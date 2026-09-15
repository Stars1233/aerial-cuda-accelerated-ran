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

#if !defined(SCF_5G_FAPI_PUSCH_PDU_PARSER_HPP_INCLUDED_)
#define SCF_5G_FAPI_PUSCH_PDU_PARSER_HPP_INCLUDED_

// This parser belongs to the FAPI store-and-replay path (ENABLE_FAPI_STORE_REPLAY).
// It supports both SCF FAPI 10.02 and 10.04: 10.04-only UCI fields are gated, and
// 10.02 rank bits are recovered from csi_part_2_bit_length when present.

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iterator>
#include <limits>
#include <ranges>
#include <span>
#include <tuple>

#include <gsl-lite/gsl-lite.hpp>

#include "aerial/casts/casts.hpp"  // aerial::casts::assume_cast
#include "scf_5g_fapi_slot_concepts.hpp"
#include "scf_5g_fapi_pusch_ue_grouping.hpp"
#include "scf_5g_fapi_lbrm.hpp"
#include "scf_5g_fapi_tags.hpp"
#include "scf_5g_fapi.h"
#include "slot_command/slot_command.hpp"
#include "nv_phy_utils.hpp"     // FAPI_SFN_MAX, nv::mu_to_slot_in_sf
#include "nvlog.h"
#include "nvlog_fmt.hpp"
#include "aerial_event_code.h"  // AERIAL_L2ADAPTER_EVENT

namespace scf_5g_fapi
{

/**
 * Self-contained fronthaul / beamforming helpers for the PUSCH parser.
 *
 * These mirror the shared utilities in scf_5g_slot_commands_common.hpp
 * (calculate_dmrs_port_mask, track_eaxcids_fh, ifAnySymbolPresent,
 * check_prb_info_size, update_prb_sym_list, update_beam_list,
 * is_latest_bfw_coff_avail) but are kept local to this header so the new UL
 * parser has no link dependency on the legacy slot-command translation unit.
 * All functions are pure (Functional Core) except where they mutate the
 * caller-owned slot-command structures they are handed.
 */
namespace pusch_fh
{

/// UL non-PRACH channel mask (PUSCH | PUCCH) — mirrors scf_5g_slot_commands.cpp.
inline constexpr uint16_t k_ul_non_prach_channel_mask =
    static_cast<uint16_t>((1u << slot_command_api::channel_type::PUSCH)
                          | (1u << slot_command_api::channel_type::PUCCH));

/// O-RAN section extension type carrying dynamic beamforming weights.
inline constexpr uint8_t k_bfw_ext_type = 11u;

/**
 * True if @p probe_sym already carries a PRB index for any channel set in @p mask.
 *
 * Used by single-sector mode to detect that the full-bandwidth UL entry for this
 * slot has already been written (so subsequent groups must not append). The probe
 * symbol is the slot's recorded UL start symbol (slot_info_t::start_symbol_ul, set
 * by append_prb_to_symbols), making the guard symbol-agnostic: it stays correct
 * when the single-sector entry is registered at a non-zero start symbol
 * (ul_start_symbol > 0), where probing the hardcoded symbol 0 would miss it.
 *
 * @param[in] symbols    Per-symbol / per-channel PRB index lists.
 * @param[in] probe_sym  OFDM symbol index to inspect (the slot's UL start symbol).
 * @param[in] mask       Channel bitmask (bit n == channel_type n).
 * @return true if any masked channel has at least one PRB index at @p probe_sym.
 */
[[nodiscard]] inline bool any_symbol_present(const slot_command_api::sym_info_list_t& symbols,
                                             std::size_t probe_sym,
                                             uint16_t mask) noexcept
{
    if (mask == 0u || probe_sym >= symbols.size()) [[unlikely]] { return false; }

    // Bit-walk the set channels rather than scanning all CHANNEL_MAX entries.
    uint16_t bits = mask;
    while (bits != 0u)
    {
        const auto ch = static_cast<std::size_t>(std::countr_zero(bits));
        if (ch < symbols[probe_sym].size() && !symbols[probe_sym][ch].empty()) { return true; }
        bits &= static_cast<uint16_t>(bits - 1u);
    }
    return false;
}

/**
 * DMRS port mask with scrambling-ID and >16-layer shifts (TS 38.211 §6.4.1.1).
 *
 * @param[in] dmrs_port_bmsk  12-bit DMRS port bitmask.
 * @param[in] scid            Scrambling ID (0 or 1) — shifts by 0 or 8 bits.
 * @param[in] nl_above16      Layers-above-16 indicator — shifts by 0 or 16 bits.
 * @return The shifted 64-bit port mask.
 */
[[nodiscard]] inline uint64_t dmrs_port_mask(uint16_t dmrs_port_bmsk,
                                             uint8_t  scid,
                                             uint8_t  nl_above16) noexcept
{
    return (static_cast<uint64_t>(dmrs_port_bmsk) << (scid * 8u)) << (16u * nl_above16);
}

/**
 * Assign each active DMRS port a sequential ap_index (eAxC-id tracking).
 *
 * @tparam UeType  UE param type exposing dmrsPortBmsk, scid, nlAbove16.
 * @param[in]     ue      UE whose DMRS ports are being mapped.
 * @param[in,out] common  PRB common info: active_eaxc_ids[] and ap_index updated.
 */
template<typename UeType>
void track_eaxc_ids(const UeType& ue, slot_command_api::prb_info_common_t& common) noexcept
{
    uint64_t mask = dmrs_port_mask(ue.dmrsPortBmsk, ue.scid, ue.nlAbove16);
    while (mask != 0u)
    {
        const auto bit = static_cast<std::size_t>(std::countr_zero(mask));
        if (bit < std::size(common.active_eaxc_ids))
        {
            common.active_eaxc_ids[bit] = static_cast<int>(common.ap_index);
        }
        ++common.ap_index;
        mask &= (mask - 1u);
    }
}

/**
 * True when the cached BFW coefficients are exactly one slot older than the
 * current slot (mirrors is_latest_bfw_coff_avail), handling SFN wraparound.
 *
 * @param[in] curr_sfn   Current system frame number.
 * @param[in] curr_slot  Current slot.
 * @param[in] prev_sfn   SFN stamped on the cached BFW buffer.
 * @param[in] prev_slot  Slot stamped on the cached BFW buffer.
 * @return true if (curr - prev) == 1 slot.
 *
 * @note Faithful port of legacy is_latest_bfw_coff_avail: mu is hardcoded to 1
 *       (30 kHz SCS), the numerology used by mMIMO/BFW deployments. slots_per_frame
 *       is therefore nv::mu_to_slot_in_sf(1) = 10*2^1 = 20 slots per 10 ms radio
 *       frame (the "sf"/"in_sf" in the helper name denotes system frame, not
 *       subframe). The freshness delta is only sensitive to this multiplier at the
 *       SFN wrap; within a frame it cancels out.
 */
[[nodiscard]] inline bool bfw_coeff_fresh(const uint16_t curr_sfn, const uint16_t curr_slot,
                                          const uint16_t prev_sfn, const uint16_t prev_slot) noexcept
{
    if (curr_slot == prev_slot) { return false; }

    // slots per 10 ms radio frame for mu=1 (= 20); see @note on the mu=1 assumption.
    const auto slots_per_frame = static_cast<uint32_t>(nv::mu_to_slot_in_sf(1));
    auto new_slots = static_cast<uint32_t>(curr_sfn) * slots_per_frame + curr_slot;
    const auto old_slots = static_cast<uint32_t>(prev_sfn) * slots_per_frame + prev_slot;
    if (old_slots > new_slots) { new_slots += FAPI_SFN_MAX * slots_per_frame; }
    return (new_slots - old_slots) == 1u;
}

/**
 * Clamp prbs_size to MAX_PRB_INFO-1 so the next write stays in bounds,
 * logging once on overflow (mirrors check_prb_info_size).
 *
 * @param[in] prbs_size  Running PRB-info count for the slot.
 * @return The clamped count (unchanged when below MAX_PRB_INFO, else MAX_PRB_INFO-1).
 */
[[nodiscard]] inline std::size_t clamp_prb_info_size(std::size_t prbs_size) noexcept
{
    if (prbs_size >= MAX_PRB_INFO) [[unlikely]]
    {
        NVLOGE_FMT(detail::k_tag, AERIAL_L2ADAPTER_EVENT,
                   "PuschPduParser: prbs_size reached MAX_PRB_INFO={}; overwriting last entry",
                   MAX_PRB_INFO);
        --prbs_size;
    }
    return prbs_size;
}

/**
 * Register @p prb_index under its START symbol only (mirrors the legacy
 * update_prb_sym_list(..., numSym=1, ...) contract).
 *
 * The PRB entry carries its own span in prb.common.numSymbols; downstream
 * consumers (FhProxy::countPuschPucchPrbs and the C-plane section generator)
 * expand the section across that span themselves. Listing the PRB under every
 * occupied symbol would therefore make those consumers count it numSymbols
 * times, inflating both the transmitted C-plane sections (RU PRB-counter
 * overflow) and the per-symbol expected-PRB map that gates the early-HARQ wait
 * kernel (Pre/Post Early HARQ wait-kernel timeout). Register the start symbol
 * only.
 *
 * @param[in,out] list        Slot symbol/PRB index structure.
 * @param[in]     prb_index   Index into list.prbs of the PRB entry.
 * @param[in]     start_sym   First OFDM symbol this PRB occupies.
 * @param[in]     channel     Channel type owning the entry.
 * @param[in]     ru          RU mode (affects single-sector start-symbol bookkeeping).
 */
inline void append_prb_to_symbols(slot_command_api::slot_info_t&        list,
                                  const std::size_t                     prb_index,
                                  const uint8_t                         start_sym,
                                  const slot_command_api::channel_type  channel,
                                  const ::ru_type                       ru) noexcept
{
    if (start_sym >= list.symbols.size()) [[unlikely]]
    {
        NVLOGE_FMT(detail::k_tag, AERIAL_L2ADAPTER_EVENT,
                   "PuschPduParser::append_prb_to_symbols: start_sym={} >= symbols={}; "
                   "dropping PRB symbol-map registration for channel={}",
                   start_sym, list.symbols.size(), static_cast<int>(channel));
        return;
    }

    auto& channel_list = list.symbols[start_sym][channel];
    // Single-sector writes one full-bandwidth entry per channel/symbol.
    if (ru == ::SINGLE_SECT_MODE && !channel_list.empty()) { return; }
    channel_list.push_back(prb_index);

    if (ru == ::SINGLE_SECT_MODE)
    {
        list.start_symbol_ul = start_sym;
    }
}

/**
 * Copy the per-PRG digital-BF beam indices from the RX beamforming PDU into the
 * PRB's beam list (mirrors the RX overload of update_beam_list; UL applies no
 * precoding so no static BF weights are derived).
 *
 * @param[in,out] prb    PRB entry whose beams_array / beams_array_size is filled.
 * @param[in]     bf     RX beamforming PDU (beam indices follow the struct).
 * @param[in]     mmimo  mMIMO enable (unused for index copy; kept for parity).
 */
inline void copy_beam_list(slot_command_api::prb_info_t&    prb,
                           const scf_fapi_rx_beamforming_t& bf,
                           const bool                       mmimo) noexcept
{
    std::ignore = mmimo;
    const uint8_t  dig_bf   = bf.dig_bf_interfaces;
    const uint16_t num_prgs = bf.num_prgs;
    if (dig_bf == 0u || num_prgs == 0u) { return; }

    const auto needed = static_cast<std::size_t>(num_prgs) * dig_bf;
    if (needed > prb.beams_array.size()) [[unlikely]]
    {
        NVLOGE_FMT(detail::k_tag, AERIAL_L2ADAPTER_EVENT,
                   "PuschPduParser: beam list overflow (need={}, cap={}); skipping beams",
                   needed, prb.beams_array.size());
        return;
    }

    // assume_cast is safe for the byte-pointer view (alignof(uint8_t) == 1). The
    // per-PRG beam indices are read with memcpy rather than assume_cast<uint16_t*>:
    // scf_fapi_rx_beamforming_t is __packed__ (alignof 1) and beam_idx begins at an
    // odd offset, so the indices are not guaranteed 2-byte aligned in the wire buffer
    // and a typed load (or assume_cast's alignment precondition) would be UB / could
    // abort. memcpy gives a correct, alignment-agnostic read.
    const auto* buf    = aerial::casts::assume_cast<uint8_t>(&bf);
    std::size_t offset = sizeof(scf_fapi_rx_beamforming_t);
    std::size_t out    = 0u;
    for (uint16_t prg = 0u; prg < num_prgs; ++prg)
    {
        const uint8_t* beam_bytes = buf + offset;
        for (uint8_t j = 0u; j < dig_bf; ++j)
        {
            uint16_t beam{};
            std::memcpy(&beam, beam_bytes + j * sizeof(uint16_t), sizeof beam);
            prb.beams_array[out++] = beam;
        }
        offset += sizeof(uint16_t) * static_cast<std::size_t>(dig_bf);
    }
    prb.beams_array_size = out;
}

} // namespace pusch_fh

/**
 * Parses a single PUSCH PDU from a UL_TTI.request message.
 *
 * Populates pusch_params in the slot-command structure, forwarding
 * group_command, cell_sub_command, staticPuschSlotNum, lbrm, bf_enabled,
 * dtx_thresholds_pusch, enable_weighted_avg_cfo, mmimo_enabled,
 * cell_stat_prm_idx, carrier_id, bfw_coeff_mem_info, and ru_type from the
 * module view.
 *
 * Before dispatching individual PUSCH PDUs, call setup_cell() once per
 * UL_TTI.request to initialize cuphyPuschCellDynPrm_t and register the
 * cell in pusch_params.
 *
 * @note This parser populates UE scalar fields, UE grouping, DMRS, the
 *       variable-length payload (Data / UCI / DFT-s-OFDM), the beamforming /
 *       fronthaul PRB and symbol maps (see setup_beamforming_and_fh /
 *       update_fh_params / apply_bfw_coefficients), and the maintenance /
 *       CSI-Part2 / extension tail.  The fronthaul PRB/symbol mapping is a
 *       temporary implementation for isolated PUSCH testing (single channel,
 *       single core); full multi-channel concurrent FH is future work.
 *
 * @tparam V  Type satisfying PuschModuleView.
 * @tparam G  UE-grouping policy; defaults to DefaultPuschUeGroupingPolicy
 *            which matches UEs by (startSym, nSym, startPrb, nPrb).
 */
template<PuschModuleView V, PuschUeGroupingPolicy G = DefaultPuschUeGroupingPolicy>
class PuschPduParser final
{
public:
    using pdu_t = scf_fapi_pusch_pdu_t;
    static constexpr scf_fapi_ul_tti_pdu_type_t pdu_type = UL_TTI_PDU_TYPE_PUSCH;

    explicit PuschPduParser(V& v) noexcept : view_{&v} {}

    /**
     * Per-message setup: initialise cell dynamic PUSCH params and register
     * the cell in pusch_params.
     *
     * Must be called once per UL_TTI.request before any parse() calls for
     * that message. Detects and skips duplicate carrier IDs.
     *
     * @param[in] req         The UL TTI request for the current message.
     * @param[in] msg_cell_id Logical cell id from the UL_TTI message header.
     *                        Used as the source key for the per-cell config
     *                        lookups (carrier_id / cell_stat_prm_idx); the
     *                        destination slot in pusch_params is still the
     *                        running nCells count.  Mirrors the SRS/PRACH
     *                        setup_cell(*req, msg.cell_id) contract so a
     *                        multi-cell UL_TTI that skips PUSCH-less cells
     *                        registers each PUSCH under its actual carrier.
     */
    void setup_cell(const scf_fapi_ul_tti_req_t& req, uint32_t msg_cell_id) noexcept;

    /**
     * Process one PUSCH PDU for the given sfn/slot.
     *
     * @param[in] sfn   System Frame Number.
     * @param[in] slot  Slot number.
     * @param[in] pdu   PUSCH PDU to process.
     * @return true on success; false if processing could not be completed
     *         (null params, UE/UE-group capacity exhausted, or an exception
     *         from slot_command()). Return value must be checked.
     * @note Marked noexcept: the body wraps view_->slot_command() and the
     *       sub-step helpers in a try-catch (slot_command() can throw via
     *       at()/getMPlaneConfig on misconfiguration).  Any exception is
     *       logged and mapped to false so the dispatch infrastructure's
     *       noexcept contract holds.
     * @pre  The caller (TtiDispatch::dispatch_pdus_typed) has validated that
     *       @p pdu.pdu_size covers the generic header and does not exceed the
     *       remaining UL_TTI payload, so @p pdu.pdu_size bytes are guaranteed
     *       readable.  This method does NOT independently bound its internal
     *       cursor walk against pdu_size — the per-section advances trust the
     *       wire fields (numPRGs, numPart2s, numPart1Params) to stay within it.
     *       A span/length-based bound threaded through the sub-steps (so every
     *       assume_cast is checked against pdu_size) is a deferred Phase-3
     *       hardening item; see MR-5325 review finding Q-2.
     */
    [[nodiscard]] bool parse(uint16_t sfn, uint16_t slot, const pdu_t& pdu) noexcept;

private:
    gsl_lite::not_null<V*> view_; //!< Non-owning; must outlive this parser.

    /// False in fapi_to_cplane_direct mode: the framework builds the O-RAN C-plane
    /// directly from FAPI, so the parser must NOT fill the normal FH sym_prb_info.
    /// Direct mode still fills order_sym_prb_info() for the UL Order kernel.
    /// Cached once per UL_TTI in setup_cell (mirrors PdschPduParser::fh_enabled_for_msg_).
    bool fh_enabled_for_msg_{true};

    // ------------------------------------------------------------------
    // Compile-time constants
    // ------------------------------------------------------------------

    /// pdu_bitmap bit 0 — optional puschData section present.
    static constexpr uint16_t k_bitmap_data = 0x1u;
    /// pdu_bitmap bit 1 — optional puschUci section present.
    static constexpr uint16_t k_bitmap_uci = 0x2u;
    /// pdu_bitmap bit 3 — optional dftsOfdm section present.
    static constexpr uint16_t k_bitmap_dftsofdm = 0x8u;
    /// pdu_bitmap bit 5 — rank bits decoded from UCI (10.02 only).
    static constexpr uint16_t k_bitmap_rank_bits = 0x20u;

    /// Default LDPC maximum decoder iterations when no PUSCH extension is present.
    static constexpr uint8_t k_default_ldpc_max_iters = 10u;
    /// OFDM symbols per slot (normal cyclic prefix) used for the DFT-s-OFDM fallback.
    static constexpr uint8_t k_n_symb_per_slot = 14u;
    /// Scale factor mapping the FAPI fo_forget_coeff (0–100) to a [0,1] float.
    static constexpr float k_fo_forget_coeff_scale = 100.0f;

    // ------------------------------------------------------------------
    // parse() helpers
    // ------------------------------------------------------------------

    /**
     * Aggregates the values produced while walking the variable-length payload
     * that later sub-steps need: the cursor positioned just past the optional
     * Data/UCI/DFT-s-OFDM sections, and whether CSI-Part2 was signalled.
     */
    struct PayloadWalk {
        const uint8_t* next;            //!< Cursor past Data/UCI/DFT-s-OFDM (points at beamforming).
        bool           csi_part2_signaled; //!< True when UCI flag_csi_part2 == 0xFFFF (10.04).
    };

    /**
     * Populate the per-UE scalar fields of ue_info[nUes] from the fixed PUSCH
     * PDU header, including the LBRM parameters when enabled.
     *
     * Does not increment cell_grp_info.nUes — the UE index is committed by
     * parse() only after all sub-steps succeed.
     *
     * @param[in,out] params  Active pusch_params for this slot.
     * @param[in]     pdu     PUSCH PDU being processed.
     */
    void setup_ue_fields(slot_command_api::pusch_params& params,
                         const pdu_t&                    pdu) noexcept;

    /**
     * Match or create a UE group for this PDU using policy G, link the current
     * UE into the group, and (for a new group) initialise its allocation and
     * structural pointers.
     *
     * @param[in,out] params  Active pusch_params for this slot.
     * @param[in]     pdu     PUSCH PDU being processed.
     * @return true on success; false if nCells==0 (setup_cell skipped) or the
     *         UE-group / per-group-UE capacity is exhausted.  On false, no
     *         group state has been mutated.  Return value must be checked.
     */
    [[nodiscard]] bool setup_ue_group(slot_command_api::pusch_params& params,
                                      const pdu_t&                    pdu) noexcept;

    /**
     * Populate the per-group DMRS configuration (semi-static parameters shared
     * by all UEs co-scheduled in the group).
     *
     * Must run after setup_ue_group() — it indexes ue_dmrs_info via the UE's
     * resolved ueGrpIdx.  Writing identical values from every UE in a group is
     * correct by 3GPP design (TS 38.211 §6.4.1.1).
     *
     * @param[in,out] params  Active pusch_params for this slot.
     * @param[in]     pdu     PUSCH PDU being processed.
     */
    void setup_dmrs(slot_command_api::pusch_params& params,
                    const pdu_t&                    pdu) noexcept;

    /**
     * Walk the variable-length payload (Data, UCI, DFT-s-OFDM) and populate the
     * corresponding ue_info / uci_info fields.
     *
     * @param[in,out] params  Active pusch_params for this slot.
     * @param[in]     pdu     PUSCH PDU being processed.
     * @return The cursor positioned at the beamforming section plus the
     *         CSI-Part2 signalling flag.  Return value must be checked.
     */
    [[nodiscard]] PayloadWalk setup_payload(slot_command_api::pusch_params& params,
                                            const pdu_t&                    pdu) noexcept;

    /**
     * Parse the variable-length beamforming section, populate the UE group
     * beamforming fields (uplink streams, PRG size, per-PRG channel estimation),
     * drive the fronthaul PRB/symbol map via update_fh_params, and advance
     * @p next past the section.
     *
     * @param[in,out] params  Active pusch_params for this slot.
     * @param[in]     pdu     PUSCH PDU being processed.
     * @param[in,out] next    Cursor at the beamforming section; advanced past it.
     * @param[in]     sfn     System Frame Number (BFW freshness check).
     * @param[in]     slot    Slot number (BFW freshness check).
     */
    void setup_beamforming_and_fh(slot_command_api::pusch_params& params,
                                  const pdu_t&                    pdu,
                                  const uint8_t*&                 next,
                                  const uint16_t                  sfn,
                                  const uint16_t                  slot) noexcept;

    /**
     * Update the cell fronthaul PRB allocation, symbol map, port mask, beam
     * list and BFW coefficients for this UE/group.
     *
     * Temporary implementation for isolated PUSCH testing: single-sector mode
     * writes one full-bandwidth UL entry per slot; otherwise a per-group PRB
     * entry is written on the first UE and only the port mask is merged for
     * subsequent mMIMO dynamic-BFW UEs.
     *
     * @param[in,out] params      Active pusch_params for this slot.
     * @param[in,out] grp         UE group owning this allocation.
     * @param[in]     is_new_grp  True when this UE created @p grp.
     * @param[in]     ue          UE being processed.
     * @param[in]     bf          RX beamforming PDU.
     * @param[in]     pdu         PUSCH PDU being processed.
     * @param[in,out] cell_cmd    Cell sub-command carrying the FH PRB/symbol map.
     * @param[in]     ru          RU mode.
     * @param[in]     carrier     Logical carrier index (cell_view / BFW lookup).
     * @param[in]     sfn         System Frame Number (BFW freshness check).
     * @param[in]     slot        Slot number (BFW freshness check).
     */
    void update_fh_params(slot_command_api::pusch_params&     params,
                          cuphyPuschUeGrpPrm_t&               grp,
                          bool                                is_new_grp,
                          const cuphyPuschUePrm_t&            ue,
                          const scf_fapi_rx_beamforming_t&    bf,
                          const pdu_t&                        pdu,
                          slot_command_api::cell_sub_command& cell_cmd,
                          slot_command_api::slot_info_t*       sym_prbs_override,
                          ::ru_type                           ru,
                          uint32_t                            carrier,
                          uint16_t sfn, uint16_t slot) noexcept;

    /**
     * Attach dynamic BFW coefficient buffers to @p prb when fresh weights are
     * available for the current slot (dynamic-BFW path, dig_bf_interfaces == 0).
     *
     * @param[in,out] params   Active pusch_params for this slot (BFW group index).
     * @param[in,out] prb      PRB entry whose bfwCoeff_buf_info / extType is set.
     * @param[in]     bf       RX beamforming PDU.
     * @param[in]     carrier  Logical carrier index (BFW memory lookup).
     * @param[in]     sfn      System Frame Number (freshness check).
     * @param[in]     slot     Slot number (freshness check).
     */
    void apply_bfw_coefficients(slot_command_api::pusch_params& params,
                                slot_command_api::prb_info_t&   prb,
                                const scf_fapi_rx_beamforming_t& bf,
                                uint32_t                        carrier,
                                uint16_t sfn, uint16_t slot) noexcept;

    /**
     * Parse the maintenance, optional CSI-Part2, and optional weighted-average
     * CFO / LDPC extension sections (SCF_FAPI_10_04), and apply the DFT-s-OFDM
     * maintenance fallback for the current UE.
     *
     * @param[in,out] params              Active pusch_params for this slot.
     * @param[in]     pdu                 PUSCH PDU being processed.
     * @param[in]     next                Cursor at the maintenance section.
     * @param[in]     csi_part2_signaled  Whether UCI signalled CSI-Part2.
     */
    void setup_maintenance_and_extension(slot_command_api::pusch_params& params,
                                         const pdu_t&                    pdu,
                                         const uint8_t*                  next,
                                         bool csi_part2_signaled) noexcept;

    /**
     * Parse the CSI-Part2 correspondence information into the current UE's
     * uci_info entry (cell_grp_info.nUes), matching the implicit "current UE"
     * convention used by the other parse() sub-steps.
     *
     * @param[in,out] params  Active pusch_params for this slot.
     * @param[in]     next    Cursor at the scf_uci_csip2_info_t header.
     * @return Cursor advanced past the CSI-Part2 section (unchanged when the
     *         report count is zero, matching legacy update_cell_command).
     *         Return value must be checked.
     */
    [[nodiscard]] const uint8_t* parse_csi_part2(slot_command_api::pusch_params& params,
                                                 const uint8_t*                  next) noexcept;
};

// ---------------------------------------------------------------------------
// Inline definitions
// ---------------------------------------------------------------------------

template<PuschModuleView V, PuschUeGroupingPolicy G>
bool PuschPduParser<V, G>::parse(uint16_t sfn, uint16_t slot, const pdu_t& pdu) noexcept
{
    // view_->slot_command() indexes slot_command_array via at() and may throw;
    // catch here so the dispatch infrastructure's noexcept contract holds.
    try
    {
        auto* params = view_->slot_command().cell_groups.get_pusch_params();
        if (!params) [[unlikely]]
        {
            NVLOGW_FMT(detail::k_tag,
                       "PuschPduParser::parse: sfn={} slot={} null pusch_params; dropping PDU",
                       sfn, slot);
            return false;
        }

        if (params->cell_grp_info.nUes
                >= static_cast<uint16_t>(slot_command_api::MAX_PUSCH_UE_PER_TTI)) [[unlikely]]
        {
            NVLOGE_FMT(detail::k_tag, AERIAL_L2ADAPTER_EVENT,
                       "PuschPduParser: sfn={} slot={} nUes={} >= MAX_PUSCH_UE_PER_TTI={}; dropping PDU",
                       sfn, slot, params->cell_grp_info.nUes, slot_command_api::MAX_PUSCH_UE_PER_TTI);
            return false;
        }

        setup_ue_fields(*params, pdu);
        if (!setup_ue_group(*params, pdu)) [[unlikely]] { return false; }

        // Exception-safety / commit invariant: setup_ue_group() has already
        // advanced the per-group state (nUeGrps, nue_grps_per_cell, ue_grp.nUes,
        // and the UE->group links).  This cannot be deferred to the tail commit
        // because setup_dmrs()/setup_payload() below index into the group it just
        // registered (e.g. ue_info[nUes].ueGrpIdx).  The per-UE commit (handle
        // list + nUes) therefore happens last, and the "all sub-steps succeeded"
        // guarantee relies on every step from here to the commit being noexcept:
        // a throw would call std::terminate rather than unwind to the catch below
        // and leave group state desynced from nUes.  If any sub-step ever becomes
        // throwing, add explicit rollback of the setup_ue_group() mutations here.
        setup_dmrs(*params, pdu);

        const PayloadWalk walk = setup_payload(*params, pdu);
        const uint8_t*    next = walk.next;
        setup_beamforming_and_fh(*params, pdu, next, sfn, slot);
        setup_maintenance_and_extension(*params, pdu, next, walk.csi_part2_signaled);

        // Deferred-commit invariant (C-4): every sub-step above must be noexcept,
        // so a throw cannot unwind into the commit region below and leave an
        // orphaned group (setup_ue_group advances nUeGrps/nue_grps_per_cell before
        // the UE handle/nUes are committed here). A future edit that drops noexcept
        // from any sub-step breaks this build deliberately. noexcept(...) operands
        // are unevaluated — zero runtime cost.
        static_assert(
            noexcept(setup_ue_fields(*params, pdu)) &&
            noexcept(setup_ue_group(*params, pdu)) &&
            noexcept(setup_dmrs(*params, pdu)) &&
            noexcept(setup_payload(*params, pdu)) &&
            noexcept(setup_beamforming_and_fh(*params, pdu, next, sfn, slot)) &&
            noexcept(setup_maintenance_and_extension(*params, pdu, next, walk.csi_part2_signaled)),
            "PuschPduParser deferred-commit invariant violated: all parse() sub-steps "
            "must be noexcept (see C-4).");

        // Commit: handle list and nUes are advanced only after every sub-step
        // succeeds, keeping scf_ul_tti_handle_list.size() == nUes.  Capacity is
        // reserved in pusch_params' constructor, so push_back never reallocates
        // on the slot path.
        params->scf_ul_tti_handle_list.push_back(pdu.handle);
        ++params->cell_grp_info.nUes;
        return true;
    }
    catch (const std::exception& ex)
    {
        NVLOGE_FMT(detail::k_tag, AERIAL_L2ADAPTER_EVENT,
                   "PuschPduParser::parse sfn={} slot={} threw: {}", sfn, slot, ex.what());
        return false;
    }
    catch (...)
    {
        NVLOGE_FMT(detail::k_tag, AERIAL_L2ADAPTER_EVENT,
                   "PuschPduParser::parse sfn={} slot={} threw unknown exception", sfn, slot);
        return false;
    }
}

template<PuschModuleView V, PuschUeGroupingPolicy G>
void PuschPduParser<V, G>::setup_ue_fields(slot_command_api::pusch_params& params,
                                           const pdu_t&                    pdu) noexcept
{
    auto& ue = params.ue_info[params.cell_grp_info.nUes];

    ue.puschIdentity  = pdu.pusch_identity;
    ue.scid           = pdu.scid;
    // Full dmrs_ports assigned verbatim (matches legacy scf_5g_slot_commands.cpp:171).
    // Unlike PDSCH, PUSCH does not extract a >16-layer indicator: nlAbove16 is a
    // DL high-layer-MU-MIMO concept (bit 13 of PDSCH dmrs_ports), whereas PUSCH is
    // <=4 layers/UE (see lbrm::k_max_layers), so bit 13 is never set and nlAbove16
    // is left at its zero-init value — consumed unshifted by calculate_dmrs_port_mask().
    ue.dmrsPortBmsk   = pdu.dmrs_ports;
    ue.mcsTableIndex  = pdu.mcs_table;
    ue.mcsIndex       = pdu.mcs_index;
    ue.rnti           = pdu.rnti;
    ue.dataScramId    = pdu.data_scrambling_id;
    ue.nUeLayers      = pdu.num_of_layers;
    ue.targetCodeRate = pdu.target_code_rate;
    ue.qamModOrder    = pdu.qam_mod_order;
    ue.pduBitmap      = pdu.pdu_bitmap;
    ue.pUciPrms       = nullptr;

    // LBRM (38.212 §5.4.2.1) — only populated when enabled by config.
    ue.i_lbrm = view_->lbrm();
    if (ue.i_lbrm)
    {
        // Prefer CONFIG.request nRxAnt (via the cell registered by setup_cell).
        // Fall back to k_max_layers when no cell is registered yet.
        uint16_t num_ul_ant = lbrm::k_max_layers;
        if (params.cell_grp_info.nCells > 0u && !params.cell_index_list.empty())
        {
            const auto carrier = static_cast<uint32_t>(params.cell_index_list.back());
            const slot_command_api::slot_indication slot_ind{};
            num_ul_ant = view_->cell_view(carrier, slot_ind).cell_params().nRxAnt;
        }
        ue.maxLayers  = lbrm::compute_max_layers(num_ul_ant);
        ue.maxQm      = lbrm::compute_max_qm(pdu.mcs_table);
        ue.n_PRB_LBRM = lbrm::compute_n_prb_lbrm(pdu.bwp.bwp_size);
    }

    // Extension defaults — overwritten by setup_maintenance_and_extension when
    // the PUSCH extension section is present.
    ue.foForgetCoeff             = 0.0f;
    ue.ldpcEarlyTerminationPerUe = 0u;
    ue.ldpcMaxNumItrPerUe        = k_default_ldpc_max_iters;
}

template<PuschModuleView V, PuschUeGroupingPolicy G>
bool PuschPduParser<V, G>::setup_ue_group(slot_command_api::pusch_params& params,
                                          const pdu_t&                    pdu) noexcept
{
    // setup_cell() must have registered at least one cell; guard against
    // unsigned underflow on cell_dyn_info[nCells-1] below.
    if (params.cell_grp_info.nCells == 0u) [[unlikely]]
    {
        NVLOGW_FMT(detail::k_tag,
                   "PuschPduParser::setup_ue_group: rnti=0x{:04X} nCells==0 (setup_cell not called); dropping UE",
                   pdu.rnti);
        return false;
    }

    const auto ue_idx  = params.cell_grp_info.nUes;
    auto&      ue      = params.ue_info[ue_idx];

    // The current PDU belongs to the most recently registered cell.
    const auto carrier   = static_cast<std::size_t>(params.cell_index_list.back());
    const auto grp_start = params.cell_ue_group_idx_start[carrier];

    // Search this cell's initialised groups for a compatible allocation.
    const auto grp_span = std::span{
        params.ue_grp_info + grp_start,
        params.ue_grp_info + params.cell_grp_info.nUeGrps};

    const auto iter = std::ranges::find_if(grp_span,
        [&](const cuphyPuschUeGrpPrm_t& g) noexcept { return G::matches(g, pdu); });

    const bool        is_new_grp = (iter == grp_span.end());
    const std::size_t grp_idx    = is_new_grp
        ? static_cast<std::size_t>(params.cell_grp_info.nUeGrps)
        : static_cast<std::size_t>(iter - grp_span.begin()) + grp_start;

    if (is_new_grp && params.cell_grp_info.nUeGrps
            >= static_cast<uint16_t>(slot_command_api::MAX_PUSCH_UE_GROUPS)) [[unlikely]]
    {
        NVLOGE_FMT(detail::k_tag, AERIAL_L2ADAPTER_EVENT,
                   "PuschPduParser: rnti=0x{:04X} nUeGrps={} >= MAX_PUSCH_UE_GROUPS={}; dropping PDU",
                   pdu.rnti, params.cell_grp_info.nUeGrps, slot_command_api::MAX_PUSCH_UE_GROUPS);
        return false;
    }

    auto& ue_grp = params.ue_grp_info[grp_idx];

    // pUePrmIdxs is sized MAX_PUSCH_UE_PER_TTI per group (see pusch_params ctor);
    // guard against overrunning that backing storage.
    if (ue_grp.nUes >= static_cast<uint16_t>(slot_command_api::MAX_PUSCH_UE_PER_TTI)) [[unlikely]]
    {
        NVLOGE_FMT(detail::k_tag, AERIAL_L2ADAPTER_EVENT,
                   "PuschPduParser: rnti=0x{:04X} grp_idx={} group nUes={} >= MAX_PUSCH_UE_PER_TTI={}; dropping UE",
                   pdu.rnti, grp_idx, ue_grp.nUes, slot_command_api::MAX_PUSCH_UE_PER_TTI);
        return false;
    }

    // Link UE -> group; register the UE index in the group's member list.
    ue.ueGrpIdx                    = static_cast<uint16_t>(grp_idx);
    ue.pUeGrpPrm                   = &ue_grp;
    ue_grp.pUePrmIdxs[ue_grp.nUes] = ue_idx;
    ++ue_grp.nUes;

    if (is_new_grp)
    {
        ++params.cell_grp_info.nUeGrps;
        ++params.nue_grps_per_cell[carrier];

        // Delegate time-frequency allocation init to the policy.
        G::init_new_group(ue_grp, params, grp_idx, pdu);

        // Structural links (owned by the parser, never the policy).
        ue_grp.pCellPrm    = &params.cell_dyn_info[params.cell_grp_info.nCells - 1u];
        ue_grp.pDmrsDynPrm = &params.ue_dmrs_info[grp_idx];
    }

    return true;
}

template<PuschModuleView V, PuschUeGroupingPolicy G>
void PuschPduParser<V, G>::setup_dmrs(slot_command_api::pusch_params& params,
                                      const pdu_t&                    pdu) noexcept
{
    const auto grp_idx = params.ue_info[params.cell_grp_info.nUes].ueGrpIdx;
    auto&      ue_dmrs = params.ue_dmrs_info[grp_idx];

    // Double-symbol DMRS when two adjacent symbol-position bits are both set.
    const uint16_t sym_pos = pdu.ul_dmrs_sym_pos;
    ue_dmrs.dmrsMaxLen = ((sym_pos & static_cast<uint16_t>(sym_pos >> 1u)) != 0u) ? 2u : 1u;

    // Additional positions = popcount / maxLen - 1.  Guard the empty-mask case
    // (validation is deferred) to avoid an unsigned underflow to 0xFF.
    const auto bit_count = static_cast<uint8_t>(std::popcount(sym_pos));
    ue_dmrs.dmrsAddlnPos = (bit_count == 0u)
        ? 0u
        : static_cast<uint8_t>(bit_count / ue_dmrs.dmrsMaxLen - 1u);

    ue_dmrs.nDmrsCdmGrpsNoData = pdu.num_dmrs_cdm_groups_no_data;
    ue_dmrs.dmrsScrmId         = pdu.ul_dmrs_scrambling_id;
}

template<PuschModuleView V, PuschUeGroupingPolicy G>
typename PuschPduParser<V, G>::PayloadWalk
PuschPduParser<V, G>::setup_payload(slot_command_api::pusch_params& params,
                                    const pdu_t&                    pdu) noexcept
{
    const auto ue_idx = params.cell_grp_info.nUes;
    auto&      ue     = params.ue_info[ue_idx];

    // SCF FAPI embeds variable-length sections in payload[]; we walk this layout
    // with pointer arithmetic and aerial::casts::assume_cast, which centralizes
    // the pointer punning behind standard-layout/trivial static_asserts and a
    // runtime alignment check (mirrors the SRS parser; SCF 222.10.0x §3.4.3).
    const uint8_t* next = &pdu.payload[0];
    bool csi_part2_signaled = false;

    // puschData (pdu_bitmap bit 0).
    if ((pdu.pdu_bitmap & k_bitmap_data) != 0u)
    {
        const auto* data = aerial::casts::assume_cast<scf_fapi_pusch_data_t>(next);
        ue.rv                     = data->rv_index;
        ue.TBSize                 = data->tb_size;
        ue.ndi                    = data->new_data_indicator;
        ue.harqProcessId          = data->harq_process_id;
        params.ue_tb_size[ue_idx] = data->tb_size;
        next += sizeof(scf_fapi_pusch_data_t);
    }

    // puschUci (pdu_bitmap bit 1).
    if ((pdu.pdu_bitmap & k_bitmap_uci) != 0u)
    {
        const auto* uci = aerial::casts::assume_cast<scf_fapi_pusch_uci_t>(next);
        ue.pUciPrms = &params.uci_info[ue_idx];
        auto* up = ue.pUciPrms;
        up->nBitsHarq         = uci->harq_ack_bit_length;
        up->nBitsCsi1         = uci->csi_part_1_bit_length;
        up->alphaScaling      = uci->alpha_scaling;
        up->betaOffsetHarqAck = uci->beta_offset_harq_ack;
        up->betaOffsetCsi1    = uci->beta_offset_csi_1;
        up->betaOffsetCsi2    = uci->beta_offset_csi_2;

#ifdef SCF_FAPI_10_04
        // 10.04 signals CSI-Part2 presence via flag_csi_part2 == 0xFFFF; the rank
        // bits are carried in the dedicated CSI-Part2 section, not encoded here.
        up->nRanksBits     = std::numeric_limits<uint8_t>::max();
        csi_part2_signaled = (uci->flag_csi_part2 == std::numeric_limits<uint16_t>::max());
#else
        // 10.02 packs the rank bit offset (low byte) and size (high byte) into
        // csi_part_2_bit_length; out-of-range values disable rank decoding. The
        // constants and decode mirror the legacy serial path in
        // scf_5g_slot_commands.cpp -- unchanged 10.02 production behaviour.
        constexpr uint16_t k_csi_part2_disabled    = 255u;   // sentinel: rank decode off
        constexpr uint16_t k_csi_part2_bit_len_max = 1707u;  // exclusive upper bound
        constexpr uint8_t  k_rank_bits_max         = 4u;     // max valid rank-bit count
        constexpr uint8_t  k_rank_bit_offset_max   = 47u;    // max valid rank-bit offset
        up->nRanksBits = std::numeric_limits<uint8_t>::max();
        if (const uint16_t csi2_bits = uci->csi_part_2_bit_length;
            csi2_bits > 0u && csi2_bits != k_csi_part2_disabled &&
            csi2_bits < k_csi_part2_bit_len_max)
        {
            up->rankBitOffset = static_cast<uint8_t>(csi2_bits & 0xFFu);
            up->nRanksBits    = static_cast<uint8_t>((csi2_bits >> 8u) & 0xFFu);
            if (up->nRanksBits > k_rank_bits_max || up->rankBitOffset > k_rank_bit_offset_max)
            {
                up->nRanksBits = std::numeric_limits<uint8_t>::max();
            }
        }
#endif
        if (up->nRanksBits != std::numeric_limits<uint8_t>::max())
        {
            ue.pduBitmap |= k_bitmap_rank_bits;
        }

        up->nCsiReports  = 1u;
        up->DTXthreshold = view_->dtx_thresholds_pusch();
        next += sizeof(scf_fapi_pusch_uci_t);
    }

    // DFT-s-OFDM transform precoding (transform_precoding == 0 enables it).
    if (pdu.transform_precoding == 0u)
    {
        ue.enableTfPrcd = 1u;
        if ((pdu.pdu_bitmap & k_bitmap_dftsofdm) != 0u)
        {
            const auto* dft = aerial::casts::assume_cast<scf_fapi_pusch_dftsofdm_t>(next);
            ue.lowPaprGroupNumber     = dft->lowPaprGroupNumber;
            ue.lowPaprSequenceNumber  = dft->lowPaprSequenceNumber;
            ue.groupOrSequenceHopping = 0u;
            next += sizeof(scf_fapi_pusch_dftsofdm_t);
        }
    }
    else
    {
        ue.enableTfPrcd = 0u;
    }

    return PayloadWalk{next, csi_part2_signaled};
}

template<PuschModuleView V, PuschUeGroupingPolicy G>
void PuschPduParser<V, G>::setup_beamforming_and_fh(
    slot_command_api::pusch_params& params,
    const pdu_t&                    pdu,
    const uint8_t*&                 next,
    const uint16_t                  sfn,
    const uint16_t                  slot) noexcept
{
    const auto& bf     = *aerial::casts::assume_cast<scf_fapi_rx_beamforming_t>(next);
    auto&       ue     = params.ue_info[params.cell_grp_info.nUes];
    auto&       ue_grp = *ue.pUeGrpPrm;
    const bool  mmimo  = view_->mmimo_enabled();

    // mMIMO uplink-stream accounting: explicit digital-BF interfaces set the
    // stream count directly; dynamic BFW accumulates per-UE layers.
    if (mmimo)
    {
        if (bf.dig_bf_interfaces != 0u)
        {
            ue_grp.nUplinkStreams = bf.dig_bf_interfaces;
        }
        else
        {
            ue_grp.nUplinkStreams =
                static_cast<uint16_t>(ue_grp.nUplinkStreams + ue.nUeLayers);
        }
    }

    ue_grp.prgSize                 = bf.prg_size;
    ue_grp.enablePerPrgChEstPerUeg = (mmimo && bf.prg_size >= 1u && bf.prg_size <= 4u) ? 1u : 0u;

    // Fronthaul PRB/symbol map fill runs in both modes; only the destination differs:
    // - non-direct mode writes the slot command's real sym_prb_info for FH.
    // - direct C-plane mode writes only order_sym_prb_info scratch; the framework
    //   builds O-RAN C-plane directly from FAPI, but the UL Order kernel still
    //   consumes PRB metadata.
    // Do not gate this block on fh_enabled_for_msg_: that would drop the Order PRB
    // metadata the UL Order kernel needs in direct mode.
    //
    // Fronthaul PRB/symbol map lives on the most recently registered cell.
    // carrier was validated < MAX_CELLS_PER_CELL_GROUP in setup_cell, but guard
    // again here: cell_sub_command() indexes cells[] via at() and this function is
    // noexcept, so an out-of-range carrier would std::terminate (bypassing parse()'s
    // exception handler) instead of dropping the update.  On a bad index, skip the
    // fronthaul/beamforming write but still advance the cursor below so the
    // maintenance/extension parse stays byte-aligned.
    const auto carrier = static_cast<uint32_t>(params.cell_index_list.back());
    if (carrier < static_cast<uint32_t>(slot_command_api::MAX_CELLS_PER_CELL_GROUP)) [[likely]]
    {
        auto&      cell_cmd   = view_->cell_sub_command(carrier);
        const auto ru         = view_->ru(carrier);
        const bool is_new_grp = (ue_grp.nUes == 1u);
        auto* const sym_prbs  = fh_enabled_for_msg_
            ? cell_cmd.sym_prb_info()
            : view_->order_sym_prb_info(carrier);

        update_fh_params(params, ue_grp, is_new_grp, ue, bf, pdu, cell_cmd, sym_prbs, ru, carrier, sfn, slot);
    }
    else [[unlikely]]
    {
        NVLOGE_FMT(detail::k_tag, AERIAL_L2ADAPTER_EVENT,
                   "PuschPduParser::setup_beamforming_and_fh: sfn={} slot={} carrier={} >= "
                   "MAX_CELLS_PER_CELL_GROUP={}; skipping fronthaul/beamforming update for this UE group",
                   sfn, slot, carrier, slot_command_api::MAX_CELLS_PER_CELL_GROUP);
    }

    // Advance the cursor past the beamforming section.  The inline beam_idx[]
    // array holds numPRGs * digBFInterfaces entries (SCF FAPI Table 3-53), so the
    // cursor must advance by that full count -- numPRGs may exceed 1, and omitting
    // the factor mislocates the maintenance/extension tail; dynamic BFW
    // (dig_bf_interfaces == 0) carries no inline beam indices.
    if (!view_->mmimo_enabled() || bf.dig_bf_interfaces != 0u)
    {
        const std::size_t num_beam_entries =
            static_cast<std::size_t>(bf.num_prgs) *
            static_cast<std::size_t>(bf.dig_bf_interfaces);
        next += sizeof(scf_fapi_rx_beamforming_t)
              + sizeof(uint16_t) * num_beam_entries;
    }
    else
    {
        next += sizeof(scf_fapi_rx_beamforming_t);
    }
}

template<PuschModuleView V, PuschUeGroupingPolicy G>
void PuschPduParser<V, G>::update_fh_params(
    slot_command_api::pusch_params&         params,
    cuphyPuschUeGrpPrm_t&                    grp,
    bool                                     is_new_grp,
    const cuphyPuschUePrm_t&                 ue,
    const scf_fapi_rx_beamforming_t&         bf,
    const pdu_t&                             pdu,
    slot_command_api::cell_sub_command&      cell_cmd,
    slot_command_api::slot_info_t*           sym_prbs_override,
    ::ru_type                                ru,
    uint32_t                                 carrier,
    uint16_t sfn, uint16_t slot) noexcept
{
    auto* sym_prbs = sym_prbs_override != nullptr ? sym_prbs_override : cell_cmd.sym_prb_info();
    if (sym_prbs == nullptr) [[unlikely]]
    {
        NVLOGW_FMT(detail::k_tag,
                   "PuschPduParser::update_fh_params: carrier={} sfn={} slot={} null sym_prb_info; "
                   "dropping fronthaul PRB/symbol update for this UE group",
                   carrier, sfn, slot);
        return;
    }
    auto& prbs = sym_prbs->prbs;

    const auto mmimo = view_->mmimo_enabled();

    // Single-sector: one full-bandwidth UL entry already covers this slot. Probe
    // the symbol where that entry was registered (start_symbol_ul, set by
    // append_prb_to_symbols) so the guard fires even when ul_start_symbol > 0.
    if (ru == ::SINGLE_SECT_MODE
        && pusch_fh::any_symbol_present(sym_prbs->symbols, sym_prbs->start_symbol_ul,
                                        pusch_fh::k_ul_non_prach_channel_mask))
    {
        return;
    }

    if (is_new_grp)
    {
        sym_prbs->prbs_size = pusch_fh::clamp_prb_info_size(sym_prbs->prbs_size);

        // Single-sector C-plane symbol span/start come from the cell's TDD slot
        // detail (mirrors legacy update_fh_params_pusch); captured here from the
        // per-cell view and applied to the symbol map below.
        uint8_t single_sect_num_sym = k_n_symb_per_slot;
        uint8_t single_sect_start   = 0u;
        if (ru == ::SINGLE_SECT_MODE)
        {
            // Full UL bandwidth + UL symbol detail from the per-cell view (per-UE
            // PRB ignored in single-sector).
            slot_command_api::slot_indication slot_ind{};
            slot_ind.sfn_  = sfn;
            slot_ind.slot_ = slot;
            const auto cv       = view_->cell_view(carrier, slot_ind);
            single_sect_num_sym = cv.ul_max_symbols(k_n_symb_per_slot);
            single_sect_start   = cv.ul_start_symbol();
            prbs[sym_prbs->prbs_size] = slot_command_api::prb_info_t(0u, cv.cell_params().nPrbUlBwp);
        }
        else
        {
            prbs[sym_prbs->prbs_size] = slot_command_api::prb_info_t(grp.startPrb, grp.nPrb);
        }
        ++sym_prbs->prbs_size;

        const std::size_t            index = sym_prbs->prbs_size - 1u;
        slot_command_api::prb_info_t& prb  = prbs[index];
        prb.common.ap_index  = 0u;  // first UE of a new group.
        prb.common.direction = slot_command_api::fh_dir_t::FH_DIR_UL;

        // Port mask: explicit digital-BF interfaces use a contiguous mask;
        // dynamic mMIMO BFW derives it from the DMRS ports and tracks eAxC ids.
        if (mmimo && bf.dig_bf_interfaces != 0u)
        {
            // Guard against UB: shifting by the full type width (dig_bf_interfaces
            // == 64) is undefined and on x86 wraps to a shift of 0, zeroing the
            // mask. A 64-antenna deployment must yield all-ones, so saturate.
            prb.common.portMask =
                (bf.dig_bf_interfaces >= std::numeric_limits<uint64_t>::digits)
                    ? ~static_cast<uint64_t>(0u)
                    : (static_cast<uint64_t>(1u) << bf.dig_bf_interfaces) - 1u;
        }
        else if (mmimo)
        {
            prb.common.portMask |= pusch_fh::dmrs_port_mask(ue.dmrsPortBmsk, ue.scid, ue.nlAbove16);
            pusch_fh::track_eaxc_ids(ue, prb.common);
        }

        if (view_->bf_enabled())
        {
            pusch_fh::copy_beam_list(prb, bf, mmimo);
            apply_bfw_coefficients(params, prb, bf, carrier, sfn, slot);
        }

        if (ru == ::SINGLE_SECT_MODE)
        {
            // UL symbol span/start from TDD slot detail (max_ul_symbols /
            // start_sym_ul), with a full-slot @ symbol-0 fallback when slot detail
            // is unavailable.  Legacy parity: the PRB is listed at its start symbol
            // only, the full span lives in numSymbols.
            prb.common.numSymbols = single_sect_num_sym;
            pusch_fh::append_prb_to_symbols(*sym_prbs, index, single_sect_start,
                                            slot_command_api::channel_type::PUSCH, ru);
        }
        else
        {
            prb.common.numSymbols = pdu.num_of_symbols;
            pusch_fh::append_prb_to_symbols(*sym_prbs, index, pdu.start_symbol_index,
                                            slot_command_api::channel_type::PUSCH, ru);
        }
    }
    else if (mmimo && bf.dig_bf_interfaces == 0u)
    {
        // Existing mMIMO dynamic-BFW group (non-single-sector): merge this UE's
        // port mask into the group's PRB entry; never append a new PRB or rewrite
        // symbols.
        //
        // SINGLE_SECT_MODE intentionally never reaches this branch: the early-out
        // above returns once the slot's single full-bandwidth UL entry exists, so a
        // 2nd UE's per-UE port mask is deliberately NOT merged in single-sector.
        // This is faithful to legacy update_fh_params_pusch, whose single-sector
        // merge predicate (startPrbc == 0 && numPrbc == ul_bandwidth) was likewise
        // unreachable behind the same early-out (dead code there too).  See MR-5330
        // review A-4 (Codex-2): kept legacy-faithful on purpose — this temporary FH
        // block is slated for removal in a follow-up MR.
        //
        // grp.puschStartSym is copied verbatim from the wire start_symbol_index, so
        // bound it against the fixed 14-entry symbol map before indexing.  A
        // malformed PDU with start_symbol_index past the last OFDM symbol would
        // otherwise read out of bounds (append_prb_to_symbols clamps the same index
        // on the new-group path).  Marked [[unlikely]] so the valid-input hot path
        // is not penalised.
        if (grp.puschStartSym >= sym_prbs->symbols.size()) [[unlikely]]
        {
            NVLOGE_FMT(detail::k_tag, AERIAL_L2ADAPTER_EVENT,
                       "PuschPduParser::update_fh_params: carrier={} sfn={} slot={} "
                       "puschStartSym={} >= symbols-per-slot={}; skipping port-mask "
                       "merge for this UE group",
                       carrier, sfn, slot, grp.puschStartSym, sym_prbs->symbols.size());
            return;
        }
        auto& idx_list = sym_prbs->symbols[grp.puschStartSym][slot_command_api::channel_type::PUSCH];
        const auto iter = std::ranges::find_if(idx_list,
            [&](std::size_t e) noexcept
            {
                if (e >= sym_prbs->prbs_size) { return false; }
                const auto& prb = prbs[e];
                return prb.common.startPrbc == grp.startPrb && prb.common.numPrbc == grp.nPrb;
            });
        if (iter != idx_list.end())
        {
            auto& prb = prbs[*iter];
            prb.common.portMask |= pusch_fh::dmrs_port_mask(ue.dmrsPortBmsk, ue.scid, ue.nlAbove16);
            pusch_fh::track_eaxc_ids(ue, prb.common);
        }
    }
}

template<PuschModuleView V, PuschUeGroupingPolicy G>
void PuschPduParser<V, G>::apply_bfw_coefficients(
    slot_command_api::pusch_params&  params,
    slot_command_api::prb_info_t&    prb,
    const scf_fapi_rx_beamforming_t& bf,
    uint32_t                         carrier,
    uint16_t sfn, uint16_t slot) noexcept
{
    // Dynamic BFW (no explicit digital-BF interfaces) is the only path that
    // carries externally computed coefficient buffers.
    if (bf.dig_bf_interfaces != 0u) { return; }

    auto* mem = view_->bfw_coeff_mem_info(carrier, static_cast<uint8_t>(slot));
    if (mem == nullptr || mem->header == nullptr) { return; }
    if (*mem->header != slot_command_api::BFW_COFF_MEM_BUSY) { return; }
    if (!pusch_fh::bfw_coeff_fresh(sfn, slot, mem->sfn, mem->slot)) { return; }

    // BFW chunks live in per-cell memory (view_->bfw_coeff_mem_info(carrier, ...)
    // selects bfwCoeff_mem_info[carrier]), and the producer fills
    // buff_addr_chunk_h/d[uegIdx] with a 0-based *per-cell* UE-group ordinal.
    // Index by the per-cell group count (not the slot-global nUeGrps, which over-
    // shoots into foreign chunks for the 2nd+ cell). nue_grps_per_cell[carrier]
    // was incremented for this group in setup_ue_group(); subtract one for the
    // 0-based index, mirroring legacy bfwUeGrpIndex (= nue_grps_per_cell[cell]
    // captured pre-increment).
    const std::size_t per_cell_grp = params.nue_grps_per_cell[carrier] == 0u
                                         ? 0u
                                         : params.nue_grps_per_cell[carrier] - 1u;
    const auto bfw_grp =
        std::min<std::size_t>(per_cell_grp, slot_command_api::MAX_DL_UL_BF_UE_GROUPS - 1u);

    prb.common.extType                   = pusch_fh::k_bfw_ext_type;
    prb.bfwCoeff_buf_info.num_prgs        = bf.num_prgs;
    prb.bfwCoeff_buf_info.prg_size        = bf.prg_size;
    prb.bfwCoeff_buf_info.dig_bf_interfaces = bf.dig_bf_interfaces;
    prb.bfwCoeff_buf_info.nGnbAnt         = mem->nGnbAnt;
    prb.bfwCoeff_buf_info.header          = mem->header;
    prb.bfwCoeff_buf_info.p_buf_bfwCoef_h = mem->buff_addr_chunk_h[bfw_grp];
    prb.bfwCoeff_buf_info.p_buf_bfwCoef_d = mem->buff_addr_chunk_d[bfw_grp];
}

template<PuschModuleView V, PuschUeGroupingPolicy G>
void PuschPduParser<V, G>::setup_maintenance_and_extension(
    slot_command_api::pusch_params& params,
    const pdu_t&                    pdu,
    const uint8_t*                  next,
    bool                            csi_part2_signaled) noexcept
{
    auto& ue = params.ue_info[params.cell_grp_info.nUes];

    const auto* maint = aerial::casts::assume_cast<scf_fapi_pusch_maintenance_t>(next);
    next += sizeof(scf_fapi_pusch_maintenance_t);

    // CSI-Part2 defaults; overwritten by parse_csi_part2 when present.
    if (ue.pUciPrms != nullptr)
    {
        ue.pUciPrms->nCsi2Reports      = 0u;
        ue.pUciPrms->pCalcCsi2SizePrms = nullptr;
    }

    if (csi_part2_signaled && (pdu.pdu_bitmap & k_bitmap_uci) != 0u)
    {
        next = parse_csi_part2(params, next);
    }

    // PUSCH extension (weighted-average CFO + LDPC controls).
    if (view_->enable_weighted_avg_cfo())
    {
        const auto* ext = aerial::casts::assume_cast<scf_fapi_pusch_extension_t>(next);
        ue.foForgetCoeff             = static_cast<float>(ext->fo_forget_coeff) / k_fo_forget_coeff_scale;
        ue.ldpcEarlyTerminationPerUe = ext->ldpc_early_termination;
        ue.ldpcMaxNumItrPerUe        = ext->n_iterations;
        next += sizeof(scf_fapi_pusch_extension_t);
    }

    // DFT-s-OFDM fallback: when transform precoding is on but no dftsOfdm
    // section was present, the hopping config comes from maintenance.
    if (pdu.transform_precoding == 0u && (pdu.pdu_bitmap & k_bitmap_dftsofdm) == 0u)
    {
        ue.groupOrSequenceHopping = maint->groupOrSequenceHopping;
        ue.N_symb_slot            = k_n_symb_per_slot;
        ue.lowPaprGroupNumber     = 0u;
        ue.lowPaprSequenceNumber  = 0u;
    }
}

template<PuschModuleView V, PuschUeGroupingPolicy G>
const uint8_t* PuschPduParser<V, G>::parse_csi_part2(slot_command_api::pusch_params& params,
                                                     const uint8_t*                  next) noexcept
{
    // Current UE index, consistent with the other parse() sub-steps (the UE is
    // committed by parse() only after all sub-steps succeed).
    const auto  ue_idx     = params.cell_grp_info.nUes;
    const auto* csip2_info = aerial::casts::assume_cast<scf_uci_csip2_info_t>(next);
    auto&       ue         = params.ue_info[ue_idx];
    ue.pUciPrms = &params.uci_info[ue_idx];
    auto* up    = ue.pUciPrms;

    const uint16_t num_parts = csip2_info->numPart2s;
    if (num_parts == 0u)
    {
        // Faithful port of the SCF_FAPI_10_04 legacy update_cell_command path
        // (scf_5g_slot_commands.cpp:409): with no CSI-Part2 reports the cursor
        // is intentionally left at the scf_uci_csip2_info_t header rather than
        // advanced past it.
        //
        // Invariant (matches legacy, validated in production): a signalled
        // CSI-Part2 with numPart2s == 0 is not expected to co-occur with the
        // weighted-average-CFO extension (view_->enable_weighted_avg_cfo()).
        // If it ever did, setup_maintenance_and_extension() would read the
        // extension struct from this same header position. This no-advance
        // behavior is kept in lock-step with legacy; revisit here (and add the
        // sizeof(scf_uci_csip2_info_t) advance) only if that wire-format
        // invariant changes.
        //
        // Detect (don't abort) a violation of that invariant: if the extension
        // is also enabled this slot, setup_maintenance_and_extension() will read
        // scf_fapi_pusch_extension_t from this un-advanced header offset and
        // misparse it.  Log at error severity (alarmable L2-adapter event) so the
        // wire-format regression is surfaced rather than lost in warning noise;
        // legacy no-advance behavior is intentionally retained (see above).
        if (view_->enable_weighted_avg_cfo()) [[unlikely]]
        {
            NVLOGE_FMT(detail::k_tag, AERIAL_L2ADAPTER_EVENT,
                       "PuschPduParser::parse_csi_part2: rnti=0x{:04X} CSI-Part2 signalled with "
                       "numPart2s==0 co-occurs with weighted-avg-CFO extension; cursor not advanced "
                       "past the CSI-Part2 header, so the extension will be read from the wrong offset",
                       ue.rnti);
        }
        up->nCsi2Reports      = 0u;
        up->pCalcCsi2SizePrms = nullptr;
        return next;
    }

    next += sizeof(scf_uci_csip2_info_t);
    // Publish nCsi2Reports as the number of reports actually written below, not the
    // raw wire numPart2s.  The loop still iterates all num_parts entries to advance
    // the payload cursor, but only in-range reports (i < CUPHY_MAX_N_CSI2_REPORTS_PER_UE)
    // are written and counted — so the published count can never exceed the backing
    // csip2_v3_params window, and downstream [0, nCsi2Reports) iteration stays within
    // the initialized entries.
    up->nCsi2Reports      = 0u;
    up->pCalcCsi2SizePrms = &params.csip2_v3_params[
        static_cast<std::size_t>(ue_idx) * CUPHY_MAX_N_CSI2_REPORTS_PER_UE];

    std::size_t offset = 0u;
    for (uint16_t i = 0u; i < num_parts; ++i)
    {
        const auto* part = aerial::casts::assume_cast<scf_uci_csip2_part_t>(next + offset);
        const uint8_t num_p1 = part->numPart1Params;
        offset += sizeof(scf_uci_csip2_part_t);

        const auto* p_offsets = aerial::casts::assume_cast<scf_uci_csip2_part_param_offset_t>(next + offset);
        const auto* p_sizes   = aerial::casts::assume_cast<scf_uci_csip2_part_param_size_t>(
            next + offset + static_cast<std::size_t>(num_p1) * sizeof(uint16_t));

        const bool report_in_range = (i < CUPHY_MAX_N_CSI2_REPORTS_PER_UE);
        uint8_t    n_p1_written    = 0u;
        for (uint8_t j = 0u; j < num_p1; ++j)
        {
            // Guard the fixed-size prmOffsets/prmSizes arrays (validation deferred).
            if (report_in_range && j < CUPHY_MAX_N_CSI1_PRMS)
            {
                up->pCalcCsi2SizePrms[i].prmOffsets[j] = p_offsets->paramOffsets[j];
                up->pCalcCsi2SizePrms[i].prmSizes[j]   = p_sizes->paramSizes[j];
                ++n_p1_written;
            }
        }

        offset += static_cast<std::size_t>(num_p1) * (sizeof(uint16_t) + sizeof(uint8_t));
        const auto* scope = aerial::casts::assume_cast<scf_uci_csip2_part_scope_t>(next + offset);

        if (report_in_range)
        {
            // Publish the number of Part-1 params actually written (capped at
            // CUPHY_MAX_N_CSI1_PRMS), not the raw wire numPart1Params, so downstream
            // [0, nPart1Prms) iteration stays within prmOffsets[]/prmSizes[].
            up->pCalcCsi2SizePrms[i].nPart1Prms    = n_p1_written;
            up->pCalcCsi2SizePrms[i].csi2sizeMapIdx = scope->part2SizeMapIndex;
            ++up->nCsi2Reports;
        }
        offset += sizeof(scf_uci_csip2_part_scope_t);
    }

    next += offset;
    return next;
}

template<PuschModuleView V, PuschUeGroupingPolicy G>
void PuschPduParser<V, G>::setup_cell(const scf_fapi_ul_tti_req_t& req,
                                      uint32_t                     msg_cell_id) noexcept
{
    // noexcept contract: view_->slot_command() and view_->cell_sub_command()
    // index slot_command_array via std::vector/array::at(), which throws
    // std::out_of_range on a bad slot/cell index.  Catch internally so a throw
    // cannot escape this noexcept function (which would call std::terminate),
    // mirroring PdschPduParser::setup_cell.
    try
    {
        // FH runtime gate (cached once per UL_TTI): in fapi_to_cplane_direct mode the
        // framework builds the O-RAN C-plane directly from FAPI, so the parser must
        // skip the sym_prb_info FH-metadata fill.  Set before any early return so it
        // is always fresh for the parse() calls that follow.
        fh_enabled_for_msg_ = !view_->fapi_to_cplane_direct_enabled();

        auto* params = view_->slot_command().cell_groups.get_pusch_params();
        if (!params) [[unlikely]] { return; }

        // dst_idx is the running destination slot in pusch_params (cell_dyn_info[],
        // cellPrmDynIdx) — the position of this cell in registration order, NOT the
        // identity of the cell.  The UL slot processor skips messages with no PUSCH
        // PDUs, so nCells advances only over PUSCH-bearing cells and does not track
        // msg_cell_id.
        const auto dst_idx = static_cast<uint32_t>(params->cell_grp_info.nCells);
        if (dst_idx >= static_cast<uint32_t>(slot_command_api::MAX_CELLS_PER_CELL_GROUP)) [[unlikely]]
        {
            NVLOGW_FMT(detail::k_tag,
                       "PuschPduParser::setup_cell: nCells={} >= MAX_CELLS_PER_CELL_GROUP; skipping",
                       dst_idx);
            return;
        }

        // The cell identity is the UL_TTI message's logical cell id, not the
        // processing order.  Source lookups (carrier_id / cell_stat_prm_idx) index
        // PHY_instances()[msg_cell_id], so they must key off msg_cell_id — mirroring
        // the SRS/PRACH setup_cell(*req, msg.cell_id) path.  Using dst_idx (nCells)
        // here would register PUSCH under the wrong carrier whenever a lower-id cell
        // in the same slot carried no PUSCH.  Bounds-check first (unchecked
        // operator[] into PHY_instances()).
        if (msg_cell_id >= static_cast<uint32_t>(slot_command_api::MAX_CELLS_PER_CELL_GROUP)) [[unlikely]]
        {
            NVLOGW_FMT(detail::k_tag,
                       "PuschPduParser::setup_cell: msg_cell_id={} >= MAX_CELLS_PER_CELL_GROUP; skipping",
                       msg_cell_id);
            return;
        }

        // carrier_id is a bounded 0-based logical carrier index
        // (phy_config.cell_config_.carrier_idx, assigned at cell config time), used
        // as the key into the per-cell arrays cell_ue_group_idx_start[] and the
        // cells[] sub-command list.
        const auto carrier = view_->carrier_id(msg_cell_id);

        // Bounds-check carrier before using it as an index into the fixed-size
        // cell_ue_group_idx_start[] / cells[] arrays.  carrier_id() returns a
        // signed int32_t sourced from L2 config; a negative or >= capacity value
        // would be an out-of-bounds write into pusch_params.  Defensive against
        // L2 bugs, mirroring the duplicate-carrier guard below.
        if (carrier < 0 ||
            carrier >= static_cast<int32_t>(slot_command_api::MAX_CELLS_PER_CELL_GROUP)) [[unlikely]]
        {
            NVLOGW_FMT(detail::k_tag,
                       "PuschPduParser::setup_cell: carrier_id={} out of range [0,{}); skipping",
                       carrier, slot_command_api::MAX_CELLS_PER_CELL_GROUP);
            return;
        }

        // Detect duplicate carrier — skip if already registered (defensive against L2 bugs).
        if (std::find(params->cell_index_list.begin(), params->cell_index_list.end(), carrier)
            != params->cell_index_list.end())
        {
            NVLOGW_FMT(detail::k_tag,
                       "PuschPduParser::setup_cell: duplicate carrier_id={} - skipping",
                       carrier);
            return;
        }

        // Record UE-group start index for this cell.
        params->cell_ue_group_idx_start[carrier] = params->cell_grp_info.nUeGrps;

        // Initialize cell dynamic info at the running destination slot (dst_idx),
        // sourcing the static-param index from the message's cell id.
        auto& cell_dyn = params->cell_dyn_info[dst_idx];
        cell_dyn.slotNum = (view_->staticPuschSlotNum() > -1)
                            ? static_cast<uint16_t>(view_->staticPuschSlotNum())
                            : req.slot;
        cell_dyn.cellPrmDynIdx  = static_cast<uint16_t>(dst_idx);
        cell_dyn.cellPrmStatIdx = view_->cell_stat_prm_idx(msg_cell_id);

        // Register cell in index lists. cell_index_list holds the logical
        // carrier id; phy_cell_index_list must hold the *physical* cell id
        // (phy::get_phy_cell_params().phyCellId) because cuphydriver resolves
        // cells via getCellByPhyId() against this list. Use the dedicated
        // phy_cell_id() accessor — mirroring PdschPduParser::setup_cell — rather
        // than cell_sub_command().cell, which is not populated in the
        // ULSlotProcessor flow and reads back 0 ("Cell 0 is not present").
        params->cell_index_list.push_back(carrier);
        // phy_cell_id() is uint16_t (PCI 0-1007); widening to the list's int32_t
        // element type is value-preserving, so no explicit cast is needed.
        params->phy_cell_index_list.push_back(view_->phy_cell_id(msg_cell_id));
        ++params->cell_grp_info.nCells;
    }
    catch (const std::exception& ex)
    {
        NVLOGE_FMT(detail::k_tag, AERIAL_L2ADAPTER_EVENT,
                   "PuschPduParser::setup_cell sfn={} slot={} threw: {}",
                   req.sfn, req.slot, ex.what());
    }
    catch (...)
    {
        NVLOGE_FMT(detail::k_tag, AERIAL_L2ADAPTER_EVENT,
                   "PuschPduParser::setup_cell sfn={} slot={} threw unknown exception",
                   req.sfn, req.slot);
    }
}

} // namespace scf_5g_fapi

#endif // SCF_5G_FAPI_PUSCH_PDU_PARSER_HPP_INCLUDED_
