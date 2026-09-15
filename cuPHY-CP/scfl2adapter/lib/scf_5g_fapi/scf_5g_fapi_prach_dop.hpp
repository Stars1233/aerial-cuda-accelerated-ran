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

#if !defined(SCF_5G_FAPI_PRACH_DOP_HPP_INCLUDED_)
#define SCF_5G_FAPI_PRACH_DOP_HPP_INCLUDED_

/**
 * @file scf_5g_fapi_prach_dop.hpp
 * @brief PRACH Tell-Don't-Ask helpers — adapter-local DOP service layer.
 *
 * MR 2a of the PRACH parser stack (GT-11843).
 *
 * Encapsulates the field-level invariants that the legacy PRACH path in
 * scf_5g_slot_commands.cpp enforces by caller discipline:
 *   * prach_params::nOccasion must be incremented after the per-occasion
 *     writes to rach[i] / freqIndex[i] / startSymbols[i] / cell_index_list /
 *     phy_cell_index_list.
 *   * prach_params::mu and ::nfft must be set as a pair.
 *   * slot_info::type (= SLOT_UPLINK) and ::slot_3gpp must be set together.
 *   * slot_info_t::prbs_size must be incremented after writing prbs[i].
 *   * prb_info_t per-occasion fields (freqOffset / numSymbols / direction /
 *     filterIndex) must all be initialised before the entry is consumed.
 *
 * These are exposed as **free functions** inside scf_5g_fapi rather than as
 * methods on the shared slot_command_api types because slot_command.hpp lives
 * in gt_common_libs and is consumed by cuphydriver / cuphycontroller / testMAC
 * — modifying its public API would widen the blast radius of this change
 * beyond what's appropriate for an adapter-layer refactor.
 *
 * cubb-review §12.1 (Tell-Don't-Ask): the invariants are owned by these
 * functions, not by caller discipline. §12.2 explicitly allows free functions
 * that operate on a type's public surface to be the encapsulation site when
 * methods-on-type isn't viable.
 *
 * All helpers are inline + noexcept; designed for hot-path call sites in the
 * new PrachPduParser. Designated-initializer construction of OccasionEntry at
 * call sites eliminates silent transposition of the four scalar fields
 * without requiring a NamedType dependency.
 */

#include <cstdint>

#include "scf_5g_fapi_tags.hpp"            // detail::k_tag + nvlog macros
#include "slot_command/slot_command.hpp"

namespace scf_5g_fapi::prach::dop
{

// ---------------------------------------------------------------------------
// OccasionEntry — value type for add_occasion()
// ---------------------------------------------------------------------------

/**
 * Per-occasion fields for prach_params::rach[i] / freqIndex[i] / startSymbols[i].
 *
 * Members are declared in size-descending order (4 B → 2 B → 1 B) so the
 * struct packs to 16 B with zero internal padding. At up to 7 occasions per
 * PRACH PDU, the stack footprint stays within two cache lines on every cuBB
 * target (x86-64, ARM Neoverse V2, ARM A78 — all 64 B lines).
 *
 * Designated-initializer call sites — e.g.
 *   add_occasion(*params, {.cell_index = c, .force_thr0 = t, .freq_index = f, ...})
 * — eliminate silent transposition of same-sized fields. C++20 requires
 * designated init to follow declaration order, so call sites must list
 * fields top-to-bottom as defined here.
 */
struct OccasionEntry final
{
    int32_t  cell_index       {};  //!< Logical cell index
    float    force_thr0       {};  //!< 0 = use cuPHY default threshold

    uint16_t phy_cell_id      {};  //!< Physical cell id (from cell_params)
    uint16_t occa_prm_stat_idx{};  //!< cell_params.start_ro_index + pdu.num_ra
    uint16_t n_uplink_streams {};  //!< 0 unless (mmimo && dig_bf_interfaces != 0)

    uint8_t  freq_index       {};  //!< pdu.num_ra
    uint8_t  start_symbol     {};  //!< pdu.prach_start_symbol
};

// Lock the layout: any future field addition that re-introduces padding fails
// at compile time, forcing the author to either keep the struct compact or
// consciously accept the growth.
static_assert(sizeof(OccasionEntry) == 16,
              "OccasionEntry must stay packed at 16 B for cache coherence");
static_assert(alignof(OccasionEntry) == 4,
              "OccasionEntry should align to its largest scalar (int32/float)");

// ---------------------------------------------------------------------------
// prach_params operations
// ---------------------------------------------------------------------------

/**
 * Capacity predicate.
 *
 * @param[in] params  prach_params to inspect.
 * @return  true if no more occasions can be added; false otherwise.
 *          Return value must be checked.
 */
[[nodiscard]] inline bool full(const slot_command_api::prach_params& params) noexcept
{
    return params.nOccasion >= slot_command_api::MAX_PRACH_OCCASIONS_PER_SLOT;
}

/**
 * Append one PRACH occasion. Owns the nOccasion++ invariant paired with the
 * rach[i] / freqIndex[i] / startSymbols[i] / cell_index_list /
 * phy_cell_index_list writes.
 *
 * @param[in,out] params  prach_params to mutate.
 * @param[in]     e       Occasion fields.
 * @return  true on success; false if @p params is full() (slot at capacity).
 *          Return value must be checked.
 */
[[nodiscard]] inline bool
add_occasion(slot_command_api::prach_params& params, const OccasionEntry& e) noexcept
{
    if (full(params)) [[unlikely]] {
        NVLOGE_FMT(detail::k_tag, AERIAL_L2ADAPTER_EVENT,
                   "prach::add_occasion: refused (slot at capacity); cell_index={} "
                   "nOccasion={} == MAX_PRACH_OCCASIONS_PER_SLOT",
                   e.cell_index, params.nOccasion);
        return false;
    }
    const auto i = params.nOccasion;
    params.cell_index_list.push_back(e.cell_index);
    params.phy_cell_index_list.push_back(static_cast<int32_t>(e.phy_cell_id));
    params.freqIndex[i]            = e.freq_index;
    params.startSymbols[i]         = e.start_symbol;
    params.rach[i].occaPrmStatIdx  = e.occa_prm_stat_idx;
    params.rach[i].occaPrmDynIdx   = i;
    params.rach[i].force_thr0      = e.force_thr0;
    params.rach[i].nUplinkStreams  = e.n_uplink_streams;
    ++params.nOccasion;
    return true;
}

/**
 * Set the cell-wide mu + nfft pair atomically. Fluent (returns the params ref).
 *
 * @param[in,out] params   prach_params to mutate.
 * @param[in]     mu_in    PRACH subcarrier-spacing numerology.
 * @param[in]     nfft_in  FFT size (short-format vs long-format).
 * @return  Reference to @p params for chaining.
 */
inline slot_command_api::prach_params&
set_global_config(slot_command_api::prach_params& params,
                  const uint8_t mu_in, const uint32_t nfft_in) noexcept
{
    params.mu   = mu_in;
    params.nfft = nfft_in;
    return params;
}

// ---------------------------------------------------------------------------
// slot_info operations
// ---------------------------------------------------------------------------

/**
 * Set both slot_info fields atomically for an uplink slot. Owns the
 * type + slot_3gpp invariant.
 *
 * @param[in,out] slot  slot_info to mutate (typically cell_cmd.slot or
 *                      group_cmd.slot).
 * @param[in]     ind   3GPP timing for this slot.
 */
inline void set_uplink(slot_command_api::slot_info& slot,
                       const slot_command_api::slot_indication& ind) noexcept
{
    slot.type      = slot_command_api::SLOT_UPLINK;
    slot.slot_3gpp = ind;
}

// ---------------------------------------------------------------------------
// slot_info_t (sym_prb_info) operations
// ---------------------------------------------------------------------------

/**
 * Capacity predicate for the per-slot PRB-info array.
 *
 * @param[in] sym_prbs  slot_info_t to inspect.
 * @return  true if no more entries can be added; false otherwise.
 *          Return value must be checked.
 */
[[nodiscard]] inline bool full(const slot_command_api::slot_info_t& sym_prbs) noexcept
{
    return sym_prbs.prbs_size >= MAX_PRB_INFO;
}

/**
 * Append one prb_info_t to the per-slot array. Owns the prbs_size++ invariant.
 *
 * Returns a reference to the appended entry so callers can fill in beamforming /
 * port-mask fields that depend on per-PDU state without re-indexing — and
 * without an "is the pointer null?" check at every call site.
 *
 * @pre @p sym_prbs must not be full(). The caller is responsible for checking
 *      full() before invoking add_prb; violation triggers gsl_Expects
 *      (configured to throw, which terminates this noexcept function).
 *
 * @param[in,out] sym_prbs  slot_info_t to mutate.
 * @param[in]     info      Fully-populated common fields. Beamforming /
 *                          portMask may be edited via the returned reference.
 * @return  Reference to the just-added entry (always valid by precondition).
 */
[[nodiscard]] inline slot_command_api::prb_info_t&
add_prb(slot_command_api::slot_info_t& sym_prbs,
        const slot_command_api::prb_info_t& info) noexcept
{
    gsl_Expects(!full(sym_prbs));  // contract: caller must check full() first
    auto& dst = sym_prbs.prbs[sym_prbs.prbs_size];
    dst = info;
    ++sym_prbs.prbs_size;
    return dst;
}

// ---------------------------------------------------------------------------
// prb_info_t factory
// ---------------------------------------------------------------------------

/**
 * Construct a prb_info_t with the per-occasion PRACH common fields populated.
 *
 * Replaces the inline four-field initialisation that the legacy
 * update_fh_params_prach (scf_5g_slot_commands.cpp:1867-1892) repeated at every
 * iteration of its per-occasion loop. Caller may still edit beams_array /
 * portMask on the returned value before passing it to add_prb().
 *
 * @param[in] startPrb     PRB start (PRACH currently always 0).
 * @param[in] numPrb       Number of PRBs (= addln_config.n_ra_rb).
 * @param[in] freqOffset   prach_config.root_sequence[num_ra].freqOffset.
 * @param[in] numSymbols   addln_config.n_ra_dur.
 * @param[in] filterIndex  Format → filter mapping (see k_prach_format_to_filter_index
 *                         once MR 3a lands).
 * @return  prb_info_t with common fields populated; beams_array and portMask
 *          remain default (caller fills based on PDU bf state).
 */
[[nodiscard]] inline slot_command_api::prb_info_t
make_prach_prb(const uint16_t startPrb,
               const uint16_t numPrb,
               const int32_t  freqOffset,
               const uint8_t  numSymbols,
               const uint8_t  filterIndex) noexcept
{
    slot_command_api::prb_info_t p{startPrb, numPrb};
    p.common.freqOffset  = freqOffset;
    p.common.numSymbols  = numSymbols;
    p.common.direction   = slot_command_api::fh_dir_t::FH_DIR_UL;
    p.common.filterIndex = filterIndex;
    return p;
}

} // namespace scf_5g_fapi::prach::dop

#endif // SCF_5G_FAPI_PRACH_DOP_HPP_INCLUDED_
