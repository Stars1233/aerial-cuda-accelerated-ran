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

#if !defined(SCF_5G_FAPI_PDSCH_FH_FILL_HPP_INCLUDED_)
#define SCF_5G_FAPI_PDSCH_FH_FILL_HPP_INCLUDED_

#include <cstdint>
#include <cstring>
#include <type_traits>
#include <utility>

#include "scf_5g_fapi.h"
#include "scf_5g_fapi_tags.hpp"
#include "slot_command/slot_command.hpp"
#include "nvlog.h"
#include "nvlog_fmt.hpp"
#include "aerial_event_code.h"

namespace scf_5g_fapi
{

/**
 * Aggregates the per-PDU inputs needed to populate one
 * @c slot_command_api::pdsch_fh_prepare_params + parallel
 * @c slot_command_api::tx_precoding_beamforming_t entry.
 *
 * Mirrors the inputs consumed by legacy @c update_cell_command in
 * @c scf_5g_slot_commands_pdsch_csirs.cpp.
 *
 * Callers must apply the legacy mMIMO null-BFW guard before constructing this
 * struct: when @c mmimo_enabled() is true and @c pm_bf.dig_bf_interfaces is 0,
 * pass @c bfw = nullptr.
 *
 * The four feature flags (is_new_grp/bf_enabled/pm_enabled/mmimo_enabled) are
 * packed into a single @c uint8_t @c flags field rather than separate @c bool
 * members. Construction uses the @c k_* bit constants below.
 */
struct FhFillArgs
{
    // ── Flag-bit positions ────────────────────────────────────────────
    static constexpr uint8_t k_is_new_grp    = 1u << 0; //!< This PDU created its UE group.
    static constexpr uint8_t k_bf_enabled    = 1u << 1; //!< Static BF enable.
    static constexpr uint8_t k_pm_enabled    = 1u << 2; //!< Static PM enable.
    static constexpr uint8_t k_mmimo_enabled = 1u << 3; //!< Runtime mMIMO flag.

    slot_command_api::cell_sub_command&         cell_cmd;     //!< Per-cell sub-command (live).
    cuphyPdschUeGrpPrm_t&                       ue_grp;       //!< UE-group params for this PDU.
    cuphyPdschUePrm_t&                          ue;           //!< UE params for this PDU.
    const scf_fapi_tx_precoding_beamforming_t&  pm_bf;        //!< Parsed Tx-Precoding+Beamforming PDU.
    slot_command_api::bfw_coeff_mem_info_t*     bfw;          //!< BFW ring entry; null for dynamic BFW.
    uint32_t                                    bfw_idx;      //!< @c ue_grp_idx_bfw_id_map entry for this group.
    uint16_t                                    ue_grp_index; //!< Group index within @c params.ue_grp_info.
    uint16_t                                    num_dl_prb;   //!< DL BWP PRB count for this cell.
    uint16_t                                    cell_index;   //!< Logical carrier id.
    uint8_t                                     flags{};      //!< Bit-or of @c k_* constants above.

    // ── Bitmask accessors (zero-cost, inline) ──────────────────────────
    [[nodiscard]] bool is_new_grp()    const noexcept { return (flags & k_is_new_grp)    != 0u; }
    [[nodiscard]] bool bf_enabled()    const noexcept { return (flags & k_bf_enabled)    != 0u; }
    [[nodiscard]] bool pm_enabled()    const noexcept { return (flags & k_pm_enabled)    != 0u; }
    [[nodiscard]] bool mmimo_enabled() const noexcept { return (flags & k_mmimo_enabled) != 0u; }
};

/**
 * Pack four feature-flag bools into one @c FhFillArgs::flags byte at the
 * call site. Constexpr so the optimizer folds construction.
 */
[[nodiscard]] constexpr uint8_t make_fh_flags(bool is_new_grp,
                                              bool bf_enabled,
                                              bool pm_enabled,
                                              bool mmimo_enabled) noexcept
{
    return static_cast<uint8_t>(
        (is_new_grp    ? FhFillArgs::k_is_new_grp    : 0u) |
        (bf_enabled    ? FhFillArgs::k_bf_enabled    : 0u) |
        (pm_enabled    ? FhFillArgs::k_pm_enabled    : 0u) |
        (mmimo_enabled ? FhFillArgs::k_mmimo_enabled : 0u));
}

/**
 * No-op callable used as the disabled-path dispatch target for
 * @c append_fh_entry. Passing @c NullFhCallable{} causes the function body to
 * be eliminated at compile time via the @c if constexpr early-out, leaving
 * only a @c ret instruction in the specialization.
 *
 * Used when @c is_fapi_to_cplane_direct_enabled() == true: C-plane is created
 * by other worker threads, so the parser must not populate @c fh_params.
 *
 * @note Must be a named type, not an empty lambda. @c append_fh_entry uses
 *       @c std::is_same_v<NF, NullFhCallable> as the if-constexpr tag.
 *       A lambda has a unique anonymous type → the tag check fails → the
 *       runtime branch runs and the disabled-mode invariants are violated:
 *         - @c fh.num_pdsch_fh_params[cell_index] gets incremented
 *         - @c fh.total_num_pdsch_pdus gets incremented
 *         - @c NVLOGD_FMT fires per call
 *       In direct mode the C-plane is built by other workers and these
 *       counters must remain untouched. Do not replace with a lambda.
 */
struct NullFhCallable
{
    void operator()(slot_command_api::pdsch_fh_prepare_params&,
                    slot_command_api::tx_precoding_beamforming_t&) const noexcept
    {
    }
};

/**
 * Shared low-level primitive: capacity check, counter increment, per-cell
 * index update, and invocation of the channel-specific fill callable.
 *
 * Specialization with @c FillFn = @c NullFhCallable collapses to a @c ret;
 * the optimizer drops the entire body via the @c if constexpr early-out.
 *
 * Designed for reuse by future CSI-RS fill (different callable, same primitive).
 *
 * @tparam FillFn   Callable with signature
 *                  @c void(pdsch_fh_prepare_params&, tx_precoding_beamforming_t&) noexcept.
 *
 * @param[in,out] fh          Active FH callback params (in @c cell_group_command::fh_params).
 * @param[in]     cell_index  Logical cell index used to increment @c num_pdsch_fh_params.
 * @param[in]     fill        Callable that writes the per-entry fields.
 * @return                    True on success; false when the per-slot capacity
 *                            (@c MAX_ALLOWED_PDSCH_PDUS_PER_SLOT) is exhausted.
 *                            The @c NullFhCallable specialization always returns true.
 *                            Return value must be checked.
 */
template <typename FillFn>
[[nodiscard]] bool append_fh_entry(
    [[maybe_unused]] slot_command_api::fh_prepare_callback_params& fh,
    [[maybe_unused]] uint16_t                                       cell_index,
    [[maybe_unused]] FillFn&&                                       fill) noexcept
{
    using NF = std::remove_cvref_t<FillFn>;
    if constexpr (std::is_same_v<NF, NullFhCallable>)
    {
        return true;
    }
    else
    {
        if (fh.total_num_pdsch_pdus
                >= slot_command_api::MAX_ALLOWED_PDSCH_PDUS_PER_SLOT) [[unlikely]]
        {
            NVLOGE_FMT(detail::k_tag, AERIAL_L2ADAPTER_EVENT,
                       "append_fh_entry: cell_index={} total_num_pdsch_pdus={} >= "
                       "MAX_ALLOWED_PDSCH_PDUS_PER_SLOT={}; dropping FH entry",
                       cell_index, fh.total_num_pdsch_pdus,
                       slot_command_api::MAX_ALLOWED_PDSCH_PDUS_PER_SLOT);
            return false;
        }

        const uint32_t total = fh.total_num_pdsch_pdus;
        std::forward<FillFn>(fill)(fh.pdsch_fh_params[total], fh.pc_bf_arr[total]);

        ++fh.num_pdsch_fh_params[cell_index];
        ++fh.total_num_pdsch_pdus;

        NVLOGD_FMT(detail::k_tag,
                   "append_fh_entry: cell_index={} total_num_pdsch_pdus={} "
                   "num_pdsch_fh_params[cell]={}",
                   cell_index, fh.total_num_pdsch_pdus,
                   fh.num_pdsch_fh_params[cell_index]);
        return true;
    }
}

namespace detail {

/**
 * Write the @c pdsch_fh_prepare_params header fields (pointer/index block).
 *
 * Cache rationale: this struct is ~56 B and lives in
 * @c fh_prepare_callback_params::pdsch_fh_params[i]. All writes target the
 * same one or two cache lines; grouping them here keeps store-buffer
 * coalescing high. No allocations.
 *
 * @param[out] pdsch_fh  Entry to populate.
 * @param[in]  args      Aggregated inputs.
 * @param[in]  pc_bf     Address of the parallel pc_bf entry to back-link.
 */
inline void fill_pdsch_fh_header(slot_command_api::pdsch_fh_prepare_params&    pdsch_fh,
                                 const FhFillArgs&                              args,
                                 slot_command_api::tx_precoding_beamforming_t& pc_bf) noexcept
{
    pdsch_fh.grp                       = &args.ue_grp;
    pdsch_fh.ue                        = &args.ue;
    pdsch_fh.cell_cmd                  = &args.cell_cmd;
    pdsch_fh.bfwCoeff_mem_info         = args.bfw;
    pdsch_fh.pc_bf                     = &pc_bf;
    pdsch_fh.ue_grp_bfw_index_per_cell = args.bfw_idx;
    pdsch_fh.ue_grp_index              = args.ue_grp_index;
    pdsch_fh.num_dl_prb                = args.num_dl_prb;
    pdsch_fh.cell_index                = args.cell_index;
    pdsch_fh.is_new_grp                = args.is_new_grp();
    pdsch_fh.bf_enabled                = args.bf_enabled();
    pdsch_fh.pm_enabled                = args.pm_enabled();
    pdsch_fh.mmimo_enabled             = args.mmimo_enabled();
    pdsch_fh.csirs_compact_mode        = false;
}

/**
 * Write the small fixed header at the start of @c pc_bf_arr[i]
 * (num_prgs, prg_size, dig_bf_interfaces — 5 bytes total).
 *
 * Cache rationale: @c tx_precoding_beamforming_t is @c __packed__ and
 * ~665 B per entry. The 5-byte fixed header sits at the start of the
 * entry's first cache line; the (much larger) @c pm_idx_and_beam_idx
 * array starts at byte 5 and spans multiple lines. Writing the header
 * before the array keeps the first line resident in L1 for the
 * subsequent memcpy.
 *
 * @param[out] pc_bf  Entry to populate.
 * @param[in]  pm_bf  Parsed Tx-Precoding+Beamforming PDU.
 */
inline void fill_pc_bf_header(slot_command_api::tx_precoding_beamforming_t& pc_bf,
                               const scf_fapi_tx_precoding_beamforming_t&     pm_bf) noexcept
{
    pc_bf.num_prgs          = pm_bf.num_prgs;
    pc_bf.prg_size          = pm_bf.prg_size;
    pc_bf.dig_bf_interfaces = pm_bf.dig_bf_interfaces;
}

/**
 * @brief Compute the number of @c uint16_t entries the source PM/beam list
 *        occupies.
 *
 * Formula: @c num_prgs * (1 + @c dig_bf_interfaces). Callers use the result
 * to (a) size the memcpy in @c copy_pm_idx_and_beam_idx and (b) compare
 * against the destination capacity in @c validate_pm_idx_and_beam_idx_capacity.
 *
 * @param[in] pm_bf  Source Tx-Precoding+Beamforming PDU.
 * @return Entry count in @c uint16_t units (must be checked against
 *         destination capacity by the caller — attribute @c [[nodiscard]]).
 */
[[nodiscard]] inline std::size_t pm_idx_and_beam_idx_entries(
    const scf_fapi_tx_precoding_beamforming_t& pm_bf) noexcept
{
    return static_cast<std::size_t>(pm_bf.num_prgs)
        * (1u + static_cast<std::size_t>(pm_bf.dig_bf_interfaces));
}

/**
 * @brief Compile-time capacity of the destination @c pm_idx_and_beam_idx
 *        array on the slot-command @c tx_precoding_beamforming_t.
 *
 * Derived from @c sizeof so any future change to the array declaration is
 * automatically reflected here.
 *
 * @return Number of @c uint16_t slots the destination can hold (must be
 *         used to bound the source entry count — attribute @c [[nodiscard]]).
 */
[[nodiscard]] inline constexpr std::size_t pm_idx_and_beam_idx_capacity() noexcept
{
    slot_command_api::tx_precoding_beamforming_t pc_bf{};
    return sizeof(pc_bf.pm_idx_and_beam_idx) / sizeof(pc_bf.pm_idx_and_beam_idx[0]);
}

/**
 * @brief Predicate: does this PDU require an actual PM/beam list copy?
 *
 * The mmimo dynamic-BFW path (@c mmimo_enabled && @c dig_bf_interfaces == 0)
 * carries no static PM/beam list and legitimately skips the copy — mirroring
 * the legacy @c update_cell_command dynamic-BFW skip.
 *
 * @param[in] pm_bf          Source PDU.
 * @param[in] mmimo_enabled  Runtime mMIMO flag.
 * @return @c true if the PDU carries a static PM/beam list that must be
 *         copied; @c false on the mmimo dynamic-BFW skip path (must be
 *         checked before invoking the memcpy — attribute @c [[nodiscard]]).
 */
[[nodiscard]] inline bool should_copy_pm_idx_and_beam_idx(
    const scf_fapi_tx_precoding_beamforming_t& pm_bf,
    bool                                       mmimo_enabled) noexcept
{
    return !(mmimo_enabled && pm_bf.dig_bf_interfaces == 0u);
}

/**
 * @brief Security predicate for the CWE-787 mitigation on the FH-fill path.
 *
 * Returns @c true iff either (a) the PDU legitimately skips the copy
 * (mmimo dynamic-BFW), or (b) the source entry count fits inside the
 * destination capacity. Called by @c fill_pdsch_fh_entry at entry so
 * overflowing PDUs are rejected with an explicit @c NVLOGE_FMT before any
 * fh state mutation.
 *
 * @param[in] pm_bf          Source PDU.
 * @param[in] mmimo_enabled  Runtime mMIMO flag.
 * @return @c true if the copy is safe (or skipped); @c false if the source
 *         entry count would overflow the destination array (must be checked
 *         — attribute @c [[nodiscard]]).
 */
[[nodiscard]] inline bool validate_pm_idx_and_beam_idx_capacity(
    const scf_fapi_tx_precoding_beamforming_t& pm_bf,
    bool                                       mmimo_enabled) noexcept
{
    return !should_copy_pm_idx_and_beam_idx(pm_bf, mmimo_enabled)
        || pm_idx_and_beam_idx_entries(pm_bf) <= pm_idx_and_beam_idx_capacity();
}

/**
 * @brief Copy the variable-length @c pm_idx_and_beam_idx payload from a
 *        FAPI PDU into a slot-command entry.
 *
 * Skipped on the dynamic-BFW path (@c mmimo_enabled && @c dig_bf_interfaces
 * == 0), mirroring the legacy @c update_cell_command dynamic-BFW skip. Callers must
 * pre-validate the source entry count via
 * @c validate_pm_idx_and_beam_idx_capacity — this helper trusts that the
 * copy fits.
 *
 * Cache / perf notes:
 *   - Size = @c num_prgs * (1 + @c dig_bf_interfaces) * @c sizeof(uint16_t),
 *     bounded by 660 B (numPRGs ≤ 10, digBFInterfaces ≤ 32).
 *   - @c std::memcpy is the right primitive: the destination is in a
 *     @c __packed__ struct so @c reinterpret_cast'd vector loads would be
 *     UB on strict-alignment ARM; compilers safely lower @c memcpy to
 *     byte-aware vector stores. Stride-1, hardware-prefetcher-friendly.
 *   - No allocation: source and destination are both pre-allocated.
 *
 * @param[out] pc_bf          Entry whose @c pm_idx_and_beam_idx is filled.
 * @param[in]  pm_bf          Source PDU.
 * @param[in]  mmimo_enabled  Runtime mMIMO flag (governs the skip).
 */
inline void copy_pm_idx_and_beam_idx(slot_command_api::tx_precoding_beamforming_t& pc_bf,
                                     const scf_fapi_tx_precoding_beamforming_t&    pm_bf,
                                     bool                                           mmimo_enabled) noexcept
{
    if (!should_copy_pm_idx_and_beam_idx(pm_bf, mmimo_enabled)) {
        return;
    }
    const std::size_t n_entries = pm_idx_and_beam_idx_entries(pm_bf);

    std::memcpy(pc_bf.pm_idx_and_beam_idx,
                pm_bf.pm_idx_and_beam_idx,
                n_entries * sizeof(uint16_t));
}

} // namespace detail

/**
 * PDSCH-specific FH entry fill.
 *
 * Composed of three inline helpers in @c detail::, each responsible for one
 * struct or one logical group of fields. The decomposition is purely for
 * readability and testability — every helper is @c inline and the optimizer
 * collapses the call chain to the same machine code as a single monolithic
 * function. No hot-path allocations.
 *
 * Field-write ordering is chosen for cache coherence:
 *   1. @c fill_pdsch_fh_header writes the ~56-byte header (1-2 cache lines).
 *   2. @c fill_pc_bf_header touches the start of the @c pc_bf entry's first
 *      cache line.
 *   3. @c copy_pm_idx_and_beam_idx streams sequentially through the rest of
 *      the @c pc_bf entry. With (2) above, the first @c pc_bf line is L1-resident
 *      when memcpy starts.
 *
 * Caller is responsible for the legacy mMIMO null-BFW guard: pass
 * @c args.bfw = nullptr when @c mmimo_enabled() and @c pm_bf.dig_bf_interfaces == 0.
 *
 * @param[in,out] fh    Active FH callback params (@c cell_group_command::fh_params).
 * @param[in]     args  Aggregated PDSCH FH inputs.
 * @return              True on success; false when the per-slot capacity is exhausted.
 *                      Return value must be checked.
 */
[[nodiscard]] inline bool fill_pdsch_fh_entry(
    slot_command_api::fh_prepare_callback_params& fh,
    const FhFillArgs&                              args) noexcept
{
    if (!detail::validate_pm_idx_and_beam_idx_capacity(args.pm_bf, args.mmimo_enabled())) [[unlikely]]
    {
        NVLOGE_FMT(detail::k_tag, AERIAL_L2ADAPTER_EVENT,
                   "fill_pdsch_fh_entry: PM/beam entries={} exceed destination capacity={}; "
                   "num_prgs={} dig_bf_interfaces={} mmimo_enabled={}",
                   detail::pm_idx_and_beam_idx_entries(args.pm_bf),
                   detail::pm_idx_and_beam_idx_capacity(),
                   args.pm_bf.num_prgs, args.pm_bf.dig_bf_interfaces, args.mmimo_enabled());
        return false;
    }

    return append_fh_entry(fh, args.cell_index,
        [&args](slot_command_api::pdsch_fh_prepare_params&    pdsch_fh,
                slot_command_api::tx_precoding_beamforming_t& pc_bf) noexcept {
            detail::fill_pdsch_fh_header(pdsch_fh, args, pc_bf);
            detail::fill_pc_bf_header(pc_bf, args.pm_bf);
            detail::copy_pm_idx_and_beam_idx(pc_bf, args.pm_bf, args.mmimo_enabled());
        });
}

} // namespace scf_5g_fapi

#endif // SCF_5G_FAPI_PDSCH_FH_FILL_HPP_INCLUDED_
