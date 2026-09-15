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

#if !defined(SCF_5G_FAPI_PRACH_SLOT_BUILDER_HPP_INCLUDED_)
#define SCF_5G_FAPI_PRACH_SLOT_BUILDER_HPP_INCLUDED_

/**
 * @file scf_5g_fapi_prach_slot_builder.hpp
 * @brief PRACH functional core (FCIS) — pure compute path, no slot-command writes.
 *
 * PRACH functional core (FCIS) — pure compute path, no slot-command writes.
 *
 * `compute_prach_params` derives every per-PDU value the imperative shells
 * need to populate `prach_params` and `sym_prb_info`, with
 * zero dependency on `slot_command_api` types or driver singletons. Inputs are
 * the raw FAPI PDU and two cached config structs read from the module view;
 * outputs are returned by value through `tl::expected<PrachComputed, PrachError>`.
 *
 * FCIS: this header / `.cpp` pair is the **pure core** —
 * branchy validation, table lookups, modular start-symbol arithmetic, but no
 * side effects (no slot-command writes, no logging, no helper calls). The
 * imperative shells (`apply_prach_to_slot_command`,
 * `apply_prach_fh_to_sym_prb_info`) live in subsequent additions.
 *
 * Layout discipline (§1.10): `PrachComputed` is `alignas(64)`, size-descending
 * member order, trivially copyable. Static asserts guarantee size <= 128 B and
 * alignof == 64.
 */

#include <algorithm>
#include <array>
#include <cstdint>
#include <type_traits>

#include <tl/expected.hpp>
#include <wise_enum/wise_enum.h>

#include "scf_5g_fapi.h"                  // scf_fapi_prach_pdu_t
#include "scf_5g_fapi_prach_dop.hpp"      // dop::set_uplink / add_occasion / set_global_config
#include "nv_phy_fapi_msg_common.hpp"     // nv::phy_config, nv::prach_addln_config_t, nv::slot_detail_t
#include "slot_command/slot_command.hpp"  // slot_command_api types used by the imperative shells
#include "aerial-fh-driver/oran.hpp"      // ru_type enum

namespace scf_5g_fapi
{
void update_beam_list(slot_command_api::beamid_array_t&        array,
                      std::size_t&                             array_size,
                      const scf_fapi_rx_beamforming_t&         pmi_bf_pdu,
                      bool                                     mmimo_enabled,
                      slot_command_api::prb_info_t&            prb_info,
                      int32_t                                  cell_idx);
} // namespace scf_5g_fapi

namespace scf_5g_fapi::prach
{

// ---------------------------------------------------------------------------
// File-scope constants — bounds and conversion tables (§1.1)
// ---------------------------------------------------------------------------

/// Max physical cell ID (3GPP TS 38.211 §7.4.2.1).
inline constexpr uint16_t k_max_phys_cell_id = 1007u;

/// Max PRACH time-domain occasions per PDU (FAPI 10.04; validated [1:7]
/// by the L2A FAPI validator). NOT MAX_PRACH_MAX_OCCASIONS_PER_CELL,
/// which is the per-cell slot-command storage capacity (a different axis).
inline constexpr uint8_t  k_max_num_prach_ocas = 7u;

/// Symbols per OFDM slot (numerology-independent).
inline constexpr uint8_t  k_ofdm_symbols_per_slot = 14u;

/// Largest PRACH format value covered by @ref k_prach_format_to_filter_index.
inline constexpr uint8_t  k_max_prach_format = 13u;

/// Max @c num_ra index — bounded by the configured root-sequence table.
inline constexpr uint8_t  k_max_num_ra =
    static_cast<uint8_t>(nv::NV_MAX_PRACH_FD_OCCASION_NUM - 1);

/// PRACH format → filter index LUT (replaces the switch in
/// scf_5g_slot_commands.cpp::update_fh_params_prach).
inline constexpr std::array<uint8_t, k_max_prach_format + 1u>
    k_prach_format_to_filter_index{
        1, 1, 1,                // formats 0..2 (long, 839 sequence)
        2,                      // format  3    (long, 139 sequence)
        3, 3, 3, 3, 3, 3, 3, 3, 3, 3  // formats 4..13 (short)
    };

/**
 * Look up the FH filter index for a PRACH format via @ref
 * k_prach_format_to_filter_index, with bounds.
 *
 * @param[in]  format  PRACH format (3GPP TS 38.211 §6.3.3, [0:13]).
 * @return  Filter index from the LUT, or 0 if @c format is out of range.
 *          Return value must be checked.
 */
[[nodiscard]] constexpr uint8_t prach_filter_index(const uint8_t format) noexcept
{
    return (format <= k_max_prach_format)
        ? k_prach_format_to_filter_index[format]
        : uint8_t{0};
}

// ---------------------------------------------------------------------------
// PrachError — typed failure modes from compute_prach_params (§1.7)
// ---------------------------------------------------------------------------

/**
 * Failure modes for @ref compute_prach_params.
 *
 * Pure-core errors are *returned*, never logged — the imperative shell is the
 * right layer to decide whether a failure needs a NVLOGE or is silently
 * dropped. Each enumerator names the specific field that failed validation so
 * callers can produce specific diagnostics.
 *
 * Uses WISE_ENUM_CLASS so `wise_enum::to_string(e)` yields the enumerator name
 * without a hand-maintained switch — the shell forwards it to NVLOGE_FMT.
 */
WISE_ENUM_CLASS((PrachError, uint8_t),
    OutOfRangePhysCellId,        // pdu.phys_cell_id        >  k_max_phys_cell_id
    OutOfRangeNumPrachOcas,      // pdu.num_prach_ocas      >  k_max_num_prach_ocas
    OutOfRangePrachFormat,       // pdu.prach_format        >  k_max_prach_format
    OutOfRangeNumRa,             // pdu.num_ra              >  k_max_num_ra
    OutOfRangePrachStartSymbol,  // pdu.prach_start_symbol  >= k_ofdm_symbols_per_slot
    UnsupportedPrachFormat,      // LUT entry is 0 (reserved / unsupported)
    ZeroNumPrachOcas             // pdu.num_prach_ocas == 0 — caller should fast-skip
);

// ---------------------------------------------------------------------------
// PrachComputed — derived per-PDU values, consumed by the imperative shell
// ---------------------------------------------------------------------------

/**
 * Result of @ref compute_prach_params.
 *
 * SoA per-occasion arrays + scalars. Members are declared in size-descending
 * order; struct is `alignas(64)` so the whole result fits in one cache line on
 * x86-64 / ARM Neoverse V2 / ARM A78 (all 64 B lines). Trivially copyable.
 *
 * Field semantics — sourced from two distinct legacy code paths:
 *
 *   rach_params path (legacy update_cell_command(PRACH),
 *   scf_5g_slot_commands.cpp:550-581) — consumed by the always-on shell
 *   (apply_prach_to_slot_command):
 *   - @c num_occasions   : copied from @c pdu.num_prach_ocas (validated)
 *   - @c occa_prm_stat_idx: `phy_cfg.prach_config_.start_ro_index + pdu.num_ra`
 *   - @c mu              : `phy_cfg.prach_config_.prach_scs`
 *   - @c nfft            : `prach_seq_length == 1 ? PRACH_SHORT_FORMAT_FFT : PRACH_LONG_FORMAT_FFT`
 *   - @c n_uplink_streams: `mmimo && bf.dig_bf_interfaces != 0 ? dig_bf_interfaces : 0`
 *   - @c phy_cell_id     : copied from `phy_cfg.cell_config_.phy_cell_id`
 *                          (NOT `pdu.phys_cell_id` — matches legacy
 *                          scf_5g_slot_commands.cpp:561 which uses the cached
 *                          cell config, not the PDU field)
 *   - @c freq_index      : copied from `pdu.num_ra` (FAPI 10.04 alias)
 *
 *   FH-shell path (legacy update_fh_params_prach,
 *   scf_5g_slot_commands.cpp:~1869) — consumed by apply_prach_fh_to_sym_prb_info:
 *   - @c start_symbols[i]: `(l0 + n_ra_dur*i + 14*n_ra_slot) %% 14`
 *   - @c filter_indices[i]: LUT lookup keyed by @c pdu.prach_format
 *   - @c freq_offset     : `phy_cfg.prach_config_.root_sequence[num_ra].freqOffset`
 *   - @c n_ra_dur, @c n_ra_rb: copied from `addln_config`
 */
struct alignas(64) PrachComputed
{
    std::array<uint8_t, k_max_num_prach_ocas> start_symbols{};    //!< Per-occasion OFDM start symbol.
    std::array<uint8_t, k_max_num_prach_ocas> filter_indices{};   //!< Per-occasion FH filter index (LUT).
    int32_t  freq_offset       {};                                //!< prach_config_.root_sequence[num_ra].freqOffset.
    uint16_t phy_cell_id       {};                                //!< phy_cfg.cell_config_.phy_cell_id (NOT pdu.phys_cell_id).
    uint16_t occa_prm_stat_idx {};                                //!< start_ro_index + num_ra.
    uint16_t nfft              {};                                //!< Short- or long-format FFT size.
    uint8_t  num_occasions     {};                                //!< pdu.num_prach_ocas.
    uint8_t  freq_index        {};                                //!< pdu.num_ra.
    uint8_t  n_ra_dur          {};                                //!< addln.n_ra_dur (symbols per occasion).
    uint8_t  n_ra_rb           {};                                //!< addln.n_ra_rb (PRBs per occasion).
    uint8_t  mu                {};                                //!< prach_config_.prach_scs.
    uint8_t  n_uplink_streams  {};                                //!< Only set when mmimo enabled and dig_bf != 0.
};

static_assert(sizeof(PrachComputed) <= 128u,
              "PrachComputed must fit in two cache lines");
static_assert(alignof(PrachComputed) == 64u,
              "PrachComputed must be cache-line aligned");
static_assert(std::is_trivially_copyable_v<PrachComputed>,
              "PrachComputed must be trivially copyable for hot-path return-by-value");

// ---------------------------------------------------------------------------
// compute_prach_params — pure functional core (§1.5)
// ---------------------------------------------------------------------------

namespace detail
{

/**
 * Per-occasion OFDM start symbol (3GPP TS 38.211 §5.3.2).
 *
 * The `14 * n_ra_slot` term carries forward unchanged from the legacy
 * @c update_fh_params_prach derivation in scf_5g_slot_commands.cpp.
 *
 * @param[in] l0          Base start symbol from the FAPI PDU (pdu.prach_start_symbol).
 * @param[in] i           Zero-based occasion index (0 .. num_prach_ocas-1).
 * @param[in] n_ra_dur    Per-occasion symbol duration (addln.n_ra_dur).
 * @param[in] n_ra_slot   Slot-index offset (addln.n_ra_slot).
 * @return  OFDM symbol index in [0, k_ofdm_symbols_per_slot).
 *          Return value must be checked.
 */
[[nodiscard]] constexpr uint8_t start_symbol_for(uint8_t l0,
                                                 uint8_t i,
                                                 uint8_t n_ra_dur,
                                                 uint8_t n_ra_slot) noexcept
{
    const uint32_t raw =
        static_cast<uint32_t>(l0)
        + static_cast<uint32_t>(n_ra_dur) * static_cast<uint32_t>(i)
        + static_cast<uint32_t>(k_ofdm_symbols_per_slot)
        * static_cast<uint32_t>(n_ra_slot);
    return static_cast<uint8_t>(raw % k_ofdm_symbols_per_slot);
}

} // namespace detail

/**
 * Derive all per-PDU PRACH state from a FAPI PRACH PDU + cached cell configs.
 *
 * Pure: no slot-command writes, no logging, no driver singletons, no
 * allocations. Validates field ranges against the constants above; on success
 * returns a fully populated @ref PrachComputed by value.
 *
 * @param[in] pdu            FAPI PRACH PDU descriptor.
 * @param[in] phy_cfg        Per-cell PHY config (provides prach_config_).
 * @param[in] addln          Per-cell PRACH additional config (n_ra_dur etc.).
 * @param[in] mmimo_enabled  Sourced from the module view (mMIMO feature gate).
 * @return                   Populated @ref PrachComputed on success;
 *                           @ref PrachError on validation failure.
 *                           Return value must be checked.
 */
[[nodiscard]] constexpr tl::expected<PrachComputed, PrachError>
compute_prach_params(const scf_fapi_prach_pdu_t&     pdu,
                     const nv::phy_config&           phy_cfg,
                     const nv::prach_addln_config_t& addln,
                     bool                            mmimo_enabled) noexcept
{
    // ---- Validation guards (defensive — every error path [[unlikely]]) ----
    if (pdu.num_prach_ocas == 0u) [[unlikely]] {
        return tl::unexpected{PrachError::ZeroNumPrachOcas};
    }
    if (pdu.num_prach_ocas > k_max_num_prach_ocas) [[unlikely]] {
        return tl::unexpected{PrachError::OutOfRangeNumPrachOcas};
    }
    if (pdu.phys_cell_id > k_max_phys_cell_id) [[unlikely]] {
        return tl::unexpected{PrachError::OutOfRangePhysCellId};
    }
    if (pdu.prach_format > k_max_prach_format) [[unlikely]] {
        return tl::unexpected{PrachError::OutOfRangePrachFormat};
    }
    if (pdu.num_ra > k_max_num_ra) [[unlikely]] {
        return tl::unexpected{PrachError::OutOfRangeNumRa};
    }
    if (pdu.prach_start_symbol >= k_ofdm_symbols_per_slot) [[unlikely]] {
        return tl::unexpected{PrachError::OutOfRangePrachStartSymbol};
    }

    // LUT lookup. Not currently reachable: OutOfRangePrachFormat above
    // already rejects prach_format > 13, and every entry in
    // k_prach_format_to_filter_index for indices 0..13 is non-zero. This
    // guard exists as a defensive backstop — if the LUT is ever reshaped
    // (a future format mapped to 0, or k_max_prach_format bumped) the
    // check prevents silently emitting filterIndex=0 to fronthaul.
    const uint8_t filter_idx = prach_filter_index(pdu.prach_format);
    if (filter_idx == 0u) [[unlikely]] {
        return tl::unexpected{PrachError::UnsupportedPrachFormat};
    }

    // ---- Population (hot path; no further branching beyond MMIMO gate) ----
    //
    // Designated initializer order MUST match the declaration order of
    // PrachComputed (C++20 requirement). Adding a member without updating
    // this initializer is a compile-time error, which is exactly the
    // discipline we want for a result struct that grows over time.
    PrachComputed out{
        .start_symbols     = {},  // filled per-occasion below
        .filter_indices    = {},  // filled per-occasion below
        .freq_offset       = phy_cfg.prach_config_.root_sequence[pdu.num_ra].freqOffset,
        .phy_cell_id       = static_cast<uint16_t>(phy_cfg.cell_config_.phy_cell_id),
        .occa_prm_stat_idx = static_cast<uint16_t>(phy_cfg.prach_config_.start_ro_index
                                                   + pdu.num_ra),
        .nfft              = (phy_cfg.prach_config_.prach_seq_length == 1u)
                                 ? static_cast<uint16_t>(nv::PRACH_SHORT_FORMAT_FFT)
                                 : static_cast<uint16_t>(nv::PRACH_LONG_FORMAT_FFT),
        .num_occasions     = pdu.num_prach_ocas,
        .freq_index        = pdu.num_ra,
        .n_ra_dur          = addln.n_ra_dur,
        .n_ra_rb           = addln.n_ra_rb,
        .mu                = phy_cfg.prach_config_.prach_scs,
        .n_uplink_streams  = (mmimo_enabled && pdu.beam_index.dig_bf_interfaces != 0u)
                                 ? pdu.beam_index.dig_bf_interfaces
                                 : uint8_t{0},
    };

    for (uint8_t i = 0; i < pdu.num_prach_ocas; ++i) {
        out.start_symbols[i]  = detail::start_symbol_for(pdu.prach_start_symbol,
                                                         i,
                                                         addln.n_ra_dur,
                                                         addln.n_ra_slot);
        out.filter_indices[i] = filter_idx;
    }
    return out;
}

// ---------------------------------------------------------------------------
// Imperative shell — always-on writes (prach_params + uplink slot state).
//
// FH/order metadata writes (sym_prb_info per-occasion fill) live further below.
// ---------------------------------------------------------------------------

/**
 * Aggregate of references + scalars passed to @ref populate_slot_command.
 *
 * Layout: 6 × 8 B pointers/refs + 2 × 4 B (int32 + ru_type) + 3 × 1 B bools +
 * 5 B tail padding = exactly 64 B (one cache line on every cuBB target —
 * x86-64, ARM Neoverse V2, ARM A78). `alignas(64)` plus static_assert
 * guarantees the whole struct fits in a single line and starts on a line
 * boundary, eliminating cache-line-straddle stalls when the parser builds
 * the context on its stack.
 *
 * `slot_indication` is held by const-ref (not by value, 24 B) so the struct
 * doesn't spill into a second line. The caller's slot_indication must
 * outlive the BuildContext — trivially satisfied at the parser call site
 * where slot_ind is a local variable declared before the BuildContext.
 *
 * Reference members are non-rebindable, which makes BuildContext
 * non-assignable. That's intentional: it's a per-call value built at the
 * point of use and discarded after `populate_slot_command` returns.
 */
struct alignas(64) BuildContext final
{
    slot_command_api::cell_group_command&    group;                  //!< Cell-group command (uplink slot + occasion appended).
    slot_command_api::cell_sub_command&      cell;                   //!< Per-cell sub-command (uplink slot + sym_prb_info).
    const slot_command_api::slot_indication& slot_ind;               //!< Held by ref to keep BuildContext one cache line.
    const nv::phy_config&                    phy_config;             //!< Per-cell PHY config (prach_config_, cell_config_).
    const nv::prach_addln_config_t&          addln_config;           //!< Per-cell PRACH additional config (n_ra_*).
    nv::slot_detail_t*                       slot_detail;            //!< TDD slot detail (null in direct mode). TODO: consumed by PrachPduParser::parse for TDD symbol masking once the parser body lands.
    int32_t                                  cell_index;             //!< Logical cell index (carrier id).
    ru_type                                  ru;                     //!< Resource-unit class (SINGLE_SECT_MODE, OTHER_MODE).
    bool                                     bf_enabled;             //!< RU-side beamforming enabled.
    bool                                     mmimo_enabled;          //!< mMIMO feature gate.
    bool                                     fapi_to_cplane_direct;  //!< Direct C-plane active; sym_prb_info still feeds UL Order.
};

static_assert(sizeof(BuildContext) <= 64u,
              "BuildContext must fit in one cache line");
static_assert(alignof(BuildContext) == 64u,
              "BuildContext must be cache-line aligned");

/**
 * Apply a validated @ref PrachComputed to the slot-command state.
 *
 * Imperative shell — writes only `prach_params` (rach[occ], freqIndex,
 * startSymbols, mu, nfft, nOccasion) and the cell/group uplink slot tags.
 * `sym_prb_info` (FH/order metadata) is left untouched here; that's the FH
 * shell's job and is appended after the always-on PRACH params.
 *
 * All field-level invariants are owned by the DOP helpers
 * (`dop::set_uplink`, `dop::add_occasion`, `dop::set_global_config`) so this
 * function does no raw field writes — cubb-review §12.1 (Tell-Don't-Ask) is
 * satisfied by delegating to those helpers.
 *
 * @param[in]     computed    Output of @ref compute_prach_params.
 * @param[in,out] group       Cell-group command to tag uplink + append occasion.
 * @param[in,out] cell        Per-cell sub-command to tag uplink.
 * @param[in]     slot_ind    3GPP slot indication (SFN/slot/hopping).
 * @param[in]     cell_index  Logical cell index (carrier id).
 * @return  true on success; false if `prach_params` is null or already at
 *          capacity (MAX_PRACH_OCCASIONS_PER_SLOT).
 *          Return value must be checked.
 */
[[nodiscard]] inline bool
apply_prach_to_slot_command(const PrachComputed&                         computed,
                            slot_command_api::cell_group_command&        group,
                            slot_command_api::cell_sub_command&          cell,
                            const slot_command_api::slot_indication&     slot_ind,
                            const int32_t                                cell_index) noexcept
{
    // Resolve prach_params and check capacity BEFORE any slot-state writes so
    // the failure path mutates nothing — preserves the "false = no mutation"
    // contract that callers (e.g. populate_slot_command) rely on when sequencing
    // dependent steps. Hoisting the full() check ahead of set_uplink avoids
    // leaving cell.slot / group.slot tagged SLOT_UPLINK on capacity refusal.
    //
    // Note: the production cell_group_command::get_prach_params() lazy-allocates
    // via create_if() and never returns nullptr (slot_command.hpp:1463-1467), so
    // this guard is currently defensive — retained as a forward-looking contract
    // check for callers that may later pass a mock or alternative group type
    // whose accessor can legitimately return null.
    auto* const params = group.get_prach_params();
    if (params == nullptr) [[unlikely]] {
        return false;
    }
    if (dop::full(*params)) [[unlikely]] {
        return false;
    }

    dop::set_uplink(cell.slot,  slot_ind);
    dop::set_uplink(group.slot, slot_ind);

    // dop::add_occasion re-checks full() defensively; with the hoisted guard
    // above it cannot return false here, but keep the [[unlikely]] branch as a
    // belt-and-braces safety net in case the DOP contract evolves.
    const bool added = dop::add_occasion(*params, dop::OccasionEntry{
        .cell_index        = cell_index,
        .force_thr0        = 0.0f,
        .phy_cell_id       = computed.phy_cell_id,
        .occa_prm_stat_idx = computed.occa_prm_stat_idx,
        .n_uplink_streams  = computed.n_uplink_streams,
        .freq_index        = computed.freq_index,
        .start_symbol      = computed.start_symbols[0],
    });
    if (!added) [[unlikely]] {
        return false;
    }

    dop::set_global_config(*params, computed.mu, computed.nfft);
    return true;
}

// ---------------------------------------------------------------------------
// MR 3c — per-occasion FH/order metadata fill
// ---------------------------------------------------------------------------

inline void update_prach_prb_sym_list(slot_command_api::slot_info_t& sym_prbs,
                                      std::size_t                    prb_index,
                                      uint8_t                        start_sym,
                                      uint8_t                        num_sym,
                                      ru_type                        ru)
{
    const auto iter_start = sym_prbs.symbols.begin() + start_sym;
    const auto iter_end = iter_start + num_sym;
    std::for_each(iter_start, iter_end, [prb_index](slot_command_api::channel_info_list_t& channel_info_list) {
        channel_info_list[slot_command_api::channel_type::PRACH].push_back(prb_index);
    });

    if (ru == SINGLE_SECT_MODE)
    {
        sym_prbs.start_symbol_ul = 0;
    }
}

/**
 * Append per-occasion FH metadata to a slot_info_t.
 *
 * In non-direct mode the framework consumes `sym_prb_info` to build O-RAN
 * C-plane locally. In direct mode C-plane packets are generated from FAPI on
 * a worker thread, but cuphydriver's UL Order kernel still counts PRACH PRBs
 * from this same `sym_prb_info`, so the metadata remains required.
 *
 * Per-occasion writes are routed through DOP helpers (`dop::full`,
 * `dop::make_prach_prb`, `dop::add_prb`) so this function does no raw
 * `prbs[i] = ...; ++prbs_size` discipline — §12.1 satisfied.
 * Beam-list append uses the legacy `update_beam_list`; PRB-symbol registration
 * mirrors the PRACH branch of the legacy helper locally to keep this parser
 * header free of the full cuphydriver include chain.
 *
 * @param[in]     computed     Output of @ref compute_prach_params.
 * @param[in,out] sym_prbs     Destination Order metadata.
 * @param[in]     beam         Rx beamforming PDU (consumed by update_beam_list).
 * @param[in]     cell_index   Logical cell index (carrier id).
 * @param[in]     ru           Resource-unit class.
 * @param[in]     bf_enabled   When true, append beam list per occasion.
 * @param[in]     mmimo_enabled Sets portMask when @c beam.dig_bf_interfaces != 0.
 * @return  true on success; false if `sym_prb_info` overflows mid-loop.
 *          On overflow, the entries appended before the failing index remain;
 *          the caller is expected to treat the slot as poisoned.
 *          Return value must be checked.
 */
[[nodiscard]] inline bool
apply_prach_fh_to_sym_prb_info(const PrachComputed&                     computed,
                               slot_command_api::slot_info_t&           sym_prbs,
                               const scf_fapi_rx_beamforming_t&         beam,
                               const int32_t                            cell_index,
                               const ru_type                            ru,
                               const bool                               bf_enabled,
                               const bool                               mmimo_enabled) noexcept
{
    for (uint8_t i = 0; i < computed.num_occasions; ++i)
    {
        if (dop::full(sym_prbs)) [[unlikely]] {
            return false;
        }
        auto& prb = dop::add_prb(sym_prbs, dop::make_prach_prb(
            /*startPrb   =*/ uint16_t{0},
            /*numPrb     =*/ static_cast<uint16_t>(computed.n_ra_rb),
            /*freqOffset =*/ computed.freq_offset,
            /*numSymbols =*/ computed.n_ra_dur,
            /*filterIndex=*/ computed.filter_indices[i]));

        if (bf_enabled) {
            // Legacy update_fh_params_prach hard-codes mmimo=false here with a
            // TODO about static-beamforming Section-Type-3 support. Mirror that
            // until the TODO is resolved upstream.
            scf_5g_fapi::update_beam_list(prb.beams_array, prb.beams_array_size,
                                          beam, /*mmimo_enabled=*/false,
                                          prb, cell_index);
        }
        if (mmimo_enabled && beam.dig_bf_interfaces != 0u) {
            // Clamp the shift to portMask's bit width. `1u << N` is UB for
            // N >= 32, and even for N in [17, 31] the result would silently
            // truncate when narrowed to uint16_t. Cap at 16 → all-ones mask.
            prb.common.portMask = (beam.dig_bf_interfaces < 16u)
                ? static_cast<uint16_t>((1u << beam.dig_bf_interfaces) - 1u)
                : uint16_t{0xFFFFu};
        }

        update_prach_prb_sym_list(
            sym_prbs,
            /*prb_index=*/ static_cast<std::size_t>(sym_prbs.prbs_size - 1u),
            /*startSym =*/ computed.start_symbols[i],
            /*numSym   =*/ uint8_t{1},
            ru);
    }
    return true;
}

/**
 * Orchestrate compute → apply_rach → apply_fh/order metadata for one PRACH PDU.
 *
 * Always-on writes (`prach_params`, uplink slot tags) run unconditionally.
 * Per-occasion sym_prb_info metadata also runs unconditionally. Direct
 * C-plane owns packet generation, but the existing UL Order kernel derives
 * PRACH reorder work from sym_prb_info.
 *
 * @param[in,out] ctx  Build context (references must outlive the call).
 * @param[in]     pdu  FAPI PRACH PDU descriptor.
 * @return  true on success; false if validation fails, capacity is reached,
 *          or sym_prb_info overflows mid-loop.
 *          Return value must be checked.
 */
[[nodiscard]] inline bool
populate_slot_command(const BuildContext& ctx,
                      const scf_fapi_prach_pdu_t& pdu,
                      slot_command_api::slot_info_t* order_sym_prb_info = nullptr) noexcept
{
    const auto computed = compute_prach_params(pdu,
                                               ctx.phy_config,
                                               ctx.addln_config,
                                               ctx.mmimo_enabled);
    if (!computed.has_value()) [[unlikely]] {
        return false;
    }
    if (!apply_prach_to_slot_command(*computed, ctx.group, ctx.cell,
                                     ctx.slot_ind, ctx.cell_index)) [[unlikely]] {
        return false;
    }
    auto* const sym_prbs = order_sym_prb_info != nullptr
        ? order_sym_prb_info
        : ctx.cell.sym_prb_info();
    if (sym_prbs == nullptr) [[unlikely]] {
        return false;
    }
    return apply_prach_fh_to_sym_prb_info(*computed, *sym_prbs, pdu.beam_index,
                                          ctx.cell_index, ctx.ru,
                                          ctx.bf_enabled, ctx.mmimo_enabled);
}

} // namespace scf_5g_fapi::prach

#endif // SCF_5G_FAPI_PRACH_SLOT_BUILDER_HPP_INCLUDED_
