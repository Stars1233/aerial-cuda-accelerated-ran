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

#if !defined(SCF_5G_FAPI_PDSCH_UE_GROUPING_HPP_INCLUDED_)
#define SCF_5G_FAPI_PDSCH_UE_GROUPING_HPP_INCLUDED_

#include <bit>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>

#include "scf_5g_fapi.h"
#include "scf_5g_fapi_tags.hpp"
#include "slot_command/slot_command.hpp"

namespace scf_5g_fapi
{

// ---------------------------------------------------------------------------
// UeGroupingPolicy concept
// ---------------------------------------------------------------------------

/**
 * C++20 concept constraining a UE-grouping policy class.
 *
 * A conforming policy must provide two static noexcept member functions:
 *
 *  - matches()        — returns true when an existing UE group is compatible
 *                       with the PDU being processed.
 *
 *  - init_new_group() — populates the PRB-allocation fields of a newly
 *                       created UE group from the current PDU.
 */
template<typename P>
concept UeGroupingPolicy = requires(
    const cuphyPdschUeGrpPrm_t&    grp,
    cuphyPdschUeGrpPrm_t&           mut_grp,  // init_new_group writes nPrb/startPrb/rbBitmap
    const scf_fapi_pdsch_pdu_end_t& end,
    const scf_fapi_pdsch_pdu_t&     pdu,
    slot_command_api::pdsch_params& params,
    std::size_t                     grp_idx)
{
    { P::matches(grp, end, pdu)                              } noexcept -> std::same_as<bool>;
    { P::init_new_group(mut_grp, params, grp_idx, end, pdu)  } noexcept -> std::same_as<void>;
};

// ---------------------------------------------------------------------------
// DefaultUeGroupingPolicy
// ---------------------------------------------------------------------------

/**
 * Default UE-grouping policy replicating update_cell_command() behaviour.
 *
 * Dispatches between RA-Type 0 (bitmap) and RA-Type 1 (contiguous) at runtime
 * via scf_fapi_pdsch_pdu_end_t::resource_alloc.
 *
 * ## Matching criteria
 *
 * **RA-Type 1** (resource_alloc == 1):
 *   pdschStartSym, nPdschSym, startPrb (bwp_start + rb_start), nPrb (rb_size).
 *
 * **RA-Type 0** (resource_alloc == 0):
 *   pdschStartSym, nPdschSym, startPrb (bwp_start), full rbBitmap.
 *
 * ## New-group init
 *
 * **RA-Type 0**: copies rbBitmap, calls compute_ra_type0_prb_allocation() to
 *   derive nPrb and fill ra_type0_info[] using C++20 std::countr_zero; sets
 *   startPrb = bwp_start.
 *
 * **RA-Type 1**: nPrb = rb_size, startPrb = bwp_start + rb_start.
 */
struct DefaultUeGroupingPolicy
{
    /**
     * Check whether @p grp is compatible with the current PDU.
     *
     * @param[in] grp  Candidate UE group from the slot-command arrays.
     * @param[in] end  Fixed PDU tail (resource_alloc, symbol/PRB fields).
     * @param[in] pdu  Full PDSCH PDU (provides bwp_start).
     * @return true if the group matches; false otherwise.
     *         Return value must be checked.
     */
    [[nodiscard]] static bool matches(
        const cuphyPdschUeGrpPrm_t&    grp,
        const scf_fapi_pdsch_pdu_end_t& end,
        const scf_fapi_pdsch_pdu_t&     pdu) noexcept
    {
        if (end.resource_alloc == 1u) { // RA-Type 1 — contiguous PRBs
            return grp.pdschStartSym == end.start_sym_index
                && grp.nPdschSym    == end.num_symbols
                && grp.startPrb     == static_cast<uint16_t>(pdu.bwp.bwp_start + end.rb_start)
                && grp.nPrb         == end.rb_size;
        }
        // RA-Type 0 — bitmap-described allocation
        return grp.pdschStartSym == end.start_sym_index
            && grp.nPdschSym    == end.num_symbols
            && grp.startPrb     == pdu.bwp.bwp_start
            && std::memcmp(grp.rbBitmap, end.rb_bitmap,
                           sizeof(uint8_t) * MAX_RBMASK_BYTE_SIZE) == 0;
    }

    /**
     * Initialise the PRB-allocation fields of a newly created UE group.
     *
     * @param[in,out] grp      UE group to initialise.
     * @param[in,out] params   Active pdsch_params (ra_type0_info written for RA-Type 0).
     * @param[in]     grp_idx  Index of @p grp within the slot-command arrays.
     * @param[in]     end      Fixed PDU tail (resource_alloc, bitmap/PRB fields).
     * @param[in]     pdu      Full PDSCH PDU (provides bwp_start).
     */
    static void init_new_group(
        cuphyPdschUeGrpPrm_t&          grp,
        slot_command_api::pdsch_params& params,
        std::size_t                     grp_idx,
        const scf_fapi_pdsch_pdu_end_t& end,
        const scf_fapi_pdsch_pdu_t&     pdu) noexcept
    {
        grp.resourceAlloc = end.resource_alloc;

        if (grp.resourceAlloc == 0u) { // RA-Type 0
            std::memcpy(grp.rbBitmap, end.rb_bitmap,
                        sizeof(uint8_t) * MAX_RBMASK_BYTE_SIZE);
            compute_ra_type0_prb_allocation(params, grp, grp_idx);
            grp.startPrb = pdu.bwp.bwp_start;
        } else {                        // RA-Type 1
            grp.nPrb     = end.rb_size;
            grp.startPrb = static_cast<uint16_t>(pdu.bwp.bwp_start + end.rb_start);
        }
    }

private:
    /**
     * Compute nPrb and fill ra_type0_info[] from the rbBitmap using C++20
     * std::countr_zero — an alternate to the file-local prepare_ra_type0_info()
     * in scf_5g_slot_commands_pdsch_csirs.cpp.
     *
     * Walks grp.rbBitmap as a std::span<const uint8_t>, detects consecutive runs
     * of set bits (PRBs) using bit-manipulation with std::countr_zero, and records
     * each run as a slot_command_api::ra_type0_info_t_ entry.  Runs spanning byte
     * boundaries are merged naturally without special-casing.
     *
     * @param[in,out] params   Active pdsch_params; ra_type0_info and
     *                         num_ra_type0_info are written here.
     * @param[in,out] grp      UE group whose rbBitmap is read; nPrb is set.
     * @param[in]     grp_idx  Column index into ra_type0_info[][grp_idx].
     */
    static void compute_ra_type0_prb_allocation(
        slot_command_api::pdsch_params& params,
        cuphyPdschUeGrpPrm_t&           grp,
        std::size_t                      grp_idx) noexcept
    {
        // Upper bound on the number of non-contiguous PRB runs that can be
        // stored; matches the first dimension of pdsch_params::ra_type0_info.
        static constexpr uint32_t k_max_runs = MAX_RBMASK_BYTE_SIZE / 2u + 1u;

        grp.nPrb = 0u;

        uint32_t info_idx  = 0u;
        int32_t  run_start = -1;
        uint32_t run_len   = 0u;

        const auto bitmap = std::span<const uint8_t, MAX_RBMASK_BYTE_SIZE>{grp.rbBitmap, MAX_RBMASK_BYTE_SIZE};

        for (std::size_t byte_idx = 0; byte_idx < bitmap.size(); ++byte_idx) {
            uint8_t remaining = bitmap[byte_idx];

            if (remaining == 0u) {
                // Whole byte is zero; flush any open run across this byte boundary.
                if (run_start != -1) {
                    if (info_idx >= k_max_runs) [[unlikely]] {
                        NVLOGW_FMT(detail::k_tag,
                                   "compute_ra_type0_prb_allocation: grp_idx={} ra_type0_info full "
                                   "at info_idx={} (byte {}), truncating",
                                   grp_idx, info_idx, byte_idx);
                        params.num_ra_type0_info[grp_idx] = info_idx;
                        return;
                    }
                    params.ra_type0_info[info_idx][grp_idx] = {
                        .start_prb = static_cast<uint32_t>(run_start),
                        .num_prb   = run_len,
                    };
                    grp.nPrb += static_cast<uint16_t>(run_len);
                    ++info_idx;
                    run_start = -1;
                    run_len   = 0u;
                }
                continue;
            }

            // Walk every set bit in this byte using std::countr_zero (C++20 <bit>).
            // clear-lowest-set-bit idiom: `remaining &= remaining - 1u`
            while (remaining != 0u) {
                const auto bit_pos = static_cast<uint32_t>(std::countr_zero(remaining));
                const auto prb     = static_cast<int32_t>(byte_idx * 8u + bit_pos);

                if (run_start == -1) {
                    run_start = prb;
                    run_len   = 1u;
                } else if (prb == run_start + static_cast<int32_t>(run_len)) {
                    ++run_len;  // contiguous — extend current run
                } else {
                    // Gap (cross-byte or intra-byte) — flush current run, start a new one.
                    if (info_idx >= k_max_runs) [[unlikely]] {
                        NVLOGW_FMT(detail::k_tag,
                                   "compute_ra_type0_prb_allocation: grp_idx={} ra_type0_info full "
                                   "at info_idx={} (byte {} bit {}), truncating",
                                   grp_idx, info_idx, byte_idx, bit_pos);
                        params.num_ra_type0_info[grp_idx] = info_idx;
                        return;
                    }
                    params.ra_type0_info[info_idx][grp_idx] = {
                        .start_prb = static_cast<uint32_t>(run_start),
                        .num_prb   = run_len,
                    };
                    grp.nPrb += static_cast<uint16_t>(run_len);
                    ++info_idx;
                    run_start = prb;
                    run_len   = 1u;
                }
                remaining &= remaining - 1u; // clear lowest set bit
            }
        }

        // Flush any run that extended to the last byte of the bitmap.
        if (run_start != -1) {
            if (info_idx >= k_max_runs) {
                NVLOGW_FMT(detail::k_tag,
                           "compute_ra_type0_prb_allocation: grp_idx={} ra_type0_info full "
                           "at final flush info_idx={}, truncating",
                           grp_idx, info_idx);
                params.num_ra_type0_info[grp_idx] = info_idx;
                return;
            }
            params.ra_type0_info[info_idx][grp_idx] = {
                .start_prb = static_cast<uint32_t>(run_start),
                .num_prb   = run_len,
            };
            grp.nPrb += static_cast<uint16_t>(run_len);
            ++info_idx;
        }

        params.num_ra_type0_info[grp_idx] = info_idx;
    }
};

static_assert(UeGroupingPolicy<DefaultUeGroupingPolicy>);

} // namespace scf_5g_fapi

#endif // SCF_5G_FAPI_PDSCH_UE_GROUPING_HPP_INCLUDED_
