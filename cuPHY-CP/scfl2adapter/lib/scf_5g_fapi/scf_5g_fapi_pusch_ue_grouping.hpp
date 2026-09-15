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

#if !defined(SCF_5G_FAPI_PUSCH_UE_GROUPING_HPP_INCLUDED_)
#define SCF_5G_FAPI_PUSCH_UE_GROUPING_HPP_INCLUDED_

#include <concepts>
#include <cstddef>
#include <cstdint>

#include "scf_5g_fapi.h"
#include "slot_command/slot_command.hpp"

namespace scf_5g_fapi
{

// ---------------------------------------------------------------------------
// PuschUeGroupingPolicy concept
// ---------------------------------------------------------------------------

/**
 * C++20 concept constraining a PUSCH UE-grouping policy class.
 *
 * A conforming policy must provide two static noexcept member functions:
 *
 *  - matches()        — returns true when an existing UE group is compatible
 *                       with the PUSCH PDU being processed.
 *
 *  - init_new_group() — populates the time-frequency allocation fields of a
 *                       newly created UE group from the current PDU.
 *
 * PUSCH only supports RA-Type 1 (contiguous PRB allocation), so unlike PDSCH
 * there is no bitmap-based matching path.
 */
template<typename P>
concept PuschUeGroupingPolicy = requires(
    const cuphyPuschUeGrpPrm_t&     grp,
    cuphyPuschUeGrpPrm_t&           mut_grp,
    const scf_fapi_pusch_pdu_t&     pdu,
    slot_command_api::pusch_params&  params,
    std::size_t                      grp_idx)
{
    { P::matches(grp, pdu)                             } noexcept -> std::same_as<bool>;
    { P::init_new_group(mut_grp, params, grp_idx, pdu) } noexcept -> std::same_as<void>;
};

// ---------------------------------------------------------------------------
// DefaultPuschUeGroupingPolicy
// ---------------------------------------------------------------------------

/**
 * Default PUSCH UE-grouping policy replicating legacy update_cell_command behaviour.
 *
 * Groups UEs that share the same time-frequency resource allocation:
 *   (puschStartSym, nPuschSym, startPrb, nPrb)
 *
 * UEs in the same MU-MIMO group share identical OFDM symbol and PRB allocation
 * (TS 38.214 §6.1.2), enabling the cuPHY kernel to process them together.
 *
 * ## Matching criteria (RA-Type 1 only — contiguous PRBs)
 *
 *   puschStartSym == pdu.start_symbol_index
 *   nPuschSym     == pdu.num_of_symbols
 *   startPrb      == bwp_start + rb_start
 *   nPrb          == rb_size
 *
 * ## New-group init
 *
 *   Sets puschStartSym, nPuschSym, startPrb, nPrb, dmrsSymLocBmsk, rssiSymLocBmsk
 *   from the PDU.
 */
struct DefaultPuschUeGroupingPolicy final
{
    /**
     * Check whether an existing group is compatible with the current PUSCH PDU.
     *
     * @param[in] grp  Candidate UE group from the slot-command arrays.
     * @param[in] pdu  PUSCH PDU being processed.
     * @return true if the group matches; false otherwise.
     */
    [[nodiscard]] static bool matches(
        const cuphyPuschUeGrpPrm_t& grp,
        const scf_fapi_pusch_pdu_t& pdu) noexcept
    {
        return grp.puschStartSym == pdu.start_symbol_index
            && grp.nPuschSym     == pdu.num_of_symbols
            && grp.startPrb      == static_cast<uint16_t>(pdu.bwp.bwp_start + pdu.rb_start)
            && grp.nPrb          == pdu.rb_size;
    }

    /**
     * Initialise the time-frequency allocation fields of a newly created UE group.
     *
     * @param[in,out] grp      UE group to initialise.
     * @param[in,out] params   Active pusch_params (unused in default policy but
     *                         available for custom policies that need it).
     * @param[in]     grp_idx  Index of the group within the slot-command arrays
     *                         (unused in default policy).
     * @param[in]     pdu      PUSCH PDU providing the allocation parameters.
     */
    static void init_new_group(
        cuphyPuschUeGrpPrm_t&                        grp,
        [[maybe_unused]] slot_command_api::pusch_params& params,
        [[maybe_unused]] std::size_t                  grp_idx,
        const scf_fapi_pusch_pdu_t&                   pdu) noexcept
    {
        grp.puschStartSym  = pdu.start_symbol_index;
        grp.nPuschSym      = pdu.num_of_symbols;
        grp.startPrb       = static_cast<uint16_t>(pdu.bwp.bwp_start + pdu.rb_start);
        grp.nPrb           = pdu.rb_size;
        grp.dmrsSymLocBmsk = pdu.ul_dmrs_sym_pos;
        grp.rssiSymLocBmsk = pdu.ul_dmrs_sym_pos;
    }
};

static_assert(PuschUeGroupingPolicy<DefaultPuschUeGroupingPolicy>);

} // namespace scf_5g_fapi

#endif // SCF_5G_FAPI_PUSCH_UE_GROUPING_HPP_INCLUDED_
