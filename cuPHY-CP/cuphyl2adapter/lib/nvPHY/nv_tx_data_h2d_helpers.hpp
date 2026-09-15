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

#if !defined(NV_TX_DATA_H2D_HELPERS_HPP_INCLUDED_)
#define NV_TX_DATA_H2D_HELPERS_HPP_INCLUDED_

#include <cstdint>
#include <cstddef>
#include "nv_ipc_utils.h"  // sfn_slot_t

namespace nv::tx_data_h2d
{

/// Matches @c PDSCH_MAX_GPU_BUFFS in @c cuphy.h (TB H2D ring depth).
inline constexpr unsigned kPdschTbGpuBufferRing = 20U;

/**
 * @brief Extract the within-frame slot index from a packed SFN/slot u32.
 *
 * Uses the @c sfn_slot_t union so the extraction is endianness-agnostic.
 */
[[nodiscard]] inline uint16_t slot_in_frame_from_slot_u32(const uint32_t slot_u32) noexcept
{
    sfn_slot_t ss;
    ss.u32 = slot_u32;
    return ss.u16.slot;
}

/**
 * @brief Circular PDSCH TB GPU buffer index used with @c Cell::get_pdsch_tb_buffer().
 *
 * Matches cuphydriver usage: @c slot % PDSCH_MAX_GPU_BUFFS.
 */
[[nodiscard]] inline uint8_t tb_gpu_buffer_index(const uint32_t slot_u32) noexcept
{
    const uint16_t sif = slot_in_frame_from_slot_u32(slot_u32);
    return static_cast<uint8_t>(static_cast<unsigned>(sif) % kPdschTbGpuBufferRing);
}

/**
 * @brief Whether a DL_TTI.request carries at least one PDSCH PDU (needs matching TX_DATA).
 *
 * @param n_pdsch_pdu_count Value from @c scf_fapi_dl_tti_req_t::nPDUsOfEachType[DL_TTI_PDU_TYPE_PDSCH].
 */
[[nodiscard]] constexpr bool dl_tti_expects_tx_data(const uint16_t n_pdsch_pdu_count) noexcept
{
    return n_pdsch_pdu_count > 0;
}

/**
 * @brief Result of @ref merge_ue_tb_ptr_ordinal — staged TB pointers merged into @c ue_tb_ptr[row].
 */
struct MergeUeTbPtrResult {
    std::uint32_t assigned      = 0; ///< Rows where a non-null staged pointer was written
    std::uint32_t overflow_count = 0; ///< PDSCH DL_TTI messages whose ordinal @c row >= max_rows
};

/**
 * @brief Merge split-phase staged GPU TB pointers into @c ue_tb_ptr using **storage order**.
 *
 * For each stored @c DL_TTI.req index @c i in order, if that message has PDSCH, ordinal
 * @c row counts 0,1,2,… (skipping SSB-only DL_TTIs). Writes
 * @c ue_tb_ptr_rows[row] = staged_ptrs[cell_id(i)] when the entry is non-null.
 *
 * Same algorithm as @c PHY_module::process_aggr_pdsch_channel when split-phase merge is active.
 *
 * @param n_dl_tti        @c dl_tti_count() upper bound for indices @c 0..n-1
 * @param has_pdsch_i     @c true iff DL_TTI at index @c i expects TX_DATA (PDSCH PDU present)
 * @param cell_id_i       IPC @c cell_id for DL_TTI at index @c i
 * @param staged_ptrs     Flat array indexed by cell_id; nullptr entries mean "not staged"
 * @param n_staged        Array size (MAX_CELLS_PER_SLOT), for bounds checking
 * @param ue_tb_ptr_rows  Row array (e.g. @c pdsch_params.ue_tb_ptr); must allow @c max_rows entries
 * @param max_rows        Typically @c MAX_PDSCH_UE_PER_TTI
 */
template<typename HasPdschFn, typename CellIdFn>
[[nodiscard]] inline MergeUeTbPtrResult merge_ue_tb_ptr_ordinal(
    std::uint16_t n_dl_tti,
    HasPdschFn&& has_pdsch_i,
    CellIdFn&& cell_id_i,
    std::uint8_t* const* staged_ptrs,
    std::size_t n_staged,
    std::uint8_t** ue_tb_ptr_rows,
    std::uint32_t max_rows)
{
    MergeUeTbPtrResult out{};
    std::uint32_t      ord = 0;
    for (std::uint16_t i = 0; i < n_dl_tti; ++i)
    {
        if (!has_pdsch_i(i))
        {
            continue;
        }
        const std::uint32_t row = ord++;
        if (row >= max_rows)
        {
            ++out.overflow_count;
            continue;
        }
        const std::uint16_t cid = cell_id_i(i);
        if (cid < n_staged && staged_ptrs[cid] != nullptr)
        {
            ue_tb_ptr_rows[row] = staged_ptrs[cid];
            ++out.assigned;
        }
    }
    return out;
}

} // namespace nv::tx_data_h2d

#endif // NV_TX_DATA_H2D_HELPERS_HPP_INCLUDED_
