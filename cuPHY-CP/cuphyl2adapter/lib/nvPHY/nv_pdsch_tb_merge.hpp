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
 * @file nv_pdsch_tb_merge.hpp
 * @brief FapiSlotStorage concept, MergeStats, patch_tb_start_offsets<>, and
 *        merge_and_patch<> — Step 6 Option A split-phase PDSCH merge.
 *
 * These template functions replace the anonymous-namespace helpers in
 * nv_phy_dl_channels.cpp with testable, dependency-injectable equivalents.
 *
 * No nvlog calls are present in the template bodies; this header must be
 * includable from unit-test builds that do not link nvlog.
 */

#ifndef NV_PDSCH_TB_MERGE_HPP_
#define NV_PDSCH_TB_MERGE_HPP_

#include <concepts>
#include <cstdint>
#include <cstring>
#include <span>

#include "aerial/casts/casts.hpp"     // aerial::casts::assume_cast (alignment-checked)
#include "nv_tx_data_h2d_helpers.hpp" // merge_ue_tb_ptr_ordinal, MergeUeTbPtrResult
#include "scf_5g_fapi.h"              // scf_fapi_header_t, scf_fapi_tx_data_req_t, …

namespace nv::pdsch_merge
{

// ---------------------------------------------------------------------------
// FapiSlotStorage concept
// ---------------------------------------------------------------------------

/**
 * Concept satisfied by any type that exposes the DL_TTI lane queries
 * needed by merge_and_patch and patch_tb_start_offsets.
 *
 * Concrete implementation: FapiSlotMessageStorage (nv_fapi_message_storage.hpp).
 * Test surrogate: any struct with the three named members below.
 */
template <typename T>
concept FapiSlotStorage = requires(const T& s, uint16_t i)
{
    {
        s.dl_tti_count()
        } -> std::convertible_to<uint16_t>;
    {
        s.dl_tti_has_pdsch_pdu(i)
        } -> std::convertible_to<bool>;
    {
        s.dl_tti_cell_id(i)
        } -> std::convertible_to<uint16_t>;
};

// ---------------------------------------------------------------------------
// MergeStats
// ---------------------------------------------------------------------------

/**
 * Counters returned by merge_and_patch.
 */
struct MergeStats
{
    uint32_t ue_tb_assigned{};     //!< Rows in ue_tb_ptr[] that received a non-null staged pointer.
    uint32_t tb_offsets_applied{}; //!< tbStartOffset fields patched from TX_DATA TLV metadata.
    uint32_t overflow_count{};     //!< PDSCH ordinals that exceeded MAX_PDSCH_UE_PER_TTI.
};

// ---------------------------------------------------------------------------
// patch_tb_start_offsets
// ---------------------------------------------------------------------------

/**
 * Walk TX_DATA.req TLV metadata and override tbStartOffset in ue_cw_info[].
 *
 * For each DL_TTI at index @p i that has a PDSCH PDU, finds the matching
 * TX_DATA.req buffer via @c staged_msg_bufs[dl_tti_cell_id(i)] and reads the
 * padded byte offset from tag == @c SCF_TX_DATA_OFFSET.  The per-cell CW start
 * index accumulates from @c tx_req->num_pdus so that multi-PDU cells work correctly.
 *
 * Promoted from the anonymous-namespace helper @c apply_tb_start_offsets_from_tx_data
 * in nv_phy_dl_channels.cpp.  No nvlog calls (template in header).
 *
 * @param[in]     slot_store      Satisfies FapiSlotStorage; provides dl_tti_* queries.
 * @param[in]     staged_msg_bufs TX_DATA msg_buf pointers indexed by cell_id.
 * @param[in,out] ue_cw_info      Span over the per-CW parameter array; tbStartOffset is written.
 * @param[in]     slot_u32        Unused in this template (kept for signature symmetry with callers
 *                                that may add logging); named parameter deliberately omitted.
 * @return Number of tbStartOffset fields successfully patched.
 *         Return value must be checked.
 */
template <FapiSlotStorage Storage, typename CwPrm>
requires requires(CwPrm& cw, uint32_t v) { cw.tbStartOffset = v; }
[[nodiscard]] uint32_t patch_tb_start_offsets(
    const Storage&               slot_store,
    std::span<const void* const> staged_msg_bufs,
    std::span<CwPrm>             ue_cw_info,
    uint32_t /*slot_u32*/) noexcept
{
    uint32_t       cw_start = 0;
    uint32_t       applied  = 0;
    const uint16_t n_dl_tti = slot_store.dl_tti_count();
    const uint32_t n_staged = static_cast<uint32_t>(staged_msg_bufs.size());

    for(uint16_t i = 0; i < n_dl_tti; ++i)
    {
        if(!slot_store.dl_tti_has_pdsch_pdu(i))
        {
            continue;
        }

        const uint16_t cid = slot_store.dl_tti_cell_id(i);
        if(cid >= n_staged || staged_msg_bufs[cid] == nullptr) [[unlikely]]
        {
            // Skip this DL_TTI message — its TX_DATA staging is missing or the
            // cell id is out of range. Subsequent cells with valid staging are
            // still processed; cell-with-missing-staging leaves its codewords
            // at their default tbStartOffset (no harm: PDSCH for that cell
            // won't transmit without TB data).
            continue;
        }

        const auto* hdr             = aerial::casts::assume_cast<const scf_fapi_header_t>(staged_msg_bufs[cid]);
        const auto* tx_req          = aerial::casts::assume_cast<const scf_fapi_tx_data_req_t>(hdr->payload);
        const auto* pdu_data        = aerial::casts::assume_cast<const uint8_t>(tx_req->payload);
        uint32_t    pdu_walk_offset = 0;

        // Bounds for the PDU walk.
        // tx_req->msg_hdr.length is the SCF body length in bytes (starting
        // after the body header). Subtract the fixed body prefix (sfn + slot
        // + num_pdus = 3 × uint16_t) to get the bytes available to the PDU
        // array. Enforce this bound before every assume_cast and before
        // advancing pdu_walk_offset — a malformed TX_DATA.req that declares
        // an impossibly large num_tlv would otherwise read past the end of
        // the IPC buffer (assume_cast checks alignment, not bounds).
        constexpr uint32_t k_tx_data_req_prefix_bytes =
            static_cast<uint32_t>(3u * sizeof(uint16_t));
        const uint32_t body_len = tx_req->msg_hdr.length;
        const uint32_t pdu_data_len = body_len > k_tx_data_req_prefix_bytes
            ? body_len - k_tx_data_req_prefix_bytes
            : 0u;

        for(uint16_t p = 0; p < tx_req->num_pdus; ++p)
        {
            // Stop on truncated buffer: not enough room for the fixed pdu_info
            // header at the current offset.
            if(pdu_walk_offset + sizeof(scf_fapi_tx_data_pdu_info_t) > pdu_data_len) [[unlikely]]
            {
                break;
            }

            const auto* dl_pdu = aerial::casts::assume_cast<const scf_fapi_tx_data_pdu_info_t>(
                pdu_data + pdu_walk_offset);

            const uint32_t tlv_bytes =
                static_cast<uint32_t>(dl_pdu->num_tlv)
                * static_cast<uint32_t>(sizeof(scf_fapi_tl_t) + sizeof(uint32_t));
            const uint32_t pdu_total_bytes =
                static_cast<uint32_t>(sizeof(scf_fapi_tx_data_pdu_info_t)) + tlv_bytes;

            // Stop on truncated buffer: declared TLV span would extend past
            // the body. Leave subsequent codewords at their default tbStartOffset.
            if(pdu_walk_offset + pdu_total_bytes > pdu_data_len) [[unlikely]]
            {
                break;
            }

            if(dl_pdu->num_tlv > 0)
            {
                const auto*    tlv    = aerial::casts::assume_cast<const scf_fapi_tl_t>(dl_pdu->tlvs);
                const uint32_t cw_idx = cw_start + static_cast<uint32_t>(p);
                if(tlv->tag == SCF_TX_DATA_OFFSET && cw_idx < static_cast<uint32_t>(ue_cw_info.size()))
                {
                    uint32_t offset{};
                    std::memcpy(&offset, tlv->val, sizeof(uint32_t));
                    ue_cw_info[cw_idx].tbStartOffset = offset;
                    ++applied;
                }
            }
            pdu_walk_offset += pdu_total_bytes;
        }
        cw_start += static_cast<uint32_t>(tx_req->num_pdus);
    }
    return applied;
}

// ---------------------------------------------------------------------------
// merge_and_patch
// ---------------------------------------------------------------------------

/**
 * Merge staged GPU TB pointers into pdsch.ue_tb_ptr[] and patch tbStartOffset.
 *
 * Performs the full split-phase PDSCH merge in a single call:
 *   1. Calls merge_ue_tb_ptr_ordinal to map staged_gpu_ptrs[cell_id] into
 *      pdsch.ue_tb_ptr[ordinal] using DL_TTI storage order.
 *   2. Sets pdsch.tb_data.pBufferType to GPU_BUFFER (value 1) unconditionally.
 *   3. If staged_msg_bufs is non-empty, calls patch_tb_start_offsets to override
 *      tbStartOffset fields from TX_DATA TLV metadata.
 *
 * Templated on both Storage (satisfies FapiSlotStorage) and PdschParams so that
 * tests can use a lightweight surrogate without CUDA/cuPHY dependencies.
 *
 * @param[in]     slot_store       Satisfies FapiSlotStorage.
 * @param[in]     staged_gpu_ptrs  Span of GPU TB pointers indexed by cell_id (size MAX_CELLS_PER_SLOT).
 * @param[in]     staged_msg_bufs  Span of TX_DATA msg_buf pointers indexed by cell_id; may be empty.
 * @param[in,out] pdsch            PDSCH params struct to populate (ue_tb_ptr, tb_data, ue_cw_info).
 * @param[in]     slot_u32         Packed SFN/slot forwarded to patch_tb_start_offsets.
 * @return MergeStats with ue_tb_assigned, tb_offsets_applied, overflow_count.
 *         Return value must be checked.
 */
template <FapiSlotStorage Storage, typename PdschParams>
[[nodiscard]] MergeStats merge_and_patch(
    const Storage&               slot_store,
    std::span<uint8_t* const>    staged_gpu_ptrs,
    std::span<const void* const> staged_msg_bufs,
    PdschParams&                 pdsch,
    uint32_t                     slot_u32) noexcept
{
    MergeStats stats{};

    // Merge gpu_ptrs into ue_tb_ptr[ordinal] using DL_TTI storage order.
    const auto merge_result = nv::tx_data_h2d::merge_ue_tb_ptr_ordinal(
        slot_store.dl_tti_count(),
        [&](std::uint16_t i) { return slot_store.dl_tti_has_pdsch_pdu(i); },
        [&](std::uint16_t i) { return slot_store.dl_tti_cell_id(i); },
        staged_gpu_ptrs.data(),
        staged_gpu_ptrs.size(),
        std::data(pdsch.ue_tb_ptr),
        static_cast<std::uint32_t>(std::size(pdsch.ue_tb_ptr)));

    stats.ue_tb_assigned = merge_result.assigned;
    stats.overflow_count = merge_result.overflow_count;

    // TB data is in GPU memory — set pBufferType = GPU_BUFFER (value 1).
    // Uses static_cast to work with both the anonymous enum in cuphyPdschDataIn_t
    // and enum class TestBufferType in unit tests (both define GPU_BUFFER = 1).
    pdsch.tb_data.pBufferType = static_cast<decltype(pdsch.tb_data.pBufferType)>(1);

    // Patch tbStartOffset from TX_DATA TLV metadata when msg bufs are present.
    if(!staged_msg_bufs.empty())
    {
        auto cw_span = std::span{std::data(pdsch.ue_cw_info), std::size(pdsch.ue_cw_info)};
        stats.tb_offsets_applied = patch_tb_start_offsets(
            slot_store, staged_msg_bufs, cw_span, slot_u32);
    }

    return stats;
}

} // namespace nv::pdsch_merge

#endif // NV_PDSCH_TB_MERGE_HPP_
