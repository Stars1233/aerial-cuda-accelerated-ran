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

#include <array>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>

#include "nv_fapi_message_storage.hpp"
#include "nv_fapi_pdu_utils.hpp"
#include "nv_phy_mac_transport.hpp"
#include "nv_tx_data_h2d_helpers.hpp"
// FAPI_SFN_MAX and MAX_CELLS_PER_SLOT are defined in nv_phy_utils.hpp /
// cuphy.h, but those headers transitively pull in the entire CUDA library.
// Only these two constants are needed here, so define them directly.
static constexpr uint16_t FAPI_SFN_MAX      = 1024;
static constexpr uint16_t MAX_CELLS_PER_SLOT = 20;
#include "scf_5g_fapi.h"
#include "nv_ipc_utils.h"
#include <aerial/casts/casts.hpp>
namespace {

nv::phy_mac_msg_desc make_msg(const uint8_t msg_id, const uint16_t cell_id)
{
    nv::phy_mac_msg_desc msg{};
    msg.msg_id  = msg_id;
    msg.cell_id = cell_id;
    return msg;
}

void store_slot_messages(nv::FapiSlotMessageStorage& storage, char slot_type, uint16_t cell_id)
{
    auto store_or_fail = [&](uint8_t msg_id) {
        EXPECT_EQ(storage.store_message(make_msg(msg_id, cell_id)), nv::StoreResult::Stored);
    };
    switch (slot_type) {
        case 'D':
            store_or_fail(SCF_FAPI_DL_TTI_REQUEST);
            store_or_fail(SCF_FAPI_UL_DCI_REQUEST);
            store_or_fail(SCF_FAPI_TX_DATA_REQUEST);
            store_or_fail(SCF_FAPI_DL_BFW_CVI_REQUEST);
            break;
        case 'U':
            store_or_fail(SCF_FAPI_UL_TTI_REQUEST);
            store_or_fail(SCF_FAPI_UL_BFW_CVI_REQUEST);
            break;
        case 'S':
            store_or_fail(SCF_FAPI_DL_TTI_REQUEST);
            store_or_fail(SCF_FAPI_UL_DCI_REQUEST);
            store_or_fail(SCF_FAPI_TX_DATA_REQUEST);
            store_or_fail(SCF_FAPI_DL_BFW_CVI_REQUEST);
            store_or_fail(SCF_FAPI_UL_TTI_REQUEST);
            store_or_fail(SCF_FAPI_UL_BFW_CVI_REQUEST);
            break;
        default:
            break;
    }
}

struct EarlyArrivalCache {
    std::array<nv::phy_mac_msg_desc, MAX_CELLS_PER_SLOT * 12> cache{};
    uint16_t cached = 0;
};

static sfn_slot_t next_slot(sfn_slot_t curr, uint16_t slot_per_frame)
{
    sfn_slot_t next = curr;
    next.u16.slot++;
    if (next.u16.slot >= slot_per_frame) {
        next.u16.slot = 0;
        next.u16.sfn = next.u16.sfn >= FAPI_SFN_MAX - 1 ? 0 : next.u16.sfn + 1;
    }
    return next;
}

static bool cache_if_early_arrival(EarlyArrivalCache& cache,
                                   sfn_slot_t current_slot,
                                   sfn_slot_t incoming_slot,
                                   bool slot_complete,
                                   uint16_t slot_per_frame)
{
    // Test-only helper: this test validates cache admission/count semantics for
    // next-slot early arrivals, not descriptor payload preservation. A default
    // descriptor is intentionally written to mark occupancy.
    if (slot_complete) {
        return false;
    }
    if (incoming_slot.u32 == next_slot(current_slot, slot_per_frame).u32) {
        if (cache.cached < cache.cache.size()) {
            cache.cache[cache.cached++] = nv::phy_mac_msg_desc{};
            return true;
        }
        return false;
    }
    return false;
}

struct Phase2StoreReplayModel {
    explicit Phase2StoreReplayModel(const std::size_t cells_in) :
        cells(cells_in),
        active_bitmap((cells_in >= 64U) ? ~0ULL : ((1ULL << cells_in) - 1ULL)),
        slot_storage(cells_in)
    {}

    [[nodiscard]] bool feed(const nv::phy_mac_msg_desc& msg, const sfn_slot_t& ss_msg)
    {
        const bool is_slot_ind = (msg.msg_id == SCF_FAPI_SLOT_INDICATION);
        const bool is_slot_rsp = (msg.msg_id == SCF_FAPI_SLOT_RESPONSE);
        const bool is_err_ind = (msg.msg_id == SCF_FAPI_ERROR_INDICATION);

        if (is_slot_ind || is_slot_rsp || is_err_ind)
        {
            control_msg_count++;
            if (is_slot_ind)
            {
                ss_curr = ss_msg;
                eom_bitmap = 0;
                slot_storage.reset_for_slot(ss_msg.u32);
                boundary_slot_ind_count++;
            }
            else if (is_slot_rsp && ss_curr.u32 != SFN_SLOT_INVALID)
            {
                if (msg.cell_id < 64)
                {
                    eom_bitmap |= (1ULL << msg.cell_id);
                }
                if ((eom_bitmap & active_bitmap) == active_bitmap)
                {
                    slot_storage.set_ready(true);
                    replayed_payload_count += payload_count(slot_storage);
                    slot_storage.clear();
                    eom_bitmap = 0;
                    boundary_eom_count++;
                    slot_complete_count++;
                    return true;
                }
            }
            return false;
        }

        if (ss_curr.u32 == SFN_SLOT_INVALID)
        {
            return false;
        }

        if (!slot_storage.has_slot() || slot_storage.tracked_slot() != ss_curr.u32)
        {
            slot_storage.reset_for_slot(ss_curr.u32);
        }

        if (slot_storage.store_message(msg) == nv::StoreResult::Stored)
        {
            payload_store_count++;
        }
        return false;
    }

    [[nodiscard]] static std::size_t payload_count(const nv::FapiSlotMessageStorage& storage)
    {
        // Cast the first term to size_t so the entire sum stays unsigned;
        // uint16_t + uint16_t would otherwise promote to signed int.
        return std::size_t{storage.dl_tti_count()} +
               storage.ul_tti_count() +
               storage.ul_dci_count() +
               storage.tx_data_count() +
               storage.dl_bfw_count() +
               storage.ul_bfw_count();
    }

    std::size_t cells = 0;
    uint64_t active_bitmap = 0;
    uint64_t eom_bitmap = 0;
    sfn_slot_t ss_curr{.u32 = SFN_SLOT_INVALID};
    nv::FapiSlotMessageStorage slot_storage;
    std::size_t control_msg_count = 0;
    std::size_t payload_store_count = 0;
    std::size_t replayed_payload_count = 0;
    std::size_t boundary_slot_ind_count = 0;
    std::size_t boundary_eom_count = 0;
    std::size_t slot_complete_count = 0;
};

} // namespace

#ifdef SCF_FAPI_10_04
// Real assertion against the production fast-path skip helper:
// for_each_tti_msg() at nv_fapi_pdu_utils.hpp:177 short-circuits messages whose
// nPDUsOfEachType[npdus_type_idx] == 0 without ever reading their payload.
// This is the gate process_aggr_csirs_channel / process_aggr_pdsch_channel /
// process_aggr_pdcch_channel / process_aggr_ssb_channel rely on.
TEST(FapiDlTtiReq, ForEachTtiMsgSkipsDlTtiWithZeroPduCount)
{
    constexpr std::size_t kBufSize =
        sizeof(scf_fapi_header_t) + sizeof(scf_fapi_dl_tti_req_t);

    alignas(64) std::array<uint8_t, kBufSize> raw_zero{};
    alignas(64) std::array<uint8_t, kBufSize> raw_nonzero{};

    auto* req_zero    = aerial::casts::assume_cast<scf_fapi_dl_tti_req_t>(
        raw_zero.data() + sizeof(scf_fapi_header_t));
    auto* req_nonzero = aerial::casts::assume_cast<scf_fapi_dl_tti_req_t>(
        raw_nonzero.data() + sizeof(scf_fapi_header_t));
    std::memset(req_zero,    0, sizeof(*req_zero));
    std::memset(req_nonzero, 0, sizeof(*req_nonzero));
    req_nonzero->nPDUsOfEachType[DL_TTI_NPDUS_IDX_CSI_RS] = 3;

    std::array<nv::phy_mac_msg_desc, 2> msgs{};
    msgs[0].msg_id  = SCF_FAPI_DL_TTI_REQUEST;
    msgs[0].cell_id = 0;
    msgs[0].msg_buf = raw_zero.data();
    msgs[0].msg_len = static_cast<int32_t>(kBufSize);
    msgs[1].msg_id  = SCF_FAPI_DL_TTI_REQUEST;
    msgs[1].cell_id = 1;
    msgs[1].msg_buf = raw_nonzero.data();
    msgs[1].msg_len = static_cast<int32_t>(kBufSize);

    std::vector<uint16_t> visited;
    nv::for_each_tti_msg<0u, scf_fapi_dl_tti_req_t>(
        msgs.data(), static_cast<uint16_t>(msgs.size()),
        [](uint16_t, const scf_fapi_dl_tti_req_t& req) { return nv::has_csi_rs(req); },
        /*slot_u32=*/0u, /*ring_idx=*/0u,
        [&visited](uint16_t i,
                   const nv::phy_mac_msg_desc&,
                   const scf_fapi_dl_tti_req_t&) {
            visited.push_back(i);
        });

    ASSERT_EQ(visited.size(), 1u);
    EXPECT_EQ(visited[0], 1u);
}

TEST(FapiDlTtiReq, NpdusEachTypeSkipsCsirsWhenCountZero)
{
    scf_fapi_dl_tti_req_t req{};
    req.nPDUsOfEachType[DL_TTI_PDU_TYPE_CSI_RS] = 0;
    EXPECT_EQ(0u, req.nPDUsOfEachType[DL_TTI_PDU_TYPE_CSI_RS]);
}

TEST(FapiSlotMessageStorage, DlTtiHasPdschPduCachedAtStore)
{
    alignas(64) std::array<uint8_t, 512> raw{};
    auto* req = aerial::casts::assume_cast<scf_fapi_dl_tti_req_t>(raw.data() + sizeof(scf_fapi_header_t));
    std::memset(req, 0, sizeof(*req));
    req->nPDUsOfEachType[DL_TTI_NPDUS_IDX_PDSCH] = 1;

    nv::phy_mac_msg_desc msg{};
    msg.msg_id  = SCF_FAPI_DL_TTI_REQUEST;
    msg.cell_id = 3;
    msg.msg_buf = raw.data();
    msg.msg_len = static_cast<int32_t>(raw.size());

    nv::FapiSlotMessageStorage storage(4);
    ASSERT_EQ(storage.store_message(msg), nv::StoreResult::Stored);
    EXPECT_EQ(storage.dl_tti_count(), 1u);
    EXPECT_TRUE(storage.dl_tti_has_pdsch_pdu(0));
    EXPECT_EQ(storage.dl_tti_pdu_counts(0).by_type[DL_TTI_NPDUS_IDX_PDSCH], 1u);

    req->nPDUsOfEachType[DL_TTI_NPDUS_IDX_PDSCH] = 0;
    ASSERT_EQ(storage.store_message(msg), nv::StoreResult::Stored);
    EXPECT_EQ(storage.dl_tti_count(), 2u);
    EXPECT_FALSE(storage.dl_tti_has_pdsch_pdu(1));
    EXPECT_EQ(storage.dl_tti_pdu_counts(1).by_type[DL_TTI_NPDUS_IDX_PDSCH], 0u);
}

TEST(FapiSlotMessageStorage, UeTbPtrMergeOrdinalMatchesStoredDlTtiOrder)
{
    constexpr std::size_t kMsgs = 3;
    std::array<std::array<uint8_t, 512>, kMsgs> raws{};
    alignas(8) std::array<uint8_t, 4> buf_a{};
    alignas(8) std::array<uint8_t, 4> buf_b{};
    alignas(8) std::array<uint8_t, 4> buf_c{};

    constexpr std::size_t kMaxCells = 64;
    std::array<uint8_t*, kMaxCells> staged{};
    staged[10] = buf_a.data();
    staged[20] = buf_b.data();
    staged[30] = buf_c.data();

    nv::FapiSlotMessageStorage storage(8);
    const std::array<uint16_t, kMsgs> store_order{20, 30, 10};
    for (std::size_t j = 0; j < kMsgs; ++j)
    {
        auto* req =
            reinterpret_cast<scf_fapi_dl_tti_req_t*>(raws[j].data() + sizeof(scf_fapi_header_t));
        std::memset(req, 0, sizeof(*req));
        req->nPDUsOfEachType[DL_TTI_PDU_TYPE_PDSCH] = 1;

        nv::phy_mac_msg_desc msg{};
        msg.msg_id  = SCF_FAPI_DL_TTI_REQUEST;
        msg.cell_id = store_order[j];
        msg.msg_buf = raws[j].data();
        msg.msg_len = static_cast<int32_t>(raws[j].size());
        ASSERT_EQ(storage.store_message(msg), nv::StoreResult::Stored);
    }

    const uint16_t                n    = storage.dl_tti_count();
    const nv::phy_mac_msg_desc*   msgs = storage.dl_tti_messages();
    std::array<uint8_t*, 16>      ue{};
    const auto mr = nv::tx_data_h2d::merge_ue_tb_ptr_ordinal(
        n,
        [&](uint16_t i) { return storage.dl_tti_has_pdsch_pdu(i); },
        [&](uint16_t i) { return static_cast<uint16_t>(msgs[i].cell_id); },
        staged.data(),
        staged.size(),
        ue.data(),
        16U);
    EXPECT_EQ(mr.assigned, 3U);
    EXPECT_EQ(mr.overflow_count, 0U);
    EXPECT_EQ(ue[0], buf_b.data());
    EXPECT_EQ(ue[1], buf_c.data());
    EXPECT_EQ(ue[2], buf_a.data());
}
#endif

TEST(FapiSlotMessageStorage, DlTtiSidecarMatchesPduUtilsAtStore)
{
    alignas(64) std::array<uint8_t, 512> raw{};
    auto* req = aerial::casts::assume_cast<scf_fapi_dl_tti_req_t>(raw.data() + sizeof(scf_fapi_header_t));
    std::memset(req, 0, sizeof(*req));
#ifdef SCF_FAPI_10_04
    req->nPDUsOfEachType[DL_TTI_NPDUS_IDX_PDCCH] = 1;
    req->nPDUsOfEachType[DL_TTI_NPDUS_IDX_PDSCH] = 2;
    req->nPDUsOfEachType[DL_TTI_NPDUS_IDX_CSI_RS] = 1;
#else
    req->num_pdus = 4;
    auto* pdus    = reinterpret_cast<scf_fapi_generic_pdu_info_t*>(req->payload);
    const uint16_t types[] = {DL_TTI_PDU_TYPE_PDCCH, DL_TTI_PDU_TYPE_PDSCH,
                              DL_TTI_PDU_TYPE_PDSCH, DL_TTI_PDU_TYPE_CSI_RS};
    for (uint16_t i = 0; i < 4; ++i)
    {
        pdus[i].pdu_type = types[i];
        pdus[i].pdu_size = sizeof(scf_fapi_generic_pdu_info_t);
    }
#endif

    nv::phy_mac_msg_desc msg{};
    msg.msg_id  = SCF_FAPI_DL_TTI_REQUEST;
    msg.cell_id = 0;
    msg.msg_buf = raw.data();
    msg.msg_len = static_cast<int32_t>(raw.size());

    nv::FapiSlotMessageStorage storage(4);
    ASSERT_EQ(storage.store_message(msg), nv::StoreResult::Stored);

    const nv::NvDlTtiPduCounts sidecar = storage.dl_tti_pdu_counts(0);
    const nv::NvDlTtiPduCounts counts  = nv::make_dl_tti_pdu_counts(*req);
    EXPECT_EQ(sidecar.by_type[DL_TTI_NPDUS_IDX_PDCCH], counts.pdcch());
    EXPECT_EQ(sidecar.by_type[DL_TTI_NPDUS_IDX_PDSCH], counts.pdsch());
    EXPECT_EQ(sidecar.by_type[DL_TTI_NPDUS_IDX_CSI_RS], counts.csi_rs());
}

#ifndef SCF_FAPI_10_04
TEST(FapiSlotMessageStorage, UlTtiSidecarUsesSummaryFieldsAtStore)
{
    alignas(64) std::array<uint8_t, 512> raw{};
    auto* req = aerial::casts::assume_cast<scf_fapi_ul_tti_req_t>(raw.data() + sizeof(scf_fapi_header_t));
    std::memset(req, 0, sizeof(*req));
    req->rach_present = 1;
    req->num_ulsch    = 2;
    req->num_ulcch    = 3;
    req->num_pdus     = 2;
    auto* pdus        = reinterpret_cast<scf_fapi_generic_pdu_info_t*>(req->payload);
    pdus[0].pdu_type  = UL_TTI_PDU_TYPE_SRS;
    pdus[0].pdu_size  = sizeof(scf_fapi_generic_pdu_info_t);
    pdus[1].pdu_type  = UL_TTI_PDU_TYPE_SRS;
    pdus[1].pdu_size  = sizeof(scf_fapi_generic_pdu_info_t);

    nv::phy_mac_msg_desc msg{};
    msg.msg_id  = SCF_FAPI_UL_TTI_REQUEST;
    msg.cell_id = 1;
    msg.msg_buf = raw.data();
    msg.msg_len = static_cast<int32_t>(raw.size());

    nv::FapiSlotMessageStorage storage(4);
    ASSERT_EQ(storage.store_message(msg), nv::StoreResult::Stored);

    const nv::NvUlTtiPduCounts sidecar = storage.ul_tti_pdu_counts(0);
    EXPECT_EQ(sidecar.by_type[UL_TTI_NPDUS_IDX_PRACH], 1u);
    EXPECT_EQ(sidecar.by_type[UL_TTI_NPDUS_IDX_PUSCH], 2u);
    EXPECT_EQ(sidecar.by_type[UL_TTI_NPDUS_IDX_PUCCH_F01], 3u);
    EXPECT_EQ(sidecar.by_type[UL_TTI_NPDUS_IDX_SRS], 2u);
    EXPECT_EQ(sidecar.pucch(), 3u);
}

// SRS-only UL_TTI: num_ulsch==0 with SRS in payload — matches SINGLE_SECT
// "SRS without PUSCH" gating input (PUSCH count from num_ulsch, SRS from walk).
TEST(FapiSlotMessageStorage, UlTtiSrsOnlySidecarHasSrsWithoutPusch)
{
    alignas(64) std::array<uint8_t, 512> raw{};
    auto* req = aerial::casts::assume_cast<scf_fapi_ul_tti_req_t>(raw.data() + sizeof(scf_fapi_header_t));
    std::memset(req, 0, sizeof(*req));
    req->rach_present = 0;
    req->num_ulsch    = 0;
    req->num_ulcch    = 0;
    req->num_pdus     = 1;
    auto* pdu         = reinterpret_cast<scf_fapi_generic_pdu_info_t*>(req->payload);
    pdu->pdu_type     = UL_TTI_PDU_TYPE_SRS;
    pdu->pdu_size     = sizeof(scf_fapi_generic_pdu_info_t);

    nv::phy_mac_msg_desc msg{};
    msg.msg_id  = SCF_FAPI_UL_TTI_REQUEST;
    msg.cell_id = 0;
    msg.msg_buf = raw.data();
    msg.msg_len = static_cast<int32_t>(raw.size());

    nv::FapiSlotMessageStorage storage(4);
    ASSERT_EQ(storage.store_message(msg), nv::StoreResult::Stored);

    const nv::NvUlTtiPduCounts sidecar = storage.ul_tti_pdu_counts(0);
    EXPECT_EQ(sidecar.by_type[UL_TTI_NPDUS_IDX_PUSCH], 0u);
    EXPECT_EQ(sidecar.by_type[UL_TTI_NPDUS_IDX_SRS], 1u);
    EXPECT_TRUE(sidecar.any_channel());
    EXPECT_EQ(nv::make_ul_tti_pdu_counts(*req).by_type[UL_TTI_NPDUS_IDX_PUSCH], 0u);
}

// Truncated UL_TTI: num_pdus advertises 3 SRS PDUs but msg_len only covers one.
// The 10.02 walk must stop at the validated body end (tti_payload_end) and count
// just the in-bounds PDU -- an unbounded walk would read all three and over-read.
TEST(FapiSlotMessageStorage, UlTtiSidecarWalkStopsAtTruncatedBody)
{
    alignas(64) std::array<uint8_t, 512> raw{};
    auto* req = aerial::casts::assume_cast<scf_fapi_ul_tti_req_t>(raw.data() + sizeof(scf_fapi_header_t));
    std::memset(req, 0, sizeof(*req));
    req->num_pdus = 3;
    auto* pdus    = reinterpret_cast<scf_fapi_generic_pdu_info_t*>(req->payload);
    for(int k = 0; k < 3; ++k)
    {
        pdus[k].pdu_type = UL_TTI_PDU_TYPE_SRS;
        pdus[k].pdu_size = sizeof(scf_fapi_generic_pdu_info_t);
    }

    // Body length covers the FAPI header + req prefix + exactly ONE PDU.
    nv::phy_mac_msg_desc msg{};
    msg.msg_id  = SCF_FAPI_UL_TTI_REQUEST;
    msg.cell_id = 0;
    msg.msg_buf = raw.data();
    msg.msg_len = static_cast<int32_t>(sizeof(scf_fapi_header_t) + nv::k_ul_tti_min_body +
                                       sizeof(scf_fapi_generic_pdu_info_t));

    nv::FapiSlotMessageStorage storage(4);
    ASSERT_EQ(storage.store_message(msg), nv::StoreResult::Stored);

    // Only the single in-bounds SRS PDU is counted; the two phantom PDUs past
    // msg_len are not walked.
    EXPECT_EQ(storage.ul_tti_pdu_counts(0).by_type[UL_TTI_NPDUS_IDX_SRS], 1u);
}

// Malformed final PDU: the header fits within msg_len but the declared pdu_size
// crosses the body end. The walk must exclude it -- counting it would activate
// work against a payload that is not fully present.
TEST(FapiSlotMessageStorage, UlTtiSidecarWalkExcludesTruncatedFinalPdu)
{
    alignas(64) std::array<uint8_t, 512> raw{};
    auto* req = aerial::casts::assume_cast<scf_fapi_ul_tti_req_t>(raw.data() + sizeof(scf_fapi_header_t));
    std::memset(req, 0, sizeof(*req));
    req->num_pdus = 1;
    auto* pdu     = reinterpret_cast<scf_fapi_generic_pdu_info_t*>(req->payload);
    pdu->pdu_type = UL_TTI_PDU_TYPE_SRS;
    pdu->pdu_size = static_cast<uint16_t>(2u * sizeof(scf_fapi_generic_pdu_info_t));  // claims a 2-header body...

    // ...but the body only covers the FAPI header + req prefix + ONE header, so
    // the SRS PDU's declared size crosses payload_end.
    nv::phy_mac_msg_desc msg{};
    msg.msg_id  = SCF_FAPI_UL_TTI_REQUEST;
    msg.cell_id = 0;
    msg.msg_buf = raw.data();
    msg.msg_len = static_cast<int32_t>(sizeof(scf_fapi_header_t) + nv::k_ul_tti_min_body +
                                       sizeof(scf_fapi_generic_pdu_info_t));

    nv::FapiSlotMessageStorage storage(4);
    ASSERT_EQ(storage.store_message(msg), nv::StoreResult::Stored);

    // The truncated SRS PDU is not counted.
    EXPECT_EQ(storage.ul_tti_pdu_counts(0).by_type[UL_TTI_NPDUS_IDX_SRS], 0u);
}
#endif

#ifdef SCF_FAPI_10_04
TEST(FapiSlotMessageStorage, UlTtiSrsOnlySidecarFromWireCounters)
{
    alignas(64) std::array<uint8_t, 512> raw{};
    auto* req = aerial::casts::assume_cast<scf_fapi_ul_tti_req_t>(raw.data() + sizeof(scf_fapi_header_t));
    std::memset(req, 0, sizeof(*req));
    req->nPDUsOfEachType[UL_TTI_NPDUS_IDX_SRS] = 1;

    nv::phy_mac_msg_desc msg{};
    msg.msg_id  = SCF_FAPI_UL_TTI_REQUEST;
    msg.cell_id = 0;
    msg.msg_buf = raw.data();
    msg.msg_len = static_cast<int32_t>(raw.size());

    nv::FapiSlotMessageStorage storage(4);
    ASSERT_EQ(storage.store_message(msg), nv::StoreResult::Stored);

    const nv::NvUlTtiPduCounts sidecar = storage.ul_tti_pdu_counts(0);
    EXPECT_EQ(sidecar.by_type[UL_TTI_NPDUS_IDX_PUSCH], 0u);
    EXPECT_EQ(sidecar.by_type[UL_TTI_NPDUS_IDX_SRS], 1u);
    EXPECT_TRUE(sidecar.any_channel());
}
#endif

// TX_DATA PDU fixed header stride must match the active FAPI version so
// store-replay H2D / tbStartOffset walks stay ABI-correct without cw_index
// on 10.02 (10 bytes) and with it on 10.04 (11 bytes, packed).
TEST(TxDataPduInfo, FixedHeaderStrideMatchesFapiVersion)
{
    constexpr std::size_t k_stride = sizeof(scf_fapi_tx_data_pdu_info_t);
#ifdef SCF_FAPI_10_04
    EXPECT_EQ(k_stride, 11u);
    EXPECT_EQ(offsetof(scf_fapi_tx_data_pdu_info_t, cw_index), 6u);
#else
    EXPECT_EQ(k_stride, 10u);
#endif
    EXPECT_EQ(k_stride, offsetof(scf_fapi_tx_data_pdu_info_t, tlvs));
}

TEST(FapiSlotMessageStorage, StoresPerCellDlTtiRequests)
{
    constexpr std::size_t kCells = 20;
    nv::FapiSlotMessageStorage storage(kCells);

    for (uint16_t cell_id = 0; cell_id < kCells; ++cell_id) {
        auto msg = make_msg(SCF_FAPI_DL_TTI_REQUEST, cell_id);
        EXPECT_EQ(storage.store_message(msg), nv::StoreResult::Stored);
    }

    EXPECT_EQ(storage.dl_tti_count(), kCells);
    EXPECT_EQ(storage.dl_tti_messages()[0].cell_id, 0);
    EXPECT_EQ(storage.dl_tti_messages()[kCells - 1].cell_id, kCells - 1);
}

TEST(FapiSlotMessageStorage, SkipsSlotResponseAsControlMessage)
{
    nv::FapiSlotMessageStorage storage(4);
    auto slot_rsp = make_msg(SCF_FAPI_SLOT_RESPONSE, 0);

    EXPECT_EQ(storage.store_message(slot_rsp), nv::StoreResult::ControlSkip);
    EXPECT_EQ(storage.dl_tti_count(), 0u);
    EXPECT_EQ(storage.ul_tti_count(), 0u);
    EXPECT_EQ(storage.ul_dci_count(), 0u);
    EXPECT_EQ(storage.tx_data_count(), 0u);
}

TEST(FapiSlotMessageStorage, Stores3D1S2U4DTDDPatternAndReportsTiming)
{
    constexpr std::size_t kCells = 20;
    constexpr std::size_t kSlots = 20;
    constexpr std::size_t kRepeatsPerSlot = 256;
    constexpr std::array<char, 10> kPattern = {'D','D','D','S','U','U','D','D','D','D'};
    std::array<long long, kSlots> slot_ns{};

    nv::FapiSlotMessageStorage storage(kCells);
    for (std::size_t slot_idx = 0; slot_idx < kSlots; ++slot_idx) {
        const char slot_type = kPattern[slot_idx % kPattern.size()];
        const auto start = std::chrono::steady_clock::now();
        for (std::size_t rep = 0; rep < kRepeatsPerSlot; ++rep) {
            storage.clear();
            for (uint16_t cell_id = 0; cell_id < kCells; ++cell_id) {
                store_slot_messages(storage, slot_type, cell_id);
            }
        }
        const auto end = std::chrono::steady_clock::now();
        slot_ns[slot_idx] = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() /
            static_cast<long long>(kRepeatsPerSlot);

        if (slot_type == 'D') {
            EXPECT_EQ(storage.dl_tti_count(), kCells);
            EXPECT_EQ(storage.ul_dci_count(), kCells);
            EXPECT_EQ(storage.tx_data_count(), kCells);
            EXPECT_EQ(storage.dl_bfw_count(), kCells);
            EXPECT_EQ(storage.ul_tti_count(), 0u);
            EXPECT_EQ(storage.ul_bfw_count(), 0u);
        } else if (slot_type == 'U') {
            EXPECT_EQ(storage.ul_tti_count(), kCells);
            EXPECT_EQ(storage.ul_bfw_count(), kCells);
            EXPECT_EQ(storage.dl_tti_count(), 0u);
            EXPECT_EQ(storage.ul_dci_count(), 0u);
            EXPECT_EQ(storage.tx_data_count(), 0u);
            EXPECT_EQ(storage.dl_bfw_count(), 0u);
        } else if (slot_type == 'S') {
            EXPECT_EQ(storage.dl_tti_count(), kCells);
            EXPECT_EQ(storage.ul_dci_count(), kCells);
            EXPECT_EQ(storage.tx_data_count(), kCells);
            EXPECT_EQ(storage.dl_bfw_count(), kCells);
            EXPECT_EQ(storage.ul_tti_count(), kCells);
            EXPECT_EQ(storage.ul_bfw_count(), kCells);
        }
    }

    long long total_ns = 0;
    std::array<long long, 3> type_totals{};
    std::array<std::size_t, 3> type_counts{};
    for (std::size_t i = 0; i < kSlots; ++i) {
        total_ns += slot_ns[i];
        const char slot_type = kPattern[i % kPattern.size()];
        const std::size_t type_idx = (slot_type == 'D') ? 0 : (slot_type == 'U' ? 1 : 2);
        type_totals[type_idx] += slot_ns[i];
        type_counts[type_idx] += 1;
    }

    std::cout << "[ INFO ] Slot timing (ns):" << std::endl;
    for (std::size_t i = 0; i < kSlots; ++i) {
        const char slot_type = kPattern[i % kPattern.size()];
        std::cout << "  slot " << i << " (" << slot_type << ") : " << slot_ns[i] << " ns" << std::endl;
    }
    const long long avg_ns = total_ns / static_cast<long long>(kSlots);
    std::cout << "[ INFO ] Average per slot: " << avg_ns << " ns" << std::endl;
    std::cout << "[ INFO ] Average per slot type: "
              << "D=" << (type_counts[0] ? (type_totals[0] / static_cast<long long>(type_counts[0])) : 0) << " ns, "
              << "U=" << (type_counts[1] ? (type_totals[1] / static_cast<long long>(type_counts[1])) : 0) << " ns, "
              << "S=" << (type_counts[2] ? (type_totals[2] / static_cast<long long>(type_counts[2])) : 0) << " ns"
              << std::endl;
}

TEST(FapiSlotMessageStorage, CellScalePatternTimingMatrix)
{
    constexpr std::array<std::size_t, 6> kCellsMatrix = {1, 2, 3, 5, 10, 20};
    constexpr std::size_t kSlots = 20;
    constexpr std::size_t kRepeatsPerSlot = 256;
    constexpr std::array<char, 10> kPattern = {'D','D','D','S','U','U','D','D','D','D'};

    std::cout << "[ INFO ] Cell scale timing matrix (ns/slot, ns/msg, ratio_vs_1cell_slot):" << std::endl;
    double baseline_slot_ns = 0.0;
    for (const std::size_t cells : kCellsMatrix)
    {
        nv::FapiSlotMessageStorage storage(cells);
        std::array<long long, kSlots> slot_ns{};
        long long total_ns = 0;

        for (std::size_t slot_idx = 0; slot_idx < kSlots; ++slot_idx)
        {
            const char slot_type = kPattern[slot_idx % kPattern.size()];
            const auto start = std::chrono::steady_clock::now();
            for (std::size_t rep = 0; rep < kRepeatsPerSlot; ++rep)
            {
                storage.clear();
                for (uint16_t cell_id = 0; cell_id < cells; ++cell_id)
                {
                    store_slot_messages(storage, slot_type, cell_id);
                }
            }
            const auto end = std::chrono::steady_clock::now();
            const auto elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() /
                static_cast<long long>(kRepeatsPerSlot);
            slot_ns[slot_idx] = elapsed_ns;
            total_ns += elapsed_ns;

            if (slot_type == 'D') {
                EXPECT_EQ(storage.dl_tti_count(), cells);
                EXPECT_EQ(storage.ul_dci_count(), cells);
                EXPECT_EQ(storage.tx_data_count(), cells);
                EXPECT_EQ(storage.dl_bfw_count(), cells);
            } else if (slot_type == 'U') {
                EXPECT_EQ(storage.ul_tti_count(), cells);
                EXPECT_EQ(storage.ul_bfw_count(), cells);
            } else {
                EXPECT_EQ(storage.dl_tti_count(), cells);
                EXPECT_EQ(storage.ul_dci_count(), cells);
                EXPECT_EQ(storage.tx_data_count(), cells);
                EXPECT_EQ(storage.dl_bfw_count(), cells);
                EXPECT_EQ(storage.ul_tti_count(), cells);
                EXPECT_EQ(storage.ul_bfw_count(), cells);
            }
        }

        const long long avg_slot_ns = total_ns / static_cast<long long>(kSlots);
        const double avg_msg_ns = static_cast<double>(total_ns) / static_cast<double>(kSlots * cells);
        if (cells == 1)
        {
            baseline_slot_ns = static_cast<double>(avg_slot_ns);
        }
        const double ratio_vs_baseline = (baseline_slot_ns > 0.0)
            ? (static_cast<double>(avg_slot_ns) / baseline_slot_ns)
            : 1.0;
        EXPECT_GT(avg_slot_ns, 0);
        std::cout << "  cells=" << cells
                  << " avg_slot_ns=" << avg_slot_ns
                  << " avg_msg_ns=" << avg_msg_ns
                  << " ratio_vs_1cell_slot=" << ratio_vs_baseline
                  << std::endl;
    }
}

TEST(FapiNonSlotMessageStorage, ReplacesConfigStartStopPerCell)
{
    nv::FapiNonSlotMessageStorage storage;

    auto cfg1 = make_msg(SCF_FAPI_CONFIG_REQUEST, 1);
    cfg1.msg_len = 100;
    auto cfg2 = make_msg(SCF_FAPI_CONFIG_REQUEST, 1);
    cfg2.msg_len = 200;
    EXPECT_TRUE(storage.store_message(cfg1));
    EXPECT_TRUE(storage.store_message(cfg2));
    ASSERT_NE(storage.config_request(), nullptr);
    EXPECT_EQ(storage.config_request()->msg_len, 200);

    auto start1 = make_msg(SCF_FAPI_START_REQUEST, 1);
    start1.msg_len = 10;
    auto start2 = make_msg(SCF_FAPI_START_REQUEST, 1);
    start2.msg_len = 20;
    EXPECT_TRUE(storage.store_message(start1));
    EXPECT_TRUE(storage.store_message(start2));
    ASSERT_NE(storage.start_request(), nullptr);
    EXPECT_EQ(storage.start_request()->msg_len, 20);

    auto stop1 = make_msg(SCF_FAPI_STOP_REQUEST, 1);
    stop1.msg_len = 11;
    auto stop2 = make_msg(SCF_FAPI_STOP_REQUEST, 1);
    stop2.msg_len = 21;
    EXPECT_TRUE(storage.store_message(stop1));
    EXPECT_TRUE(storage.store_message(stop2));
    ASSERT_NE(storage.stop_request(), nullptr);
    EXPECT_EQ(storage.stop_request()->msg_len, 21);
}

TEST(FapiSlotMessageStorage, CachesEarlyArrivalForNextSlot)
{
    EarlyArrivalCache cache{};
    sfn_slot_t current{};
    current.u16.sfn = 0;
    current.u16.slot = 1;
    sfn_slot_t next{};
    next.u16.sfn = 0;
    next.u16.slot = 2;
    sfn_slot_t beyond{};
    beyond.u16.sfn = 0;
    beyond.u16.slot = 3;

    constexpr uint16_t kSlotsPerFrame = 10;
    EXPECT_TRUE(cache_if_early_arrival(cache, current, next, false, kSlotsPerFrame));
    EXPECT_EQ(cache.cached, 1);
    EXPECT_FALSE(cache_if_early_arrival(cache, current, beyond, false, kSlotsPerFrame));
    EXPECT_EQ(cache.cached, 1);
}

TEST(FapiSlotMessageStorage, Phase2EomGatedReplayCellMatrix)
{
    constexpr std::array<std::size_t, 6> kCellsMatrix = {1, 2, 3, 5, 10, 20};

    for (const std::size_t cells : kCellsMatrix)
    {
        Phase2StoreReplayModel model(cells);
        sfn_slot_t ss{};
        ss.u16.sfn = 0;
        ss.u16.slot = 1;

        nv::phy_mac_msg_desc slot_ind = make_msg(SCF_FAPI_SLOT_INDICATION, 0);
        EXPECT_FALSE(model.feed(slot_ind, ss));

        nv::phy_mac_msg_desc err_ind = make_msg(SCF_FAPI_ERROR_INDICATION, 0);
        EXPECT_FALSE(model.feed(err_ind, ss));

        for (uint16_t cell_id = 0; cell_id < cells; ++cell_id)
        {
            EXPECT_FALSE(model.feed(make_msg(SCF_FAPI_DL_TTI_REQUEST, cell_id), ss));
            EXPECT_FALSE(model.feed(make_msg(SCF_FAPI_TX_DATA_REQUEST, cell_id), ss));
        }

        for (uint16_t cell_id = 0; cell_id < cells; ++cell_id)
        {
            const bool complete = model.feed(make_msg(SCF_FAPI_SLOT_RESPONSE, cell_id), ss);
            if (cell_id == cells - 1)
            {
                EXPECT_TRUE(complete);
            }
            else
            {
                EXPECT_FALSE(complete);
            }
        }

        EXPECT_EQ(model.boundary_slot_ind_count, 1U);
        EXPECT_EQ(model.boundary_eom_count, 1U);
        EXPECT_EQ(model.slot_complete_count, 1U);
        EXPECT_EQ(model.control_msg_count, cells + 2U);
        EXPECT_EQ(model.payload_store_count, 2U * cells);
        EXPECT_EQ(model.replayed_payload_count, 2U * cells);
    }
}

// ---------------------------------------------------------------------------
// Gap coverage: FapiSlotMessageStorage (items 1–13)
// ---------------------------------------------------------------------------

// 1. Lane overflow — (capacity+1)th store returns Rejected, not Stored.
TEST(FapiSlotMessageStorage, LaneOverflowReturnsRejected)
{
    constexpr std::size_t kCapacity = 2;
    nv::FapiSlotMessageStorage storage(kCapacity);

    EXPECT_EQ(storage.store_message(make_msg(SCF_FAPI_DL_TTI_REQUEST, 0)), nv::StoreResult::Stored);
    EXPECT_EQ(storage.store_message(make_msg(SCF_FAPI_DL_TTI_REQUEST, 1)), nv::StoreResult::Stored);
    EXPECT_EQ(storage.dl_tti_count(), 2u);
    // Capacity exhausted — third store must be rejected.
    EXPECT_EQ(storage.store_message(make_msg(SCF_FAPI_DL_TTI_REQUEST, 2)), nv::StoreResult::Rejected);
    // Count must remain at capacity; no overrun.
    EXPECT_EQ(storage.dl_tti_count(), 2u);
}

// 2. Unknown msg_id → default branch → Rejected.
TEST(FapiSlotMessageStorage, UnknownMsgIdReturnsRejected)
{
    nv::FapiSlotMessageStorage storage(4);
    EXPECT_EQ(storage.store_message(make_msg(0xFF, 0)), nv::StoreResult::Rejected);
    EXPECT_EQ(storage.store_message(make_msg(0x00, 0)), nv::StoreResult::Rejected);
}

// 3. SLOT_INDICATION and ERROR_INDICATION return ControlSkip.
TEST(FapiSlotMessageStorage, ControlIndicationsReturnControlSkip)
{
    nv::FapiSlotMessageStorage storage(4);
    EXPECT_EQ(storage.store_message(make_msg(SCF_FAPI_SLOT_INDICATION, 0)), nv::StoreResult::ControlSkip);
    EXPECT_EQ(storage.store_message(make_msg(SCF_FAPI_ERROR_INDICATION, 0)), nv::StoreResult::ControlSkip);
    // None of these should touch payload counts.
    EXPECT_EQ(storage.dl_tti_count(), 0u);
    EXPECT_EQ(storage.ul_tti_count(), 0u);
}

// 4. All six lanes store and count independently.
TEST(FapiSlotMessageStorage, AllSixLanesStoreAndCountIndependently)
{
    nv::FapiSlotMessageStorage storage(4);

    EXPECT_EQ(storage.store_message(make_msg(SCF_FAPI_DL_TTI_REQUEST,     0)), nv::StoreResult::Stored);
    EXPECT_EQ(storage.store_message(make_msg(SCF_FAPI_UL_TTI_REQUEST,     0)), nv::StoreResult::Stored);
    EXPECT_EQ(storage.store_message(make_msg(SCF_FAPI_UL_DCI_REQUEST,     0)), nv::StoreResult::Stored);
    EXPECT_EQ(storage.store_message(make_msg(SCF_FAPI_TX_DATA_REQUEST,    0)), nv::StoreResult::Stored);
    EXPECT_EQ(storage.store_message(make_msg(SCF_FAPI_DL_BFW_CVI_REQUEST, 0)), nv::StoreResult::Stored);
    EXPECT_EQ(storage.store_message(make_msg(SCF_FAPI_UL_BFW_CVI_REQUEST, 0)), nv::StoreResult::Stored);

    EXPECT_EQ(storage.dl_tti_count(),  1u);
    EXPECT_EQ(storage.ul_tti_count(),  1u);
    EXPECT_EQ(storage.ul_dci_count(),  1u);
    EXPECT_EQ(storage.tx_data_count(), 1u);
    EXPECT_EQ(storage.dl_bfw_count(),  1u);
    EXPECT_EQ(storage.ul_bfw_count(),  1u);
}

// 5. Payload fields are preserved byte-for-byte after store.
TEST(FapiSlotMessageStorage, PreservesPayloadFieldsOnStore)
{
    nv::FapiSlotMessageStorage storage(4);

    nv::phy_mac_msg_desc original{};
    original.msg_id  = SCF_FAPI_DL_TTI_REQUEST;
    original.cell_id = 7;
    original.msg_len = 12345;

    ASSERT_EQ(storage.store_message(original), nv::StoreResult::Stored);
    const nv::phy_mac_msg_desc& stored = storage.dl_tti_messages()[0];
    EXPECT_EQ(stored.msg_id,  original.msg_id);
    EXPECT_EQ(stored.cell_id, original.cell_id);
    EXPECT_EQ(stored.msg_len, original.msg_len);
}

// 6. clear() zeroes all counts, clears slot binding, and resets ready flag.
TEST(FapiSlotMessageStorage, ClearResetsAllCountsAndState)
{
    nv::FapiSlotMessageStorage storage(4);
    storage.reset_for_slot(99);
    (void)storage.store_message(make_msg(SCF_FAPI_DL_TTI_REQUEST, 0));
    (void)storage.store_message(make_msg(SCF_FAPI_UL_TTI_REQUEST, 0));
    storage.set_ready(true);

    storage.clear();

    EXPECT_EQ(storage.dl_tti_count(),  0u);
    EXPECT_EQ(storage.ul_tti_count(),  0u);
    EXPECT_EQ(storage.ul_dci_count(),  0u);
    EXPECT_EQ(storage.tx_data_count(), 0u);
    EXPECT_EQ(storage.dl_bfw_count(),  0u);
    EXPECT_EQ(storage.ul_bfw_count(),  0u);
    EXPECT_FALSE(storage.has_slot());
    EXPECT_FALSE(storage.ready());

    // Storage must be usable again after clear.
    EXPECT_EQ(storage.store_message(make_msg(SCF_FAPI_DL_TTI_REQUEST, 0)), nv::StoreResult::Stored);
    EXPECT_EQ(storage.dl_tti_count(), 1u);
}

// 7. reset_for_slot() binds the slot tag and clears counts.
TEST(FapiSlotMessageStorage, ResetForSlotBindsSlotAndClearsCounts)
{
    nv::FapiSlotMessageStorage storage(4);
    (void)storage.store_message(make_msg(SCF_FAPI_DL_TTI_REQUEST, 0));
    EXPECT_EQ(storage.dl_tti_count(), 1u);

    constexpr uint32_t kSlot = 42u;
    storage.reset_for_slot(kSlot);

    EXPECT_TRUE(storage.has_slot());
    EXPECT_EQ(storage.tracked_slot(), kSlot);
    EXPECT_EQ(storage.dl_tti_count(), 0u);
    EXPECT_FALSE(storage.ready());
}

// bind_to_slot() drives the three-way BindOutcome contract that bind_ring_slot()
// and the store path rely on: FreshBind clears from cold, AlreadyBound preserves
// the ring's accumulated state (no mid-slot wipe), Rebound clears a drained ring's
// stale state. store_slot_message discards the outcome, so this is the only place
// the "same slot => AlreadyBound => no clear" invariant is pinned.
TEST(FapiSlotMessageStorage, BindToSlotOutcomeTransitions)
{
    using BindOutcome = nv::FapiSlotMessageStorage::BindOutcome;
    nv::FapiSlotMessageStorage storage(4);
    constexpr uint32_t SLOT_X = 0x0001'0002U;
    constexpr uint32_t SLOT_Y = 0x0001'0003U;

    // Unbound -> bind_to_slot(X): FreshBind, binds X and clears from cold.
    (void)storage.store_message(make_msg(SCF_FAPI_DL_TTI_REQUEST, 0));
    EXPECT_EQ(storage.dl_tti_count(), 1U);
    EXPECT_EQ(storage.bind_to_slot(SLOT_X), BindOutcome::FreshBind);
    EXPECT_TRUE(storage.has_slot());
    EXPECT_EQ(storage.tracked_slot(), SLOT_X);
    EXPECT_EQ(storage.dl_tti_count(), 0U);

    // Bound-to-X -> bind_to_slot(X): AlreadyBound, ring state preserved (no wipe).
    (void)storage.store_message(make_msg(SCF_FAPI_DL_TTI_REQUEST, 0));
    EXPECT_EQ(storage.dl_tti_count(), 1U);
    EXPECT_EQ(storage.bind_to_slot(SLOT_X), BindOutcome::AlreadyBound);
    EXPECT_EQ(storage.tracked_slot(), SLOT_X);
    EXPECT_EQ(storage.dl_tti_count(), 1U);

    // Bound-to-X -> bind_to_slot(Y): Rebound, binds Y and clears the stale state.
    EXPECT_EQ(storage.bind_to_slot(SLOT_Y), BindOutcome::Rebound);
    EXPECT_TRUE(storage.has_slot());
    EXPECT_EQ(storage.tracked_slot(), SLOT_Y);
    EXPECT_EQ(storage.dl_tti_count(), 0U);
}

// 8. set_ready / ready round-trip; clear() resets the flag.
TEST(FapiSlotMessageStorage, SetReadyFlagAndClearResetsIt)
{
    nv::FapiSlotMessageStorage storage(4);
    EXPECT_FALSE(storage.ready());

    storage.set_ready(true);
    EXPECT_TRUE(storage.ready());

    storage.set_ready(false);
    EXPECT_FALSE(storage.ready());

    storage.set_ready(true);
    storage.clear();
    EXPECT_FALSE(storage.ready());
}

// 9. Early-arrival cache: SFN slot wraparound (last slot → slot 0 of next SFN).
TEST(FapiSlotMessageStorage, EarlyArrivalCachesSfnWraparound)
{
    EarlyArrivalCache cache{};
    constexpr uint16_t kSlotsPerFrame = 10;

    sfn_slot_t current{};
    current.u16.sfn  = 5;
    current.u16.slot = kSlotsPerFrame - 1;  // last slot of frame

    sfn_slot_t next_frame_slot0{};
    next_frame_slot0.u16.sfn  = 6;
    next_frame_slot0.u16.slot = 0;

    EXPECT_TRUE(cache_if_early_arrival(cache, current, next_frame_slot0, false, kSlotsPerFrame));
    EXPECT_EQ(cache.cached, 1u);

    // A slot two ahead must not be admitted.
    sfn_slot_t two_ahead{};
    two_ahead.u16.sfn  = 6;
    two_ahead.u16.slot = 1;
    EXPECT_FALSE(cache_if_early_arrival(cache, current, two_ahead, false, kSlotsPerFrame));
    EXPECT_EQ(cache.cached, 1u);
}

// 10. Early-arrival cache: admission fails when cache is full.
TEST(FapiSlotMessageStorage, EarlyArrivalCacheFullRejectsFurtherAdmissions)
{
    EarlyArrivalCache cache{};
    constexpr uint16_t kSlotsPerFrame = 20;

    sfn_slot_t current{};
    current.u16.sfn  = 0;
    current.u16.slot = 0;
    sfn_slot_t next_slot_val{};
    next_slot_val.u16.sfn  = 0;
    next_slot_val.u16.slot = 1;

    // Fill the cache to capacity.
    while (cache.cached < static_cast<uint16_t>(cache.cache.size()))
    {
        ASSERT_TRUE(cache_if_early_arrival(cache, current, next_slot_val, false, kSlotsPerFrame));
    }
    EXPECT_EQ(cache.cached, static_cast<uint16_t>(cache.cache.size()));

    // One more admission must fail.
    EXPECT_FALSE(cache_if_early_arrival(cache, current, next_slot_val, false, kSlotsPerFrame));
    EXPECT_EQ(cache.cached, static_cast<uint16_t>(cache.cache.size()));
}

// 11. Phase2: payload arriving before the first SLOT_INDICATION is silently dropped.
TEST(FapiSlotMessageStorage, Phase2PayloadBeforeSlotIndicationIsDropped)
{
    Phase2StoreReplayModel model(4);
    sfn_slot_t ss{};
    ss.u16.sfn  = 0;
    ss.u16.slot = 1;

    // No SLOT_IND has been fed yet — ss_curr is SFN_SLOT_INVALID.
    EXPECT_FALSE(model.feed(make_msg(SCF_FAPI_DL_TTI_REQUEST, 0), ss));
    EXPECT_FALSE(model.feed(make_msg(SCF_FAPI_TX_DATA_REQUEST, 1), ss));

    EXPECT_EQ(model.payload_store_count, 0u);
    EXPECT_EQ(model.slot_complete_count, 0u);
}

// 12. Phase2: multi-slot run accumulates correct cumulative counters.
TEST(FapiSlotMessageStorage, Phase2MultiSlotRunAccumulatesCounts)
{
    constexpr std::size_t kCells = 3;
    constexpr std::size_t kSlots = 3;
    Phase2StoreReplayModel model(kCells);

    sfn_slot_t ss{};
    ss.u16.sfn  = 0;
    ss.u16.slot = 0;

    for (std::size_t slot_idx = 0; slot_idx < kSlots; ++slot_idx)
    {
        // SLOT_IND opens the slot.
        EXPECT_FALSE(model.feed(make_msg(SCF_FAPI_SLOT_INDICATION, 0), ss));

        // Two payload types per cell.
        for (uint16_t cell_id = 0; cell_id < kCells; ++cell_id)
        {
            EXPECT_FALSE(model.feed(make_msg(SCF_FAPI_DL_TTI_REQUEST,  cell_id), ss));
            EXPECT_FALSE(model.feed(make_msg(SCF_FAPI_TX_DATA_REQUEST, cell_id), ss));
        }

        // EOM per cell — last one completes the slot.
        for (uint16_t cell_id = 0; cell_id < kCells; ++cell_id)
        {
            const bool complete = model.feed(make_msg(SCF_FAPI_SLOT_RESPONSE, cell_id), ss);
            EXPECT_EQ(complete, (cell_id == kCells - 1));
        }

        ss.u16.slot++;
    }

    EXPECT_EQ(model.boundary_slot_ind_count, kSlots);
    EXPECT_EQ(model.boundary_eom_count,      kSlots);
    EXPECT_EQ(model.slot_complete_count,     kSlots);
    EXPECT_EQ(model.payload_store_count,     kSlots * 2U * kCells);
    EXPECT_EQ(model.replayed_payload_count,  kSlots * 2U * kCells);
}

// 13. Phase2: duplicate SLOT_RESPONSE for the same cell is idempotent — slot
//     must not complete early and the bitmap must not be double-counted.
TEST(FapiSlotMessageStorage, Phase2DuplicateSlotResponseIsIdempotent)
{
    constexpr std::size_t kCells = 3;
    Phase2StoreReplayModel model(kCells);

    sfn_slot_t ss{};
    ss.u16.sfn  = 0;
    ss.u16.slot = 1;

    EXPECT_FALSE(model.feed(make_msg(SCF_FAPI_SLOT_INDICATION, 0), ss));

    for (uint16_t cell_id = 0; cell_id < kCells; ++cell_id)
    {
        EXPECT_FALSE(model.feed(make_msg(SCF_FAPI_DL_TTI_REQUEST, cell_id), ss));
    }

    // Send SLOT_RESPONSE for cell 0 twice — should not complete on the duplicate.
    EXPECT_FALSE(model.feed(make_msg(SCF_FAPI_SLOT_RESPONSE, 0), ss));
    EXPECT_FALSE(model.feed(make_msg(SCF_FAPI_SLOT_RESPONSE, 0), ss));  // duplicate

    // Remaining cells complete normally.
    EXPECT_FALSE(model.feed(make_msg(SCF_FAPI_SLOT_RESPONSE, 1), ss));
    EXPECT_TRUE(model.feed(make_msg(SCF_FAPI_SLOT_RESPONSE, 2), ss));

    EXPECT_EQ(model.slot_complete_count, 1U);
    EXPECT_EQ(model.boundary_eom_count,  1U);
}

// ---------------------------------------------------------------------------
// Gap coverage: FapiNonSlotMessageStorage (items 14–17)
// ---------------------------------------------------------------------------

// 14. Initial state: all three accessors return nullptr before any store.
TEST(FapiNonSlotMessageStorage, InitialStateAllAccessorsReturnNullptr)
{
    nv::FapiNonSlotMessageStorage storage;
    EXPECT_EQ(storage.config_request(), nullptr);
    EXPECT_EQ(storage.start_request(),  nullptr);
    EXPECT_EQ(storage.stop_request(),   nullptr);
}

// 15. Unknown msg_id returns false and no slot is populated.
TEST(FapiNonSlotMessageStorage, UnknownMsgIdReturnsFalse)
{
    nv::FapiNonSlotMessageStorage storage;
    EXPECT_FALSE(storage.store_message(make_msg(SCF_FAPI_DL_TTI_REQUEST, 0)));
    EXPECT_FALSE(storage.store_message(make_msg(0xFF, 0)));
    EXPECT_EQ(storage.config_request(), nullptr);
    EXPECT_EQ(storage.start_request(),  nullptr);
    EXPECT_EQ(storage.stop_request(),   nullptr);
}

// 16. clear() resets all three slots to nullptr.
TEST(FapiNonSlotMessageStorage, ClearResetsAllToNullptr)
{
    nv::FapiNonSlotMessageStorage storage;
    EXPECT_TRUE(storage.store_message(make_msg(SCF_FAPI_CONFIG_REQUEST, 0)));
    EXPECT_TRUE(storage.store_message(make_msg(SCF_FAPI_START_REQUEST,  0)));
    EXPECT_TRUE(storage.store_message(make_msg(SCF_FAPI_STOP_REQUEST,   0)));

    storage.clear();

    EXPECT_EQ(storage.config_request(), nullptr);
    EXPECT_EQ(storage.start_request(),  nullptr);
    EXPECT_EQ(storage.stop_request(),   nullptr);
}

// 17. Replace without transport: new message is stored, previous value is gone.
//     (The transport-with-release path requires integration testing because
//     phy_mac_transport_wrapper::rx_release is non-virtual and cannot be mocked.)
TEST(FapiNonSlotMessageStorage, ReplaceWithoutTransportPreservesNewValue)
{
    nv::FapiNonSlotMessageStorage storage;

    auto first = make_msg(SCF_FAPI_CONFIG_REQUEST, 0);
    first.msg_len = 111;
    auto second = make_msg(SCF_FAPI_CONFIG_REQUEST, 0);
    second.msg_len = 222;

    EXPECT_TRUE(storage.store_message(first,  nv::NullFapiTransport{}));
    const auto* p1 = storage.config_request();
    ASSERT_NE(p1, nullptr);
    EXPECT_EQ(p1->msg_len, 111u);

    EXPECT_TRUE(storage.store_message(second, nv::NullFapiTransport{}));
    const auto* p2 = storage.config_request();
    ASSERT_NE(p2, nullptr);
    EXPECT_EQ(p2->msg_len, 222u);
}

// ===========================================================================
// Per-ring telemetry stamping at store-path ingestion.
// ===========================================================================

#include <atomic>
#include <thread>

#include "nv_slot_telemetry.hpp"
#include "nv_slot_task_pool.hpp"   // SLOT_STORAGE_DEPTH, slot_index_from_u32

namespace {

constexpr uint32_t kSpfMu1 = 20U;  // 20 slots/frame (numerology µ=1)

constexpr uint32_t make_slot_u32(uint16_t sfn, uint16_t slot)
{
    return (static_cast<uint32_t>(sfn) << 16U) | static_cast<uint32_t>(slot);
}

constexpr sfn_slot_t ss(uint16_t sfn, uint16_t slot)
{
    return sfn_slot_t{.u32 = make_slot_u32(sfn, slot)};
}

// Stub DL_TTI payload carrying n_csi_rs CSI-RS PDUs. The PDUs are what 10.02
// counts; 10.04 additionally carries the per-type counters.
struct DlTtiPayloadStub final {
    static constexpr std::size_t kMaxPdus = 4;
    std::array<uint8_t, sizeof(scf_fapi_header_t) + sizeof(scf_fapi_dl_tti_req_t)
                            + kMaxPdus * sizeof(scf_fapi_generic_pdu_info_t)> bytes{};

    explicit DlTtiPayloadStub(uint8_t n_csi_rs)
    {
        assert(n_csi_rs <= kMaxPdus);
        auto* req = reinterpret_cast<scf_fapi_dl_tti_req_t*>(bytes.data() + sizeof(scf_fapi_header_t));
        req->num_pdus = n_csi_rs;
        auto* pdus = reinterpret_cast<scf_fapi_generic_pdu_info_t*>(req->payload);
        for (uint16_t i = 0; i < n_csi_rs; ++i)
        {
            pdus[i].pdu_type = DL_TTI_PDU_TYPE_CSI_RS;
            pdus[i].pdu_size = sizeof(scf_fapi_generic_pdu_info_t);
        }
#ifdef SCF_FAPI_10_04
        req->nPDUsOfEachType[DL_TTI_NPDUS_IDX_CSI_RS] = n_csi_rs;
#endif
    }

    [[nodiscard]] nv::phy_mac_msg_desc desc(uint16_t cell_id)
    {
        return nv_ipc_msg_t{
            .msg_id    = SCF_FAPI_DL_TTI_REQUEST,
            .cell_id   = cell_id,
            .msg_len   = static_cast<int32_t>(bytes.size()),
            .data_len  = 0,
            .data_pool = 0,
            .msg_buf   = bytes.data(),
            .data_buf  = nullptr,
        };
    }
};

// Test-only model that mirrors the production per-ring telemetry path:
// payload+store -> stamp_ring_telemetry; SLOT.RESP -> per-ring tick stamp;
// new (ring, slot) binding -> snapshot reset; ERROR.IND -> no-op.
struct StorePathTelemetrySim final {
    static constexpr uint32_t DEPTH = nv::SLOT_STORAGE_DEPTH;

    explicit StorePathTelemetrySim(std::size_t cells)
        : storages{nv::FapiSlotMessageStorage{cells},
                   nv::FapiSlotMessageStorage{cells},
                   nv::FapiSlotMessageStorage{cells}}
    {}

    [[nodiscard]] static uint32_t ring_idx(uint32_t slot_u32)
    {
        return nv::slot_index_from_u32(slot_u32, kSpfMu1) % DEPTH;
    }

    void payload(const nv::phy_mac_msg_desc& msg, sfn_slot_t ss_msg,
                 std::chrono::nanoseconds now)
    {
        const uint32_t r = ring_idx(ss_msg.u32);
        auto& store = storages[r];
        // Storage rebind + snapshot reset via the single bind_to_slot tell (mirrors
        // PHY_module::bind_ring_slot): the snapshot is cleared only when the bind
        // actually transitions state (FreshBind / Rebound), not when AlreadyBound.
        if (store.bind_to_slot(ss_msg.u32) != nv::FapiSlotMessageStorage::BindOutcome::AlreadyBound) {
            snaps[r] = nv::SlotTelemetrySnapshot{};
        }
        if (store.store_message(msg) != nv::StoreResult::Stored) {
            return;
        }
        bool has_csirs = false;
        if (msg.msg_id == SCF_FAPI_DL_TTI_REQUEST) {
            const uint16_t n = store.dl_tti_count();
            if (n > 0) {
                has_csirs = store.dl_tti_pdu_counts(static_cast<uint16_t>(n - 1))
                                .by_type[DL_TTI_NPDUS_IDX_CSI_RS] > 0;
            }
        }
        nv::stamp_ring_telemetry(snaps[r], now, msg.msg_id, has_csirs);
    }

    void slot_resp(sfn_slot_t ss_msg, std::chrono::nanoseconds now)
    {
        snaps[ring_idx(ss_msg.u32)].last_fapi_msg_tick = now;
    }

    // Snapshot stays until the next ring rebind clears it. 
    void error_ind(sfn_slot_t /*ss_msg*/, std::chrono::nanoseconds /*now*/) {}

    std::array<nv::FapiSlotMessageStorage, DEPTH> storages{};
    std::array<nv::SlotTelemetrySnapshot, DEPTH>  snaps{};
};

} // namespace

TEST(StoreTelemetry, MultiSlotIngestionStampsPerRingFields)
{
    // 5 slots S/D/U/CSIRS-D/S, partly interleaved. Slot N+3 reuses ring(N)
    // and slot N+4 reuses ring(N+1), exercising the rebind-resets-snapshot
    // path. Cross-ring isolation, CSI-RS detection, UL_DCI is_dl flip,
    // TX_DATA gate-skip, and first-msg gate are all covered.

    StorePathTelemetrySim sim(/*cells=*/3);

    DlTtiPayloadStub no_csirs{/*n_csi_rs=*/0};
    DlTtiPayloadStub with_csirs{/*n_csi_rs=*/2};

    using ns = std::chrono::nanoseconds;
    const ns t_n_first(1000), t_n_dlc(1001), t_n_tx(1002), t_n_ul(1003), t_n_resp(1100);
    const ns t_n1_dl(2000), t_n1_dci(2001), t_n1_tx(2002), t_n1_resp(2100);
    const ns t_n2_ul(3000), t_n2_resp(3100);
    const ns t_n3_dl(4000), t_n3_dci(4001), t_n3_resp(4100);
    const ns t_n4_dl(5000), t_n4_dci(5001), t_n4_tx(5002), t_n4_ul(5003), t_n4_resp(5100);

    // --- Slot N (ring 0) — interleaved with N+1's start ---
    sim.payload(no_csirs.desc(/*cell=*/0), ss(0, 0), t_n_first);  // first DL_TTI for N
    sim.payload(no_csirs.desc(/*cell=*/1), ss(0, 1), t_n1_dl);    // first DL_TTI for N+1 (ring 1)
    sim.payload(make_msg(SCF_FAPI_UL_DCI_REQUEST, 0), ss(0, 0), t_n_dlc);
    sim.payload(make_msg(SCF_FAPI_TX_DATA_REQUEST, 0), ss(0, 0), t_n_tx);
    sim.payload(make_msg(SCF_FAPI_UL_TTI_REQUEST, 0), ss(0, 0), t_n_ul);
    sim.slot_resp(ss(0, 0), t_n_resp);

    // --- Slot N+1 (ring 1) — remaining payloads ---
    sim.payload(make_msg(SCF_FAPI_UL_DCI_REQUEST, 1), ss(0, 1), t_n1_dci);
    sim.payload(make_msg(SCF_FAPI_TX_DATA_REQUEST, 1), ss(0, 1), t_n1_tx);
    sim.slot_resp(ss(0, 1), t_n1_resp);

    // --- Slot N+2 (ring 2) — UL-only ---
    sim.payload(make_msg(SCF_FAPI_UL_TTI_REQUEST, 2), ss(0, 2), t_n2_ul);
    sim.slot_resp(ss(0, 2), t_n2_resp);

    // --- Slot N+3 (ring 0 again — rebind) — DL with CSI-RS ---
    sim.payload(with_csirs.desc(/*cell=*/0), ss(0, 3), t_n3_dl);
    sim.payload(make_msg(SCF_FAPI_UL_DCI_REQUEST, 0), ss(0, 3), t_n3_dci);
    sim.slot_resp(ss(0, 3), t_n3_resp);

    // --- Slot N+4 (ring 1 again — rebind) — full S pattern ---
    sim.payload(no_csirs.desc(/*cell=*/1), ss(0, 4), t_n4_dl);
    sim.payload(make_msg(SCF_FAPI_UL_DCI_REQUEST, 1), ss(0, 4), t_n4_dci);
    sim.payload(make_msg(SCF_FAPI_TX_DATA_REQUEST, 1), ss(0, 4), t_n4_tx);
    sim.payload(make_msg(SCF_FAPI_UL_TTI_REQUEST, 1), ss(0, 4), t_n4_ul);
    sim.slot_resp(ss(0, 4), t_n4_resp);

    // Ring 0 → slot N+3 (DL + CSI-RS + UL_DCI). N's stamps must have been
    // wiped by the rebind.
    {
        SCOPED_TRACE("ring 0 — final occupant: slot N+3 (DL with CSI-RS, UL_DCI)");
        const auto& r0 = sim.snaps[StorePathTelemetrySim::ring_idx(make_slot_u32(0, 3))];
        EXPECT_EQ(r0.l2a_start_tick.count(), t_n3_dl.count())
            << "gate must fire on first slot-typed msg (DL_TTI), not re-stamp on later ones";
        EXPECT_EQ(r0.last_fapi_msg_tick.count(), t_n3_resp.count())
            << "SLOT.RESP per-ring stamp updates last_fapi_msg_tick";
        EXPECT_TRUE(r0.is_dl_slot)  << "DL_TTI sets is_dl_slot";
        EXPECT_FALSE(r0.is_ul_slot) << "no UL msgs in slot N+3";
        EXPECT_TRUE(r0.is_csirs_slot) << "DL_TTI CSI-RS count must set is_csirs_slot";
    }

    // Ring 1 → slot N+4 (full S pattern). The earlier slot N+1 stamps in this
    // ring must have been wiped by the rebind.
    {
        SCOPED_TRACE("ring 1 — final occupant: slot N+4 (S pattern)");
        const auto& r1 = sim.snaps[StorePathTelemetrySim::ring_idx(make_slot_u32(0, 4))];
        EXPECT_EQ(r1.l2a_start_tick.count(), t_n4_dl.count());
        EXPECT_EQ(r1.last_fapi_msg_tick.count(), t_n4_resp.count());
        EXPECT_TRUE(r1.is_dl_slot)  << "DL_TTI and UL_DCI both flip is_dl_slot";
        EXPECT_TRUE(r1.is_ul_slot)  << "UL_TTI flips is_ul_slot";
        EXPECT_FALSE(r1.is_csirs_slot) << "DL_TTI without CSI-RS keeps is_csirs_slot=false";
    }

    // Ring 2 → slot N+2 (UL only).
    {
        SCOPED_TRACE("ring 2 — slot N+2 (UL only)");
        const auto& r2 = sim.snaps[StorePathTelemetrySim::ring_idx(make_slot_u32(0, 2))];
        EXPECT_EQ(r2.l2a_start_tick.count(), t_n2_ul.count())
            << "first slot-typed msg is UL_TTI";
        EXPECT_EQ(r2.last_fapi_msg_tick.count(), t_n2_resp.count());
        EXPECT_FALSE(r2.is_dl_slot);
        EXPECT_TRUE(r2.is_ul_slot);
        EXPECT_FALSE(r2.is_csirs_slot);
    }

    // TX_DATA-only slot: gate must not fire.
    {
        SCOPED_TRACE("TX_DATA-only slot: gate must not fire");
        StorePathTelemetrySim sim2(/*cells=*/1);
        const auto t_tx0(ns{6000}), t_tx1(ns{6001}), t_resp(ns{6100});
        sim2.payload(make_msg(SCF_FAPI_TX_DATA_REQUEST, 0), ss(0, 0), t_tx0);
        sim2.payload(make_msg(SCF_FAPI_TX_DATA_REQUEST, 0), ss(0, 0), t_tx1);
        sim2.slot_resp(ss(0, 0), t_resp);
        const auto& r = sim2.snaps[0];
        EXPECT_EQ(r.l2a_start_tick.count(), 0) << "TX_DATA must not stamp l2a_start_tick";
        EXPECT_EQ(r.last_fapi_msg_tick.count(), t_resp.count());
        EXPECT_FALSE(r.is_dl_slot);
        EXPECT_FALSE(r.is_ul_slot);
    }

    // Gate fires once: subsequent slot-typed msgs must not re-stamp l2a_start_tick.
    {
        SCOPED_TRACE("UL_TTI first then DL_TTI/TX_DATA/SLOT.RESP — gate stays at T0");
        StorePathTelemetrySim sim3(/*cells=*/1);
        const auto t0(ns{7000}), t1(ns{7001}), t2(ns{7002}), tr(ns{7100});
        sim3.payload(make_msg(SCF_FAPI_UL_TTI_REQUEST, 0), ss(0, 0), t0);
        sim3.payload(make_msg(SCF_FAPI_DL_TTI_REQUEST, 0), ss(0, 0), t1);
        sim3.payload(make_msg(SCF_FAPI_TX_DATA_REQUEST, 0), ss(0, 0), t2);
        sim3.slot_resp(ss(0, 0), tr);
        const auto& r = sim3.snaps[0];
        EXPECT_EQ(r.l2a_start_tick.count(), t0.count())
            << "gate fires once on first slot-typed msg; subsequent DL_TTI must not re-stamp";
        EXPECT_EQ(r.last_fapi_msg_tick.count(), tr.count());
        EXPECT_TRUE(r.is_ul_slot) << "UL_TTI flips is_ul_slot";
        EXPECT_TRUE(r.is_dl_slot) << "DL_TTI flips is_dl_slot (both set on S slot)";
    }
}

TEST(StoreTelemetry, ErrorIndDoesNotWipeSnapshotRebindDoes)
{
    // ERROR.IND leaves the snapshot untouched; the next slot N+SLOT_STORAGE_DEPTH
    // ingest hits ring(N) and clears it via the rebind path.

    StorePathTelemetrySim sim(/*cells=*/1);
    DlTtiPayloadStub no_csirs(0);
    using ns = std::chrono::nanoseconds;

    // Populate ring 0 with slot N's stamps.
    sim.payload(no_csirs.desc(0), ss(0, 0), ns{1000});
    sim.payload(make_msg(SCF_FAPI_UL_TTI_REQUEST, 0), ss(0, 0), ns{1001});

    {
        const auto& r0 = sim.snaps[0];
        ASSERT_EQ(r0.l2a_start_tick.count(), 1000);
        ASSERT_TRUE(r0.is_dl_slot);
        ASSERT_TRUE(r0.is_ul_slot);
    }

    // ERROR.IND must not mutate the snapshot.
    sim.error_ind(ss(0, 0), ns{1200});
    {
        SCOPED_TRACE("post-ERROR.IND: ring(0) snapshot unchanged");
        const auto& r0 = sim.snaps[0];
        EXPECT_EQ(r0.l2a_start_tick.count(), 1000);
        EXPECT_TRUE(r0.is_dl_slot);
        EXPECT_TRUE(r0.is_ul_slot);
    }

    // Slot N+3 (ring 0 rebind) wipes the snapshot.
    sim.payload(make_msg(SCF_FAPI_UL_DCI_REQUEST, 0), ss(0, 3), ns{2000});
    {
        SCOPED_TRACE("post-rebind: ring(0) shows only slot N+3's data");
        const auto& r0 = sim.snaps[0];
        EXPECT_EQ(r0.l2a_start_tick.count(), 2000) << "fresh gate for N+3";
        EXPECT_TRUE(r0.is_dl_slot)  << "UL_DCI flips is_dl_slot";
        EXPECT_FALSE(r0.is_ul_slot) << "N's is_ul_slot wiped on rebind";
    }
}

// Worker-publish vs msg-thread-rebind race on slot_telemetry_[ring], for TSan.
//
// Prod protocol (GT-12162 ordering invariant + GT-12390 bind_ring_slot):
//   - The completing task holds tasks_in_flight[ring] >= 1 across publish_slot_command,
//     which value-copies slot_telemetry_[ring] (the worker's read).
//   - store_slot_message rebinds a ring (bind_ring_slot -> reset slot_telemetry_[ring],
//     the msg thread's write) only when tasks_in_flight[old_ring] == 0.
// The read and the write are therefore mutually exclusive via the tasks_in_flight gate.
//
// This drives the real FapiSlotMessageStorage + SlotTelemetrySnapshot + bind_to_slot with
// an atomic mirroring tasks_in_flight, a worker stall to widen the read window, and slots
// that all hash to one contended ring so bind_to_slot takes the Rebound path each time.
// Under TSan, a broken gate (the msg thread resetting the snapshot while the worker reads
// it) surfaces as a data race on `snap`; as written it must be clean.
TEST(StoreTelemetry, WorkerPublishRebindGateNoRaceUnderTSan)
{
    constexpr int NUM_ITERS   = 4000;
    constexpr int STALL_SPINS = 128;   // widen the worker's read window over the gate

    nv::FapiSlotMessageStorage store(/*capacity_per_type=*/3);
    nv::SlotTelemetrySnapshot   snap{};
    std::atomic<int32_t>        tasks_in_flight{0};   // 0 = ring idle (rebind allowed)
    std::atomic<bool>           producer_done{false};
    int64_t                     worker_last_seen = 0; // read after join (synchronized)

    // Msg thread: rebind the ring to the next slot + reset the snapshot (mirrors
    // bind_ring_slot), but only while the worker is not publishing (the gate), then arm.
    std::thread msg_thread([&] {
        for (int i = 1; i <= NUM_ITERS; ++i)
        {
            // Gate: never rebind/reset the snapshot while the worker holds ownership.
            while (tasks_in_flight.load(std::memory_order_acquire) != 0)
            {
                std::this_thread::yield();
            }
            // Distinct slot each iteration, all on the same ring -> Rebound path.
            const uint32_t slot_u32 = static_cast<uint32_t>(i) * nv::SLOT_STORAGE_DEPTH;
            if (store.bind_to_slot(slot_u32)
                != nv::FapiSlotMessageStorage::BindOutcome::AlreadyBound)
            {
                snap = nv::SlotTelemetrySnapshot{};
            }
            snap.l2a_start_tick = std::chrono::nanoseconds{i};   // an ingestion-time write
            // Arm: release-store so the worker's snapshot read happens-after this write.
            tasks_in_flight.store(1, std::memory_order_release);
        }
        producer_done.store(true, std::memory_order_release);
    });

    // Worker thread: publish = value-copy the snapshot while holding ownership, stall to
    // widen the window, then release the ring so the msg thread may rebind.
    std::thread worker_thread([&] {
        for (;;)
        {
            if (tasks_in_flight.load(std::memory_order_acquire) == 0)
            {
                if (producer_done.load(std::memory_order_acquire)) { break; }
                std::this_thread::yield();
                continue;
            }
            // Race-relevant access: value-copy like publish_slot_command's
            // `snap = slot_telemetry_[ring_idx]`.
            const nv::SlotTelemetrySnapshot local = snap;
            worker_last_seen = local.l2a_start_tick.count();
            for (volatile int s = 0; s < STALL_SPINS; ++s) { /* stall to overlap the gate */ }
            // Release ownership: the msg thread's next rebind happens-after this store.
            tasks_in_flight.store(0, std::memory_order_release);
        }
    });

    msg_thread.join();
    worker_thread.join();

    // The primary check is TSan cleanliness (no data race on `snap`); this also confirms
    // the worker observed a real stamped value, never a torn/default snapshot mid-rebind.
    EXPECT_GT(worker_last_seen, 0);
}
