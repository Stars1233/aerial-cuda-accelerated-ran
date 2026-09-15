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

#ifndef CUPHY_CP_TESTS_PUSCH_PARSER_DRIVERS_HPP_
#define CUPHY_CP_TESTS_PUSCH_PARSER_DRIVERS_HPP_

/**
 * @file pusch_parser_drivers.hpp
 * @brief Shared (GTest-free) drivers for the legacy PUSCH populator path.
 *
 * Used by both the functional equivalence test and the benchmark so the two
 * front-ends drive the legacy path through one definition.
 */

#include <cstdint>

#include "ul_module_view_mock.hpp"  // MockCellView::kBandwidthPrb

#include "aerial/casts/casts.hpp"   // aerial::casts::assume_cast
#include "scf_5g_fapi.h"
#include "scf_5g_slot_commands.hpp"
#include "scf_5g_fapi_pusch_pdu_parser.hpp"

#include "nv_fapi_pdu_utils.hpp"     // nv::for_each_pdu
#include "nv_phy_driver_proxy.hpp"   // nv::PHYDriverProxy singleton (legacy populator dependency)
#include "nvlog.hpp"                 // NVLOG_TAG_BASE_L2_ADAPTER

namespace cuphy_cp::tests
{

/// Log tag shared by the PUSCH equivalence test/bench PDU walkers.
inline constexpr int kPuschParserTestTag = NVLOG_TAG_BASE_L2_ADAPTER + 14;

/**
 * @brief Lazily install the @c nv::PHYDriverProxy singleton the legacy populator
 *        depends on, once per process.
 *
 * @c update_cell_command resolves the per-cell M-plane config through the global
 * @c nv::PHYDriverProxy. The full L1 application creates that singleton during
 * bring-up; this standalone harness has no driver, so it installs the proxy's
 * "standalone" form (@c make() with @c driver_ == nullptr) and sets every cell's
 * RU type to @c OTHER_MODE to mirror @c MockUlModuleView::ru -- so the legacy and
 * new parser paths observe identical RU inputs. Without this, @c getInstance()
 * dereferences a null @c unique_ptr and the populator segfaults.
 *
 * Idempotent and thread-safe: the setup body runs exactly once via a
 * function-local static, so it is cheap to call on every legacy invocation
 * (including inside the benchmark's timed loop, where it costs one guard check).
 */
inline void ensure_legacy_phydriver_proxy()
{
    static const bool initialized = []() {
        nv::PHYDriverProxy::make();  // standalone proxy: driver_ == nullptr, zeroed M-plane config
        for (auto& mplane_cfg : nv::PHYDriverProxy::getInstance().getMPlaneConfigList())
        {
            mplane_cfg.ru = OTHER_MODE;
        }
        return true;
    }();
    (void)initialized;
}

/**
 * @brief Drive the legacy populator for every PUSCH PDU in @p req.
 * @param[in]  req        UL_TTI.request carrying >=1 PUSCH PDU.
 * @param[in]  cell_index Carrier/cell index passed to the populator.
 * @param[out] out        Cell-group command receiving the populated pusch_params.
 * @param[out] out_cell   Cell sub-command scratch required by the populator.
 *
 * Inputs mirror @c MockUlModuleView defaults so the legacy and new paths see
 * identical arguments.
 */
inline void run_legacy_pusch(const scf_fapi_ul_tti_req_t&          req,
                             int32_t                               cell_index,
                             slot_command_api::cell_group_command& out,
                             slot_command_api::cell_sub_command&   out_cell)
{
    ensure_legacy_phydriver_proxy();

    slot_command_api::slot_indication slot_ind{req.sfn, req.slot, 0};
    nv::for_each_pdu<kPuschParserTestTag>(static_cast<uint32_t>(cell_index),
        aerial::casts::assume_cast<scf_fapi_generic_pdu_info_t>(req.payload),
        req.num_pdus,
        [&](uint16_t, const scf_fapi_generic_pdu_info_t& pdu) {
            if (pdu.pdu_type != UL_TTI_PDU_TYPE_PUSCH) { return; }
            const auto& pusch = *aerial::casts::assume_cast<scf_fapi_pusch_pdu_t>(pdu.pdu_config);
            scf_5g_fapi::update_cell_command(
                &out, out_cell, pusch, cell_index, slot_ind,
                /*staticPuschSlotNum=*/-1, /*lbrm=*/0u, /*bf_enabled=*/false,
                /*cell_stat_prm_idx=*/0u, /*dtx_threshold=*/0.0f,
                /*bfwCoeff_mem_info=*/nullptr, /*mmimo_enabled=*/false,
                /*slot_detail=*/nullptr, /*ul_bandwidth=*/MockCellView::kBandwidthPrb,
                /*num_ul_ant=*/MockCellView::kDefaultAntCount);
        });
}

} // namespace cuphy_cp::tests

#endif // CUPHY_CP_TESTS_PUSCH_PARSER_DRIVERS_HPP_
