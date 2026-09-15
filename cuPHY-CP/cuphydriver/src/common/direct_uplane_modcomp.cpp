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

#define TAG (NVLOG_TAG_BASE_CUPHY_DRIVER + 4) // DRV.API

#include "direct_uplane_modcomp.hpp"

#include "cell.hpp"
#include "dlbuffer.hpp"
#include "nvlog.hpp"
#include "nvlog_fmt.hpp"
#include "slot_map_dl.hpp"

#include "aerial-fh-driver/api.hpp"
#include "aerial-fh-driver/oran.hpp"
#include "cuda_driver_utils/cuda_driver_utils.hpp"
#include <oran/oran_errors.hpp>

#include <QAM_param.cuh>

#include <array>
#include <cstring>
#include <span>

using ran::oran::make_error_code;
using ran::oran::OranErrc;

[[nodiscard]] std::error_code fill_mod_compression_params(
    std::span<const ran::oran::ModCompUplaneConfigEntry> entries,
    const std::uint16_t active_pdsch_antennas,
    mod_compression_params& dst)
{
    if (active_pdsch_antennas > API_MAX_ANTENNAS) {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT,
            "modcomp adapter: active_pdsch_antennas {} > API_MAX_ANTENNAS {}",
            active_pdsch_antennas, API_MAX_ANTENNAS);
        return make_error_code(OranErrc::InvalidParameter);
    }

    std::uint8_t num_messages_per_list[API_MAX_ANTENNAS][ORAN_ALL_SYMBOLS]{};

    for (std::size_t i = 0; i < entries.size(); ++i) {
        const auto& e = entries[i];
        if (e.flow_index >= active_pdsch_antennas || e.symbol_id >= ORAN_ALL_SYMBOLS) {
            NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT,
                "modcomp adapter: invalid entry {}: flow_index={} active={} symbol_id={}",
                i, e.flow_index, active_pdsch_antennas, e.symbol_id);
            return make_error_code(OranErrc::InvalidParameter);
        }

        auto& num = num_messages_per_list[e.flow_index][e.symbol_id];
        if (num >= MAX_SECTIONS_PER_UPLANE_SYMBOL) {
            NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT,
                "modcomp adapter: too many sections for flow={} symbol={} count={}",
                e.flow_index, e.symbol_id, num);
            return make_error_code(OranErrc::InvalidParameter);
        }

        const auto idx = num++;
        dst.nprbs_per_list[e.flow_index][e.symbol_id][idx]       = e.num_prbu;
        dst.prb_start_per_list[e.flow_index][e.symbol_id][idx]   = e.start_prbu;
        dst.scaling[e.flow_index][e.symbol_id][idx]              = float2{e.scaling_0, e.scaling_1};
        dst.params_per_list[e.flow_index][e.symbol_id][idx].set(
            static_cast<QamListParam::qamwidth>(e.iq_width), e.csf_0, e.csf_1);
        dst.prb_params_per_list[e.flow_index][e.symbol_id][idx].set(e.re_mask_0, e.re_mask_1);
    }

    std::memcpy(
        dst.num_messages_per_list,
        num_messages_per_list,
        sizeof(dst.num_messages_per_list));
    return {};
}

[[nodiscard]] std::error_code fill_direct_modcomp_config(
    bool dl_modcomp_enabled,
    std::span<const ran::oran::ModCompUplaneConfigEntry> entries,
    std::uint16_t active_pdsch_antennas,
    mod_compression_params* modcomp_temp)
{
    if (!dl_modcomp_enabled) {
        return {};
    }

    if (modcomp_temp == nullptr) {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT,
            "direct C-plane modcomp enabled but host config storage is missing");
        return make_error_code(OranErrc::InvalidParameter);
    }

    const auto fill_ec =
        fill_mod_compression_params(entries, active_pdsch_antennas, *modcomp_temp);
    if (fill_ec) {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT,
            "direct C-plane to U-plane modcomp adapter failed: {}", fill_ec.message());
        std::memset(
            modcomp_temp->num_messages_per_list,
            0,
            sizeof(modcomp_temp->num_messages_per_list));
        return fill_ec;
    }

    return {};
}

[[nodiscard]] std::error_code clear_direct_modcomp_config(
    bool dl_modcomp_enabled,
    mod_compression_params* modcomp_temp)
{
    if (!dl_modcomp_enabled) {
        return {};
    }
    if (modcomp_temp == nullptr) {
        return make_error_code(OranErrc::InvalidParameter);
    }

    std::memset(
        modcomp_temp->num_messages_per_list,
        0,
        sizeof(modcomp_temp->num_messages_per_list));
    return {};
}

[[nodiscard]] std::error_code issue_direct_modcomp_config_copies(
    SlotMapDl* slot_map,
    const int first_cell,
    const int num_cells,
    CUstream stream)
{
    if (slot_map == nullptr || stream == nullptr) {
        return make_error_code(OranErrc::InvalidParameter);
    }

    // cuphyBatchedMemcpyHelper still exposes cudaStream_t& (legacy cuPHY API).
    // CUstream and cudaStream_t are both CUstream_st*; keep the cast explicit.
    cudaStream_t helper_stream = static_cast<cudaStream_t>(stream);

    auto& helper = slot_map->getBatchedMemcpyHelper();
    helper.reset();

    bool has_modcomp_copy = false;
    for (int i = first_cell;
         i < first_cell + num_cells && i < slot_map->getNumCells();
         ++i) {
        Cell* cell_ptr = slot_map->aggr_cell_list[i];
        DLOutputBuffer* dlbuf = slot_map->aggr_dlbuf_list[i];
        if (cell_ptr == nullptr || dlbuf == nullptr) {
            continue;
        }
        if (cell_ptr->getDLCompMeth() !=
            static_cast<int>(aerial_fh::UserDataCompressionMethod::MODULATION_COMPRESSION)) {
            continue;
        }
        if (dlbuf->getModCompressionConfig() == nullptr ||
            dlbuf->getModCompressionTempConfig() == nullptr) {
            NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT,
                "direct C-plane modcomp H2D storage missing for cell {}", i);
            helper.reset();
            return make_error_code(OranErrc::InvalidParameter);
        }

        helper.updateMemcpy(
            dlbuf->getModCompressionConfig(),
            dlbuf->getModCompressionTempConfig(),
            sizeof(mod_compression_params),
            cudaMemcpyHostToDevice,
            helper_stream);
        has_modcomp_copy = true;
    }

    if (!has_modcomp_copy) {
        helper.reset();
        return {};
    }

    const cuphyStatus_t status = helper.launchBatchedMemcpy(helper_stream);
    if (status != CUPHY_STATUS_SUCCESS) {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT,
            "direct C-plane modcomp H2D batch failed");
        return make_error_code(OranErrc::InvalidParameter);
    }
    return {};
}

void reset_partial_uplane_preserving_modcomp_backing(PartialUplaneSlotInfo_t* out)
{
    if (out == nullptr) {
        return;
    }
    std::array<ModCompPartialSectionInfoPerMessagePerSymbol_t*, PARTIAL_UPLANE_SYMBOLS_PER_SLOT>
        saved{};
    for (std::size_t sym = 0; sym < saved.size(); ++sym) {
        saved[sym] = out->message_info[sym].mod_comp_params;
    }
    std::memset(out, 0, sizeof(*out));
    for (std::size_t sym = 0; sym < saved.size(); ++sym) {
        out->message_info[sym].mod_comp_params = saved[sym];
        if (saved[sym] != nullptr) {
            std::memset(saved[sym], 0, sizeof(*saved[sym]));
        }
    }
}
