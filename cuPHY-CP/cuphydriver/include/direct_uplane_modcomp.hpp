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

#ifndef CUPHYDRIVER_DIRECT_UPLANE_MODCOMP_HPP_
#define CUPHYDRIVER_DIRECT_UPLANE_MODCOMP_HPP_

#include "compression_types.hpp"

#include <aerial-fh-driver/partial_uplane_slot_info.hpp>
#include <oran/cplane_types.hpp>

#include <cuda.h>

#include <cstdint>
#include <system_error>
#include <span>

class SlotMapDl;

/**
 * Pack framework-emitted modcomp entries into cuphydriver mod-compression parameters.
 *
 * @param[in] entries U-plane modcomp records emitted by the framework C-plane converter.
 * @param[in] active_pdsch_antennas Number of active PDSCH antenna flows.
 * @param[out] dst Destination cuphydriver mod-compression workspace.
 * @return Empty error_code on success; non-zero when input dimensions are invalid.
 */
[[nodiscard]] std::error_code fill_mod_compression_params(
    std::span<const ran::oran::ModCompUplaneConfigEntry> entries,
    std::uint16_t active_pdsch_antennas,
    mod_compression_params& dst);

/**
 * Pack entries into pinned host modcomp workspace when direct DL modcomp is enabled.
 *
 * @param[in] dl_modcomp_enabled True when DL modulation compression is active.
 * @param[in] entries U-plane modcomp records emitted by the framework C-plane converter.
 * @param[in] active_pdsch_antennas Number of active PDSCH antenna flows.
 * @param[out] modcomp_temp Nullable pinned host workspace to populate.
 * @return Empty error_code on success; non-zero when enabled storage is missing or invalid.
 */
[[nodiscard]] std::error_code fill_direct_modcomp_config(
    bool dl_modcomp_enabled,
    std::span<const ran::oran::ModCompUplaneConfigEntry> entries,
    std::uint16_t active_pdsch_antennas,
    mod_compression_params* modcomp_temp);

/**
 * Clear per-antenna/symbol section counts on the pinned modcomp workspace.
 *
 * @param[in] dl_modcomp_enabled True when DL modulation compression is active.
 * @param[out] modcomp_temp Nullable pinned host workspace to clear.
 * @return Empty error_code on success; non-zero when enabled storage is missing.
 */
[[nodiscard]] std::error_code clear_direct_modcomp_config(
    bool dl_modcomp_enabled,
    mod_compression_params* modcomp_temp);

/**
 * H2D-copy pinned modcomp workspaces to device for all modcomp cells in the batch.
 *
 * @param[in,out] slot_map DL slot map that owns the batched memcpy helper and cell buffers.
 * @param[in] first_cell First cell index in the aggregate batch.
 * @param[in] num_cells Number of cells in the aggregate batch.
 * @param[in] stream CUDA stream used for the batched copy.
 * @return Empty error_code on success; non-zero when storage or copy launch fails.
 */
[[nodiscard]] std::error_code issue_direct_modcomp_config_copies(
    SlotMapDl* slot_map,
    int first_cell,
    int num_cells,
    CUstream stream);

/**
 * Zero PartialUplaneSlotInfo while preserving caller-owned mod_comp_params backing.
 *
 * @param[in,out] out Partial slot info to reset; null is accepted as a no-op.
 */
void reset_partial_uplane_preserving_modcomp_backing(PartialUplaneSlotInfo_t* out);

#endif // ifndef CUPHYDRIVER_DIRECT_UPLANE_MODCOMP_HPP_
