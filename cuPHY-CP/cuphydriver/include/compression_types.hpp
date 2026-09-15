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
 * @file compression_types.hpp
 * @brief Lightweight header providing compression parameter types for CUDA kernels.
 *
 * Defines compression_params, mod_compression_params, and re-exports CleanupDlBufInfo
 * (via aerial-fh-driver/oran.hpp). This header is safe to include from CUDA (.cu) files
 * compiled with nvcc -std=c++17 because it does not include slot_command/slot_command.hpp
 * or any other C++20 header directly.
 *
 * Background: generic_cuda_kernels.cu previously included the full cuphydriver_api.hpp
 * solely to access these three types. cuphydriver_api.hpp transitively pulls in
 * slot_command, cuphy_api.h, and aerial-fh-driver/api.hpp — none of which are needed
 * by the CUDA kernel file. This lightweight header breaks that dependency.
 *
 * All existing consumers of cuphydriver_api.hpp are unaffected: cuphydriver_api.hpp
 * includes this header and continues to expose all types.
 */

#ifndef CUPHYDRIVER_COMPRESSION_TYPES_HPP__
#define CUPHYDRIVER_COMPRESSION_TYPES_HPP__

#include <cstdint>
#include "aerial-fh-driver/oran.hpp"  // API_MAX_ANTENNAS, ORAN_ALL_SYMBOLS, CleanupDlBufInfo (includes <inttypes.h> for size_t)
#include "constant.hpp"               // MAX_SECTIONS_PER_UPLANE_SYMBOL, DL_MAX_CELLS_PER_SLOT
#include <QAM_param.cuh>              // QamListParam, QamPrbParam

/// Maximum number of cells per GPU device. Mirrors DL_MAX_CELLS_PER_SLOT from cuphy.h.
inline constexpr uint32_t MAX_NUM_CELLS_PER_DEVICE = DL_MAX_CELLS_PER_SLOT;

/// Number of valid entries in aerial_fh::UserDataCompressionMethod (i.e. excluding RESERVED).
/// Duplicated here so CUDA (.cu) files can include this lightweight header without pulling in
/// aerial-fh-driver/api.hpp. A static_assert in cuphydriver_api.cpp cross-checks that this
/// constant equals static_cast<std::size_t>(UserDataCompressionMethod::RESERVED) at build time.
inline constexpr std::size_t NUM_USER_DATA_COMPRESSION_METHODS = 7;

/**
 * @brief Per-cell modulation compression configuration for GPU kernel.
 *
 * Holds scaling factors, PRB layout, and QAM parameters for each antenna,
 * symbol, and U-plane section. Passed to the GPU compression kernel via
 * compression_params::mod_compression_config[].
 */
struct mod_compression_params final {
    float2   scaling[API_MAX_ANTENNAS][ORAN_ALL_SYMBOLS][MAX_SECTIONS_PER_UPLANE_SYMBOL];          ///< Scaling factors per antenna/symbol/section
    uint16_t nprbs_per_list[API_MAX_ANTENNAS][ORAN_ALL_SYMBOLS][MAX_SECTIONS_PER_UPLANE_SYMBOL];   ///< Number of PRBs per section (range 0-273)
    uint16_t prb_start_per_list[API_MAX_ANTENNAS][ORAN_ALL_SYMBOLS][MAX_SECTIONS_PER_UPLANE_SYMBOL]; ///< Starting PRB index per section (range 0-273)
    uint8_t  num_messages_per_list[API_MAX_ANTENNAS][ORAN_ALL_SYMBOLS];                            ///< Number of sections per antenna/symbol (range 0-MAX_SECTIONS_PER_UPLANE_SYMBOL)
    QamListParam params_per_list[API_MAX_ANTENNAS][ORAN_ALL_SYMBOLS][MAX_SECTIONS_PER_UPLANE_SYMBOL];   ///< QAM modulation parameters per section (IQ width, CSF, etc.)
    QamPrbParam  prb_params_per_list[API_MAX_ANTENNAS][ORAN_ALL_SYMBOLS][MAX_SECTIONS_PER_UPLANE_SYMBOL]; ///< RE mask of PRBs per section
};

/**
 * @brief Aggregated compression parameters passed to the GPU compression kernel.
 *
 * Collects per-cell buffer pointers, PRB counts, and modulation compression configs
 * for a single slot's downlink transmission.
 */
struct compression_params final {
    uint8_t  comp_meth[MAX_NUM_CELLS_PER_DEVICE];               ///< Compression method per cell
    uint8_t  bit_width[MAX_NUM_CELLS_PER_DEVICE];               ///< Compressed bit width per cell
    uint8_t  *input_ptrs[MAX_NUM_CELLS_PER_DEVICE];             ///< Input buffer pointers per cell
    uint8_t **prb_ptrs[MAX_NUM_CELLS_PER_DEVICE];               ///< Per-PRB buffer pointers per cell
    int      num_prbs[MAX_NUM_CELLS_PER_DEVICE];                ///< Number of PRBs per cell
    float    beta[MAX_NUM_CELLS_PER_DEVICE];                    ///< Beta scaling factor per cell
    uint16_t max_num_prb_per_symbol[MAX_NUM_CELLS_PER_DEVICE];  ///< Maximum PRBs per symbol per cell
    uint8_t  num_antennas[MAX_NUM_CELLS_PER_DEVICE];            ///< Number of antennas per cell
    uint8_t  num_cells;                                          ///< Total number of active cells
    bool     gpu_comms;                                          ///< GPU direct communication enabled (true=enabled, false=via CPU)

    mod_compression_params *mod_compression_config[MAX_NUM_CELLS_PER_DEVICE];  ///< Modulation compression config per cell (null if disabled)
};

#endif // ifndef CUPHYDRIVER_COMPRESSION_TYPES_HPP__
