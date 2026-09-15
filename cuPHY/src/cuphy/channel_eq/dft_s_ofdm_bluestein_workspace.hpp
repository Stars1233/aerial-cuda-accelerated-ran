/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#if !defined(DFT_S_OFDM_BLUESTEIN_WORKSPACE_HPP_INCLUDED_)
#define DFT_S_OFDM_BLUESTEIN_WORKSPACE_HPP_INCLUDED_

#include "cuphy.h"
#include "tensor_desc.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>

#if defined(__CUDACC__)
#define CUPHY_DFT_S_OFDM_HD __host__ __device__
#else
#define CUPHY_DFT_S_OFDM_HD
#endif

namespace cuphy
{

inline constexpr uint32_t CUPHY_PUSCH_DFT_S_OFDM_NUM_BLUESTEIN_ROWS = 53U;
inline constexpr uint16_t CUPHY_PUSCH_DFT_S_OFDM_MAX_DFT_SIZE        = 3240U; // 270 PRB * 12 tones/PRB
inline constexpr uint32_t CUPHY_PUSCH_DFT_S_OFDM_NPRB_TO_ROW_SIZE    = 271U; // indices [0, 270]

// 3GPP transform-precoding DFT sizes (index == locBluesteinWorkspace). Single source of truth.
inline constexpr std::array<uint16_t, CUPHY_PUSCH_DFT_S_OFDM_NUM_BLUESTEIN_ROWS> CUPHY_PUSCH_DFT_S_OFDM_DFT_SIZES = {
    // FFT128
    12, 24, 36, 48, 60,
    // FFT256
    72, 96, 108, 120,
    // FFT512
    144, 180, 192, 216, 240,
    // FFT1024
    288, 300, 324, 360, 384, 432, 480,
    // FFT2048
    540, 576, 600, 648, 720, 768, 864, 900, 960, 972,
    // FFT4096
    1080, 1152, 1200, 1296, 1440, 1500, 1536, 1620, 1728, 1800, 1920, 1944,
    // FFT8192
    2160, 2304, 2400, 2592, 2700, 2880, 2916, 3000, 3072, 3240};

// Invalid Bluestein workspace row (nPrb is not a supported transform-precoding size).
inline constexpr uint8_t CUPHY_PUSCH_DFT_S_OFDM_INVALID_ROW = 0xFFU;

// Bluestein workspace-init FFT widths (index 0..N-1 == FFT128 .. FFT8192).
inline constexpr std::array<uint32_t, 7> CUPHY_PUSCH_DFT_S_OFDM_BLUESTEIN_FFT_WIDTHS = {
    FFT128, FFT256, FFT512, FFT1024, FFT2048, FFT4096, FFT8192};
inline constexpr uint32_t CUPHY_PUSCH_DFT_S_OFDM_NUM_BLUESTEIN_FFT_WIDTHS =
    static_cast<uint32_t>(CUPHY_PUSCH_DFT_S_OFDM_BLUESTEIN_FFT_WIDTHS.size());
inline constexpr int CUPHY_PUSCH_DFT_S_OFDM_MAX_FFT_KERNEL_INDEX =
    static_cast<int>(CUPHY_PUSCH_DFT_S_OFDM_NUM_BLUESTEIN_FFT_WIDTHS) - 1;

namespace detail
{

// Build nPrb -> row from DFT_SIZES so the reverse map cannot drift from the size list.
constexpr std::array<uint8_t, CUPHY_PUSCH_DFT_S_OFDM_NPRB_TO_ROW_SIZE> makePuschDftSOfdmNprbToRow()
{
    std::array<uint8_t, CUPHY_PUSCH_DFT_S_OFDM_NPRB_TO_ROW_SIZE> table{};
    for(uint32_t i = 0; i < CUPHY_PUSCH_DFT_S_OFDM_NPRB_TO_ROW_SIZE; ++i)
    {
        table[i] = CUPHY_PUSCH_DFT_S_OFDM_INVALID_ROW;
    }
    for(uint32_t row = 0; row < CUPHY_PUSCH_DFT_S_OFDM_NUM_BLUESTEIN_ROWS; ++row)
    {
        const uint16_t dftSize = CUPHY_PUSCH_DFT_S_OFDM_DFT_SIZES[row];
        const uint16_t nPrb    = static_cast<uint16_t>(dftSize / CUPHY_N_TONES_PER_PRB);
        table[nPrb]            = static_cast<uint8_t>(row);
    }
    return table;
}

constexpr bool validatePuschDftSOfdmDftSizes()
{
    for(uint32_t row = 0; row < CUPHY_PUSCH_DFT_S_OFDM_NUM_BLUESTEIN_ROWS; ++row)
    {
        const uint16_t dftSize = CUPHY_PUSCH_DFT_S_OFDM_DFT_SIZES[row];
        if((dftSize % CUPHY_N_TONES_PER_PRB) != 0U)
        {
            return false;
        }
        if(row > 0U && dftSize <= CUPHY_PUSCH_DFT_S_OFDM_DFT_SIZES[row - 1U])
        {
            return false;
        }
        const uint16_t nPrb = static_cast<uint16_t>(dftSize / CUPHY_N_TONES_PER_PRB);
        if(nPrb >= CUPHY_PUSCH_DFT_S_OFDM_NPRB_TO_ROW_SIZE)
        {
            return false;
        }
    }
    // Reverse map must be unique (no two DFT sizes share an nPrb).
    auto seen = makePuschDftSOfdmNprbToRow();
    uint32_t mapped = 0U;
    for(uint32_t nPrb = 0; nPrb < CUPHY_PUSCH_DFT_S_OFDM_NPRB_TO_ROW_SIZE; ++nPrb)
    {
        if(seen[nPrb] != CUPHY_PUSCH_DFT_S_OFDM_INVALID_ROW)
        {
            ++mapped;
        }
    }
    return mapped == CUPHY_PUSCH_DFT_S_OFDM_NUM_BLUESTEIN_ROWS;
}

static_assert(validatePuschDftSOfdmDftSizes(),
              "CUPHY_PUSCH_DFT_S_OFDM_DFT_SIZES must be strictly increasing, 12*nPrb, and unique");

inline constexpr std::array<uint8_t, CUPHY_PUSCH_DFT_S_OFDM_NPRB_TO_ROW_SIZE> CUPHY_PUSCH_DFT_S_OFDM_NPRB_TO_ROW_ARR =
    makePuschDftSOfdmNprbToRow();

} // namespace detail

/**
 * Host-facing pointer to the generated nPrb -> Bluestein workspace row map.
 *
 * Do not duplicate this table in .cu sources; seed device tables from
 * getDftSOfdmNprbToRowTable() instead.
 */
inline constexpr uint8_t const* CUPHY_PUSCH_DFT_S_OFDM_NPRB_TO_ROW = detail::CUPHY_PUSCH_DFT_S_OFDM_NPRB_TO_ROW_ARR.data();

/**
 * Contiguous host nPrb -> row table used to seed device __constant__ memory.
 *
 * @return Const reference to the full nPrb-indexed row map
 *         (unsupported nPrb entries are CUPHY_PUSCH_DFT_S_OFDM_INVALID_ROW).
 */
inline constexpr std::array<uint8_t, CUPHY_PUSCH_DFT_S_OFDM_NPRB_TO_ROW_SIZE> const& getDftSOfdmNprbToRowTable()
{
    return detail::CUPHY_PUSCH_DFT_S_OFDM_NPRB_TO_ROW_ARR;
}

/**
 * Map an uplink PRB count to its Bluestein workspace row index.
 *
 * Host-safe lookup. Unsupported nPrb values return CUPHY_PUSCH_DFT_S_OFDM_INVALID_ROW (0xFF).
 *
 * @param[in] nPrb Number of PRBs for the UE group / DFT size / 12.
 * @return Bluestein workspace row in [0, CUPHY_PUSCH_DFT_S_OFDM_NUM_BLUESTEIN_ROWS),
 *         or CUPHY_PUSCH_DFT_S_OFDM_INVALID_ROW if unsupported.
 */
inline constexpr uint8_t getDftSOfdmBluesteinRowForNprb(uint16_t nPrb)
{
    return (nPrb < CUPHY_PUSCH_DFT_S_OFDM_NPRB_TO_ROW_SIZE)
               ? detail::CUPHY_PUSCH_DFT_S_OFDM_NPRB_TO_ROW_ARR[nPrb]
               : CUPHY_PUSCH_DFT_S_OFDM_INVALID_ROW;
}

struct DftSOfdmBluesteinWorkspaceDims
{
    uint32_t numRows;
    uint32_t fftWidth;
};

inline uint32_t normalizePuschMaxPrbForDftSOfdm(uint32_t nMaxPrb)
{
    if(nMaxPrb == 0U)
    {
        return MAX_N_PRBS_SUPPORTED;
    }
    return nMaxPrb;
}

inline uint32_t getDftSOfdmBluesteinNumRowsForMaxPrb(uint32_t nMaxPrb)
{
    const uint32_t maxPrb  = normalizePuschMaxPrbForDftSOfdm(nMaxPrb);
    const uint32_t maxDft  = std::min<uint32_t>(CUPHY_N_TONES_PER_PRB * maxPrb, CUPHY_PUSCH_DFT_S_OFDM_MAX_DFT_SIZE);
    uint32_t       numRows = 0U;
    for(uint32_t i = 0; i < CUPHY_PUSCH_DFT_S_OFDM_NUM_BLUESTEIN_ROWS; ++i)
    {
        if(CUPHY_PUSCH_DFT_S_OFDM_DFT_SIZES[i] <= maxDft)
        {
            numRows = i + 1U;
        }
        else
        {
            break;
        }
    }
    return std::max(numRows, 1U);
}

/**
 * Smallest power-of-two Bluestein FFT width that covers workspace rows [0, numRows).
 *
 * Used as the second tensor dimension when sizing the Bluestein chirp tables.
 *
 * @param[in] numRows Number of Bluestein workspace rows to cover.
 * @return FFT width in {FFT128, FFT256, FFT512, FFT1024, FFT2048, FFT4096, FFT8192}.
 */
CUPHY_DFT_S_OFDM_HD inline constexpr uint32_t getDftSOfdmBluesteinFftWidthForNumRows(uint32_t numRows)
{
    if(numRows <= 5U)
    {
        return FFT128;
    }
    if(numRows <= 9U)
    {
        return FFT256;
    }
    if(numRows <= 14U)
    {
        return FFT512;
    }
    if(numRows <= 21U)
    {
        return FFT1024;
    }
    if(numRows <= 31U)
    {
        return FFT2048;
    }
    if(numRows <= 43U)
    {
        return FFT4096;
    }
    return FFT8192;
}

/**
 * Bluestein FFT width required by a single workspace row.
 *
 * Uses the same FFT-width buckets as workspace initialization.
 *
 * @param[in] row Bluestein workspace row index.
 * @return FFT width required to process @p row.
 */
CUPHY_DFT_S_OFDM_HD inline constexpr uint32_t getDftSOfdmBluesteinFftWidthForRow(uint8_t row)
{
    return getDftSOfdmBluesteinFftWidthForNumRows(static_cast<uint32_t>(row) + 1U);
}

/**
 * Contiguous DFT_SIZES row span that uses a given Bluestein FFT width.
 *
 * Shared by bluestein_workspace_kernel launch geometry and IDFT row selection via
 * getDftSOfdmBluesteinFftWidthForRow. Do not hard-code FFT_OFFSETS elsewhere.
 */
struct DftSOfdmBluesteinRowRange
{
    uint8_t offset; //!< First DFT_SIZES row that uses the FFT width.
    uint8_t count;  //!< Number of contiguous rows that use the FFT width.
};

/**
 * Contiguous [offset, offset+count) of DFT_SIZES rows for a Bluestein FFT width.
 *
 * @param[in] blueFftSize Bluestein FFT width (FFT128 .. FFT8192).
 * @return Row range; count is 0 if @p blueFftSize is not a supported FFT width.
 */
CUPHY_DFT_S_OFDM_HD inline constexpr DftSOfdmBluesteinRowRange
getDftSOfdmBluesteinRowRangeForFftWidth(uint32_t blueFftSize)
{
    DftSOfdmBluesteinRowRange range{0U, 0U};
    bool                      inRange = false;
    for(uint32_t row = 0; row < CUPHY_PUSCH_DFT_S_OFDM_NUM_BLUESTEIN_ROWS; ++row)
    {
        if(getDftSOfdmBluesteinFftWidthForRow(static_cast<uint8_t>(row)) == blueFftSize)
        {
            if(!inRange)
            {
                range.offset = static_cast<uint8_t>(row);
                inRange      = true;
            }
            ++range.count;
        }
        else if(inRange)
        {
            break;
        }
    }
    return range;
}

namespace detail
{

// Rows that share an FFT width must form a single contiguous block (workspace launch assumes this).
constexpr bool validatePuschDftSOfdmFftWidthGroupsContiguous()
{
    for(uint32_t fftWidth = FFT128; fftWidth <= FFT8192; fftWidth <<= 1U)
    {
        bool seen  = false;
        bool left  = false;
        for(uint32_t row = 0; row < CUPHY_PUSCH_DFT_S_OFDM_NUM_BLUESTEIN_ROWS; ++row)
        {
            const bool match = getDftSOfdmBluesteinFftWidthForRow(static_cast<uint8_t>(row)) == fftWidth;
            if(match)
            {
                if(left)
                {
                    return false;
                }
                seen = true;
            }
            else if(seen)
            {
                left = true;
            }
        }
        const auto range = getDftSOfdmBluesteinRowRangeForFftWidth(fftWidth);
        if(seen && (range.count == 0U))
        {
            return false;
        }
    }
    return true;
}

static_assert(validatePuschDftSOfdmFftWidthGroupsContiguous(),
              "DFT-s-OFDM Bluestein rows must be contiguous per FFT width");

} // namespace detail

/**
 * Map nPrb and launched Bluestein FFT size to a workspace row.
 *
 * Host helper. Returns CUPHY_PUSCH_DFT_S_OFDM_INVALID_ROW if nPrb is unsupported or
 * the row's required FFT width does not match @p blueFftSize.
 *
 * @param[in] nPrb        Number of PRBs for the UE group / DFT size / 12.
 * @param[in] blueFftSize Bluestein FFT width used by the launched kernel.
 * @return Matching Bluestein workspace row, or CUPHY_PUSCH_DFT_S_OFDM_INVALID_ROW.
 */
inline constexpr uint8_t getDftSOfdmBluesteinRowForNprbAndFft(uint16_t nPrb, uint16_t blueFftSize)
{
    const uint8_t row = getDftSOfdmBluesteinRowForNprb(nPrb);
    if(row == CUPHY_PUSCH_DFT_S_OFDM_INVALID_ROW)
    {
        return CUPHY_PUSCH_DFT_S_OFDM_INVALID_ROW;
    }
    return (getDftSOfdmBluesteinFftWidthForRow(row) == blueFftSize) ? row : CUPHY_PUSCH_DFT_S_OFDM_INVALID_ROW;
}

inline DftSOfdmBluesteinWorkspaceDims getDftSOfdmBluesteinWorkspaceDims(uint32_t nMaxPrb)
{
    const uint32_t numRows = getDftSOfdmBluesteinNumRowsForMaxPrb(nMaxPrb);
    return DftSOfdmBluesteinWorkspaceDims{numRows, getDftSOfdmBluesteinFftWidthForNumRows(numRows)};
}

inline size_t getDftSOfdmBluesteinWorkspaceSizeBytes(uint32_t nMaxPrb)
{
    const auto dims = getDftSOfdmBluesteinWorkspaceDims(nMaxPrb);
    return static_cast<size_t>(dims.numRows) * dims.fftWidth * sizeof(data_type_traits<CUPHY_C_32F>::type) * 2U;
}

/**
 * Map a Bluestein workspace FFT width to workspace-init kernel index.
 *
 * Indices correspond to CUPHY_PUSCH_DFT_S_OFDM_BLUESTEIN_FFT_WIDTHS
 * (FFT128 .. FFT8192; see getDftSOfdmBluesteinRowRangeForFftWidth).
 *
 * @param[in] bluesteinWorkspaceFftWidth Provisioned Bluestein FFT width.
 * @return Kernel index in [0, CUPHY_PUSCH_DFT_S_OFDM_MAX_FFT_KERNEL_INDEX],
 *         or -1 if @p bluesteinWorkspaceFftWidth is invalid.
 */
inline constexpr int getDftSOfdmBluesteinMaxFftKernelIndex(uint32_t bluesteinWorkspaceFftWidth)
{
    for(int idx = 0; idx <= CUPHY_PUSCH_DFT_S_OFDM_MAX_FFT_KERNEL_INDEX; ++idx)
    {
        if(CUPHY_PUSCH_DFT_S_OFDM_BLUESTEIN_FFT_WIDTHS[static_cast<std::size_t>(idx)] == bluesteinWorkspaceFftWidth)
        {
            return idx;
        }
    }
    return -1;
}

} // namespace cuphy

#undef CUPHY_DFT_S_OFDM_HD

#endif // DFT_S_OFDM_BLUESTEIN_WORKSPACE_HPP_INCLUDED_
