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

#if !defined(LDPC_PARAMS_HPP_INCLUDED_)
#define LDPC_PARAMS_HPP_INCLUDED_

#include <array>
#include <cstdint>

namespace cuphy {
namespace ldpc {

// Valid lifting sizes from 3GPP TS 38.212 Table 5.3.2-1
inline constexpr std::array<uint32_t, 51> Z_TABLE{
     2,  4,  8,  16,  32,  64, 128, 256,
     3,  6, 12,  24,  48,  96, 192, 384,
     5, 10, 20,  40,  80, 160, 320,
     7, 14, 28,  56, 112, 224,
     9, 18, 36,  72, 144, 288,
    11, 22, 44,  88, 176, 352,
    13, 26, 52, 104, 208,
    15, 30, 60, 120, 240
};

/**
 * Derives the LDPC base graph from transport-block size and code rate.
 * Based on 3GPP TS 38.212 Section 6.2.2.
 *
 * @param[in] tbSize Transport-block size in bits.
 * @param[in] codeRate Target code rate.
 * @return Base graph identifier (1 or 2).
 */
[[nodiscard]] inline int derive_base_graph(uint32_t tbSize, double codeRate)
{
    if ((tbSize <= 292) || ((tbSize <= 3824) && (codeRate <= 0.67)) || (codeRate <= 0.25))
        return 2;
    else
        return 1;
}

/**
 * Derives the number of systematic columns for an LDPC codeblock.
 * Based on 3GPP TS 38.212 Section 5.2.2.
 *
 * @param[in] bg Base graph identifier.
 * @param[in] B Transport-block size including CRC bits.
 * @return Number of systematic columns.
 */
[[nodiscard]] inline int derive_Kb(int bg, uint32_t B)
{
    if (bg == 1)
        return 22;
    else if (B > 640)
        return 10;
    else if (B > 560)
        return 9;
    else if (B > 192)
        return 8;
    else
        return 6;
}

/**
 * Derives the smallest valid lifting size for an LDPC codeblock.
 * Based on 3GPP TS 38.212 Section 5.2.2.
 *
 * @param[in] Kb Number of systematic columns.
 * @param[in] K_prime Information bits per codeblock before filler bits.
 * @return Lifting size.
 */
[[nodiscard]] inline int derive_Zc(int Kb, uint32_t K_prime)
{
    uint32_t best_product = 1'000'000;
    int Zc = 0;
    for (const auto z : Z_TABLE)
    {
        uint32_t product = z * Kb;
        if ((product >= K_prime) && (product < best_product))
        {
            best_product = product;
            Zc = z;
        }
    }
    return Zc;
}

/**
 * Holds LDPC parameters derived from a transport block.
 */
struct LdpcParams {
    int      bg;      ///< Base graph identifier (1 or 2).
    int      Kb;      ///< Number of systematic columns.
    int      Zc;      ///< Lifting size.
    int      K;       ///< Systematic bits per codeblock.
    uint32_t F;       ///< Filler bits per codeblock.
    uint32_t nCb;     ///< Number of codeblocks.
};

/**
 * Derives all LDPC parameters from transport-block size and code rate.
 *
 * @param[in] tbSize Transport-block size in bits.
 * @param[in] codeRate Target code rate.
 * @return Derived base graph, segmentation, lifting, and filler parameters.
 */
[[nodiscard]] inline LdpcParams derive_ldpc_params(uint32_t tbSize, double codeRate)
{
    LdpcParams params{};

    // Derive base graph
    params.bg = derive_base_graph(tbSize, codeRate);

    // Max codeblock size
    uint32_t K_cb = (params.bg == 1) ? 8448 : 3840;

    // B = TB + CRC bits
    uint32_t B = (tbSize <= 3824) ? (tbSize + 16) : (tbSize + 24);

    // Number of codeblocks
    if (B <= K_cb)
    {
        params.nCb = 1;
    }
    else
    {
        params.nCb = (B + (K_cb - 24) - 1) / (K_cb - 24);  // div_round_up(B, K_cb - 24)
    }

    // B' = B + CB-CRCs
    uint32_t B_prime = (B <= K_cb) ? B : (B + params.nCb * 24);

    // K' = bits per codeblock before filler
    uint32_t K_prime = B_prime / params.nCb;

    // Derive Kb and Zc
    params.Kb = derive_Kb(params.bg, B);
    params.Zc = derive_Zc(params.Kb, K_prime);

    // K = systematic bits, F = filler bits
    params.K = ((params.bg == 1) ? 22 : 10) * params.Zc;
    params.F = params.K - K_prime;

    return params;
}

} // namespace ldpc
} // namespace cuphy

#endif // !defined(LDPC_PARAMS_HPP_INCLUDED_)
