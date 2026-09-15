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

#ifndef SLS_TABLE_H
#define SLS_TABLE_H

#include <cstdint>

// Matrix dimensions
inline constexpr int LOS_MATRIX_SIZE = 7;  // 7x7 matrix for LOS cases
inline constexpr int NLOS_MATRIX_SIZE = 6; // 6x6 matrix for NLOS cases
inline constexpr int O2I_MATRIX_SIZE = 6;  // 6x6 matrix for O2I cases

// Legacy TR 36.873 Table 7.4.3-3 O2I building penetration-loss model.
inline constexpr std::uint8_t LEGACY_O2I_BUILDING_PENETR_LOSS_IND = 4U;
inline constexpr float LEGACY_O2I_MAX_FREQUENCY_HZ = 6e9F;
inline constexpr float LEGACY_O2I_MAX_INDOOR_DISTANCE_M = 25.0F;

// 3GPP maximum antenna element gain (dBi); used for antenna pattern (antTheta/antPhi) in small-scale
// and for pathloss+antenna-gain aggregation in sls_chan
inline constexpr float SLS_ANTENNA_GAIN_MAX_DBI = 8.0F;
// Table 7.3-1 A_max: clamp on the COMBINED vertical+horizontal cut attenuation (dB),
// A''(theta,phi) = -min{-(A_V + A_H), A_max}
inline constexpr float SLS_ANTENNA_PATTERN_A_MAX_DB = 30.0F;

// Parameter indices for correlation matrices
inline constexpr int SF_IDX = 0;   // Shadow Fading
inline constexpr int K_IDX = 1;    // K-factor
inline constexpr int DS_IDX = 2;   // Delay Spread
inline constexpr int ASD_IDX = 3;  // Azimuth Spread of Departure
inline constexpr int ASA_IDX = 4;  // Azimuth Spread of Arrival
inline constexpr int ZSD_IDX = 5;  // Zenith Spread of Departure
inline constexpr int ZSA_IDX = 6;  // Zenith Spread of Arrival

// UMa LOS correlation matrix (7x7)
// Order: SF (Shadow Fading), K (K-factor), DS (Delay Spread), ASD (Azimuth Spread of Departure),
//        ASA (Azimuth Spread of Arrival), ZSD (Zenith Spread of Departure), ZSA (Zenith Spread of Arrival)
inline constexpr float CORR_MAT_UMA_LOS[7][7] = {
    //   SF,     K,     DS,    ASD,    ASA,    ZSD,    ZSA
    { 1.0F,   0.0F,  -0.4F,  -0.5F,  -0.5F,   0.0F,  -0.8F},  // SF
    { 0.0F,   1.0F,  -0.4F,   0.0F,  -0.2F,   0.0F,   0.0F},  // K
    {-0.4F,  -0.4F,   1.0F,   0.4F,   0.8F,  -0.2F,   0.0F},  // DS
    {-0.5F,   0.0F,   0.4F,   1.0F,   0.0F,   0.5F,   0.0F},  // ASD
    {-0.5F,  -0.2F,   0.8F,   0.0F,   1.0F,  -0.3F,   0.4F},  // ASA
    { 0.0F,   0.0F,  -0.2F,   0.5F,  -0.3F,   1.0F,   0.0F},  // ZSD
    {-0.8F,   0.0F,   0.0F,   0.0F,   0.4F,   0.0F,   1.0F}   // ZSA
};

// UMa LOS square root correlation matrix (7x7) based on Cholesky decomposition (lower triangular)
inline constexpr float SQRT_CORR_MAT_UMA_LOS[7][7] = {
    //   SF,     K,     DS,    ASD,    ASA,    ZSD,    ZSA
    { 1.0000F,  0.0000F,  0.0000F,  0.0000F,  0.0000F,  0.0000F,  0.0000F},  // SF
    { 0.0000F,  1.0000F,  0.0000F,  0.0000F,  0.0000F,  0.0000F,  0.0000F},  // K
    {-0.4000F, -0.4000F,  0.8246F,  0.0000F,  0.0000F,  0.0000F,  0.0000F},  // DS
    {-0.5000F,  0.0000F,  0.2425F,  0.8314F,  0.0000F,  0.0000F,  0.0000F},  // ASD
    {-0.5000F, -0.2000F,  0.6306F, -0.4847F,  0.2783F,  0.0000F,  0.0000F},  // ASA
    { 0.0000F,  0.0000F, -0.2425F,  0.6722F,  0.6422F,  0.2774F,  0.0000F},  // ZSD
    {-0.8000F,  0.0000F, -0.3881F, -0.3679F,  0.2385F, -0.0000F,  0.1309F}   // ZSA
};

// UMa NLOS correlation matrix (6x6)
// Order: SF (Shadow Fading), DS (Delay Spread), ASD (Azimuth Spread of Departure),
//        ASA (Azimuth Spread of Arrival), ZSD (Zenith Spread of Departure), ZSA (Zenith Spread of Arrival)
inline constexpr float CORR_MAT_UMA_NLOS[6][6] = {
    //   SF,     DS,    ASD,    ASA,    ZSD,    ZSA
    { 1.0F,  -0.4F,  -0.6F,   0.0F,   0.0F,  -0.4F},  // SF
    {-0.4F,   1.0F,   0.4F,   0.6F,  -0.5F,   0.0F},  // DS
    {-0.6F,   0.4F,   1.0F,   0.4F,   0.5F,  -0.1F},  // ASD
    { 0.0F,   0.6F,   0.4F,   1.0F,   0.0F,   0.0F},  // ASA
    { 0.0F,  -0.5F,   0.5F,   0.0F,   1.0F,   0.0F},  // ZSD
    {-0.4F,   0.0F,  -0.1F,   0.0F,   0.0F,   1.0F}   // ZSA
};

// UMa NLOS square root correlation matrix (6x6) based on Cholesky decomposition (lower triangular)
inline constexpr float SQRT_CORR_MAT_UMA_NLOS[6][6] = {
    //   SF,     DS,    ASD,    ASA,    ZSD,    ZSA
    { 1.0000F,  0.0000F,  0.0000F,  0.0000F,  0.0000F,  0.0000F},  // SF
    {-0.4000F,  0.9165F,  0.0000F,  0.0000F,  0.0000F,  0.0000F},  // DS
    {-0.6000F,  0.1746F,  0.7807F,  0.0000F,  0.0000F,  0.0000F},  // ASD
    { 0.0000F,  0.6547F,  0.3660F,  0.6614F,  0.0000F,  0.0000F},  // ASA
    { 0.0000F, -0.5455F,  0.7624F,  0.1181F,  0.3273F,  0.0000F},  // ZSD
    {-0.4000F, -0.1746F, -0.3965F,  0.3921F,  0.4910F,  0.5074F}   // ZSA
};

// UMa O2I correlation matrix (6x6)
// Order: SF (Shadow Fading), DS (Delay Spread), ASD (Azimuth Spread of Departure),
//        ASA (Azimuth Spread of Arrival), ZSD (Zenith Spread of Departure), ZSA (Zenith Spread of Arrival)
inline constexpr float CORR_MAT_UMA_O2I[6][6] = {
    //   SF,     DS,    ASD,    ASA,    ZSD,    ZSA
    { 1.0F,  -0.5F,   0.2F,   0.0F,   0.0F,   0.0F},  // SF
    {-0.5F,   1.0F,   0.4F,   0.4F,  -0.6F,  -0.2F},  // DS
    { 0.2F,   0.4F,   1.0F,   0.0F,  -0.2F,   0.0F},  // ASD
    { 0.0F,   0.4F,   0.0F,   1.0F,   0.0F,   0.5F},  // ASA
    { 0.0F,  -0.6F,  -0.2F,   0.0F,   1.0F,   0.5F},  // ZSD
    { 0.0F,  -0.2F,   0.0F,   0.5F,   0.5F,   1.0F}   // ZSA
};

// UMa O2I square root correlation matrix (6x6) based on Cholesky decomposition (lower triangular)
inline constexpr float SQRT_CORR_MAT_UMA_O2I[6][6] = {
    //   SF,     DS,    ASD,    ASA,    ZSD,    ZSA
    { 1.0000F,  0.0000F,  0.0000F,  0.0000F,  0.0000F,  0.0000F},  // SF
    {-0.5000F,  0.8660F,  0.0000F,  0.0000F,  0.0000F,  0.0000F},  // DS
    { 0.2000F,  0.5774F,  0.7916F,  0.0000F,  0.0000F,  0.0000F},  // ASD
    { 0.0000F,  0.4619F, -0.3369F,  0.8205F,  0.0000F,  0.0000F},  // ASA
    { 0.0000F, -0.6928F,  0.2526F,  0.4937F,  0.4609F,  0.0000F},  // ZSD
    { 0.0000F, -0.2309F,  0.1684F,  0.8086F, -0.2208F,  0.4645F}   // ZSA
};

// UMi LOS correlation matrix (7x7)
// Order: SF (Shadow Fading), K (K-factor), DS (Delay Spread), ASD (Azimuth Spread of Departure),
//        ASA (Azimuth Spread of Arrival), ZSD (Zenith Spread of Departure), ZSA (Zenith Spread of Arrival)
inline constexpr float CORR_MAT_UMI_LOS[7][7] = {
    //   SF,     K,     DS,    ASD,    ASA,    ZSD,    ZSA
    { 1.0F,   0.5F,  -0.4F,  -0.5F,  -0.4F,   0.0F,   0.0F},  // SF
    { 0.5F,   1.0F,  -0.7F,  -0.2F,  -0.3F,   0.0F,   0.0F},  // K
    {-0.4F,  -0.7F,   1.0F,   0.5F,   0.8F,   0.0F,   0.2F},  // DS
    {-0.5F,  -0.2F,   0.5F,   1.0F,   0.4F,   0.5F,   0.3F},  // ASD
    {-0.4F,  -0.3F,   0.8F,   0.4F,   1.0F,   0.0F,   0.0F},  // ASA
    { 0.0F,   0.0F,   0.0F,   0.5F,   0.0F,   1.0F,   0.0F},  // ZSD
    { 0.0F,   0.0F,   0.2F,   0.3F,   0.0F,   0.0F,   1.0F}   // ZSA
};

// UMi LOS square root correlation matrix (7x7) based on Cholesky decomposition (lower triangular)
inline constexpr float SQRT_CORR_MAT_UMI_LOS[7][7] = {
    //   SF,     K,     DS,    ASD,    ASA,    ZSD,    ZSA
    { 1.0000F,  0.0000F,  0.0000F,  0.0000F,  0.0000F,  0.0000F,  0.0000F},  // SF
    { 0.5000F,  0.8660F,  0.0000F,  0.0000F,  0.0000F,  0.0000F,  0.0000F},  // K
    {-0.4000F, -0.5774F,  0.7118F,  0.0000F,  0.0000F,  0.0000F,  0.0000F},  // DS
    {-0.5000F,  0.0577F,  0.4683F,  0.7262F,  0.0000F,  0.0000F,  0.0000F},  // ASD
    {-0.4000F, -0.1155F,  0.8055F, -0.2348F,  0.3504F,  0.0000F,  0.0000F},  // ASA
    { 0.0000F,  0.0000F,  0.0000F,  0.6885F,  0.4615F,  0.5595F,  0.0000F},  // ZSD
    { 0.0000F,  0.0000F,  0.2810F,  0.2319F, -0.4905F,  0.1192F,  0.7826F}   // ZSA
};

// UMi NLOS correlation matrix (6x6)
// Order: SF (Shadow Fading), DS (Delay Spread), ASD (Azimuth Spread of Departure),
//        ASA (Azimuth Spread of Arrival), ZSD (Zenith Spread of Departure), ZSA (Zenith Spread of Arrival)
inline constexpr float CORR_MAT_UMI_NLOS[6][6] = {
    //   SF,     DS,    ASD,    ASA,    ZSD,    ZSA
    { 1.0F,  -0.7F,   0.0F,  -0.4F,   0.0F,   0.0F},  // SF
    {-0.7F,   1.0F,   0.0F,   0.4F,  -0.5F,   0.0F},  // DS
    { 0.0F,   0.0F,   1.0F,   0.0F,   0.5F,   0.5F},  // ASD
    {-0.4F,   0.4F,   0.0F,   1.0F,   0.0F,   0.2F},  // ASA
    { 0.0F,  -0.5F,   0.5F,   0.0F,   1.0F,   0.0F},  // ZSD
    { 0.0F,   0.0F,   0.5F,   0.2F,   0.0F,   1.0F}   // ZSA
};

// UMi NLOS square root correlation matrix (6x6) based on Cholesky decomposition (lower triangular)
inline constexpr float SQRT_CORR_MAT_UMI_NLOS[6][6] = {
    //   SF,     DS,    ASD,    ASA,    ZSD,    ZSA
    { 1.0000F,  0.0000F,  0.0000F,  0.0000F,  0.0000F,  0.0000F},  // SF
    {-0.7000F,  0.7141F,  0.0000F,  0.0000F,  0.0000F,  0.0000F},  // DS
    { 0.0000F,  0.0000F,  1.0000F,  0.0000F,  0.0000F,  0.0000F},  // ASD
    {-0.4000F,  0.1680F,  0.0000F,  0.9010F,  0.0000F,  0.0000F},  // ASA
    { 0.0000F, -0.7001F,  0.5000F,  0.1306F,  0.4927F,  0.0000F},  // ZSD
    { 0.0000F,  0.0000F,  0.5000F,  0.2220F, -0.5662F,  0.6165F}   // ZSA
};

// UMi O2I correlation matrix (6x6)
// Order: SF (Shadow Fading), DS (Delay Spread), ASD (Azimuth Spread of Departure),
//        ASA (Azimuth Spread of Arrival), ZSD (Zenith Spread of Departure), ZSA (Zenith Spread of Arrival)
inline constexpr float CORR_MAT_UMI_O2I[6][6] = {
    //   SF,     DS,    ASD,    ASA,    ZSD,    ZSA
    { 1.0F,  -0.5F,   0.2F,   0.0F,   0.0F,   0.0F},  // SF
    {-0.5F,   1.0F,   0.4F,   0.4F,  -0.6F,  -0.2F},  // DS
    { 0.2F,   0.4F,   1.0F,   0.0F,  -0.2F,   0.0F},  // ASD
    { 0.0F,   0.4F,   0.0F,   1.0F,   0.0F,   0.5F},  // ASA
    { 0.0F,  -0.6F,  -0.2F,   0.0F,   1.0F,   0.5F},  // ZSD
    { 0.0F,  -0.2F,   0.0F,   0.5F,   0.5F,   1.0F}   // ZSA
};

// UMi O2I square root correlation matrix (6x6) based on Cholesky decomposition (lower triangular)
inline constexpr float SQRT_CORR_MAT_UMI_O2I[6][6] = {
    //   SF,     DS,    ASD,    ASA,    ZSD,    ZSA
    { 1.0000F,  0.0000F,  0.0000F,  0.0000F,  0.0000F,  0.0000F},  // SF
    {-0.5000F,  0.8660F,  0.0000F,  0.0000F,  0.0000F,  0.0000F},  // DS
    { 0.2000F,  0.5774F,  0.7916F,  0.0000F,  0.0000F,  0.0000F},  // ASD
    { 0.0000F,  0.4619F, -0.3369F,  0.8205F,  0.0000F,  0.0000F},  // ASA
    { 0.0000F, -0.6928F,  0.2526F,  0.4937F,  0.4609F,  0.0000F},  // ZSD
    { 0.0000F, -0.2309F,  0.1684F,  0.8086F, -0.2208F,  0.4645F}   // ZSA
};

// RMa LOS correlation matrix (7x7)
// Order: SF (Shadow Fading), K (K-factor), DS (Delay Spread), ASD (Azimuth Spread of Departure),
//        ASA (Azimuth Spread of Arrival), ZSD (Zenith Spread of Departure), ZSA (Zenith Spread of Arrival)
inline constexpr float CORR_MAT_RMA_LOS[7][7] = {
    //   SF,     K,     DS,    ASD,    ASA,    ZSD,    ZSA
    { 1.0F,   0.0F,  -0.5F,   0.0F,   0.0F,  0.01F, -0.17F},  // SF
    { 0.0F,   1.0F,   0.0F,   0.0F,   0.0F,   0.0F, -0.02F},  // K
    {-0.5F,   0.0F,   1.0F,   0.0F,   0.0F, -0.05F,  0.27F},  // DS
    { 0.0F,   0.0F,   0.0F,   1.0F,   0.0F,  0.73F, -0.14F},  // ASD
    { 0.0F,   0.0F,   0.0F,   0.0F,   1.0F,  -0.2F,  0.24F},  // ASA
    { 0.01F,  0.0F, -0.05F,  0.73F,  -0.2F,   1.0F, -0.07F},  // ZSD
    {-0.17F, -0.02F, 0.27F, -0.14F,  0.24F, -0.07F,   1.0F}   // ZSA
};

// RMa LOS square root correlation matrix (7x7) based on Cholesky decomposition (lower triangular)
inline constexpr float SQRT_CORR_MAT_RMA_LOS[7][7] = {
    //   SF,     K,     DS,    ASD,    ASA,    ZSD,    ZSA
    { 1.0000F,  0.0000F,  0.0000F,  0.0000F,  0.0000F,  0.0000F,  0.0000F},  // SF
    { 0.0000F,  1.0000F,  0.0000F,  0.0000F,  0.0000F,  0.0000F,  0.0000F},  // K
    {-0.5000F,  0.0000F,  0.8660F,  0.0000F,  0.0000F,  0.0000F,  0.0000F},  // DS
    { 0.0000F,  0.0000F,  0.0000F,  1.0000F,  0.0000F,  0.0000F,  0.0000F},  // ASD
    { 0.0000F,  0.0000F,  0.0000F,  0.0000F,  1.0000F,  0.0000F,  0.0000F},  // ASA
    { 0.0100F,  0.0000F, -0.0520F,  0.7300F, -0.2000F,  0.6514F,  0.0000F},  // ZSD
    {-0.1700F, -0.0200F,  0.2136F, -0.1400F,  0.2400F,  0.1428F,  0.9097F}   // ZSA
};

// RMa NLOS correlation matrix (6x6)
// Order: SF (Shadow Fading), DS (Delay Spread), ASD (Azimuth Spread of Departure),
//        ASA (Azimuth Spread of Arrival), ZSD (Zenith Spread of Departure), ZSA (Zenith Spread of Arrival)
inline constexpr float CORR_MAT_RMA_NLOS[6][6] = {
    //   SF,     DS,    ASD,    ASA,    ZSD,    ZSA
    { 1.0F,  -0.5F,   0.6F,   0.0F, -0.04F, -0.25F},  // SF
    {-0.5F,   1.0F,  -0.4F,   0.0F,  -0.1F,  -0.4F},   // DS
    { 0.6F,  -0.4F,   1.0F,   0.0F,  0.42F, -0.27F},  // ASD
    { 0.0F,   0.0F,   0.0F,   1.0F, -0.18F,  0.26F},  // ASA
    {-0.04F, -0.1F,  0.42F, -0.18F,   1.0F, -0.27F},  // ZSD
    {-0.25F, -0.4F, -0.27F,  0.26F, -0.27F,   1.0F}    // ZSA
};

// RMa NLOS square root correlation matrix (6x6) based on Cholesky decomposition (lower triangular)
inline constexpr float SQRT_CORR_MAT_RMA_NLOS[6][6] = {
    //   SF,     DS,    ASD,    ASA,    ZSD,    ZSA
    { 1.0000F,  0.0000F,  0.0000F,  0.0000F,  0.0000F,  0.0000F},  // SF
    {-0.5000F,  0.8660F,  0.0000F,  0.0000F,  0.0000F,  0.0000F},  // DS
    { 0.6000F, -0.1155F,  0.7916F,  0.0000F,  0.0000F,  0.0000F},  // ASD
    { 0.0000F,  0.0000F,  0.0000F,  1.0000F,  0.0000F,  0.0000F},  // ASA
    {-0.0400F, -0.1386F,  0.5407F, -0.1800F,  0.8090F,  0.0000F},  // ZSD
    {-0.2500F, -0.6062F, -0.2400F,  0.2600F, -0.2317F,  0.6254F}   // ZSA
};


// RMa O2I correlation matrix (6x6)
// Order: SF (Shadow Fading), DS (Delay Spread), ASD (Azimuth Spread of Departure),
//        ASA (Azimuth Spread of Arrival), ZSD (Zenith Spread of Departure), ZSA (Zenith Spread of Arrival)
inline constexpr float CORR_MAT_RMA_O2I[6][6] = {
    //   SF,     DS,    ASD,    ASA,    ZSD,    ZSA
    { 1.0F,   0.0F,   0.0F,   0.0F,   0.0F,   0.0F},  // SF
    { 0.0F,   1.0F,   0.0F,   0.0F,   0.0F,   0.0F},  // DS
    { 0.0F,   0.0F,   1.0F,  -0.7F,  0.66F,  0.47F},  // ASD
    { 0.0F,   0.0F,  -0.7F,   1.0F, -0.55F, -0.22F},  // ASA
    { 0.0F,   0.0F,  0.66F, -0.55F,   1.0F,   0.0F},  // ZSD
    { 0.0F,   0.0F,  0.47F, -0.22F,   0.0F,   1.0F}   // ZSA
};

// RMa O2I square root correlation matrix (6x6) based on Cholesky decomposition (lower triangular)
inline constexpr float SQRT_CORR_MAT_RMA_O2I[6][6] = {
    //   SF,     DS,    ASD,    ASA,    ZSD,    ZSA
    { 1.0000F,  0.0000F,  0.0000F,  0.0000F,  0.0000F,  0.0000F},  // SF
    { 0.0000F,  1.0000F,  0.0000F,  0.0000F,  0.0000F,  0.0000F},  // DS
    { 0.0000F,  0.0000F,  1.0000F,  0.0000F,  0.0000F,  0.0000F},  // ASD
    { 0.0000F,  0.0000F, -0.7000F,  0.7141F,  0.0000F,  0.0000F},  // ASA
    { 0.0000F,  0.0000F,  0.6600F, -0.1232F,  0.7411F,  0.0000F},  // ZSD
    { 0.0000F,  0.0000F,  0.4700F,  0.1526F, -0.3932F,  0.7754F}   // ZSA
};

/** Correlation distances for large-scale parameters, in meters. */
struct CorrDist {
    float sf;   //!< Shadow Fading.
    float k;    //!< K-factor.
    float ds;   //!< Delay Spread.
    float asd;  //!< Azimuth Spread of Departure.
    float asa;  //!< Azimuth Spread of Arrival.
    float zsd;  //!< Zenith Spread of Departure.
    float zsa;  //!< Zenith Spread of Arrival.
    float dt;   //!< Delta Tau (excess delay) per 3GPP TR 38.901 Table 7.6.9-1.
};

// UMa correlation distances
inline constexpr CorrDist CORR_DIST_UMA_LOS = {
    37.0F,  // SF (LOS)
    12.0F,  // K (LOS)
    30.0F,  // DS (LOS)
    18.0F,  // ASD (LOS)
    15.0F,  // ASA (LOS)
    15.0F,  // ZSD (LOS)
    15.0F,  // ZSA (LOS)
    50.0F   // DT (Delta Tau) per 3GPP TR 38.901 Table 7.6.9-1 UMa
};

inline constexpr CorrDist CORR_DIST_UMA_NLOS = {
    50.0F,  // SF (NLOS)
    0.0F,   // K (NLOS) - not applicable for NLOS
    40.0F,  // DS (NLOS)
    50.0F,  // ASD (NLOS)
    50.0F,  // ASA (NLOS)
    50.0F,  // ZSD (NLOS)
    50.0F,  // ZSA (NLOS)
    50.0F   // DT (Delta Tau) per 3GPP TR 38.901 Table 7.6.9-1 UMa
};

inline constexpr CorrDist CORR_DIST_UMA_O2I = {
    7.0F,   // SF (O2I)
    0.0F,   // K (O2I) - not applicable for O2I
    10.0F,  // DS (O2I)
    11.0F,  // ASD (O2I)
    17.0F,  // ASA (O2I)
    25.0F,  // ZSD (O2I)
    25.0F,  // ZSA (O2I)
    10.0F   // DT (Delta Tau) per 3GPP TR 38.901 Table 7.6.9-1 InH (indoor)
};

// UMi correlation distances
inline constexpr CorrDist CORR_DIST_UMI_LOS = {
    10.0F,  // SF (LOS)
    15.0F,  // K (LOS)
    7.0F,   // DS (LOS)
    8.0F,   // ASD (LOS)
    8.0F,   // ASA (LOS)
    12.0F,  // ZSD (LOS)
    12.0F,  // ZSA (LOS)
    15.0F   // DT (Delta Tau) per 3GPP TR 38.901 Table 7.6.9-1 UMi
};

inline constexpr CorrDist CORR_DIST_UMI_NLOS = {
    13.0F,  // SF (NLOS)
    0.0F,   // K (NLOS) - not applicable for NLOS
    10.0F,  // DS (NLOS)
    10.0F,  // ASD (NLOS)
    9.0F,   // ASA (NLOS)
    10.0F,  // ZSD (NLOS)
    10.0F,  // ZSA (NLOS)
    15.0F   // DT (Delta Tau) per 3GPP TR 38.901 Table 7.6.9-1 UMi
};

inline constexpr CorrDist CORR_DIST_UMI_O2I = {
    7.0F,   // SF (O2I)
    0.0F,   // K (O2I) - not applicable for O2I
    10.0F,  // DS (O2I)
    11.0F,  // ASD (O2I)
    17.0F,  // ASA (O2I)
    25.0F,  // ZSD (O2I)
    25.0F,  // ZSA (O2I)
    10.0F   // DT (Delta Tau) per 3GPP TR 38.901 Table 7.6.9-1 InH (indoor)
};

// RMa correlation distances
inline constexpr CorrDist CORR_DIST_RMA_LOS = {
    37.0F,  // SF (LOS)
    40.0F,  // K (LOS)
    50.0F,  // DS (LOS)
    25.0F,  // ASD (LOS)
    35.0F,  // ASA (LOS)
    15.0F,  // ZSD (LOS)
    15.0F,  // ZSA (LOS)
    50.0F   // DT (Delta Tau) per 3GPP TR 38.901 Table 7.6.9-1 RMa
};

inline constexpr CorrDist CORR_DIST_RMA_NLOS = {
    120.0F, // SF (NLOS)
    0.0F,   // K (NLOS) - not applicable for NLOS
    36.0F,  // DS (NLOS)
    30.0F,  // ASD (NLOS)
    40.0F,  // ASA (NLOS)
    50.0F,  // ZSD (NLOS)
    50.0F,  // ZSA (NLOS)
    50.0F   // DT (Delta Tau) per 3GPP TR 38.901 Table 7.6.9-1 RMa
};

// RMa O2I correlation distances
inline constexpr CorrDist CORR_DIST_RMA_O2I = {
    120.0F, // SF (O2I)
    0.0F,   // K (O2I) - not applicable for O2I
    36.0F,  // DS (O2I)
    30.0F,  // ASD (O2I)
    40.0F,  // ASA (O2I)
    50.0F,  // ZSD (O2I)
    50.0F,  // ZSA (O2I)
    50.0F   // DT (Delta Tau) per 3GPP TR 38.901 Table 7.6.9-1 RMa/indoor
};

// Table 7.5-2: Scaling factors for AOA, AOD generation (C_phi^NLOS)
inline constexpr int N_SCALING_FACTORS_AOA_AOD = 12;
inline constexpr int CLUSTER_COUNTS_AOA_AOD[N_SCALING_FACTORS_AOA_AOD] = {4, 5, 8, 10, 11, 12, 14, 15, 16, 19, 20, 25};
inline constexpr float SCALING_FACTORS_AOA_AOD[N_SCALING_FACTORS_AOA_AOD] = {
    0.779F, 0.860F, 1.018F, 1.090F, 1.123F, 1.146F,
    1.190F, 1.211F, 1.226F, 1.273F, 1.289F, 1.358F
};

// Table 7.5-4: Scaling factors for ZOA, ZOD generation (C_theta^NLOS)
inline constexpr int N_SCALING_FACTORS_ZOA_ZOD = 8;
inline constexpr int CLUSTER_COUNTS_ZOA_ZOD[N_SCALING_FACTORS_ZOA_ZOD] = {8, 10, 11, 12, 15, 19, 20, 25};
inline constexpr float SCALING_FACTORS_ZOA_ZOD[N_SCALING_FACTORS_ZOA_ZOD] = {
    0.889F, 0.957F, 1.031F, 1.104F, 1.1088F, 1.184F, 1.178F, 1.282F
};

inline constexpr int N_SUB_CLUSTER = 3;
inline constexpr int RAYS_IN_SUB_CLUSTER_SIZES[N_SUB_CLUSTER] = {10, 6, 4};
// original 1-indexing in the Table 7.5-5
// constexpr uint16_t RAYS_IN_SUB_CLUSTER_0[10] = {1, 2, 3, 4, 5, 6, 7, 8, 19, 20};
// constexpr uint16_t RAYS_IN_SUB_CLUSTER_1[6]  = {9, 10, 11, 12, 17, 18};
// constexpr uint16_t RAYS_IN_SUB_CLUSTER_2[4]  = {13, 14, 15, 16};
// 0-indexing in the code
inline constexpr uint16_t RAYS_IN_SUB_CLUSTER_0[10] = {0, 1, 2, 3, 4, 5, 6, 7, 18, 19};
inline constexpr uint16_t RAYS_IN_SUB_CLUSTER_1[6]  = {8, 9, 10, 11, 16, 17};
inline constexpr uint16_t RAYS_IN_SUB_CLUSTER_2[4]  = {12, 13, 14, 15};

#endif // SLS_TABLE_H
