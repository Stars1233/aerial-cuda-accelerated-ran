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
constexpr int LOS_MATRIX_SIZE = 7;  // 7x7 matrix for LOS cases
constexpr int NLOS_MATRIX_SIZE = 6; // 6x6 matrix for NLOS cases
constexpr int O2I_MATRIX_SIZE = 6;  // 6x6 matrix for O2I cases

// 3GPP maximum antenna element gain (dBi); used for antenna pattern (antTheta/antPhi) in small-scale
// and for pathloss+antenna-gain aggregation in sls_chan
constexpr float SLS_ANTENNA_GAIN_MAX_DBI = 8.0f;

// Parameter indices for correlation matrices
constexpr int SF_IDX = 0;   // Shadow Fading
constexpr int K_IDX = 1;    // K-factor
constexpr int DS_IDX = 2;   // Delay Spread
constexpr int ASD_IDX = 3;  // Azimuth Spread of Departure
constexpr int ASA_IDX = 4;  // Azimuth Spread of Arrival
constexpr int ZSD_IDX = 5;  // Zenith Spread of Departure
constexpr int ZSA_IDX = 6;  // Zenith Spread of Arrival

// UMa LOS correlation matrix (7x7)
// Order: SF (Shadow Fading), K (K-factor), DS (Delay Spread), ASD (Azimuth Spread of Departure),
//        ASA (Azimuth Spread of Arrival), ZSD (Zenith Spread of Departure), ZSA (Zenith Spread of Arrival)
const float corrMatUmaLos[7][7] = {
    //   SF,     K,     DS,    ASD,    ASA,    ZSD,    ZSA
    { 1.0f,   0.0f,  -0.4f,  -0.5f,  -0.5f,   0.0f,  -0.8f},  // SF
    { 0.0f,   1.0f,  -0.4f,   0.0f,  -0.2f,   0.0f,   0.0f},  // K
    {-0.4f,  -0.4f,   1.0f,   0.4f,   0.8f,  -0.2f,   0.0f},  // DS
    {-0.5f,   0.0f,   0.4f,   1.0f,   0.0f,   0.5f,   0.0f},  // ASD
    {-0.5f,  -0.2f,   0.8f,   0.0f,   1.0f,  -0.3f,   0.4f},  // ASA
    { 0.0f,   0.0f,  -0.2f,   0.5f,  -0.3f,   1.0f,   0.0f},  // ZSD
    {-0.8f,   0.0f,   0.0f,   0.0f,   0.4f,   0.0f,   1.0f}   // ZSA
};

// UMa LOS square root correlation matrix (7x7) based on Cholesky decomposition (lower triangular)
const float sqrtCorrMatUmaLos[7][7] = {
    //   SF,     K,     DS,    ASD,    ASA,    ZSD,    ZSA
    { 1.0000f,  0.0000f,  0.0000f,  0.0000f,  0.0000f,  0.0000f,  0.0000f},  // SF
    { 0.0000f,  1.0000f,  0.0000f,  0.0000f,  0.0000f,  0.0000f,  0.0000f},  // K
    {-0.4000f, -0.4000f,  0.8246f,  0.0000f,  0.0000f,  0.0000f,  0.0000f},  // DS
    {-0.5000f,  0.0000f,  0.2425f,  0.8314f,  0.0000f,  0.0000f,  0.0000f},  // ASD
    {-0.5000f, -0.2000f,  0.6306f, -0.4847f,  0.2783f,  0.0000f,  0.0000f},  // ASA
    { 0.0000f,  0.0000f, -0.2425f,  0.6722f,  0.6422f,  0.2774f,  0.0000f},  // ZSD
    {-0.8000f,  0.0000f, -0.3881f, -0.3679f,  0.2385f, -0.0000f,  0.1309f}   // ZSA
};

// UMa NLOS correlation matrix (6x6)
// Order: SF (Shadow Fading), DS (Delay Spread), ASD (Azimuth Spread of Departure),
//        ASA (Azimuth Spread of Arrival), ZSD (Zenith Spread of Departure), ZSA (Zenith Spread of Arrival)
const float corrMatUmaNlos[6][6] = {
    //   SF,     DS,    ASD,    ASA,    ZSD,    ZSA
    { 1.0f,  -0.4f,  -0.6f,   0.0f,   0.0f,  -0.4f},  // SF
    {-0.4f,   1.0f,   0.4f,   0.6f,  -0.5f,   0.0f},  // DS
    {-0.6f,   0.4f,   1.0f,   0.4f,   0.5f,  -0.1f},  // ASD
    { 0.0f,   0.6f,   0.4f,   1.0f,   0.0f,   0.0f},  // ASA
    { 0.0f,  -0.5f,   0.5f,   0.0f,   1.0f,   0.0f},  // ZSD
    {-0.4f,   0.0f,  -0.1f,   0.0f,   0.0f,   1.0f}   // ZSA
};

// UMa NLOS square root correlation matrix (6x6) based on Cholesky decomposition (lower triangular)
const float sqrtCorrMatUmaNlos[6][6] = {
    //   SF,     DS,    ASD,    ASA,    ZSD,    ZSA
    { 1.0000f,  0.0000f,  0.0000f,  0.0000f,  0.0000f,  0.0000f},  // SF
    {-0.4000f,  0.9165f,  0.0000f,  0.0000f,  0.0000f,  0.0000f},  // DS
    {-0.6000f,  0.1746f,  0.7807f,  0.0000f,  0.0000f,  0.0000f},  // ASD
    { 0.0000f,  0.6547f,  0.3660f,  0.6614f,  0.0000f,  0.0000f},  // ASA
    { 0.0000f, -0.5455f,  0.7624f,  0.1181f,  0.3273f,  0.0000f},  // ZSD
    {-0.4000f, -0.1746f, -0.3965f,  0.3921f,  0.4910f,  0.5074f}   // ZSA
};

// UMa O2I correlation matrix (6x6)
// Order: SF (Shadow Fading), DS (Delay Spread), ASD (Azimuth Spread of Departure),
//        ASA (Azimuth Spread of Arrival), ZSD (Zenith Spread of Departure), ZSA (Zenith Spread of Arrival)
const float corrMatUmaO2i[6][6] = {
    //   SF,     DS,    ASD,    ASA,    ZSD,    ZSA
    { 1.0f,  -0.5f,   0.2f,   0.0f,   0.0f,   0.0f},  // SF
    {-0.5f,   1.0f,   0.4f,   0.4f,  -0.6f,  -0.2f},  // DS
    { 0.2f,   0.4f,   1.0f,   0.0f,  -0.2f,   0.0f},  // ASD
    { 0.0f,   0.4f,   0.0f,   1.0f,   0.0f,   0.5f},  // ASA
    { 0.0f,  -0.6f,  -0.2f,   0.0f,   1.0f,   0.5f},  // ZSD
    { 0.0f,  -0.2f,   0.0f,   0.5f,   0.5f,   1.0f}   // ZSA
};

// UMa O2I square root correlation matrix (6x6) based on Cholesky decomposition (lower triangular)
const float sqrtCorrMatUmaO2i[6][6] = {
    //   SF,     DS,    ASD,    ASA,    ZSD,    ZSA
    { 1.0000f,  0.0000f,  0.0000f,  0.0000f,  0.0000f,  0.0000f},  // SF
    {-0.5000f,  0.8660f,  0.0000f,  0.0000f,  0.0000f,  0.0000f},  // DS
    { 0.2000f,  0.5774f,  0.7916f,  0.0000f,  0.0000f,  0.0000f},  // ASD
    { 0.0000f,  0.4619f, -0.3369f,  0.8205f,  0.0000f,  0.0000f},  // ASA
    { 0.0000f, -0.6928f,  0.2526f,  0.4937f,  0.4609f,  0.0000f},  // ZSD
    { 0.0000f, -0.2309f,  0.1684f,  0.8086f, -0.2208f,  0.4645f}   // ZSA
};

// UMi LOS correlation matrix (7x7)
// Order: SF (Shadow Fading), K (K-factor), DS (Delay Spread), ASD (Azimuth Spread of Departure),
//        ASA (Azimuth Spread of Arrival), ZSD (Zenith Spread of Departure), ZSA (Zenith Spread of Arrival)
const float corrMatUmiLos[7][7] = {
    //   SF,     K,     DS,    ASD,    ASA,    ZSD,    ZSA
    { 1.0f,   0.5f,  -0.4f,  -0.5f,  -0.4f,   0.0f,   0.0f},  // SF
    { 0.5f,   1.0f,  -0.7f,  -0.2f,  -0.3f,   0.0f,   0.0f},  // K
    {-0.4f,  -0.7f,   1.0f,   0.5f,   0.8f,   0.0f,   0.2f},  // DS
    {-0.5f,  -0.2f,   0.5f,   1.0f,   0.4f,   0.5f,   0.3f},  // ASD
    {-0.4f,  -0.3f,   0.8f,   0.4f,   1.0f,   0.0f,   0.0f},  // ASA
    { 0.0f,   0.0f,   0.0f,   0.5f,   0.0f,   1.0f,   0.0f},  // ZSD
    { 0.0f,   0.0f,   0.2f,   0.3f,   0.0f,   0.0f,   1.0f}   // ZSA
};

// UMi LOS square root correlation matrix (7x7) based on Cholesky decomposition (lower triangular)
const float sqrtCorrMatUmiLos[7][7] = {
    //   SF,     K,     DS,    ASD,    ASA,    ZSD,    ZSA
    { 1.0000f,  0.0000f,  0.0000f,  0.0000f,  0.0000f,  0.0000f,  0.0000f},  // SF
    { 0.5000f,  0.8660f,  0.0000f,  0.0000f,  0.0000f,  0.0000f,  0.0000f},  // K
    {-0.4000f, -0.5774f,  0.7118f,  0.0000f,  0.0000f,  0.0000f,  0.0000f},  // DS
    {-0.5000f,  0.0577f,  0.4683f,  0.7262f,  0.0000f,  0.0000f,  0.0000f},  // ASD
    {-0.4000f, -0.1155f,  0.8055f, -0.2348f,  0.3504f,  0.0000f,  0.0000f},  // ASA
    { 0.0000f,  0.0000f,  0.0000f,  0.6885f,  0.4615f,  0.5595f,  0.0000f},  // ZSD
    { 0.0000f,  0.0000f,  0.2810f,  0.2319f, -0.4905f,  0.1192f,  0.7826f}   // ZSA
};

// UMi NLOS correlation matrix (6x6)
// Order: SF (Shadow Fading), DS (Delay Spread), ASD (Azimuth Spread of Departure),
//        ASA (Azimuth Spread of Arrival), ZSD (Zenith Spread of Departure), ZSA (Zenith Spread of Arrival)
const float corrMatUmiNlos[6][6] = {
    //   SF,     DS,    ASD,    ASA,    ZSD,    ZSA
    { 1.0f,  -0.7f,   0.0f,  -0.4f,   0.0f,   0.0f},  // SF
    {-0.7f,   1.0f,   0.0f,   0.4f,  -0.5f,   0.0f},  // DS
    { 0.0f,   0.0f,   1.0f,   0.0f,   0.5f,   0.5f},  // ASD
    {-0.4f,   0.4f,   0.0f,   1.0f,   0.0f,   0.2f},  // ASA
    { 0.0f,  -0.5f,   0.5f,   0.0f,   1.0f,   0.0f},  // ZSD
    { 0.0f,   0.0f,   0.5f,   0.2f,   0.0f,   1.0f}   // ZSA
};

// UMi NLOS square root correlation matrix (6x6) based on Cholesky decomposition (lower triangular)
const float sqrtCorrMatUmiNlos[6][6] = {
    //   SF,     DS,    ASD,    ASA,    ZSD,    ZSA
    { 1.0000f,  0.0000f,  0.0000f,  0.0000f,  0.0000f,  0.0000f},  // SF
    {-0.7000f,  0.7141f,  0.0000f,  0.0000f,  0.0000f,  0.0000f},  // DS
    { 0.0000f,  0.0000f,  1.0000f,  0.0000f,  0.0000f,  0.0000f},  // ASD
    {-0.4000f,  0.1680f,  0.0000f,  0.9010f,  0.0000f,  0.0000f},  // ASA
    { 0.0000f, -0.7001f,  0.5000f,  0.1306f,  0.4927f,  0.0000f},  // ZSD
    { 0.0000f,  0.0000f,  0.5000f,  0.2220f, -0.5662f,  0.6165f}   // ZSA
};

// UMi O2I correlation matrix (6x6)
// Order: SF (Shadow Fading), DS (Delay Spread), ASD (Azimuth Spread of Departure),
//        ASA (Azimuth Spread of Arrival), ZSD (Zenith Spread of Departure), ZSA (Zenith Spread of Arrival)
const float corrMatUmiO2i[6][6] = {
    //   SF,     DS,    ASD,    ASA,    ZSD,    ZSA
    { 1.0f,  -0.5f,   0.2f,   0.0f,   0.0f,   0.0f},  // SF
    {-0.5f,   1.0f,   0.4f,   0.4f,  -0.6f,  -0.2f},  // DS
    { 0.2f,   0.4f,   1.0f,   0.0f,  -0.2f,   0.0f},  // ASD
    { 0.0f,   0.4f,   0.0f,   1.0f,   0.0f,   0.5f},  // ASA
    { 0.0f,  -0.6f,  -0.2f,   0.0f,   1.0f,   0.5f},  // ZSD
    { 0.0f,  -0.2f,   0.0f,   0.5f,   0.5f,   1.0f}   // ZSA
};

// UMi O2I square root correlation matrix (6x6) based on Cholesky decomposition (lower triangular)
const float sqrtCorrMatUmiO2i[6][6] = {
    //   SF,     DS,    ASD,    ASA,    ZSD,    ZSA
    { 1.0000f,  0.0000f,  0.0000f,  0.0000f,  0.0000f,  0.0000f},  // SF
    {-0.5000f,  0.8660f,  0.0000f,  0.0000f,  0.0000f,  0.0000f},  // DS
    { 0.2000f,  0.5774f,  0.7916f,  0.0000f,  0.0000f,  0.0000f},  // ASD
    { 0.0000f,  0.4619f, -0.3369f,  0.8205f,  0.0000f,  0.0000f},  // ASA
    { 0.0000f, -0.6928f,  0.2526f,  0.4937f,  0.4609f,  0.0000f},  // ZSD
    { 0.0000f, -0.2309f,  0.1684f,  0.8086f, -0.2208f,  0.4645f}   // ZSA
};

// RMa LOS correlation matrix (7x7)
// Order: SF (Shadow Fading), K (K-factor), DS (Delay Spread), ASD (Azimuth Spread of Departure),
//        ASA (Azimuth Spread of Arrival), ZSD (Zenith Spread of Departure), ZSA (Zenith Spread of Arrival)
const float corrMatRmaLos[7][7] = {
    //   SF,     K,     DS,    ASD,    ASA,    ZSD,    ZSA
    { 1.0f,   0.0f,  -0.5f,   0.0f,   0.0f,  0.01f, -0.17f},  // SF
    { 0.0f,   1.0f,   0.0f,   0.0f,   0.0f,   0.0f, -0.02f},  // K
    {-0.5f,   0.0f,   1.0f,   0.0f,   0.0f, -0.05f,  0.27f},  // DS
    { 0.0f,   0.0f,   0.0f,   1.0f,   0.0f,  0.73f, -0.14f},  // ASD
    { 0.0f,   0.0f,   0.0f,   0.0f,   1.0f,  -0.2f,  0.24f},  // ASA
    { 0.01f,  0.0f, -0.05f,  0.73f,  -0.2f,   1.0f, -0.07f},  // ZSD
    {-0.17f, -0.02f, 0.27f, -0.14f,  0.24f, -0.07f,   1.0f}   // ZSA
};

// RMa LOS square root correlation matrix (7x7) based on Cholesky decomposition (lower triangular)
const float sqrtCorrMatRmaLos[7][7] = {
    //   SF,     K,     DS,    ASD,    ASA,    ZSD,    ZSA
    { 1.0000f,  0.0000f,  0.0000f,  0.0000f,  0.0000f,  0.0000f,  0.0000f},  // SF
    { 0.0000f,  1.0000f,  0.0000f,  0.0000f,  0.0000f,  0.0000f,  0.0000f},  // K
    {-0.5000f,  0.0000f,  0.8660f,  0.0000f,  0.0000f,  0.0000f,  0.0000f},  // DS
    { 0.0000f,  0.0000f,  0.0000f,  1.0000f,  0.0000f,  0.0000f,  0.0000f},  // ASD
    { 0.0000f,  0.0000f,  0.0000f,  0.0000f,  1.0000f,  0.0000f,  0.0000f},  // ASA
    { 0.0100f,  0.0000f, -0.0520f,  0.7300f, -0.2000f,  0.6514f,  0.0000f},  // ZSD
    {-0.1700f, -0.0200f,  0.2136f, -0.1400f,  0.2400f,  0.1428f,  0.9097f}   // ZSA
};

// RMa NLOS correlation matrix (6x6)
// Order: SF (Shadow Fading), DS (Delay Spread), ASD (Azimuth Spread of Departure),
//        ASA (Azimuth Spread of Arrival), ZSD (Zenith Spread of Departure), ZSA (Zenith Spread of Arrival)
const float corrMatRmaNlos[6][6] = {
    //   SF,     DS,    ASD,    ASA,    ZSD,    ZSA
    { 1.0f,  -0.5f,   0.6f,   0.0f, -0.04f, -0.25f},  // SF
    {-0.5f,   1.0f,  -0.4f,   0.0f,  -0.1f,  -0.4f},   // DS
    { 0.6f,  -0.4f,   1.0f,   0.0f,  0.42f, -0.27f},  // ASD
    { 0.0f,   0.0f,   0.0f,   1.0f, -0.18f,  0.26f},  // ASA
    {-0.04f, -0.1f,  0.42f, -0.18f,   1.0f, -0.27f},  // ZSD
    {-0.25f, -0.4f, -0.27f,  0.26f, -0.27f,   1.0f}    // ZSA
};

// RMa NLOS square root correlation matrix (6x6) based on Cholesky decomposition (lower triangular)
const float sqrtCorrMatRmaNlos[6][6] = {
    //   SF,     DS,    ASD,    ASA,    ZSD,    ZSA
    { 1.0000f,  0.0000f,  0.0000f,  0.0000f,  0.0000f,  0.0000f},  // SF
    {-0.5000f,  0.8660f,  0.0000f,  0.0000f,  0.0000f,  0.0000f},  // DS
    { 0.6000f, -0.1155f,  0.7916f,  0.0000f,  0.0000f,  0.0000f},  // ASD
    { 0.0000f,  0.0000f,  0.0000f,  1.0000f,  0.0000f,  0.0000f},  // ASA
    {-0.0400f, -0.1386f,  0.5407f, -0.1800f,  0.8090f,  0.0000f},  // ZSD
    {-0.2500f, -0.6062f, -0.2400f,  0.2600f, -0.2317f,  0.6254f}   // ZSA
};


// RMa O2I correlation matrix (6x6)
// Order: SF (Shadow Fading), DS (Delay Spread), ASD (Azimuth Spread of Departure),
//        ASA (Azimuth Spread of Arrival), ZSD (Zenith Spread of Departure), ZSA (Zenith Spread of Arrival)
const float corrMatRmaO2i[6][6] = {
    //   SF,     DS,    ASD,    ASA,    ZSD,    ZSA
    { 1.0f,   0.0f,   0.0f,   0.0f,   0.0f,   0.0f},  // SF
    { 0.0f,   1.0f,   0.0f,   0.0f,   0.0f,   0.0f},  // DS
    { 0.0f,   0.0f,   1.0f,  -0.7f,  0.66f,  0.47f},  // ASD
    { 0.0f,   0.0f,  -0.7f,   1.0f, -0.55f, -0.22f},  // ASA
    { 0.0f,   0.0f,  0.66f, -0.55f,   1.0f,   0.0f},  // ZSD
    { 0.0f,   0.0f,  0.47f, -0.22f,   0.0f,   1.0f}   // ZSA
};

// RMa O2I square root correlation matrix (6x6) based on Cholesky decomposition (lower triangular)
const float sqrtCorrMatRmaO2i[6][6] = {
    //   SF,     DS,    ASD,    ASA,    ZSD,    ZSA
    { 1.0000f,  0.0000f,  0.0000f,  0.0000f,  0.0000f,  0.0000f},  // SF
    { 0.0000f,  1.0000f,  0.0000f,  0.0000f,  0.0000f,  0.0000f},  // DS
    { 0.0000f,  0.0000f,  1.0000f,  0.0000f,  0.0000f,  0.0000f},  // ASD
    { 0.0000f,  0.0000f, -0.7000f,  0.7141f,  0.0000f,  0.0000f},  // ASA
    { 0.0000f,  0.0000f,  0.6600f, -0.1232f,  0.7411f,  0.0000f},  // ZSD
    { 0.0000f,  0.0000f,  0.4700f,  0.1526f, -0.3932f,  0.7754f}   // ZSA
};

// Correlation distances for different scenarios
// Correlation distances for different LSPs (in meters)
struct corrDist_t {
    float SF;   // Shadow Fading
    float K;    // K-factor
    float DS;   // Delay Spread
    float ASD;  // Azimuth Spread of Departure
    float ASA;  // Azimuth Spread of Arrival
    float ZSD;  // Zenith Spread of Departure
    float ZSA;  // Zenith Spread of Arrival
    float DT;   // Delta Tau (excess delay) per 3GPP TR 38.901 Table 7.6.9-1
};

// UMa correlation distances
const corrDist_t corrDistUmaLos = {
    37.0f,  // SF (LOS)
    12.0f,  // K (LOS)
    30.0f,  // DS (LOS)
    18.0f,  // ASD (LOS)
    15.0f,  // ASA (LOS)
    15.0f,  // ZSD (LOS)
    15.0f,  // ZSA (LOS)
    50.0f   // DT (Delta Tau) per 3GPP TR 38.901 Table 7.6.9-1 UMa
};

const corrDist_t corrDistUmaNlos = {
    50.0f,  // SF (NLOS)
    0.0f,   // K (NLOS) - not applicable for NLOS
    40.0f,  // DS (NLOS)
    50.0f,  // ASD (NLOS)
    50.0f,  // ASA (NLOS)
    50.0f,  // ZSD (NLOS)
    50.0f,  // ZSA (NLOS)
    50.0f   // DT (Delta Tau) per 3GPP TR 38.901 Table 7.6.9-1 UMa
};

const corrDist_t corrDistUmaO2i = {
    7.0f,   // SF (O2I)
    0.0f,   // K (O2I) - not applicable for O2I
    10.0f,  // DS (O2I)
    11.0f,  // ASD (O2I)
    17.0f,  // ASA (O2I)
    25.0f,  // ZSD (O2I)
    25.0f,  // ZSA (O2I)
    10.0f   // DT (Delta Tau) per 3GPP TR 38.901 Table 7.6.9-1 InH (indoor)
};

// UMi correlation distances
const corrDist_t corrDistUmiLos = {
    10.0f,  // SF (LOS)
    15.0f,  // K (LOS)
    7.0f,   // DS (LOS)
    8.0f,   // ASD (LOS)
    8.0f,   // ASA (LOS)
    12.0f,  // ZSD (LOS)
    12.0f,  // ZSA (LOS)
    15.0f   // DT (Delta Tau) per 3GPP TR 38.901 Table 7.6.9-1 UMi
};

const corrDist_t corrDistUmiNlos = {
    13.0f,  // SF (NLOS)
    0.0f,   // K (NLOS) - not applicable for NLOS
    10.0f,  // DS (NLOS)
    10.0f,  // ASD (NLOS)
    9.0f,   // ASA (NLOS)
    10.0f,  // ZSD (NLOS)
    10.0f,  // ZSA (NLOS)
    15.0f   // DT (Delta Tau) per 3GPP TR 38.901 Table 7.6.9-1 UMi
};

const corrDist_t corrDistUmiO2i = {
    7.0f,   // SF (O2I)
    0.0f,   // K (O2I) - not applicable for O2I
    10.0f,  // DS (O2I)
    11.0f,  // ASD (O2I)
    17.0f,  // ASA (O2I)
    25.0f,  // ZSD (O2I)
    25.0f,  // ZSA (O2I)
    10.0f   // DT (Delta Tau) per 3GPP TR 38.901 Table 7.6.9-1 InH (indoor)
};

// RMa correlation distances
const corrDist_t corrDistRmaLos = {
    37.0f,  // SF (LOS)
    40.0f,  // K (LOS)
    50.0f,  // DS (LOS)
    25.0f,  // ASD (LOS)
    35.0f,  // ASA (LOS)
    15.0f,  // ZSD (LOS)
    15.0f,  // ZSA (LOS)
    50.0f   // DT (Delta Tau) per 3GPP TR 38.901 Table 7.6.9-1 RMa
};

const corrDist_t corrDistRmaNlos = {
    120.0f, // SF (NLOS)
    0.0f,   // K (NLOS) - not applicable for NLOS
    36.0f,  // DS (NLOS)
    30.0f,  // ASD (NLOS)
    40.0f,  // ASA (NLOS)
    50.0f,  // ZSD (NLOS)
    50.0f,  // ZSA (NLOS)
    50.0f   // DT (Delta Tau) per 3GPP TR 38.901 Table 7.6.9-1 RMa
};

// RMa O2I correlation distances
const corrDist_t corrDistRmaO2i = {
    120.0f, // SF (O2I)
    0.0f,   // K (O2I) - not applicable for O2I
    36.0f,  // DS (O2I)
    30.0f,  // ASD (O2I)
    40.0f,  // ASA (O2I)
    50.0f,  // ZSD (O2I)
    50.0f,  // ZSA (O2I)
    50.0f   // DT (Delta Tau) per 3GPP TR 38.901 Table 7.6.9-1 RMa/indoor
};

// Table 7.5-2: Scaling factors for AOA, AOD generation (C_phi^NLOS)
constexpr int nScalingFactorsAoaAod = 12;
constexpr int clusterCountsAoaAod[nScalingFactorsAoaAod] = {4, 5, 8, 10, 11, 12, 14, 15, 16, 19, 20, 25};
constexpr float scalingFactorsAoaAod[nScalingFactorsAoaAod] = {
    0.779f, 0.860f, 1.018f, 1.090f, 1.123f, 1.146f, 
    1.190f, 1.211f, 1.226f, 1.273f, 1.289f, 1.358f
};

// Table 7.5-4: Scaling factors for ZOA, ZOD generation (C_theta^NLOS)
constexpr int nScalingFactorsZoaZod = 8;
constexpr int clusterCountsZoaZod[nScalingFactorsZoaZod] = {8, 10, 11, 12, 15, 19, 20, 25};
constexpr float scalingFactorsZoaZod[nScalingFactorsZoaZod] = {
    0.889f, 0.957f, 1.031f, 1.104f, 1.1088f, 1.184f, 1.178f, 1.282f
};

constexpr int nSubCluster = 3;
constexpr int raysInSubClusterSizes[nSubCluster] = {10, 6, 4};
// original 1-indexing in the Table 7.5-5
// constexpr uint16_t raysInSubCluster0[10] = {1, 2, 3, 4, 5, 6, 7, 8, 19, 20};
// constexpr uint16_t raysInSubCluster1[6]  = {9, 10, 11, 12, 17, 18};
// constexpr uint16_t raysInSubCluster2[4]  = {13, 14, 15, 16};
// 0-indexing in the code
constexpr uint16_t raysInSubCluster0[10] = {0, 1, 2, 3, 4, 5, 6, 7, 18, 19};
constexpr uint16_t raysInSubCluster1[6]  = {8, 9, 10, 11, 16, 17};
constexpr uint16_t raysInSubCluster2[4]  = {12, 13, 14, 15};

#endif // SLS_TABLE_H
