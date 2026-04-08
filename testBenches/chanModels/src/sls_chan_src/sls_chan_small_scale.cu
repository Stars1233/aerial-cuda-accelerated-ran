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

#include "sls_chan.cuh"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <curand.h>
#include <cassert>
#include <algorithm>  // For std::min
#include <cmath>      // For std::pow, std::cos, std::sin
#include <cstdint>    // For uint32_t, uint16_t, uint8_t
#include <vector>     // For std::vector
#include <random>     // For random number generation
#include <string>     // For std::string
#include <algorithm>  // For std::clamp, std::shuffle


// Helper function to calculate field components
inline void calculateFieldComponents(
    const AntPanelConfig& antConfig, float theta, float phi, float zeta, float& F_theta, float& F_phi) {
    // Convert angles to radians
    float zeta_rad = zeta * M_PI / 180.0f;

    // Wrap theta into [0, 360] then map to [0, 180] using symmetry
    int theta_idx = static_cast<int>(round(theta));
    if (theta_idx < 0 || theta_idx >= 360) {
        theta_idx = theta_idx % 360;  // First modulo
        if (theta_idx < 0) {
            theta_idx += 360;  // Only add 360 if negative, avoiding second modulo
        }
    }
    theta_idx = (theta_idx > 180) ? 360 - theta_idx : theta_idx;
    
    // Handle phi: only do modulo if outside [0, 359] range
    int phi_idx = static_cast<int>(round(phi));
    if (phi_idx < 0 || phi_idx > 359) {
        phi_idx = phi_idx % 360;  // First modulo
        if (phi_idx < 0) {
            phi_idx += 360;  // Only add 360 if negative, avoiding second modulo
        }
    }
    float A_db_3D = antConfig.antTheta[theta_idx] + antConfig.antPhi[phi_idx] + (antConfig.antModel == 1 ? SLS_ANTENNA_GAIN_MAX_DBI : 0.0f);
    float A_3D_sqrt = powf(10.0f, A_db_3D / 20.0f); // equivalent to sqrt(10^(A_db_3D/10))
    F_theta = A_3D_sqrt * cosf(zeta_rad);
    F_phi = A_3D_sqrt * sinf(zeta_rad);
}


// Inline function to generate cluster delays and powers
inline void genClusterDelayAndPower(
    float delaySpread,
    float r_tao,
    uint8_t losInd,
    uint16_t& nCluster,
    float K,
    float xi,
    uint8_t outdoor_ind,
    float* delays,           // Changed from vector to pointer
    float* powers,           // Changed from vector to pointer
    uint16_t* strongest2clustersIdx,  // Changed from vector to pointer
    std::mt19937& gen,
    std::uniform_real_distribution<float>& uniformDist,
    std::normal_distribution<float>& normalDist)
{
    // Generate initial delays using exponential distribution
    // Use epsilon to avoid log(0) which would result in -infinity
    constexpr float epsilon = 1e-10f;
    for (uint16_t clusterIdx = 0; clusterIdx < nCluster; clusterIdx++) {
        float delay = -delaySpread * r_tao * std::log(std::max(uniformDist(gen), epsilon));
        delays[clusterIdx] = delay;  // Use array indexing
    }
    
    // Sort delays
    std::sort(delays, delays + nCluster);  // Sort array range
    float minDelay = delays[0];
    if (losInd && outdoor_ind) {
        float C_tao = 0.7705f - 0.0433f * K + 0.0002f * std::pow(K, 2) + 0.000017f * std::pow(K, 3);
        for (uint16_t i = 0; i < nCluster; i++) {
            delays[i] = (delays[i] - minDelay) / C_tao;
        }
    }
    else {
        for (uint16_t i = 0; i < nCluster; i++) {
            delays[i] -= minDelay;
        }
    }
    
    // Generate cluster powers with exponential decay
    float totalPower = 0.0f;
    for (uint16_t clusterIdx = 0; clusterIdx < nCluster; clusterIdx++) {
        float power = std::exp(-delays[clusterIdx] * (r_tao - 1.0f) / (r_tao * delaySpread));
        power *= std::pow(10.0f, - (xi * normalDist(gen)) / 10.0f);  // Add some randomness
        powers[clusterIdx] = power;  // Use array indexing
        totalPower += power;
    }
    
    // Normalize powers
    for (uint16_t i = 0; i < nCluster; i++) {
        powers[i] /= totalPower;
    }
    
    // Find max power and threshold for filtering (AFTER LOS is added per 3GPP)
    float maxPower = powers[0];
    for (uint16_t i = 1; i < nCluster; i++) {
        if (powers[i] > maxPower) {
            maxPower = powers[i];
        }
    }
    float powerThreshold = maxPower * std::pow(10.0f, -25.0f / 10.0f);
    
    // Filter out weak clusters (25 dB below max power)
    // Count valid clusters and compact arrays
    uint16_t validClusterCount = 0;
    
    // in LOS case, the first cluster will be set to 0 if it's below the threshold
    // but kept in the cluster count, validClusterCount = 1 in later steps
    // in NLOS case, it's direcly removed, validClusterCount = 0 in later steps
    if (powers[0] < powerThreshold) {
        if (losInd && outdoor_ind) {
            // implicit delays[0] = delays[0];
            powers[0] = 0.0f;
            validClusterCount = 1;
        }
    }
    // other clusters are filtered normally
    for (uint16_t i = 1; i < nCluster; i++) {
        if (powers[i] >= powerThreshold) {
            if (validClusterCount != i) {
                // Compact the arrays
                delays[validClusterCount] = delays[i];
                powers[validClusterCount] = powers[i];
            }
            validClusterCount++;
        }
    }
    
    // Update nCluster to reflect valid clusters only
    nCluster = validClusterCount;
    
    // Find strongest 2 clusters after filtering
    if (nCluster >= 2) {
        uint16_t maxIdx1 = 0;
        uint16_t maxIdx2 = 1;
        
        if (powers[1] > powers[0]) {
            maxIdx1 = 1;
            maxIdx2 = 0;
        }
        
        for (uint16_t i = 2; i < nCluster; i++) {
            if (powers[i] > powers[maxIdx1]) {
                maxIdx2 = maxIdx1;
                maxIdx1 = i;
            } else if (powers[i] > powers[maxIdx2]) {
                maxIdx2 = i;
            }
        }
        
        strongest2clustersIdx[0] = maxIdx1;
        strongest2clustersIdx[1] = maxIdx2;
    } else if (nCluster == 1) {
        strongest2clustersIdx[0] = 0;
        strongest2clustersIdx[1] = 0;
    }

    // Handle LOS case - adjust first cluster power (no need to redo normlization)
    if (losInd && outdoor_ind) {
        float K_R = std::pow(10.0f, K / 10.0f);
        float P1_LOS = K_R / (K_R + 1.0f);
        float P_n_LOS = 1.0f / (K_R + 1.0f);
        
        // Scale all powers by P_n_LOS
        for (uint16_t i = 0; i < nCluster; i++) {
            powers[i] *= P_n_LOS;
        }
        
        // Add P1_LOS to first cluster
        powers[0] += P1_LOS;
    }
}


// Static member function to generate cluster angles
template <typename Tscalar, typename Tcomplex>
void slsChan<Tscalar, Tcomplex>::genClusterAngle(
    uint8_t nCluster,
    float C_ASA,
    float C_ASD,
    float C_phi_NLOS,
    float C_phi_LOS,
    float c_phi_O2I,
    float C_theta_LOS,
    float C_theta_NLOS,
    float C_theta_O2I,
    float ASA,
    float ASD,
    float ZSA,
    float ZSD,
    float phi_LOS_AOA,
    float phi_LOS_AOD,
    float theta_LOS_ZOA,
    float theta_LOS_ZOD,
    float mu_offset_ZOD,
    bool losInd,
    bool outdoor_ind,
    float K,    
    float* powers,
    float* phi_n_AoA,           // Changed from vector to pointer
    float* phi_n_AoD,           // Changed from vector to pointer
    float* theta_n_ZOD,         // Changed from vector to pointer
    float* theta_n_ZOA,         // Changed from vector to pointer
    std::mt19937& gen,
    std::uniform_real_distribution<float>& uniformDist,
    std::normal_distribution<float>& normalDist)
{    

    // calculate C_phi and C_theta
    float C_phi, C_theta;
    if (outdoor_ind == 0) { // indoor UE, O2I
        C_phi = c_phi_O2I;
        C_theta = C_theta_O2I;
    } else {
        if (losInd) { // outdoor LOS
            float scalingFactor_phi = (1.1035f - 0.028f * K - 0.002f * K * K + 0.0001f * K * K * K);
            float scalingFactor_theta = (1.3086f + 0.0339f * K - 0.0077f * K * K + 0.0002f * K * K * K);
            C_phi = C_phi_LOS * scalingFactor_phi;
            C_theta = C_theta_LOS * scalingFactor_theta;
        } else { // outdoor NLOS
            C_phi = C_phi_NLOS;
            C_theta = C_theta_NLOS;
        }
    }
    
    // find the maximum power of the clusters
    float max_p_n = *std::max_element(powers, powers + nCluster);
    
    // Safety check for max power
    if (max_p_n <= 0.0f || std::isnan(max_p_n)) {
        printf("ERROR: Invalid max_p_n detected: %f\n", max_p_n);
        return;
    }

    // Generate AOA (Azimuth of Arrival)
    for (uint16_t n = 0; n < nCluster; n++) {
        float Xn = (uniformDist(gen) < 0.5f) ? 1.0f : -1.0f;
        float Yn = ASA / 7.0f * normalDist(gen);
        
        // Safety check for power ratio to prevent log(0) and sqrt(negative)
        float power_ratio = powers[n] / max_p_n;
        if (power_ratio <= 0.0f) {
            printf("ERROR: Invalid power ratio: %f (powers[%d]=%f, max_p_n=%f)\n", 
                   power_ratio, n, powers[n], max_p_n);
            return;
        }
        power_ratio = std::min(power_ratio, 1.0f);   // Ensure ratio <= 1
        
        float log_term = -std::log(power_ratio);
        if (std::isnan(log_term) || std::isinf(log_term) || log_term < 0.0f) {
            printf("ERROR: Invalid log term: %f (power_ratio=%f)\n", log_term, power_ratio);
            return;
        }
        
        float phi_prime_AOA = 2.0f * (ASA / 1.4f) * std::sqrt(log_term) / std::max(C_phi, 1e-6f);
        phi_n_AoA[n] = Xn * phi_prime_AOA + Yn + phi_LOS_AOA;
    }
    
    if (losInd && outdoor_ind) {
        for (uint16_t n = 0; n < nCluster; n++) {
            phi_n_AoA[n] -= (phi_n_AoA[1] - phi_LOS_AOA);
        }
    }
    
    // Generate AOD (Azimuth of Departure)
    for (uint16_t n = 0; n < nCluster; n++) {
        float Xn = (uniformDist(gen) < 0.5f) ? 1.0f : -1.0f;
        float Yn = ASD / 7.0f * normalDist(gen);
        float phi_prime_AOD = 2.0f * (ASD / 1.4f) * sqrtf(-logf(powers[n] / max_p_n)) / C_phi;
        phi_n_AoD[n] = Xn * phi_prime_AOD + Yn + phi_LOS_AOD;
    }
    
    if (losInd && outdoor_ind) {
        for (uint16_t n = 0; n < nCluster; n++) {
            phi_n_AoD[n] -= (phi_n_AoD[1] - phi_LOS_AOD);
        }
    }
    
    // Generate ZOA (Zenith of Arrival)
    for (uint16_t n = 0; n < nCluster; n++) {
        float Xn = (uniformDist(gen) < 0.5f) ? 1.0f : -1.0f;
        float Yn = ZSA / 7.0f * normalDist(gen);
        float theta_prime_ZOA = -ZSA * logf(powers[n] / max_p_n) / C_theta;
        float theta_bar_ZOA = outdoor_ind ? theta_LOS_ZOA : 90.0f;
        theta_n_ZOA[n] = Xn * theta_prime_ZOA + Yn + theta_bar_ZOA;
    }
    
    if (losInd && outdoor_ind) {
        for (uint16_t n = 0; n < nCluster; n++) {
            theta_n_ZOA[n] -= (theta_n_ZOA[1] - theta_LOS_ZOA);
        }
    }
    
    // Generate ZOD (Zenith of Departure)
    for (uint16_t n = 0; n < nCluster; n++) {
        float Xn = (uniformDist(gen) < 0.5f) ? 1.0f : -1.0f;
        float Yn = ZSD / 7.0f * normalDist(gen);
        float theta_prime_ZOD = -ZSD * logf(powers[n] / max_p_n) / C_theta;
        theta_n_ZOD[n] = Xn * theta_prime_ZOD + Yn + theta_LOS_ZOD + mu_offset_ZOD;
    }
    
    if (losInd && outdoor_ind) {
        for (uint16_t n = 0; n < nCluster; n++) {
            theta_n_ZOD[n] -= (theta_n_ZOD[1] - theta_LOS_ZOD - mu_offset_ZOD);
        }
    }
}

// Static member function to generate ray angles within clusters
template <typename Tscalar, typename Tcomplex>
void slsChan<Tscalar, Tcomplex>::genRayAngle(
    uint8_t nCluster,
    uint16_t nRayPerCluster,
    const float* phi_n_AoA,
    const float* phi_n_AoD,
    const float* theta_n_ZOD,
    const float* theta_n_ZOA,
    float* phi_n_m_AoA,
    float* phi_n_m_AoD,
    float* theta_n_m_ZOD,
    float* theta_n_m_ZOA,
    float C_ASA,
    float C_ASD,
    float C_ZSA,
    float C_ZSD,
    std::mt19937& gen,
    std::uniform_real_distribution<float>& uniformDist)
{
    // Standardized ray offset angles (3GPP specifications - constant for all scenarios)
    const float rayOffsets[20] = {
        0.0447f, -0.0447f, 0.1413f, -0.1413f, 0.2492f, -0.2492f, 0.3715f, -0.3715f,
        0.5129f, -0.5129f, 0.6797f, -0.6797f, 0.8844f, -0.8844f, 1.1481f, -1.1481f,
        1.5195f, -1.5195f, 2.1551f, -2.1551f
    };
    
    // For each cluster
    for (uint8_t n = 0; n < nCluster; n++) {
        // Generate random permutations for each angle type (like MATLAB randperm)
        std::vector<uint16_t> idxASA(nRayPerCluster), idxASD(nRayPerCluster), idxZSA(nRayPerCluster), idxZSD(nRayPerCluster);
        
        // Initialize permutation arrays
        for (uint16_t i = 0; i < nRayPerCluster; i++) {
            idxASA[i] = i;
            idxASD[i] = i;
            idxZSA[i] = i;
            idxZSD[i] = i;
        }
        
        // Apply random shuffle for each angle type
        std::shuffle(idxASA.begin(), idxASA.end(), gen);
        std::shuffle(idxASD.begin(), idxASD.end(), gen);
        std::shuffle(idxZSA.begin(), idxZSA.end(), gen);
        std::shuffle(idxZSD.begin(), idxZSD.end(), gen);
        
        // For each ray in the cluster
        for (uint16_t m = 0; m < nRayPerCluster; m++) {
            uint16_t rayIdx = n * nRayPerCluster + m;
            uint8_t offsetIdx_ASA = idxASA[m];
            uint8_t offsetIdx_ASD = idxASD[m];
            uint8_t offsetIdx_ZSA = idxZSA[m];
            uint8_t offsetIdx_ZSD = idxZSD[m];
            
            // Generate AOA (Azimuth of Arrival)
            phi_n_m_AoA[rayIdx] = phi_n_AoA[n] + C_ASA * rayOffsets[offsetIdx_ASA];
            
            // Generate AOD (Azimuth of Departure)
            phi_n_m_AoD[rayIdx] = phi_n_AoD[n] + C_ASD * rayOffsets[offsetIdx_ASD];
            
            // Generate ZOA (Zenith of Arrival) with angle wrapping
            float temp_ZOA = theta_n_ZOA[n] + C_ZSA * rayOffsets[offsetIdx_ZSA];
            // Normalize to [0°, 360°) range only if needed to avoid expensive fmod
            if (temp_ZOA < 0.0f || temp_ZOA >= 360.0f) {
                temp_ZOA = std::fmod(temp_ZOA, 360.0f);
                if (temp_ZOA < 0.0f) {
                    temp_ZOA += 360.0f;
                }
            }
            // Apply zenith angle reflection for [0°, 180°] range
            theta_n_m_ZOA[rayIdx] = (temp_ZOA > 180.0f) ? 360.0f - temp_ZOA : temp_ZOA;
            
            // Generate ZOD (Zenith of Departure) with angle wrapping  
            float temp_ZOD = theta_n_ZOD[n] + C_ZSD * rayOffsets[offsetIdx_ZSD];
            // Normalize to [0°, 360°) range only if needed to avoid expensive fmod
            if (temp_ZOD < 0.0f || temp_ZOD >= 360.0f) {
                temp_ZOD = std::fmod(temp_ZOD, 360.0f);
                if (temp_ZOD < 0.0f) {
                    temp_ZOD += 360.0f;
                }
            }
            // Apply zenith angle reflection for [0°, 180°] range
            theta_n_m_ZOD[rayIdx] = (temp_ZOD > 180.0f) ? 360.0f - temp_ZOD : temp_ZOD;
        }
    }
}


// Helper function to find indices of N strongest clusters
inline std::vector<uint16_t> findStrongestClusters(const std::vector<float>& powers, uint16_t n) {
    // Create vector of indices
    std::vector<uint16_t> indices(powers.size());
    std::iota(indices.begin(), indices.end(), 0);
    
    // Sort indices based on power values in descending order
    std::sort(indices.begin(), indices.end(), 
        [&powers](uint16_t a, uint16_t b) { return powers[a] > powers[b]; });
    
    // Return first n indices
    return std::vector<uint16_t>(indices.begin(), indices.begin() + n);
}


// Helper function to calculate ray coefficient
inline cuComplex calculateRayCoefficient(
    const AntPanelConfig& utAntConfig, int ueAntIdx, float theta_ZOA, float phi_AOA, float zetaOffsetUtAnt,
    const AntPanelConfig& bsAntConfig, int bsAntIdx, float theta_ZOD, float phi_AOD, float zetaOffsetBsAnt,
    float xpr, float * randomPhase, float currentTime, float * utVelocityPolar, float lambda_0)
{
    // Compute d_bar_rx (Rx antenna element position)
    // Assume antSize = [M_g, N_g, M, N, P] and antSpacing = [d_g_h, d_g_v, d_h, d_v]
    int M = utAntConfig.antSize[2];
    int N = utAntConfig.antSize[3];
    int P = utAntConfig.antSize[4];
    // int m = (ueAntIdx / (N * P)) % M;
    // int n = (ueAntIdx / P) % N;
    int p_rx = ueAntIdx % P;
    float d_h_rx = utAntConfig.antSpacing[2];
    float d_v_rx = utAntConfig.antSpacing[3];
    float d_bar_rx[3] = { (ueAntIdx / (N * P)) % M * d_h_rx, (ueAntIdx / P) % N * d_v_rx, 0.0f };

    // Convert angles to radians
    float d2pi = M_PI / 180.0f;
    float theta_ZOA_rad = theta_ZOA * d2pi;
    float phi_AOA_rad = phi_AOA * d2pi;
    float theta_ZOD_rad = theta_ZOD * d2pi;
    float phi_AOD_rad = phi_AOD * d2pi;

    // Calculate field patterns for Rx
    float F_rx_theta, F_rx_phi, F_tx_theta, F_tx_phi;
    calculateFieldComponents(utAntConfig, theta_ZOA, phi_AOA, utAntConfig.antPolarAngles[p_rx] + zetaOffsetUtAnt, F_rx_theta, F_rx_phi);

    // Compute d_bar_tx (Tx antenna element position)
    M = bsAntConfig.antSize[2];
    N = bsAntConfig.antSize[3];
    P = bsAntConfig.antSize[4];
    // int m_tx = (bsAntIdx / (N * P)) % M;
    // int n_tx = (bsAntIdx / P) % N;
    int p_tx = bsAntIdx % P;
    float d_h_tx = bsAntConfig.antSpacing[2];
    float d_v_tx = bsAntConfig.antSpacing[3];
    float d_bar_tx[3] = { (bsAntIdx / (N * P)) % M * d_h_tx, (bsAntIdx / P) % N * d_v_tx, 0.0f };
    
    // Calculate field patterns for Tx
    calculateFieldComponents(bsAntConfig, theta_ZOD, phi_AOD, bsAntConfig.antPolarAngles[p_tx] + zetaOffsetBsAnt, F_tx_theta, F_tx_phi);

    // Term 1: Rx antenna field pattern
    cuComplex term1[2] = {make_cuComplex(F_rx_theta, 0.0f), make_cuComplex(F_rx_phi, 0.0f)};

    // Term 2: Polarization matrix
    float kappa = xpr; // Use the input xpr
    float sqrt_kappa = sqrtf(kappa);
    cuComplex term2[2][2];
    term2[0][0] = make_cuComplex(cosf(randomPhase[0]), sinf(randomPhase[0]));
    term2[0][1] = make_cuComplex(cosf(randomPhase[1]), sinf(randomPhase[1]));
    term2[0][1].x /= sqrt_kappa; term2[0][1].y /= sqrt_kappa;
    term2[1][0] = make_cuComplex(cosf(randomPhase[2]), sinf(randomPhase[2]));
    term2[1][0].x /= sqrt_kappa; term2[1][0].y /= sqrt_kappa;
    term2[1][1] = make_cuComplex(cosf(randomPhase[3]), sinf(randomPhase[3]));

    // Term 3: Tx antenna field pattern
    cuComplex term3[2] = {make_cuComplex(F_tx_theta, 0.0f), make_cuComplex(F_tx_phi, 0.0f)};

    // Term 4: Rx antenna array response
    float r_head_rx[3] = {
        sinf(theta_ZOA_rad) * cosf(phi_AOA_rad),
        sinf(theta_ZOA_rad) * sinf(phi_AOA_rad),
        cosf(theta_ZOA_rad)
    };
    float phase_rx = 2.0f * M_PI * (r_head_rx[0] * d_bar_rx[0] + r_head_rx[1] * d_bar_rx[1] + r_head_rx[2] * d_bar_rx[2]);
    cuComplex term4 = make_cuComplex(cosf(phase_rx), sinf(phase_rx));

    // Term 5: Tx antenna array response
    float r_head_tx[3] = {
        sinf(theta_ZOD_rad) * cosf(phi_AOD_rad),
        sinf(theta_ZOD_rad) * sinf(phi_AOD_rad),
        cosf(theta_ZOD_rad)
    };
    float phase_tx = 2.0f * M_PI * (r_head_tx[0] * d_bar_tx[0] + r_head_tx[1] * d_bar_tx[1] + r_head_tx[2] * d_bar_tx[2]);
    cuComplex term5 = make_cuComplex(cosf(phase_tx), sinf(phase_tx));

    // Term 6: Doppler effect
    float v_bar[3] = {
        utVelocityPolar[2] * sinf(utVelocityPolar[0]) * cosf(utVelocityPolar[1]),
        utVelocityPolar[2] * sinf(utVelocityPolar[0]) * sinf(utVelocityPolar[1]),
        utVelocityPolar[2] * cosf(utVelocityPolar[0])
    };
    float doppler_phase = 2.0f * M_PI * (r_head_rx[0] * v_bar[0] + r_head_rx[1] * v_bar[1] + r_head_rx[2] * v_bar[2]) * currentTime / lambda_0;
    cuComplex term6 = make_cuComplex(cosf(doppler_phase), sinf(doppler_phase));

    // Combine all terms according to equation 7.5-22
    cuComplex result = make_cuComplex(0.0f, 0.0f);
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            cuComplex temp = cuCmulf(term1[i], term2[i][j]);
            temp = cuCmulf(temp, term3[j]);
            temp = cuCmulf(temp, term4);
            temp = cuCmulf(temp, term5);
            temp = cuCmulf(temp, term6);
            result = cuCaddf(result, temp);
        }
    }
    return result;
}

// Helper function to calculate LOS ray coefficient (LOS case, similar to calculateRayCoefficient)
inline cuComplex calculateLOSCoefficient(
    const AntPanelConfig& utAntConfig, int ueAntIdx, float theta_LOS_ZOA, float phi_LOS_AOA, float zetaOffsetUtAnt,
    const AntPanelConfig& bsAntConfig, int bsAntIdx, float theta_LOS_ZOD, float phi_LOS_AOD, float zetaOffsetBsAnt,
    float currentTime, float* utVelocityPolar, float lambda_0, float d_3d)
{
    // Compute d_bar_rx (Rx antenna element position)
    int M = utAntConfig.antSize[2];
    int N = utAntConfig.antSize[3];
    int P = utAntConfig.antSize[4];
    // int m = (ueAntIdx / (N * P)) % M;
    // int n = (ueAntIdx / P) % N;
    int p_rx = ueAntIdx % P;
    float d_h_rx = utAntConfig.antSpacing[2];
    float d_v_rx = utAntConfig.antSpacing[3];
    float d_bar_rx[3] = { (ueAntIdx / (N * P)) % M * d_h_rx, (ueAntIdx / P) % N * d_v_rx, 0.0f };

    // Convert angles to radians
    float d2pi = M_PI / 180.0f;
    float theta_LOS_ZOA_rad = theta_LOS_ZOA * d2pi;
    float phi_LOS_AOA_rad = phi_LOS_AOA * d2pi;
    float theta_LOS_ZOD_rad = theta_LOS_ZOD * d2pi;
    float phi_LOS_AOD_rad = phi_LOS_AOD * d2pi;

    // Calculate field patterns for Rx
    float F_rx_theta, F_rx_phi, F_tx_theta, F_tx_phi;
    calculateFieldComponents(utAntConfig, theta_LOS_ZOA, phi_LOS_AOA, utAntConfig.antPolarAngles[p_rx] + zetaOffsetUtAnt, F_rx_theta, F_rx_phi);

    // Compute d_bar_tx (Tx antenna element position)
    M = bsAntConfig.antSize[2];
    N = bsAntConfig.antSize[3];
    P = bsAntConfig.antSize[4];
    // int m_tx = (bsAntIdx / (N * P)) % M;
    // int n_tx = (bsAntIdx / P) % N;
    int p_tx = bsAntIdx % P;
    float d_h_tx = bsAntConfig.antSpacing[2];
    float d_v_tx = bsAntConfig.antSpacing[3];
    float d_bar_tx[3] = { (bsAntIdx / (N * P)) % M * d_h_tx, (bsAntIdx / P) % N * d_v_tx, 0.0f };
    // Calculate field patterns for Tx
    calculateFieldComponents(bsAntConfig, theta_LOS_ZOD, phi_LOS_AOD, bsAntConfig.antPolarAngles[p_tx] + zetaOffsetBsAnt, F_tx_theta, F_tx_phi);

    // Term 1: Rx antenna field pattern
    cuComplex term1[2] = {make_cuComplex(F_rx_theta, 0.0f), make_cuComplex(F_rx_phi, 0.0f)};

    // Term 2: LOS polarization matrix [1 0; 0 -1]
    cuComplex term2[2][2];
    term2[0][0] = make_cuComplex(1.0f, 0.0f);
    term2[0][1] = make_cuComplex(0.0f, 0.0f);
    term2[1][0] = make_cuComplex(0.0f, 0.0f);
    term2[1][1] = make_cuComplex(-1.0f, 0.0f);

    // Term 3: Tx antenna field pattern
    cuComplex term3[2] = {make_cuComplex(F_tx_theta, 0.0f), make_cuComplex(F_tx_phi, 0.0f)};

    // Term 4: Rx antenna array response
    float r_head_rx[3] = {
        sinf(theta_LOS_ZOA_rad) * cosf(phi_LOS_AOA_rad),
        sinf(theta_LOS_ZOA_rad) * sinf(phi_LOS_AOA_rad),
        cosf(theta_LOS_ZOA_rad)
    };
    float phase_rx = 2.0f * M_PI * (r_head_rx[0] * d_bar_rx[0] + r_head_rx[1] * d_bar_rx[1] + r_head_rx[2] * d_bar_rx[2]);
    cuComplex term4 = make_cuComplex(cosf(phase_rx), sinf(phase_rx));

    // Term 5: Tx antenna array response
    float r_head_tx[3] = {
        sinf(theta_LOS_ZOD_rad) * cosf(phi_LOS_AOD_rad),
        sinf(theta_LOS_ZOD_rad) * sinf(phi_LOS_AOD_rad),
        cosf(theta_LOS_ZOD_rad)
    };
    float phase_tx = 2.0f * M_PI * (r_head_tx[0] * d_bar_tx[0] + r_head_tx[1] * d_bar_tx[1] + r_head_tx[2] * d_bar_tx[2]);
    cuComplex term5 = make_cuComplex(cosf(phase_tx), sinf(phase_tx));

    // Term 6: Doppler effect
    float v_bar[3] = {
        utVelocityPolar[2] * sinf(utVelocityPolar[0]) * cosf(utVelocityPolar[1]),
        utVelocityPolar[2] * sinf(utVelocityPolar[0]) * sinf(utVelocityPolar[1]),
        utVelocityPolar[2] * cosf(utVelocityPolar[0])
    };
    float doppler_phase = 2.0f * M_PI * (r_head_rx[0] * v_bar[0] + r_head_rx[1] * v_bar[1] + r_head_rx[2] * v_bar[2]) * currentTime / lambda_0;
    cuComplex term6 = make_cuComplex(cosf(doppler_phase), sinf(doppler_phase));

    // Term 7: LOS phase term exp(-j*2*pi*d_3d/lambda_0)
    float los_phase = -2.0f * M_PI * d_3d / lambda_0;
    cuComplex term7 = make_cuComplex(cosf(los_phase), sinf(los_phase));

    // Combine all terms according to equation 7.5-22
    cuComplex result = make_cuComplex(0.0f, 0.0f);
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            cuComplex temp = cuCmulf(term1[i], term2[i][j]);
            temp = cuCmulf(temp, term3[j]);
            temp = cuCmulf(temp, term7); // LOS phase
            temp = cuCmulf(temp, term4);
            temp = cuCmulf(temp, term5);
            temp = cuCmulf(temp, term6);
            result = cuCaddf(result, temp);
        }
    }
    return result;
}

template <typename Tscalar, typename Tcomplex>
void slsChan<Tscalar, Tcomplex>::calClusterRay()
{
    // For each link
    for(uint16_t siteIdx = 0; siteIdx < m_topology.nSite; siteIdx++) {
        for(uint16_t ueIdx = 0; ueIdx < m_topology.nUT; ueIdx++) {
            uint32_t linkIdx = siteIdx * m_topology.nUT + ueIdx;
            bool losInd = m_linkParams[linkIdx].losInd;
            
            // Add indoor status for O2I logic similar to large scale params
            // assume BS is always outdoor
            uint8_t isO2I = (m_topology.utParams[ueIdx].outdoor_ind == 0);  // 1 if indoor (O2I), 0 if outdoor
            
            // Calculate proper index for cmnLinkParams arrays: 2 for O2I, 1 for LOS, 0 for NLOS
            uint8_t lspIdx = isO2I ? 2 : losInd;
            
            // Set number of clusters and rays for this link
            m_clusterParams[linkIdx].nCluster = m_cmnLinkParams.nCluster[lspIdx];
            m_clusterParams[linkIdx].nRayPerCluster = m_cmnLinkParams.nRayPerCluster[lspIdx];
            
            // generate cluster delays and powers
            // find the indexes of the two strongest clusters
            genClusterDelayAndPower(m_linkParams[linkIdx].DS,
                                  m_cmnLinkParams.r_tao[lspIdx],
                                  losInd,
                                  m_clusterParams[linkIdx].nCluster,
                                  m_linkParams[linkIdx].K,
                                  m_cmnLinkParams.xi[lspIdx],
                                  m_topology.utParams[ueIdx].outdoor_ind,
                                  m_clusterParams[linkIdx].delays,
                                  m_clusterParams[linkIdx].powers,
                                  m_clusterParams[linkIdx].strongest2clustersIdx,
                                  m_gen, m_uniformDist, m_normalDist);

            
            // Generate arrival and departure angles
            genClusterAngle(m_clusterParams[linkIdx].nCluster,
                          m_cmnLinkParams.C_ASA[lspIdx],
                          m_cmnLinkParams.C_ASD[lspIdx],
                          m_cmnLinkParams.C_phi_NLOS,
                          m_cmnLinkParams.C_phi_LOS,
                          m_cmnLinkParams.C_phi_O2I,
                          m_cmnLinkParams.C_theta_NLOS,
                          m_cmnLinkParams.C_theta_LOS,
                          m_cmnLinkParams.C_theta_O2I,
                          m_linkParams[linkIdx].ASA,
                          m_linkParams[linkIdx].ASD,
                          m_linkParams[linkIdx].ZSA,
                          m_linkParams[linkIdx].ZSD,
                          m_linkParams[linkIdx].phi_LOS_AOA,
                          m_linkParams[linkIdx].phi_LOS_AOD,
                          m_linkParams[linkIdx].theta_LOS_ZOA,
                          m_linkParams[linkIdx].theta_LOS_ZOD,
                          m_linkParams[linkIdx].mu_offset_ZOD,
                          losInd,
                          m_topology.utParams[ueIdx].outdoor_ind,
                          m_linkParams[linkIdx].K,
                          m_clusterParams[linkIdx].powers,
                          m_clusterParams[linkIdx].phi_n_AoA,
                          m_clusterParams[linkIdx].phi_n_AoD,
                          m_clusterParams[linkIdx].theta_n_ZOD,
                          m_clusterParams[linkIdx].theta_n_ZOA,
                          m_gen,
                          m_uniformDist,
                          m_normalDist);

            // generate ray angles
            // Extract cluster spread factors based on LOS state
            float C_ASA = m_cmnLinkParams.C_ASA[lspIdx];
            float C_ASD = m_cmnLinkParams.C_ASD[lspIdx];
            float C_ZSA = m_cmnLinkParams.C_ZSA[lspIdx];
            // Use mu_lgZSD from link parameters following MATLAB reference: (3/8)*10^mu_lgZSD
            float C_ZSD = (3.0f/8.0f) * std::pow(10.0f, m_linkParams[linkIdx].mu_lgZSD);
            
            genRayAngle(m_clusterParams[linkIdx].nCluster,
                        m_clusterParams[linkIdx].nRayPerCluster,
                        m_clusterParams[linkIdx].phi_n_AoA,
                        m_clusterParams[linkIdx].phi_n_AoD,
                        m_clusterParams[linkIdx].theta_n_ZOD,
                        m_clusterParams[linkIdx].theta_n_ZOA,
                        m_clusterParams[linkIdx].phi_n_m_AoA,
                        m_clusterParams[linkIdx].phi_n_m_AoD,
                        m_clusterParams[linkIdx].theta_n_m_ZOD,
                        m_clusterParams[linkIdx].theta_n_m_ZOA,
                        C_ASA,
                        C_ASD,
                        C_ZSA,
                        C_ZSD,
                        m_gen,
                        m_uniformDist);
            
            // generate XPR and random phases
            for (uint16_t clusterIdx = 0; clusterIdx < m_clusterParams[linkIdx].nCluster; clusterIdx++) {
                for (uint16_t rayIdx = 0; rayIdx < m_clusterParams[linkIdx].nRayPerCluster; rayIdx++) {
                    // Generate XPR values
                                m_clusterParams[linkIdx].xpr[clusterIdx * m_clusterParams[linkIdx].nRayPerCluster + rayIdx] = std::pow(10.0f, (m_cmnLinkParams.mu_XPR[lspIdx] +
                                                          m_cmnLinkParams.sigma_XPR[lspIdx] * m_normalDist(m_gen)) / 10.0f);
                
                    // Generate random phases
                    m_clusterParams[linkIdx].randomPhases[(clusterIdx * m_clusterParams[linkIdx].nRayPerCluster + rayIdx) * 4] = (m_uniformDist(m_gen) - 0.5f) * 2 * M_PI;
                    m_clusterParams[linkIdx].randomPhases[(clusterIdx * m_clusterParams[linkIdx].nRayPerCluster + rayIdx) * 4 + 1] = (m_uniformDist(m_gen) - 0.5f) * 2 * M_PI;
                    m_clusterParams[linkIdx].randomPhases[(clusterIdx * m_clusterParams[linkIdx].nRayPerCluster + rayIdx) * 4 + 2] = (m_uniformDist(m_gen) - 0.5f) * 2 * M_PI;
                    m_clusterParams[linkIdx].randomPhases[(clusterIdx * m_clusterParams[linkIdx].nRayPerCluster + rayIdx) * 4 + 3] = (m_uniformDist(m_gen) - 0.5f) * 2 * M_PI;
                }
            }
        }
    }
}


template <typename Tscalar, typename Tcomplex>
void slsChan<Tscalar, Tcomplex>::generateCIR()
{
    // Calculate time offset between snapshots
    const float timeOffset = 1e-3f / (m_simConfig->sc_spacing_hz * 15e3f * m_simConfig->n_snapshot_per_slot);

    // For each active link
    for (size_t activeLinkIdx = 0; activeLinkIdx < m_activeLinkParams.size(); activeLinkIdx++) {
        auto & activeLink = m_activeLinkParams[activeLinkIdx];
        uint16_t & cid = activeLink.cid;
        uint16_t & uid = activeLink.uid;
        uint32_t & linkIdx = activeLink.linkIdx;
        uint32_t & lspReadIdx = activeLink.lspReadIdx;
        Tcomplex * cirCoe = activeLink.cirCoe;
        uint16_t * cirNormDelay = activeLink.cirNormDelay;
        uint16_t * cirNtaps = activeLink.cirNtaps;
        
        uint8_t & losInd = m_linkParams[lspReadIdx].losInd;
        
        // Calculate O2I status for this link (needed for LOS processing)
        uint8_t isO2I = (m_topology.utParams[uid].outdoor_ind == 0);  // 1 if indoor (O2I), 0 if outdoor
        // Calculate proper index for cmnLinkParams arrays: 2 for O2I, 1 for LOS, 0 for NLOS
        uint8_t lspIdx = isO2I ? 2 : losInd;

        float C_DS = m_cmnLinkParams.C_DS[lspIdx];
        uint16_t nCluster = m_clusterParams[lspReadIdx].nCluster;
        uint16_t nRayPerCluster = m_clusterParams[lspReadIdx].nRayPerCluster;
        float K = m_linkParams[lspReadIdx].K;
        float K_R = pow(10.0f, K / 10.0f);

        // Get antenna parameters
        uint32_t utAntPanelIdx = m_topology.utParams[uid].antPanelIdx;
        uint32_t cellAntPanelIdx = m_topology.cellParams[cid].antPanelIdx;
        const AntPanelConfig & utAntPanelConfig = (*m_antPanelConfig)[utAntPanelIdx];
        const AntPanelConfig & cellAntPanelConfig = (*m_antPanelConfig)[cellAntPanelIdx];
        uint32_t nUtAnt = utAntPanelConfig.nAnt;
        uint32_t nCellAnt = cellAntPanelConfig.nAnt;
        float * utAntPanelOrientation = m_topology.utParams[uid].antPanelOrientation;
        float * cellAntPanelOrientation = m_topology.cellParams[cid].antPanelOrientation;

        // convert ut velocity to vector
        auto utVelocityXYZ = m_topology.utParams[uid].velocity;
        float utVelocityPolar[3] = {
            atan2f(utVelocityXYZ[1], utVelocityXYZ[0]),  // azimuth
            0,  // zenith will always be zero
            sqrtf(utVelocityXYZ[0] * utVelocityXYZ[0] + utVelocityXYZ[1] * utVelocityXYZ[1] + utVelocityXYZ[2] * utVelocityXYZ[2])  // speed
        };

        // calculate the tap indices
        std::vector<uint16_t> H_tapIdx(N_MAX_TAPS, 0);

        // For each snapshot
        for (uint16_t snapshotIdx = 0; snapshotIdx < m_simConfig->n_snapshot_per_slot; snapshotIdx++) {
            // Calculate time for this snapshot
            float snapshotTime = m_refTime + snapshotIdx * timeOffset;
            
            // Add propagation delay if enabled
            if (m_sysConfig->enable_propagation_delay == 1) {
                snapshotTime += m_linkParams[lspReadIdx].d3d / 3.0e8f;  // d_3d / speed_of_light
            }
            size_t snapshotOffset = snapshotIdx * nUtAnt * nCellAnt * N_MAX_TAPS;
            
            // Check if small scale fading is disabled
            if (m_sysConfig->disable_small_scale_fading == 1) {
                // Small scale fading disabled: only apply path loss (fast fading = 1)
                // Reset cirCoe to zero first
                for (uint16_t utAntIdx = 0; utAntIdx < nUtAnt; utAntIdx++) {
                    for (uint16_t bsAntIdx = 0; bsAntIdx < nCellAnt; bsAntIdx++) {
                        for (uint16_t tap = 0; tap < N_MAX_TAPS; tap++) {
                            cirCoe[snapshotOffset + (utAntIdx * nCellAnt + bsAntIdx) * N_MAX_TAPS + tap] = make_cuComplex(0.0f, 0.0f);
                        }
                    }
                }
                
                // Apply path loss, shadowing, and antenna patterns deterministically (Phase-1)
                // For Phase-1: Apply all pattern- or array-related effects deterministically,
                // since only large-scale, non-random effects are considered
                if (m_sysConfig->disable_pl_shadowing != 1) {
                    // The sign of the shadow fading is defined so that positive SF means more received power at UT than predicted by the path loss model
                    const float pathGain = -(m_linkParams[lspReadIdx].pathloss - m_linkParams[lspReadIdx].SF);
                    const float path_scale = std::pow(10.0f, pathGain / 20.0f);
                    
                    // Get LOS angles and apply antenna orientation corrections (cell-specific for different sectors!)
                    // For co-sited BSs (same site, different sectors), antenna orientations differ (e.g., 30°, 150°, 270°)
                    const float theta_LOS_ZOD = wrapZenith(m_linkParams[lspReadIdx].theta_LOS_ZOD - cellAntPanelOrientation[0]);
                    const float phi_LOS_AOD = wrapAzimuth(m_linkParams[lspReadIdx].phi_LOS_AOD - cellAntPanelOrientation[1]);
                    const float theta_LOS_ZOA = wrapZenith(m_linkParams[lspReadIdx].theta_LOS_ZOA - utAntPanelOrientation[0]);
                    const float phi_LOS_AOA = wrapAzimuth(m_linkParams[lspReadIdx].phi_LOS_AOA - utAntPanelOrientation[1]);
                    const float d_3d = m_linkParams[lspReadIdx].d3d;
                    
                    // Get antenna polarization offsets
                    const float zetaOffsetBsAnt = cellAntPanelOrientation[2];
                    const float zetaOffsetUtAnt = utAntPanelOrientation[2];
                    
                    // Set first tap with path loss + antenna patterns
                    // The corrected angles above ensure different sectors get different antenna gains
                    for (uint16_t utAntIdx = 0; utAntIdx < nUtAnt; utAntIdx++) {
                        for (uint16_t bsAntIdx = 0; bsAntIdx < nCellAnt; bsAntIdx++) {
                            // Calculate deterministic antenna response using LOS component
                            // This includes: antenna field pattern + array response
                            const Tcomplex antennaResponse = calculateLOSCoefficient(
                                utAntPanelConfig, utAntIdx, theta_LOS_ZOA, phi_LOS_AOA, zetaOffsetUtAnt,
                                cellAntPanelConfig, bsAntIdx, theta_LOS_ZOD, phi_LOS_AOD, zetaOffsetBsAnt,
                                snapshotTime, utVelocityPolar, m_cmnLinkParams.lambda_0, d_3d
                            );
                            
                            // Combine path loss/shadowing with antenna gain
                            // antennaResponse contains: F_rx * F_tx * array_response * doppler
                            cirCoe[snapshotOffset + (utAntIdx * nCellAnt + bsAntIdx) * N_MAX_TAPS + 0] = 
                                make_cuComplex(
                                    antennaResponse.x * path_scale,
                                    antennaResponse.y * path_scale
                                );
                        }
                    }
                } else {
                    // Both path loss and small scale fading disabled: set unit channel (1+0j)
                    for (uint16_t utAntIdx = 0; utAntIdx < nUtAnt; utAntIdx++) {
                        for (uint16_t bsAntIdx = 0; bsAntIdx < nCellAnt; bsAntIdx++) {
                            cirCoe[snapshotOffset + (utAntIdx * nCellAnt + bsAntIdx) * N_MAX_TAPS + 0] = make_cuComplex(1.0f, 0.0f);
                        }
                    }
                }
                
                // Set delay and taps info for simplified channel
                if (snapshotIdx == 0) {  // Only set once per link
                    cirNormDelay[0] = 0;  // Single tap at delay 0
                    for (uint16_t tap = 1; tap < N_MAX_TAPS; tap++) {
                        cirNormDelay[tap] = 0;
                    }
                    cirNtaps[0] = 1;  // Only one tap
                }
                
                continue;  // Skip the complex small scale fading calculations
            }
            
            // Initialize channel matrix for this link and snapshot
            std::vector<Tcomplex> H_link(nUtAnt * nCellAnt * N_MAX_TAPS, Tcomplex{0.0f, 0.0f});

            // For each cluster
            uint16_t tapCount = 0;
            for (uint16_t clusterIdx = 0; clusterIdx < nCluster; clusterIdx++) {
                // Check if this is one of the strongest 2 clusters
                bool isStrongest2 = false;
                for (uint16_t i = 0; i < 2; i++) {
                    if (clusterIdx == m_clusterParams[lspReadIdx].strongest2clustersIdx[i]) {
                        isStrongest2 = true;
                        break;
                    }
                }

                // Get cluster parameters
                float clusterPower = m_clusterParams[lspReadIdx].powers[clusterIdx];
                
                // For LOS case, subtract LOS component from first cluster before splitting
                // The LOS component K/(K+1) will be added later as a dedicated LOS path
                if (losInd && !isO2I && clusterIdx == 0) {
                    float K_R_plus_1 = K_R + 1.0f;
                    float losPower = K_R / K_R_plus_1;
                    clusterPower -= losPower;
                    clusterPower = std::max(clusterPower, 0.0f);  // Clamp to zero if LOS dominated (cluster 0 was weak)
                }
                
                float normPower = std::sqrt(clusterPower / nRayPerCluster);

                // Handle subclusters for strongest 2 clusters
                if (isStrongest2) {
                    // Use subcluster ray arrays from cmnLinkParams struct (3GPP Table 7.5-5)
                    // Process each subcluster
                    for (int subClusterIdx = 0; subClusterIdx < m_cmnLinkParams.nSubCluster; ++subClusterIdx) {
                        const uint16_t* rays = nullptr;
                        int nRays = 0;
                        float subClusterPower = 0.0f;
                        if (subClusterIdx == 0) {
                            rays = m_cmnLinkParams.raysInSubCluster0;
                            nRays = m_cmnLinkParams.raysInSubClusterSizes[0];
                            subClusterPower = sqrt(10.0f/20.0f);
                        } else if (subClusterIdx == 1) {
                            rays = m_cmnLinkParams.raysInSubCluster1;
                            nRays = m_cmnLinkParams.raysInSubClusterSizes[1];
                            subClusterPower = sqrt(6.0f/20.0f);
                        } else if (subClusterIdx == 2) {
                            rays = m_cmnLinkParams.raysInSubCluster2;
                            nRays = m_cmnLinkParams.raysInSubClusterSizes[2];
                            subClusterPower = sqrt(4.0f/20.0f);
                        }
                        for (int rayIdx = 0; rayIdx < nRays; ++rayIdx) {
                            // Get ray parameters and apply antenna orientation corrections (following API spec)
                            float theta_ZOA = m_clusterParams[lspReadIdx].theta_n_ZOA[clusterIdx * nRayPerCluster + rays[rayIdx]] - utAntPanelOrientation[0];
                            float phi_AOA = m_clusterParams[lspReadIdx].phi_n_AoA[clusterIdx * nRayPerCluster + rays[rayIdx]] - utAntPanelOrientation[1];
                            float theta_ZOD = m_clusterParams[lspReadIdx].theta_n_ZOD[clusterIdx * nRayPerCluster + rays[rayIdx]] - cellAntPanelOrientation[0];
                            float phi_AOD = m_clusterParams[lspReadIdx].phi_n_AoD[clusterIdx * nRayPerCluster + rays[rayIdx]] - cellAntPanelOrientation[1];                            

                            // Add to channel matrix at the correct tap
                            for (uint16_t utAntIdx = 0; utAntIdx < nUtAnt; utAntIdx++) {
                                for (uint16_t bsAntIdx = 0; bsAntIdx < nCellAnt; bsAntIdx++) {
                                    // Calculate ray coefficient with snapshot time
                                    Tcomplex rayCoeff = calculateRayCoefficient(
                                        utAntPanelConfig, utAntIdx, theta_ZOA, phi_AOA, utAntPanelOrientation[2],
                                        cellAntPanelConfig, bsAntIdx, theta_ZOD, phi_AOD, cellAntPanelOrientation[2],
                                        m_clusterParams[lspReadIdx].xpr[clusterIdx * nRayPerCluster + rays[rayIdx]],
                                        m_clusterParams[lspReadIdx].randomPhases + (clusterIdx * nRayPerCluster + rays[rayIdx]) * 4,
                                        snapshotTime,
                                        utVelocityPolar,
                                        m_cmnLinkParams.lambda_0
                                    );
                                    rayCoeff = make_cuComplex(rayCoeff.x * normPower * subClusterPower, rayCoeff.y * normPower * subClusterPower);

                                    H_link[(utAntIdx * nCellAnt + bsAntIdx) * N_MAX_TAPS + tapCount] = cuCaddf(
                                        H_link[(utAntIdx * nCellAnt + bsAntIdx) * N_MAX_TAPS + tapCount], rayCoeff);
                                }
                            }
                        }
                        if (snapshotIdx == 0) {
                            float clusterDelay = m_clusterParams[lspReadIdx].delays[clusterIdx];
                            if (subClusterIdx == 1) {
                                clusterDelay += 1.28 * C_DS;
                            }
                            else if (subClusterIdx == 2) {
                                clusterDelay += 2.56 * C_DS;
                            }
                            if (m_sysConfig->enable_propagation_delay == 1) {
                                H_tapIdx[tapCount] = static_cast<uint16_t>(std::round((clusterDelay * 1e-9 + m_linkParams[lspReadIdx].d3d / 3.0e8f) * m_simConfig->sc_spacing_hz * m_simConfig->fft_size));
                            }
                            else {
                                H_tapIdx[tapCount] = static_cast<uint16_t>(std::round(clusterDelay * 1e-9 * m_simConfig->sc_spacing_hz * m_simConfig->fft_size));
                            }
                        }
#ifdef SLS_DEBUG_
                        // Check for buffer overflow before incrementing tapCount
                        if (tapCount >= N_MAX_TAPS - 1) {
                            printf("ERROR: tapCount (%d) exceeds N_MAX_TAPS (%d) limit. Cluster processing stopped to prevent buffer overflow.\n", 
                                   tapCount + 1, N_MAX_TAPS);
                            return;
                        }
#endif
                        tapCount++;
                    }
                } else {
                    // Process all rays for non-strongest clusters
                    for (uint16_t rayIdx = 0; rayIdx < nRayPerCluster; rayIdx++) {
                        // Get ray parameters and apply antenna orientation corrections (following API spec)
                        float theta_ZOA = m_clusterParams[lspReadIdx].theta_n_ZOA[clusterIdx * nRayPerCluster + rayIdx] - utAntPanelOrientation[0];
                        float phi_AOA = m_clusterParams[lspReadIdx].phi_n_AoA[clusterIdx * nRayPerCluster + rayIdx] - utAntPanelOrientation[1];
                        float theta_ZOD = m_clusterParams[lspReadIdx].theta_n_ZOD[clusterIdx * nRayPerCluster + rayIdx] - cellAntPanelOrientation[0];
                        float phi_AOD = m_clusterParams[lspReadIdx].phi_n_AoD[clusterIdx * nRayPerCluster + rayIdx] - cellAntPanelOrientation[1];

                        // Add to channel matrix at the correct tap
                        for (uint16_t utAntIdx = 0; utAntIdx < nUtAnt; utAntIdx++) {
                            for (uint16_t bsAntIdx = 0; bsAntIdx < nCellAnt; bsAntIdx++) {
                            // Calculate ray coefficient with snapshot time
                            Tcomplex rayCoeff = calculateRayCoefficient(
                                utAntPanelConfig, utAntIdx, theta_ZOA, phi_AOA, utAntPanelOrientation[2],
                                cellAntPanelConfig, bsAntIdx, theta_ZOD, phi_AOD, cellAntPanelOrientation[2],
                                m_clusterParams[lspReadIdx].xpr[clusterIdx * nRayPerCluster + rayIdx],
                                m_clusterParams[lspReadIdx].randomPhases + (clusterIdx * nRayPerCluster + rayIdx) * 4,
                                snapshotTime,
                                utVelocityPolar,
                                m_cmnLinkParams.lambda_0
                            );
                            rayCoeff = make_cuComplex(rayCoeff.x * normPower, rayCoeff.y * normPower);

                            H_link[(utAntIdx * nCellAnt + bsAntIdx) * N_MAX_TAPS + tapCount] = cuCaddf(
                                H_link[(utAntIdx * nCellAnt + bsAntIdx) * N_MAX_TAPS + tapCount], rayCoeff);
                            }
                        }
                    }
                    if (snapshotIdx == 0) {
                        float clusterDelay = m_clusterParams[lspReadIdx].delays[clusterIdx];
                        if (m_sysConfig->enable_propagation_delay == 1) {
                            H_tapIdx[tapCount] = static_cast<uint16_t>(std::round((clusterDelay * 1e-9 + m_linkParams[lspReadIdx].d3d / 3.0e8f) * m_simConfig->sc_spacing_hz * m_simConfig->fft_size));
                        }
                        else {
                            H_tapIdx[tapCount] = static_cast<uint16_t>(std::round(clusterDelay * 1e-9 * m_simConfig->sc_spacing_hz * m_simConfig->fft_size));
                        }
                    }
#ifdef SLS_DEBUG_
                    // Check for buffer overflow before incrementing tapCount
                    if (tapCount >= N_MAX_TAPS - 1) {
                        printf("ERROR: tapCount (%d) exceeds N_MAX_TAPS (%d) limit. Cluster processing stopped to prevent buffer overflow.\n", 
                               tapCount + 1, N_MAX_TAPS);
                        return;
                    }
#endif
                    tapCount++;
                }

                // Handle LOS case if present
                if (losInd && !isO2I) {
                    // Calculate LOS component with snapshot time for each antenna pair
                    for (uint16_t utAntIdx = 0; utAntIdx < nUtAnt; utAntIdx++) {
                        for (uint16_t bsAntIdx = 0; bsAntIdx < nCellAnt; bsAntIdx++) {
                            // Apply antenna orientation corrections to LOS angles (following API spec)
                            float theta_LOS_ZOA_corrected = wrapZenith(m_linkParams[lspReadIdx].theta_LOS_ZOA - utAntPanelOrientation[0]);
                            float phi_LOS_AOA_corrected = wrapAzimuth(m_linkParams[lspReadIdx].phi_LOS_AOA - utAntPanelOrientation[1]);
                            float theta_LOS_ZOD_corrected = wrapZenith(m_linkParams[lspReadIdx].theta_LOS_ZOD - cellAntPanelOrientation[0]);
                            float phi_LOS_AOD_corrected = wrapAzimuth(m_linkParams[lspReadIdx].phi_LOS_AOD - cellAntPanelOrientation[1]);
                            
                            Tcomplex H_LOS = calculateLOSCoefficient(
                                utAntPanelConfig, utAntIdx,
                                theta_LOS_ZOA_corrected,
                                phi_LOS_AOA_corrected,
                                utAntPanelOrientation[2],
                                cellAntPanelConfig, bsAntIdx,
                                theta_LOS_ZOD_corrected,
                                phi_LOS_AOD_corrected,
                                cellAntPanelOrientation[2],
                                snapshotTime,
                                utVelocityPolar,
                                m_cmnLinkParams.lambda_0,
                                m_linkParams[lspReadIdx].d3d
                            );
                            // Combine LOS and NLOS components at tap 0
                            float los_scale = std::sqrt(K_R / (K_R + 1));
                            // nlos already been scaled by K_R / (K_R + 1) in cluster power
                            H_link[(utAntIdx * nCellAnt + bsAntIdx) * N_MAX_TAPS] = cuCaddf(
                                make_cuComplex(los_scale * H_LOS.x, los_scale * H_LOS.y),
                                make_cuComplex(H_link[(utAntIdx * nCellAnt + bsAntIdx) * N_MAX_TAPS].x,
                                               H_link[(utAntIdx * nCellAnt + bsAntIdx) * N_MAX_TAPS].y));
                        }
                    }
                }
            }

            if (snapshotIdx == 0) {
                // process H_tapIdx to get unique items in ascending order
                std::vector<uint16_t> unique_taps;
                for (uint16_t tap = 0; tap < tapCount; tap++) {
                    unique_taps.push_back(H_tapIdx[tap]);
                }
                // Sort in ascending order and remove duplicates (only if we have taps)
                if (!unique_taps.empty()) {
                    std::sort(unique_taps.begin(), unique_taps.end());
                    unique_taps.erase(std::unique(unique_taps.begin(), unique_taps.end()), unique_taps.end());
                }
                // copy to cirNormDelay
                for (uint16_t tap = 0; tap < unique_taps.size(); tap++) {
                    cirNormDelay[tap] = unique_taps[tap];
                }
                for (uint16_t tap = unique_taps.size(); tap < N_MAX_TAPS; tap++) {
                    cirNormDelay[tap] = 0;
                }
                cirNtaps[0] = unique_taps.size();
                
                // Find index of each element in unique_taps and update H_tapIdx
                for (uint16_t tap = 0; tap < tapCount; tap++) {
                    auto it = std::find(unique_taps.begin(), unique_taps.end(), H_tapIdx[tap]);
                    H_tapIdx[tap] = std::distance(unique_taps.begin(), it);
                }
                // Set remaining elements to 0
                for (uint16_t tap = tapCount; tap < N_MAX_TAPS; tap++) {
                    H_tapIdx[tap] = 0;
                }
            }            

            // Apply path loss and shadowing if not disabled
            if (m_sysConfig->disable_pl_shadowing != 1) {
                // The sign of the shadow fading is defined so that positive SF means more received power at UT than predicted by the path loss model
                float pathGain = -(m_linkParams[lspReadIdx].pathloss - m_linkParams[lspReadIdx].SF);
                float path_scale = std::pow(10.0f, pathGain / 20.0f);
                for (uint16_t utAntIdx = 0; utAntIdx < nUtAnt; utAntIdx++) {
                    for (uint16_t bsAntIdx = 0; bsAntIdx < nCellAnt; bsAntIdx++) {
                        for (uint16_t tap = 0; tap < N_MAX_TAPS; tap++) {
                            H_link[(utAntIdx * nCellAnt + bsAntIdx) * N_MAX_TAPS + tap] = make_cuComplex(
                                H_link[(utAntIdx * nCellAnt + bsAntIdx) * N_MAX_TAPS + tap].x * path_scale,
                                H_link[(utAntIdx * nCellAnt + bsAntIdx) * N_MAX_TAPS + tap].y * path_scale);
                        }
                    }
                }
            }

            // reset cirCoe / cirNormDelay = 0
            for (uint16_t utAntIdx = 0; utAntIdx < nUtAnt; utAntIdx++) {
                for (uint16_t bsAntIdx = 0; bsAntIdx < nCellAnt; bsAntIdx++) {
                    for (uint16_t tap = 0; tap < N_MAX_TAPS; tap++) {
                        cirCoe[snapshotOffset + (utAntIdx * nCellAnt + bsAntIdx) * N_MAX_TAPS + tap] = make_cuComplex(0.0f, 0.0f);
                    }
                }
            }

            // combine the channel matrix with the tap indices
            for (uint16_t tapIdx = 0; tapIdx < tapCount; tapIdx++) {
                // if same tap index, then add the channel matrix
                for (uint16_t utAntIdx = 0; utAntIdx < nUtAnt; utAntIdx++) {
                    for (uint16_t bsAntIdx = 0; bsAntIdx < nCellAnt; bsAntIdx++) {
                        cirCoe[snapshotOffset + (utAntIdx * nCellAnt + bsAntIdx) * N_MAX_TAPS + H_tapIdx[tapIdx]] = cuCaddf(
                            cirCoe[snapshotOffset + (utAntIdx * nCellAnt + bsAntIdx) * N_MAX_TAPS + H_tapIdx[tapIdx]],
                            H_link[(utAntIdx * nCellAnt + bsAntIdx) * N_MAX_TAPS + tapIdx]);
                    }
                }
            }
        }
    }
}

// Generate channel frequency response
template <typename Tscalar, typename Tcomplex>
void slsChan<Tscalar, Tcomplex>::generateCFR() {
    // Implementation of CFR generation
    // This would typically involve:
    // 1. Taking FFT of CIR
    // 2. Storing results in m_freqChanSc and m_freqChanPrbg
    // TODO: Implementation details for CFR generation
    throw std::runtime_error("CPU CFR generation not implemented. Use GPU method generateCFRGPU() instead.");
}

// Explicit template instantiations
template class slsChan<float, float2>; 