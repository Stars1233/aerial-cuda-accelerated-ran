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
#include <numeric>    // For std::iota
#include <stdexcept>


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
    // Table 7.3-1: A''(theta,phi) = -min{-(A_V + A_H), A_max}; the separable table sum
    // alone would reach -60 dB in back lobes, so re-clamp the combined attenuation to A_max
    float A_db_3D = antConfig.antTheta[theta_idx] + antConfig.antPhi[phi_idx];
    if (antConfig.antModel == 1) {
        A_db_3D = std::max(A_db_3D, -SLS_ANTENNA_PATTERN_A_MAX_DB) + SLS_ANTENNA_GAIN_MAX_DBI;
    }
    float A_3D_sqrt = powf(10.0f, A_db_3D / 20.0f); // equivalent to sqrt(10^(A_db_3D/10))
    F_theta = A_3D_sqrt * cosf(zeta_rad);
    F_phi = A_3D_sqrt * sinf(zeta_rad);
}


// Generate cluster delays and powers (also called from sls_chan_isac.cpp via a forward
// declaration -- needs external linkage, so must NOT be inline)
void genClusterDelayAndPower(
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
    
    // Sort delays and normalize to start from zero
    std::sort(delays, delays + nCluster);  // Sort array range
    float minDelay = delays[0];
    for (uint16_t i = 0; i < nCluster; i++) {
        delays[i] -= minDelay;
    }

    // Generate cluster powers with exponential decay (eq 7.5-5)
    // NOTE: powers must be computed from the UNSCALED delays — per Step 5 the C_tau-scaled
    // LOS delays "are not to be used in cluster power generation"
    float totalPower = 0.0f;
    for (uint16_t clusterIdx = 0; clusterIdx < nCluster; clusterIdx++) {
        float power = std::exp(-delays[clusterIdx] * (r_tao - 1.0f) / (r_tao * delaySpread));
        power *= std::pow(10.0f, - (xi * normalDist(gen)) / 10.0f);  // Add some randomness
        powers[clusterIdx] = power;  // Use array indexing
        totalPower += power;
    }

    // Apply C_tau scaling (eq 7.5-3/7.5-4) AFTER the power generation; the scaled delays
    // are what Steps 11-12 consume as CIR tap delays
    if (losInd && outdoor_ind) {
        float C_tao = 0.7705f - 0.0433f * K + 0.0002f * std::pow(K, 2) + 0.000017f * std::pow(K, 3);
        for (uint16_t i = 0; i < nCluster; i++) {
            delays[i] /= C_tao;
        }
    }

    // Normalize powers
    for (uint16_t i = 0; i < nCluster; i++) {
        powers[i] /= totalPower;
    }

    // Apply the LOS K-factor split before deriving the -25 dB threshold. The
    // threshold is defined against the final cluster powers, including the
    // specular component added to cluster zero.
    if (losInd && outdoor_ind) {
        const float K_R = std::pow(10.0f, K / 10.0f);
        const float P1_LOS = K_R / (K_R + 1.0f);
        const float P_n_LOS = 1.0f / (K_R + 1.0f);

        for (uint16_t i = 0; i < nCluster; i++) {
            powers[i] *= P_n_LOS;
        }
        powers[0] += P1_LOS;
    }

    // Find max power and threshold for filtering.
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
    } else {
        // cluster 0 survives the -25 dB threshold: keep it in slot 0 (without this the
        // compaction below overwrites the delay-0 cluster on every link)
        validClusterCount = 1;
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
    
    // Select the strongest clusters only after pruning the final LOS powers.
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
        throw std::runtime_error("genClusterAngle received an invalid maximum cluster power");
    }

    // Generate AOA (Azimuth of Arrival)
    for (uint16_t n = 0; n < nCluster; n++) {
        float Xn = (uniformDist(gen) < 0.5f) ? 1.0f : -1.0f;
        float Yn = ASA / 7.0f * normalDist(gen);
        
        // Safety check for power ratio to prevent log(0) and sqrt(negative)
        float power_ratio = powers[n] / max_p_n;
        if (power_ratio <= 0.0f) {
            throw std::runtime_error("genClusterAngle received a non-positive cluster power ratio");
        }
        power_ratio = std::min(power_ratio, 1.0f);   // Ensure ratio <= 1
        
        float log_term = -std::log(power_ratio);
        if (std::isnan(log_term) || std::isinf(log_term) || log_term < 0.0f) {
            throw std::runtime_error("genClusterAngle produced an invalid logarithmic power term");
        }
        
        float phi_prime_AOA = 2.0f * (ASA / 1.4f) * std::sqrt(log_term) / std::max(C_phi, 1e-6f);
        phi_n_AoA[n] = Xn * phi_prime_AOA + Yn + phi_LOS_AOA;
    }
    
    if (losInd && outdoor_ind) {
        // eq 7.5-12: re-center so the FIRST cluster coincides with the LOS direction
        const float aoaShift = phi_n_AoA[0] - phi_LOS_AOA;
        for (uint16_t n = 0; n < nCluster; n++) {
            phi_n_AoA[n] -= aoaShift;
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
        // eq 7.5-12: re-center so the FIRST cluster coincides with the LOS direction
        const float aodShift = phi_n_AoD[0] - phi_LOS_AOD;
        for (uint16_t n = 0; n < nCluster; n++) {
            phi_n_AoD[n] -= aodShift;
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
        // eq 7.5-17: re-center so the FIRST cluster coincides with the LOS direction
        const float zoaShift = theta_n_ZOA[0] - theta_LOS_ZOA;
        for (uint16_t n = 0; n < nCluster; n++) {
            theta_n_ZOA[n] -= zoaShift;
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
        // eq 7.5-19 with the 7.5-17 LOS substitution: anchor the FIRST cluster
        const float zodShift = theta_n_ZOD[0] - theta_LOS_ZOD - mu_offset_ZOD;
        for (uint16_t n = 0; n < nCluster; n++) {
            theta_n_ZOD[n] -= zodShift;
        }
    }
}

// 3GPP TR 38.901 Table 7.5-5 sub-cluster ray sets (0-based ray indices; same sets as
// cmnLinkParams.raysInSubCluster0/1/2). Step 8 requires the coupling permutations of the
// two strongest clusters to stay within each set, so every delay sub-cluster keeps its
// own designed Table 7.5-3 offset footprint.
static const uint16_t kSubClusterRays[3][10] = {
    {0, 1, 2, 3, 4, 5, 6, 7, 18, 19},
    {8, 9, 10, 11, 16, 17, 0, 0, 0, 0},
    {12, 13, 14, 15, 0, 0, 0, 0, 0, 0}
};
static const uint16_t kSubClusterSizes[3] = {10, 6, 4};

// Shuffle a 20-entry offset-index permutation within each Table 7.5-5 sub-cluster ray set,
// so idx restricted to each set is a permutation of that set.
static inline void shuffleWithinSubClusters(std::vector<uint16_t>& idx, std::mt19937& gen)
{
    for (int s = 0; s < 3; s++) {
        uint16_t tmp[10];
        for (uint16_t k = 0; k < kSubClusterSizes[s]; k++) {
            tmp[k] = idx[kSubClusterRays[s][k]];
        }
        std::shuffle(tmp, tmp + kSubClusterSizes[s], gen);
        for (uint16_t k = 0; k < kSubClusterSizes[s]; k++) {
            idx[kSubClusterRays[s][k]] = tmp[k];
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
    const uint16_t* strongest2clustersIdx,
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
        
        // Apply random shuffle for each angle type. Step 8: for the two strongest clusters
        // (which Step 11 splits into Table 7.5-5 delay sub-clusters by ray index) the
        // coupling must stay within each sub-cluster ray set; same equality test as the
        // CIR builders so coupling scope always matches the actual split.
        bool splitCluster = (nRayPerCluster == 20) &&
                            (n == strongest2clustersIdx[0] || n == strongest2clustersIdx[1]);
        if (splitCluster) {
            shuffleWithinSubClusters(idxASA, gen);
            shuffleWithinSubClusters(idxASD, gen);
            shuffleWithinSubClusters(idxZSA, gen);
            shuffleWithinSubClusters(idxZSD, gen);
        } else {
            std::shuffle(idxASA.begin(), idxASA.end(), gen);
            std::shuffle(idxASD.begin(), idxASD.end(), gen);
            std::shuffle(idxZSA.begin(), idxZSA.end(), gen);
            std::shuffle(idxZSD.begin(), idxZSD.end(), gen);
        }
        
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
// theta_ZOA/phi_AOA are panel-local (orientation-corrected) for the pattern and array phase;
// theta_ZOA_gcs/phi_AOA_gcs are the uncorrected GCS angles for the eq 7.5-25 Doppler term,
// where r_rx and the (GCS, Cartesian vx/vy/vz) UT velocity must share one frame.
inline cuComplex calculateRayCoefficient(
    const AntPanelConfig& utAntConfig, int ueAntIdx, float theta_ZOA, float phi_AOA, float psi_rx,
    const AntPanelConfig& bsAntConfig, int bsAntIdx, float theta_ZOD, float phi_AOD, float psi_tx,
    float theta_ZOA_gcs, float phi_AOA_gcs,
    float xpr, float * randomPhase, float currentTime, const float * utVelocity, float lambda_0)
{
    // Compute d_bar_rx (Rx antenna element position)
    // Assume antSize = [M_g, N_g, M, N, P] and antSpacing = [d_g_h, d_g_v, d_h, d_v]
    // 38.901 Fig 7.3-1: panel in the y-z plane (column n -> y*d_H, row m -> z*d_V)
    int M = utAntConfig.antSize[2];
    int N = utAntConfig.antSize[3];
    int P = utAntConfig.antSize[4];
    int p_rx = ueAntIdx % P;
    float d_h_rx = utAntConfig.antSpacing[2];
    float d_v_rx = utAntConfig.antSpacing[3];
    float d_bar_rx[3] = { 0.0f, (ueAntIdx / P) % N * d_h_rx, (ueAntIdx / (N * P)) % M * d_v_rx };

    // Convert angles to radians
    float d2pi = M_PI / 180.0f;
    float theta_ZOA_rad = theta_ZOA * d2pi;
    float phi_AOA_rad = phi_AOA * d2pi;
    float theta_ZOD_rad = theta_ZOD * d2pi;
    float phi_AOD_rad = phi_AOD * d2pi;

    // Calculate field patterns for Rx
    float F_rx_theta, F_rx_phi, F_tx_theta, F_tx_phi;
    calculateFieldComponents(utAntConfig, theta_ZOA, phi_AOA, utAntConfig.antPolarAngles[p_rx], F_rx_theta, F_rx_phi);
    // eq 7.1-11: rotate the LCS field components into the GCS spherical basis
    {
        float c_psi = cosf(psi_rx), s_psi = sinf(psi_rx);
        float Ft = c_psi * F_rx_theta - s_psi * F_rx_phi;
        float Fp = s_psi * F_rx_theta + c_psi * F_rx_phi;
        F_rx_theta = Ft;
        F_rx_phi = Fp;
    }

    // Compute d_bar_tx (Tx antenna element position), y-z plane as above
    M = bsAntConfig.antSize[2];
    N = bsAntConfig.antSize[3];
    P = bsAntConfig.antSize[4];
    int p_tx = bsAntIdx % P;
    float d_h_tx = bsAntConfig.antSpacing[2];
    float d_v_tx = bsAntConfig.antSpacing[3];
    float d_bar_tx[3] = { 0.0f, (bsAntIdx / P) % N * d_h_tx, (bsAntIdx / (N * P)) % M * d_v_tx };

    // Calculate field patterns for Tx
    calculateFieldComponents(bsAntConfig, theta_ZOD, phi_AOD, bsAntConfig.antPolarAngles[p_tx], F_tx_theta, F_tx_phi);
    // eq 7.1-11: rotate the LCS field components into the GCS spherical basis
    {
        float c_psi = cosf(psi_tx), s_psi = sinf(psi_tx);
        float Ft = c_psi * F_tx_theta - s_psi * F_tx_phi;
        float Fp = s_psi * F_tx_theta + c_psi * F_tx_phi;
        F_tx_theta = Ft;
        F_tx_phi = Fp;
    }

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

    // Term 6: Doppler effect per eq 7.5-25: r_rx (from GCS ray angles) dot GCS velocity
    float theta_gcs_rad = theta_ZOA_gcs * d2pi;
    float phi_gcs_rad = phi_AOA_gcs * d2pi;
    float sin_zoa_gcs = sinf(theta_gcs_rad);
    float doppler_phase = 2.0f * M_PI
        * (sin_zoa_gcs * cosf(phi_gcs_rad) * utVelocity[0]
         + sin_zoa_gcs * sinf(phi_gcs_rad) * utVelocity[1]
         + cosf(theta_gcs_rad)             * utVelocity[2])
        * currentTime / lambda_0;
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
// theta_LOS_*/phi_LOS_* are panel-local (orientation-corrected); the *_gcs pair carries the
// uncorrected GCS arrival angles for the eq 7.5-25 Doppler term (GCS Cartesian velocity).
inline cuComplex calculateLOSCoefficient(
    const AntPanelConfig& utAntConfig, int ueAntIdx, float theta_LOS_ZOA, float phi_LOS_AOA, float psi_rx,
    const AntPanelConfig& bsAntConfig, int bsAntIdx, float theta_LOS_ZOD, float phi_LOS_AOD, float psi_tx,
    float theta_LOS_ZOA_gcs, float phi_LOS_AOA_gcs,
    float currentTime, const float* utVelocity, float lambda_0, float d_3d)
{
    // Compute d_bar_rx (Rx antenna element position)
    // 38.901 Fig 7.3-1: panel in the y-z plane (column n -> y*d_H, row m -> z*d_V)
    int M = utAntConfig.antSize[2];
    int N = utAntConfig.antSize[3];
    int P = utAntConfig.antSize[4];
    int p_rx = ueAntIdx % P;
    float d_h_rx = utAntConfig.antSpacing[2];
    float d_v_rx = utAntConfig.antSpacing[3];
    float d_bar_rx[3] = { 0.0f, (ueAntIdx / P) % N * d_h_rx, (ueAntIdx / (N * P)) % M * d_v_rx };

    // Convert angles to radians
    float d2pi = M_PI / 180.0f;
    float theta_LOS_ZOA_rad = theta_LOS_ZOA * d2pi;
    float phi_LOS_AOA_rad = phi_LOS_AOA * d2pi;
    float theta_LOS_ZOD_rad = theta_LOS_ZOD * d2pi;
    float phi_LOS_AOD_rad = phi_LOS_AOD * d2pi;

    // Calculate field patterns for Rx
    float F_rx_theta, F_rx_phi, F_tx_theta, F_tx_phi;
    calculateFieldComponents(utAntConfig, theta_LOS_ZOA, phi_LOS_AOA, utAntConfig.antPolarAngles[p_rx], F_rx_theta, F_rx_phi);
    // eq 7.1-11: rotate the LCS field components into the GCS spherical basis
    {
        float c_psi = cosf(psi_rx), s_psi = sinf(psi_rx);
        float Ft = c_psi * F_rx_theta - s_psi * F_rx_phi;
        float Fp = s_psi * F_rx_theta + c_psi * F_rx_phi;
        F_rx_theta = Ft;
        F_rx_phi = Fp;
    }

    // Compute d_bar_tx (Tx antenna element position), y-z plane as above
    M = bsAntConfig.antSize[2];
    N = bsAntConfig.antSize[3];
    P = bsAntConfig.antSize[4];
    int p_tx = bsAntIdx % P;
    float d_h_tx = bsAntConfig.antSpacing[2];
    float d_v_tx = bsAntConfig.antSpacing[3];
    float d_bar_tx[3] = { 0.0f, (bsAntIdx / P) % N * d_h_tx, (bsAntIdx / (N * P)) % M * d_v_tx };
    // Calculate field patterns for Tx
    calculateFieldComponents(bsAntConfig, theta_LOS_ZOD, phi_LOS_AOD, bsAntConfig.antPolarAngles[p_tx], F_tx_theta, F_tx_phi);
    // eq 7.1-11: rotate the LCS field components into the GCS spherical basis
    {
        float c_psi = cosf(psi_tx), s_psi = sinf(psi_tx);
        float Ft = c_psi * F_tx_theta - s_psi * F_tx_phi;
        float Fp = s_psi * F_tx_theta + c_psi * F_tx_phi;
        F_tx_theta = Ft;
        F_tx_phi = Fp;
    }

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

    // Term 6: Doppler effect per eq 7.5-25: r_rx (from GCS LOS angles) dot GCS velocity
    float theta_gcs_rad = theta_LOS_ZOA_gcs * d2pi;
    float phi_gcs_rad = phi_LOS_AOA_gcs * d2pi;
    float sin_zoa_gcs = sinf(theta_gcs_rad);
    float doppler_phase = 2.0f * M_PI
        * (sin_zoa_gcs * cosf(phi_gcs_rad) * utVelocity[0]
         + sin_zoa_gcs * sinf(phi_gcs_rad) * utVelocity[1]
         + cosf(theta_gcs_rad)             * utVelocity[2])
        * currentTime / lambda_0;
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
                          m_cmnLinkParams.C_theta_LOS,   // signature order is (C_theta_LOS, C_theta_NLOS, C_theta_O2I)
                          m_cmnLinkParams.C_theta_NLOS,
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
                        m_clusterParams[linkIdx].strongest2clustersIdx,
                        m_gen,
                        m_uniformDist);
            
            // generate XPR and random phases
            for (uint16_t clusterIdx = 0; clusterIdx < m_clusterParams[linkIdx].nCluster; clusterIdx++) {
                for (uint16_t rayIdx = 0; rayIdx < m_clusterParams[linkIdx].nRayPerCluster; rayIdx++) {
                    // Generate XPR values
                                m_clusterParams[linkIdx].xpr[clusterIdx * m_clusterParams[linkIdx].nRayPerCluster + rayIdx] = std::pow(10.0f, (m_cmnLinkParams.mu_XPR[lspIdx] +
                                                          m_cmnLinkParams.sigma_XPR[lspIdx] * m_normalDist(m_gen)) / 10.0f);
                
                    // Generate random phases (3GPP Step 10): drawn INDEPENDENTLY per co-sited
                    // sector into per-sector blocks; Steps 1-9 (clusters, XPR) stay shared per site.
                    const uint32_t rayBase = (clusterIdx * m_clusterParams[linkIdx].nRayPerCluster + rayIdx) * 4;
                    for (uint32_t s = 0; s < m_topology.n_sector_per_site && s < ClusterParams::MAX_SECTORS; ++s) {
                        const uint32_t off = s * ClusterParams::PHASE_SECTOR_STRIDE + rayBase;
                        m_clusterParams[linkIdx].randomPhases[off]     = (m_uniformDist(m_gen) - 0.5f) * 2 * M_PI;
                        m_clusterParams[linkIdx].randomPhases[off + 1] = (m_uniformDist(m_gen) - 0.5f) * 2 * M_PI;
                        m_clusterParams[linkIdx].randomPhases[off + 2] = (m_uniformDist(m_gen) - 0.5f) * 2 * M_PI;
                        m_clusterParams[linkIdx].randomPhases[off + 3] = (m_uniformDist(m_gen) - 0.5f) * 2 * M_PI;
                    }
                }
            }
        }
    }
}


template <typename Tscalar, typename Tcomplex>
void slsChan<Tscalar, Tcomplex>::generateCIR()
{
    // Calculate time offset between snapshots: slot duration = 1 ms / (scs / 15 kHz),
    // split evenly across the n_snapshot_per_slot intra-slot snapshots
    const float timeOffset = 1e-3f * 15e3f / (m_simConfig->sc_spacing_hz * m_simConfig->n_snapshot_per_slot);

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

        // UT velocity in GCS Cartesian components (vx, vy, vz), as the Doppler term expects
        const float utVelocity[3] = {
            m_topology.utParams[uid].velocity[0],
            m_topology.utParams[uid].velocity[1],
            m_topology.utParams[uid].velocity[2]
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
                    
                    // Map the GCS LOS angles into each panel's LCS per eq 7.1-7/-8 and get the
                    // eq 7.1-15 polarization rotation (cell-specific for different sectors!)
                    // Orientation convention: [1] = bearing alpha, [0] = downtilt beta, [2] = roll gamma
                    float theta_LOS_ZOD, phi_LOS_AOD, psi_tx;
                    gcsToLcs(cellAntPanelOrientation[1], cellAntPanelOrientation[0], cellAntPanelOrientation[2],
                             m_linkParams[lspReadIdx].theta_LOS_ZOD, m_linkParams[lspReadIdx].phi_LOS_AOD,
                             theta_LOS_ZOD, phi_LOS_AOD, psi_tx);
                    float theta_LOS_ZOA, phi_LOS_AOA, psi_rx;
                    gcsToLcs(utAntPanelOrientation[1], utAntPanelOrientation[0] - 90.0f, utAntPanelOrientation[2],
                             m_linkParams[lspReadIdx].theta_LOS_ZOA, m_linkParams[lspReadIdx].phi_LOS_AOA,
                             theta_LOS_ZOA, phi_LOS_AOA, psi_rx);
                    const float d_3d = m_linkParams[lspReadIdx].d3d;

                    // Set first tap with path loss + antenna patterns
                    // The corrected angles above ensure different sectors get different antenna gains
                    for (uint16_t utAntIdx = 0; utAntIdx < nUtAnt; utAntIdx++) {
                        for (uint16_t bsAntIdx = 0; bsAntIdx < nCellAnt; bsAntIdx++) {
                            // Calculate deterministic antenna response using LOS component
                            // This includes: antenna field pattern + array response
                            const Tcomplex antennaResponse = calculateLOSCoefficient(
                                utAntPanelConfig, utAntIdx, theta_LOS_ZOA, phi_LOS_AOA, psi_rx,
                                cellAntPanelConfig, bsAntIdx, theta_LOS_ZOD, phi_LOS_AOD, psi_tx,
                                m_linkParams[lspReadIdx].theta_LOS_ZOA, m_linkParams[lspReadIdx].phi_LOS_AOA,
                                snapshotTime, utVelocity, m_cmnLinkParams.lambda_0, d_3d
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
                    // Each ray keeps the plain eq 7.5-28 amplitude sqrt(P_n/M); the
                    // 10/20-6/20-4/20 sub-cluster power split arises from the ray-set sizes
                    // alone, so no extra per-sub-cluster amplitude factor may be applied.
                    for (int subClusterIdx = 0; subClusterIdx < m_cmnLinkParams.nSubCluster; ++subClusterIdx) {
                        const uint16_t* rays = nullptr;
                        int nRays = 0;
                        if (subClusterIdx == 0) {
                            rays = m_cmnLinkParams.raysInSubCluster0;
                            nRays = m_cmnLinkParams.raysInSubClusterSizes[0];
                        } else if (subClusterIdx == 1) {
                            rays = m_cmnLinkParams.raysInSubCluster1;
                            nRays = m_cmnLinkParams.raysInSubClusterSizes[1];
                        } else if (subClusterIdx == 2) {
                            rays = m_cmnLinkParams.raysInSubCluster2;
                            nRays = m_cmnLinkParams.raysInSubClusterSizes[2];
                        }
                        for (int rayIdx = 0; rayIdx < nRays; ++rayIdx) {
                            // Get per-RAY angles (theta_n_m_* arrays); eq 7.1-7/-8/-15 map GCS
                            // -> panel LCS (orientation: [1]=bearing, [0]=downtilt, [2]=roll)
                            int rayGlobalIdx = clusterIdx * nRayPerCluster + rays[rayIdx];
                            float theta_ZOA_gcs = m_clusterParams[lspReadIdx].theta_n_m_ZOA[rayGlobalIdx];
                            float phi_AOA_gcs = m_clusterParams[lspReadIdx].phi_n_m_AoA[rayGlobalIdx];
                            float theta_ZOA, phi_AOA, psi_rx;
                            gcsToLcs(utAntPanelOrientation[1], utAntPanelOrientation[0] - 90.0f, utAntPanelOrientation[2],
                                     theta_ZOA_gcs, phi_AOA_gcs, theta_ZOA, phi_AOA, psi_rx);
                            float theta_ZOD, phi_AOD, psi_tx;
                            gcsToLcs(cellAntPanelOrientation[1], cellAntPanelOrientation[0], cellAntPanelOrientation[2],
                                     m_clusterParams[lspReadIdx].theta_n_m_ZOD[rayGlobalIdx],
                                     m_clusterParams[lspReadIdx].phi_n_m_AoD[rayGlobalIdx],
                                     theta_ZOD, phi_AOD, psi_tx);

                            // Add to channel matrix at the correct tap
                            for (uint16_t utAntIdx = 0; utAntIdx < nUtAnt; utAntIdx++) {
                                for (uint16_t bsAntIdx = 0; bsAntIdx < nCellAnt; bsAntIdx++) {
                                    // Calculate ray coefficient with snapshot time
                                    Tcomplex rayCoeff = calculateRayCoefficient(
                                        utAntPanelConfig, utAntIdx, theta_ZOA, phi_AOA, psi_rx,
                                        cellAntPanelConfig, bsAntIdx, theta_ZOD, phi_AOD, psi_tx,
                                        theta_ZOA_gcs, phi_AOA_gcs,
                                        m_clusterParams[lspReadIdx].xpr[rayGlobalIdx],
                                        m_clusterParams[lspReadIdx].randomPhases + (m_topology.n_sector_per_site > 0 ? (cid % m_topology.n_sector_per_site) % ClusterParams::MAX_SECTORS : 0) * ClusterParams::PHASE_SECTOR_STRIDE + rayGlobalIdx * 4,
                                        snapshotTime,
                                        utVelocity,
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
                            if (subClusterIdx == 1) {
                                clusterDelay += 1.28 * C_DS;
                            }
                            else if (subClusterIdx == 2) {
                                clusterDelay += 2.56 * C_DS;
                            }
                            if (m_sysConfig->enable_propagation_delay == 1) {
                                H_tapIdx[tapCount] = static_cast<uint16_t>(std::round((clusterDelay * 1e-9 + m_linkParams[lspReadIdx].d3d / 3.0e8f + m_linkParams[lspReadIdx].delta_tau) * m_simConfig->sc_spacing_hz * m_simConfig->fft_size));
                            }
                            else {
                                H_tapIdx[tapCount] = static_cast<uint16_t>(std::round(clusterDelay * 1e-9 * m_simConfig->sc_spacing_hz * m_simConfig->fft_size));
                            }
                        }
#ifdef SLS_DEBUG_
                        // Overflow means the NEXT dense slot index would reach N_MAX_TAPS.
                        // Dense index N_MAX_TAPS-1 (the 24th tap: 20 clusters + 2x2 extra
                        // sub-cluster slots) is legal, so compare post-increment against
                        // N_MAX_TAPS, not N_MAX_TAPS-1.
                        if (tapCount + 1 > N_MAX_TAPS) {
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
                        // Get per-RAY angles (theta_n_m_* arrays); eq 7.1-7/-8/-15 map GCS
                        // -> panel LCS (orientation: [1]=bearing, [0]=downtilt, [2]=roll)
                        int rayGlobalIdx = clusterIdx * nRayPerCluster + rayIdx;
                        float theta_ZOA_gcs = m_clusterParams[lspReadIdx].theta_n_m_ZOA[rayGlobalIdx];
                        float phi_AOA_gcs = m_clusterParams[lspReadIdx].phi_n_m_AoA[rayGlobalIdx];
                        float theta_ZOA, phi_AOA, psi_rx;
                        gcsToLcs(utAntPanelOrientation[1], utAntPanelOrientation[0] - 90.0f, utAntPanelOrientation[2],
                                 theta_ZOA_gcs, phi_AOA_gcs, theta_ZOA, phi_AOA, psi_rx);
                        float theta_ZOD, phi_AOD, psi_tx;
                        gcsToLcs(cellAntPanelOrientation[1], cellAntPanelOrientation[0], cellAntPanelOrientation[2],
                                 m_clusterParams[lspReadIdx].theta_n_m_ZOD[rayGlobalIdx],
                                 m_clusterParams[lspReadIdx].phi_n_m_AoD[rayGlobalIdx],
                                 theta_ZOD, phi_AOD, psi_tx);

                        // Add to channel matrix at the correct tap
                        for (uint16_t utAntIdx = 0; utAntIdx < nUtAnt; utAntIdx++) {
                            for (uint16_t bsAntIdx = 0; bsAntIdx < nCellAnt; bsAntIdx++) {
                            // Calculate ray coefficient with snapshot time
                            Tcomplex rayCoeff = calculateRayCoefficient(
                                utAntPanelConfig, utAntIdx, theta_ZOA, phi_AOA, psi_rx,
                                cellAntPanelConfig, bsAntIdx, theta_ZOD, phi_AOD, psi_tx,
                                theta_ZOA_gcs, phi_AOA_gcs,
                                m_clusterParams[lspReadIdx].xpr[rayGlobalIdx],
                                m_clusterParams[lspReadIdx].randomPhases + (m_topology.n_sector_per_site > 0 ? (cid % m_topology.n_sector_per_site) % ClusterParams::MAX_SECTORS : 0) * ClusterParams::PHASE_SECTOR_STRIDE + rayGlobalIdx * 4,
                                snapshotTime,
                                utVelocity,
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
                            H_tapIdx[tapCount] = static_cast<uint16_t>(std::round((clusterDelay * 1e-9 + m_linkParams[lspReadIdx].d3d / 3.0e8f + m_linkParams[lspReadIdx].delta_tau) * m_simConfig->sc_spacing_hz * m_simConfig->fft_size));
                        }
                        else {
                            H_tapIdx[tapCount] = static_cast<uint16_t>(std::round(clusterDelay * 1e-9 * m_simConfig->sc_spacing_hz * m_simConfig->fft_size));
                        }
                    }
#ifdef SLS_DEBUG_
                    // Overflow means the NEXT dense slot index would reach N_MAX_TAPS
                    // (index N_MAX_TAPS-1 is a legal write; see comment at the sub-cluster site)
                    if (tapCount + 1 > N_MAX_TAPS) {
                        printf("ERROR: tapCount (%d) exceeds N_MAX_TAPS (%d) limit. Cluster processing stopped to prevent buffer overflow.\n",
                               tapCount + 1, N_MAX_TAPS);
                        return;
                    }
#endif
                    tapCount++;
                }
            }

            // Handle LOS case if present (eq 7.5-30: the specular ray is added ONCE to the
            // first tap, outside the cluster loop)
            if (losInd && !isO2I) {
                // eq 7.1-7/-8/-15: map GCS LOS angles into each panel's LCS
                // (orientation: [1]=bearing alpha, [0]=downtilt beta, [2]=roll gamma)
                float theta_LOS_ZOA_corrected, phi_LOS_AOA_corrected, psi_rx_los;
                gcsToLcs(utAntPanelOrientation[1], utAntPanelOrientation[0] - 90.0f, utAntPanelOrientation[2],
                         m_linkParams[lspReadIdx].theta_LOS_ZOA, m_linkParams[lspReadIdx].phi_LOS_AOA,
                         theta_LOS_ZOA_corrected, phi_LOS_AOA_corrected, psi_rx_los);
                float theta_LOS_ZOD_corrected, phi_LOS_AOD_corrected, psi_tx_los;
                gcsToLcs(cellAntPanelOrientation[1], cellAntPanelOrientation[0], cellAntPanelOrientation[2],
                         m_linkParams[lspReadIdx].theta_LOS_ZOD, m_linkParams[lspReadIdx].phi_LOS_AOD,
                         theta_LOS_ZOD_corrected, phi_LOS_AOD_corrected, psi_tx_los);
                float los_scale = std::sqrt(K_R / (K_R + 1));

                // Calculate LOS component with snapshot time for each antenna pair
                for (uint16_t utAntIdx = 0; utAntIdx < nUtAnt; utAntIdx++) {
                    for (uint16_t bsAntIdx = 0; bsAntIdx < nCellAnt; bsAntIdx++) {
                        Tcomplex H_LOS = calculateLOSCoefficient(
                            utAntPanelConfig, utAntIdx,
                            theta_LOS_ZOA_corrected,
                            phi_LOS_AOA_corrected,
                            psi_rx_los,
                            cellAntPanelConfig, bsAntIdx,
                            theta_LOS_ZOD_corrected,
                            phi_LOS_AOD_corrected,
                            psi_tx_los,
                            m_linkParams[lspReadIdx].theta_LOS_ZOA,
                            m_linkParams[lspReadIdx].phi_LOS_AOA,
                            snapshotTime,
                            utVelocity,
                            m_cmnLinkParams.lambda_0,
                            m_linkParams[lspReadIdx].d3d
                        );
                        // Combine LOS and NLOS components at tap 0
                        // nlos already been scaled by 1 / (K_R + 1) in cluster power
                        H_link[(utAntIdx * nCellAnt + bsAntIdx) * N_MAX_TAPS] = cuCaddf(
                            make_cuComplex(los_scale * H_LOS.x, los_scale * H_LOS.y),
                            make_cuComplex(H_link[(utAntIdx * nCellAnt + bsAntIdx) * N_MAX_TAPS].x,
                                           H_link[(utAntIdx * nCellAnt + bsAntIdx) * N_MAX_TAPS].y));
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

// Host equivalents of the GPU CFR helpers (sls_chan_small_scale_GPU.cu), same layouts.
static inline uint32_t hostCalculateCfrOffset(uint16_t batchIdx, uint16_t ueAntIdx, uint16_t bsAntIdx,
                                              uint16_t prbgIdx, uint16_t nUtAnt, uint16_t nCellAnt,
                                              uint16_t N_Prbg, bool optionalCfrDim) {
    if (optionalCfrDim) {
        // Layout: [nActiveUtForThisCell, n_snapshot_per_slot, nPrbg, nUtAnt, nBsAnt]
        return ((batchIdx * N_Prbg + prbgIdx) * nUtAnt + ueAntIdx) * nCellAnt + bsAntIdx;
    } else {
        // Default layout: [nActiveUtForThisCell, n_snapshot_per_slot, nUtAnt, nBsAnt, nPrbg]
        return ((batchIdx * nUtAnt + ueAntIdx) * nCellAnt + bsAntIdx) * N_Prbg + prbgIdx;
    }
}

static inline uint32_t hostCalculateScCfrOffset(uint16_t batchIdx, uint16_t ueAntIdx, uint16_t bsAntIdx,
                                                uint16_t scIdx, uint16_t nUtAnt, uint16_t nCellAnt,
                                                uint16_t N_sc, bool optionalCfrDim) {
    if (optionalCfrDim) {
        // Layout: [nActiveUtForThisCell, n_snapshot_per_slot, nSc, nUtAnt, nBsAnt]
        return ((batchIdx * N_sc + scIdx) * nUtAnt + ueAntIdx) * nCellAnt + bsAntIdx;
    } else {
        // Default layout: [nActiveUtForThisCell, n_snapshot_per_slot, nUtAnt, nBsAnt, nSc]
        return ((batchIdx * nUtAnt + ueAntIdx) * nCellAnt + bsAntIdx) * N_sc + scIdx;
    }
}

template <typename Tcomplex>
static inline Tcomplex hostCalCfrbyCir(float freqKHz, uint16_t cirNtaps,
                                       const float* cirNormDelayUs2Pi,
                                       const Tcomplex* cirCoeff,
                                       float /* normalizationFactor: 1.0, matches GPU */) {
    Tcomplex cfr = {0.0f, 0.0f};
    for (uint16_t tapIdx = 0; tapIdx < cirNtaps; tapIdx++) {
        const float delay = cirNormDelayUs2Pi[tapIdx];
        const Tcomplex coeff = cirCoeff[tapIdx];
        const float phase = freqKHz * delay * 1e-3f; // kHz * 2*pi*delayUs * 1e-3 = rad
        if (fabsf(phase) > 1e6f) {
            continue; // match GPU kernel: skip taps with extreme phase values
        }
        const float cosPhase = cosf(phase);
        const float sinPhase = sinf(phase);
        // cirCoeff[tapIdx] * exp(-j * 2*pi*f*tau)
        cfr.x += coeff.x * cosPhase + coeff.y * sinPhase;
        cfr.y += coeff.y * cosPhase - coeff.x * sinPhase;
    }
    return cfr;
}

// Generate channel frequency response (CPU port of generateCFRKernel_runMode1/23).
// Converts each active link's CIR taps to CFR on subcarriers and/or PRBGs, writing
// through the same activeLink.freqChanSc / freqChanPrbg pointers as the GPU kernels
// (host memory in cpu_only_mode).
template <typename Tscalar, typename Tcomplex>
void slsChan<Tscalar, Tcomplex>::generateCFR() {
    if (m_activeLinkParams.empty()) {
        return;
    }
    if (m_simConfig->run_mode == 4) {
        throw std::runtime_error("CPU CFR generation for run_mode 4 (full N_FFT) not implemented.");
    }

    bool needScLevel = false;
    bool needPrbgLevel = false;
    for (const auto& activeLinkParam : m_activeLinkParams) {
        if (activeLinkParam.freqChanSc != nullptr) needScLevel = true;
        if (activeLinkParam.freqChanPrbg != nullptr) needPrbgLevel = true;
    }
    if (!needScLevel && !needPrbgLevel) {
        return;
    }

    const uint16_t N_Prbg = m_simConfig->n_prbg;
    const uint16_t N_sc = m_simConfig->n_prb * 12;
    const uint16_t N_sc_Prbg = (uint16_t)ceilf((float)(N_sc) / N_Prbg);
    const uint16_t N_sc_last_Prbg = N_sc - (N_Prbg - 1) * N_sc_Prbg;
    const uint16_t N_sc_over_2 = N_sc >> 1;
    const uint8_t freqConvertType = m_simConfig->freq_convert_type;
    const uint8_t scSampling = std::max<uint8_t>(1, m_simConfig->sc_sampling);
    const bool optionalCfrDim = (m_simConfig->optional_cfr_dim == 1);
    const float cfrNormalizationFactor = 1.0f; // no FFT energy normalization (matches GPU)

    std::vector<float> cirNormDelayUs2Pi(N_MAX_TAPS, 0.0f);
    std::vector<Tcomplex> timeChanLocal(N_MAX_TAPS);

    for (const auto& link : m_activeLinkParams) {
        if (link.freqChanSc == nullptr && link.freqChanPrbg == nullptr) {
            continue;
        }
        const uint16_t cid = link.cid;
        const uint16_t uid = link.uid;
        const uint32_t nUtAnt = (*m_antPanelConfig)[m_topology.utParams[uid].antPanelIdx].nAnt;
        const uint32_t nCellAnt = (*m_antPanelConfig)[m_topology.cellParams[cid].antPanelIdx].nAnt;
        // Element stride between consecutive SCs in freqChanSc: the optional layout is
        // [.., nSc, nUtAnt, nBsAnt] so neighboring SCs sit nUtAnt*nBsAnt apart, while the
        // default layout keeps SC innermost (stride 1)
        const uint32_t scStride = optionalCfrDim ? nUtAnt * nCellAnt : 1u;

        const uint16_t cirNtaps = std::min<uint16_t>(link.cirNtaps[0], N_MAX_TAPS);
        for (uint16_t tapIdx = 0; tapIdx < cirNtaps; tapIdx++) {
            const float delayUs = link.cirNormDelay[tapIdx] * 1e6f /
                                  (m_simConfig->sc_spacing_hz * m_simConfig->fft_size);
            cirNormDelayUs2Pi[tapIdx] = 2.0f * M_PI * delayUs;
        }

        for (uint16_t batchIdx = 0; batchIdx < m_simConfig->n_snapshot_per_slot; batchIdx++) {
            for (uint16_t ueAntIdx = 0; ueAntIdx < nUtAnt; ueAntIdx++) {
                for (uint16_t bsAntIdx = 0; bsAntIdx < nCellAnt; bsAntIdx++) {
                    const size_t cirOffset =
                        (size_t)batchIdx * nUtAnt * nCellAnt * N_MAX_TAPS +
                        (size_t)(ueAntIdx * nCellAnt + bsAntIdx) * N_MAX_TAPS;
                    for (uint16_t tapIdx = 0; tapIdx < cirNtaps; tapIdx++) {
                        // CFO rotation is identity (matches GPU placeholder)
                        timeChanLocal[tapIdx] = link.cirCoe[cirOffset + tapIdx];
                    }

                    for (uint16_t prbgIdx = 0; prbgIdx < N_Prbg; prbgIdx++) {
                        const uint32_t prbgOffset = hostCalculateCfrOffset(
                            batchIdx, ueAntIdx, bsAntIdx, prbgIdx, nUtAnt, nCellAnt, N_Prbg,
                            optionalCfrDim);
                        const uint16_t localScOffset = prbgIdx * N_sc_Prbg;
                        const uint16_t N_sc_current_Prbg =
                            (prbgIdx < N_Prbg - 1) ? N_sc_Prbg : N_sc_last_Prbg;

                        Tcomplex tempSum = {0.0f, 0.0f};
                        uint16_t sampledScCount = 0;
                        const bool accumulate =
                            link.freqChanPrbg != nullptr &&
                            freqConvertType == 3;
                        // Per-SC sweep: needed for the SC-level output and/or the
                        // averaging convert types; skipped otherwise (first/center/last
                        // SC types recompute their single SC directly below).
                        if (link.freqChanSc != nullptr || accumulate) {
                            const uint32_t scStartOffset = hostCalculateScCfrOffset(
                                batchIdx, ueAntIdx, bsAntIdx, localScOffset, nUtAnt, nCellAnt,
                                N_sc, optionalCfrDim);
                            for (uint16_t scInPrbgIdx = 0; scInPrbgIdx < N_sc_current_Prbg;
                                 scInPrbgIdx += scSampling) {
                                const float freqKHz =
                                    (localScOffset + scInPrbgIdx - N_sc_over_2) *
                                    m_simConfig->sc_spacing_hz * 1e-3f;
                                const Tcomplex cfrOnFreqKHz = hostCalCfrbyCir(
                                    freqKHz, cirNtaps, cirNormDelayUs2Pi.data(),
                                    timeChanLocal.data(), cfrNormalizationFactor);
                                if (link.freqChanSc != nullptr) {
                                    link.freqChanSc[scStartOffset + scInPrbgIdx * scStride] = cfrOnFreqKHz;
                                }
                                if (accumulate) {
                                    tempSum.x += cfrOnFreqKHz.x;
                                    tempSum.y += cfrOnFreqKHz.y;
                                    sampledScCount++;
                                }
                            }
                        }

                        if (link.freqChanPrbg == nullptr) {
                            continue;
                        }
                        uint16_t pickScIdx = 0;
                        switch (freqConvertType) {
                            case 0: // first SC of the PRBG
                                pickScIdx = 0;
                                break;
                            case 1: // center SC of the PRBG
                                pickScIdx = N_sc_current_Prbg / 2;
                                break;
                            case 2: // last SC of the PRBG
                                pickScIdx = N_sc_current_Prbg - 1;
                                break;
                            case 3:
                                link.freqChanPrbg[prbgOffset].x = tempSum.x / sampledScCount;
                                link.freqChanPrbg[prbgOffset].y = tempSum.y / sampledScCount;
                                continue;
                            case 4: // center SC after removing the within-PRBG ramp
                                pickScIdx = N_sc_current_Prbg / 2;
                                break;
                            default:
                                throw std::runtime_error("generateCFR: invalid freq_convert_type " +
                                                         std::to_string(freqConvertType));
                        }
                        if (link.freqChanSc != nullptr && pickScIdx % scSampling == 0) {
                            const uint32_t scStartOffset = hostCalculateScCfrOffset(
                                batchIdx, ueAntIdx, bsAntIdx, localScOffset, nUtAnt, nCellAnt,
                                N_sc, optionalCfrDim);
                            link.freqChanPrbg[prbgOffset] = link.freqChanSc[scStartOffset + pickScIdx * scStride];
                        } else {
                            const float freqKHz =
                                (localScOffset + pickScIdx - N_sc_over_2) *
                                m_simConfig->sc_spacing_hz * 1e-3f;
                            link.freqChanPrbg[prbgOffset] = hostCalCfrbyCir(
                                freqKHz, cirNtaps, cirNormDelayUs2Pi.data(),
                                timeChanLocal.data(), cfrNormalizationFactor);
                        }
                    }
                }
            }
        }
    }
}

// Explicit template instantiations
template class slsChan<float, float2>;
