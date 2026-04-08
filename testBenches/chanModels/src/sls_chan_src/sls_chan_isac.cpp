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
 * @file sls_chan_isac.cpp
 * @brief ISAC (Integrated Sensing and Communication) channel implementation
 * 
 * Implements target channel model per 3GPP TR 38.901 Section 7.9.
 */

#include "sls_chan_isac.hpp"
#include "sls_chan.cuh"
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <random>
#include <fstream>
#include <chrono>
#include <cstdint>
#include <array>
#include <limits>

// Forward declarations of small-scale helpers (defined in sls_chan_small_scale.cu)
void genClusterDelayAndPower(float delaySpread, float r_tao, uint8_t losInd,
                             uint16_t& nCluster, float K, float xi, uint8_t outdoor_ind,
                             float* delays, float* powers, uint16_t* strongest2clustersIdx,
                             std::mt19937& gen,
                             std::uniform_real_distribution<float>& uniformDist,
                             std::normal_distribution<float>& normalDist);


// Forward declaration for shared path loss helper implemented in sls_chan_large_scale.cu
float calPL(scenario_t scenario, bool isLos, float d_2d, float d_3d, float h_bs, float h_ut, float fc,
            bool optionalPlInd, std::mt19937& gen, std::uniform_real_distribution<float>& uniformDist,
            bool is_aerial);
float calLosProb(scenario_t scenario, float d_2d_out, float h_ut, const float force_los_prob[2],
                 uint8_t outdoor_ind, bool is_aerial);

// ============================================================================
// Target CIR Memory Management
// ============================================================================

// ============================================================================
// ISAC wrappers (member definitions)
// ============================================================================

// Initialize ISAC (monostatic RPs or bistatic links)
template <typename Tscalar, typename Tcomplex>
void slsChan<Tscalar, Tcomplex>::initializeIsacParams() {
    if (!m_sysConfig || m_sysConfig->isac_type == 0) {
        return;
    }
    m_targetChannelParams.isac_type = m_sysConfig->isac_type;
    
    if (m_sysConfig->isac_type == 1) {
        initializeMonostaticRefPoints();
    } else if (m_sysConfig->isac_type == 2) {
        initializeBistaticLinks();
    }
}

template <typename Tscalar, typename Tcomplex>
void slsChan<Tscalar, Tcomplex>::initializeMonostaticRefPoints() {
    auto& cellParams = m_topology.cellParams;
    const auto& antPanelConfigs = *m_antPanelConfig;
    const auto& sysConfig = *m_sysConfig;
    auto& refPoints = m_targetChannelParams.monostaticRefPoints;

    refPoints.clear();
    refPoints.reserve(cellParams.size());

    // Get Gamma distribution parameters for TRP monostatic background channel per 3GPP TR 38.901 Table 7.9.4.2-1
    const MonostaticRpGammaParams gammaParams = getTrpMonostaticGammaParams(sysConfig.scenario);

    // STX/SRX velocity for TRP monostatic (TRP is stationary)
    const float stx_srx_velocity[3] = {0.0f, 0.0f, 0.0f};

    // Track last generated site's RPs (cells are ordered by siteId)
    uint32_t lastSiteId = std::numeric_limits<uint32_t>::max();  // Invalid initial value
    MonostaticReferencePoint currentSiteRps[3];

    // For each cell, create monostatic reference points
    for (size_t bs_idx = 0; bs_idx < cellParams.size(); ++bs_idx) {
        const auto& cell = cellParams[bs_idx];

        // Generate new RPs only if this is a new site
        // TODO: check whether RP locations should be regenerated per panel (not shared across co-sited cells)
        if (cell.siteId != lastSiteId) {
            generateMonostaticReferencePoints(
                cell.loc,                      // Site location
                gammaParams,
                cell.antPanelOrientation,      // RP inherits orientation
                stx_srx_velocity,              // TRP is stationary
                currentSiteRps);
            lastSiteId = cell.siteId;
        }

        MonostaticReferencePoints ref;
        ref.bs_id = static_cast<uint32_t>(bs_idx);

        // STX reference point (TX antenna panel)
        ref.stx_loc = cell.loc;
        ref.stx_ant_panel_idx = cell.antPanelIdx;
        ref.stx_orientation[0] = cell.antPanelOrientation[0];  // Azimuth
        ref.stx_orientation[1] = cell.antPanelOrientation[1];  // Downtilt
        ref.stx_orientation[2] = cell.antPanelOrientation[2];  // Slant

        // SRX reference point (same or different antenna panel)
        // Per 3GPP TR 38.901, monostatic sensing can use same panel (full duplex) or different panel on same BS
        ref.same_antenna_panel = 1;  // Default: same panel
        ref.srx_loc = cell.loc;
        ref.srx_ant_panel_idx = cell.antPanelIdx;
        ref.srx_orientation[0] = cell.antPanelOrientation[0];
        ref.srx_orientation[1] = cell.antPanelOrientation[1];
        ref.srx_orientation[2] = cell.antPanelOrientation[2];

        // Distance between STX and SRX (0 if same panel)
        ref.stx_srx_distance = 0.0f;

        // Use the generated RPs for this site and align orientation to this cell
        // Note: LSP and clusters are generated later by generateRpClusters() after CRN is ready
        for (int i = 0; i < 3; ++i) {
            ref.background_rps[i] = currentSiteRps[i];
            ref.background_rps[i].orientation[0] = cell.antPanelOrientation[0];
            ref.background_rps[i].orientation[1] = cell.antPanelOrientation[1];
            ref.background_rps[i].orientation[2] = cell.antPanelOrientation[2];
            
            auto& rp = ref.background_rps[i];
            
            // Calculate distances
            rp.d_2d = calculateDistance2DTarget(cell.loc, rp.loc);
            rp.d_3d = calculateDistance3DTarget(cell.loc, rp.loc);
            
            // Calculate angles (Step 4: AOA = AOD for monostatic)
            calculateAngles3D(cell.loc, rp.loc, rp.los_zod, rp.los_aod);
            rp.los_zoa = rp.los_zod;  // Monostatic: arrival = departure
            rp.los_aoa = rp.los_aod;
        }

        refPoints.push_back(ref);
    }
}

template <typename Tscalar, typename Tcomplex>
void slsChan<Tscalar, Tcomplex>::generateRpClusters() {
    // Generate LSP and clusters for monostatic reference points
    // This must be called AFTER generateCRN() since calIsacLsp requires CRN tables
    
    if (m_sysConfig->isac_type != 1) {
        return;  // Only for monostatic
    }
    
    auto& refPoints = m_targetChannelParams.monostaticRefPoints;
    const auto& cellParams = m_topology.cellParams;
    const float fc = m_simConfig->center_freq_hz;
    const Scenario scenario = m_sysConfig->scenario;
    const bool optionalPlInd = (m_sysConfig->optional_pl_ind != 0);
    
    for (size_t bs_idx = 0; bs_idx < refPoints.size(); ++bs_idx) {
        auto& ref = refPoints[bs_idx];
        const auto& cell = cellParams[bs_idx];
        
        for (int i = 0; i < 3; ++i) {
            auto& rp = ref.background_rps[i];
            
            // Step 3: NLOS propagation condition for all RP channels
            const bool isLos = false;  // NLOS per spec
            const bool isIndoor = false;
            
            // Generate LSP for this RP (using existing calIsacLsp)
            float delta_tau_rp{};
            calIsacLsp(static_cast<scenario_t>(scenario), isLos, isIndoor, fc,
                      rp.d_2d, rp.d_3d, cell.loc.z, rp.loc.z,
                      rp.loc.x, rp.loc.y,
                      rp.delay_spread, rp.asd, rp.asa, rp.shadow_fading,
                      rp.ricean_k, rp.zsd, rp.zsa, delta_tau_rp,
                      cell.siteId);
            // Note: delta_tau for RPs is not currently used in RP background channel generation
            
            // Generate clusters/rays for this RP
            genTargetClustersFromLsp(
                rp.delay_spread, rp.asd, rp.asa, rp.zsd, rp.zsa, rp.ricean_k,
                rp.los_aoa, rp.los_aod, rp.los_zoa, rp.los_zod,
                isLos,
                rp.clusters);
            
            // Calculate pathloss (convert Hz to GHz for calPL)
            rp.pathloss = calPL(
                static_cast<scenario_t>(scenario), isLos,
                rp.d_2d, rp.d_3d, cell.loc.z, rp.loc.z,
                fc / 1e9f, optionalPlInd, m_gen, m_uniformDist, false);
        }
    }
}

template <typename Tscalar, typename Tcomplex>
void slsChan<Tscalar, Tcomplex>::initializeBistaticLinks() {
    auto& cellParams = m_topology.cellParams;
    auto& utParams = m_topology.utParams;
    const auto& antPanelConfigs = *m_antPanelConfig;
    const auto& sysConfig = *m_sysConfig;
    auto& refPoints = m_targetChannelParams.bistaticLinks;

    refPoints.clear();

    // Option 1: BS-to-BS bistatic (different BS as RX)
    for (size_t tx_idx = 0; tx_idx < cellParams.size(); ++tx_idx) {
        for (size_t rx_idx = 0; rx_idx < cellParams.size(); ++rx_idx) {
            if (tx_idx == rx_idx) {
                continue;  // Skip same BS
            }

            const auto& tx_cell = cellParams[tx_idx];
            const auto& rx_cell = cellParams[rx_idx];

            BistaticLinkEndpoints ref;

            // STX (transmitter BS)
            ref.stx_id = static_cast<uint32_t>(tx_idx);
            ref.stx_is_cell = 1;
            ref.stx_loc = tx_cell.loc;
            ref.stx_ant_panel_idx = tx_cell.antPanelIdx;
            ref.stx_orientation[0] = tx_cell.antPanelOrientation[0];
            ref.stx_orientation[1] = tx_cell.antPanelOrientation[1];
            ref.stx_orientation[2] = tx_cell.antPanelOrientation[2];

            // SRX (receiver BS)
            ref.srx_id = static_cast<uint32_t>(rx_idx);
            ref.srx_is_cell = 1;
            ref.srx_loc = rx_cell.loc;
            ref.srx_ant_panel_idx = rx_cell.antPanelIdx;
            ref.srx_orientation[0] = rx_cell.antPanelOrientation[0];
            ref.srx_orientation[1] = rx_cell.antPanelOrientation[1];
            ref.srx_orientation[2] = rx_cell.antPanelOrientation[2];

            // Baseline distance
            ref.baseline_distance = calculateDistance3DTarget(ref.stx_loc, ref.srx_loc);

            refPoints.push_back(ref);
        }
    }

    // Option 2: BS-to-UE bistatic (UE as sensing receiver)
    for (size_t tx_idx = 0; tx_idx < cellParams.size(); ++tx_idx) {
        for (size_t rx_idx = 0; rx_idx < utParams.size(); ++rx_idx) {
            const auto& tx_cell = cellParams[tx_idx];
            const auto& rx_ut = utParams[rx_idx];

            BistaticLinkEndpoints ref;

            // STX (transmitter BS)
            ref.stx_id = static_cast<uint32_t>(tx_idx);
            ref.stx_is_cell = 1;
            ref.stx_loc = tx_cell.loc;
            ref.stx_ant_panel_idx = tx_cell.antPanelIdx;
            ref.stx_orientation[0] = tx_cell.antPanelOrientation[0];
            ref.stx_orientation[1] = tx_cell.antPanelOrientation[1];
            ref.stx_orientation[2] = tx_cell.antPanelOrientation[2];

            // SRX (receiver UE)
            ref.srx_id = static_cast<uint32_t>(rx_idx);
            ref.srx_is_cell = 0;
            ref.srx_loc = rx_ut.loc;
            ref.srx_ant_panel_idx = rx_ut.antPanelIdx;
            ref.srx_orientation[0] = rx_ut.antPanelOrientation[0];
            ref.srx_orientation[1] = rx_ut.antPanelOrientation[1];
            ref.srx_orientation[2] = rx_ut.antPanelOrientation[2];

            // Baseline distance
            ref.baseline_distance = calculateDistance3DTarget(ref.stx_loc, ref.srx_loc);

            refPoints.push_back(ref);
        }
    }
}

void TargetCIR::allocate(uint32_t nTxAnt_, uint32_t nRxAnt_, uint32_t nSnapshots_, uint16_t nMaxTaps) {
    if (ownsMemory) {
        deallocate();
    }
    
    nTxAnt = nTxAnt_;
    nRxAnt = nRxAnt_;
    nSnapshots = nSnapshots_;
    
    const size_t cirSize = static_cast<size_t>(nSnapshots) * nRxAnt * nTxAnt * nMaxTaps;
    cirCoe = new cuComplex[cirSize];
    ownsMemory = true;
    cirNormDelay = new uint16_t[nMaxTaps];
    cirNtaps = new uint16_t[1];
    
    std::fill(cirCoe, cirCoe + cirSize, make_cuComplex(0.0f, 0.0f));
    std::fill(cirNormDelay, cirNormDelay + nMaxTaps, 0);
    cirNtaps[0] = 0;
}

void TargetCIR::deallocate() {
    if (ownsMemory) {
        delete[] cirCoe;
        delete[] cirNormDelay;
        delete[] cirNtaps;
        cirCoe = nullptr;
        cirNormDelay = nullptr;
        cirNtaps = nullptr;
        ownsMemory = false;
    }
}

// ============================================================================
// Geometric Calculations
// ============================================================================

template <typename Tscalar, typename Tcomplex>
void slsChan<Tscalar, Tcomplex>::calculateAngles3D(const Coordinate& loc1, const Coordinate& loc2, 
                       float& theta_ZOD, float& phi_AOD) {
    // Vector from loc1 to loc2
    const float dx = loc2.x - loc1.x;
    const float dy = loc2.y - loc1.y;
    const float dz = loc2.z - loc1.z;
    
    // 3D distance
    const float d3d = std::sqrt(dx * dx + dy * dy + dz * dz);
    
    if (d3d < 1e-6f) {
        // Points are essentially the same
        theta_ZOD = 90.0f;
        phi_AOD = 0.0f;
        return;
    }
    
    // Azimuth angle (in horizontal plane)
    phi_AOD = std::atan2(dy, dx) * 180.0f / M_PI;
    
    // Zenith angle (from vertical axis)
    // theta = acos(dz / d3d), where 0° is zenith (up), 90° is horizontal, 180° is nadir (down)
    theta_ZOD = std::acos(std::clamp(dz / d3d, -1.0f, 1.0f)) * 180.0f / M_PI;
}

template <typename Tscalar, typename Tcomplex>
float slsChan<Tscalar, Tcomplex>::calculateBistaticAngle(float theta_incident, float phi_incident,
                             float theta_scattered, float phi_scattered) {
    // Convert angles to radians
    const float theta_i_rad = theta_incident * M_PI / 180.0f;
    const float phi_i_rad = phi_incident * M_PI / 180.0f;
    const float theta_s_rad = theta_scattered * M_PI / 180.0f;
    const float phi_s_rad = phi_scattered * M_PI / 180.0f;
    
    // Convert to unit direction vectors
    // k_i: incident direction (TX → Target)
    const float k_i_x = std::sin(theta_i_rad) * std::cos(phi_i_rad);
    const float k_i_y = std::sin(theta_i_rad) * std::sin(phi_i_rad);
    const float k_i_z = std::cos(theta_i_rad);
    
    // k_s: scattered direction (Target → RX)
    const float k_s_x = std::sin(theta_s_rad) * std::cos(phi_s_rad);
    const float k_s_y = std::sin(theta_s_rad) * std::sin(phi_s_rad);
    const float k_s_z = std::cos(theta_s_rad);
    
    // Dot product: k_i · k_s
    const float dot_product = k_i_x * k_s_x + k_i_y * k_s_y + k_i_z * k_s_z;
    
    // Bistatic angle: β = acos(k_i · k_s)
    // Note: For forward scattering (same direction), β ≈ 0°
    //       For backscattering (opposite direction), β ≈ 180°
    const float bistatic_angle_rad = std::acos(std::clamp(dot_product, -1.0f, 1.0f));
    
    return bistatic_angle_rad * 180.0f / M_PI;
}

// RCS Constants for UAV per 3GPP TR 38.901 Table 7.9.2.1-1 and 7.9.2.2-1
// ============================================================================

// RCS Model 1: UAV with small size (Table 7.9.2.1-1)
constexpr float UAV_SMALL_SIGMA_M_DBSM = -12.81f;   // 10*log10(σ_M) in dBsm
constexpr float UAV_SMALL_SIGMA_S_DB = 3.74f;       // σ_σS_dB in dB

// XPR parameters for UAV (Table 7.9.2.2-1)
constexpr float UAV_XPR_MU_DB = 13.75f;             // μ_XPR in dB
constexpr float UAV_XPR_SIGMA_DB = 7.07f;           // σ_XPR in dB

// ============================================================================
// RCS Calculation (3GPP TR 38.901 Section 7.9.2.1)
// ============================================================================

/**
 * Calculate the stochastic component σ_S of RCS per 3GPP TR 38.901 Eq. 7.9.2-1
 * 
 * σ_S follows log-normal distribution where:
 * μ_σS_dB = -ln(10)/20 * σ_σS_dB²
 * 
 * @param sigma_sigma_s_db Standard deviation σ_σS_dB in dB
 * @return float Stochastic RCS component σ_S in linear scale
 */
template <typename Tscalar, typename Tcomplex>
float slsChan<Tscalar, Tcomplex>::calculateRcsSigmaS(float sigma_sigma_s_db) {
    // Per Eq. 7.9.2-1: μ_σS_dB = -ln(10)/20 * σ_σS_dB²
    const float mu_sigma_s_db = -std::log(10.0f) / 20.0f * sigma_sigma_s_db * sigma_sigma_s_db;
    
    // Generate 10*log10(σ_S) ~ N(μ_σS_dB, σ_σS_dB²)
    std::normal_distribution<float> normal_dist(mu_sigma_s_db, sigma_sigma_s_db);
    const float sigma_s_db = normal_dist(m_gen);
    
    // Convert to linear scale
    return std::pow(10.0f, sigma_s_db / 10.0f);
}

/**
 * Calculate XPR (Cross Polarization Ratio) for UAV per Table 7.9.2.2-1
 * 
 * @return float XPR κ in linear scale
 */
template <typename Tscalar, typename Tcomplex>
float slsChan<Tscalar, Tcomplex>::calculateUavXpr() {
    // XPR is log-Normal distributed: κ = 10^(X/10) where X ~ N(μ_XPR, σ_XPR²)
    std::normal_distribution<float> normal_dist(UAV_XPR_MU_DB, UAV_XPR_SIGMA_DB);
    const float xpr_db = normal_dist(m_gen);
    
    return std::pow(10.0f, xpr_db / 10.0f);
}

template <typename Tscalar, typename Tcomplex>
float slsChan<Tscalar, Tcomplex>::calculateRCS(const StParam& st, uint32_t spst_idx,
                   float theta_incident, float phi_incident,
                   float theta_scattered, float phi_scattered,
                   float bistatic_angle) {
    if (spst_idx >= st.n_spst || spst_idx >= st.spst_configs.size()) {
        throw std::out_of_range("SPST index out of range");
    }
    
    const SpstParam& spst = st.spst_configs[spst_idx];
    
    float sigma_RCS_linear;
    
    if (st.rcs_model == 1) {
        // ============================================================
        // RCS Model 1: UAV with small size / Human (Table 7.9.2.1-1)
        // ============================================================
        // Per 3GPP TR 38.901 Eq. 7.9.2-2:
        // σ_MD_dB(θ_i, φ_i, θ_s, φ_s) = max(10*lg(σ_M) - 3*sin(β/2), σ_FS)
        // where:
        //   β ∈ [0°, 180°] is the bistatic angle
        //   σ_FS is for forward scattering effect (set to -∞, i.e., not applicable)
        //
        // Total RCS: σ_RCS = σ_M * σ_D * σ_S
        // where σ_D = 1 for Model 1 (angular independent)
        
        const float sigma_M_dbsm = spst.rcs_sigma_m_dbsm;  // 10*lg(σ_M)
        
        // Bistatic attenuation: -3*sin(β/2) dB
        const float bistatic_angle_rad = bistatic_angle * M_PI / 180.0f;
        const float bistatic_attenuation_db = 3.0f * std::sin(bistatic_angle_rad / 2.0f);
        
        // σ_MD_dB = 10*lg(σ_M) - 3*sin(β/2)
        // Note: σ_FS = -∞ means we don't apply forward scattering floor
        const float sigma_MD_dbsm = sigma_M_dbsm - bistatic_attenuation_db;
        
        // σ_D = 1 for Model 1 (angular independent), so σ_MD = σ_M in linear
        // But we apply the bistatic attenuation to σ_M directly
        const float sigma_MD_linear = std::pow(10.0f, sigma_MD_dbsm / 10.0f);
        
        // Generate stochastic component σ_S
        const float sigma_S = calculateRcsSigmaS(spst.rcs_sigma_s_db);
        
        // Total RCS: σ_RCS = σ_MD * σ_S
        sigma_RCS_linear = sigma_MD_linear * sigma_S;
        
    } else if (st.rcs_model == 2) {
        // ============================================================
        // RCS Model 2: UAV with large size / Vehicle / AGV (Eq. 7.9.2-3)
        // ============================================================
        // Per 3GPP TR 38.901 Eq. 7.9.2-3:
        // σ_MD_dB(θ_i, φ_i, θ_s, φ_s) = max(G_max - min{-(σ^V_dB(θ) + σ^H_dB(φ)), σ_max}
        //                                    - k1*sin(k2*β/2) + 5*log10(cos(β/2)),
        //                                    G_max - σ_max,
        //                                    σ_FS)
        //
        // For UAV with large size: k1 = 6.05, k2 = 1.33
        // (θ, φ) are zenith/azimuth of the bisector of incident and scattered rays
        
        const float bistatic_angle_rad = bistatic_angle * M_PI / 180.0f;
        
        float theta_bisector = (theta_incident + theta_scattered) / 2.0f;
        // Use vector-based circular mean for azimuth to handle +/-180 wraparound
        float phi_bisector = std::atan2(
            std::sin(phi_incident * static_cast<float>(M_PI) / 180.0f) + std::sin(phi_scattered * static_cast<float>(M_PI) / 180.0f),
            std::cos(phi_incident * static_cast<float>(M_PI) / 180.0f) + std::cos(phi_scattered * static_cast<float>(M_PI) / 180.0f)
        ) * 180.0f / static_cast<float>(M_PI);
        
        // Normalize phi_bisector to [0, 360)
        while (phi_bisector < 0.0f) phi_bisector += 360.0f;
        while (phi_bisector >= 360.0f) phi_bisector -= 360.0f;
        
        // RCS pattern parameters for UAV with large size (Table 7.9.2.1-2)
        // Select parameters based on angular region
        float G_max, sigma_max, phi_center, phi_3dB, theta_center, theta_3dB;
        bool disable_sigma_H = false;  // Per NOTE: σ^H_dB(φ) = 0 for roof/bottom
        
        if (theta_bisector >= 0.0f && theta_bisector < 45.0f) {
            // Roof region: θ ∈ [0, 45)
            G_max = 13.55f; sigma_max = 20.42f;
            theta_center = 0.0f; theta_3dB = 4.93f;
            phi_center = 0.0f; phi_3dB = 1.0f;  // Not used (σ^H_dB = 0)
            disable_sigma_H = true;
        } else if (theta_bisector >= 135.0f && theta_bisector <= 180.0f) {
            // Bottom region: θ ∈ [135, 180]
            G_max = 13.55f; sigma_max = 20.42f;
            theta_center = 180.0f; theta_3dB = 4.93f;
            phi_center = 0.0f; phi_3dB = 1.0f;  // Not used (σ^H_dB = 0)
            disable_sigma_H = true;
        } else if (phi_bisector >= 45.0f && phi_bisector < 135.0f) {
            // Left region: θ ∈ [45, 135), φ ∈ [45, 135)
            G_max = 7.43f; sigma_max = 14.30f;
            phi_center = 90.0f; phi_3dB = 7.13f;
            theta_center = 90.0f; theta_3dB = 8.68f;
        } else if (phi_bisector >= 135.0f && phi_bisector < 225.0f) {
            // Back region: θ ∈ [45, 135), φ ∈ [135, 225)
            G_max = 3.99f; sigma_max = 10.86f;
            phi_center = 180.0f; phi_3dB = 10.09f;
            theta_center = 90.0f; theta_3dB = 11.43f;
        } else if (phi_bisector >= 225.0f && phi_bisector < 315.0f) {
            // Right region: θ ∈ [45, 135), φ ∈ [225, 315)
            G_max = 7.43f; sigma_max = 14.30f;
            phi_center = 270.0f; phi_3dB = 7.13f;
            theta_center = 90.0f; theta_3dB = 8.68f;
        } else {
            // Front region: θ ∈ [45, 135), φ ∈ [-45, 45) = [315, 360) ∪ [0, 45)
            G_max = 1.02f; sigma_max = 7.89f;
            phi_center = 0.0f; phi_3dB = 14.19f;
            theta_center = 90.0f; theta_3dB = 16.53f;
        }
        
        // k1, k2 parameters (per target type)
        const float k1 = 6.05f;   // UAV with large size
        const float k2 = 1.33f;   // UAV with large size
        
        // Calculate angular pattern attenuation
        // σ^V_dB(θ) = -min{12*((θ - θ_center)/θ_3dB)², σ_max}
        const float theta_term = std::pow((theta_bisector - theta_center) / theta_3dB, 2.0f);
        const float sigma_V_db = -std::min(12.0f * theta_term, sigma_max);
        
        // σ^H_dB(φ) = -min{12*((φ - φ_center)/φ_3dB)², σ_max}
        // Per NOTE: When θ ∈ [0,45] or [135,180], σ^H_dB(φ) = 0
        float sigma_H_db = 0.0f;
        if (!disable_sigma_H) {
            float phi_diff = phi_bisector - phi_center;
            // Normalize to [-180, 180]
            while (phi_diff > 180.0f) phi_diff -= 360.0f;
            while (phi_diff < -180.0f) phi_diff += 360.0f;
            const float phi_term = std::pow(phi_diff / phi_3dB, 2.0f);
            sigma_H_db = -std::min(12.0f * phi_term, sigma_max);
        }
        
        // Bistatic components
        const float bistatic_sin_term = k1 * std::sin(k2 * bistatic_angle_rad / 2.0f);
        const float cos_half_beta = std::cos(bistatic_angle_rad / 2.0f);
        const float bistatic_cos_term = (std::abs(cos_half_beta) > 1e-6f) ? 
                                        5.0f * std::log10(std::abs(cos_half_beta)) : -50.0f;
        
        // Combined σ_MD_dB per Eq. 7.9.2-3
        const float angular_pattern = G_max - std::min(-(sigma_V_db + sigma_H_db), sigma_max);
        const float sigma_MD_dbsm = std::max({
            angular_pattern - bistatic_sin_term + bistatic_cos_term,
            G_max - sigma_max,
            -100.0f  // σ_FS = -∞ (use -100 dBsm as floor)
        });
        
        const float sigma_MD_linear = std::pow(10.0f, sigma_MD_dbsm / 10.0f);
        
        // Generate stochastic component σ_S
        const float sigma_S = calculateRcsSigmaS(spst.rcs_sigma_s_db);
        
        // Total RCS: σ_RCS = σ_MD * σ_S
        sigma_RCS_linear = sigma_MD_linear * sigma_S;
        
    } else {
        throw std::invalid_argument("Invalid RCS model: must be 1 or 2");
    }
    
    return sigma_RCS_linear;
}

// ============================================================================
// Target Doppler Calculation
// ============================================================================

template <typename Tscalar, typename Tcomplex>
float slsChan<Tscalar, Tcomplex>::calculateTargetDoppler(const float target_velocity[3],
                             float theta_incident, float phi_incident,
                             float theta_scattered, float phi_scattered,
    float lambda_0) {
    // Convert angles to radians
    const float theta_i_rad = theta_incident * M_PI / 180.0f;
    const float phi_i_rad = phi_incident * M_PI / 180.0f;
    const float theta_s_rad = theta_scattered * M_PI / 180.0f;
    const float phi_s_rad = phi_scattered * M_PI / 180.0f;
    
    // Unit vectors in incident and scattered directions
    // k_i: TX → Target
    const float k_i_x = std::sin(theta_i_rad) * std::cos(phi_i_rad);
    const float k_i_y = std::sin(theta_i_rad) * std::sin(phi_i_rad);
    const float k_i_z = std::cos(theta_i_rad);
    
    // k_s: Target → RX
    const float k_s_x = std::sin(theta_s_rad) * std::cos(phi_s_rad);
    const float k_s_y = std::sin(theta_s_rad) * std::sin(phi_s_rad);
    const float k_s_z = std::cos(theta_s_rad);
    
    // Target velocity vector
    const float vx = target_velocity[0];
    const float vy = target_velocity[1];
    const float vz = target_velocity[2];
    
    // Bistatic Doppler formula:
    // f_d = (v · (k_i - k_s)) / λ
    // where k_i points toward target (incident), k_s points away from target (scattered)
    const float doppler_component = (vx * (k_i_x - k_s_x) + 
                                     vy * (k_i_y - k_s_y) + 
                                     vz * (k_i_z - k_s_z)) / lambda_0;
    
    return doppler_component;
}

// ============================================================================
// Path Parameter Calculation
// ============================================================================

/**
 * Calculate LOS probability for ISAC link per 3GPP TR 38.901 Section 7.6.3.3
 * Uses same LOS probability models as communication channels
 * 
 * @param d_2d 2D distance in meters
 * @param h_target Target height in meters
 * @param scenario Network scenario
 * @return float LOS probability [0, 1]
 */
template <typename Tscalar, typename Tcomplex>
float slsChan<Tscalar, Tcomplex>::calculateIsacLosProb(float d_2d, float h_target,
                                                       const StParam& target, Scenario scenario) {
    // Derive LOS probability parameters from target attributes and system config.
    const float* force_los_prob = m_sysConfig->force_los_prob;  // system-level overrides (-1 uses model)
    const uint8_t outdoor_ind = target.outdoor_ind;             // 0: indoor, 1: outdoor
    const bool is_aerial = (target.target_type == SensingTargetType::UAV);
    const float losProb = calLosProb(static_cast<scenario_t>(scenario), d_2d, h_target,
                                     force_los_prob, outdoor_ind, is_aerial);
    return std::clamp(losProb, 0.0f, 1.0f);
}

// Member wrapper
template <typename Tscalar, typename Tcomplex>
void slsChan<Tscalar, Tcomplex>::calculateIncidentPath(
    const Coordinate& tx_loc,
    const Coordinate& target_loc,
    const StParam& target,
    float fc, Scenario scenario,
    TargetIncidentPath& incident,
    uint32_t crnSiteIdx) {
    // Calculate distances
    incident.d3d = calculateDistance3DTarget(tx_loc, target_loc);
    incident.d2d = calculateDistance2DTarget(tx_loc, target_loc);
    
    // Calculate departure angles (from TX perspective)
    calculateAngles3D(tx_loc, target_loc, incident.theta_ZOD, incident.phi_AOD);
    
    // Calculate arrival angles at target (incident direction)
    // These are opposite to departure angles from TX
    calculateAngles3D(target_loc, tx_loc, incident.theta_ZOA_i, incident.phi_AOA_i);
    
    // Step 2: Determine LOS/NLOS for STX-SPST link
    const float los_prob = calculateIsacLosProb(incident.d2d, target_loc.z, target, scenario);
    const bool need_new_los = m_updateLosState || !incident.los_initialized;
    if (need_new_los) {
        std::uniform_real_distribution<float> uniform_dist(0.0f, 1.0f);
        incident.los_ind = (uniform_dist(m_gen) < los_prob) ? 1 : 0;
        incident.los_initialized = true;
    }
    
    // Step 3: Calculate path loss (convert Hz to GHz for calPL)
    const bool is_aerial_target = (target.target_type == SensingTargetType::UAV);
    const bool optionalPlInd = (m_sysConfig->optional_pl_ind != 0);
    const float fc_ghz = fc / 1e9f;
    incident.pathloss = calPL(
        static_cast<scenario_t>(scenario),
        incident.los_ind != 0,
        incident.d2d, incident.d3d,
        tx_loc.z, target_loc.z,
        fc_ghz,
        optionalPlInd,
        m_gen, m_uniformDist,
        is_aerial_target);
    
    // Step 4: Generate full LSP set (SF/DS/ASD/ASA/ZSD/ZSA/K/DT) using shared model
    float ds{}, asd{}, asa{}, sf{}, k{}, zsd{}, zsa{}, delta_tau{};
    calIsacLsp(static_cast<scenario_t>(scenario),
               incident.los_ind != 0,
               target.outdoor_ind == 0,  // indoor_ind: true if indoor
               fc,
               incident.d2d, incident.d3d,
               tx_loc.z, target_loc.z,
               target_loc.x, target_loc.y,
               ds, asd, asa, sf, k, zsd, zsa, delta_tau,
               crnSiteIdx, is_aerial_target);
    incident.shadow_fading = sf;
    incident.delay_spread = ds;
    incident.asd = asd;
    incident.asa = asa;
    incident.zsd = zsd;
    incident.zsa = zsa;
    incident.ricean_k = k;
    incident.delta_tau = delta_tau;
}

template <typename Tscalar, typename Tcomplex>
void slsChan<Tscalar, Tcomplex>::calculateScatteredPath(
    const Coordinate& target_loc,
    const Coordinate& rx_loc,
    const StParam& target,
    float fc, Scenario scenario,
    TargetScatteredPath& scattered,
    bool is_monostatic,
    const TargetIncidentPath* incident_path,
    uint32_t crnSiteIdx) {
    const bool is_aerial_target = (target.target_type == SensingTargetType::UAV);
    // Calculate distances
    scattered.d3d = calculateDistance3DTarget(target_loc, rx_loc);
    scattered.d2d = calculateDistance2DTarget(target_loc, rx_loc);
    
    // Calculate departure angles from target (scattered direction)
    calculateAngles3D(target_loc, rx_loc, scattered.theta_ZOD_s, scattered.phi_AOD_s);
    
    // Calculate arrival angles at RX
    calculateAngles3D(rx_loc, target_loc, scattered.theta_ZOA, scattered.phi_AOA);
    
    if (is_monostatic && incident_path != nullptr) {
        scattered.los_ind = incident_path->los_ind;
        scattered.pathloss = incident_path->pathloss;
        scattered.shadow_fading = incident_path->shadow_fading;
        scattered.delay_spread = incident_path->delay_spread;
        scattered.asd = incident_path->asd;
        scattered.asa = incident_path->asa;
        scattered.zsd = incident_path->zsd;
        scattered.zsa = incident_path->zsa;
        scattered.ricean_k = incident_path->ricean_k;
        scattered.los_initialized = incident_path->los_initialized;
    } else {
        const float los_prob = calculateIsacLosProb(scattered.d2d, target_loc.z, target, scenario);
        const bool need_new_los = m_updateLosState || !scattered.los_initialized;
        if (need_new_los) {
            std::uniform_real_distribution<float> uniform_dist(0.0f, 1.0f);
            scattered.los_ind = (uniform_dist(m_gen) < los_prob) ? 1 : 0;
            scattered.los_initialized = true;
        }
        
        const bool optionalPlInd = (m_sysConfig->optional_pl_ind != 0);
        const float fc_ghz = fc / 1e9f;  // Convert Hz to GHz for calPL
        scattered.pathloss = calPL(
            static_cast<scenario_t>(scenario),
            scattered.los_ind != 0,
            scattered.d2d, scattered.d3d,
            target_loc.z, rx_loc.z,
            fc_ghz,
            optionalPlInd,
            m_gen, m_uniformDist,
            is_aerial_target);
        
        float ds{}, asd{}, asa{}, sf{}, k{}, zsd{}, zsa{}, delta_tau{};
        calIsacLsp(static_cast<scenario_t>(scenario),
                   scattered.los_ind != 0,
                   target.outdoor_ind == 0,  // indoor_ind: true if indoor
                   fc,
                   scattered.d2d, scattered.d3d,
                   target_loc.z, rx_loc.z,
                   target_loc.x, target_loc.y,
                   ds, asd, asa, sf, k, zsd, zsa, delta_tau,
                   crnSiteIdx, is_aerial_target);
        scattered.shadow_fading = sf;
        scattered.delay_spread = ds;
        scattered.asd = asd;
        scattered.asa = asa;
        scattered.zsd = zsd;
        scattered.zsa = zsa;
        scattered.ricean_k = k;
        scattered.delta_tau = delta_tau;
    }
}

// ============================================================================
// Target Reflection Coefficient
// ============================================================================

template <typename Tscalar, typename Tcomplex>
cuComplex slsChan<Tscalar, Tcomplex>::calculateTargetReflectionCoefficient(
    const AntPanelConfig& txAntConfig, uint32_t txAntIdx,
    const AntPanelConfig& rxAntConfig, uint32_t rxAntIdx,
    const TargetLinkParams& targetLink,
    float currentTime, float lambda_0) {
    
    // Per 3GPP TR 38.901 Eq. 7.9.4-14:
    // H_u,s^(k)(τ,t) = Σ_p (10^{-(PL_tx + PL_rx + SF_tx + SF_rx)/20} * √(4π*σ_M/λ_0²) * H_u,s^{(k,p)}(τ,t))
    //
    // Where:
    // - PL_tx, PL_rx: Path loss for STX-SPST and SPST-SRX links
    // - SF_tx, SF_rx: Shadow fading for both links
    // - σ_M: First component of RCS (deterministic mean)
    // - λ_0: Wavelength
    // - H_u,s^{(k,p)}: Channel impulse response including σ_D, σ_S effects
    
    // TODO: Implement full antenna pattern and array response
    // This is a simplified placeholder implementation
    
    // 1. TX antenna pattern (incident direction)
    // F_tx(θ_i, φ_i) - would use calculateFieldComponents()
    
    // 2. RX antenna pattern (scattered direction)
    // F_rx(θ_s, φ_s) - would use calculateFieldComponents()
    
    // 3. RCS scaling factor per Eq. 7.9.4-14: √(4π*σ_M/λ_0²)
    // Note: targetLink.rcs_linear includes σ_M * σ_D * σ_S
    // For Step 15, we use √(4π*σ_RCS/λ_0²) as the full RCS scaling
    const float rcs_scale = std::sqrt(4.0f * M_PI * targetLink.rcs_linear / (lambda_0 * lambda_0));
    
    // 4. Path loss scaling
    // Combined incident + scattered path loss: 10^{-(PL_tx + PL_rx)/20}
    const float total_pathloss_linear = std::pow(10.0f, -targetLink.total_pathloss / 20.0f);
    
    // 5. Bistatic path phase
    // φ_path = -2π * (d_incident + d_scattered) / λ
    const float total_distance = targetLink.incident.d3d + targetLink.scattered.d3d;
    const float path_phase = -2.0f * M_PI * total_distance / lambda_0;
    
    // 6. Target Doppler phase per Eq. 7.9.4-5
    // f_D = (r_rx^T * v_rx + r_k,p^T * v_k,p) / λ + (r_tx^T * v_tx + r_k,p^T * v_k,p) / λ
    // φ_doppler = 2π * ∫f_D dt ≈ 2π * f_d * t (for constant Doppler)
    const float doppler_phase = 2.0f * M_PI * targetLink.doppler_shift_hz * currentTime;
    
    // Combine phases
    const float total_phase = path_phase + doppler_phase;
    
    // Complex coefficient (simplified without full antenna patterns)
    const float magnitude = rcs_scale * total_pathloss_linear;
    
    return make_cuComplex(magnitude * std::cos(total_phase), 
                         magnitude * std::sin(total_phase));
}

// ============================================================================
// Target Location Update
// ============================================================================

template <typename Tscalar, typename Tcomplex>
Coordinate slsChan<Tscalar, Tcomplex>::transformSpstToGCS(const Coordinate& spst_loc_lcs,
                              const Coordinate& target_loc_gcs,
                              const float target_orientation[2]) {
    // Transform from target local coordinate system (LCS) to global coordinate system (GCS)
    // Using target orientation [azimuth, elevation]
    
    const float azimuth_rad = target_orientation[0] * M_PI / 180.0f;
    const float elevation_rad = target_orientation[1] * M_PI / 180.0f;
    
    // Rotation matrix (simplified - assume azimuth rotation around z-axis)
    const float cos_az = std::cos(azimuth_rad);
    const float sin_az = std::sin(azimuth_rad);
    
    // Rotate SPST location
    Coordinate spst_gcs;
    spst_gcs.x = target_loc_gcs.x + (spst_loc_lcs.x * cos_az - spst_loc_lcs.y * sin_az);
    spst_gcs.y = target_loc_gcs.y + (spst_loc_lcs.x * sin_az + spst_loc_lcs.y * cos_az);
    spst_gcs.z = target_loc_gcs.z + spst_loc_lcs.z;  // Simplified (ignores elevation rotation)
    
    return spst_gcs;
}

template <typename Tscalar, typename Tcomplex>
void slsChan<Tscalar, Tcomplex>::updateAllTargetLocations(std::vector<StParam>& targets,
                             float current_time, float ref_time) {
    const float delta_t = current_time - ref_time;
    
    if (delta_t < 0.0f) {
        throw std::invalid_argument("current_time must be >= ref_time");
    }
    
    // Update each target's location
    for (auto& target : targets) {
        // Update target center position (straight-line movement)
        target.loc.x += target.velocity[0] * delta_t;
        target.loc.y += target.velocity[1] * delta_t;
        target.loc.z += target.velocity[2] * delta_t;
        
        // Note: SPST locations in LCS remain constant relative to target center
        // They are transformed to GCS when needed using transformSpstToGCS()
    }
}

// ============================================================================
// Channel Combination
// ============================================================================

template <typename Tscalar, typename Tcomplex>
void slsChan<Tscalar, Tcomplex>::combineBackgroundAndTargetCIR(
    const cuComplex* cirCoe_bg, const uint16_t* cirNormDelay_bg, const uint16_t* cirNtaps_bg,
    const cuComplex* cirCoe_tgt, const uint16_t* cirNormDelay_tgt, const uint16_t* cirNtaps_tgt,
    cuComplex* cirCoe_combined, uint16_t* cirNormDelay_combined, uint16_t* cirNtaps_combined,
    uint32_t nRxAnt, uint32_t nTxAnt, uint32_t nSnapshots,
    const IsacCirConfig& isacConfig) {
    
    // Get tap counts from config
    const uint16_t bg_stride = isacConfig.bg_stride;              // Background buffer stride (24 bistatic, 72 monostatic)
    const uint32_t total_max_taps = isacConfig.total_max_taps;    // Output buffer tap count
    
    // Get background dimensions from config
    // For monostatic: ISAC panels (64×64), for bistatic: comm link (4×64)
    const uint32_t bg_nRxAnt = isacConfig.bg_n_rx_ant;
    const uint32_t bg_nTxAnt = isacConfig.bg_n_tx_ant;
    
    // If background is disabled (calibration mode), just copy target CIR directly
    if (isacConfig.disable_background) {
        const uint16_t nTaps_tgt = cirNtaps_tgt[0];
        const uint16_t nTapsCopy = std::min(nTaps_tgt, static_cast<uint16_t>(total_max_taps));
        
        // Copy target delays and zero remaining
        for (uint32_t i = 0; i < total_max_taps; ++i) {
            cirNormDelay_combined[i] = (i < nTapsCopy) ? cirNormDelay_tgt[i] : 0;
        }
        cirNtaps_combined[0] = nTapsCopy;
        
        // Copy target coefficients
        const size_t total_size = static_cast<size_t>(nSnapshots) * nRxAnt * nTxAnt * total_max_taps;
        std::memcpy(cirCoe_combined, cirCoe_tgt, total_size * sizeof(cuComplex));
        return;
    }
    
    // If target is disabled (background-only calibration mode), just copy background CIR
    if (isacConfig.disable_target) {
        const uint16_t nTaps_bg = cirNtaps_bg[0];
        const uint16_t nTapsCopy = std::min(nTaps_bg, static_cast<uint16_t>(total_max_taps));
        
        // Copy background delays (may need to handle stride difference)
        for (uint32_t i = 0; i < total_max_taps; ++i) {
            cirNormDelay_combined[i] = (i < nTapsCopy) ? cirNormDelay_bg[i] : 0;
        }
        cirNtaps_combined[0] = nTapsCopy;
        
        // Copy background coefficients
        // Note: Background uses bg_nRxAnt × bg_nTxAnt × bg_stride layout
        // Output uses nRxAnt × nTxAnt × total_max_taps layout
        // For monostatic, these should match; for bistatic, we may need to adjust
        const size_t total_size = static_cast<size_t>(nSnapshots) * bg_nRxAnt * bg_nTxAnt * bg_stride;
        std::memcpy(cirCoe_combined, cirCoe_bg, total_size * sizeof(cuComplex));
        return;
    }
    
    // Strategy: Merge tap delay indices and add corresponding coefficients
    
    // 1. Create merged delay vector (sorted, unique)
    std::vector<uint16_t> merged_delays;
    merged_delays.reserve(total_max_taps);  // Reserve space for efficiency
    
    const uint16_t nTaps_bg = cirNtaps_bg[0];
    const uint16_t nTaps_tgt = cirNtaps_tgt[0];
    
    // Add background delays (input uses bg_stride)
    for (uint16_t i = 0; i < nTaps_bg && i < bg_stride; ++i) {
        merged_delays.push_back(cirNormDelay_bg[i]);
    }
    
    // Add target delays (all target taps are valid within total_max_taps stride)
    for (uint16_t i = 0; i < nTaps_tgt; ++i) {
        const uint16_t delay = cirNormDelay_tgt[i];
        if (std::find(merged_delays.begin(), merged_delays.end(), delay) == merged_delays.end()) {
            merged_delays.push_back(delay);
        }
    }
    
    // Sort merged delays
    std::sort(merged_delays.begin(), merged_delays.end());
    
    // Limit to total_max_taps (larger buffer for ISAC)
    if (merged_delays.size() > total_max_taps) {
        merged_delays.resize(total_max_taps);
    }
    
    // Store merged delays and zero-initialize remaining slots
    for (size_t i = 0; i < total_max_taps; ++i) {
        cirNormDelay_combined[i] = (i < merged_delays.size()) ? merged_delays[i] : 0;
    }
    cirNtaps_combined[0] = static_cast<uint16_t>(merged_delays.size());
    
    // 2. Initialize combined coefficients to zero (output uses total_max_taps)
    const size_t total_size = static_cast<size_t>(nSnapshots) * nRxAnt * nTxAnt * total_max_taps;
    std::fill(cirCoe_combined, cirCoe_combined + total_size, make_cuComplex(0.0f, 0.0f));
    
    // 3. Add background coefficients
    // Input buffer: uses bg_nRxAnt × bg_nTxAnt × bg_stride
    // Output buffer: uses nRxAnt × nTxAnt × total_max_taps
    // For monostatic: bg_nRxAnt == nRxAnt, bg_nTxAnt == nTxAnt (both 64×64)
    // For bistatic: dimensions may differ
    const uint32_t antLoopRx = std::min(bg_nRxAnt, nRxAnt);
    const uint32_t antLoopTx = std::min(bg_nTxAnt, nTxAnt);
    
    for (uint32_t snapshot = 0; snapshot < nSnapshots; ++snapshot) {
        for (uint32_t rxAnt = 0; rxAnt < antLoopRx; ++rxAnt) {
            for (uint32_t txAnt = 0; txAnt < antLoopTx; ++txAnt) {
                for (uint16_t tap_bg = 0; tap_bg < nTaps_bg && tap_bg < bg_stride; ++tap_bg) {
                    // Find corresponding tap in merged delays
                    const uint16_t delay = cirNormDelay_bg[tap_bg];
                    auto it = std::find(merged_delays.begin(), merged_delays.end(), delay);
                    if (it != merged_delays.end()) {
                        const uint16_t tap_merged = static_cast<uint16_t>(std::distance(merged_delays.begin(), it));
                        
                        // Input uses bg dimensions and bg_stride stride
                        const size_t idx_bg = snapshot * (bg_nRxAnt * bg_nTxAnt * bg_stride) + 
                                             (rxAnt * bg_nTxAnt + txAnt) * bg_stride + tap_bg;
                        // Output uses target dimensions and total_max_taps stride
                        const size_t idx_merged = snapshot * (nRxAnt * nTxAnt * total_max_taps) + 
                                                 (rxAnt * nTxAnt + txAnt) * total_max_taps + tap_merged;
                        
                        cirCoe_combined[idx_merged] = cuCaddf(cirCoe_combined[idx_merged], cirCoe_bg[idx_bg]);
                    }
                }
            }
        }
    }
    
    // 4. Add target coefficients (use target stride = total_max_taps)
    for (uint32_t snapshot = 0; snapshot < nSnapshots; ++snapshot) {
        for (uint32_t rxAnt = 0; rxAnt < nRxAnt; ++rxAnt) {
            for (uint32_t txAnt = 0; txAnt < nTxAnt; ++txAnt) {
                for (uint16_t tap_tgt = 0; tap_tgt < nTaps_tgt; ++tap_tgt) {
                    // Find corresponding tap in merged delays
                    const uint16_t delay = cirNormDelay_tgt[tap_tgt];
                    auto it = std::find(merged_delays.begin(), merged_delays.end(), delay);
                    if (it != merged_delays.end()) {
                        const uint16_t tap_merged = static_cast<uint16_t>(std::distance(merged_delays.begin(), it));
                        
                        // Target buffer uses total_max_taps stride
                        const size_t idx_tgt = snapshot * (nRxAnt * nTxAnt * total_max_taps) + 
                                              (rxAnt * nTxAnt + txAnt) * total_max_taps + tap_tgt;
                        // Output uses total_max_taps stride
                        const size_t idx_merged = snapshot * (nRxAnt * nTxAnt * total_max_taps) + 
                                                 (rxAnt * nTxAnt + txAnt) * total_max_taps + tap_merged;
                        
                        cirCoe_combined[idx_merged] = cuCaddf(cirCoe_combined[idx_merged], cirCoe_tgt[idx_tgt]);
                    }
                }
            }
        }
    }
}


// Explicit instantiation for current precision (float/FP32)
template void slsChan<float, cuComplex>::initializeIsacParams();
template void slsChan<float, cuComplex>::initializeMonostaticRefPoints();
template void slsChan<float, cuComplex>::initializeBistaticLinks();
template void slsChan<float, cuComplex>::generateMonostaticReferencePoints(
    const Coordinate& stx_srx_loc,
    const MonostaticRpGammaParams& params,
    const float stx_srx_orientation[3],
    const float stx_srx_velocity[3],
    MonostaticReferencePoint rps[3]);
template void slsChan<float, cuComplex>::generateMonostaticBackgroundCIR(
    const MonostaticBackgroundParams& bgParams,
    const AntPanelConfig& txAntConfig,
    const AntPanelConfig& rxAntConfig,
    float fc, float lambda_0,
    uint32_t nSnapshots,
    float currentTime,
    float sampleRate,
    TargetCIR& backgroundCIR);
template void slsChan<float, cuComplex>::calculateIncidentPath(
    const Coordinate& tx_loc,
    const Coordinate& target_loc,
    const StParam& target,
    float fc, Scenario scenario,
    TargetIncidentPath& incident,
    uint32_t crnSiteIdx);
template void slsChan<float, cuComplex>::calculateScatteredPath(
    const Coordinate& target_loc,
    const Coordinate& rx_loc,
    const StParam& target,
    float fc, Scenario scenario,
    TargetScatteredPath& scattered,
    bool is_monostatic,
    const TargetIncidentPath* incident_path,
    uint32_t crnSiteIdx);
template void slsChan<float, cuComplex>::calculateAllTargetLinkParams();
template void slsChan<float, cuComplex>::calculateAngles3D(
    const Coordinate& loc1, const Coordinate& loc2,
    float& theta_ZOD, float& phi_AOD);
template float slsChan<float, cuComplex>::calculateBistaticAngle(
    float theta_incident, float phi_incident,
    float theta_scattered, float phi_scattered);
template float slsChan<float, cuComplex>::calculateTargetDoppler(
    const float target_velocity[3],
    float theta_incident, float phi_incident,
    float theta_scattered, float phi_scattered,
    float lambda_0);
template float slsChan<float, cuComplex>::calculateIsacLosProb(
    float d_2d, float h_target, const StParam& target, Scenario scenario);
template void slsChan<float, cuComplex>::generateTargetCIR(
    const std::vector<TargetLinkParams>& targetLinks,
    const AntPanelConfig& txAntConfig,
    const AntPanelConfig& rxAntConfig,
    uint32_t nSnapshots,
    float currentTime,
    float lambda_0,
    float sampleRate,
    const IsacCirConfig& isacConfig,
    TargetCIR& targetCIR);

// Local helper: bilinear interpolation of CRN (mirrors large-scale helper)
namespace {
inline float getLspAtLocationIsac(float x, float y,
                                  float maxX, float minX,
                                  float maxY, float minY,
                                  const std::vector<std::vector<float>>& crn) {
    if (crn.empty() || crn[0].empty()) return 0.0f;
    const float rangeX = maxX - minX;
    const float rangeY = maxY - minY;
    if (rangeX < 1e-6f || rangeY < 1e-6f) return 0.0f;

    float normX = (x - minX) / rangeX;
    float normY = (y - minY) / rangeY;
    normX = std::clamp(normX, 0.0f, 1.0f);
    normY = std::clamp(normY, 0.0f, 1.0f);

    const int nX = static_cast<int>(crn.size());
    const int nY = static_cast<int>(crn[0].size());
    const float gridX = normX * static_cast<float>(nX - 1);
    const float gridY = normY * static_cast<float>(nY - 1);

    const int x0 = static_cast<int>(std::floor(gridX));
    const int y0 = static_cast<int>(std::floor(gridY));
    const int x1 = std::min(x0 + 1, nX - 1);
    const int y1 = std::min(y0 + 1, nY - 1);
    const float dx = gridX - static_cast<float>(x0);
    const float dy = gridY - static_cast<float>(y0);

    const float v00 = crn[x0][y0];
    const float v01 = crn[x0][y1];
    const float v10 = crn[x1][y0];
    const float v11 = crn[x1][y1];

    const float v0 = v00 * (1.0f - dy) + v01 * dy;
    const float v1 = v10 * (1.0f - dy) + v11 * dy;
    return v0 * (1.0f - dx) + v1 * dx;
}
}  // namespace

// ============================================================================
// ISAC LSP generation (mirrors comm LSP flow)
// ============================================================================
template <typename Tscalar, typename Tcomplex>
void slsChan<Tscalar, Tcomplex>::calIsacLsp(scenario_t scenario, bool isLos, bool isIndoor, float fc,
                                            float d_2d, float d_3d,
                                            float h_bs, float h_ut,
                                            float utX, float utY,
                                            float& ds, float& asd, float& asa, float& sf, float& k, float& zsd, float& zsa, float& delta_tau,
                                            uint32_t crnSiteIdx, bool is_aerial) {
    // Pick CRN based on site index and indoor/LOS state.

    const auto siteCount = static_cast<uint32_t>(m_crnLos.size());
    const uint32_t siteIdx = siteCount > 0 ? std::min(crnSiteIdx, siteCount - 1) : 0u;

    // Use target location to fetch CRN.
    const float locX = utX;
    const float locY = utY;

    const float r_SF  = getLspAtLocationIsac(locX, locY, m_maxX, m_minX, m_maxY, m_minY,
                                             isIndoor ? m_crnO2i[siteIdx][0] : (isLos ? m_crnLos[siteIdx][0] : m_crnNlos[siteIdx][0]));
    const float r_K   = isIndoor ? 0.0f
                                 : (isLos ? getLspAtLocationIsac(locX, locY, m_maxX, m_minX, m_maxY, m_minY, m_crnLos[siteIdx][1])
                                          : 0.0f);
    const float r_DS  = getLspAtLocationIsac(locX, locY, m_maxX, m_minX, m_maxY, m_minY,
                                             isIndoor ? m_crnO2i[siteIdx][1] : (isLos ? m_crnLos[siteIdx][2] : m_crnNlos[siteIdx][1]));
    const float r_ASD = getLspAtLocationIsac(locX, locY, m_maxX, m_minX, m_maxY, m_minY,
                                             isIndoor ? m_crnO2i[siteIdx][2] : (isLos ? m_crnLos[siteIdx][3] : m_crnNlos[siteIdx][2]));
    const float r_ASA = getLspAtLocationIsac(locX, locY, m_maxX, m_minX, m_maxY, m_minY,
                                             isIndoor ? m_crnO2i[siteIdx][3] : (isLos ? m_crnLos[siteIdx][4] : m_crnNlos[siteIdx][3]));
    const float r_ZSD = getLspAtLocationIsac(locX, locY, m_maxX, m_minX, m_maxY, m_minY,
                                             isIndoor ? m_crnO2i[siteIdx][4] : (isLos ? m_crnLos[siteIdx][5] : m_crnNlos[siteIdx][4]));
    const float r_ZSA = getLspAtLocationIsac(locX, locY, m_maxX, m_minX, m_maxY, m_minY,
                                             isIndoor ? m_crnO2i[siteIdx][5] : (isLos ? m_crnLos[siteIdx][6] : m_crnNlos[siteIdx][5]));
    
    // Delta Tau CRN: only read if enable_propagation_delay is enabled
    float r_DT = 0.0f;
    if (m_sysConfig->enable_propagation_delay != 0) {
        r_DT = getLspAtLocationIsac(locX, locY, m_maxX, m_minX, m_maxY, m_minY,
                                    isIndoor ? m_crnO2i[siteIdx][6] : (isLos ? m_crnLos[siteIdx][7] : m_crnNlos[siteIdx][6]));
    }

    float uncorrVars[LOS_MATRIX_SIZE] = {r_SF, r_K, r_DS, r_ASD, r_ASA, r_ZSD, r_ZSA};
    float corrVars[LOS_MATRIX_SIZE] = {0.0f};

    if (isIndoor) {
        for (int i = 0; i < O2I_MATRIX_SIZE; ++i) {
            for (int j = 0; j <= i; ++j) {
                const int src_i = (i >= K_IDX) ? i + 1 : i;
                const int src_j = (j >= K_IDX) ? j + 1 : j;
                corrVars[src_i] += m_cmnLinkParams.sqrtCorrMatO2i[i * O2I_MATRIX_SIZE + j] * uncorrVars[src_j];
            }
        }
        corrVars[K_IDX] = 0.0f;
    } else if (isLos) {
        for (int i = 0; i < LOS_MATRIX_SIZE; ++i) {
            for (int j = 0; j <= i; ++j) {
                corrVars[i] += m_cmnLinkParams.sqrtCorrMatLos[i * LOS_MATRIX_SIZE + j] * uncorrVars[j];
            }
        }
    } else {
        for (int i = 0; i < NLOS_MATRIX_SIZE; ++i) {
            for (int j = 0; j <= i; ++j) {
                const int src_i = (i >= K_IDX) ? i + 1 : i;
                const int src_j = (j >= K_IDX) ? j + 1 : j;
                corrVars[src_i] += m_cmnLinkParams.sqrtCorrMatNlos[i * NLOS_MATRIX_SIZE + j] * uncorrVars[src_j];
            }
        }
        corrVars[K_IDX] = 0.0f;
    }

    const uint8_t lspIdx = isIndoor ? 2 : (isLos ? 1 : 0);
    const float lgfc = m_cmnLinkParams.lgfc;

    const bool optionalPlInd = (m_sysConfig->optional_pl_ind != 0);
    // For aerial targets (UAV), use 3GPP TR 36.777 Table B-3 height-dependent SF std
    sf = corrVars[SF_IDX] * slsChan::calSfStd(static_cast<scenario_t>(scenario), isLos, fc, optionalPlInd, d_2d, h_bs, h_ut, is_aerial);

    const float muK = m_cmnLinkParams.mu_K[lspIdx];
    const float sigmaK = m_cmnLinkParams.sigma_K[lspIdx];
    k = (lspIdx == 1) ? corrVars[K_IDX] * sigmaK + muK : 0.0f;
    
    // Override K-factor for aerial targets if specified (e.g., Phase 2 UAV calibration uses K=15dB per TR 36.777 B.1.3)
    if (is_aerial && isLos && !std::isnan(m_sysConfig->st_override_k_db)) {
        k = m_sysConfig->st_override_k_db;
    }

    const float muDS = m_cmnLinkParams.mu_lgDS[lspIdx];
    const float sigmaDS = m_cmnLinkParams.sigma_lgDS[lspIdx];
    ds = std::pow(10.0f, corrVars[DS_IDX] * sigmaDS + muDS);

    const float muASD = m_cmnLinkParams.mu_lgASD[lspIdx];
    const float sigmaASD = m_cmnLinkParams.sigma_lgASD[lspIdx];
    const float muASA = m_cmnLinkParams.mu_lgASA[lspIdx];
    const float sigmaASA = m_cmnLinkParams.sigma_lgASA[lspIdx];
    asd = std::min(std::pow(10.0f, corrVars[ASD_IDX] * sigmaASD + muASD), 104.0f);
    asa = std::min(std::pow(10.0f, corrVars[ASA_IDX] * sigmaASA + muASA), 104.0f);

    float muZSD{};
    float sigmaZSD{};
    const float d2d_km = std::max(d_2d, 1.0f) / 1000.0f;
    switch (scenario) {
        case scenario_t::UMa:
            if (isLos) {
                muZSD = std::max(-0.5f, -2.1f * d2d_km - 0.01f * (h_ut - 1.5f) + 0.75f);
                sigmaZSD = 0.4f;
            } else {
                muZSD = std::max(-0.5f, -2.1f * d2d_km - 0.01f * (h_ut - 1.5f) + 0.9f);
                sigmaZSD = 0.49f;
            }
            break;
        case scenario_t::UMi:
            if (isLos) {
                muZSD = std::max(-0.21f, -14.8f * d2d_km - 0.01f * std::abs(h_ut - h_bs) + 0.83f);
                sigmaZSD = 0.35f;
            } else {
                muZSD = std::max(-0.5f, -3.1f * d2d_km + 0.01f * std::max(h_ut - h_bs, 0.0f) + 0.2f);
                sigmaZSD = 0.35f;
            }
            break;
        case scenario_t::RMa:
            if (isLos) {
                muZSD = std::max(-1.0f, -0.17f * d2d_km - 0.01f * (h_ut - 1.5f) + 0.22f);
                sigmaZSD = 0.34f;
            } else {
                muZSD = std::max(-1.0f, -0.19f * d2d_km - 0.01f * (h_ut - 1.5f) + 0.28f);
                sigmaZSD = 0.30f;
            }
            break;
        default:
            muZSD = -0.5f;
            sigmaZSD = 0.35f;
            break;
    }
    const float muZSA = m_cmnLinkParams.mu_lgZSA[lspIdx];
    const float sigmaZSA = m_cmnLinkParams.sigma_lgZSA[lspIdx];
    zsd = std::min(std::pow(10.0f, corrVars[ZSD_IDX] * sigmaZSD + muZSD + lgfc), 52.0f);
    zsa = std::min(std::pow(10.0f, corrVars[ZSA_IDX] * sigmaZSA + muZSA + lgfc), 52.0f);
    
    // Generate excess delay Δτ per 3GPP TR 38.901 Table 7.6.9-1
    // Only compute if enable_propagation_delay is enabled (saves computation)
    // LOS: Δτ = 0 (per Eq. 7.6-44)
    // NLOS: lg(Δτ) = log10(Δτ/1s) ~ N(mu_lg_DT, sigma_lg_DT)
    // Note: mu and sigma are time-invariant, so compute once using spatially correlated CRN
    if (m_sysConfig->enable_propagation_delay != 0) {
        if (isLos) {
            // LOS: Delta Tau = 0
            delta_tau = 0.0f;
        } else {
            // NLOS: Generate from lognormal distribution per Table 7.6.9-1
            float mu_lg_dt{}, sigma_lg_dt{};
            switch (scenario) {
                case scenario_t::UMi:
                    mu_lg_dt = -7.5f;
                    sigma_lg_dt = 0.5f;
                    break;
                case scenario_t::UMa:
                    mu_lg_dt = -7.4f;
                    sigma_lg_dt = 0.2f;
                    break;
                case scenario_t::RMa:
                    mu_lg_dt = -8.33f;
                    sigma_lg_dt = 0.26f;
                    break;
                default:
                    // Default to UMi
                    mu_lg_dt = -7.5f;
                    sigma_lg_dt = 0.5f;
                    break;
            }
            // lg(Delta Tau) = mu + sigma * r_DT (where r_DT is spatially correlated)
            const float lg_delta_tau = mu_lg_dt + sigma_lg_dt * r_DT;
            // Delta Tau in seconds = 10^(lg(Delta Tau))
            delta_tau = std::pow(10.0f, lg_delta_tau);
        }
    } else {
        // enable_propagation_delay disabled: skip delta_tau computation
        delta_tau = 0.0f;
    }
}
template cuComplex slsChan<float, cuComplex>::calculateTargetReflectionCoefficient(
    const AntPanelConfig& txAntConfig, uint32_t txAntIdx,
    const AntPanelConfig& rxAntConfig, uint32_t rxAntIdx,
    const TargetLinkParams& targetLink,
    float currentTime, float lambda_0);
template Coordinate slsChan<float, cuComplex>::transformSpstToGCS(
    const Coordinate& spst_loc_lcs,
    const Coordinate& target_loc_gcs,
    const float target_orientation[2]);
template void slsChan<float, cuComplex>::updateAllTargetLocations(
    std::vector<StParam>& targets,
    float current_time, float ref_time);
template void slsChan<float, cuComplex>::combineBackgroundAndTargetCIR(
    const cuComplex* cirCoe_bg, const uint16_t* cirNormDelay_bg, const uint16_t* cirNtaps_bg,
    const cuComplex* cirCoe_tgt, const uint16_t* cirNormDelay_tgt, const uint16_t* cirNtaps_tgt,
    cuComplex* cirCoe_combined, uint16_t* cirNormDelay_combined, uint16_t* cirNtaps_combined,
    uint32_t nRxAnt, uint32_t nTxAnt, uint32_t nSnapshots,
    const IsacCirConfig& isacConfig);
template MonostaticRpGammaParams slsChan<float, cuComplex>::getTrpMonostaticGammaParams(Scenario scenario);
template MonostaticRpGammaParams slsChan<float, cuComplex>::getUtMonostaticGammaParams(Scenario scenario, float h_ut, bool is_aerial);

// ============================================================================
// Target Link Parameter Calculation
// ============================================================================

template <typename Tscalar, typename Tcomplex>
void slsChan<Tscalar, Tcomplex>::genTargetClustersFromLsp(
    float delay_spread,
    float asd, float asa,
    float zsd, float zsa,
    float ricean_k,
    float phi_AOA, float phi_AOD,
    float theta_ZOA, float theta_ZOD,
    bool los,
    TargetClusterParams& out) {

    const uint8_t lspIdx = los ? 1 : 0;             // reuse LOS/NLOS bins (ignore O2I for targets)
    uint16_t nCluster = m_cmnLinkParams.nCluster[lspIdx];
    const uint16_t nRayPerCluster = m_cmnLinkParams.nRayPerCluster[lspIdx];

    out.n_clusters = nCluster;
    out.n_rays_per_cluster = nRayPerCluster;
    out.cluster_delays.assign(nCluster, 0.0f);
    out.cluster_powers.assign(nCluster, 0.0f);
    out.ray_phi_AOA.assign(nCluster, std::vector<float>(nRayPerCluster, 0.0f));
    out.ray_phi_AOD.assign(nCluster, std::vector<float>(nRayPerCluster, 0.0f));
    out.ray_theta_ZOA.assign(nCluster, std::vector<float>(nRayPerCluster, 0.0f));
    out.ray_theta_ZOD.assign(nCluster, std::vector<float>(nRayPerCluster, 0.0f));
    out.ray_rcs.assign(nCluster, std::vector<float>(nRayPerCluster, 1.0f));
    out.ray_xpr.assign(nCluster, std::vector<float>(nRayPerCluster, 0.0f));
    out.ray_phase_theta_theta.assign(nCluster, std::vector<float>(nRayPerCluster, 0.0f));
    out.ray_phase_theta_phi.assign(nCluster, std::vector<float>(nRayPerCluster, 0.0f));
    out.ray_phase_phi_theta.assign(nCluster, std::vector<float>(nRayPerCluster, 0.0f));
    out.ray_phase_phi_phi.assign(nCluster, std::vector<float>(nRayPerCluster, 0.0f));
    // SPST scattering phases (Eq. 7.9.4-6)
    out.spst_phase_theta_theta.assign(nCluster, std::vector<float>(nRayPerCluster, 0.0f));
    out.spst_phase_theta_phi.assign(nCluster, std::vector<float>(nRayPerCluster, 0.0f));
    out.spst_phase_phi_theta.assign(nCluster, std::vector<float>(nRayPerCluster, 0.0f));
    out.spst_phase_phi_phi.assign(nCluster, std::vector<float>(nRayPerCluster, 0.0f));
    out.spst_xpr.assign(nCluster, std::vector<float>(nRayPerCluster, 0.0f));
    out.ray_subcluster_delay.assign(nCluster, std::vector<float>(nRayPerCluster, 0.0f));

    // Working arrays sized like comm cluster generation
    std::vector<float> delays(nCluster, 0.0f);
    std::vector<float> powers(nCluster, 0.0f);
    std::vector<uint16_t> strongest2(2, 0);
    std::vector<float> phi_n_AoA(nCluster, 0.0f);
    std::vector<float> phi_n_AoD(nCluster, 0.0f);
    std::vector<float> theta_n_ZOD(nCluster, 0.0f);
    std::vector<float> theta_n_ZOA(nCluster, 0.0f);
    std::vector<float> phi_n_m_AoA(nCluster * nRayPerCluster, 0.0f);
    std::vector<float> phi_n_m_AoD(nCluster * nRayPerCluster, 0.0f);
    std::vector<float> theta_n_m_ZOD(nCluster * nRayPerCluster, 0.0f);
    std::vector<float> theta_n_m_ZOA(nCluster * nRayPerCluster, 0.0f);

    const float k_lin = ricean_k;
    const float k_db = (k_lin > 0.0f) ? 10.0f * std::log10(k_lin) : 0.0f;
    const float mu_offset_ZOD = 0.0f;  // no offset available for targets; default to 0
    // approximate mu_lgZSD so that (3/8)*10^(mu_lgZSD) ≈ zsd
    const float mu_lgZSD = std::log10(std::max(zsd * 8.0f / 3.0f, 1e-3f));

    genClusterDelayAndPower(delay_spread,
                            m_cmnLinkParams.r_tao[lspIdx],
                            static_cast<uint8_t>(los),
                            nCluster,
                            k_db,
                            m_cmnLinkParams.xi[lspIdx],
                            /*outdoor_ind=*/1,
                            delays.data(),
                            powers.data(),
                            strongest2.data(),
                            m_gen, m_uniformDist, m_normalDist);

    slsChan::genClusterAngle(nCluster,
                             m_cmnLinkParams.C_ASA[lspIdx],
                             m_cmnLinkParams.C_ASD[lspIdx],
                             m_cmnLinkParams.C_phi_NLOS,
                             m_cmnLinkParams.C_phi_LOS,
                             m_cmnLinkParams.C_phi_O2I,
                             m_cmnLinkParams.C_theta_NLOS,
                             m_cmnLinkParams.C_theta_LOS,
                             m_cmnLinkParams.C_theta_O2I,
                             asa,
                             asd,
                             zsa,
                             zsd,
                             phi_AOA,
                             phi_AOD,
                             theta_ZOA,
                             theta_ZOD,
                             mu_offset_ZOD,
                             static_cast<uint8_t>(los),
                             /*outdoor_ind=*/1,
                             k_db,
                             powers.data(),
                             phi_n_AoA.data(),
                             phi_n_AoD.data(),
                             theta_n_ZOD.data(),
                             theta_n_ZOA.data(),
                             m_gen,
                             m_uniformDist,
                             m_normalDist);

    const float C_ASA = m_cmnLinkParams.C_ASA[lspIdx];
    const float C_ASD = m_cmnLinkParams.C_ASD[lspIdx];
    const float C_ZSA = m_cmnLinkParams.C_ZSA[lspIdx];
    const float C_ZSD = (3.0f / 8.0f) * std::pow(10.0f, mu_lgZSD);

    slsChan::genRayAngle(nCluster,
                         nRayPerCluster,
                         phi_n_AoA.data(),
                         phi_n_AoD.data(),
                         theta_n_ZOD.data(),
                         theta_n_ZOA.data(),
                         phi_n_m_AoA.data(),
                         phi_n_m_AoD.data(),
                         theta_n_m_ZOD.data(),
                         theta_n_m_ZOA.data(),
                         C_ASA,
                         C_ASD,
                         C_ZSA,
                         C_ZSD,
                         m_gen,
                         m_uniformDist);

    for (uint16_t c = 0; c < nCluster; ++c) {
        out.cluster_delays[c] = delays[c];
        out.cluster_powers[c] = powers[c];
        out.ray_rcs[c].assign(nRayPerCluster, 1.0f);
        for (uint16_t r = 0; r < nRayPerCluster; ++r) {
            const uint32_t idx = c * nRayPerCluster + r;
            out.ray_phi_AOA[c][r] = phi_n_m_AoA[idx];
            out.ray_phi_AOD[c][r] = phi_n_m_AoD[idx];
            out.ray_theta_ZOD[c][r] = theta_n_m_ZOD[idx];
            out.ray_theta_ZOA[c][r] = theta_n_m_ZOA[idx];

            // XPR (log-normal)
            out.ray_xpr[c][r] = std::pow(10.0f,
                (m_cmnLinkParams.mu_XPR[lspIdx] +
                 m_cmnLinkParams.sigma_XPR[lspIdx] * m_normalDist(m_gen)) / 10.0f);

            // Random initial phase for CPM_tx/CPM_rx (Eq. 7.9.4-7/8)
            out.ray_phase_theta_theta[c][r] = (m_uniformDist(m_gen) - 0.5f) * 2.0f * M_PI;
            out.ray_phase_theta_phi[c][r]   = (m_uniformDist(m_gen) - 0.5f) * 2.0f * M_PI;
            out.ray_phase_phi_theta[c][r]   = (m_uniformDist(m_gen) - 0.5f) * 2.0f * M_PI;
            out.ray_phase_phi_phi[c][r]     = (m_uniformDist(m_gen) - 0.5f) * 2.0f * M_PI;
            
            // SPST scattering phases for CPM_spst (Eq. 7.9.4-6)
            // These are independent random phases representing polarization transformation at the target
            out.spst_phase_theta_theta[c][r] = (m_uniformDist(m_gen) - 0.5f) * 2.0f * M_PI;
            out.spst_phase_theta_phi[c][r]   = (m_uniformDist(m_gen) - 0.5f) * 2.0f * M_PI;
            out.spst_phase_phi_theta[c][r]   = (m_uniformDist(m_gen) - 0.5f) * 2.0f * M_PI;
            out.spst_phase_phi_phi[c][r]     = (m_uniformDist(m_gen) - 0.5f) * 2.0f * M_PI;
            // SPST XPR (log-normal, similar to ray XPR)
            out.spst_xpr[c][r] = std::pow(10.0f,
                (m_cmnLinkParams.mu_XPR[lspIdx] +
                 m_cmnLinkParams.sigma_XPR[lspIdx] * m_normalDist(m_gen)) / 10.0f);
        }
    }
    
    // Pre-compute NLOS ray coupling for time continuity (3GPP TR 38.901 Section 7.9.4 Step 9)
    // This shuffle is done ONCE per cluster set and reused across all snapshots
    out.nlos_ray_coupling.resize(nCluster);
    for (uint16_t c = 0; c < nCluster; ++c) {
        out.nlos_ray_coupling[c].resize(nRayPerCluster);
        // Initialize with identity mapping, then shuffle for NLOS rays
        for (uint16_t r = 0; r < nRayPerCluster; ++r) {
            out.nlos_ray_coupling[c][r] = r;
        }
        // Shuffle all rays except ray 0 (which is LOS in cluster 0 for LOS links)
        // The actual LOS check happens in generateTargetCIR, but pre-shuffle here
        if (nRayPerCluster > 1) {
            std::shuffle(out.nlos_ray_coupling[c].begin() + 1, 
                        out.nlos_ray_coupling[c].end(), m_gen);
        }
    }
    
    // Store strongest 2 cluster indices and compute sub-cluster delays (Eq. 7.5-26)
    // Per 3GPP TR 38.901 Table 7.5-6, sub-cluster delay offsets for UMa/UMi:
    // Sub-cluster 1 (rays 1,2,3,4,5,6,19,20): 0 ns
    // Sub-cluster 2 (rays 7,8,9,10,17,18): 5 ns
    // Sub-cluster 3 (rays 11,12,13,14,15,16): 10 ns
    out.strongest_clusters[0] = strongest2[0];
    out.strongest_clusters[1] = strongest2[1];
    
    // Initialize sub-cluster delays (0 for non-strongest clusters)
    out.ray_subcluster_delay.assign(nCluster, std::vector<float>(nRayPerCluster, 0.0f));
    
    // Sub-cluster delay offsets in seconds (from Table 7.5-6)
    constexpr float subcluster_offsets_ns[3] = {0.0f, 5.0f, 10.0f};
    // Ray indices for each sub-cluster (1-indexed in spec, 0-indexed here)
    // Sub-cluster 1: rays 0,1,2,3,4,5,18,19 (indices for 20-ray cluster)
    // Sub-cluster 2: rays 6,7,8,9,16,17
    // Sub-cluster 3: rays 10,11,12,13,14,15
    auto getSubclusterIndex = [](uint16_t rayIdx) -> int {
        // Map ray index to sub-cluster (0, 1, or 2)
        if (rayIdx <= 5 || rayIdx >= 18) return 0;  // Sub-cluster 1
        if ((rayIdx >= 6 && rayIdx <= 9) || (rayIdx >= 16 && rayIdx <= 17)) return 1;  // Sub-cluster 2
        return 2;  // Sub-cluster 3 (rays 10-15)
    };
    
    // Apply sub-cluster delays only to the 2 strongest clusters
    for (int sc = 0; sc < 2; ++sc) {
        const uint16_t c = strongest2[sc];
        if (c < nCluster) {
            for (uint16_t r = 0; r < nRayPerCluster; ++r) {
                const int subcluster = getSubclusterIndex(r);
                out.ray_subcluster_delay[c][r] = subcluster_offsets_ns[subcluster] * 1e-9f;
            }
        }
    }
}

template <typename Tscalar, typename Tcomplex>
void slsChan<Tscalar, Tcomplex>::calculateAllTargetLinkParams() {
    
    m_targetChannelParams.targetLinks.clear();
    
    const float c0 = 3e8f;  // Speed of light
    if (m_simConfig->center_freq_hz <= 0.0f) {
        throw std::invalid_argument("center_freq_hz must be positive for ISAC channel model");
    }
    const float lambda_0 = c0 / m_simConfig->center_freq_hz;
    
    if (m_targetChannelParams.isac_type == 1) {
        // Monostatic: BS -> Target -> same BS
        // Per 3GPP TR 38.901 Section 7.9.4.1 Step 4:
        // "For monostatic sensing mode, the large scale parameters generated in
        // step 2 to Step 4 are identical for a STX-SPST link and the corresponding
        // SPST-SRX link of the same SPST."
        constexpr bool is_monostatic = true;
        
        for (const auto& refPoint : m_targetChannelParams.monostaticRefPoints) {
            const Coordinate& tx_loc = refPoint.stx_loc;
            const Coordinate& rx_loc = refPoint.srx_loc;
            
            for (const auto& st : m_topology.stParams) {
                for (uint32_t spst_idx = 0; spst_idx < st.n_spst; ++spst_idx) {
                    // Transform SPST location to GCS
                    Coordinate spst_gcs = transformSpstToGCS(
                        st.spst_configs[spst_idx].loc_in_st_lcs,
                        st.loc,
                        st.orientation
                    );
                    
                    TargetLinkParams link;
                    link.tx_id = refPoint.bs_id;
                    link.rx_id = refPoint.bs_id;
                    link.tx_is_cell = 1;
                    link.rx_is_cell = 1;
                    link.st_id = st.sid;
                    link.spst_id = spst_idx;
                    link.target_loc = spst_gcs;
                    link.target_velocity[0] = st.velocity[0];
                    link.target_velocity[1] = st.velocity[1];
                    link.target_velocity[2] = st.velocity[2];
                    
                    // Calculate incident path (TX -> Target)
                    // This determines LOS/NLOS and LSPs for STX-SPST link
                    calculateIncidentPath(tx_loc, spst_gcs, st, m_simConfig->center_freq_hz, m_sysConfig->scenario, link.incident, refPoint.bs_id);
                    
                    // Calculate scattered path (Target -> RX)
                    // For monostatic: LSPs copied from incident path
                    calculateScatteredPath(spst_gcs, rx_loc, st, m_simConfig->center_freq_hz, m_sysConfig->scenario, link.scattered,
                                          is_monostatic, &link.incident, refPoint.bs_id);
                    
                    // Calculate bistatic angle and RCS
                    // For monostatic: bistatic angle β = 0° (TX and RX are co-located)
                    link.bistatic_angle = calculateBistaticAngle(
                        link.incident.theta_ZOA_i, link.incident.phi_AOA_i,
                        link.scattered.theta_ZOD_s, link.scattered.phi_AOD_s
                    );
                    
                    // Calculate RCS (Radar Cross Section)
                    // This is critical for target channel power scaling per Eq. 7.9.4-14
                    link.rcs_linear = calculateRCS(
                        st, spst_idx,
                        link.incident.theta_ZOA_i, link.incident.phi_AOA_i,
                        link.scattered.theta_ZOD_s, link.scattered.phi_AOD_s,
                        link.bistatic_angle
                    );
                    link.rcs_dbsm = (link.rcs_linear > 0.0f) ? 10.0f * std::log10(link.rcs_linear) : -200.0f;
                    
                    // Calculate target Doppler shift
                    link.doppler_shift_hz = calculateTargetDoppler(
                        link.target_velocity,
                        link.incident.theta_ZOA_i, link.incident.phi_AOA_i,
                        link.scattered.theta_ZOD_s, link.scattered.phi_AOD_s,
                        lambda_0
                    );
                    
                    // Total delay and path loss (include shadow fading per Eq. 7.9.4-14)
                    link.total_delay = (link.incident.d3d + link.scattered.d3d) / c0;
                    link.total_pathloss = link.incident.pathloss + link.scattered.pathloss
                                        + link.incident.shadow_fading + link.scattered.shadow_fading;

                    // Use spatially correlated excess delay Δτ from LSP generation (calIsacLsp)
                    // For monostatic: Δτ_rx = Δτ_tx per Eq. 7.9.4-2 note
                    // Both use the same spatially correlated value from incident link
                    link.delta_tau_tx = link.incident.delta_tau;
                    link.delta_tau_rx = link.incident.delta_tau;

                    // Precompute clusters/rays and store on link
                    genTargetClustersFromLsp(link.incident.delay_spread,
                                             link.incident.asd, link.incident.asa,
                                             link.incident.zsd, link.incident.zsa,
                                             link.incident.ricean_k,
                                             link.incident.phi_AOA_i, link.incident.phi_AOD,
                                             link.incident.theta_ZOA_i, link.incident.theta_ZOD,
                                             link.incident.los_ind != 0,
                                             link.incident.clusters);
                    // Monostatic per 7.9.4.1 Step 4: reuse incident clusters for scattered leg
                    link.scattered.clusters = link.incident.clusters;
                    
                    
                    m_targetChannelParams.targetLinks.push_back(link);
                }
            }
        }
    } else if (m_targetChannelParams.isac_type == 2) {
        // Bistatic: TX -> Target -> different RX
        // For bistatic mode, STX-SPST and SPST-SRX links have independent LOS/LSPs
        constexpr bool is_monostatic = false;
        
        for (const auto& refPoint : m_targetChannelParams.bistaticLinks) {
            Coordinate tx_loc = refPoint.stx_loc;
            Coordinate rx_loc = refPoint.srx_loc;
            
            for (const auto& st : m_topology.stParams) {
                for (uint32_t spst_idx = 0; spst_idx < st.n_spst; ++spst_idx) {
                    // Transform SPST location to GCS
                    Coordinate spst_gcs = transformSpstToGCS(
                        st.spst_configs[spst_idx].loc_in_st_lcs,
                        st.loc,
                        st.orientation
                    );
                    
                    TargetLinkParams link;
                    link.tx_id = refPoint.stx_id;
                    link.rx_id = refPoint.srx_id;
                    link.tx_is_cell = refPoint.stx_is_cell;
                    link.rx_is_cell = refPoint.srx_is_cell;
                    link.st_id = st.sid;
                    link.spst_id = spst_idx;
                    link.target_loc = spst_gcs;
                    link.target_velocity[0] = st.velocity[0];
                    link.target_velocity[1] = st.velocity[1];
                    link.target_velocity[2] = st.velocity[2];
                    
                    // Calculate incident path (TX -> Target)
                    // This determines LOS/NLOS and LSPs for STX-SPST link
                    calculateIncidentPath(tx_loc, spst_gcs, st, m_simConfig->center_freq_hz, m_sysConfig->scenario, link.incident, refPoint.stx_id);
                    
                    // Calculate scattered path (Target -> RX)
                    // For bistatic: LSPs are independent (not copied from incident)
                    calculateScatteredPath(spst_gcs, rx_loc, st, m_simConfig->center_freq_hz, m_sysConfig->scenario, link.scattered,
                                          is_monostatic, nullptr, refPoint.srx_id);
                    
                    // Calculate Doppler shift
                    link.doppler_shift_hz = calculateTargetDoppler(
                        link.target_velocity,
                        link.incident.theta_ZOA_i, link.incident.phi_AOA_i,
                        link.scattered.theta_ZOD_s, link.scattered.phi_AOD_s,
                        lambda_0
                    );
                    
                    // Total delay and path loss (include shadow fading per Eq. 7.9.4-14)
                    link.total_delay = (link.incident.d3d + link.scattered.d3d) / c0;
                    link.total_pathloss = link.incident.pathloss + link.scattered.pathloss
                                        + link.incident.shadow_fading + link.scattered.shadow_fading;

                    // Use spatially correlated excess delay Δτ from LSP generation (calIsacLsp)
                    // For bistatic: Δτ_tx and Δτ_rx are independent (from incident and scattered links)
                    link.delta_tau_tx = link.incident.delta_tau;
                    link.delta_tau_rx = link.scattered.delta_tau;

                    // Precompute clusters/rays and store on link
                    genTargetClustersFromLsp(link.incident.delay_spread,
                                             link.incident.asd, link.incident.asa,
                                             link.incident.zsd, link.incident.zsa,
                                             link.incident.ricean_k,
                                             link.incident.phi_AOA_i, link.incident.phi_AOD,
                                             link.incident.theta_ZOA_i, link.incident.theta_ZOD,
                                             link.incident.los_ind != 0,
                                             link.incident.clusters);
                    genTargetClustersFromLsp(link.scattered.delay_spread,
                                             link.scattered.asd, link.scattered.asa,
                                             link.scattered.zsd, link.scattered.zsa,
                                             link.scattered.ricean_k,
                                             link.scattered.phi_AOA, link.scattered.phi_AOD_s,
                                             link.scattered.theta_ZOA, link.scattered.theta_ZOD_s,
                                             link.scattered.los_ind != 0,
                                             link.scattered.clusters);

                    // Per-ray RCS using ray angles
                    const uint16_t nClusterPair = std::min(link.incident.clusters.n_clusters,
                                                           link.scattered.clusters.n_clusters);
                    const uint16_t nRayPair = std::min(link.incident.clusters.n_rays_per_cluster,
                                                       link.scattered.clusters.n_rays_per_cluster);
                    for (uint16_t c = 0; c < nClusterPair; ++c) {
                        for (uint16_t r = 0; r < nRayPair; ++r) {
                            const float bistatic_angle = calculateBistaticAngle(
                                link.incident.clusters.ray_theta_ZOA[c][r],
                                link.incident.clusters.ray_phi_AOA[c][r],
                                link.scattered.clusters.ray_theta_ZOD[c][r],
                                link.scattered.clusters.ray_phi_AOD[c][r]);
                            const float rcs_ray = calculateRCS(st, spst_idx,
                                                               link.incident.clusters.ray_theta_ZOA[c][r],
                                                               link.incident.clusters.ray_phi_AOA[c][r],
                                                               link.scattered.clusters.ray_theta_ZOD[c][r],
                                                               link.scattered.clusters.ray_phi_AOD[c][r],
                                                               bistatic_angle);
                            link.incident.clusters.ray_rcs[c][r] = rcs_ray;
                            link.scattered.clusters.ray_rcs[c][r] = rcs_ray;
                        }
                    }
                    
                    m_targetChannelParams.targetLinks.push_back(link);
                }
            }
        }
    }
}

// ============================================================================
// Target CIR Generation
// ============================================================================

template <typename Tscalar, typename Tcomplex>
void slsChan<Tscalar, Tcomplex>::generateTargetCIR(
    const std::vector<TargetLinkParams>& targetLinks,
    const AntPanelConfig& txAntConfig,
    const AntPanelConfig& rxAntConfig,
    uint32_t nSnapshots,
    float currentTime,
    float lambda_0,
    float sampleRate,
    const IsacCirConfig& isacConfig,
    TargetCIR& targetCIR) {
    
    const uint32_t nTxAnt = txAntConfig.nAnt;
    const uint32_t nRxAnt = rxAntConfig.nAnt;

    // Local helpers for antenna patterns/array response (mirrors comm link logic)
    auto wrapTheta = [](float theta) {
        float wrapped = std::fmod(theta, 360.0f);
        if (wrapped < 0.0f) {
            wrapped += 360.0f;
        }
        return (wrapped > 180.0f) ? 360.0f - wrapped : wrapped;
    };
    auto wrapPhi = [](float phi) {
        float wrapped = std::fmod(phi, 360.0f);
        if (wrapped < 0.0f) {
            wrapped += 360.0f;
        }
        return wrapped;
    };
    auto calcField = [&](const AntPanelConfig& cfg, float theta, float phi, float zeta, float& F_theta, float& F_phi) {
        constexpr float G_max = 8.0f;
        const int theta_idx = static_cast<int>(std::round(wrapTheta(theta)));
        const int phi_idx = static_cast<int>(std::round(wrapPhi(phi))) % 360;
        const float A_db_3D = cfg.antTheta[theta_idx] + cfg.antPhi[phi_idx] + (cfg.antModel == 1 ? G_max : 0.0f);
        const float A_3D_sqrt = std::pow(10.0f, A_db_3D / 20.0f);
        const float zeta_rad = zeta * static_cast<float>(M_PI) / 180.0f;
        F_theta = A_3D_sqrt * std::cos(zeta_rad);
        F_phi   = A_3D_sqrt * std::sin(zeta_rad);
    };
    auto elemPos = [](const AntPanelConfig& cfg, uint32_t antIdx) {
        const int M = static_cast<int>(cfg.antSize[2]);
        const int N = static_cast<int>(cfg.antSize[3]);
        const int P = static_cast<int>(cfg.antSize[4]);
        const float d_h = cfg.antSpacing[2];
        const float d_v = cfg.antSpacing[3];
        const int m = static_cast<int>((antIdx / (N * P)) % M);
        const int n = static_cast<int>((antIdx / P) % N);
        return std::array<float, 3>{m * d_h, n * d_v, 0.0f};
    };
    auto arrayPhase = [](const std::array<float, 3>& d_bar, float theta_deg, float phi_deg) {
        const float theta = theta_deg * static_cast<float>(M_PI) / 180.0f;
        const float phi   = phi_deg   * static_cast<float>(M_PI) / 180.0f;
        const float r_head[3] = {std::sin(theta) * std::cos(phi),
                                 std::sin(theta) * std::sin(phi),
                                 std::cos(theta)};
        const float phase = 2.0f * static_cast<float>(M_PI) *
                            (r_head[0] * d_bar[0] + r_head[1] * d_bar[1] + r_head[2] * d_bar[2]);
        return make_cuComplex(std::cos(phase), std::sin(phase));
    };
    
    // Tap budget: use config or fall back to a safe cap
    const uint16_t nMaxTaps = std::max<uint16_t>(1u, isacConfig.total_max_taps);
    
    // (Re)allocate target CIR if shape changed
    if (!targetCIR.ownsMemory || targetCIR.nTxAnt != nTxAnt || 
        targetCIR.nRxAnt != nRxAnt || targetCIR.nSnapshots != nSnapshots) {
        targetCIR.allocate(nTxAnt, nRxAnt, nSnapshots, nMaxTaps);
    }
    
    // Clear buffers
    const size_t totalSize = static_cast<size_t>(nSnapshots) * nRxAnt * nTxAnt * nMaxTaps;
    std::fill(targetCIR.cirCoe, targetCIR.cirCoe + totalSize, make_cuComplex(0.0f, 0.0f));
    std::fill(targetCIR.cirNormDelay, targetCIR.cirNormDelay + nMaxTaps, 0);
    targetCIR.cirNtaps[0] = 0;
    
    struct PathCandidate {
        uint16_t c_inc{};
        uint16_t r_inc{};
        uint16_t c_sca{};
        uint16_t r_sca{};
        float delay_tx{};
        float delay_rx{};
        float path_power{};
        float rcs{};           // Pre-calculated RCS for time continuity
        size_t link_idx{};     // Index into targetLinks
    };

    auto computeRayPower = [](bool linkLos, bool firstCluster, uint16_t rIdx,
                              float P_cluster, float K_lin, uint16_t nRays) -> float {
        // Per 3GPP TR 38.901 Section 7.9.4.2: power scaling for Ricean K-factor
        // LOS ray: P_LOS = K/(K+1) * P_cluster
        // NLOS ray m in LOS cluster: P = (1/(K+1)) * (P_cluster / M), where M = nRays
        if (linkLos && firstCluster && rIdx == 0) {
            // LOS ray gets K/(K+1) fraction of cluster power
            return (K_lin / (K_lin + 1.0f)) * P_cluster;
        }
        if (linkLos && firstCluster) {
            // NLOS rays in LOS cluster: (1/(K+1)) * (P_cluster / M)
            return (1.0f / (K_lin + 1.0f)) * (P_cluster / static_cast<float>(nRays));
        }
        // Non-LOS clusters: rays share power equally
        return P_cluster / static_cast<float>(nRays);
    };

    // ========================================================================
    // PHASE 1: Build coupled paths with pre-calculated RCS (time-continuous)
    // Ray coupling is done ONCE per link to ensure time continuity across snapshots
    // ========================================================================
    const float dropThreshDb = m_sysConfig->path_drop_threshold_db;
    std::vector<PathCandidate> allCandidates;
    
    for (size_t linkIdx = 0; linkIdx < targetLinks.size(); ++linkIdx) {
        const auto& link = targetLinks[linkIdx];
        const auto& incClusters = link.incident.clusters;
        const auto& scaClusters = link.scattered.clusters;

        const uint16_t nClusterPair = std::min(incClusters.n_clusters, scaClusters.n_clusters);
        // ricean_k is stored in dB (from calIsacLsp), convert to linear for ray power computation
        const float K_tx = (link.incident.ricean_k > 0.0f) ? std::pow(10.0f, link.incident.ricean_k / 10.0f) : 1.0f;
        const float K_rx = (link.scattered.ricean_k > 0.0f) ? std::pow(10.0f, link.scattered.ricean_k / 10.0f) : 1.0f;

        const uint16_t nRayIncMax = incClusters.n_rays_per_cluster;
        const uint16_t nRayScaMax = scaClusters.n_rays_per_cluster;

        for (uint16_t c = 0; c < nClusterPair; ++c) {
            const float delay_tx = incClusters.cluster_delays[c];
            const float delay_rx = scaClusters.cluster_delays[c];
            const float P_tx_cluster = incClusters.cluster_powers[c];
            const float P_rx_cluster = scaClusters.cluster_powers[c];

            const uint16_t nRayInc = incClusters.n_rays_per_cluster;
            const uint16_t nRaySca = scaClusters.n_rays_per_cluster;

            // Identify LOS rays
            const bool losIncRay = (link.incident.los_ind != 0) && (c == 0) && (nRayInc > 0);
            const bool losScaRay = (link.scattered.los_ind != 0) && (c == 0) && (nRaySca > 0);

            std::vector<uint16_t> incNlos, scaNlos;
            incNlos.reserve(nRayInc);
            scaNlos.reserve(nRaySca);
            for (uint16_t r = 0; r < nRayInc; ++r) {
                if (!(losIncRay && r == 0)) {
                    incNlos.push_back(r);
                }
            }
            for (uint16_t r = 0; r < nRaySca; ++r) {
                if (!(losScaRay && r == 0)) {
                    scaNlos.push_back(r);
                }
            }

            // Lambda to add a coupled path with pre-calculated RCS
            auto addPath = [&](uint16_t rInc, uint16_t rSca) {
                const float P_tx_ray = computeRayPower(link.incident.los_ind != 0, c == 0, rInc, P_tx_cluster, K_tx, nRayInc);
                const float P_rx_ray = computeRayPower(link.scattered.los_ind != 0, c == 0, rSca, P_rx_cluster, K_rx, nRaySca);
                const auto& st_ref = m_topology.stParams[link.st_id];
                const uint32_t spst_idx = link.spst_id;
                
                // Use pre-calculated RCS from ray_rcs if indices match, otherwise calculate
                float rcs_ray{};
                if (rInc == rSca && c < incClusters.ray_rcs.size() && rInc < incClusters.ray_rcs[c].size()) {
                    rcs_ray = incClusters.ray_rcs[c][rInc];
                } else {
                    const float bistatic_angle = calculateBistaticAngle(
                        incClusters.ray_theta_ZOA[c][rInc],
                        incClusters.ray_phi_AOA[c][rInc],
                        scaClusters.ray_theta_ZOD[c][rSca],
                        scaClusters.ray_phi_AOD[c][rSca]);
                    rcs_ray = calculateRCS(st_ref, spst_idx,
                                           incClusters.ray_theta_ZOA[c][rInc],
                                           incClusters.ray_phi_AOA[c][rInc],
                                           scaClusters.ray_theta_ZOD[c][rSca],
                                           scaClusters.ray_phi_AOD[c][rSca],
                                           bistatic_angle);
                }
                
                // Apply RCS scaling per 3GPP TR 38.901 Eq. 7.9.4-14:
                // Reflection coefficient includes √(4π*σ_RCS/λ²) factor
                const float rcs_scale = std::sqrt(4.0f * static_cast<float>(M_PI) * rcs_ray / (lambda_0 * lambda_0));
                const float path_power = rcs_scale * P_tx_ray * P_rx_ray;
                if (path_power <= 0.0f) {
                    return;
                }
                allCandidates.push_back(PathCandidate{c, rInc, c, rSca, delay_tx, delay_rx, path_power, rcs_ray, linkIdx});
            };

            // LOS-LOS
            if (losIncRay && losScaRay) {
                addPath(0, 0);
            }
            // LOS (inc) with all NLOS (sca)
            if (losIncRay) {
                for (uint16_t rSca : scaNlos) {
                    addPath(0, rSca);
                }
            }
            // LOS (sca) with all NLOS (inc)
            if (losScaRay) {
                for (uint16_t rInc : incNlos) {
                    addPath(rInc, 0);
                }
            }
            // NLOS-NLOS Option 2: use pre-computed coupling for time continuity
            // nlos_ray_coupling was generated once in genTargetClustersFromLsp
            if (!incNlos.empty() && !scaNlos.empty()) {
                const size_t pairCount = std::min(incNlos.size(), scaNlos.size());
                // Use pre-computed coupling: pair incident ray i with scattered ray using permutation
                const bool haveCoupling = (c < scaClusters.nlos_ray_coupling.size() &&
                                          !scaClusters.nlos_ray_coupling[c].empty());
                for (size_t idx = 0; idx < pairCount; ++idx) {
                    const uint16_t rInc = incNlos[idx];
                    uint16_t rSca{};
                    if (haveCoupling && rInc < scaClusters.nlos_ray_coupling[c].size()) {
                        // Use pre-computed permutation
                        rSca = scaClusters.nlos_ray_coupling[c][rInc];
                    } else {
                        // Fallback to sequential if coupling not available
                        rSca = scaNlos[idx];
                    }
                    addPath(rInc, rSca);
                }
            }
        }
    }

    // Path dropping based on max power across all paths
    float maxPowerAll = 0.0f;
    for (const auto& p : allCandidates) {
        if (p.path_power > maxPowerAll) {
            maxPowerAll = p.path_power;
        }
    }
    const float dropThreshLin = maxPowerAll * std::pow(10.0f, -dropThreshDb / 10.0f);

    // Filter paths that survive dropping
    std::vector<PathCandidate> survivingPaths;
    survivingPaths.reserve(allCandidates.size());
    for (const auto& p : allCandidates) {
        if (p.path_power >= dropThreshLin) {
            survivingPaths.push_back(p);
        }
    }

#ifdef SLS_DEBUG_
    printf("DEBUG: Pre-built %zu candidates, %zu survive after %.1f dB threshold (maxPower=%.6e)\n",
           allCandidates.size(), survivingPaths.size(), dropThreshDb, maxPowerAll);
#endif

    // ========================================================================
    // PHASE 2: Generate CIR coefficients per snapshot using pre-built paths
    // ========================================================================
    for (uint32_t snapshot = 0; snapshot < nSnapshots; ++snapshot) {
        const float snapshotTime = currentTime + static_cast<float>(snapshot) / sampleRate;
        uint16_t tapCount = targetCIR.cirNtaps[0];
        
#ifdef SLS_DEBUG_
        int keptPaths = 0;
        constexpr int kLogPaths = 5;
        int logged = 0;
        float loggedPower[kLogPaths]{};
        uint16_t loggedDelay[kLogPaths]{};
        uint16_t loggedCinc[kLogPaths]{};
        uint16_t loggedRinc[kLogPaths]{};
        uint16_t loggedCsca[kLogPaths]{};
        uint16_t loggedRsca[kLogPaths]{};
#endif

        for (const auto& path : survivingPaths) {
            const auto& link = targetLinks[path.link_idx];
            const auto& incClusters = link.incident.clusters;
            const auto& scaClusters = link.scattered.clusters;

            const uint16_t c = path.c_inc;

                // Per 3GPP TR 38.901 Eq. 7.9.4-2:
                // τ = τ_rx + d_rx/c + Δτ_rx + τ_tx + d_tx/c + Δτ_tx
                // where Δτ = 0 for LOS rays
                // Also includes sub-cluster delay offsets for 2 strongest clusters (Eq. 7.5-26)
                const bool isLosTxRay = (link.incident.los_ind != 0) && (path.c_inc == 0) && (path.r_inc == 0);
                const bool isLosRxRay = (link.scattered.los_ind != 0) && (path.c_sca == 0) && (path.r_sca == 0);
                // Δτ applies only to NLOS rays
                const float delta_tx = isLosTxRay ? 0.0f : link.delta_tau_tx;
                const float delta_rx = isLosRxRay ? 0.0f : link.delta_tau_rx;
                
                // Sub-cluster delay offsets for 2 strongest clusters
                const float subcluster_tx = incClusters.ray_subcluster_delay[path.c_inc][path.r_inc];
                const float subcluster_rx = scaClusters.ray_subcluster_delay[path.c_sca][path.r_sca];
                
                const float absDelay = link.total_delay + path.delay_tx + path.delay_rx 
                                      + delta_tx + delta_rx + subcluster_tx + subcluster_rx;
                const float delaySamplesF = absDelay * sampleRate;
                if (std::isnan(delaySamplesF) || delaySamplesF < 0.0f || 
                    delaySamplesF > static_cast<float>(std::numeric_limits<uint16_t>::max())) {
                    continue;
                }
                const uint16_t delaySamples = static_cast<uint16_t>(delaySamplesF);

#ifdef SLS_DEBUG_
                static int delayPrintCount = 0;
                if (snapshot == 0 && delayPrintCount < 60) {
                    printf("DEBUG delay: link=%zu c=%u r=%u/%u prop=%.3fus tau=%.3f/%.3fns delta=%.3f/%.3fus sub=%.1f/%.1fns abs=%.3fus samp=%u\n",
                           path.link_idx, path.c_inc, path.r_inc, path.r_sca,
                           link.total_delay * 1e6f, 
                           path.delay_tx * 1e9f, path.delay_rx * 1e9f,
                           delta_tx * 1e6f, delta_rx * 1e6f,
                           subcluster_tx * 1e9f, subcluster_rx * 1e9f,
                           absDelay * 1e6f, delaySamples);
                    ++delayPrintCount;
                }
#endif

                int tapPos = -1;
                for (uint16_t t = 0; t < tapCount; ++t) {
                    if (targetCIR.cirNormDelay[t] == delaySamples) {
                        tapPos = static_cast<int>(t);
                        break;
                    }
                }
                if (tapPos == -1) {
                    if (tapCount >= nMaxTaps) {
                        continue;
                    }
                    tapPos = static_cast<int>(tapCount);
                    targetCIR.cirNormDelay[tapCount++] = delaySamples;
#ifdef SLS_DEBUG_
                    if (snapshot == 0) {
                        printf("DEBUG new tap: tapPos=%d delaySamples=%u tapCount=%u\n", tapPos, delaySamples, tapCount);
                    }
#endif
                }

#ifdef SLS_DEBUG_
                    ++keptPaths;
                    if (logged < kLogPaths) {
                        loggedPower[logged] = path.path_power;
                        loggedDelay[logged] = delaySamples;
                        loggedCinc[logged] = path.c_inc;
                        loggedRinc[logged] = path.r_inc;
                        loggedCsca[logged] = path.c_sca;
                        loggedRsca[logged] = path.r_sca;
                        ++logged;
                    }
#endif

                // Per-ray Doppler per 3GPP TR 38.901 Eq. 7.9.4-5 using the coupled ray angles
                const float doppler_hz = calculateTargetDoppler(
                    link.target_velocity,
                    incClusters.ray_theta_ZOA[path.c_inc][path.r_inc],
                    incClusters.ray_phi_AOA[path.c_inc][path.r_inc],
                    scaClusters.ray_theta_ZOD[path.c_sca][path.r_sca],
                    scaClusters.ray_phi_AOD[path.c_sca][path.r_sca],
                    lambda_0);

                const float rayPhase = incClusters.ray_phase_theta_theta[path.c_inc][path.r_inc]
                                     + scaClusters.ray_phase_theta_theta[path.c_sca][path.r_sca];
                const float pathPhase = -2.0f * M_PI * (link.incident.d3d + link.scattered.d3d) / lambda_0;
                const float dopplerPhase = 2.0f * M_PI * doppler_hz * snapshotTime;
                const float totalPhase = rayPhase + pathPhase + dopplerPhase;

                // Per 3GPP TR 38.901 Eq. 7.9.4-14: apply pathloss + shadow fading scaling
                // total_pathloss = PL_tx + PL_rx + SF_tx + SF_rx (all in dB)
                // Amplitude scaling: 10^(-total_pathloss/20)
                const float pathloss_linear = std::pow(10.0f, -link.total_pathloss / 20.0f);
                const float amp = std::sqrt(path.path_power) * pathloss_linear;
                const cuComplex baseCoef = make_cuComplex(amp * std::cos(totalPhase),
                                                          amp * std::sin(totalPhase));

                for (uint32_t rxAnt = 0; rxAnt < nRxAnt; ++rxAnt) {
                    const int p_rx = static_cast<int>(rxAnt % rxAntConfig.antSize[4]);
                    float F_rx_theta{}, F_rx_phi{};
                    calcField(rxAntConfig,
                              scaClusters.ray_theta_ZOA[path.c_sca][path.r_sca],
                              scaClusters.ray_phi_AOA[path.c_sca][path.r_sca],
                              rxAntConfig.antPolarAngles[p_rx],
                              F_rx_theta, F_rx_phi);
                    const auto d_bar_rx = elemPos(rxAntConfig, rxAnt);
                    const cuComplex arr_rx = arrayPhase(d_bar_rx,
                                                        scaClusters.ray_theta_ZOA[path.c_sca][path.r_sca],
                                                        scaClusters.ray_phi_AOA[path.c_sca][path.r_sca]);
                    for (uint32_t txAnt = 0; txAnt < nTxAnt; ++txAnt) {
                        const int p_tx = static_cast<int>(txAnt % txAntConfig.antSize[4]);
                        float F_tx_theta{}, F_tx_phi{};
                        calcField(txAntConfig,
                                  incClusters.ray_theta_ZOD[path.c_inc][path.r_inc],
                                  incClusters.ray_phi_AOD[path.c_inc][path.r_inc],
                                  txAntConfig.antPolarAngles[p_tx],
                                  F_tx_theta, F_tx_phi);
                        const auto d_bar_tx = elemPos(txAntConfig, txAnt);
                        const cuComplex arr_tx = arrayPhase(d_bar_tx,
                                                            incClusters.ray_theta_ZOD[path.c_inc][path.r_inc],
                                                            incClusters.ray_phi_AOD[path.c_inc][path.r_inc]);

                        const float xpr_tx = incClusters.ray_xpr[path.c_inc][path.r_inc];
                        const float xpr_rx = scaClusters.ray_xpr[path.c_sca][path.r_sca];
                        const float kappa_tx = std::max(xpr_tx, 1e-6f);
                        const float kappa_rx = std::max(xpr_rx, 1e-6f);
                        
                        // SPST XPR (Eq. 7.9.4-6) - use incident cluster's SPST XPR
                        const float xpr_spst = incClusters.spst_xpr[path.c_inc][path.r_inc];
                        const float kappa_spst = std::max(xpr_spst, 1e-6f);

                        const bool isLosTxRay = (link.incident.los_ind != 0) && (path.c_inc == 0) && (path.r_inc == 0);
                        const bool isLosRxRay = (link.scattered.los_ind != 0) && (path.c_sca == 0) && (path.r_sca == 0);
                        const bool isLosPath = isLosTxRay && isLosRxRay;

                        auto makePhase = [](float phase) {
                            return make_cuComplex(std::cos(phase), std::sin(phase));
                        };

                        // CPM_tx (Eq. 7.9.4-7): STX-SPST link polarization
                        cuComplex cpm_tx[2][2];
                        if (isLosTxRay) {
                            cpm_tx[0][0] = make_cuComplex(1.0f, 0.0f);
                            cpm_tx[0][1] = make_cuComplex(0.0f, 0.0f);
                            cpm_tx[1][0] = make_cuComplex(0.0f, 0.0f);
                            cpm_tx[1][1] = make_cuComplex(-1.0f, 0.0f);
                        } else {
                            const float sqrtKappaTx = std::sqrt(kappa_tx);
                            cpm_tx[0][0] = makePhase(incClusters.ray_phase_theta_theta[path.c_inc][path.r_inc]);
                            cpm_tx[0][1] = makePhase(incClusters.ray_phase_theta_phi[path.c_inc][path.r_inc]);
                            cpm_tx[0][1].x /= sqrtKappaTx; cpm_tx[0][1].y /= sqrtKappaTx;
                            cpm_tx[1][0] = makePhase(incClusters.ray_phase_phi_theta[path.c_inc][path.r_inc]);
                            cpm_tx[1][0].x /= sqrtKappaTx; cpm_tx[1][0].y /= sqrtKappaTx;
                            cpm_tx[1][1] = makePhase(incClusters.ray_phase_phi_phi[path.c_inc][path.r_inc]);
                        }

                        // CPM_spst (Eq. 7.9.4-6): SPST scattering polarization
                        cuComplex cpm_spst[2][2];
                        if (isLosPath) {
                            // For LOS-LOS path, SPST acts as specular reflector
                            cpm_spst[0][0] = make_cuComplex(1.0f, 0.0f);
                            cpm_spst[0][1] = make_cuComplex(0.0f, 0.0f);
                            cpm_spst[1][0] = make_cuComplex(0.0f, 0.0f);
                            cpm_spst[1][1] = make_cuComplex(1.0f, 0.0f);
                        } else {
                            const float sqrtKappaSpst = std::sqrt(kappa_spst);
                            cpm_spst[0][0] = makePhase(incClusters.spst_phase_theta_theta[path.c_inc][path.r_inc]);
                            cpm_spst[0][1] = makePhase(incClusters.spst_phase_theta_phi[path.c_inc][path.r_inc]);
                            cpm_spst[0][1].x /= sqrtKappaSpst; cpm_spst[0][1].y /= sqrtKappaSpst;
                            cpm_spst[1][0] = makePhase(incClusters.spst_phase_phi_theta[path.c_inc][path.r_inc]);
                            cpm_spst[1][0].x /= sqrtKappaSpst; cpm_spst[1][0].y /= sqrtKappaSpst;
                            cpm_spst[1][1] = makePhase(incClusters.spst_phase_phi_phi[path.c_inc][path.r_inc]);
                        }

                        // CPM_rx (Eq. 7.9.4-8): SPST-SRX link polarization
                        cuComplex cpm_rx[2][2];
                        if (isLosRxRay) {
                            cpm_rx[0][0] = make_cuComplex(1.0f, 0.0f);
                            cpm_rx[0][1] = make_cuComplex(0.0f, 0.0f);
                            cpm_rx[1][0] = make_cuComplex(0.0f, 0.0f);
                            cpm_rx[1][1] = make_cuComplex(-1.0f, 0.0f);
                        } else {
                            const float sqrtKappaRx = std::sqrt(kappa_rx);
                            cpm_rx[0][0] = makePhase(scaClusters.ray_phase_theta_theta[path.c_sca][path.r_sca]);
                            cpm_rx[0][1] = makePhase(scaClusters.ray_phase_theta_phi[path.c_sca][path.r_sca]);
                            cpm_rx[0][1].x /= sqrtKappaRx; cpm_rx[0][1].y /= sqrtKappaRx;
                            cpm_rx[1][0] = makePhase(scaClusters.ray_phase_phi_theta[path.c_sca][path.r_sca]);
                            cpm_rx[1][0].x /= sqrtKappaRx; cpm_rx[1][0].y /= sqrtKappaRx;
                            cpm_rx[1][1] = makePhase(scaClusters.ray_phase_phi_phi[path.c_sca][path.r_sca]);
                        }

                        cuComplex term_rx[2] = { make_cuComplex(F_rx_theta, 0.0f), make_cuComplex(F_rx_phi, 0.0f) };
                        cuComplex term_tx[2] = { make_cuComplex(F_tx_theta, 0.0f), make_cuComplex(F_tx_phi, 0.0f) };

                        auto mulAdd = [](const cuComplex a, const cuComplex b, const cuComplex acc) {
                            return cuCaddf(acc, cuCmulf(a, b));
                        };

                        // Matrix multiplication: CPM_rx × CPM_spst × CPM_tx (Eq. 7.9.4-4)
                        // Step 1: temp = CPM_spst × CPM_tx
                        cuComplex temp[2][2];
                        for (int i = 0; i < 2; ++i) {
                            for (int j = 0; j < 2; ++j) {
                                temp[i][j] = make_cuComplex(0.0f, 0.0f);
                                temp[i][j] = mulAdd(cpm_spst[i][0], cpm_tx[0][j], temp[i][j]);
                                temp[i][j] = mulAdd(cpm_spst[i][1], cpm_tx[1][j], temp[i][j]);
                            }
                        }
                        
                        // Step 2: combined = CPM_rx × temp
                        cuComplex combined[2][2];
                        for (int i = 0; i < 2; ++i) {
                            for (int j = 0; j < 2; ++j) {
                                combined[i][j] = make_cuComplex(0.0f, 0.0f);
                                combined[i][j] = mulAdd(cpm_rx[i][0], temp[0][j], combined[i][j]);
                                combined[i][j] = mulAdd(cpm_rx[i][1], temp[1][j], combined[i][j]);
                            }
                        }
                        
                        // Normalization factor: 1/sqrt((|d^θθ|² + |d^φφ|²)/2) per Eq. 7.9.4-4
                        const float d_theta_theta_sq = combined[0][0].x * combined[0][0].x + 
                                                       combined[0][0].y * combined[0][0].y;
                        const float d_phi_phi_sq = combined[1][1].x * combined[1][1].x + 
                                                   combined[1][1].y * combined[1][1].y;
                        const float norm_factor = 1.0f / std::sqrt(std::max((d_theta_theta_sq + d_phi_phi_sq) * 0.5f, 1e-12f));
                        
                        // Apply normalization to combined matrix
                        for (int i = 0; i < 2; ++i) {
                            for (int j = 0; j < 2; ++j) {
                                combined[i][j].x *= norm_factor;
                                combined[i][j].y *= norm_factor;
                            }
                        }

                        // v1 = F_rx^T × combined
                        cuComplex v1[2] = { make_cuComplex(0.0f, 0.0f), make_cuComplex(0.0f, 0.0f) };
                        v1[0] = mulAdd(term_rx[0], combined[0][0], v1[0]);
                        v1[0] = mulAdd(term_rx[1], combined[1][0], v1[0]);
                        v1[1] = mulAdd(term_rx[0], combined[0][1], v1[1]);
                        v1[1] = mulAdd(term_rx[1], combined[1][1], v1[1]);

                        // pol = v1 × F_tx
                        cuComplex pol = make_cuComplex(0.0f, 0.0f);
                        pol = mulAdd(v1[0], term_tx[0], pol);
                        pol = mulAdd(v1[1], term_tx[1], pol);

                        const cuComplex coef = cuCmulf(cuCmulf(baseCoef, arr_rx), cuCmulf(arr_tx, pol));

                        const size_t idx = static_cast<size_t>(snapshot) * nRxAnt * nTxAnt * nMaxTaps +
                                           (rxAnt * nTxAnt + txAnt) * nMaxTaps +
                                           static_cast<size_t>(tapPos);
                        targetCIR.cirCoe[idx] = cuCaddf(targetCIR.cirCoe[idx], coef);
                    }  // txAnt loop
                }  // rxAnt loop
        }  // path loop
        
#ifdef SLS_DEBUG_
        printf("DEBUG ISAC snapshot=%u maxPower=%.4e dropThreshDb=%.1f kept=%zu tapCount=%u\n",
               snapshot, maxPowerAll, dropThreshDb, survivingPaths.size(), tapCount);
#endif

        targetCIR.cirNtaps[0] = tapCount;
    }  // snapshot loop
}

// ============================================================================
// Monostatic Background Channel (3GPP TR 38.901 Section 7.9.4.2)
// ============================================================================

template <typename Tscalar, typename Tcomplex>
MonostaticRpGammaParams slsChan<Tscalar, Tcomplex>::getTrpMonostaticGammaParams(Scenario scenario) {
    // Per 3GPP TR 38.901 Table 7.9.4.2-1: Parameters for TRP monostatic sensing
    MonostaticRpGammaParams params;
    
    switch (scenario) {
        case Scenario::UMi:
            // UMi scenario
            params.alpha_d = 6.1996f;
            params.beta_d = 0.1558f;
            params.c_d = 15.2697f;
            params.alpha_h = 12.0487f;
            params.beta_h = 2.3261f;
            params.c_h = 0.0157f;
            break;
            
        case Scenario::UMa:
            // UMa / Urban grid / Highway(FR2) / HST(FR2)
            params.alpha_d = 10.3370f;
            params.beta_d = 0.1317f;
            params.c_d = 68.7778f;
            params.alpha_h = 16.2253f;
            params.beta_h = 1.9218f;
            params.c_h = 2.6142f;
            break;
            
        case Scenario::RMa:
            // RMa / Highway(FR1) / HST(FR1)
            params.alpha_d = 6.2025f;
            params.beta_d = 0.0391f;
            params.c_d = 1.2940f;
            params.alpha_h = 0.0007f;
            params.beta_h = 5.0146f;
            params.c_h = 0.0522f;
            break;
            
        case Scenario::Indoor:
            // Indoor office
            params.alpha_d = 4.236f;
            params.beta_d = 0.19255f;
            params.c_d = 4.99f;
            params.alpha_h = 1.3293f;
            params.beta_h = 0.1442f;
            params.c_h = -13.19f;
            break;
            
        case Scenario::InF:
            // Indoor Factory
            params.alpha_d = 0.039836f;
            params.beta_d = 0.179783f;
            params.c_d = 1.130020f;
            params.alpha_h = 0.283447f;
            params.beta_h = 0.435965f;
            params.c_h = -17.043530f;
            break;
            
        default:
            // Default to UMi parameters
            params.alpha_d = 6.1996f;
            params.beta_d = 0.1558f;
            params.c_d = 15.2697f;
            params.alpha_h = 12.0487f;
            params.beta_h = 2.3261f;
            params.c_h = 0.0157f;
            break;
    }
    
    return params;
}

template <typename Tscalar, typename Tcomplex>
MonostaticRpGammaParams slsChan<Tscalar, Tcomplex>::getUtMonostaticGammaParams(Scenario scenario, float h_ut, bool is_aerial) {
    // Per 3GPP TR 38.901 Table 7.9.4.2-2: Parameters for UT monostatic sensing
    MonostaticRpGammaParams params;
    constexpr float kBetaDenEps = 1e-3f;
    const auto safeDiv = [&](float num, float den) {
        if (std::abs(den) < kBetaDenEps) {
            den = (den >= 0.0f) ? kBetaDenEps : -kBetaDenEps;
        }
        return num / den;
    };
    
    if (is_aerial) {
        // Aerial UE parameters (Table 7.9.4.2-2 Part-2)
        // Parameters are height-dependent: f(h) where h is the aerial UE height
        switch (scenario) {
            case Scenario::UMi:
                // UMi-AV
                params.alpha_d = 0.0156f * h_ut + 5.5399f;
                params.beta_d = 40.4517f / (h_ut + 254.6318f);
                params.c_d = 0.0140f * h_ut + 15.1184f;
                params.alpha_h = 0.0123f * h_ut + 11.9569f;
                params.beta_h = safeDiv(17.8047f, h_ut - 0.2202f);
                params.c_h = 0.0532f * h_ut - 0.0120f;
                break;
                
            case Scenario::UMa:
                // UMa-AV
                params.alpha_d = 0.83f + 0.00015f * h_ut;
                params.beta_d = 1.0f / (536.305f + 1.0279f * h_ut);
                params.c_d = 13.824f + 0.03085f * h_ut;
                params.alpha_h = 0.9054f - 0.0001117f * h_ut;
                params.beta_h = safeDiv(1.0f, 38.672f - 0.04658f * h_ut);
                params.c_h = 25.4898f - 0.02398f * h_ut;
                break;
                
            case Scenario::RMa:
                // RMa-AV
                params.alpha_d = 4.423f + 0.001926f * h_ut;
                params.beta_d = 1.0f / (3.8467f + 0.6547f * h_ut);
                params.c_d = 3.864f + 0.1538f * h_ut;
                params.alpha_h = 1.4231f + 0.00192f * h_ut;
                params.beta_h = safeDiv(1.0f, 1.7157f - 0.00538f * h_ut);
                params.c_h = 2.6541f - 0.003851f * h_ut;
                break;
                
            default:
                // Default to UMi-AV
                params.alpha_d = 0.0156f * h_ut + 5.5399f;
                params.beta_d = 40.4517f / (h_ut + 254.6318f);
                params.c_d = 0.0140f * h_ut + 15.1184f;
                params.alpha_h = 0.0123f * h_ut + 11.9569f;
                params.beta_h = safeDiv(17.8047f, h_ut - 0.2202f);
                params.c_h = 0.0532f * h_ut - 0.0120f;
                break;
        }
    } else {
        // Terrestrial UT parameters (Table 7.9.4.2-2 Part-1)
        switch (scenario) {
            case Scenario::UMi:
                params.alpha_d = 10.0220f;
                params.beta_d = 1.2522f;
                params.c_d = 11.0040f;
                params.alpha_h = 3.0487f;
                params.beta_h = 1.9128f;
                params.c_h = 0.1785f;
                break;
                
            case Scenario::UMa:
                params.alpha_d = 2.9072f;
                params.beta_d = 0.1031f;
                params.c_d = 3.8471f;
                params.alpha_h = 1.6640f;
                params.beta_h = 1.6215f;
                params.c_h = -1.4205f;
                break;
                
            case Scenario::RMa:
                params.alpha_d = 10.2421f;
                params.beta_d = 0.0526f;
                params.c_d = 3.3131f;
                params.alpha_h = 0.3175f;
                params.beta_h = 1.4150f;
                params.c_h = 1.5906f;
                break;
                
            case Scenario::Indoor:
                params.alpha_d = 4.3733f;
                params.beta_d = 0.4457f;
                params.c_d = 4.6302f;
                params.alpha_h = 0.2974f;
                params.beta_h = 0.4103f;
                params.c_h = 2.9711f;
                break;
                
            case Scenario::InF:
                params.alpha_d = 0.231418f;
                params.beta_d = 0.128133f;
                params.c_d = 2.004903f;
                params.alpha_h = 0.462968f;
                params.beta_h = 0.281526f;
                params.c_h = -16.921515f;
                break;
                
            default:
                params.alpha_d = 10.0220f;
                params.beta_d = 1.2522f;
                params.c_d = 11.0040f;
                params.alpha_h = 3.0487f;
                params.beta_h = 1.9128f;
                params.c_h = 0.1785f;
                break;
        }
    }
    
    return params;
}

namespace {
template <typename TRng>
inline float sampleGammaDistribution(TRng& gen, const float alpha, const float beta, const float offset) {
    std::gamma_distribution<float> gamma_dist(alpha, 1.0f / beta);  // use 1.0f / beta for C++ gamma distribution definition
    return gamma_dist(gen) + offset;
}
}  // namespace

template <typename Tscalar, typename Tcomplex>
void slsChan<Tscalar, Tcomplex>::generateMonostaticReferencePoints(
    const Coordinate& stx_srx_loc,
    const MonostaticRpGammaParams& params,
    const float stx_srx_orientation[3],
    const float stx_srx_velocity[3],
    MonostaticReferencePoint rps[3]) {
    
    std::uniform_real_distribution<float> uniform_dist(-180.0f, 180.0f);
    
    // Step 2a: Draw LOS AOD for first RP uniformly from [-180deg, 180deg]
    const float los_aod_0_deg = uniform_dist(m_gen);
    
    // Generate 3 Reference Points
    for (uint32_t r = 0; r < 3; ++r) {
        rps[r].rp_id = r;
        
        // Step 2b: Draw 2D distance from Γ(α_d, β_d) + c_d
        rps[r].d_2d = sampleGammaDistribution(m_gen, params.alpha_d, params.beta_d, params.c_d);
        
        // Step 2c: Draw height from Γ(α_h, β_h) + c_h
        rps[r].height = sampleGammaDistribution(m_gen, params.alpha_h, params.beta_h, params.c_h);
        
        // Ensure non-negative values
        rps[r].d_2d = std::max(0.0f, rps[r].d_2d);
        rps[r].height = std::max(0.0f, rps[r].height);
        
        // Step 2d: LOS AOD for each RP (rotate by 2π/3 for 2nd and 4π/3 for 3rd)
        const float los_aod_deg = los_aod_0_deg + r * (120.0f);
        rps[r].los_aod = los_aod_deg;
        
        // Normalize to [-180, 180]
        while (rps[r].los_aod > 180.0f) rps[r].los_aod -= 360.0f;
        while (rps[r].los_aod < -180.0f) rps[r].los_aod += 360.0f;

        // Step 2e: Calculate 3D location of RP
        rps[r].loc.x = stx_srx_loc.x + rps[r].d_2d * std::cos(los_aod_deg * M_PI / 180.0f);
        rps[r].loc.y = stx_srx_loc.y + rps[r].d_2d * std::sin(los_aod_deg * M_PI / 180.0f);
        rps[r].loc.z = rps[r].height;  // Height above ground
        
        // Calculate LOS ZOD (zenith angle)
        const float d_3d = std::sqrt(rps[r].d_2d * rps[r].d_2d + 
                                    std::pow(rps[r].loc.z - stx_srx_loc.z, 2.0f));
        if (d_3d > 1e-6f) {
            rps[r].los_zod = std::acos(std::clamp((rps[r].loc.z - stx_srx_loc.z) / d_3d, 
                                                   -1.0f, 1.0f)) * 180.0f / M_PI;
        } else {
            rps[r].los_zod = 90.0f;
        }
        
        // Step 2f: Copy inherited properties from STX/SRX
        // Per 3GPP TR 38.901 Section 7.9.4.2 Step 2:
        // "Set each RP the same array orientations... as the STX/SRX"
        // "Set each RP the same velocity as the STX/SRX"
        rps[r].orientation[0] = stx_srx_orientation[0];  // Azimuth (bearing)
        rps[r].orientation[1] = stx_srx_orientation[1];  // Downtilt
        rps[r].orientation[2] = stx_srx_orientation[2];  // Slant
        
        rps[r].velocity[0] = stx_srx_velocity[0];
        rps[r].velocity[1] = stx_srx_velocity[1];
        rps[r].velocity[2] = stx_srx_velocity[2];
    }
}

template <typename Tscalar, typename Tcomplex>
void slsChan<Tscalar, Tcomplex>::generateMonostaticBackgroundCIR(
    const MonostaticBackgroundParams& bgParams,
    const AntPanelConfig& txAntConfig,
    const AntPanelConfig& rxAntConfig,
    float fc, float lambda_0,
    uint32_t nSnapshots,
    float currentTime,
    float sampleRate,
    TargetCIR& backgroundCIR) {
    
    const uint32_t nTxAnt = txAntConfig.nAnt;
    const uint32_t nRxAnt = rxAntConfig.nAnt;
    // 3 RPs × 24 taps per RP = 72 taps for monostatic background
    const uint16_t nMaxTaps = ISAC_NUM_REFERENCE_POINTS * ISAC_BG_MAX_TAPS;  // 3 × 24 = 72
    const float c0 = 3e8f;
    
    // Allocate if needed
    if (!backgroundCIR.ownsMemory || backgroundCIR.nTxAnt != nTxAnt ||
        backgroundCIR.nRxAnt != nRxAnt || backgroundCIR.nSnapshots != nSnapshots) {
        backgroundCIR.allocate(nTxAnt, nRxAnt, nSnapshots, nMaxTaps);
    }
    
    // Initialize to zero
    const size_t totalSize = static_cast<size_t>(nSnapshots) * nRxAnt * nTxAnt * nMaxTaps;
    std::fill(backgroundCIR.cirCoe, backgroundCIR.cirCoe + totalSize, make_cuComplex(0.0f, 0.0f));
    std::fill(backgroundCIR.cirNormDelay, backgroundCIR.cirNormDelay + nMaxTaps, 0);
    backgroundCIR.cirNtaps[0] = 0;
    
    // Helpers for antenna patterns (same as generateTargetCIR)
    auto wrapTheta = [](float theta) {
        float wrapped = std::fmod(theta, 360.0f);
        if (wrapped < 0.0f) wrapped += 360.0f;
        return (wrapped > 180.0f) ? 360.0f - wrapped : wrapped;
    };
    auto wrapPhi = [](float phi) {
        float wrapped = std::fmod(phi, 360.0f);
        if (wrapped < 0.0f) wrapped += 360.0f;
        return wrapped;
    };
    auto calcField = [&](const AntPanelConfig& cfg, float theta, float phi, float& F_theta, float& F_phi) {
        constexpr float G_max = 8.0f;
        const int theta_idx = static_cast<int>(std::round(wrapTheta(theta)));
        const int phi_idx = static_cast<int>(std::round(wrapPhi(phi))) % 360;
        const float A_db_3D = cfg.antTheta[theta_idx] + cfg.antPhi[phi_idx] + (cfg.antModel == 1 ? G_max : 0.0f);
        const float A_3D_sqrt = std::pow(10.0f, A_db_3D / 20.0f);
        F_theta = A_3D_sqrt;
        F_phi = A_3D_sqrt;
    };
    auto elemPos = [](const AntPanelConfig& cfg, uint32_t antIdx) {
        const int M = static_cast<int>(cfg.antSize[2]);
        const int N = static_cast<int>(cfg.antSize[3]);
        const int P = static_cast<int>(cfg.antSize[4]);
        const float d_h = cfg.antSpacing[2];
        const float d_v = cfg.antSpacing[3];
        const int m = static_cast<int>((antIdx / (N * P)) % M);
        const int n = static_cast<int>((antIdx / P) % N);
        return std::array<float, 3>{m * d_h, n * d_v, 0.0f};
    };
    auto arrayPhase = [](const std::array<float, 3>& d_bar, float theta_deg, float phi_deg) {
        const float theta = theta_deg * static_cast<float>(M_PI) / 180.0f;
        const float phi = phi_deg * static_cast<float>(M_PI) / 180.0f;
        const float r_head[3] = {std::sin(theta) * std::cos(phi),
                                 std::sin(theta) * std::sin(phi),
                                 std::cos(theta)};
        const float phase = 2.0f * static_cast<float>(M_PI) *
                            (r_head[0] * d_bar[0] + r_head[1] * d_bar[1] + r_head[2] * d_bar[2]);
        return make_cuComplex(std::cos(phase), std::sin(phase));
    };
    
    // Collect all path candidates from all 3 RPs
    struct PathCandidate {
        uint32_t rp_idx{};
        uint16_t cluster{};
        uint16_t ray{};
        float total_delay{};
        float power{};
        float phase{};
        float theta_tx{}, phi_tx{};
        float theta_rx{}, phi_rx{};
    };
    std::vector<PathCandidate> allCandidates;
    
    // Per Eq. 7.9.4-15: H_u,s^bk(τ,t) = Σ_{r=0}^{2} 10^{-(PL+SF)/20} * H_u,s^{bk,r}(τ,t)
    // Per 3GPP TR 38.901 Table 7.9.6.1-1: coupling loss is one-way (TX → RP)
    for (uint32_t r = 0; r < 3; ++r) {
        const auto& rp = bgParams.rps[r];
        const auto& clusters = rp.clusters;
        
        if (clusters.n_clusters == 0) continue;
        
        // Power scale: 10^{-(PL+SF)/20} per Eq. 7.9.4-15 (one-way amplitude)
        const float power_scale = std::pow(10.0f, -(rp.pathloss + rp.shadow_fading) / 20.0f);
        
        // Base delay: round-trip for monostatic
        const float base_delay = 2.0f * rp.d_3d / c0;
        
        const uint16_t nCluster = clusters.n_clusters;
        const uint16_t nRayPerCluster = clusters.n_rays_per_cluster;
        
        for (uint16_t c = 0; c < nCluster; ++c) {
            const float cluster_delay = clusters.cluster_delays[c];
            const float cluster_power = clusters.cluster_powers[c];
            
            for (uint16_t ray = 0; ray < nRayPerCluster; ++ray) {
                PathCandidate cand;
                cand.rp_idx = r;
                cand.cluster = c;
                cand.ray = ray;
                
                // Total delay: base + cluster delay (doubled for round-trip)
                cand.total_delay = base_delay + 2.0f * cluster_delay;
                
                // Power includes path loss, cluster power, and ray distribution
                cand.power = power_scale * std::sqrt(cluster_power / static_cast<float>(nRayPerCluster));
                
                // Random phase per ray
                cand.phase = clusters.ray_phase_theta_theta[c][ray];
                
                // For monostatic: AOA = AOD
                cand.theta_tx = clusters.ray_theta_ZOD[c][ray];
                cand.phi_tx = clusters.ray_phi_AOD[c][ray];
                cand.theta_rx = clusters.ray_theta_ZOA[c][ray];
                cand.phi_rx = clusters.ray_phi_AOA[c][ray];
                
                allCandidates.push_back(cand);
            }
        }
    }
    
    if (allCandidates.empty()) {
        return;
    }
    
    // Sort by delay and build tap indices
    std::sort(allCandidates.begin(), allCandidates.end(),
              [](const PathCandidate& a, const PathCandidate& b) { return a.total_delay < b.total_delay; });
    
    std::vector<uint16_t> tapDelays;
    for (const auto& cand : allCandidates) {
        const uint16_t delaySamples = static_cast<uint16_t>(cand.total_delay * sampleRate);
        if (std::find(tapDelays.begin(), tapDelays.end(), delaySamples) == tapDelays.end()) {
            if (tapDelays.size() < nMaxTaps) {
                tapDelays.push_back(delaySamples);
            }
        }
    }
    
    // Store tap delays
    for (size_t i = 0; i < tapDelays.size(); ++i) {
        backgroundCIR.cirNormDelay[i] = tapDelays[i];
    }
    backgroundCIR.cirNtaps[0] = static_cast<uint16_t>(tapDelays.size());
    
    // Generate CIR coefficients
    for (uint32_t snapshot = 0; snapshot < nSnapshots; ++snapshot) {
        for (const auto& cand : allCandidates) {
            const uint16_t delaySamples = static_cast<uint16_t>(cand.total_delay * sampleRate);
            
            // Find tap index
            auto it = std::find(tapDelays.begin(), tapDelays.end(), delaySamples);
            if (it == tapDelays.end()) continue;
            const uint16_t tapIdx = static_cast<uint16_t>(std::distance(tapDelays.begin(), it));
            
            // Generate coefficients for all antenna pairs
            for (uint32_t rxAnt = 0; rxAnt < nRxAnt; ++rxAnt) {
                for (uint32_t txAnt = 0; txAnt < nTxAnt; ++txAnt) {
                    // Antenna field patterns
                    float F_tx_theta, F_tx_phi, F_rx_theta, F_rx_phi;
                    calcField(txAntConfig, cand.theta_tx, cand.phi_tx, F_tx_theta, F_tx_phi);
                    calcField(rxAntConfig, cand.theta_rx, cand.phi_rx, F_rx_theta, F_rx_phi);
                    
                    // Array response (phase from antenna element positions)
                    const auto d_tx = elemPos(txAntConfig, txAnt);
                    const auto d_rx = elemPos(rxAntConfig, rxAnt);
                    const cuComplex arr_tx = arrayPhase(d_tx, cand.theta_tx, cand.phi_tx);
                    const cuComplex arr_rx = arrayPhase(d_rx, cand.theta_rx, cand.phi_rx);
                    
                    // Combined antenna response
                    const float ant_gain = F_tx_theta * F_rx_theta;
                    
                    // Path coefficient
                    const float path_phase = cand.phase - 2.0f * M_PI * cand.total_delay * fc;
                    const cuComplex path_coef = make_cuComplex(
                        cand.power * ant_gain * std::cos(path_phase),
                        cand.power * ant_gain * std::sin(path_phase)
                    );
                    
                    // Apply array response
                    cuComplex coef = cuCmulf(path_coef, arr_tx);
                    coef = cuCmulf(coef, arr_rx);
                    
                    // Add to CIR
                    const size_t idx = snapshot * (nRxAnt * nTxAnt * nMaxTaps) +
                                      (rxAnt * nTxAnt + txAnt) * nMaxTaps + tapIdx;
                    backgroundCIR.cirCoe[idx] = cuCaddf(backgroundCIR.cirCoe[idx], coef);
                }
            }
        }
    }
}

// Explicit template instantiation for the types used
template class slsChan<float, float2>;
