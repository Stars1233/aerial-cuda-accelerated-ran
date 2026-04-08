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

#pragma once

/**
 * @file sls_chan_isac.hpp
 * @brief ISAC (Integrated Sensing and Communication) channel structures and functions
 * 
 * This file implements the target channel model for ISAC systems per 3GPP TR 38.901 Section 7.9.
 * It combines background communication channels with target reflection channels.
 */

#include "../chanModelsDataset.hpp"
#include <vector>
#include <cstdint>
#include <cuComplex.h>

// ============================================================================
// ISAC Constants and Configuration
// ============================================================================

/**
 * Default max taps for background communication channel (no ISAC)
 * This is kept at 24 for backward compatibility with existing code
 */
constexpr uint16_t ISAC_BG_MAX_TAPS = 24;

/**
 * Number of Reference Points for monostatic background channel
 * Per 3GPP TR 38.901 Section 7.9.4.2 Step 2
 */
constexpr uint16_t ISAC_NUM_REFERENCE_POINTS = 3;

/**
 * Calculate total max taps for ISAC combined channel
 * 
 * For monostatic:  (nSPST + 3) × N_MAX_TAPS
 *   - nSPST: Total SPSTs across all STs (each SPST contributes its own taps)
 *   - 3: Reference Points for background channel
 * 
 * For bistatic:    (nSPST + 1) × N_MAX_TAPS
 *   - nSPST: Total SPSTs across all STs
 *   - 1: Background communication channel
 * 
 * @param[in] nSPST Total number of SPSTs across all STs
 * @param[in] is_monostatic True for monostatic, false for bistatic
 * @param[in] bg_max_taps Max taps per individual channel (default 24)
 * @return Total max taps for combined ISAC channel
 */
inline uint32_t calculateIsacMaxTaps(uint32_t nSPST, bool is_monostatic, 
                                     uint16_t bg_max_taps = ISAC_BG_MAX_TAPS) {
    if (is_monostatic) {
        // Monostatic: (nSPST + 3 RPs) × bg_max_taps
        return (nSPST + ISAC_NUM_REFERENCE_POINTS) * bg_max_taps;
    } else {
        // Bistatic: (nSPST + 1 background) × bg_max_taps
        return (nSPST + 1) * bg_max_taps;
    }
}

/**
 * ISAC CIR buffer configuration
 * 
 * Holds the max taps configuration for ISAC channel generation.
 * This allows dynamic sizing based on number of STs/SPSTs.
 * 
 * Usage:
 * - If ISAC is disabled (no STs): Use N_MAX_TAPS = 24 (backward compatible)
 * - If ISAC is enabled (nST > 0): Use larger buffer for combined channel
 *   - Monostatic: (nSPST + 3) × 24  (3 for monostatic reference points)
 *   - Bistatic:   (nSPST + 1) × 24  (1 for background channel)
 * 
 * CFR Conversion Impact:
 * When ISAC is enabled and combined CIR has more than 24 taps, the CFR
 * conversion kernels must use getEffectiveMaxTaps() for:
 * - Dynamic shared memory sizing (s_timeChanLocal)
 * - Static shared memory declaration (s_cirNormDelayUs2Pi) - requires kernel update
 */
struct IsacCirConfig {
    uint16_t bg_max_taps{ISAC_BG_MAX_TAPS};     //!< Max taps per individual channel component (24)
    uint16_t bg_stride{static_cast<uint16_t>(3 * ISAC_BG_MAX_TAPS)};  //!< Background buffer stride (bg_max_taps for bistatic, 3*bg_max_taps for monostatic)
    uint32_t total_max_taps{ISAC_BG_MAX_TAPS};  //!< Total max taps for combined channel
    uint32_t n_spst_total{};                     //!< Total SPSTs across all STs
    uint32_t n_st{};                             //!< Number of sensing targets
    uint32_t bg_n_rx_ant{};                      //!< Background CIR RX antenna count (comm link: nUeAnt)
    uint32_t bg_n_tx_ant{};                      //!< Background CIR TX antenna count (comm link: nBsAnt)
    bool is_monostatic{true};                    //!< True: monostatic, false: bistatic
    bool isac_enabled{};                         //!< True if ISAC is enabled (nST > 0)
    bool disable_background{false};              //!< True: skip background, use target CIR only (for calibration)
    bool disable_target{false};                  //!< True: skip target, use background CIR only (for calibration)
    
    /**
     * Get effective max taps for buffer allocation
     * 
     * Use this for all buffer allocations and kernel shared memory sizing.
     * Returns 24 for non-ISAC (backward compatible), larger for ISAC.
     * 
     * @return 24 if ISAC disabled, total_max_taps if ISAC enabled
     */
    [[nodiscard]] uint32_t getEffectiveMaxTaps() const {
        return isac_enabled ? total_max_taps : bg_max_taps;
    }
    
    /**
     * Get shared memory size for CFR conversion kernel
     * 
     * @param[in] nTxAnt Number of TX antennas per block
     * @param[in] nRxAnt Number of RX antennas per block
     * @return Shared memory size in bytes
     */
    [[nodiscard]] size_t getCfrKernelSharedMemSize(uint32_t nTxAnt, uint32_t nRxAnt) const {
        const uint32_t effectiveTaps = getEffectiveMaxTaps();
        // s_timeChanLocal (cuComplex) + s_cirNormDelayUs2Pi (float)
        return effectiveTaps * nTxAnt * nRxAnt * sizeof(cuComplex) + effectiveTaps * sizeof(float);
    }
    
    /**
     * Recalculate total_max_taps based on current configuration
     */
    void updateTotalMaxTaps() {
        if (isac_enabled && n_spst_total > 0) {
            total_max_taps = calculateIsacMaxTaps(n_spst_total, is_monostatic, bg_max_taps);
        } else {
            total_max_taps = bg_max_taps;
        }
    }
    
    /**
     * Configure for ISAC operation
     * 
     * Call this when at least one ST exists (ISAC enabled)
     * 
     * @param[in] nST Number of sensing targets (if 0, ISAC is disabled)
     * @param[in] nSPST Total number of SPSTs across all STs
     * @param[in] monostatic True for monostatic mode
     */
    void configure(uint32_t nST, uint32_t nSPST, bool monostatic) {
        n_st = nST;
        n_spst_total = nSPST;
        is_monostatic = monostatic;
        isac_enabled = (nST > 0);  // ISAC enabled if at least one ST
        bg_stride = monostatic ? static_cast<uint16_t>(3 * bg_max_taps) : bg_max_taps;
        updateTotalMaxTaps();
    }
    
    /**
     * Reset to default (no ISAC)
     */
    void reset() {
        n_st = 0;
        n_spst_total = 0;
        is_monostatic = true;
        isac_enabled = false;
        bg_stride = static_cast<uint16_t>(3 * bg_max_taps);
        total_max_taps = bg_max_taps;
    }
};

// ============================================================================
// Data Structures for ISAC Target Channel
// ============================================================================

/**
 * @brief Cluster parameters for ISAC target links
 * 
 * Per 3GPP TR 38.901 Section 7.9.4.1 Steps 5-8:
 * - Each STX-SPST link has its own cluster parameters
 * - Each SPST-SRX link has its own cluster parameters
 * - For monostatic: SPST-SRX clusters are mirrored from STX-SPST (with angle swap)
 */
struct TargetClusterParams {
    // Cluster delays (Step 5)
    std::vector<float> cluster_delays;        //!< τ_n: Cluster delays in seconds
    
    // Cluster powers (Step 6)
    std::vector<float> cluster_powers;        //!< P_n: Cluster powers (normalized)
    
    // Per-ray angles (Step 7) - [nCluster][nRayPerCluster]
    std::vector<std::vector<float>> ray_phi_AOA;   //!< φ_n,m,AOA: Azimuth of Arrival
    std::vector<std::vector<float>> ray_phi_AOD;   //!< φ_n,m,AOD: Azimuth of Departure  
    std::vector<std::vector<float>> ray_theta_ZOA; //!< θ_n,m,ZOA: Zenith of Arrival
    std::vector<std::vector<float>> ray_theta_ZOD; //!< θ_n,m,ZOD: Zenith of Departure
    
    // Per-ray RCS (linear) for ISAC reflections
    std::vector<std::vector<float>> ray_rcs;        //!< RCS per ray (linear)
    
    // Cluster XPR (Step 12)
    std::vector<std::vector<float>> ray_xpr;       //!< κ_n,m: Cross-polarization ratio
    
    // Cluster random phases (Step 13) - for CPM_tx/CPM_rx
    std::vector<std::vector<float>> ray_phase_theta_theta;
    std::vector<std::vector<float>> ray_phase_theta_phi;
    std::vector<std::vector<float>> ray_phase_phi_theta;
    std::vector<std::vector<float>> ray_phase_phi_phi;
    
    // SPST scattering phases (Eq. 7.9.4-6) - for CPM_spst
    // Represents polarization transformation at the target scattering point
    std::vector<std::vector<float>> spst_phase_theta_theta;
    std::vector<std::vector<float>> spst_phase_theta_phi;
    std::vector<std::vector<float>> spst_phase_phi_theta;
    std::vector<std::vector<float>> spst_phase_phi_phi;
    std::vector<std::vector<float>> spst_xpr;  //!< XPR for SPST scattering
    
    // Pre-computed NLOS ray coupling for time continuity (3GPP TR 38.901 Section 7.9.4 Step 9)
    // For each cluster, stores the shuffled scattered ray indices to pair with incident rays
    // nlos_ray_coupling[cluster][inc_ray_idx] = coupled_sca_ray_idx
    std::vector<std::vector<uint16_t>> nlos_ray_coupling;
    
    // Sub-cluster delay offsets for 2 strongest clusters (3GPP TR 38.901 Eq. 7.5-26)
    // ray_subcluster_delay[cluster][ray] = additional delay offset (seconds)
    std::vector<std::vector<float>> ray_subcluster_delay;
    uint16_t strongest_clusters[2]{0, 1};     //!< Indices of 2 strongest clusters
    
    uint16_t n_clusters{};                    //!< Number of clusters
    uint16_t n_rays_per_cluster{};            //!< Number of rays per cluster
};

/**
 * Reference Point (RP) for monostatic background channel
 * Per 3GPP TR 38.901 Section 7.9.4.2 Step 2
 * 
 * Each RP is generated with:
 * - 2D distance from Γ(α_d, β_d) + c_d
 * - Height from Γ(α_h, β_h) + c_h
 * - LOS AOD: first RP uniform [-π, π], rotated by 2π/3 and 4π/3 for 2nd/3rd
 * 
 * Per 3GPP Step 2, each RP inherits from STX/SRX:
 * - Same antenna field patterns (F_rx)
 * - Same array orientations (Ω_RP,α, Ω_RP,β, Ω_RP,γ)
 * - Same velocity
 */
struct MonostaticReferencePoint {
    uint32_t rp_id{};          //!< Reference point ID (0, 1, or 2)
    Coordinate loc{};           //!< RP location in GCS
    float d_2d{};               //!< 2D distance from STX/SRX to RP
    float d_3d{};               //!< 3D distance from STX/SRX to RP
    float height{};             //!< Height of RP
    float los_aod{};            //!< LOS AOD from STX/SRX to RP (degrees)
    float los_zod{};            //!< LOS ZOD from STX/SRX to RP (degrees)
    float los_aoa{};            //!< LOS AOA (= AOD for monostatic)
    float los_zoa{};            //!< LOS ZOA (= ZOD for monostatic)
    float pathloss{};           //!< Path loss to RP (dB)
    float shadow_fading{};      //!< Shadow fading for this RP (dB)
    
    // LSP for this RP (generated during initialization)
    float delay_spread{};       //!< DS: RMS delay spread
    float asd{};                //!< ASD: Azimuth spread of departure
    float asa{};                //!< ASA: Azimuth spread of arrival
    float zsd{};                //!< ZSD: Zenith spread of departure
    float zsa{};                //!< ZSA: Zenith spread of arrival
    float ricean_k{};           //!< K: Ricean K-factor (dB)
    
    // Pre-computed clusters/rays (generated once, reused per TTI)
    TargetClusterParams clusters;  //!< Cluster parameters for this RP channel
    
    // Per 3GPP TR 38.901 Section 7.9.4.2 Step 2: Inherited from STX/SRX
    float orientation[3]{};     //!< RP orientation [azimuth, downtilt, slant] - same as STX/SRX
    float velocity[3]{};        //!< RP velocity vector [vx, vy, vz] in m/s - same as STX/SRX
};

/**
 * @brief Monostatic reference points per 3GPP TR 38.901 Section 7.9
 * 
 * For monostatic sensing (isac_type=1), TX and RX are co-located on the same BS.
 * Contains:
 * - STX: Sensing Transmitter reference point (antenna panel)
 * - SRX: Sensing Receiver reference point (same or different antenna panel)
 * - 3 RPs: Background channel reference points (per Step 2)
 */
struct MonostaticReferencePoints {
    uint32_t bs_id{0};                //!< Base station ID
    
    // STX (Sensing Transmitter) reference point
    Coordinate stx_loc{};             //!< TX antenna panel center location in GCS
    uint32_t stx_ant_panel_idx{0};    //!< TX antenna panel index
    float stx_orientation[3]{0.0f};   //!< TX antenna panel orientation [azimuth, downtilt, slant]
    
    // SRX (Sensing Receiver) reference point  
    Coordinate srx_loc{};             //!< RX antenna panel center location in GCS
    uint32_t srx_ant_panel_idx{0};    //!< RX antenna panel index (can be same as TX)
    float srx_orientation[3]{0.0f};   //!< RX antenna panel orientation [azimuth, downtilt, slant]
    
    uint8_t same_antenna_panel{1};    //!< 1: same panel for TX/RX, 0: different panels
    
    // Distance between STX and SRX (for self-interference considerations)
    float stx_srx_distance{0.0f};     //!< Distance between TX and RX panels (0 if same panel)
    
    // 3 Reference Points for background channel per 3GPP TR 38.901 Section 7.9.4.2 Step 2
    // Generated using Gamma distributions for distance and height
    // First RP has random azimuth, 2nd/3rd rotated by 2π/3 and 4π/3
    MonostaticReferencePoint background_rps[3];  //!< 3 RPs for monostatic background channel
};

/**
 * @brief Bistatic link endpoints (no RPs in bistatic mode)
 * 
 * For bistatic sensing (isac_type=2), TX and RX are on different BS/UE.
 */
struct BistaticLinkEndpoints {
    // STX (Sensing Transmitter) - typically a BS
    uint32_t stx_id{0};               //!< TX entity ID (BS or UE)
    uint8_t stx_is_cell{1};           //!< 1: TX is BS, 0: TX is UE
    Coordinate stx_loc{};             //!< TX location in GCS
    uint32_t stx_ant_panel_idx{0};    //!< TX antenna panel index
    float stx_orientation[3]{0.0f};   //!< TX antenna panel orientation
    
    // SRX (Sensing Receiver) - can be BS or UE
    uint32_t srx_id{0};               //!< RX entity ID (BS or UE)
    uint8_t srx_is_cell{1};           //!< 1: RX is BS, 0: RX is UE
    Coordinate srx_loc{};             //!< RX location in GCS
    uint32_t srx_ant_panel_idx{0};    //!< RX antenna panel index
    float srx_orientation[3]{0.0f};   //!< RX antenna panel orientation
    
    // Baseline distance between STX and SRX
    float baseline_distance{0.0f};    //!< 3D distance between TX and RX
};

/**
 * @brief Incident path parameters (TX → Target)
 * 
 * Describes the forward propagation path from transmitter to sensing target.
 * Per 3GPP TR 38.901 Section 7.9.4.1, LOS/NLOS and LSPs are link-specific.
 */
struct TargetIncidentPath {
    float d3d{0.0f};           //!< 3D distance TX → Target (meters)
    float d2d{0.0f};           //!< 2D distance (horizontal plane, meters)
    
    // Link-specific propagation condition (Step 2 of 7.9.4.1)
    uint8_t los_ind{};         //!< LOS indicator for STX-SPST link (0: NLOS, 1: LOS)
    bool los_initialized{false}; //!< True once LOS draw has been cached
    
    // Link-specific large-scale parameters (Step 3-4 of 7.9.4.1)
    float pathloss{0.0f};      //!< Path loss PL_tx in dB
    float shadow_fading{0.0f}; //!< Shadow fading SF_tx in dB
    float delay_spread{0.0f};  //!< DS (s)
    float asd{0.0f};           //!< Azimuth spread of departure (deg)
    float asa{0.0f};           //!< Azimuth spread of arrival (deg)
    float zsd{0.0f};           //!< Zenith spread of departure (deg)
    float zsa{0.0f};           //!< Zenith spread of arrival (deg)
    float ricean_k{0.0f};      //!< Ricean K-factor (linear)
    
    // TX side angles (departure from TX perspective)
    float theta_ZOD{0.0f};     //!< Zenith angle of departure from TX (degrees)
    float phi_AOD{0.0f};       //!< Azimuth angle of departure from TX (degrees)
    
    // Target side angles (arrival at target, incident direction)
    float theta_ZOA_i{0.0f};   //!< Zenith angle of arrival at target (degrees)
    float phi_AOA_i{0.0f};     //!< Azimuth angle of arrival at target (degrees)

    // Excess delay per 3GPP TR 38.901 Section 7.6.9
    float delta_tau{0.0f};     //!< Excess delay for STX-SPST link (seconds), only for NLOS

    // Small-scale: Cluster parameters
    TargetClusterParams clusters;              //!< Cluster params for this link
};

/**
 * @brief Scattered path parameters (Target → RX)
 * 
 * Describes the backward propagation path from sensing target to receiver.
 * Per 3GPP TR 38.901 Section 7.9.4.1, LOS/NLOS and LSPs are link-specific.
 * Note: For monostatic mode, los_ind and LSPs are identical to incident path (Step 4).
 */
struct TargetScatteredPath {
    float d3d{0.0f};           //!< 3D distance Target → RX (meters)
    float d2d{0.0f};           //!< 2D distance (horizontal plane, meters)
    
    // Link-specific propagation condition (Step 2 of 7.9.4.1)
    uint8_t los_ind{};         //!< LOS indicator for SPST-SRX link (0: NLOS, 1: LOS)
    bool los_initialized{false}; //!< True once LOS draw has been cached
    
    // Link-specific large-scale parameters (Step 3-4 of 7.9.4.1)
    float pathloss{0.0f};      //!< Path loss PL_rx in dB
    float shadow_fading{0.0f}; //!< Shadow fading SF_rx in dB
    float delay_spread{0.0f};  //!< DS (s)
    float asd{0.0f};           //!< Azimuth spread of departure (deg)
    float asa{0.0f};           //!< Azimuth spread of arrival (deg)
    float zsd{0.0f};           //!< Zenith spread of departure (deg)
    float zsa{0.0f};           //!< Zenith spread of arrival (deg)
    float ricean_k{0.0f};      //!< Ricean K-factor (linear)
    
    // Target side angles (departure from target, scattered direction)
    float theta_ZOD_s{0.0f};   //!< Zenith angle of departure from target (degrees)
    float phi_AOD_s{0.0f};     //!< Azimuth angle of departure from target (degrees)
    
    // RX side angles (arrival at RX perspective)
    float theta_ZOA{0.0f};     //!< Zenith angle of arrival at RX (degrees)
    float phi_AOA{0.0f};       //!< Azimuth angle of arrival at RX (degrees)

    // Excess delay per 3GPP TR 38.901 Section 7.6.9
    float delta_tau{0.0f};     //!< Excess delay for SPST-SRX link (seconds), only for NLOS

    // Small-scale: Cluster parameters
    TargetClusterParams clusters;              //!< Cluster params for this link
};

/**
 * @brief Complete target link parameters
 * 
 * Represents one complete bistatic reflection path: TX → SPST → RX
 * This is the fundamental unit for ISAC channel generation.
 */
struct TargetLinkParams {
    // Link identifiers
    uint32_t tx_id{0};         //!< Transmitter ID (cell or UE index)
    uint32_t rx_id{0};         //!< Receiver ID (cell or UE index)
    uint32_t st_id{0};         //!< Sensing target ID
    uint32_t spst_id{0};       //!< SPST ID within target (scattering point index)
    
    uint8_t tx_is_cell{1};     //!< 1: TX is cell/BS, 0: TX is UE
    uint8_t rx_is_cell{1};     //!< 1: RX is cell/BS, 0: RX is UE
    
    // Path parameters
    TargetIncidentPath incident;   //!< TX → Target path
    TargetScatteredPath scattered; //!< Target → RX path
    
    // Bistatic parameters
    float bistatic_angle{0.0f};     //!< Angle between incident and scattered rays (degrees)
    float total_delay{0.0f};        //!< Round-trip delay: (d_incident + d_scattered) / c (seconds)
    float total_pathloss{0.0f};     //!< Total path loss: PL_incident + PL_scattered (dB)
    
    // Excess delay per 3GPP TR 38.901 Section 7.6.9 and Eq. 7.9.4-2
    float delta_tau_tx{0.0f};       //!< Excess delay for STX-SPST link (seconds)
    float delta_tau_rx{0.0f};       //!< Excess delay for SPST-SRX link (seconds)
                                    //!< For monostatic: delta_tau_rx = delta_tau_tx
    
    // RCS parameters
    float rcs_linear{0.0f};         //!< RCS value in linear scale (m²) after angular dependence
    float rcs_dbsm{0.0f};           //!< RCS value in dBsm (for debugging)
    
    // Target motion parameters
    float doppler_shift_hz{0.0f};   //!< Doppler shift due to target motion (Hz)
    float target_velocity[3]{0.0f}; //!< Target velocity vector [vx, vy, vz] (m/s)
    
    // Target location (for convenience)
    Coordinate target_loc{};        //!< Target SPST location in GCS
};

/**
 * @brief Link parameters from one entity (BS or UE) to one SPST
 * 
 * This represents a single STX-SPST or SPST-SRX link.
 * For monostatic: same params used for both directions (with angle swap)
 * For bistatic: each direction has independent params
 */
struct EntityToSpstLink {
    uint32_t entity_id{};                      //!< Entity ID (BS or UE index)
    uint8_t entity_is_cell{1};                 //!< 1: entity is BS, 0: entity is UE
    uint32_t st_id{};                          //!< Sensing target ID
    uint32_t spst_id{};                        //!< SPST index within ST
    
    // Geometry
    float d3d{};                               //!< 3D distance entity ↔ SPST
    float d2d{};                               //!< 2D distance
    
    // Large-scale params (link-specific)
    uint8_t los_ind{};                         //!< LOS indicator
    float pathloss{};                          //!< Path loss in dB
    float shadow_fading{};                     //!< Shadow fading in dB
    
    // Angles
    float theta_ZOD{};                         //!< Zenith of departure from entity
    float phi_AOD{};                           //!< Azimuth of departure from entity
    float theta_ZOA{};                         //!< Zenith of arrival at SPST
    float phi_AOA{};                           //!< Azimuth of arrival at SPST
    
    // Large-scale params for cluster generation
    float ricean_k{};                          //!< Ricean K-factor (LOS)
    float delay_spread{};                      //!< DS
    float asd{};                               //!< Azimuth spread of departure
    float asa{};                               //!< Azimuth spread of arrival
    float zsd{};                               //!< Zenith spread of departure
    float zsa{};                               //!< Zenith spread of arrival
    float delta_tau{};                         //!< Excess delay per 3GPP TR 38.901 Table 7.6.9-1 (0 for LOS, lognormal for NLOS)
    
    // Small-scale: Cluster parameters
    TargetClusterParams clusters;              //!< Cluster params for this link
};

/**
 * @brief Complete target link for one SPST
 * 
 * For monostatic mode (STX = SRX):
 *   - Use entity_link for both STX-SPST and SPST-SRX (with angle swap)
 *   - srx_link is empty/unused
 * 
 * For bistatic mode (STX ≠ SRX):
 *   - stx_link: STX → SPST link (incident)
 *   - srx_link: SPST → SRX link (scattered) with independent params
 */
struct TargetLinkFullParams {
    // Common info
    uint32_t st_id{};                          //!< Sensing target ID
    uint32_t spst_id{};                        //!< SPST index within ST
    Coordinate spst_loc{};                     //!< SPST location in GCS
    float target_velocity[3]{};                //!< Target velocity [vx, vy, vz]
    
    // RCS (computed from angles between STX-SPST-SRX)
    float rcs_linear{};                        //!< RCS in linear scale (m²)
    float rcs_dbsm{};                          //!< RCS in dBsm
    float bistatic_angle{};                    //!< Bistatic angle β
    
    // Doppler
    float doppler_shift_hz{};                  //!< Target Doppler shift
    
    // STX → SPST link
    EntityToSpstLink stx_link;                 //!< Always used
    
    // SPST → SRX link (only for bistatic, empty for monostatic)
    EntityToSpstLink srx_link;                 //!< Only used when is_bistatic = true
    
    uint8_t is_bistatic{};                     //!< 0: monostatic (stx_link only), 1: bistatic (both)
};

/**
 * @brief Link parameters for monostatic background Reference Point
 * 
 * Per 3GPP TR 38.901 Section 7.9.4.2, each monostatic BS has 3 RPs.
 */
struct RpLinkParams {
    uint32_t bs_id{};                          //!< BS ID
    uint32_t rp_id{};                          //!< RP index (0, 1, or 2)
    MonostaticReferencePoint rp{};             //!< RP location and geometry
    
    // Large-scale parameters (NLOS assigned per Step 3)
    float pathloss{};                          //!< Path loss to RP
    float shadow_fading{};                     //!< Shadow fading for RP link
    
    // Small-scale parameters (Step 4)
    TargetClusterParams clusters;              //!< Cluster parameters for RP link
};

/**
 * @brief Container for all target channel parameters
 * 
 * Complete link structure for ISAC channel generation:
 * 
 * 1. Background Communication: nSite × nUE (handled by existing LinkParams)
 * 
 * 2. Target Channel:
 *    - Monostatic: nSite × nST × nSPST links
 *    - Bistatic: (nSite-1 + nUE) × nSite × nST × nSPST links
 * 
 * 3. Monostatic Background (3 RPs per BS):
 *    - nSite × 3 RP links
 * 
 * Total ISAC links = nSite × nST × nSPST (target) + nSite × 3 (background RPs)
 */
struct TargetChannelParams {
    // =========================================================================
    // Target Links (STX → SPST → SRX)
    // =========================================================================
    std::vector<TargetLinkParams> targetLinks;       //!< All target reflection links
    uint32_t nTargetLinks{};                         //!< Total number of target links
    
    // =========================================================================
    // Reference Points per 3GPP TR 38.901 Section 7.9
    // =========================================================================
    uint8_t isac_type{};                             //!< 0: none, 1: monostatic, 2: bistatic
    
    // Monostatic reference points (used when isac_type=1)
    std::vector<MonostaticReferencePoints> monostaticRefPoints;  //!< One per BS
    
    // Bistatic link endpoints (used when isac_type=2)
    std::vector<BistaticLinkEndpoints> bistaticLinks;            //!< TX-RX pairs
    
    // =========================================================================
    // Monostatic Background Channel (3 RPs per BS)
    // =========================================================================
    std::vector<RpLinkParams> rpLinks;               //!< nSite × 3 RP links
    uint32_t nRpLinks{};                             //!< Total number of RP links (nSite × 3)
    
    // =========================================================================
    // Indexing Helpers
    // =========================================================================
    uint32_t nSites{};                               //!< Number of BS sites
    uint32_t nST{};                                  //!< Number of sensing targets
    uint32_t nSPSTTotal{};                           //!< Total SPSTs across all STs
    
    /**
     * Get target link index for monostatic mode
     * Index = site_id × nST × max_spst + st_id × max_spst + spst_id
     */
    uint32_t getTargetLinkIndex(uint32_t site_id, uint32_t st_id, uint32_t spst_id, uint32_t max_spst_per_st) const {
        return site_id * nST * max_spst_per_st + st_id * max_spst_per_st + spst_id;
    }
    
    /**
     * Get RP link index
     * Index = site_id × 3 + rp_id
     */
    uint32_t getRpLinkIndex(uint32_t site_id, uint32_t rp_id) const {
        return site_id * 3 + rp_id;
    }
};

/**
 * @brief Target CIR (Channel Impulse Response) storage
 * 
 * Stores channel coefficients for target reflections.
 * Format matches background CIR for easy combination.
 */
struct TargetCIR {
    cuComplex* cirCoe{nullptr};         //!< CIR coefficients [nSnapshot][nRxAnt][nTxAnt][N_MAX_TAPS]
    uint16_t* cirNormDelay{nullptr};    //!< Normalized delay indices [N_MAX_TAPS]
    uint16_t* cirNtaps{nullptr};        //!< Number of taps (typically 1 per SPST)
    
    uint32_t nTxAnt{0};                 //!< Number of TX antennas
    uint32_t nRxAnt{0};                 //!< Number of RX antennas
    uint32_t nSnapshots{0};             //!< Number of time snapshots
    
    // Owning pointers - must be freed
    bool ownsMemory{false};
    
    // Constructor
    TargetCIR() = default;
    
    // Non-copyable to avoid double free on owning pointers
    TargetCIR(const TargetCIR&) = delete;
    TargetCIR& operator=(const TargetCIR&) = delete;
    
    // Movable to transfer ownership safely
    TargetCIR(TargetCIR&& other) noexcept {
        *this = std::move(other);
    }
    TargetCIR& operator=(TargetCIR&& other) noexcept {
        if (this != &other) {
            // Free existing ownership
            if (ownsMemory) {
                deallocate();
            }
            cirCoe = other.cirCoe;
            cirNormDelay = other.cirNormDelay;
            cirNtaps = other.cirNtaps;
            nTxAnt = other.nTxAnt;
            nRxAnt = other.nRxAnt;
            nSnapshots = other.nSnapshots;
            ownsMemory = other.ownsMemory;
            
            // Reset source
            other.cirCoe = nullptr;
            other.cirNormDelay = nullptr;
            other.cirNtaps = nullptr;
            other.nTxAnt = 0;
            other.nRxAnt = 0;
            other.nSnapshots = 0;
            other.ownsMemory = false;
        }
        return *this;
    }
    
    // Allocate memory
    void allocate(uint32_t nTxAnt, uint32_t nRxAnt, uint32_t nSnapshots, uint16_t nMaxTaps);
    
    // Free memory
    void deallocate();
    
    // Destructor
    ~TargetCIR() {
        if (ownsMemory) {
            deallocate();
        }
    }
};

// ============================================================================
// ISAC Target Channel Functions
// ============================================================================

/**
 * @brief Calculate 3D Euclidean distance between two points
 * 
 * @param loc1 First coordinate
 * @param loc2 Second coordinate
 * @return float Distance in meters
 */
inline float calculateDistance3DTarget(const Coordinate& loc1, const Coordinate& loc2);

/**
 * @brief Calculate 2D horizontal distance between two points
 * 
 * @param loc1 First coordinate
 * @param loc2 Second coordinate
 * @return float Distance in meters (ignores z-component)
 */
inline float calculateDistance2DTarget(const Coordinate& loc1, const Coordinate& loc2);

/**
 * @brief Calculate incident path parameters (TX → Target)
 * 
 * Computes geometry, angles, path loss, and determines LOS/NLOS for STX-SPST link.
 * Per 3GPP TR 38.901 Section 7.9.4.1 Steps 2-4:
 * - Step 2: Assign LOS/NLOS based on LOS probability
 * - Step 3: Calculate path loss
 * - Step 4: Generate shadow fading
 * 
 * @param[in] tx_loc Transmitter location
 * @param[in] target_loc Target SPST location (in GCS)
 * @param[in] fc Center frequency (Hz)
 * @param[in] scenario Network scenario
 * @param[out] incident Output incident path parameters (including los_ind, pathloss, SF)
 */
void calculateIncidentPath(const Coordinate& tx_loc,
                          const Coordinate& target_loc,
                          const StParam& target,
                          float fc, Scenario scenario,
                          TargetIncidentPath& incident,
                          uint32_t crnSiteIdx = 0);

/**
 * @brief Calculate scattered path parameters (Target → RX)
 * 
 * Computes geometry, angles, path loss, and determines LOS/NLOS for SPST-SRX link.
 * Per 3GPP TR 38.901 Section 7.9.4.1:
 * - For monostatic mode: LSPs are identical to incident path (Step 4 note)
 * - For bistatic mode: LSPs are calculated independently
 * 
 * @param[in] target_loc Target SPST location (in GCS)
 * @param[in] rx_loc Receiver location
 * @param[in] fc Center frequency (Hz)
 * @param[in] scenario Network scenario
 * @param[out] scattered Output scattered path parameters
 * @param[in] is_monostatic True if monostatic mode (STX=SRX)
 * @param[in] incident_path Incident path (used for monostatic mode to copy LSPs)
 */
void calculateScatteredPath(const Coordinate& target_loc,
                           const Coordinate& rx_loc,
                           const StParam& target,
                           float fc, Scenario scenario,
                           TargetScatteredPath& scattered,
                           bool is_monostatic = false,
                           const TargetIncidentPath* incident_path = nullptr,
                           uint32_t crnSiteIdx = 0);

/**
 * @brief Calculate target reflection coefficient for antenna pair
 * 
 * Similar to calculateRayCoefficient() but for target reflections.
 * 
 * Combines:
 * - TX antenna pattern (incident direction)
 * - RX antenna pattern (scattered direction)
 * - RCS scaling: √(σ_RCS / (4π)³)
 * - Bistatic path phase: exp(-j·2π·(d1+d2)/λ)
 * - Target Doppler: exp(j·2π·f_d·t)
 * 
 * @param txAntConfig TX antenna panel configuration
 * @param txAntIdx TX antenna element index
 * @param rxAntConfig RX antenna panel configuration
 * @param rxAntIdx RX antenna element index
 * @param targetLink Target link parameters (angles, RCS, Doppler)
 * @param currentTime Current time for Doppler phase
 * @param lambda_0 Wavelength in meters
 * @return cuComplex Complex channel coefficient for this antenna pair
 */
cuComplex calculateTargetReflectionCoefficient(
    const AntPanelConfig& txAntConfig, uint32_t txAntIdx,
    const AntPanelConfig& rxAntConfig, uint32_t rxAntIdx,
    const TargetLinkParams& targetLink,
    float currentTime, float lambda_0);

/**
 * @brief Initialize all ISAC parameters at setup time
 * 
 * Complete initialization flow per 3GPP TR 38.901 Section 7.9:
 * 
 * 1. Initialize reference points (monostatic or bistatic)
 * 2. Generate 3 RPs per BS for monostatic background (7.9.4.2 Step 2)
 * 3. Calculate all target link large-scale parameters (7.9.4.1 Steps 2-4)
 * 4. Generate cluster parameters for all links (7.9.4.1 Steps 5-8)
 * 
 * Link counts:
 * - Background comm: nSite × nUE (handled separately)
 * - Target channel:  nSite × nST × max_nSPST
 * - RP background:   nSite × 3
 * 
 * @param[in] cellParams Vector of cell/BS parameters
 * @param[in] stParams Vector of sensing target parameters
 * @param[in] utParams Vector of UE parameters (for bistatic)
 * @param[in] sysConfig System configuration
 * @param[in] fc Center frequency in Hz
 * @param[in] scenario Network scenario
 * @param[out] targetChannelParams Output: all ISAC channel parameters
 */
void initializeIsacParameters(
    const std::vector<CellParam>& cellParams,
    const std::vector<StParam>& stParams,
    const std::vector<UtParam>& utParams,
    const SystemLevelConfig& sysConfig,
    float fc, Scenario scenario,
    TargetChannelParams& targetChannelParams);

/**
 * @brief Generate cluster parameters for a link
 * 
 * Per 3GPP TR 38.901 Section 7.9.4.1 Steps 5-8:
 * - Step 5: Generate cluster delays
 * - Step 6: Generate cluster powers
 * - Step 7: Generate arrival/departure angles
 * - Step 8: Couple rays within clusters
 * 
 * @param[in] d_3d 3D distance of the link
 * @param[in] los_ind LOS indicator (1: LOS, 0: NLOS)
 * @param[in] scenario Network scenario
 * @param[in] ricean_k Ricean K-factor (for LOS)
 * @param[in] ds Delay spread
 * @param[in] asd Azimuth spread of departure
 * @param[in] asa Azimuth spread of arrival
 * @param[in] zsd Zenith spread of departure
 * @param[in] zsa Zenith spread of arrival
 * @param[out] clusters Output cluster parameters
 */
void generateClusterParams(
    float d_3d, uint8_t los_ind, Scenario scenario,
    float ricean_k, float ds, float asd, float asa, float zsd, float zsa,
    TargetClusterParams& clusters);

/**
 * @brief Mirror cluster parameters for monostatic mode
 * 
 * Per 3GPP TR 38.901 Section 7.9.4.1 Step 7:
 * "For monostatic sensing mode, the cluster parameters are equal
 * with AOA↔AOD and ZOA↔ZOD swapped"
 * 
 * @param[in] incident_clusters Source clusters (STX-SPST)
 * @param[out] scattered_clusters Output clusters (SPST-SRX) with swapped angles
 */
void mirrorClusterParamsForMonostatic(
    const TargetClusterParams& incident_clusters,
    TargetClusterParams& scattered_clusters);

// ============================================================================
// Monostatic Background Channel (3GPP TR 38.901 Section 7.9.4.2)
// ============================================================================

/**
 * Gamma distribution parameters for Reference Point generation
 * Per Tables 7.9.4.2-1 and 7.9.4.2-2 in 3GPP TR 38.901
 */
struct MonostaticRpGammaParams {
    // Distance parameters: Γ(α_d, β_d) + c_d
    float alpha_d{};  //!< Shape parameter for distance
    float beta_d{};   //!< Scale parameter for distance
    float c_d{};      //!< Offset for distance
    
    // Height parameters: Γ(α_h, β_h) + c_h
    float alpha_h{};  //!< Shape parameter for height
    float beta_h{};   //!< Scale parameter for height
    float c_h{};      //!< Offset for height
};

/**
 * Background channel parameters for monostatic sensing
 */
struct MonostaticBackgroundParams {
    uint32_t bs_id{};           //!< BS ID for monostatic sensing
    Coordinate stx_srx_loc{};   //!< STX/SRX location (same for monostatic)
    float stx_srx_height{};     //!< STX/SRX height
    MonostaticReferencePoint rps[3];  //!< 3 Reference Points
    Scenario scenario{};         //!< Network scenario
};

/**
 * Generate 3 Reference Points for monostatic background channel
 * Per 3GPP TR 38.901 Section 7.9.4.2 Step 2
 * 
 * Steps:
 * 1. Draw 2D distance from Γ(α_d, β_d) + c_d
 * 2. Draw height from Γ(α_h, β_h) + c_h
 * 3. Draw LOS AOD uniformly from [-π, π] for first RP
 * 4. Rotate by 2π/3 and 4π/3 for 2nd and 3rd RPs
 * 5. Calculate 3D location of each RP
 * 6. Copy inherited properties from STX/SRX (orientation, velocity)
 * 
 * @param[in] stx_srx_loc Location of STX/SRX (same for monostatic)
 * @param[in] params Gamma distribution parameters
 * @param[in] stx_srx_orientation STX/SRX orientation [azimuth, downtilt, slant] to inherit
 * @param[in] stx_srx_velocity STX/SRX velocity [vx, vy, vz] to inherit (typically 0 for TRP)
 * @param[out] rps Array of 3 Reference Points
 */
void generateMonostaticReferencePoints(
    const Coordinate& stx_srx_loc,
    const MonostaticRpGammaParams& params,
    const float stx_srx_orientation[3],
    const float stx_srx_velocity[3],
    MonostaticReferencePoint rps[3]);

/**
 * Generate background channel for monostatic sensing
 * Per 3GPP TR 38.901 Section 7.9.4.2 Steps 3-5
 * 
 * H_u,s^bk(τ,t) = Σ_{r=0}^{2} 10^{-(PL + SF)/20} * H_u,s^{bk,r}(τ,t)
 * 
 * @param[in] bgParams Background channel parameters
 * @param[in] txAntConfig TX antenna panel configuration
 * @param[in] rxAntConfig RX antenna panel configuration
 * @param[in] fc Center frequency in Hz
 * @param[in] lambda_0 Wavelength in meters
 * @param[in] nSnapshots Number of time snapshots
 * @param[in] currentTime Current simulation time
 * @param[in] sampleRate Sample rate for delay calculation
 * @param[out] backgroundCIR Output background CIR storage
 */

// ============================================================================
// Inline Implementations
// ============================================================================

inline float calculateDistance3DTarget(const Coordinate& loc1, const Coordinate& loc2) {
    const float dx = loc1.x - loc2.x;
    const float dy = loc1.y - loc2.y;
    const float dz = loc1.z - loc2.z;
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

inline float calculateDistance2DTarget(const Coordinate& loc1, const Coordinate& loc2) {
    const float dx = loc1.x - loc2.x;
    const float dy = loc1.y - loc2.y;
    return std::sqrt(dx * dx + dy * dy);
}

