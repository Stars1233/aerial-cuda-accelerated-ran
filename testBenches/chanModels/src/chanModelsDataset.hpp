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

#ifndef CHAN_MODELS_DATASET_HPP
#define CHAN_MODELS_DATASET_HPP

#include <vector>
#include <cstdint>  // for uint8_t, uint16_t, uint32_t
#include <cuda_runtime.h>
#include <cuda_fp16.h>  // For __half and __half2
#include <string>
#include <random>  // For random number generation
#include <array>
#include <algorithm>  // for std::copy
#include <stdexcept>  // for std::invalid_argument
#include <cmath>     // for std::isnan
#include <limits>    // for std::numeric_limits

// Scenario enum
enum class Scenario {
    UMa,
    UMi,
    RMa,
    Indoor,  // TODO: Not supported yet
    InF,     // TODO: Not supported yet
    SMa      // TODO: Not supported yet
};

// UE type enumeration (3GPP categorization for different device types)
enum class UeType {
    TERRESTRIAL = 0,  // Traditional handheld/fixed UE (smartphones, tablets, fixed CPE)
    VEHICLE = 1,      // Vehicular UE (cars, trucks, buses for V2X communication)
    AERIAL = 2,       // Aerial UE (drones, UAVs used for communication, not sensing)
    AGV = 3,          // Automated Guided Vehicle (industrial robots, warehouse vehicles)
    RSU = 4           // Road Side Unit (fixed infrastructure for V2X, pseudo-UE)
};

// Sensing target type enumeration (3GPP TR 38.901 Section 7.9.2)
enum class SensingTargetType {
    UAV = 0,         // UAV, Table 7.9.1-1
    AUTOMOTIVE = 1,  // Automotive, Table 7.9.1-2
    HUMAN = 2,       // Human target, Table 7.9.1-3
    AGV = 3,         // Automated Guided Vehicle, Table 7.9.1-4
    HAZARD = 4       // Hazards on roads/railways, Table 7.9.1-5
};

// Coordinate structure
struct Coordinate {
    float x = 0.0f;  // x-coordinate in global coordinate system
    float y = 0.0f;  // y-coordinate in global coordinate system
    float z = 0.0f;  // z-coordinate in global coordinate system
    
    // Default constructor
    Coordinate() = default;
    
    // Constructor with parameters
    Coordinate(float x, float y, float z) : x(x), y(y), z(z) {}
};

// Antenna panel parameters
struct AntPanelConfig {
    uint16_t nAnt = 4;  // Number of antennas in the array
    uint16_t antSize[5] = {1, 1, 1, 2, 2};  // Dimensions (M_g,N_g,M,N,P)
    float antSpacing[4] = {0, 0, 0.5, 0.5};  // Spacing in wavelengths
    float antTheta[181] = {0.0f};  // Antenna pattern A(theta, phi=0) in dB, size should be 181 (0-180 degrees)
    float antPhi[360] = {0.0f};    // Antenna pattern A(theta=90, phi) in dB, size should be 360 (0-360 degrees)
    float antPolarAngles[2] = {45, -45};  // Polar angles
    uint8_t antModel = 1;  // 0: isotropic, 1: directional, 2: direct pattern
    
    // Default constructor
    AntPanelConfig() = default;
    
    // Basic constructor
    AntPanelConfig(uint16_t nAnt, uint8_t antModel = 1) : nAnt(nAnt), antModel(antModel) {}
    
    // Constructor with antenna size array for antMode 0 and 1
    // report error if antModel is not 0 or 1
    AntPanelConfig(uint16_t nAnt, const std::array<uint16_t, 5>& antSize, const std::array<float, 4>& antSpacing, const std::array<float, 2>& antPolarAngles, uint8_t antModel = 1) {
        if (antModel != 0 && antModel != 1) {
            throw std::invalid_argument("antModel must be 0 or 1");
        }
        this->nAnt = nAnt;
        this->antModel = antModel;
        std::copy(antSize.begin(), antSize.end(), this->antSize);
        std::copy(antSpacing.begin(), antSpacing.end(), this->antSpacing);
        std::copy(antPolarAngles.begin(), antPolarAngles.end(), this->antPolarAngles);
    }
    
    // Constructor with antenna size array using direct pattern
    AntPanelConfig(uint16_t nAnt, const std::array<uint16_t, 5>& antSize, const std::array<float, 4>& antSpacing, const std::array<float, 181>& antTheta, const std::array<float, 360>& antPhi, const std::array<float, 2>& antPolarAngles, uint8_t antModel = 2) 
        : nAnt(nAnt), antModel(antModel) {
        std::copy(antSize.begin(), antSize.end(), this->antSize);
        std::copy(antSpacing.begin(), antSpacing.end(), this->antSpacing);
        std::copy(antTheta.begin(), antTheta.end(), this->antTheta);
        std::copy(antPhi.begin(), antPhi.end(), this->antPhi);
        std::copy(antPolarAngles.begin(), antPolarAngles.end(), this->antPolarAngles);
    }
};

// Scattering Point of Sensing Target (SPST) parameters (3GPP TR 38.901 Section 7.9.2.1)
struct SpstParam {
    uint32_t spst_id{};  // SPST ID within the ST (0-indexed)
    
    Coordinate loc_in_st_lcs{};  // Location of this SPST in ST's local coordinate system (LCS)
                                  // Relative to ST center position
    
    // RCS parameters per Table 7.9.2.1-1
    float rcs_sigma_m_dbsm{-12.81f};  // Mean monostatic RCS sigma_M in dBsm (10*log10(sigma_M))
    float rcs_sigma_d_dbsm{1.0f};     // Mean monostatic RCS sigma_D in dBsm (10*log10(sigma_D))
    float rcs_sigma_s_db{3.74f};      // Standard deviation sigma_s_dB in dB
    
    // Incident and scattered angles in radians for angular dependent RCS
    float rcs_incident_theta_i{};   // Incident zenith angle
    float rcs_incident_phi_i{};     // Incident azimuth angle
    float rcs_scattered_theta_s{};  // Scattered zenith angle
    float rcs_scattered_phi_s{};    // Scattered azimuth angle
    
    float rcs_k_sp{};  // XPR of the pair i of incident/scattered angles in dB (Table 7.9.2.2-1)
    
    float rcs_cpm_spi[4]{};  // Cross-polarization mismatch (CPM) - Equation 7.9.2-5
    
    uint8_t enable_forward_scattering{1};  // Control forward scattering effect (Eq 7.9.2-2)
                                            // 0: disable, 1: enable (default)
    
    // Default constructor
    SpstParam() = default;
    
    // Basic constructor
    SpstParam(uint32_t spst_id, const Coordinate& loc_in_st_lcs, 
                 float rcs_sigma_m_dbsm, float rcs_sigma_d_dbsm, float rcs_sigma_s_db)
        : spst_id(spst_id), loc_in_st_lcs(loc_in_st_lcs), 
          rcs_sigma_m_dbsm(rcs_sigma_m_dbsm), rcs_sigma_d_dbsm(rcs_sigma_d_dbsm), 
          rcs_sigma_s_db(rcs_sigma_s_db) {}
};

// Sensing Target (ST) parameters (3GPP TR 38.901 Section 7.9.2)
struct StParam {
    // Basic identification and location
    uint32_t sid{};         // Global ST ID (Sensing Target ID)
    
    // Target characteristics
    SensingTargetType target_type{SensingTargetType::UAV};  // Type of sensing target
    
    // Indoor or outdoor indicator (ST-specific property)
    // Note: LOS/NLOS is link-specific, not ST-specific per 3GPP TR 38.901 Section 7.9.4.1 Step 2:
    // "Assign propagation condition (LOS/NLOS) for each pair of STX and SPST, and each pair of SPST and SRX"
    // LOS is determined separately for STX-SPST and SPST-SRX links in TargetIncidentPath/TargetScatteredPath
    uint8_t outdoor_ind{1};   //!< 0: indoor, 1: outdoor (ST-specific)
    
    // Mobility parameters
    float velocity[3]{};    // Velocity vector (vx, vy, vz) in m/s
    
    // 3D location in GCS
    Coordinate loc{};       // ST location in GCS at beginning [x, y, z] in meters
    
    // Target orientation in GCS (for RCS calculation)
    float orientation[2]{}; // [azimuth, elevation] in degrees
                            // theta: zenith angle (0° = top faces zenith)
                            // phi: azimuth angle (0° = front faces azimuth 0°)
    
    // Physical dimensions
    float physical_size[3]{};  // Physical dimensions [length, width, height] in meters
    
    // RCS model configuration
    uint8_t rcs_model{1};  // RCS model type:
                           // 1: Single SPST with deterministic monostatic RCS (angular independent sigma_D)
                           // 2: Single/multiple SPSTs with angular dependent RCS (angular dependent sigma_D)
    
    uint32_t n_spst{1};  // Number of scattering points (SPSTs)
                         // RCS Model 1: n_spst MUST be 1 (enforced)
                         // RCS Model 2: n_spst can be 1 or more (5 for automotive/AGV)
    
    std::vector<SpstParam> spst_configs{};  // SPST parameter objects, one per scattering point
    
    // Default constructor
    StParam() = default;
    
    // Basic constructor (parameter order matches member declaration order)
    StParam(uint32_t sid, uint8_t outdoor_ind, const Coordinate& loc)
        : sid(sid), outdoor_ind(outdoor_ind), loc(loc) {}
    
    // Constructor with target type and RCS model (parameter order matches member declaration order)
    StParam(uint32_t sid, SensingTargetType target_type, uint8_t outdoor_ind,
            const Coordinate& loc, uint8_t rcs_model)
        : sid(sid), target_type(target_type), outdoor_ind(outdoor_ind), 
          loc(loc), rcs_model(rcs_model) {
        // Enforce n_spst based on rcs_model per 3GPP TR 38.901
        if (rcs_model == 1) {
            n_spst = 1;  // Model 1: MUST have single SPST
        } else if (rcs_model == 2) {
            // Model 2: Default based on target type
            if (target_type == SensingTargetType::AUTOMOTIVE || target_type == SensingTargetType::AGV) {
                n_spst = 5;  // 5 SPSTs: front, left, back, right, roof
            } else {
                n_spst = 1;  // Single SPST for others
            }
        }
    }

    /**
     * @brief Validate consistency between n_spst and spst_configs.
     *
     * Ensures that the declared number of scattering points (n_spst) does not
     * exceed the actual configurations stored in spst_configs.
     *
     * @param allow_empty_spst When true (default), an empty spst_configs is
     *        accepted without error — useful before SPST auto-configuration
     *        has run.  When false, an empty spst_configs with n_spst > 0 will
     *        trigger the size check and throw.
     *
     * @throws std::invalid_argument if n_spst > spst_configs.size().
     *
     * @note This method is const and does not modify the StParam object.
     */
    void validateSpstConsistency(bool allow_empty_spst = true) const {
        if (allow_empty_spst && spst_configs.empty()) {
            return;
        }
        if (n_spst > spst_configs.size()) {
            throw std::invalid_argument(
                "StParam invalid: n_spst (" + std::to_string(n_spst) +
                ") exceeds spst_configs size (" + std::to_string(spst_configs.size()) + ")"
            );
        }
    }
};

// UT parameters for public API (user-configurable parameters only)
struct UtParamCfg {
    uint32_t uid = 0;  // Global UE ID
    Coordinate loc;  // UE location
    uint8_t outdoor_ind = 0;  // 0: indoor, 1: outdoor
    UeType ue_type = UeType::TERRESTRIAL;  // UE type: TERRESTRIAL, VEHICLE, AERIAL, AGV, RSU
    uint32_t antPanelIdx = 0;  // Antenna panel configuration index
    float antPanelOrientation[3] = {0, 0, 0};  // (theta, phi, slant offset)
    float velocity[3] = {0, 0, 0};  // (vx, vy, vz), abs(velocity_direction) = speed in m/s, vz = 0 per 3GPP spec)
    
    // ISAC monostatic sensing parameters (for UE acting as sensing receiver)
    uint8_t monostatic_ind = 0;  // 0: not a monostatic sensing receiver, 1: monostatic sensing receiver (UE receives sensing reflections)
    uint8_t same_antenna_panel_ind = 0;  // 0: use second antenna panel for sensing, 1: use same antenna panel for sensing
    uint32_t second_ant_panel_idx = 0;  // Second antenna panel index for sensing RX (when monostatic_ind = 1)
    float second_ant_panel_orientation[3] = {0, 0, 0};  // Second antenna panel orientation for sensing RX
    
    // Default constructor
    UtParamCfg() = default;
    
    // Basic constructor
    UtParamCfg(uint32_t uid, const Coordinate& loc, uint8_t outdoor_ind = 0, uint32_t antPanelIdx = 0, UeType ue_type = UeType::TERRESTRIAL)
        : uid(uid), loc(loc), outdoor_ind(outdoor_ind), ue_type(ue_type), antPanelIdx(antPanelIdx) {}
    
    // Constructor with velocity and antenna panel orientation
    UtParamCfg(uint32_t uid, const Coordinate& loc, uint8_t outdoor_ind, uint32_t antPanelIdx, const float antPanelOrientation[3], const float velocity[3], UeType ue_type = UeType::TERRESTRIAL)
        : uid(uid), loc(loc), outdoor_ind(outdoor_ind), ue_type(ue_type), antPanelIdx(antPanelIdx) {
        this->antPanelOrientation[0] = antPanelOrientation[0];
        this->antPanelOrientation[1] = antPanelOrientation[1];
        this->antPanelOrientation[2] = antPanelOrientation[2];
        this->velocity[0] = velocity[0];
        this->velocity[1] = velocity[1];
        this->velocity[2] = velocity[2];
    }
};

// UT parameters for internal implementation (inherits public API + adds internal fields)
struct UtParam : public UtParamCfg {
    float d_2d_in = -1.0f;  //!< 2D distance of an indoor UE (internal use, default -1)
    float o2i_penetration_loss = 0.0f;  //!< O2I building penetration loss in dB (UT-specific, same for all BSs per 3GPP TR 38.901 Section 7.4.3)
    
    // Default constructor
    UtParam() = default;
    
    // Constructor from public API struct
    UtParam(const UtParamCfg& cfg) : UtParamCfg(cfg) {}
};

// Cell parameters
struct CellParam {
    uint32_t cid = 0;  // Global cell ID
    uint32_t siteId = 0;  // Site ID for LSP access
    Coordinate loc;  // Cell location
    uint32_t antPanelIdx = 0;  // Antenna panel configuration index
    float antPanelOrientation[3] = {0, 0, 0};  // (theta, phi, slant offset)
    
    // ISAC monostatic sensing parameters
    uint8_t monostatic_ind = 0;  // 0: not a monostatic target, 1: monostatic target (BS acts as both TX and RX for sensing)
    uint32_t second_ant_panel_idx = 0;  // Second antenna panel index for sensing RX (when monostatic_ind = 1)
    float second_ant_panel_orientation[3] = {0, 0, 0};  // Second antenna panel orientation for sensing RX
    
    // Default constructor
    CellParam() = default;
    
    // Basic constructor
    CellParam(uint32_t cid, uint32_t siteId, Coordinate loc, uint32_t antPanelIdx)
        : cid(cid), siteId(siteId), loc(loc), antPanelIdx(antPanelIdx) {}
    
    // Constructor with parameters
    CellParam(uint32_t cid, uint32_t siteId, Coordinate loc, uint32_t antPanelIdx, const float antPanelOrientation[3]) 
        : cid(cid), siteId(siteId), loc(loc), antPanelIdx(antPanelIdx) {
        this->antPanelOrientation[0] = antPanelOrientation[0];
        this->antPanelOrientation[1] = antPanelOrientation[1];
        this->antPanelOrientation[2] = antPanelOrientation[2];
    }
};

// System-level configuration
struct SystemLevelConfig {
    static constexpr uint32_t kMaxSitesSupported = 19;
    static constexpr uint32_t kMaxSectorsPerSiteSupported = 3;
    static constexpr uint32_t kMaxUtDropCells = kMaxSitesSupported * kMaxSectorsPerSiteSupported;

    Scenario scenario = Scenario::UMa;
    float isd = 1732.0f;  // Inter-site distance in meters
    uint32_t n_site = 1;  // Number of sites
    uint8_t n_sector_per_site = 3;  // Sectors per site
    uint32_t n_ut = 100;  // Total number of UTs
    uint32_t n_ut_drop_cells = 0;  // Number of configured drop cells; 0 means all cells [0, n_site * n_sector_per_site)
    uint32_t ut_drop_cells[kMaxUtDropCells] = {0};  // Allowed UE drop cell IDs
    uint8_t ut_drop_option = 0;  // 0: randomly across whole region; 1: same number of UTs per site; 2: same number of UTs per sector
    float ut_cell_2d_dist[2] = {-1.0f, -1.0f};  // UT-to-serving-cell 2D distance range [min, max] in meters; [-1, -1] means default model behavior
    
    // ISAC (Integrated Sensing and Communications) configuration
    uint8_t isac_type = 0;  // ISAC type: 0: communication only (no sensing targets), 
                            // 1: monostatic sensing (BS acts as both TX and RX), 
                            // 2: bistatic sensing (separate TX and RX)
    uint32_t n_st = 0;  // Total number of STs (sensing targets)
    
    // ISAC sensing target distribution parameters
    float st_horizontal_speed[2] = {8.33f, 8.33f};  // Horizontal speed range [min, max] in m/s for ISAC sensing targets
                                                     // Default: fixed 30 km/h (8.33 m/s)
                                                     // Typical values: 3 km/h (0.83 m/s) indoor, 160 km/h (44.4 m/s) aerial
    float st_vertical_velocity = 0.0f;   // Vertical velocity in m/s (vz component)
                                         // Default: 0.0 for ground targets, can be set for aerial targets
    uint8_t st_distribution_option[2] = {0, 0};  //!< Distribution option for STs: [horizontal, vertical]
                                                  //!< horizontal: 0=Option A (uniform in cell), 1=Option B (per cell), 2=Option C (any area)
                                                  //!< vertical: 0=Option A (uniform 1.5-300m), 1=Option B (fixed height)
    float st_height[2] = {100.0f, 100.0f};  //!< ST height range [min, max] for vertical Option B per TR 38.901 Table 7.9.1-1
                                             //!< Scalar legacy key `st_fixed_height` maps to [h, h]
                                             //!< Typical values: {25, 50, 100, 200, 300}m
    float st_drop_radius = -1.0f;  //!< Override ST drop radius in meters; <= 0 uses default cell radius (ISD/sqrt(3))
    float st_override_k_db = std::numeric_limits<float>::quiet_NaN();  //!< LOS K-factor override (dB) for aerial ST links; NaN disables override
    float st_minimum_distance = 0.0f;  //!< Minimum distance between STs in meters
                                        // 0: Option A (at least larger than physical size of target)
                                        // Other values: minimum separation in meters
    uint8_t st_size_ind = 0;  // Size index for STs: 0: small, 1: medium, 2: large
    float st_min_dist_from_tx_rx = 10.0f;  // Minimum 3D distance from ST to any STX/SRX (BS/UE) in meters
                                            // Default: 10 meters per 3GPP TR 38.901 for UAV
    SensingTargetType st_target_type = SensingTargetType::UAV;  // Default target type for auto-generated STs
                                                                 // Per 3GPP TR 38.901 Section 7.9.2
    uint8_t st_rcs_model = 1;  // RCS model for STs: 1: deterministic monostatic (single SPST)
                                // 2: angular dependent (can have multiple SPSTs for automotive/AGV)
    
    uint8_t optional_pl_ind = 0;  // 0: standard pathloss, 1: optional pathloss
    float path_drop_threshold_db = 40.0f;  // Path power drop threshold (dB) for ISAC ray/path pruning
    uint8_t isac_disable_background = 0;   // 0: combine target with background, 1: target CIR only (for calibration)
    uint8_t isac_disable_target = 0;       // 0: include target CIR, 1: background CIR only (for calibration)
    uint8_t o2i_building_penetr_loss_ind = 1;  // 0: none, 1: low-loss, 2: 50% low-loss, 50% high-loss, 3: 100% high-loss
    uint8_t o2i_car_penetr_loss_ind = 0;  // 0: none, 1: basic, 2: 50% basic, 50% metallized, 3: 100% metallized
    uint8_t enable_near_field_effect = 0;  // 0: disable, 1: enable
    uint8_t enable_non_stationarity = 0;  // 0: disable, 1: enable
    float force_los_prob[2] = {-1, -1};  // Force LOS probability
    float force_ut_speed[2] = {-1, -1};  // Force UT speed in m/s
    float force_indoor_ratio = -1;  // Force indoor ratio
    uint8_t disable_pl_shadowing = 0;  // 0: calculate, 1: disable
    uint8_t disable_small_scale_fading = 0;  // 0: calculate, 1: disable
    uint8_t enable_per_tti_lsp = 1;  // 0: disable, 1: update PL/O2I/shadowing, 2: update all
    uint8_t enable_propagation_delay = 1;  // 0: disable propagation delay in CIR generation, 1: enable propagation delay in CIR generation. Propagation delay is link-specific, distance / speed of light
    
    // UE Mobility and Wrap-Around configuration
    uint8_t enable_ue_mobility_wraparound = 0;  // 0: disable wrap-around, 1: enable wrap-around (only for 1, 7, or 19 sites)
    
    // ST (sensing target) mobility wrap-around configuration
    // 0: disable (default), 1: enable wrap-around for ST locations (1, 7, or 19 sites only)
    uint8_t enable_st_mobility_wraparound = 0;
    
    // Aerial UE configuration (per 3GPP TR 36.777)
    // Aerial UE ratio = N_aerial / (N_outdoor_terrestrial + N_indoor_terrestrial)
    // Case 1: 0% (baseline, no aerial UEs)
    // Case 2: 0.67% (corresponding to N_aerial = 0.1 when N_terrestrial ≈ 15)
    float aerial_ue_ratio = 0.0f;          // Ratio of aerial UEs to terrestrial UEs (default: 0.0 = no aerial UEs)
    float aerial_ue_height_min = 1.5f;     // Minimum aerial UE height in meters (default: 1.5m per TR 36.777)
    float aerial_ue_height_max = 300.0f;   // Maximum aerial UE height in meters (default: 300m per TR 36.777)
    float aerial_ue_indoor_ratio = 0.0f;   // Ratio of indoor aerial UEs (0.0-1.0, default: 0.0 = all outdoor)
    
    // Aerial UE fast fading alternatives (per 3GPP TR 36.777 Annex B)
    // 1: Alternative B.1.1 - CDL-D based with height/distance dependent K-factor
    // 2: Alternative B.1.2 - TDL-D based with height/distance dependent K-factor
    // 3: Alternative B.1.3 - Traditional UMa/UMi/RMa with K=15 dB for LOS (currently supported)
    uint8_t aerial_ue_fast_fading_alt = 3;  // Only alternative 3 (B.1.3) is currently supported
    
    /** Calculate the number of hexagonal tiers from the site count.
     *
     * @return Number of tiers: 0 for 1 site, 1 for 7 sites, 2 for 19 sites.
     * @throws std::invalid_argument if n_site is not 1, 7, or 19.
     */
    [[nodiscard]] uint32_t getNumTiers() const {
        if (n_site == 1) return 0;
        if (n_site == 7) return 1;
        if (n_site == 19) return 2;
        
        throw std::invalid_argument(
            "Wrap-around only supports 1, 7, or 19 sites. Got " + 
            std::to_string(n_site) + " sites."
        );
    }
    
    /** Check whether the current site count supports wrap-around mobility.
     *
     * @return True if n_site is 1, 7, or 19 (valid hexagonal tier counts).
     */
    [[nodiscard]] bool isWrapAroundSupported() const {
        return (n_site == 1 || n_site == 7 || n_site == 19);
    }

    // Default constructor
    SystemLevelConfig() = default;
    
    // Basic constructor
    SystemLevelConfig(Scenario scenario, uint32_t n_site, uint8_t n_sector_per_site, uint32_t n_ut, float isd = 1732.0f)
        : scenario(scenario), isd(isd), n_site(n_site), n_sector_per_site(n_sector_per_site), n_ut(n_ut) {}
    
    // Constructor with full parameters
    SystemLevelConfig(Scenario scenario, uint32_t n_site, uint8_t n_sector_per_site, uint32_t n_ut, float isd, uint8_t optional_pl_ind, uint8_t o2i_building_penetr_loss_ind, uint8_t o2i_car_penetr_loss_ind, uint8_t enable_near_field_effect, uint8_t enable_non_stationarity, float force_los_prob[2], float force_ut_speed[2], float force_indoor_ratio, uint8_t disable_pl_shadowing, uint8_t disable_small_scale_fading, uint8_t enable_per_tti_lsp, uint8_t enable_propagation_delay)
        : scenario(scenario), isd(isd), n_site(n_site), n_sector_per_site(n_sector_per_site), n_ut(n_ut) {
        this->optional_pl_ind = optional_pl_ind;
        this->o2i_building_penetr_loss_ind = o2i_building_penetr_loss_ind;
        this->o2i_car_penetr_loss_ind = o2i_car_penetr_loss_ind;
        this->enable_near_field_effect = enable_near_field_effect;
        this->enable_non_stationarity = enable_non_stationarity;
        this->force_los_prob[0] = force_los_prob[0];
        this->force_los_prob[1] = force_los_prob[1];
        this->force_ut_speed[0] = force_ut_speed[0];
        this->force_ut_speed[1] = force_ut_speed[1];
        this->force_indoor_ratio = force_indoor_ratio;
        this->disable_pl_shadowing = disable_pl_shadowing;
        this->disable_small_scale_fading = disable_small_scale_fading;
        this->enable_per_tti_lsp = enable_per_tti_lsp;
        this->enable_propagation_delay = enable_propagation_delay;
    }
};

// Link-level configuration
struct LinkLevelConfig {
    int fast_fading_type = 0;  // 0: AWGN, 1: TDL, 2: CDL
    char delay_profile = 'A';  // 'A' to 'C'
    float delay_spread = 30.0f;  // in nanoseconds
    float velocity[3] = {0, 0, 0};  // (vx, vy, vz), abs(velocity_direction) = speed in m/s, vz = 0 per 3GPP spec
    int num_ray = 20;  // number of rays to add per path; defualt 48 for TDL, 20 for CDL
    float cfo_hz = 200.0f;  // carrier frequency offset in Hz
    float delay = 0.0f;  // delay in seconds
    
    // Default constructor
    LinkLevelConfig() = default;
    
    // Basic constructor
    LinkLevelConfig(int fast_fading_type, char delay_profile = 'A', float delay_spread = 30.0f)
        : fast_fading_type(fast_fading_type), delay_profile(delay_profile), delay_spread(delay_spread) {}

    // Constructor with velocity
    LinkLevelConfig(int fast_fading_type, char delay_profile, float delay_spread, const float velocity[3])
        : fast_fading_type(fast_fading_type), delay_profile(delay_profile), delay_spread(delay_spread) {
        this->velocity[0] = velocity[0];
        this->velocity[1] = velocity[1];
        this->velocity[2] = velocity[2];
    }
};

// Test configuration
struct SimConfig {
    int link_sim_ind = 0;  // Link simulation indicator
    float center_freq_hz = 3e9f;  // Center frequency
    float bandwidth_hz = 100e6f;  // Bandwidth
    float sc_spacing_hz = 15e3f * 2;  // Subcarrier spacing
    int fft_size = 4096;  // FFT size
    int n_prb = 273;  // Number of PRBs
    int n_prbg = 137;  // Number of PRBGs
    int n_snapshot_per_slot = 1;  // Channel realizations per slot
    int run_mode = 0;  // 0: CIR only, 1: CIR+CFR on PRBG, 2: CIR+CFR on PRB/Sc
    int internal_memory_mode = 0;  // 0: external memory, 1: internal memory
    int freq_convert_type = 1;  // Frequency conversion type
    int sc_sampling = 1;  // Subcarrier sampling
    float * tx_sig_in = nullptr;  // Input signal for transmission
    int proc_sig_freq = 0;  // Signal processing frequency indicator
    int optional_cfr_dim = 0;  // Optional CFR dimension: 0: [nActiveUtForThisCell, n_snapshot_per_slot, nUtAnt, nBsAnt, nPrbg / nSc], 1: [nActiveUtForThisCell, n_snapshot_per_slot, nPrbg / nSc, nUtAnt, nBsAnt]
    int cpu_only_mode = 0;  // 0: GPU mode, 1: CPU only mode
    std::uint8_t h5_dump_level = 1;  // 0: minimal (topology + CIR/CFR + config), 1: full (includes link/cluster params, etc.)
    
    // Default constructor
    SimConfig() = default;
    
    // Constructor with basic parameters  
    SimConfig(float center_freq_hz, float bandwidth_hz, int run_mode = 0)
        : center_freq_hz(center_freq_hz), bandwidth_hz(bandwidth_hz), run_mode(run_mode) {
        // Set reasonable defaults for other fields
        this->n_prb = 273;
        this->n_prbg = 137;
        this->n_snapshot_per_slot = 1;
    }
    
    // Constructor with detailed parameters
    SimConfig(float center_freq_hz, float bandwidth_hz, float sc_spacing_hz, int fft_size, int run_mode = 0)
        : center_freq_hz(center_freq_hz), bandwidth_hz(bandwidth_hz), sc_spacing_hz(sc_spacing_hz), 
          fft_size(fft_size), run_mode(run_mode) {
        // Set reasonable defaults for other fields
        this->n_prb = 273;
        this->n_prbg = 137;
        this->n_snapshot_per_slot = 1;
    }
    
    // Constructor with full parameters (fixed member initialization order)
    SimConfig(int link_sim_ind, float center_freq_hz, float bandwidth_hz, float sc_spacing_hz, int fft_size, 
        int n_prb, int n_prbg, int n_snapshot_per_slot, int run_mode, int internal_memory_mode, int freq_convert_type, int sc_sampling, float * tx_sig_in = nullptr, 
        int proc_sig_freq = 0, int optional_cfr_dim = 0, int cpu_only_mode = 0)
        : link_sim_ind(link_sim_ind), center_freq_hz(center_freq_hz), bandwidth_hz(bandwidth_hz), 
          sc_spacing_hz(sc_spacing_hz), fft_size(fft_size), n_prb(n_prb), n_prbg(n_prbg), 
          n_snapshot_per_slot(n_snapshot_per_slot), run_mode(run_mode), internal_memory_mode(internal_memory_mode), 
          freq_convert_type(freq_convert_type), sc_sampling(sc_sampling), tx_sig_in(tx_sig_in), proc_sig_freq(proc_sig_freq), optional_cfr_dim(optional_cfr_dim), cpu_only_mode(cpu_only_mode) {
    }
};

// External configuration (public API - uses UtParamCfg for user interface)
struct ExternalConfig {
    std::vector<CellParam> cell_config;  // Cell configuration
    std::vector<UtParamCfg> ut_config;  // UT configuration (public API struct)
    std::vector<AntPanelConfig> ant_panel_config;  // Antenna panel configurations
    std::vector<StParam> st_config;  // Sensing target parameters (for ISAC)

    // Default constructor
    ExternalConfig() = default;

    // Constructor with parameters (without ST config for backward compatibility)
    ExternalConfig(const std::vector<CellParam>& cell_config, const std::vector<UtParamCfg>& ut_config, const std::vector<AntPanelConfig>& ant_panel_config)
        : cell_config(cell_config), ut_config(ut_config), ant_panel_config(ant_panel_config) {}
    
    // Constructor with ST configuration (for ISAC)
    ExternalConfig(const std::vector<CellParam>& cell_config, const std::vector<UtParamCfg>& ut_config, 
                   const std::vector<AntPanelConfig>& ant_panel_config, const std::vector<StParam>& st_config)
        : cell_config(cell_config), ut_config(ut_config), ant_panel_config(ant_panel_config), st_config(st_config) {}
};

#endif // CHAN_MODELS_DATASET_HPP