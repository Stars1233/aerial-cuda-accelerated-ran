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

#pragma once

#include <string>
#include <vector>
#include <cmath>      // For math functions like log10, pow, exp, fmod, clamp
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda/std/complex>
#include <cuda_fp16.h>  // for __half and __half2
#include <random>
#include <curand_kernel.h>
#include <curand.h>
#include <cstdint>  // for uint8_t, uint16_t, uint32_t
#include <cassert>
#include <cstring>
#include "sls_table.h"
#include "../gpu3gppchanDataset.hpp"
#include "../fastFadingCommon.cuh"
#include "sls_chan_isac.hpp"

#include <algorithm>  // for std::clamp
#include <cstdio>     // for std::fprintf (setLosOverride diagnostics)

#define N_MAX_TAPS 24

// Entries with this value keep the model's TR 38.901 LOS probability draw.
constexpr uint8_t SLS_LOS_OVERRIDE_NONE = 0xFF;
// #define SLS_DEBUG_  // for debugging output

// TR 38.901 Clause 7.1.3: exact GCS->LCS transformation for a panel with orientation
// alpha (bearing, about z), beta (downtilt, about y), gamma (roll, about x), all degrees.
// Maps the GCS direction (theta, phi) [deg] to the panel-local direction (theta', phi')
// [deg] per eq 7.1-7/-8, and returns the polarization basis rotation angle psi [rad] of
// eq 7.1-15 (the field components evaluated in the LCS must be rotated by R(psi) into the
// GCS spherical basis, eq 7.1-11). For a bearing-only orientation (beta = gamma = 0) this
// reduces exactly to theta' = theta, phi' = phi - alpha, psi = 0.
// Storage convention note: BS antPanelOrientation[0] holds the downtilt beta directly,
// while UT antPanelOrientation[0] holds the boresight zenith angle (90 = horizontal), so
// UT callers pass orientation[0] - 90 as beta.
__host__ __device__ inline void gcsToLcs(float alpha_deg, float beta_deg, float gamma_deg,
                                         float theta_deg, float phi_deg,
                                         float& theta_lcs_deg, float& phi_lcs_deg,
                                         float& psi_rad)
{
    const float d2r = static_cast<float>(M_PI) / 180.0f;
    const float r2d = 180.0f / static_cast<float>(M_PI);
    const float b = beta_deg * d2r;
    const float g = gamma_deg * d2r;
    const float th = theta_deg * d2r;
    const float dphi = (phi_deg - alpha_deg) * d2r;
    const float sb = sinf(b), cb = cosf(b);
    const float sg = sinf(g), cg = cosf(g);
    const float st = sinf(th), ct = cosf(th);
    const float sd = sinf(dphi), cd = cosf(dphi);

    // eq 7.1-7
    float cos_theta_p = cb * cg * ct + (sb * cg * cd - sg * sd) * st;
    cos_theta_p = fminf(1.0f, fmaxf(-1.0f, cos_theta_p));
    theta_lcs_deg = acosf(cos_theta_p) * r2d;

    // eq 7.1-8
    const float re_phi = cb * st * cd - sb * ct;
    const float im_phi = cb * sg * ct + (sb * sg * cd + cg * sd) * st;
    phi_lcs_deg = atan2f(im_phi, re_phi) * r2d;

    // eq 7.1-15
    const float re_psi = sg * ct * sd + cg * (cb * st - sb * ct * cd);
    const float im_psi = sg * cd + sb * cg * sd;
    psi_rad = atan2f(im_psi, re_psi);
}
// #define CALIBRATION_CFG_  // for calibration config (min BS-UT is 0, diff generation of d_2d_in)
// #define SHOW_TRAJECTORY  // for showing UE and ST trajectory

// Angle wrapping helpers shared by channel components
inline float wrapAzimuth(const float phi) {
    float wrapped = std::fmod(phi + 180.0f, 360.0f);
    if (wrapped < 0.0f) {
        wrapped += 360.0f;
    }
    return wrapped - 180.0f;
}

inline float wrapZenith(const float theta) {
    float wrapped = std::fmod(theta, 360.0f);
    if (wrapped < 0.0f) {
        wrapped += 360.0f;
    }
    if (wrapped > 180.0f) {
        wrapped = 360.0f - wrapped;
    }
    return wrapped;
}

// Topology parameters
struct TopologyParam {
    uint32_t nSite;  // Number of sites
    uint32_t nSector;  // Number of sectors per site
    uint32_t n_sector_per_site;  // Number of sectors per site
    uint32_t nUT;  // Number of user terminals
    std::vector<CellParam> cellParams;  // Base station parameters
    std::vector<UtParam> utParams;  // User terminal parameters
    float ISD;  // Inter-site distance
    float bsHeight;  // Base station height
    float minBsUeDist2d;  // Minimum 2D distance between base station and user terminal
    float maxBsUeDist2dIndoor;  // Maximum 2D distance between base station and user terminal in indoor
    float indoorUtPercent;  // Percentage of user terminals in indoor
    
    // ISAC (Integrated Sensing and Communications) parameters
    uint32_t nST;  // Number of sensing targets
    std::vector<StParam> stParams;  // Sensing target parameters
};

// Link parameters
// 85 bytes (was 81 bytes, +4 for delta_tau)
struct LinkParams {
    float d2d;  // 2D distance between BS and UT in meters
    float d2d_in;  // 2D distance in indoor environment in meters
    float d2d_out;  // 2D distance in outdoor environment in meters
    float d3d;  // 3D distance between BS and UT in meters
    float d3d_in;  // 3D distance in indoor environment in meters
    float d3d_out;  // 3D distance in outdoor environment in meters
    float phi_LOS_AOD;  // Line-of-Sight (LOS) Azimuth Angle of Departure in degrees
    float theta_LOS_ZOD;  // Line-of-Sight (LOS) Zenith Angle of Departure in degrees
    float phi_LOS_AOA;  // Line-of-Sight (LOS) Azimuth Angle of Arrival in degrees
    float theta_LOS_ZOA;  // Line-of-Sight (LOS) Zenith Angle of Arrival in degrees    
    uint8_t losInd;  // Line-of-Sight indicator (1: LOS, 0: NLOS)
    float pathloss;  // Path loss in dB
    float SF;  // Shadow Fading in dB
    float K;  // K-factor (Ricean factor) in dB
    float DS; // Delay Spread in seconds
    float ASD; // Azimuth Spread of Departure in degrees
    float ASA; // Azimuth Spread of Arrival in degrees
    float mu_lgZSD; // Mean of log10(Zenith Spread of Departure) in degrees
    float sigma_lgZSD; // Standard deviation of log10(Zenith Spread of Departure) in degrees
    float mu_offset_ZOD; // Mean of offset of ZOD
    float ZSD; // Zenith Spread of Departure in degrees
    float ZSA; // Zenith Spread of Arrival in degrees
    float delta_tau; // Excess delay per 3GPP TR 38.901 Table 7.6.9-1 (0 for LOS, lognormal for NLOS, in seconds)
    float h_e; // UMa effective environment height (Table 7.4.1-1 NOTE 1), drawn ONCE per link
               // per drop and reused across PL re-evaluations; 0 = not drawn yet
};

// Common link parameters
// Updated bytes count due to array size changes
struct CmnLinkParams {
    // Large-scale parameters
    float mu_lgDS[3];  // [NLOS, LOS, O2I] mean of log10(DS)
    float sigma_lgDS[3];  // [NLOS, LOS, O2I] std of log10(DS)
    float mu_lgASD[3];  // [NLOS, LOS, O2I] mean of log10(ASD)
    float sigma_lgASD[3];  // [NLOS, LOS, O2I] std of log10(ASD)
    float mu_lgASA[3];  // [NLOS, LOS, O2I] mean of log10(ASA)
    float sigma_lgASA[3];  // [NLOS, LOS, O2I] std of log10(ASA)
    float mu_lgZSA[3];  // [NLOS, LOS, O2I] mean of log10(ZSA)
    float sigma_lgZSA[3];  // [NLOS, LOS, O2I] std of log10(ZSA)
    float mu_K[3];  // [NLOS, LOS, O2I] mean of K
    float sigma_K[3];  // [NLOS, LOS, O2I] std of K
    float r_tao[3];  // [NLOS, LOS, O2I] delay scaling factor
    float mu_XPR[3];  // [NLOS, LOS, O2I] mean of XPR
    float sigma_XPR[3];  // [NLOS, LOS, O2I] std of XPR
    uint16_t nCluster[3];  // [NLOS, LOS, O2I] number of clusters
    uint16_t nRayPerCluster[3];  // [NLOS, LOS, O2I] number of rays per cluster
    
    // cluster parameters
    float C_DS[3];  // [NLOS, LOS, O2I] cluster DS (not used with O2I)
    float C_ASD[3];  // [NLOS, LOS, O2I] cluster ASD
    float C_ASA[3];  // [NLOS, LOS, O2I] cluster ASA
    float C_ZSA[3];  // [NLOS, LOS, O2I] cluster ZSA
    float xi[3];  // [NLOS, LOS, O2I] cluster shadowing std
    float C_phi_LOS;  // LOS azimuth offset, need to be scaled based on K
    float C_phi_NLOS;  // NLOS azimuth offset
    float C_phi_O2I;  // O2I azimuth offset
    float C_theta_LOS;  // LOS elevation offset, need to be scaled based on K
    float C_theta_NLOS;  // NLOS elevation offset
    float C_theta_O2I;  // O2I elevation offset
    
    // updated lgfc
    float lgfc;
    
    // Ray offset angles (no need to set since they are same for all scenarios)
    // float RayOffsetAngles[20];

    // Correlation matrices for LOS and NLOS cases
    float sqrtCorrMatLos[LOS_MATRIX_SIZE * LOS_MATRIX_SIZE];
    float sqrtCorrMatNlos[NLOS_MATRIX_SIZE * NLOS_MATRIX_SIZE];
    float sqrtCorrMatO2i[O2I_MATRIX_SIZE * O2I_MATRIX_SIZE];

    uint32_t nLink;  // Number of links from a Site to a UE
    uint16_t nUeAnt; // Max number of UE antennas (must be <= 65535); TODO: should be the same for all UEs
    uint16_t nBsAnt; // Max number of BS antennas (must be <= 65535); TODO: should be the same for all BSs
    float lambda_0;  // Wavelength in meters
    
    // Subcluster ray definitions (3GPP Table 7.5-5)
    static constexpr int nSubCluster = 3;
    static constexpr int maxRaysInSubCluster = 10;
    int raysInSubClusterSizes[nSubCluster];  // {10, 6, 4}
    uint16_t raysInSubCluster0[10];  // {0, 1, 2, 3, 4, 5, 6, 7, 18, 19}
    uint16_t raysInSubCluster1[6];   // {8, 9, 10, 11, 16, 17}
    uint16_t raysInSubCluster2[4];   // {12, 13, 14, 15}
};

// Cluster parameters
struct ClusterParams {
    // Use maximum possible sizes
    static constexpr uint8_t MAX_CLUSTERS = 20;
    static constexpr uint8_t MAX_RAYS = 20;
    // Max sectors per site. Per 3GPP TR 38.901, co-sited sectors share Steps 1-9 (this
    // per-site ClusterParams) but draw INDEPENDENT Step-10 random phases per sector. We
    // therefore store one phase block per sector; index with sectorIdx = cid % n_sector_per_site.
    static constexpr uint8_t MAX_SECTORS = 3;
    static constexpr uint32_t PHASE_SECTOR_STRIDE = MAX_CLUSTERS * MAX_RAYS * 4;

    // Actual number of clusters and rays for this instance
    uint16_t nCluster = 0;
    uint16_t nRayPerCluster = 0;
    
    // Arrays with maximum size
    float delays[MAX_CLUSTERS];
    float powers[MAX_CLUSTERS];
    uint16_t strongest2clustersIdx[2];
    float phi_n_AoA[MAX_CLUSTERS];
    float phi_n_AoD[MAX_CLUSTERS];
    float theta_n_ZOD[MAX_CLUSTERS];
    float theta_n_ZOA[MAX_CLUSTERS];
    float xpr[MAX_CLUSTERS * MAX_RAYS];
    // Per-sector Step-10 phases: [MAX_SECTORS][MAX_CLUSTERS*MAX_RAYS*4], sector block stride
    // = PHASE_SECTOR_STRIDE. sector 0 holds the legacy layout for single-sector use.
    float randomPhases[MAX_SECTORS * MAX_CLUSTERS * MAX_RAYS * 4];
    float phi_n_m_AoA[MAX_CLUSTERS * MAX_RAYS];
    float phi_n_m_AoD[MAX_CLUSTERS * MAX_RAYS];
    float theta_n_m_ZOD[MAX_CLUSTERS * MAX_RAYS];
    float theta_n_m_ZOA[MAX_CLUSTERS * MAX_RAYS];
};

// Active link parameters
template <typename Tcomplex>
struct activeLink {
    // Default constructor to initialize all pointer members to nullptr
    activeLink() : cirCoe(nullptr), cirNormDelay(nullptr), cirNtaps(nullptr), 
                  freqChanPrbg(nullptr), freqChanSc(nullptr) {}

    // Constructor with all parameters
    activeLink(uint16_t cid_, uint16_t uid_, uint32_t linkIdx_, uint32_t lspReadIdx_,
              Tcomplex* cirCoe_, uint16_t* cirNormDelay_, uint16_t* cirNtaps_,
              Tcomplex* freqChanPrbg_, Tcomplex* freqChanSc_)
        : cid(cid_), uid(uid_), linkIdx(linkIdx_), lspReadIdx(lspReadIdx_),
          cirCoe(cirCoe_), cirNormDelay(cirNormDelay_), cirNtaps(cirNtaps_),
          freqChanPrbg(freqChanPrbg_), freqChanSc(freqChanSc_) {}

    // link indexes
    uint16_t cid;
    uint16_t uid;
    uint32_t linkIdx;
    uint32_t lspReadIdx;
    
    // place to save the channel coefficients
    Tcomplex * cirCoe;
    uint16_t * cirNormDelay;
    uint16_t * cirNtaps;
    Tcomplex * freqChanPrbg;  // CFR on PRBG level
    Tcomplex * freqChanSc;    // CFR on SC level - reused for mode 2/3 (n_prb*12) and mode 4 (N_FFT)
};

// Use types from gpu3gppchanApi.hpp
using scenario_t = Scenario;
using cmnLinkParams_t = CmnLinkParams;
using clusterParams_t = ClusterParams;
using linkParams_t = LinkParams;
using topologyParam_t = TopologyParam;

// Random number type enum
enum class RandomNumberType {
    UNIFORM,
    NORMAL
};

// Main channel model class
template <typename Tscalar, typename Tcomplex>
class slsChan {
public:
    /**
     * @brief Constructor for SLS channel class
     * 
     * @param simConfig Simulation configuration
     * @param sysConfig System level configuration
     * @param randSeed Random seed for simulation
     * @param strm CUDA stream to run SLS class
     */
    slsChan(const SimConfig * simConfig, const SystemLevelConfig * sysConfig, const ExternalConfig * extConfig, uint32_t randSeed, cudaStream_t strm);

    /**
     * @brief Setup SLS channel class by overriding the default configuration
     * 
     * @param extConfig External configuration
     */
    void setup(const ExternalConfig * extConfig);

    /**
     * @brief Run SLS channel generation
     * 
     * @param refTime Timestamp for the start of transmitted symbol (default: 0.0f)
     * @param continuous_fading Flag to enable continuous fading (default: 1)
     * @param activeCell Vector of active cell indices (default: empty)
     * @param activeUt Vector of vectors containing active UT indices per cell (default: empty)
     * @param utNewLoc Vector of new UT locations (default: empty)
     * @param utNewVelocity Vector of new UT mobility parameters (default: empty)
     * @param cirCoePerCell Vector of pointers to store CIR coefficients per cell (default: empty)
     * @param cirNormDelayPerCell Vector of pointers to store CIR normalized delays per cell (default: empty)
     * @param cirNTapsPerCell Vector of pointers to store number of CIR taps per cell (default: empty)
     * @param cfrScPerCell Vector of pointers to store CFR on subcarriers per cell (default: empty)
     * @param cfrPrbgPerCell Vector of pointers to store CFR on PRBGs per cell (default: empty)
     */
    void run(const float refTime = 0.0f,
             const uint8_t continuous_fading = 1,
             const std::vector<uint16_t>& activeCell = {},
             const std::vector<std::vector<uint16_t>>& activeUt = {},
             const std::vector<Coordinate>& utNewLoc = {},
             const std::vector<float3>& utNewVelocity = {},
             const std::vector<Tcomplex*>& cirCoePerCell = {},
             const std::vector<uint16_t*>& cirNormDelayPerCell = {},
             const std::vector<uint16_t*>& cirNTapsPerCell = {},
             const std::vector<Tcomplex*>& cfrScPerCell = {},
             const std::vector<Tcomplex*>& cfrPrbgPerCell = {});

    /**
     * @brief Get the Channel Impulse Response (CIR) coefficients
     * 
     * @param cellIdx Cell index to access
     * @return Tcomplex* Pointer to CIR coefficient data for the specified cell
     * @note CIR is saved as a row-major 1D array [nUe, nBatch, nUeAnt, nBsAnt, firNzLen] per cell
     *       CIR is saved as sparse matrix. Works for both internal and external memory modes.
     */
    Tcomplex* getCirCoe(uint32_t cellIdx = 0) {
        if (m_cirCoePerCell.empty()) {
            // Internal memory mode: return owning pointer (contiguous allocation)
            if (cellIdx == 0) {
                return m_cirCoe;
            }
            // For cellIdx != 0 in internal mode, would need offset calculation
            return nullptr;
        }
        if (cellIdx >= m_cirCoePerCell.size()) {
            return nullptr;
        }
        return m_cirCoePerCell[cellIdx];
    };

    /**
     * @brief Get the Channel Impulse Response (CIR) tap indices
     * 
     * @param cellIdx Cell index to access
     * @return uint16_t* Pointer to CIR tap indices for the specified cell
     * @note Works for both internal and external memory modes.
     */
    uint16_t* getCirIndex(uint32_t cellIdx = 0) {
        if (m_cirNormDelayPerCell.empty()) {
            // Internal memory mode: return owning pointer (contiguous allocation)
            if (cellIdx == 0) {
                return m_cirNormDelay;
            }
            // For cellIdx != 0 in internal mode, would need offset calculation
            return nullptr;
        }
        if (cellIdx >= m_cirNormDelayPerCell.size()) {
            return nullptr;
        }
        return m_cirNormDelayPerCell[cellIdx];
    };

    /**
     * @brief Get the number of CIR taps
     * 
     * @param cellIdx Cell index to access  
     * @return uint16_t* Pointer to number of CIR taps for the specified cell
     * @note Works for both internal and external memory modes.
     */
    uint16_t* getCirNtaps(uint32_t cellIdx = 0) {
        if (m_cirNtapsPerCell.empty()) {
            // Internal memory mode: return owning pointer (contiguous allocation)
            if (cellIdx == 0) {
                return m_cirNtaps;
            }
            // For cellIdx != 0 in internal mode, would need offset calculation
            return nullptr;
        }
        if (cellIdx >= m_cirNtapsPerCell.size()) {
            return nullptr;
        }
        return m_cirNtapsPerCell[cellIdx];
    };

    /** Host-side per-link large-scale parameters, valid after run(). */
    [[nodiscard]] const LinkParams* getLinkParamsHost() const {
        return m_linkParams.data();
    }

    /** Host-side per-link cluster parameters, valid after run(). */
    [[nodiscard]] const ClusterParams* getClusterParamsHost() const {
        return m_clusterParams.data();
    }

    /** Number of site-UT links represented by the host snapshots. */
    [[nodiscard]] uint32_t getNumLinks() const {
        return static_cast<uint32_t>(m_linkParams.size());
    }

    /**
     * Override LOS state per site-UT link for the next regeneration.
     * Values are 0 (NLOS), 1 (LOS), or SLS_LOS_OVERRIDE_NONE.
     */
    [[nodiscard]] bool setLosOverride(const uint8_t* losInd, size_t nLinks) {
        if (losInd == nullptr && nLinks == 0) {
            clearLosOverride();
            return true;
        }
        if (losInd == nullptr) {
            return false;
        }
        if (nLinks != m_linkParams.size()) {
            std::fprintf(
                stderr,
                "setLosOverride: nLinks=%zu does not match nSite*nUT=%zu -- override ignored\n",
                nLinks, m_linkParams.size());
            return false;
        }
        for (size_t linkIdx = 0; linkIdx < nLinks; ++linkIdx) {
            if (losInd[linkIdx] != 0 && losInd[linkIdx] != 1 &&
                losInd[linkIdx] != SLS_LOS_OVERRIDE_NONE) {
                std::fprintf(
                    stderr,
                    "setLosOverride: entry %zu has invalid value %u -- override ignored\n",
                    linkIdx, static_cast<unsigned int>(losInd[linkIdx]));
                return false;
            }
        }
        m_losOverride.assign(losInd, losInd + nLinks);
        m_losOverrideDirty = true;
        m_updateLosState = true;
        m_updateAllLSPs = true;
        m_updatePLAndPenetrationLoss = true;
        return true;
    }

    /** Remove a previously configured LOS override. */
    void clearLosOverride() {
        const bool hadOverride = !m_losOverride.empty();
        // Retain host/device capacity for deterministic reuse. An empty host
        // override makes the GPU kernel receive nullptr; the destructor frees
        // cached device storage.
        m_losOverride.clear();
        m_losOverrideDirty = false;
        if (hadOverride) {
            m_updateLosState = true;
            m_updateAllLSPs = true;
            m_updatePLAndPenetrationLoss = true;
        }
    }

    /**
     * @brief Get Channel Frequency Response (CFR) on PRBG
     * 
     * @param cellIdx Cell index to access
     * @return Tcomplex* Pointer to CFR on PRBG data for the specified cell
     * @note CFR on PRBG is saved as a row-major 1D array [nUe, nBatch, nUeAnt, nBsAnt, nPRBG] per cell
     *       Works for both internal and external memory modes.
     */
    Tcomplex* getFreqChanPrbg(uint32_t cellIdx = 0) {
        if (m_freqChanPrbgPerCell.empty()) {
            // Internal memory mode: return owning pointer (contiguous allocation)
            if (cellIdx == 0) {
                return m_freqChanPrbg;
            }
            // For cellIdx != 0 in internal mode, would need offset calculation
            return nullptr;
        }
        if (cellIdx >= m_freqChanPrbgPerCell.size()) {
            return nullptr;
        }
        return m_freqChanPrbgPerCell[cellIdx];
    };

    /**
     * @brief Get Channel Frequency Response (CFR) on SC
     * 
     * @param cellIdx Cell index to access
     * @return Tcomplex* Pointer to CFR on SC data for the specified cell
     * @note CFR per cell is a row-major 1D array [nUe, nBatch, nUeAnt, nBsAnt, nSc]
     *       Works for both internal and external memory modes.
     */
    Tcomplex* getFreqChanSc(uint32_t cellIdx = 0) {
        if (m_freqChanScPerCell.empty()) {
            // Internal memory mode: return owning pointer (contiguous allocation)
            if (cellIdx == 0) {
                return m_freqChanSc;
            }
            // For cellIdx != 0 in internal mode, would need offset calculation
            return nullptr;
        }
        if (cellIdx >= m_freqChanScPerCell.size()) {
            return nullptr;
        }
        return m_freqChanScPerCell[cellIdx];
    };

    /**
     * @brief Get received signal output
     * 
     * @return Tcomplex* Pointer to received signal data
     * @note Rx samples is saved as a row-major 1D array [nCell, nUe, nUeAnt or nBsAnt, sigLenPerAnt]
     */
    Tcomplex* getRxSigOut() {return m_rxSigOut;};

    /**
     * @brief Reset the SLS channel class by regenerating random numbers
     */
    void reset();



    /**
     * @brief Destructor
     */
    ~slsChan();
                          
    /**
     * @brief Generate network topology
     */
    void generateTopology() {
        bsUeDropping();  // Call the private implementation
        
        // Drop sensing targets if ISAC is enabled
        if (m_sysConfig->isac_type == 1 || m_sysConfig->isac_type == 2) {
            stDropping();
        }
    }

    /**
     * @brief Get GPU memory usage in MB
     * @return float Memory usage in MB (returns 0 in CPU-only mode)
     */
    float getGpuMemUseMB() {
        // Skip GPU memory query in CPU-only mode to avoid creating CUDA context
        if (m_simConfig && m_simConfig->cpu_only_mode == 1) {
            return 0.0f;
        }
        size_t free, total;
        cudaMemGetInfo(&free, &total);
        return (total - free) / (1024.0f * 1024.0f);
    }

    /**
     * @brief Dump network topology to YAML file
     * @param filename Output YAML filename
     */
    void dumpTopologyToYaml(const std::string& filename);

    /**
     * @brief Dump LOS/NLOS statistics for all links
     * 
     * @param lost_nlos_stats Pointer to array for storing LOS/NLOS stats, dimension: [n_sector, n_ut]
     *                        Values: 0 = NLOS (Non-Line-of-Sight), 1 = LOS (Line-of-Sight)
     */
    void dump_los_nlos_stats(float* lost_nlos_stats);

    /**
     * @brief Dump pathloss and shadowing statistics (negative value in dB)
     * 
     * @param pl_sf Pointer to array for storing pathloss+shadowing stats (required)
     *                          If activeCell and activeUt are provided: dimension [activeCell.size(), activeUt.size()]
     *                          If activeCell or activeUt are empty: use dimension n_sector*n_site or n_ut for the empty one
     *                          Values are total loss = - (pathloss - shadow_fading) in dB
     * @param activeCell Vector of active cell IDs (optional, empty vector dumps all cells)
     * @param activeUt Vector of active UT IDs (optional, empty vector dumps all UEs)
     */
    void dump_pl_sf_stats(float* pl_sf,
                          const std::vector<uint16_t>& activeCell = {},
                          const std::vector<uint16_t>& activeUt = {});

    /**
     * @brief Dump pathloss, shadowing and antenna gain statistics
     *
     * Outputs total channel gain in dB = antGain - pathloss + SF, so it can replace
     * antGain is per antenna element only (no 10*log10(nAnt) array gain); downstream may add array/beamforming gain.
     *
     * @param pl_sf_ant_gain Pointer to array for storing gain stats (required)
     *                                    Same dimension rules as dump_pl_sf_stats.
     *                                    Values are total channel gain in dB = antGain - pathloss + SF.
     * @param activeCell Vector of active cell IDs (optional, empty vector dumps all cells)
     * @param activeUt Vector of active UT IDs (optional, empty vector dumps all UEs)
     */
    void dump_pl_sf_ant_gain_stats(float* pl_sf_ant_gain,
                                   const std::vector<uint16_t>& activeCell = {},
                                   const std::vector<uint16_t>& activeUt = {});

    /**
     * @brief Save SLS channel data to H5 file for debugging
     * @param filenameEnding Optional string to append to filename
     */
    void saveSlsChanToH5File(std::string_view filenameEnding = "");

    /**
     * @brief Generate sensing channel for ISAC (Integrated Sensing and Communications)
     * 
     * Generates sensing channel based on ISAC type:
     * - Type 1 (Monostatic): BS -> ST -> BS (round-trip)
     * - Type 2 (Bistatic): BS_TX -> ST -> BS_RX or UE (one-way)
     * 
     * The sensing channel includes:
     * - RCS-weighted reflection from sensing targets
     * - Doppler shift for moving targets
     * - Multi-SPST support for complex targets (automotive/AGV)
     */
    void generateSensingChannel();

    /**
     * @brief Calculate RCS (Radar Cross Section) for sensing targets
     * 
     * Implements 3GPP TR 38.901 Section 7.9.2 RCS models:
     * - Model 1: Deterministic monostatic RCS (n_spst = 1)
     * - Model 2: Angular dependent RCS (n_spst >= 1)
     * 
     * @param stConfig Sensing target parameters
     * @param incidentAngle Incident angle (theta_i, phi_i)
     * @param scatteredAngle Scattered angle (theta_s, phi_s)
     * @param bistaticBeta Bistatic angle in degrees
     * @return float RCS value in linear scale (not dB)
     */
    float calculateRCS(const StParam& stConfig, 
                      float incidentTheta, float incidentPhi,
                      float scatteredTheta, float scatteredPhi,
                      float bistaticBeta);

private:
    const SystemLevelConfig * m_sysConfig;
    const SimConfig * m_simConfig;
    const ExternalConfig * m_extConfig;
    uint32_t m_randSeed;
    cudaStream_t m_strm;
    bool m_updatePerTTILinkParams;  // update link parameters (for each TTI)
    // Target clusters and Option-2 coupling form one sensing realization per
    // simulation drop. They are invalidated by reset(), not by communication
    // UE mobility or active-link selection changes.
    bool m_updateTargetLinkParams{false};
    bool m_updatePLAndPenetrationLoss;  // update pathloss and penetration loss
    bool m_updateAllLSPs;  // update all LSPs
    bool m_updateLosState;  // update LOS/NLOS state (only true at start or after reset)

public:
    /**
     * @brief Number of active communication links currently configured.
     * @return Count of entries in the active link parameter list.
     */
    uint32_t getNumActiveLinks() const { return (uint32_t)m_activeLinkParams.size(); }
    /**
     * @brief Base station antenna count for the common link geometry.
     * @return `nBsAnt` from common link parameters.
     */
    uint16_t getNBsAnt() const { return m_cmnLinkParams.nBsAnt; }
    /**
     * @brief User equipment antenna count for the common link geometry.
     * @return `nUeAnt` from common link parameters.
     */
    uint16_t getNUeAnt() const { return m_cmnLinkParams.nUeAnt; }
    /** @brief Effective transmit-antenna count, including monostatic ISAC. */
    [[nodiscard]] uint32_t getEffectiveNBsAnt() const {
        uint32_t antennas = m_effectiveBsAnt > 0 ? m_effectiveBsAnt : m_cmnLinkParams.nBsAnt;
        if (m_sysConfig && m_sysConfig->isac_type == 1 &&
            !m_targetChannelParams.monostaticRefPoints.empty()) {
            const auto& ref = m_targetChannelParams.monostaticRefPoints.front();
            antennas = std::max(
                antennas,
                static_cast<uint32_t>((*m_antPanelConfig)[ref.stx_ant_panel_idx].nAnt));
        }
        return antennas;
    }
    /** @brief Effective receive-antenna count, including monostatic ISAC. */
    [[nodiscard]] uint32_t getEffectiveNUeAnt() const {
        uint32_t antennas = m_effectiveUeAnt > 0 ? m_effectiveUeAnt : m_cmnLinkParams.nUeAnt;
        if (m_sysConfig && m_sysConfig->isac_type == 1 &&
            !m_targetChannelParams.monostaticRefPoints.empty()) {
            const auto& ref = m_targetChannelParams.monostaticRefPoints.front();
            antennas = std::max(
                antennas,
                static_cast<uint32_t>((*m_antPanelConfig)[ref.srx_ant_panel_idx].nAnt));
        }
        return antennas;
    }
    /**
     * @brief Maximum tap count to use for buffers and kernels (comm vs ISAC).
     *
     * Returns `N_MAX_TAPS` for non-ISAC. When ISAC is enabled (`isac_type` > 0),
     * returns the combined-channel tap budget from `IsacCirConfig` (depends on
     * ST/SPST layout and monostatic vs bistatic mode).
     *
     * @return Effective maximum taps for CIR allocation and CFR conversion sizing.
     */
    uint32_t getEffectiveMaxTaps() const;

private:
    /**
     * @brief Perform BS and UE dropping in the network topology
     * 
     * This function handles the spatial distribution of base stations and user equipment
     * according to the specified network topology parameters. It determines the positions
     * of all network elements based on the configuration settings.
     */
    void bsUeDropping();

    /**
     * @brief Reconcile indoor-distance and penetration-loss state after an
     *        external per-UE indoor/outdoor override.
     *
     * @param ut UT whose public configuration has already been overridden.
     * @param outdoorIndChanged Whether the external flag changed the dropped
     *        indoor/outdoor state.
     */
    void reconcileExternalUtO2iState(UtParam& ut, bool outdoorIndChanged);

    /**
     * @brief Perform sensing target (ST) dropping for ISAC
     * 
     * This function handles the spatial distribution of sensing targets (STs) for 
     * ISAC (Integrated Sensing and Communications) simulations per 3GPP TR 38.901.
     * It ensures STs are placed at minimum 3D distances from all BS/UE positions to
     * guarantee valid sensing geometries. Called when isac_type is 1 or 2.
     * 
     * The function:
     * - Generates n_st sensing targets according to distribution options
     * - Ensures minimum separation from all STX/SRX (BS/UE) positions
     * - Populates m_topology.stParams with generated sensing targets
     */
    void stDropping();

    /**
     * @brief Initialize antenna panel configuration
     */
    void initializeAntPanelConfig();

    /**
     * @brief Calculate Large Scale Parameters (LSP)
     * 
     * Generates and calculates the large scale parameters including:
     * - Delay Spread (DS)
     * - Angular Spread of Arrival (ASA)
     * - Angular Spread of Departure (ASD)
     * - Shadow Fading (SF)
     * - K-factor
     * These parameters are correlated according to 3GPP specifications.
     */
    [[nodiscard]] IsacLsp calIsacLsp(
        scenario_t scenario, bool isLos, bool isIndoor, float fc,
        float d_2d, float d_3d, float h_bs, float h_ut, float utX, float utY,
        uint32_t crnSiteIdx = 0, bool is_aerial = false);

    /**
     * @brief Calculate cluster and ray parameters
     * 
     * Generates cluster and ray parameters including:
     * - Cluster delays
     * - Cluster powers
     * - Arrival and departure angles
     * - Ray coupling
     * - Cross-polarization ratios (XPR)
     * - Initial phases
     */
    void calClusterRay();

    /**
     * @brief Calculate shadow fading standard deviation
     * 
     * Computes the standard deviation of shadow fading (σ_SF) according to 3GPP specifications,
     * with support for aerial UE/targets as per TR 36.777.
     * 
     * @param scenario Deployment scenario (UMa, UMi, RMa)
     * @param isLos Line-of-sight indicator (true for LOS, false for NLOS)
     * @param fc Carrier frequency in Hz
     * @param optionalPlInd Optional pathloss indicator
     * @param d_2d 2D distance in meters
     * @param h_bs Base station height in meters
     * @param h_ut User terminal/target height in meters
     * @param is_aerial Whether the UT/target is aerial (for TR 36.777 aerial UE model)
     * @return Shadow fading standard deviation in dB
     */
    static float calSfStd(scenario_t scenario, bool isLos, float fc, bool optionalPlInd,
                          float d_2d, float h_bs, float h_ut, bool is_aerial = false,
                          bool isIndoor = false);

    /**
     * @brief Generate cluster angles for multipath propagation
     * 
     * Generates azimuth and zenith angles for each cluster according to 3GPP TR 38.901
     * specifications. Applies LOS/NLOS-specific scaling factors and adjustments.
     * 
     * @param nCluster Number of clusters
     * @param C_ASA Cluster ASA parameter
     * @param C_ASD Cluster ASD parameter
     * @param C_phi_NLOS Azimuth cluster angle scaling for NLOS
     * @param C_phi_LOS Azimuth cluster angle scaling for LOS
     * @param c_phi_O2I Azimuth cluster angle scaling for O2I
     * @param C_theta_LOS Zenith cluster angle scaling for LOS
     * @param C_theta_NLOS Zenith cluster angle scaling for NLOS
     * @param C_theta_O2I Zenith cluster angle scaling for O2I
     * @param ASA Azimuth spread of arrival in degrees
     * @param ASD Azimuth spread of departure in degrees
     * @param ZSA Zenith spread of arrival in degrees
     * @param ZSD Zenith spread of departure in degrees
     * @param phi_LOS_AOA LOS azimuth angle of arrival in degrees
     * @param phi_LOS_AOD LOS azimuth angle of departure in degrees
     * @param theta_LOS_ZOA LOS zenith angle of arrival in degrees
     * @param theta_LOS_ZOD LOS zenith angle of departure in degrees
     * @param mu_offset_ZOD Zenith offset for departure angles
     * @param losInd Line-of-sight indicator
     * @param outdoor_ind Outdoor indicator (1: outdoor, 0: indoor)
     * @param K K-factor in dB for Ricean fading
     * @param powers Cluster powers array
     * @param phi_n_AoA Output: cluster azimuth angles of arrival
     * @param phi_n_AoD Output: cluster azimuth angles of departure
     * @param theta_n_ZOD Output: cluster zenith angles of departure
     * @param theta_n_ZOA Output: cluster zenith angles of arrival
     * @param gen Random number generator
     * @param uniformDist Uniform distribution for random generation
     * @param normalDist Normal distribution for random generation
     */
    static void genClusterAngle(uint8_t nCluster,
                                float C_ASA, float C_ASD,
                                float C_phi_NLOS, float C_phi_LOS, float c_phi_O2I,
                                float C_theta_LOS, float C_theta_NLOS, float C_theta_O2I,
                                float ASA, float ASD, float ZSA, float ZSD,
                                float phi_LOS_AOA, float phi_LOS_AOD,
                                float theta_LOS_ZOA, float theta_LOS_ZOD, float mu_offset_ZOD,
                                bool losInd, bool outdoor_ind, float K,
                                float* powers,
                                float* phi_n_AoA, float* phi_n_AoD,
                                float* theta_n_ZOD, float* theta_n_ZOA,
                                std::mt19937& gen,
                                std::uniform_real_distribution<float>& uniformDist,
                                std::normal_distribution<float>& normalDist);

    /**
     * @brief Generate ray angles within clusters
     * 
     * Generates individual ray angles within each cluster according to 3GPP specifications.
     * Uses standardized ray offset angles and random permutations.
     * 
     * @param nCluster Number of clusters
     * @param nRayPerCluster Number of rays per cluster
     * @param phi_n_AoA Cluster azimuth angles of arrival
     * @param phi_n_AoD Cluster azimuth angles of departure
     * @param theta_n_ZOD Cluster zenith angles of departure
     * @param theta_n_ZOA Cluster zenith angles of arrival
     * @param phi_n_m_AoA Output: ray azimuth angles of arrival
     * @param phi_n_m_AoD Output: ray azimuth angles of departure
     * @param theta_n_m_ZOD Output: ray zenith angles of departure
     * @param theta_n_m_ZOA Output: ray zenith angles of arrival
     * @param C_ASA Cluster ASA scaling
     * @param C_ASD Cluster ASD scaling
     * @param C_ZSA Cluster ZSA scaling
     * @param C_ZSD Cluster ZSD scaling
     * @param strongest2clustersIdx The two strongest cluster indices (Step 8 restricts their
     *                              coupling permutations to the Table 7.5-5 sub-cluster ray sets)
     * @param gen Random number generator
     * @param uniformDist Uniform distribution for random permutation
     */
    static void genRayAngle(uint8_t nCluster, uint16_t nRayPerCluster,
                            const float* phi_n_AoA, const float* phi_n_AoD,
                            const float* theta_n_ZOD, const float* theta_n_ZOA,
                            float* phi_n_m_AoA, float* phi_n_m_AoD,
                            float* theta_n_m_ZOD, float* theta_n_m_ZOA,
                            float C_ASA, float C_ASD, float C_ZSA, float C_ZSD,
                            const uint16_t* strongest2clustersIdx,
                            std::mt19937& gen,
                            std::uniform_real_distribution<float>& uniformDist);

    /**
     * @brief Generate Common Random Numbers (CRN) for correlated LSP generation
     */
    void generateCRN();

    /**
     * @brief Generate Channel Impulse Response (CIR)
     * 
     */
    void generateCIR();

    /**
     * @brief Generate Channel Frequency Response (CFR)
     * 
     */
    void generateCFR();

    /**
     * ISAC call flow (reference):
     * 1) initializeIsacParams() → initializeMonostaticRefPoints() / initializeBistaticLinks()
     * 2) calculateAllTargetLinkParams() builds per-link LSPs:
     *    calculateIncidentPath() / calculateScatteredPath(), calculateBistaticAngle(),
     *    calculateRCS(), calculateTargetDoppler()
     * 3) generateTargetCIR() with calculateTargetReflectionCoefficient()
     * 4) combineBackgroundAndTargetCIR() merges background + target CIR using IsacCirConfig tap sizing
     * Utilities: transformSpstToGCS(), updateAllTargetLocations() when mobility is applied.
     */
    // ISAC initialization (monostatic RPs or bistatic links)
    void initializeIsacParams();
    void initializeMonostaticRefPoints();
    void generateRpClusters();  //!< Generate LSP/clusters for RPs (must be called after generateCRN)
    void initializeBistaticLinks();
    void generateMonostaticReferencePoints(
        const Coordinate& stx_srx_loc,
        const MonostaticRpGammaParams& params,
        const float stx_srx_orientation[3],
        const float stx_srx_velocity[3],
        MonostaticReferencePoint rps[3]);
    void generateMonostaticBackgroundCIR(
        const MonostaticBackgroundParams& bgParams,
        const AntPanelConfig& txAntConfig,
        const AntPanelConfig& rxAntConfig,
        float fc, float lambda_0,
        uint32_t nSnapshots,
        float currentTime,
        float sampleRate,
        TargetCIR& backgroundCIR);
    void calculateIncidentPath(
        const Coordinate& tx_loc,
        const Coordinate& target_loc,
        const StParam& target,
        float fc, Scenario scenario,
        TargetIncidentPath& incident,
        uint32_t crnSiteIdx = 0);
    void calculateScatteredPath(
        const Coordinate& target_loc,
        const Coordinate& rx_loc,
        const StParam& target,
        float fc, Scenario scenario,
        TargetScatteredPath& scattered,
        bool is_monostatic,
        const TargetIncidentPath* incident_path,
        uint32_t crnSiteIdx = 0);
    float calculateRcsSigmaS(float sigma_sigma_s_db);
    float calculateUavXpr();
    float calculateRCS(const StParam& st, uint32_t spst_idx,
                       float theta_incident, float phi_incident,
                       float theta_scattered, float phi_scattered,
                       float bistatic_angle);
    void calculateAngles3D(const Coordinate& loc1, const Coordinate& loc2,
                           float& theta_ZOD, float& phi_AOD);
    float calculateBistaticAngle(float theta_incident, float phi_incident,
                                 float theta_scattered, float phi_scattered);
    float calculateTargetDoppler(const float target_velocity[3],
                                 float theta_incident, float phi_incident,
                                 float theta_scattered, float phi_scattered,
                                 float lambda_0);
    float calculateIsacLosProb(float d_2d, float h_target, const StParam& target, Scenario scenario);
    void calculateAllTargetLinkParams(bool preserveNlosCoupling = false);
    void generateTargetCIR(std::vector<TargetLinkParams>& targetLinks,
                           const AntPanelConfig& txAntConfig,
                           const AntPanelConfig& rxAntConfig,
                           uint32_t nSnapshots,
                           float currentTime,
                           float lambda_0,
                           float sampleRate,
                           const IsacCirConfig& isacConfig,
                           TargetCIR& targetCIR);
        void genTargetClustersFromLsp(float delay_spread_ns,
                                      float asd, float asa,
                                      float zsd, float zsa,
                                      float ricean_k,
                                      float phi_AOA, float phi_AOD,
                                      float theta_ZOA, float theta_ZOD,
                                      bool los,
                                      TargetClusterParams& out);
    cuComplex calculateTargetReflectionCoefficient(
        const AntPanelConfig& txAntConfig, uint32_t txAntIdx,
        const AntPanelConfig& rxAntConfig, uint32_t rxAntIdx,
        const TargetLinkParams& targetLink,
        float currentTime, float lambda_0);
    Coordinate transformSpstToGCS(const Coordinate& spst_loc_lcs,
                                  const Coordinate& target_loc_gcs,
                                  const float target_orientation[2]);
    void updateAllTargetLocations(std::vector<StParam>& targets,
                                  float current_time, float ref_time);
    void combineBackgroundAndTargetCIR(
        const cuComplex* cirCoe_bg, const uint16_t* cirNormDelay_bg, const uint16_t* cirNtaps_bg,
        const cuComplex* cirCoe_tgt, const uint16_t* cirNormDelay_tgt, const uint16_t* cirNtaps_tgt,
        cuComplex* cirCoe_combined, uint16_t* cirNormDelay_combined, uint16_t* cirNtaps_combined,
        uint32_t nRxAnt, uint32_t nTxAnt, uint32_t nSnapshots,
        const IsacCirConfig& isacConfig);
    MonostaticRpGammaParams getTrpMonostaticGammaParams(Scenario scenario);
    MonostaticRpGammaParams getUtMonostaticGammaParams(Scenario scenario, float h_ut, bool is_aerial);

    /**
     * @brief Process transmitted samples
     * 
     * Handles the processing of transmitted signal samples through the channel.
     */
    void processTxSamples() {
        // TODO: Implement this function
    }

    /**
     * @brief Allocate GPU memory for internal data structures
     */
    void allocateStaticGpuMem();
    void allocateDynamicGpuMem(uint32_t nLink);
    
    /**
     * @brief Copy data from contiguous internal storage to per-cell external arrays
     * 
     * Used when internal memory mode is enabled but external per-cell arrays are provided
     */
    void copyContiguousToPerCell(const std::vector<uint16_t>& activeCell,
                                const std::vector<std::vector<uint16_t>>& activeUt);
    
    /**
     * @brief Copy data from per-cell external arrays to contiguous internal storage
     * 
     * Used when internal memory mode is enabled and external per-cell arrays need to be processed
     */
    void copyPerCellToContiguous(const std::vector<uint16_t>& activeCell,
                                const std::vector<std::vector<uint16_t>>& activeUt);

    /**
     * @brief Calculate common link parameters
     */
    void calCmnLinkParams();

    /**
     * @brief Calculate link parameters
     */
    void calLinkParam();

    /**
     * @brief Calculate link parameters using GPU
     */
    void calLinkParamGPU();
    void initializeLargeScaleGpuKernels();
    
    /**
     * @brief Calculate cluster ray parameters using GPU
     */
    void calClusterRayGPU();

    /**
     * @brief Resolve small-scale CUDA kernels and configure their launch limits
     *
     * Called during construction so the per-TTI run path only uses cached
     * driver handles and precomputed launch limits.
     */
    void initializeSmallScaleGpuKernels();
    
    /**
     * @brief Generate Channel Impulse Response using GPU
     */
    void generateCIRGPU();
    
    /**
     * @brief Generate Channel Frequency Response using GPU
     */
    void generateCFRGPU();
    
    /**
     * @brief Generate Common Random Numbers (CRN) on GPU for correlated LSP generation
     */
    void generateCRNGPU();

    /**
     * @brief Generate sensing channel on GPU for ISAC
     * 
     * GPU implementation of sensing channel generation including:
     * - RCS calculation for all SPSTs
     * - Path computation (BS->ST->BS or BS_TX->ST->BS_RX or BS_TX->ST->UE)
     * - Doppler shift for moving targets
     */
    void generateSensingChannelGPU();

    /**
     * @brief Calculate RCS on GPU for all sensing targets and SPSTs
     * 
     * Parallel RCS computation for:
     * - Multiple sensing targets
     * - Multiple SPSTs per target (up to 5 for automotive/AGV)
     * - Both RCS Model 1 (deterministic) and Model 2 (angular dependent)
     */
    void calculateRCSGPU();

    /**
     * @brief Update active link indices based on active cells and UTs
     * 
     * @param activeCell Vector of active cell indices
     * @param activeUt Vector of vectors containing active UT indices per cell
     * 
     * If both inputs are empty, creates links for all BS-UE pairs.
     * Otherwise, creates links between each active cell and its corresponding active UTs.
     */
    void updateActiveLinkInd(const std::vector<uint16_t>& activeCell,
                           const std::vector<std::vector<uint16_t>>& activeUt);

    uint32_t m_nSiteUeLink;
    size_t m_lastAllocatedSize;
    // Add topology parameters
    topologyParam_t m_topology;  // Network topology parameters
    
    // ISAC target channel parameters
    TargetChannelParams m_targetChannelParams;  // Target channel parameters for sensing
    TargetCIR m_targetCIR;                       // Target CIR storage (reused per link)
    TargetCIR m_monostaticBackgroundCIR;         // Monostatic background CIR (BS->RPs)
    
    // Internal data structures
    size_t m_lastAllocatedLinks;
    size_t m_lastAllocatedActiveLinks;  // Track allocated active link memory
    uint32_t m_lastAllocatedMaxTaps{N_MAX_TAPS}; // Track tap budget used for current allocations
    uint32_t m_effectiveBsAnt{0}; // Effective BS antennas used for allocations (comm vs ISAC)
    uint32_t m_effectiveUeAnt{0}; // Effective UE antennas used for allocations (comm vs ISAC)
    float m_refTime{0.0f};       // reference time for CIR and CFR generation
    float m_prevRefTime{0.0f};   // previous ref time (for mobility delta_t)
    bool m_hasPrevRefTime{false};
    
    /** Memory ownership clarification */
    
    /** OWNING POINTERS - Internal contiguous storage (for modes 1,2 - performance)
     * 
     * MEMORY OWNERSHIP CONTRACT:
     * - These pointers OWN the memory they point to
     * - Memory is allocated via cudaMalloc() in allocateInternalMemory()
     * - Memory is deallocated via cudaFree() in deallocateInternalMemory()
     * - NEVER call delete[] on these pointers - use cudaFree() only
     * - These pointers become invalid after calling deallocateInternalMemory()
     */
    Tcomplex* m_cirCoe;        // OWNING: Contiguous CIR coefficients for internal allocation
    uint16_t* m_cirNormDelay;  // OWNING: Contiguous CIR indices for internal allocation  
    uint16_t* m_cirNtaps;      // OWNING: Contiguous number of CIR taps for internal allocation
    Tcomplex* m_freqChanPrbg;  // OWNING: Contiguous CFR on PRBG data for internal allocation
    Tcomplex* m_freqChanSc;    // OWNING: Contiguous CFR on SC data for internal allocation
    
    /** NON-OWNING POINTERS - External per-cell views (for mode 0 and API compatibility) */
    // IMPORTANT: These vectors contain raw pointers that DO NOT own the memory.
    // The memory is owned and managed by external code (user-provided arrays).
    // Do NOT delete these pointers - they are views into externally managed memory.
    std::vector<Tcomplex*> m_cirCoePerCell;         // NON-OWNING: Per-cell views into external CIR arrays
    std::vector<uint16_t*> m_cirNormDelayPerCell;   // NON-OWNING: Per-cell views into external delay arrays
    std::vector<uint16_t*> m_cirNtapsPerCell;       // NON-OWNING: Per-cell views into external ntaps arrays
    std::vector<Tcomplex*> m_freqChanPrbgPerCell;   // NON-OWNING: Per-cell views into external PRBG arrays
    std::vector<Tcomplex*> m_freqChanScPerCell;     // NON-OWNING: Per-cell views into external SC arrays
    
    /** TODO: Consider clarifying ownership of this pointer */
    Tcomplex* m_rxSigOut;    // Raw pointer for received signal data (ownership TBD)

    // antenna panel configuration
    const std::vector<AntPanelConfig>* m_antPanelConfig; ///< Pointer to the active antenna panel config (external or owned)
    std::vector<AntPanelConfig>  m_ownAntPanelConfig; ///< Owned antenna panel config if no external config is provided 

    // Add link parameters
    // for generate correlated random variables
    float m_maxX;
    float m_minX;
    float m_maxY;
    float m_minY;
    std::vector<std::vector<std::vector<std::vector<float>>>> m_crnLos;  // [site][LSP][x][y]: SF, K, DS, ASD, ASA, ZSD, ZSA per site
    std::vector<std::vector<std::vector<std::vector<float>>>> m_crnNlos;  // [site][LSP][x][y]: SF, DS, ASD, ASA, ZSD, ZSA per site
    std::vector<std::vector<std::vector<std::vector<float>>>> m_crnO2i;  // [site][LSP][x][y]: SF, DS, ASD, ASA, ZSD, ZSA per site

    cmnLinkParams_t m_cmnLinkParams;
    std::vector<linkParams_t> m_linkParams;  // Link-specific parameters

    std::vector<ClusterParams> m_clusterParams;  // Cluster parameters for each link

    // Random number generators
    std::mt19937 m_gen;  // Mersenne Twister random number generator
    std::uniform_real_distribution<float> m_uniformDist;  // Uniform distribution for [0,1)
    std::normal_distribution<float> m_normalDist;  // Normal distribution with mean 0 and std dev 1

    // active link indices
    std::vector<activeLink<Tcomplex>> m_activeLinkParams;
    
    // Store active cell and UE mappings for H5 file generation
    std::vector<uint16_t> m_activeCell;
    std::vector<std::vector<uint16_t>> m_activeUt;

    // GPU buffers
    // Additional GPU memory pointers for static allocations
    CellParam* m_d_cellParams;
    UtParam* m_d_utParams;
    SystemLevelConfig* m_d_sysConfig;
    SimConfig* m_d_simConfig;
    CmnLinkParams* m_d_cmnLinkParams;
    LinkParams* m_d_linkParams;
    ClusterParams* m_d_clusterParams;

    // Per-link LOS override; the device copy is allocated lazily outside kernels.
    std::vector<uint8_t> m_losOverride;
    uint8_t* m_d_losOverride{};
    size_t m_losOverrideCapacity{};
    bool m_losOverrideDirty{};

    // Additional pointers for small-scale functions
    AntPanelConfig* m_d_antPanelConfigs;
    activeLink<Tcomplex>* m_d_activeLinkParams;
    CUfunction m_generateCirFunction{nullptr};
    CUfunction m_calLinkParamFunction{nullptr};
    CUfunction m_calClusterRayFunction{nullptr};
    CUfunction m_generateCfrRunMode1Function{nullptr};
    CUfunction m_generateCfrRunMode23Function{nullptr};
    CUfunction m_generateCfrRunMode4Function{nullptr};
    int m_cirSharedBudget{48 * 1024};
    
    // Common Random Number (CRN) arrays for correlated LSP generation
    float** m_d_crnLos{};   // CRN for LOS scenarios [nSite * m_nCrnLosLsp]
    float** m_d_crnNlos{};  // CRN for NLOS scenarios [nSite * m_nCrnNlosLsp]
    float** m_d_crnO2i{};   // CRN for O2I scenarios [nSite * m_nCrnO2iLsp]
    uint8_t m_nCrnLosLsp{7};   // SF, K, DS, ASD, ASA, ZSD, ZSA, and optional DT
    uint8_t m_nCrnNlosLsp{6};  // SF, DS, ASD, ASA, ZSD, ZSA, and optional DT
    uint8_t m_nCrnO2iLsp{6};   // SF, DS, ASD, ASA, ZSD, ZSA, and optional DT
    uint32_t m_crnSeed{};   // Seed for CRN generation
    uint32_t m_crnGridSize{}; // Grid size for CRN (calculated from spatial bounds)
    uint16_t m_crnAllocatedNSite{}; // Number of sites for which CRN was allocated (to detect size changes)
    bool m_crnGridsAllocated{}; // Track if individual CRN grids have been allocated
    
    // GPU correlation distance arrays (allocated on GPU)
    float* m_d_corrDistLos{};   // Correlation distances for LOS LSPs
    float* m_d_corrDistNlos{};  // Correlation distances for NLOS LSPs
    float* m_d_corrDistO2i{};   // Correlation distances for O2I LSPs
    
    // Universal curandState array for consistent random number generation across kernel launches
    curandState* m_d_curandStates;  // OWNING: Pre-initialized curandState array for all threads
    uint32_t m_maxCurandStates;     // Maximum number of curandState elements allocated
    
    // ISAC (Integrated Sensing and Communications) GPU buffers
    StParam* m_d_stParams;       // OWNING: Sensing target parameters on GPU
    SpstParam* m_d_spstParams;   // OWNING: Flattened SPST parameters on GPU (for all STs)
    uint32_t* m_d_spstOffsets;      // OWNING: Offset indices for accessing SPSTs per ST
    float* m_d_sensingChannelRCS;   // OWNING: Computed RCS values for each ST-BS pair
    uint32_t m_totalSpstCount;      // Total number of SPSTs across all sensing targets
};

// Allocate and destroy the concrete SLS object in the CUDA translation unit
// that defines its methods. This prevents host-only consumers compiled with a
// different C++ language mode from using their view of sizeof(slsChan).
template <typename Tscalar, typename Tcomplex>
[[nodiscard]] slsChan<Tscalar, Tcomplex>* createSlsChan(const SimConfig* simConfig,
                                                       const SystemLevelConfig* sysConfig,
                                                       const ExternalConfig* extConfig,
                                                       uint32_t randSeed,
                                                       cudaStream_t strm);

template <typename Tscalar, typename Tcomplex>
void destroySlsChan(slsChan<Tscalar, Tcomplex>* channel) noexcept;

// UE Mobility Helper Functions (implemented in sls_chan_ue_mobility.cpp)

/**
 * Calculate 3D distance between two coordinates
 * 
 * @param[in] loc1 First coordinate
 * @param[in] loc2 Second coordinate
 * @return Distance in meters
 */
inline float calculateDistance3D(const Coordinate& loc1, const Coordinate& loc2) {
    const float dx = loc1.x - loc2.x;
    const float dy = loc1.y - loc2.y;
    const float dz = loc1.z - loc2.z;
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

/**
 * Calculate 2D distance between two coordinates (ignoring z)
 * 
 * @param[in] loc1 First coordinate
 * @param[in] loc2 Second coordinate
 * @return Distance in meters
 */
inline float calculateDistance2D(const Coordinate& loc1, const Coordinate& loc2) {
    const float dx = loc1.x - loc2.x;
    const float dy = loc1.y - loc2.y;
    return std::sqrt(dx * dx + dy * dy);
}

/**
 * Get unique site locations from cell configuration
 * 
 * @param[in] cells Vector of cell parameters
 * @return Vector of unique site locations
 */
std::vector<Coordinate> getUniqueSiteLocations(const std::vector<CellParam>& cells);

/**
 * Calculate speed (magnitude of velocity vector)
 * 
 * @param[in] velocity Velocity vector [vx, vy, vz]
 * @return Speed in m/s
 */
float calculateSpeed(const float velocity[3]);

/**
 * Generate new random velocity direction with same speed
 * Uses uniform distribution on sphere for 3D or circle for 2D (vz=0)
 * 
 * @param[in,out] velocity Velocity vector to update [vx, vy, vz]
 * @param[in,out] rng Random number generator
 */
void generateRandomDirection(float velocity[3], std::mt19937& rng);

/**
 * Check if UE is too close to any site and needs redirection
 * Uses 2D distance (horizontal plane) per 3GPP specification
 * 
 * @param[in] ut_loc UE location
 * @param[in] site_locations Vector of site locations
 * @param[in] min_distance Minimum allowed 2D distance to site in meters (minBsUeDist2d)
 * @return True if UE is too close to any site
 */
bool isUtTooCloseToSite(const Coordinate& ut_loc, const std::vector<Coordinate>& site_locations, 
                        float min_distance);

/**
 * Update single UE location based on velocity and time
 * Redirects UE if it gets too close to any site (uses 2D distance check per 3GPP)
 * 
 * @param[in,out] ut UE parameter to update
 * @param[in] current_time Current simulation time in seconds
 * @param[in] ref_time Reference time (time0) in seconds
 * @param[in] site_locations Vector of site locations
 * @param[in] min_distance Minimum allowed 2D distance to site in meters (minBsUeDist2d)
 * @param[in,out] rng Random number generator for direction changes
 */
void updateUtLocation(UtParam& ut, float current_time, float ref_time, 
                      const std::vector<Coordinate>& site_locations, float min_distance,
                      std::mt19937& rng);

/**
 * Update all UE locations based on velocity and time
 * 
 * @param[in,out] uts Vector of UE parameters to update
 * @param[in] current_time Current simulation time in seconds
 * @param[in] ref_time Reference time (time0) in seconds
 * @param[in] cells Vector of cell parameters (to extract site locations)
 * @param[in] min_distance Minimum allowed 2D distance to site in meters (minBsUeDist2d)
 * @param[in] seed Random seed for direction changes (default: 0 for time-based seed)
 */
void updateAllUtLocations(std::vector<UtParam>& uts, float current_time, float ref_time,
                          const std::vector<CellParam>& cells, float min_distance,
                          uint32_t seed = 0);

// Wrap-Around Helper Functions (hexagonal cellular network wrap-around per 3GPP)
// NOTE: Wrap-around only supports 1, 7, or 19 sites (0, 1, or 2 tiers)

/**
 * Calculate minimum 2D distance to site considering wrap-around
 * Implements hexagonal wrap-around by checking 7 positions (original + 6 wrapped)
 * 
 * @param[in] ut_loc UE location
 * @param[in] site_loc Site location
 * @param[in] num_tiers Number of tiers in hexagonal network (0, 1, or 2)
 * @param[in] isd Inter-site distance in meters
 * @param[in] enable Enable wrap-around (if false, returns direct distance)
 * @return Minimum 2D distance considering wrap-around in meters
 */
float calculateMinDistance2DWithWrapAround(const Coordinate& ut_loc, const Coordinate& site_loc,
                                            uint32_t num_tiers, float isd, bool enable);

/**
 * Check if UE is too close to any site with wrap-around
 * Uses hexagonal wrap-around to check distance to all sites and their wrapped copies
 * 
 * @param[in] ut_loc UE location
 * @param[in] site_locations Vector of site locations
 * @param[in] min_distance Minimum allowed 2D distance in meters
 * @param[in] num_tiers Number of tiers in hexagonal network (0, 1, or 2)
 * @param[in] isd Inter-site distance in meters
 * @param[in] enable Enable wrap-around (if false, uses non-wrap distance check)
 * @return True if UE is too close to any site (including wrapped copies)
 */
bool isUtTooCloseToSiteWithWrapAround(const Coordinate& ut_loc, 
                                       const std::vector<Coordinate>& site_locations,
                                       float min_distance,
                                       uint32_t num_tiers, float isd, bool enable);

/**
 * Apply wrap-around to UE location
 * Wraps UE back into simulation area based on hexagonal network geometry
 * 
 * @param[in,out] ut_loc UE location to wrap
 * @param[in] num_tiers Number of tiers in hexagonal network (0, 1, or 2)
 * @param[in] isd Inter-site distance in meters
 * @param[in] enable Enable wrap-around (if false, no action)
 */
void applyWrapAroundToUe(Coordinate& ut_loc, uint32_t num_tiers, float isd, bool enable);

/**
 * Apply wrap-around to all UE locations
 * 
 * @param[in,out] uts Vector of UE parameters
 * @param[in] num_tiers Number of tiers in hexagonal network (0, 1, or 2)
 * @param[in] isd Inter-site distance in meters
 * @param[in] enable Enable wrap-around (if false, no action)
 */
void applyWrapAroundToAllUes(std::vector<UtParam>& uts, uint32_t num_tiers, float isd, bool enable);

/**
 * Update single UE location with wrap-around support
 * 
 * @param[in,out] ut UE parameter to update
 * @param[in] current_time Current simulation time in seconds
 * @param[in] ref_time Reference time (time0) in seconds
 * @param[in] site_locations Vector of site locations
 * @param[in] min_distance Minimum allowed 2D distance in meters
 * @param[in] num_tiers Number of tiers in hexagonal network (0, 1, or 2)
 * @param[in] isd Inter-site distance in meters
 * @param[in] enable Enable wrap-around (if false, uses non-wrap update)
 * @param[in,out] rng Random number generator for direction changes
 */
void updateUtLocationWithWrapAround(UtParam& ut, float current_time, float ref_time,
                                     const std::vector<Coordinate>& site_locations,
                                     float min_distance,
                                     uint32_t num_tiers, float isd, bool enable,
                                     std::mt19937& rng);

/**
 * Update all UE locations with wrap-around support
 * 
 * @param[in,out] uts Vector of UE parameters to update
 * @param[in] current_time Current simulation time in seconds
 * @param[in] ref_time Reference time (time0) in seconds
 * @param[in] cells Vector of cell parameters (to extract site locations)
 * @param[in] min_distance Minimum allowed 2D distance in meters
 * @param[in] num_tiers Number of tiers in hexagonal network (0, 1, or 2)
 * @param[in] isd Inter-site distance in meters
 * @param[in] enable Enable wrap-around (if false, uses non-wrap update)
 * @param[in] seed Random seed for direction changes (default: 0 for time-based seed)
 */
void updateAllUtLocationsWithWrapAround(std::vector<UtParam>& uts, float current_time, float ref_time,
                                         const std::vector<CellParam>& cells,
                                         float min_distance,
                                         uint32_t num_tiers, float isd, bool enable,
                                         uint32_t seed = 0);
