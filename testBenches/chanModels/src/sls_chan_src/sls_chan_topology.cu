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

// This file contains the non-template implementations for the slsChan class
// related to network topology and visualization. Template implementations
// have been moved to sls_chan.cuh.

#include <string>
#include <vector>
#include <random>
#include <cmath>
#include <cassert>
#include <stdexcept>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include "sls_chan.cuh"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <curand.h>
#include <fstream>

// Forward declaration for O2I penetration loss calculation (defined in sls_chan_large_scale.cu)
float calPenetrLos(scenario_t scenario, bool outdoor_ind, float fc, float d_2d_in, 
                   std::uint8_t o2iBuildingPenetrLossInd, std::uint8_t o2iCarPenetrLossInd,
                   std::mt19937& gen, 
                   std::uniform_real_distribution<float>& uniformDist,
                   std::normal_distribution<float>& normalDist);

// ============================================================================
// Helper: Generate 2D coordinates within a cell/deployment area
// ============================================================================
struct DroppedCoordinate {
    float x{};
    float y{};
    bool valid{false};
};

/**
 * @brief Generate 2D coordinates within a cell sector
 * 
 * Per 3GPP TR 38.901, generates UE/target position within a hexagonal cell sector.
 * Uses sqrt(uniform) for distance to achieve uniform density per unit area.
 * 
 * @param cell_loc Cell center location (x, y)
 * @param sector_orientation Sector boresight angle in degrees
 * @param cell_radius Cell radius in meters (ISD / sqrt(3))
 * @param ISD Inter-site distance in meters
 * @param min_dist_2d Minimum 2D distance from cell center
 * @param gen Random number generator
 * @param uniformDist Uniform distribution [0, 1)
 * @param max_iterations Maximum attempts to find valid position
 * @return DroppedCoordinate with (x, y) and validity flag
 */
inline DroppedCoordinate dropCoordinateInCell(
    const Coordinate& cell_loc,
    float sector_orientation,
    float cell_radius,
    float ISD,
    float min_dist_2d,
    std::mt19937& gen,
    std::uniform_real_distribution<float>& uniformDist,
    int max_iterations = 1000)
{
    DroppedCoordinate result{};
    
    for (int iter = 0; iter < max_iterations; ++iter) {
        // Generate random angle within sector coverage (-60° to +60° from sector orientation)
        float randomAngle = 2.0f * M_PI * uniformDist(gen) / 3.0f - M_PI / 3.0f;
        
        // Generate random distance with min distance constraint [rmin, rmax]
        // sqrt(uniform) ensures uniform density per unit area
        float rmin = min_dist_2d;
        float rmax = cell_radius;
        if (rmax < rmin) {
            (void)fprintf(stderr, "dropCoordinateInCell: cell_radius (%.3f) < min_dist_2d (%.3f); clamping rmax = rmin.\n", cell_radius, min_dist_2d);
            rmax = rmin;
        }
        float randomDistance = (rmax - rmin) * std::sqrt(uniformDist(gen)) + rmin;
        randomDistance = std::max(randomDistance, 0.0f);  // ensure non-negative
        
        // Check if the position is within the hexagonal cell boundary
        float tempAngle = std::abs(randomAngle);
        tempAngle = tempAngle > M_PI / 6.0f ? M_PI / 3.0f - tempAngle : tempAngle;
        float maxDistanceAngle = ISD / 2.0f / std::cos(tempAngle);
        
        if (randomDistance <= maxDistanceAngle) {
            // Valid position found - apply sector orientation
            randomAngle += sector_orientation * M_PI / 180.0f;
            result.x = std::cos(randomAngle) * randomDistance + cell_loc.x;
            result.y = std::sin(randomAngle) * randomDistance + cell_loc.y;
            result.valid = true;
            return result;
        }
    }
    
    // Fallback: deterministic position at 2 * minimum distance
    float fallbackAngle = sector_orientation * M_PI / 180.0f;
    float fallbackDist = min_dist_2d * 2.2f;
    result.x = std::cos(fallbackAngle) * fallbackDist + cell_loc.x;
    result.y = std::sin(fallbackAngle) * fallbackDist + cell_loc.y;
    result.valid = true;  // Still valid, just fallback
    return result;
}

/**
 * @brief Generate 2D coordinates within a circular deployment area
 * 
 * Alternative dropping method for deployment-area based (non-cell-specific) placement.
 * 
 * @param center_loc Center of deployment area
 * @param radius Deployment area radius
 * @param min_dist_2d Minimum 2D distance from center (optional)
 * @param gen Random number generator
 * @param uniformDist Uniform distribution [0, 1)
 * @return DroppedCoordinate with (x, y)
 */
inline DroppedCoordinate dropCoordinateInCircle(
    const Coordinate& center_loc,
    float radius,
    float min_dist_2d,
    std::mt19937& gen,
    std::uniform_real_distribution<float>& uniformDist)
{
    DroppedCoordinate result{};

    if (radius <= min_dist_2d) {
        result.valid = false;
        return result;
    }
    
    // sqrt(uniform) for uniform area density
    float r = (radius - min_dist_2d) * std::sqrt(uniformDist(gen)) + min_dist_2d;
    float theta = 2.0f * M_PI * uniformDist(gen);
    
    result.x = r * std::cos(theta) + center_loc.x;
    result.y = r * std::sin(theta) + center_loc.y;
    result.valid = true;
    
    return result;
}

template <typename Tscalar, typename Tcomplex>
void slsChan<Tscalar, Tcomplex>::bsUeDropping()
{
    // BS and UE dropping per 3GPP TR 38.901 and TR 36.777
    // Supports:
    // - Terrestrial UEs: Indoor/outdoor ground users (heights: 1.5-30m)
    // - Aerial UEs (per 3GPP TR 36.777): UAVs/drones at configurable heights (default: 1.5-300m)
    //   * Can be indoor or outdoor (configurable via aerial_ue_indoor_ratio)
    //   * 3D mobility with vertical velocity component (vz)
    //   * Antenna orientation depends on height (low: horizontal, high: downward-looking)
    //
    // Aerial UE ratio per TR 36.777:
    // aerial_ue_ratio = N_aerial / (N_outdoor_terrestrial + N_indoor_terrestrial)
    // Total UEs = N_terrestrial + N_aerial
    
    // Clear existing topology
    m_topology.cellParams.clear();
    m_topology.utParams.clear();
    m_topology.n_sector_per_site = m_sysConfig->n_sector_per_site;
    m_topology.nSite = m_sysConfig->n_site;
    m_topology.nSector = m_sysConfig->n_site * m_topology.n_sector_per_site;
    m_topology.nUT = m_sysConfig->n_ut;
    
    // Support n_sector_per_site = 1 for single-site single-cell simulations
    // Multi-site scenarios require 3 sectors per site for proper hexagonal layout
    if (m_topology.nSite == 1) {
        assert((m_topology.n_sector_per_site == 1 || m_topology.n_sector_per_site == 3) && 
               "Single site supports 1 or 3 sectors per site");
    } else {
        assert(m_topology.n_sector_per_site == 3 && 
               "Multi-site deployments require 3 sectors per site for hexagonal layout");
    }
    
    // Calculate split between terrestrial and aerial UEs
    // aerial_ue_ratio = N_aerial / N_terrestrial
    // n_ut = N_terrestrial + N_aerial
    // Solving: N_terrestrial = n_ut / (1 + aerial_ue_ratio)
    //          N_aerial = n_ut * aerial_ue_ratio / (1 + aerial_ue_ratio)
    const uint32_t n_terrestrial = static_cast<uint32_t>(
        m_topology.nUT / (1.0f + m_sysConfig->aerial_ue_ratio)
    );
    const uint32_t n_aerial = m_topology.nUT - n_terrestrial;
    
    if (m_sysConfig->aerial_ue_ratio > 0.0f) {
        printf("UE distribution: %u terrestrial + %u aerial = %u total (aerial ratio: %.2f%%)\n",
               n_terrestrial, n_aerial, m_topology.nUT, 
               m_sysConfig->aerial_ue_ratio * 100.0f);
    }

    // Set scenario-specific parameters
    switch (m_sysConfig->scenario) {
        case Scenario::UMa:
            m_topology.ISD = 500.0f;
            m_topology.bsHeight = 25.0f;
            m_topology.minBsUeDist2d = 35.0f;
            m_topology.maxBsUeDist2dIndoor = 25.0f;
            m_topology.indoorUtPercent = 0.8f;
            break;
        case Scenario::UMi:
            m_topology.ISD = 200.0f;
            m_topology.bsHeight = 10.0f;
            m_topology.minBsUeDist2d = 10.0f;
            m_topology.maxBsUeDist2dIndoor = 25.0f;
            m_topology.indoorUtPercent = 0.8f;
            break;
        case Scenario::RMa:
            assert(m_sysConfig->isd == 1732.0f || m_sysConfig->isd == 5000.0f);
            m_topology.ISD = m_sysConfig->isd;
            m_topology.bsHeight = 35.0f;
            m_topology.minBsUeDist2d = 35.0f;
            m_topology.maxBsUeDist2dIndoor = 10.0f;
            m_topology.indoorUtPercent = 0.5f;
            break;
        default:
            assert(false && "Unknown scenario");
            break;
    }

    // force indoor percentage
    if (m_sysConfig->force_indoor_ratio >= 0 && m_sysConfig->force_indoor_ratio <= 1) {
        m_topology.indoorUtPercent = m_sysConfig->force_indoor_ratio;
    }

    float cellRadius = m_topology.ISD / std::sqrt(3.0f); // cell radius is ISD/sqrt(3)
    // Sector orientations for 3-sector deployment (120° apart)
    // For single-sector deployment, only sectorOrien[0] is used (boresight direction)
    float sectorOrien[3] = {30.0f, 150.0f, 270.0f};  // Sector orientations
    const float cfgUeDistMin = m_sysConfig->ut_cell_2d_dist[0];
    const float cfgUeDistMax = m_sysConfig->ut_cell_2d_dist[1];
    const bool hasMixedSentinel = ((cfgUeDistMin < 0.0f) != (cfgUeDistMax < 0.0f));
    if (hasMixedSentinel) {
        throw std::runtime_error("Invalid ut_cell_2d_dist: use [-1, -1] for default or provide both [min, max]");
    }
    const bool useConfiguredUeDistRange = (cfgUeDistMin >= 0.0f && cfgUeDistMax >= 0.0f);
    if (useConfiguredUeDistRange) {
        if (cfgUeDistMax > cellRadius) {
            throw std::runtime_error("Invalid ut_cell_2d_dist: max must be <= cell radius");
        }
        if (cfgUeDistMin > cfgUeDistMax) {
            throw std::runtime_error("Invalid ut_cell_2d_dist: min must be <= max");
        }
        if (cfgUeDistMin < m_topology.minBsUeDist2d) {
            printf("Warning: ut_cell_2d_dist min (%.1f) is less than 3GPP minimum BS-UE distance (%.1f)\n",
                   cfgUeDistMin, m_topology.minBsUeDist2d);
        }
    }
    
    // Generate BS positions in hexagonal layout
    for (uint32_t siteIdx = 0; siteIdx < m_sysConfig->n_site; siteIdx++) {
        float coorX = 0.0f;
        float coorY = 0.0f;
        
        // Calculate cell center position
        if (siteIdx < 1) {
            // Center site
            coorX = 0.0f;
            coorY = 0.0f;
        }
        else if (siteIdx < 7) {
            // First ring (6 sites)
            float angle = (siteIdx - 1) * M_PI / 3.0f + M_PI / 6.0f;
            coorX = cos(angle) * m_topology.ISD;
            coorY = sin(angle) * m_topology.ISD;
        }
        else if (siteIdx < 19) {
            float angle = (siteIdx - 7) * M_PI / 6.0f;
            if (siteIdx % 2 == 1) {
                coorX = cos(angle) * 3.0f * cellRadius;
                coorY = sin(angle) * 3.0f * cellRadius;
            }
            else {
                coorX = cos(angle) * 2.0f * m_topology.ISD;
                coorY = sin(angle) * 2.0f * m_topology.ISD;
            }
        }
        else {
            fprintf(stderr, "unsupported number of sites: %d\n", m_topology.nSite);
            exit(-1);
        }
        
        // Create BS parameters
        CellParam cell;
        cell.siteId = siteIdx;
        cell.loc = {coorX, coorY, m_topology.bsHeight};
        cell.antPanelIdx = 0;
        if (m_sysConfig->scenario == Scenario::UMa || m_sysConfig->scenario == Scenario::UMi) {
            cell.antPanelOrientation[0] = 12.0f;  // default value for UMa/UMi
        }
        else {
            cell.antPanelOrientation[0] = 10.0f;  // default value for RMa
        }
        cell.antPanelOrientation[1] = 0.0f;
        cell.antPanelOrientation[2] = 0.0f;
        
        for (int i = 0; i < m_topology.n_sector_per_site; i++) {
            cell.cid = siteIdx * m_topology.n_sector_per_site + i;
            cell.antPanelOrientation[1] = sectorOrien[i];
            m_topology.cellParams.push_back(cell);
        }
    }
    if (m_topology.cellParams.size() != m_topology.nSector) {
        throw std::runtime_error("cellParams.size() (" + std::to_string(m_topology.cellParams.size()) +
                                 ") must equal nSector (" + std::to_string(m_topology.nSector) + ")");
    }

    // Build UE drop-cell pool from config. If empty, default to all sectors.
    std::vector<uint32_t> dropCells;
    if (m_sysConfig->n_ut_drop_cells > 0) {
        dropCells.reserve(m_sysConfig->n_ut_drop_cells);
        for (uint32_t i = 0; i < m_sysConfig->n_ut_drop_cells; ++i) {
            const uint32_t cid = m_sysConfig->ut_drop_cells[i];
            if (cid >= m_topology.nSector) {
                throw std::runtime_error("ut_drop_cells contains invalid cell id " + std::to_string(cid) +
                                         " (must be < n_site * n_sector_per_site = " + std::to_string(m_topology.nSector) + ")");
            }
            if (std::find(dropCells.begin(), dropCells.end(), cid) == dropCells.end()) {
                dropCells.push_back(cid);
            }
        }
    }
    if (dropCells.empty()) {
        dropCells.resize(m_topology.nSector);
        for (uint32_t cid = 0; cid < m_topology.nSector; ++cid) {
            dropCells[cid] = cid;
        }
    }

    // Group drop cells by site for ut_drop_option=1.
    std::vector<std::vector<uint32_t>> dropCellsBySite(m_topology.nSite);
    std::vector<uint32_t> activeDropSites;
    for (const uint32_t cid : dropCells) {
        const uint32_t siteIdx = cid / m_topology.n_sector_per_site;
        if (siteIdx < m_topology.nSite) {
            if (dropCellsBySite[siteIdx].empty()) {
                activeDropSites.push_back(siteIdx);
            }
            dropCellsBySite[siteIdx].push_back(cid);
        }
    }

    auto drawDropCell = [&](const std::vector<uint32_t>& cells) -> uint32_t {
        if (cells.empty()) {
            throw std::runtime_error("ut_drop_cells sampling set is empty");
        }
        std::uniform_int_distribution<uint32_t> dist(0, static_cast<uint32_t>(cells.size() - 1));
        return cells[dist(m_gen)];
    };

    // Generate UEs
    for (uint32_t uIdx = 0; uIdx < m_topology.nUT; uIdx++) {

        // create a new UT parameter
        UtParam ut;

        // Determine if this UE is aerial based on calculated split (per 3GPP TR 36.777)
        // First n_terrestrial UEs are terrestrial, remaining are aerial
        const bool is_aerial = (uIdx >= n_terrestrial);
        
        // Set UE type
        ut.ue_type = is_aerial ? UeType::AERIAL : UeType::TERRESTRIAL;

        // Set indoor/outdoor status
        // force_indoor_ratio applies to both aerial and terrestrial UEs when set
        if (m_sysConfig->force_indoor_ratio >= 0 && m_sysConfig->force_indoor_ratio <= 1) {
            ut.outdoor_ind = (m_uniformDist(m_gen) > m_sysConfig->force_indoor_ratio) ? 1 : 0;
            if (!is_aerial) {
                m_topology.indoorUtPercent = m_sysConfig->force_indoor_ratio;
            }
        } else if (is_aerial) {
            // Aerial UE indoor/outdoor based on aerial_ue_indoor_ratio
            ut.outdoor_ind = (m_uniformDist(m_gen) > m_sysConfig->aerial_ue_indoor_ratio) ? 1 : 0;
        } else {
            // Terrestrial UE indoor/outdoor based on scenario default
            ut.outdoor_ind = (m_uniformDist(m_gen) > m_topology.indoorUtPercent) ? 1 : 0;
        }

        // Select sector for this UE based on ut_drop_option, constrained to ut_drop_cells.
        uint32_t secIdx;
        switch (m_sysConfig->ut_drop_option) {
            case 0:  // Randomly across configured drop cells
                secIdx = drawDropCell(dropCells);
                break;
            case 1:  // Same number of UTs in each configured site, then randomly across that site's configured cells
                {
                    const uint32_t nActiveSite = static_cast<uint32_t>(activeDropSites.size());
                    if (nActiveSite == 0) {
                        throw std::runtime_error("activeDropSites must not be empty");
                    }
                    const uint32_t nUePerSite = m_topology.nUT / nActiveSite;
                    const uint32_t remSite = m_topology.nUT % nActiveSite;
                    uint32_t activeSiteListIdx;
                    if (nUePerSite == 0) {
                        activeSiteListIdx = uIdx;  // more active sites than UEs: first nUT active sites get one UE each
                    } else if (uIdx < remSite * (nUePerSite + 1)) {
                        activeSiteListIdx = uIdx / (nUePerSite + 1);
                    } else {
                        activeSiteListIdx = remSite + (uIdx - remSite * (nUePerSite + 1)) / nUePerSite;
                    }
                    const uint32_t siteIdx = activeDropSites[activeSiteListIdx];
                    const std::vector<uint32_t>& siteDropCells = dropCellsBySite[siteIdx];
                    secIdx = drawDropCell(siteDropCells);
                }
                break;
            case 2:  // Same number of UTs in each configured drop cell
                {
                    const uint32_t nDropCells = static_cast<uint32_t>(dropCells.size());
                    const uint32_t nUePerSector = m_topology.nUT / nDropCells;
                    const uint32_t remSec = m_topology.nUT % nDropCells;
                    uint32_t dropCellListIdx;
                    if (nUePerSector == 0) {
                        dropCellListIdx = uIdx;  // more drop cells than UEs: first nUT drop cells get one UE each
                    } else if (uIdx < remSec * (nUePerSector + 1)) {
                        dropCellListIdx = uIdx / (nUePerSector + 1);
                    } else {
                        dropCellListIdx = remSec + (uIdx - remSec * (nUePerSector + 1)) / nUePerSector;
                    }
                    secIdx = dropCells[dropCellListIdx];
                }
                break;
            default: 
                printf("unknown ut_drop_option: %d, fallback to randomly across ut_drop_cells\n", m_sysConfig->ut_drop_option);
                secIdx = drawDropCell(dropCells);
                break;
        }

        // Generate UE location
        float randomDistance = 0.0f;
        float randomAngle = 0.0f;
        const int maxIterations = 1000;  // Maximum attempts to find valid position
        int iterationCount = 0;
        bool positionFound = false;
        
        // Both aerial and terrestrial UEs use same (x, y) placement (cell-based)
        // Only the Z height differs between them
        while(iterationCount < maxIterations) {
            // Generate random angle within sector coverage (-60° to +60° from sector orientation) 
            randomAngle = 2.0f * M_PI * m_uniformDist(m_gen) / 3.0f - M_PI/3.0f;
            // Generate random distance between minBsUeDist2d and cell radius
            if (useConfiguredUeDistRange) {
                randomDistance = (cfgUeDistMax - cfgUeDistMin) * sqrt(m_uniformDist(m_gen)) + cfgUeDistMin;
            }
            else {
#ifdef CALIBRATION_CFG_
                if (ut.outdoor_ind == 0) {  // no min BS-UT distance for indoor UEs in calibration config
                    randomDistance = cellRadius * sqrt(m_uniformDist(m_gen));
                }
                else {  // min BS-UT distance for outdoor UEs in calibration config
                    randomDistance = (cellRadius - m_topology.minBsUeDist2d) * sqrt(m_uniformDist(m_gen)) + m_topology.minBsUeDist2d;
                }
#else
                // min BS-UT distance for all UEs in non-calibration config
                randomDistance = (cellRadius - m_topology.minBsUeDist2d) * sqrt(m_uniformDist(m_gen)) + m_topology.minBsUeDist2d;
#endif
            }
            // check if the UE is within the cell
            float tempAngle = std::abs(randomAngle);
            tempAngle = tempAngle > M_PI / 6.0f ? M_PI / 3.0f - tempAngle : tempAngle;  // convert to angle inside a right triangle
            float maxDistanceAngle = m_topology.ISD / 2.0f / cos(tempAngle);
            if (randomDistance <= maxDistanceAngle) {
                randomAngle += (m_topology.cellParams[secIdx].antPanelOrientation[1]) * M_PI / 180.0f;
                positionFound = true;
                break;
            }
            iterationCount++;
        }
        
        // Fallback: assign deterministic position if valid position not found
        if (!positionFound) {
            // Use deterministic fallback position at 2 * minimum distance from sector center
            randomAngle = (m_topology.cellParams[secIdx].antPanelOrientation[1]) * M_PI / 180.0f;
            randomDistance = m_topology.minBsUeDist2d * 1.1f * 2.0f;  // 2 * minimum distance
            printf("Warning: UE %u positioned using fallback after %d iterations\n", uIdx, maxIterations);
        }
        
        // Calculate UE position (same x, y for both aerial and terrestrial)
        ut.uid = uIdx;
        ut.loc.x = cos(randomAngle) * randomDistance + m_topology.cellParams[secIdx].loc.x;
        ut.loc.y = sin(randomAngle) * randomDistance + m_topology.cellParams[secIdx].loc.y;

        // Calculate height based on UE type
        if (is_aerial) {
            // Aerial UE height per 3GPP TR 36.777 calibration: uniform in [1.5m, 300m]
            ut.loc.z = m_sysConfig->aerial_ue_height_min + 
                      (m_sysConfig->aerial_ue_height_max - m_sysConfig->aerial_ue_height_min) * 
                       m_uniformDist(m_gen);
        }
        // Terrestrial UE height
        else if (m_sysConfig->scenario == Scenario::UMa || m_sysConfig->scenario == Scenario::UMi) {
            // Generate random number of total floors (N_fl) between 4 and 8 (inclusive)
            uint16_t N_fl = 4 + static_cast<uint16_t>(m_uniformDist(m_gen) * 5);  // uniformDist[4,8] - multiply by 5 to include 8
            // Generate random floor number (n_fl) between 1 and N_fl (inclusive)
            uint16_t n_floor = ut.outdoor_ind ? 1 : 1 + static_cast<uint16_t>(m_uniformDist(m_gen) * N_fl);  // uniformDist(1,N_fl) for indoor, 1 for outdoor
            ut.loc.z = 3.0f * (n_floor - 1) + 1.5f; // 3m per floor + 1.5m base height
        }
        else {
            ut.loc.z = 1.5f;
        }
        
        ut.antPanelIdx = 1;
        
        // Generate velocity based on UE type
        float direction = 2.0f * M_PI * m_uniformDist(m_gen);   
        float speed = 10.0f;
        float vertical_speed = 0.0f;  // Vertical velocity, set to 0
        
        if (is_aerial) {
            // Aerial UE velocity per 3GPP TR 36.777
            // Horizontal speed: 160 km/h typical for UAVs
            if (m_sysConfig->force_ut_speed[1] >= 0) {
                speed = m_sysConfig->force_ut_speed[1];  // Use outdoor speed config
            } else {
                speed = 160.0f;  // 160 km/h (default)
            }
            // Note: Vertical velocity kept at 0
            
        } else {
            // Terrestrial UE velocity
            if (ut.outdoor_ind == 0) {
                if (m_sysConfig->force_ut_speed[0] >= 0) {
                    speed = m_sysConfig->force_ut_speed[0];
                }
                else {
                    speed = (m_sysConfig->scenario == Scenario::RMa) ? 60.0f : 3.0f;
                }
            }
            if (ut.outdoor_ind == 1) {
                if (m_sysConfig->force_ut_speed[1] >= 0) {
                    speed = m_sysConfig->force_ut_speed[1];
                }
                else {
                    speed = 3.0f;  
                }
            }
        }
        speed /= 3.6f;  // convert km/h to m/s
        direction = 2.0f * M_PI * m_uniformDist(m_gen); 
        ut.velocity[0] = speed * cos(direction);
        ut.velocity[1] = speed * sin(direction);
        ut.velocity[2] = vertical_speed;  // Vertical velocity, set to 0
        
        // UE antenna orientation
        if (is_aerial) {
            // Aerial UE antenna orientation per 3GPP TR 36.777:
            // [0] = β (zenith/downtilt): depends on altitude
            //       - Low altitude (< 50m): near-horizontal (80-100°)
            //       - High altitude (≥ 50m): downward-looking (110-135°)
            // [1] = α (azimuth/bearing): aligned with horizontal movement direction
            // [2] = γ (slant): fixed at 0°
            const float horizontal_speed = std::sqrt(ut.velocity[0] * ut.velocity[0] + 
                                                    ut.velocity[1] * ut.velocity[1]);
            float orientation_deg = (horizontal_speed > 0.0f) 
                ? std::atan2(ut.velocity[1], ut.velocity[0]) * 180.0f / M_PI
                : 360.0f * m_uniformDist(m_gen);  // Random if stationary
            // Wrap to [0, 360) range
            if (orientation_deg < 0.0f) {
                orientation_deg += 360.0f;
            }
            
            // Zenith angle depends on altitude
            float zenith_angle;
            if (ut.loc.z < 50.0f) {
                // Low altitude: near-horizontal orientation (80-100°)
                zenith_angle = 80.0f + 20.0f * m_uniformDist(m_gen);  // [80°, 100°]
            } else {
                // High altitude: downward-looking (110-135°)
                zenith_angle = 110.0f + 25.0f * m_uniformDist(m_gen);  // [110°, 135°]
            }
            
            ut.antPanelOrientation[0] = zenith_angle;
            ut.antPanelOrientation[1] = orientation_deg;
            ut.antPanelOrientation[2] = 0.0f;
            
        } else {
            // Terrestrial UE antenna orientation per 3GPP TR 38.901:
            // Align azimuth with movement direction (physical assumption: handheld device orientation)
            // [0] = β (zenith/downtilt): fixed at 90° (horizontal)
            // [1] = α (azimuth/bearing): aligned with velocity direction (or random if stationary)
            // [2] = γ (slant): fixed at 0°
            float orientation_deg = (speed > 0.0f) 
                ? std::atan2(ut.velocity[1], ut.velocity[0]) * 180.0f / M_PI
                : 360.0f * m_uniformDist(m_gen);  // Random orientation for stationary UEs
            // Wrap to [0, 360) range
            if (orientation_deg < 0.0f) {
                orientation_deg += 360.0f;
            }
            ut.antPanelOrientation[0] = 90.0f;           // Fixed zenith at 90°
            ut.antPanelOrientation[1] = orientation_deg;  // Azimuth aligned with movement
            ut.antPanelOrientation[2] = 0.0f;            // Fixed slant at 0°
        }
        
        // Generate indoor distance for indoor UEs
        // must be within closest cell distance
        if (ut.outdoor_ind == 1) {
            ut.d_2d_in = 0.0f;
        }
        else {
            const int siteIdx = secIdx / m_topology.n_sector_per_site;
            const float closestCellDistance = sqrt(pow(m_topology.cellParams[siteIdx].loc.x - ut.loc.x, 2) + pow(m_topology.cellParams[siteIdx].loc.y - ut.loc.y, 2));
            
            // Try to generate indoor distance with iteration limit to prevent infinite loop
            int iterationCount = 0;
            while (iterationCount < maxIterations) {
#ifdef CALIBRATION_CFG_
                ut.d_2d_in = m_uniformDist(m_gen) * m_topology.maxBsUeDist2dIndoor;
#else
                ut.d_2d_in = std::min(m_uniformDist(m_gen), m_uniformDist(m_gen)) * m_topology.maxBsUeDist2dIndoor;
#endif
                // check if the indoor distance is within closest cell distance
                if (ut.d_2d_in <= closestCellDistance) {
                    break;
                }
                iterationCount++;
            }
            
            // Fallback: if max iterations reached, use uniform random within closestCellDistance
            if (iterationCount >= maxIterations) {
                ut.d_2d_in = m_uniformDist(m_gen) * closestCellDistance;
                std::fprintf(stderr, "WARNING: Max iterations (%d) reached for indoor distance generation (UE %u, closest cell distance: %.2f m, maxBsUeDist2dIndoor: %.2f m). Using fallback: %.2f m\n", 
                            maxIterations, ut.uid, closestCellDistance, m_topology.maxBsUeDist2dIndoor, ut.d_2d_in);
            }
        }

        // Generate O2I penetration loss ONCE per UE (UT-specifically generated per 3GPP TR 38.901 Section 7.4.3)
        // This value is the SAME for all BSs connecting to this UE (building characteristic, not link characteristic)
        // Applies to both terrestrial and aerial indoor UEs
        ut.o2i_penetration_loss = calPenetrLos(
            m_sysConfig->scenario,
            ut.outdoor_ind,
            m_simConfig->center_freq_hz,
            ut.d_2d_in,
            m_sysConfig->o2i_building_penetr_loss_ind,
            m_sysConfig->o2i_car_penetr_loss_ind,
            m_gen,
            m_uniformDist,
            m_normalDist
        );

        // Add UE to topology
        m_topology.utParams.push_back(ut);
    }
}

template <typename Tscalar, typename Tcomplex>
void slsChan<Tscalar, Tcomplex>::dumpTopologyToYaml(const std::string& filename) {
    std::ofstream outfile(filename);
    if (!outfile.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    // Write topology parameters
    outfile << "topology:\n";
    outfile << "  nSite: " << m_topology.nSite << "\n";
    outfile << "  nSector: " << m_topology.nSector << "\n";
    outfile << "  n_sector_per_site: " << m_topology.n_sector_per_site << "\n";
    outfile << "  nUT: " << m_topology.nUT << "\n";
    outfile << "  ISD: " << m_topology.ISD << "\n";
    outfile << "  bsHeight: " << m_topology.bsHeight << "\n";
    outfile << "  minBsUeDist2d: " << m_topology.minBsUeDist2d << "\n";
    outfile << "  maxBsUeDist2dIndoor: " << m_topology.maxBsUeDist2dIndoor << "\n";
    outfile << "  indoorUtPercent: " << m_topology.indoorUtPercent << "\n\n";

    // Write BS parameters
    outfile << "base_stations:\n";
    for (const auto& bs : m_topology.cellParams) {
        outfile << "  - cid: " << bs.cid << "\n";
        outfile << "    siteId: " << bs.siteId << "\n";
        outfile << "    location:\n";
        outfile << "      x: " << bs.loc.x << "\n";
        outfile << "      y: " << bs.loc.y << "\n";
        outfile << "      z: " << bs.loc.z << "\n";
        outfile << "    antPanelIdx: " << bs.antPanelIdx << "\n";
        outfile << "    antPanelOrientation: [" 
                << bs.antPanelOrientation[0] << ", "
                << bs.antPanelOrientation[1] << ", "
                << bs.antPanelOrientation[2] << "]\n";
    }
    outfile << "\n";

    // Write UE parameters
    outfile << "user_equipment:\n";
    for (const auto& ut : m_topology.utParams) {
        outfile << "  - uid: " << ut.uid << "\n";
        
        // UE type
        outfile << "    ue_type: ";
        switch (ut.ue_type) {
            case UeType::TERRESTRIAL: outfile << "TERRESTRIAL\n"; break;
            case UeType::VEHICLE:     outfile << "VEHICLE\n"; break;
            case UeType::AERIAL:      outfile << "AERIAL\n"; break;
            case UeType::AGV:         outfile << "AGV\n"; break;
            case UeType::RSU:         outfile << "RSU\n"; break;
            default:                  outfile << "UNKNOWN\n"; break;
        }
        
        outfile << "    location:\n";
        outfile << "      x: " << ut.loc.x << "\n";
        outfile << "      y: " << ut.loc.y << "\n";
        outfile << "      z: " << ut.loc.z << "\n";
        outfile << "    outdoor_ind: " << static_cast<int>(ut.outdoor_ind) << "\n";
        outfile << "    d_2d_in: " << ut.d_2d_in << "\n";
        outfile << "    antPanelIdx: " << ut.antPanelIdx << "\n";
        outfile << "    antPanelOrientation: ["
                << ut.antPanelOrientation[0] << ", "
                << ut.antPanelOrientation[1] << ", "
                << ut.antPanelOrientation[2] << "]\n";
        outfile << "    velocity: ["
                << ut.velocity[0] << ", "
                << ut.velocity[1] << ", "
                << ut.velocity[2] << "]\n";
        outfile << "    o2i_penetration_loss: " << ut.o2i_penetration_loss << "\n";
    }

    outfile.close();
}

// Helper function to configure ST physical size based on target type and size index
// Physical dimensions [length, width, height] in meters per 3GPP TR 38.901 Section 7.9
static void configureStPhysicalSize(StParam& st, uint8_t size_ind) {
    switch (st.target_type) {
        case SensingTargetType::UAV:
            // TR 38.901 Table 7.9.1-1: UAV physical size (size_ind matches Option number)
            if (size_ind == 0) {  // Option 1: Large UAV 1.6m x 1.5m x 0.7m
                st.physical_size[0] = 1.6f; st.physical_size[1] = 1.5f; st.physical_size[2] = 0.7f;
            } else {  // Option 2: Small UAV 0.3m x 0.4m x 0.2m
                st.physical_size[0] = 0.3f; st.physical_size[1] = 0.4f; st.physical_size[2] = 0.2f;
            }
            break;
        case SensingTargetType::AUTOMOTIVE:
            // TR 37.885 vehicle types, Section 6.1.2
            if (size_ind == 0) {  // Type 1/2: Passenger vehicle, 5.0m x 2.0m x 1.6m
                st.physical_size[0] = 5.0f; st.physical_size[1] = 2.0f; st.physical_size[2] = 1.6f;
            } else {  // Type 3: Truck/Bus, 13.0m x 2.6m x 3.0m
                st.physical_size[0] = 13.0f; st.physical_size[1] = 2.6f; st.physical_size[2] = 3.0f;
            }
            break;
        case SensingTargetType::HUMAN:
            // TR 38.901 Table 7.9.1-3:
            if (size_ind == 0) {  // Child: 0.2m x 0.3m x 1.0m
                st.physical_size[0] = 0.2f; st.physical_size[1] = 0.3f; st.physical_size[2] = 1.0f;
            } else {  // Adult: 0.5m x 0.5m x 1.75m
                st.physical_size[0] = 0.5f; st.physical_size[1] = 0.5f; st.physical_size[2] = 1.75f;
            }
            break;
        case SensingTargetType::AGV:
            // TR 38.901 Table 7.9.1-4: AGV physical size
            if (size_ind == 0) {  // Small AGV: 0.5m x 1.0m x 0.5m (Option 1)
                st.physical_size[0] = 0.5f; st.physical_size[1] = 1.0f; st.physical_size[2] = 0.5f;
            } else {  // Large AGV: 1.5m x 3.0m x 1.5m (Option 2)
                st.physical_size[0] = 1.5f; st.physical_size[1] = 3.0f; st.physical_size[2] = 1.5f;
            }
            break;
        case SensingTargetType::HAZARD:
            // TR 38.901 Table 7.9.1-5: Objects creating hazards on roads/railways
            if (size_ind == 0) {  // Child: 0.2m x 0.3m x 1.0m
                st.physical_size[0] = 0.2f; st.physical_size[1] = 0.3f; st.physical_size[2] = 1.0f;
            } else if (size_ind == 1) {  // Adult: 0.5m x 0.5m x 1.75m
                st.physical_size[0] = 0.5f; st.physical_size[1] = 0.5f; st.physical_size[2] = 1.75f;
            } else {  // Animal: 1.5m x 0.5m x 1.0m
                st.physical_size[0] = 1.5f; st.physical_size[1] = 0.5f; st.physical_size[2] = 1.0f;
            }
            break;
        default:
            // Default to small UAV size
            st.physical_size[0] = 0.3f; st.physical_size[1] = 0.4f; st.physical_size[2] = 0.2f;
    }
}

// Helper function to configure SPSTs based on target type and RCS model
// Per 3GPP TR 38.901 Section 7.9.2 and Tables 7.9.2.1-1 to 7.9.2.1-7
static void configureSpstParams(StParam& st, uint8_t size_ind) {
    st.spst_configs.clear();
    
    // Determine n_spst based on rcs_model and target type
    // RCS Model 1: angular independent (Table 7.9.2.1-1) - UAV small, Human
    // RCS Model 2: angular dependent (Tables 7.9.2.1-2 to 7.9.2.1-7) - UAV large, Human, Vehicle, AGV
    // For UAVs, TR 38.901 defines:
    // - small UAV: Model 1 (Table 7.9.2.1-1)
    // - large UAV: Model 2 (Table 7.9.2.1-2)
    const uint8_t effective_rcs_model =
        (st.target_type == SensingTargetType::UAV) ? (size_ind == 1 ? 1 : 2) : st.rcs_model;

    if (effective_rcs_model == 1) {
        st.n_spst = 1;  // Model 1: single SPST (deterministic monostatic)
    } else if (effective_rcs_model == 2) {
        // Model 2: Multiple SPSTs for automotive/AGV (5 SPSTs: front, left, back, right, roof)
        if (st.target_type == SensingTargetType::AUTOMOTIVE || st.target_type == SensingTargetType::AGV) {
            st.n_spst = 5;
        } else {
            st.n_spst = 1;
        }
    }
    
    // Get RCS parameters based on target type and RCS model per TR 38.901 Tables 7.9.2.1-1 to 7.9.2.1-7
    // σ_D is 0 dB for Model 1 (angular independent), computed from angle for Model 2
    float sigma_m_dbsm, sigma_s_db;
    switch (st.target_type) {
        case SensingTargetType::UAV:
            if (size_ind == 1) {
                // UAV small size (Table 7.9.2.1-1)
                sigma_m_dbsm = -12.81f; sigma_s_db = 3.74f;
            } else {
                // UAV large size (Table 7.9.2.1-2)
                sigma_m_dbsm = -5.85f; sigma_s_db = 2.50f;
            }
            break;
        case SensingTargetType::AUTOMOTIVE:
            // Table 7.9.2.1-4/5: Vehicle (single or multiple SPSTs)
            sigma_m_dbsm = 11.25f; sigma_s_db = 3.41f;
            break;
        case SensingTargetType::HUMAN:
            // Table 7.9.2.1-1 (Model 1) or 7.9.2.1-3 (Model 2): same values
            sigma_m_dbsm = -1.37f; sigma_s_db = 3.94f;
            break;
        case SensingTargetType::AGV:
            // Table 7.9.2.1-6/7: AGV (single or multiple SPSTs)
            sigma_m_dbsm = -4.25f; sigma_s_db = 2.51f;
            break;
        case SensingTargetType::HAZARD:
            // Uses Human RCS model (child/adult/animal on roads)
            sigma_m_dbsm = -1.37f; sigma_s_db = 3.94f;
            break;
        default:
            // Default to UAV small
            sigma_m_dbsm = -12.81f; sigma_s_db = 3.74f;
    }
    
    // Create SPSTs
    // σ_D is 0 dB for Model 1; for Model 2, it's computed during channel generation based on angles
    const float sigma_d_dbsm = 0.0f;
    
    if (st.n_spst == 1) {
        // Single SPST at target center
        SpstParam spst{};
        spst.spst_id = 0;
        spst.loc_in_st_lcs = Coordinate(0.0f, 0.0f, 0.0f);
        spst.rcs_sigma_m_dbsm = sigma_m_dbsm;
        spst.rcs_sigma_d_dbsm = sigma_d_dbsm;
        spst.rcs_sigma_s_db = sigma_s_db;
        spst.enable_forward_scattering = 1;
        st.spst_configs.push_back(spst);
    } else if (st.n_spst == 5) {
        // 5 SPSTs for automotive/AGV: front, left, back, right, roof
        // Positions relative to target center in LCS
        const float half_length = st.physical_size[0] / 2.0f;
        const float half_width = st.physical_size[1] / 2.0f;
        const float half_height = st.physical_size[2] / 2.0f;
        
        Coordinate spst_positions[5] = {
            Coordinate(half_length, 0.0f, 0.0f),   // Front
            Coordinate(0.0f, half_width, 0.0f),    // Left
            Coordinate(-half_length, 0.0f, 0.0f),  // Back
            Coordinate(0.0f, -half_width, 0.0f),   // Right
            Coordinate(0.0f, 0.0f, half_height)    // Roof
        };
        
        for (uint32_t i = 0; i < 5; ++i) {
            SpstParam spst{};
            spst.spst_id = i;
            spst.loc_in_st_lcs = spst_positions[i];
            spst.rcs_sigma_m_dbsm = sigma_m_dbsm;
            spst.rcs_sigma_d_dbsm = sigma_d_dbsm;
            spst.rcs_sigma_s_db = sigma_s_db;
            spst.enable_forward_scattering = 1;
            st.spst_configs.push_back(spst);
        }
    }
}

template <typename Tscalar, typename Tcomplex>
void slsChan<Tscalar, Tcomplex>::stDropping()
{
    // Sensing target (ST) dropping per 3GPP TR 38.901 Section 7.9
    // This function is called when isac_type is 1 (monostatic) or 2 (bistatic)
    
    // Initialize topology ST parameters
    m_topology.nST = m_sysConfig->n_st;
    m_topology.stParams.clear();
    m_topology.stParams.reserve(m_topology.nST);
    
    if (m_topology.nST == 0) {
        return;  // No sensing targets to drop
    }
    
    // Use same deployment area as UE dropping: cell radius = ISD / sqrt(3)
    // This ensures STs are placed within the same area as UEs per 3GPP TR 38.901
    // However, st_drop_radius can override this for calibration purposes
    const float defaultCellRadius = m_topology.ISD / std::sqrt(3.0f);
    const float st_drop_radius = (m_sysConfig->st_drop_radius > 0.0f) 
        ? m_sysConfig->st_drop_radius 
        : defaultCellRadius;
    
    // Determine height for STs based on distribution option per TR 38.901 Table 7.9.1-1
    // st_distribution_option[1]: vertical distribution for UAV sensing targets
    // 0: Option A - Uniform between 1.5m and 300m
    // 1: Option B - Configured height range via st_height (or legacy st_fixed_height)
    float height_min{}, height_max{};
    
    switch (m_sysConfig->st_distribution_option[1]) {
        case 0:  // Option A: Uniform [1.5m, 300m] per TR 38.901 Table 7.9.1-1
            height_min = 1.5f;
            height_max = 300.0f;
            break;
        case 1:  // Option B: Height range (uniform) per configuration
            height_min = m_sysConfig->st_height[0];
            height_max = m_sysConfig->st_height[1];
            if (height_max < height_min) {
                std::swap(height_min, height_max);
            }
            break;
        default:
            // Default to Option A
            height_min = 1.5f;
            height_max = 300.0f;
            break;
    }
    
    // Maximum attempts to find valid ST position
    constexpr int maxAttempts = 1000;
    const float min_dist_from_tx_rx = m_sysConfig->st_min_dist_from_tx_rx;
    const float min_dist_between_sts = (m_sysConfig->st_minimum_distance > 0.0f) 
        ? m_sysConfig->st_minimum_distance 
        : 10.0f;  // Default 10m minimum separation between STs
    
    // Generate sensing targets
    for (uint32_t st_idx = 0; st_idx < m_topology.nST; ++st_idx) {
        StParam st{};
        st.sid = st_idx;
        
        bool valid_position = false;
        int attempt = 0;
        
        while (!valid_position && attempt < maxAttempts) {
            // Generate random position within deployment area
            // Horizontal distribution based on st_distribution_option[0]
            float x{}, y{};
            
            switch (m_sysConfig->st_distribution_option[0]) {
                case 0:  // Option A: Uniform distribution within st_drop_radius (exclude min2D)
                {
                    auto coord = dropCoordinateInCircle(
                        Coordinate(0.0f, 0.0f, 0.0f), st_drop_radius, m_topology.minBsUeDist2d, 
                        m_gen, m_uniformDist);
                    if (!coord.valid) {
                        ++attempt;
                        continue;
                    }
                    x = coord.x;
                    y = coord.y;
                    break;
                }
                case 1:  // Option B: Uniform distribution within st_drop_radius (no exclusion zone)
                {
                    const float r = st_drop_radius * std::sqrt(m_uniformDist(m_gen));
                    const float theta = 2.0f * M_PI * m_uniformDist(m_gen);
                    x = r * std::cos(theta);
                    y = r * std::sin(theta);
                    break;
                }
                case 2:  // Option C: Per-sector dropping (use sector wedge of a cell)
                {
                    // Assign each ST to a sector in round-robin order.
                    // If n_st is a multiple of nSector, this gives equal targets per sector.
                    const uint32_t sectorIdx = (m_topology.nSector > 0) ? (st_idx % m_topology.nSector) : 0;
                    if (sectorIdx >= m_topology.cellParams.size()) {
                        // Fallback to circular dropping if sector index is invalid
                        auto coord = dropCoordinateInCircle(
                            Coordinate(0.0f, 0.0f, 0.0f), st_drop_radius, m_topology.minBsUeDist2d,
                            m_gen, m_uniformDist);
                        if (!coord.valid) {
                            ++attempt;
                            continue;
                        }
                        x = coord.x;
                        y = coord.y;
                    } else {
                        const CellParam& cell = m_topology.cellParams[sectorIdx];
                        const float cellRadius = m_topology.ISD / std::sqrt(3.0f);
                        auto coord = dropCoordinateInCell(
                            cell.loc,
                            cell.antPanelOrientation[1],  // sector boresight (degrees)
                            cellRadius,
                            m_topology.ISD,
                            m_topology.minBsUeDist2d,
                            m_gen,
                            m_uniformDist
                        );
                        x = coord.x;
                        y = coord.y;
                    }
                    break;
                }
                default:
                    // Default fallback to option 0
                    auto coord = dropCoordinateInCircle(
                        Coordinate(0.0f, 0.0f, 0.0f), st_drop_radius, m_topology.minBsUeDist2d, 
                        m_gen, m_uniformDist);
                    if (!coord.valid) {
                        ++attempt;
                        continue;
                    }
                    x = coord.x;
                    y = coord.y;
                    break;
            }
            
            // Generate random height
            const float z = height_min + (height_max - height_min) * m_uniformDist(m_gen);
            
            st.loc = Coordinate(x, y, z);
            
            // Check minimum distance from all BS positions
            bool far_enough_from_bs = true;
            for (const auto& cell : m_topology.cellParams) {
                const float dx = st.loc.x - cell.loc.x;
                const float dy = st.loc.y - cell.loc.y;
                const float dz = st.loc.z - cell.loc.z;
                const float dist_3d = std::sqrt(dx * dx + dy * dy + dz * dz);
                
                if (dist_3d < min_dist_from_tx_rx) {
                    far_enough_from_bs = false;
                    break;
                }
            }
            
            if (!far_enough_from_bs) {
                ++attempt;
                continue;
            }
            
            // Check minimum distance from all UE positions
            bool far_enough_from_ue = true;
            for (const auto& ut : m_topology.utParams) {
                const float dx = st.loc.x - ut.loc.x;
                const float dy = st.loc.y - ut.loc.y;
                const float dz = st.loc.z - ut.loc.z;
                const float dist_3d = std::sqrt(dx * dx + dy * dy + dz * dz);
                
                if (dist_3d < min_dist_from_tx_rx) {
                    far_enough_from_ue = false;
                    break;
                }
            }
            
            if (!far_enough_from_ue) {
                ++attempt;
                continue;
            }
            
            // Check minimum distance from other already placed STs
            bool far_enough_from_other_sts = true;
            for (const auto& other_st : m_topology.stParams) {
                const float dx = st.loc.x - other_st.loc.x;
                const float dy = st.loc.y - other_st.loc.y;
                const float dz = st.loc.z - other_st.loc.z;
                const float dist_3d = std::sqrt(dx * dx + dy * dy + dz * dz);
                
                if (dist_3d < min_dist_between_sts) {
                    far_enough_from_other_sts = false;
                    break;
                }
            }
            
            if (!far_enough_from_other_sts) {
                ++attempt;
                continue;
            }
            
            // Valid position found
            valid_position = true;
        }
        
        if (!valid_position) {
            std::fprintf(stderr, 
                "WARNING: Could not find valid position for ST %u after %d attempts. "
                "Consider reducing st_min_dist_from_tx_rx (current: %.2fm) or n_st.\n",
                st_idx, maxAttempts, min_dist_from_tx_rx);
            // Use last generated position anyway
        }
        
        // Set ST parameters based on configuration
        // Use configured target type or default to UAV
        // Note: LOS is link-specific and determined during link parameter calculation
        st.target_type = m_sysConfig->st_target_type;
        st.outdoor_ind = 1;  // Default to outdoor
        
        // Set velocity per 3GPP TR 38.901 (full 3D velocity vector)
        // Generate random horizontal direction with configured speed
        float speed_min = m_sysConfig->st_horizontal_speed[0];
        float speed_max = m_sysConfig->st_horizontal_speed[1];
        if (speed_max < speed_min) {
            std::swap(speed_min, speed_max);
        }
        const float horizontal_speed = speed_min + (speed_max - speed_min) * m_uniformDist(m_gen);
        const float azimuth_angle = 2.0f * M_PI * m_uniformDist(m_gen);  // Random direction [0, 2π)
        
        st.velocity[0] = horizontal_speed * std::cos(azimuth_angle);  // vx
        st.velocity[1] = horizontal_speed * std::sin(azimuth_angle);  // vy
        st.velocity[2] = m_sysConfig->st_vertical_velocity;           // vz
        
        // Set orientation (default: facing forward in direction of velocity)
        st.orientation[0] = azimuth_angle * 180.0f / M_PI;  // Azimuth matches velocity direction
        st.orientation[1] = 0.0f;  // Elevation
        
        // Set physical size based on target type and size index
        configureStPhysicalSize(st, m_sysConfig->st_size_ind);
        
        // Set RCS model from configuration
        st.rcs_model = m_sysConfig->st_rcs_model;
        
        // Configure SPSTs based on target type, RCS model, and size per TR 38.901
        configureSpstParams(st, m_sysConfig->st_size_ind);
        
        // Add ST to topology
        m_topology.stParams.push_back(st);
    }
    
    printf("ISAC: Dropped %u sensing targets with min distance %.2fm from STX/SRX\n",
           m_topology.nST, min_dist_from_tx_rx);
}

// Explicit template instantiations
template class slsChan<float, float2>;
