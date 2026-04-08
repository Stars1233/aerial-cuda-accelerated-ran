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

#include <cmath>
#include <ctime>
#include <array>

#include "sls_chan.cuh"

// UE Mobility Helper Functions Implementation
// Note: calculateDistance3D and calculateDistance2D are now inline in sls_chan.cuh

/**
 * @brief Extract unique site locations from cell configuration.
 * @param[in] cells Cell list; uniqueness is determined by `CellParam::siteId`.
 * @return Unique site coordinates preserving first-seen order.
 */
std::vector<Coordinate> getUniqueSiteLocations(const std::vector<CellParam>& cells) {
    std::vector<Coordinate> site_locations{};
    std::vector<uint32_t> processed_sites{};
    
    for (const auto& cell : cells) {
        // Check if this site has already been processed
        if (std::find(processed_sites.begin(), processed_sites.end(), cell.siteId) == processed_sites.end()) {
            site_locations.push_back(cell.loc);
            processed_sites.push_back(cell.siteId);
        }
    }
    
    return site_locations;
}

/**
 * @brief Compute the scalar speed from a 3D velocity vector.
 * @param[in] velocity Velocity vector [vx, vy, vz].
 * @return Magnitude of the velocity vector.
 */
float calculateSpeed(const float velocity[3]) {
    return std::sqrt(velocity[0] * velocity[0] + velocity[1] * velocity[1] + velocity[2] * velocity[2]);
}

/**
 * @brief Randomize velocity direction while preserving speed magnitude.
 *
 * For near-2D motion (|vz| < 1e-6), samples angle uniformly in [0, 2pi).
 * For 3D motion, samples a uniform direction on the unit sphere.
 *
 * @param[in,out] velocity Velocity vector updated in-place.
 * @param[in,out] rng Pseudo-random generator used for sampling.
 */
void generateRandomDirection(float velocity[3], std::mt19937& rng) {
    const float current_speed = calculateSpeed(velocity);
    
    if (current_speed < 1e-6f) {
        // If speed is essentially zero, keep it zero
        return;
    }
    
    // Check if this is 2D movement (vz = 0)
    const bool is_2d_movement = (std::abs(velocity[2]) < 1e-6f);
    
    if (is_2d_movement) {
        // Generate random angle in [0, 2π) for 2D movement
        std::uniform_real_distribution<float> angle_dist(0.0f, 2.0f * M_PI);
        const float angle = angle_dist(rng);
        
        velocity[0] = current_speed * std::cos(angle);
        velocity[1] = current_speed * std::sin(angle);
        velocity[2] = 0.0f;
    } else {
        // Generate uniform random direction on sphere for 3D movement
        std::uniform_real_distribution<float> uniform_dist(-1.0f, 1.0f);
        std::uniform_real_distribution<float> angle_dist(0.0f, 2.0f * M_PI);
        
        // Use spherical coordinates with uniform distribution
        const float cos_theta = uniform_dist(rng);  // cos(theta) uniform in [-1, 1]
        const float sin_theta = std::sqrt(1.0f - cos_theta * cos_theta);
        const float phi = angle_dist(rng);
        
        velocity[0] = current_speed * sin_theta * std::cos(phi);
        velocity[1] = current_speed * sin_theta * std::sin(phi);
        velocity[2] = current_speed * cos_theta;
    }
}

/**
 * @brief Check whether a UE is within a minimum 2D distance of any site.
 * @param[in] ut_loc UE location.
 * @param[in] site_locations Candidate site locations.
 * @param[in] min_distance Minimum allowed horizontal distance.
 * @return True if any site is closer than `min_distance`.
 */
bool isUtTooCloseToSite(const Coordinate& ut_loc, const std::vector<Coordinate>& site_locations,
                        const float min_distance) {
    for (const auto& site_loc : site_locations) {
        // Use 2D distance per 3GPP spec (horizontal plane only)
        const float distance = calculateDistance2D(ut_loc, site_loc);
        if (distance < min_distance) {
            return true;
        }
    }
    return false;
}

/**
 * @brief Update one UE location from velocity and apply collision avoidance.
 * @details Integrates position using delta_t = current_time - ref_time and
 * retries random redirection when the UE violates minimum site distance.
 * @param[out] ut UE state updated in-place (location and possibly velocity).
 * @param[in] current_time Current simulation time.
 * @param[in] ref_time Reference time.
 * @param[in] site_locations Unique site locations used for distance checks.
 * @param[in] min_distance Minimum allowed horizontal distance to any site.
 * @param[in,out] rng Pseudo-random generator used for redirection sampling.
 * @throws std::invalid_argument If current_time < ref_time.
 * @note Time is expected in seconds; coordinates/velocity use consistent units.
 */
void updateUtLocation(UtParam& ut, const float current_time, const float ref_time,
                      const std::vector<Coordinate>& site_locations, const float min_distance,
                      std::mt19937& rng) {
    // Calculate time delta
    const float delta_t = current_time - ref_time;
    
    if (delta_t < 0.0f) {
        // Cannot go back in time
        throw std::invalid_argument("current_time must be >= ref_time");
    }
    
    // Update position based on velocity (straight-line movement)
    ut.loc.x += ut.velocity[0] * delta_t;
    ut.loc.y += ut.velocity[1] * delta_t;
    ut.loc.z += ut.velocity[2] * delta_t;
    
    // Check if UE is too close to any site
    static constexpr int MAX_REDIRECTION_ATTEMPTS = 100;
    int redirection_attempts{};
    
    while (isUtTooCloseToSite(ut.loc, site_locations, min_distance) && 
           redirection_attempts < MAX_REDIRECTION_ATTEMPTS) {
        // Generate new random direction with same speed
        generateRandomDirection(ut.velocity, rng);
        
        // Move UE slightly away from its current position with new direction
        const float escape_distance = min_distance * 0.1f;  // Move 10% of min_distance
        const float speed = calculateSpeed(ut.velocity);
        if (speed > 1e-6f) {
            ut.loc.x += ut.velocity[0] / speed * escape_distance;
            ut.loc.y += ut.velocity[1] / speed * escape_distance;
            ut.loc.z += ut.velocity[2] / speed * escape_distance;
        }
        
        redirection_attempts++;
    }
    
    // If still too close after max attempts, reverse direction in horizontal plane to move away
    if (redirection_attempts >= MAX_REDIRECTION_ATTEMPTS && 
        isUtTooCloseToSite(ut.loc, site_locations, min_distance)) {
        ut.velocity[0] = -ut.velocity[0];
        ut.velocity[1] = -ut.velocity[1];
        ut.velocity[2] = ut.velocity[2];
    }
}

/**
 * @brief Batch-update all UE locations in-place.
 * @param[in,out] uts UE list to update.
 * @param[in] current_time Current simulation time.
 * @param[in] ref_time Reference time.
 * @param[in] cells Cell list used to derive unique site locations.
 * @param[in] min_distance Minimum allowed horizontal distance to sites.
 * @param[in] seed RNG seed; 0 uses current wall-clock time.
 * @note Calls updateUtLocation() once per UE using a shared std::mt19937.
 */
void updateAllUtLocations(std::vector<UtParam>& uts, const float current_time, const float ref_time,
                          const std::vector<CellParam>& cells, const float min_distance,
                          const uint32_t seed) {
    // Get unique site locations
    const std::vector<Coordinate> site_locations = getUniqueSiteLocations(cells);
    
    // Initialize random number generator
    std::mt19937 rng(seed == 0 ? static_cast<uint32_t>(std::time(nullptr)) : seed);
    
    // Update each UE
    for (auto& ut : uts) {
        updateUtLocation(ut, current_time, ref_time, site_locations, min_distance, rng);
    }
}

// Wrap-Around Helper Functions Implementation (Hexagonal Cellular Network)

/**
 * @brief Compute minimum 2D distance using optional hexagonal wrap-around.
 * @param[in] ut_loc UE location.
 * @param[in] site_loc Site location in the central layout.
 * @param[in] num_tiers Wrap-around tier count.
 * @param[in] isd Inter-site distance.
 * @param[in] enable Whether wrap-around is enabled.
 * @return Minimum horizontal distance to the original or wrapped site.
 * @note Uses six wrapped replicas plus the original site when enabled.
 */
float calculateMinDistance2DWithWrapAround(const Coordinate& ut_loc, const Coordinate& site_loc,
                                            const uint32_t num_tiers, const float isd, const bool enable) {
    if (!enable) {
        // No wrap-around, return direct distance
        return calculateDistance2D(ut_loc, site_loc);
    }
    
    // Hexagonal wrap-around: check 7 positions (original + 6 wrapped)
    // Per 3GPP hexagonal geometry and MATLAB reference
    const float deltaX = isd * std::sqrt(3.0f) / 2.0f;
    const float deltaY = isd / 2.0f;
    const uint32_t n = num_tiers;
    
    // Initialize with direct distance
    float min_dist = calculateDistance2D(ut_loc, site_loc);
    
    // Check 6 wrap-around positions
    std::array<std::pair<float, float>, 6> wrap_offsets = {{
        { n * deltaX,          (3 * n + 2) * deltaY},  // Position 2
        {-n * deltaX,         -(3 * n + 2) * deltaY},  // Position 3
        { (n + 1) * deltaX,   -(3 * n + 1) * deltaY},  // Position 4
        {-(n + 1) * deltaX,    (3 * n + 1) * deltaY},  // Position 5
        { (2 * n + 1) * deltaX,  deltaY},              // Position 6
        {-(2 * n + 1) * deltaX, -deltaY}               // Position 7
    }};
    
    for (const auto& offset : wrap_offsets) {
        Coordinate wrapped_site = site_loc;
        wrapped_site.x += offset.first;
        wrapped_site.y += offset.second;
        
        const float dist = calculateDistance2D(ut_loc, wrapped_site);
        min_dist = std::min(min_dist, dist);
    }
    
    return min_dist;
}

/**
 * @brief Check minimum-distance violation with optional wrap-around.
 * @param[in] ut_loc UE location.
 * @param[in] site_locations Site locations.
 * @param[in] min_distance Minimum allowed horizontal distance.
 * @param[in] num_tiers Wrap-around tier count.
 * @param[in] isd Inter-site distance.
 * @param[in] enable Whether wrap-around is enabled.
 * @return True if any site (or wrapped replica) is too close.
 * @note Distance is computed via calculateMinDistance2DWithWrapAround().
 */
bool isUtTooCloseToSiteWithWrapAround(const Coordinate& ut_loc,
                                       const std::vector<Coordinate>& site_locations,
                                       const float min_distance,
                                       const uint32_t num_tiers, const float isd, const bool enable) {
    for (const auto& site_loc : site_locations) {
        const float distance = calculateMinDistance2DWithWrapAround(ut_loc, site_loc, num_tiers, isd, enable);
        if (distance < min_distance) {
            return true;
        }
    }
    return false;
}

/**
 * @brief Wrap a UE coordinate into the approximate network extent.
 * @param[in,out] ut_loc UE coordinate modified in-place.
 * @param[in] num_tiers Wrap-around tier count.
 * @param[in] isd Inter-site distance.
 * @param[in] enable Whether wrap-around is enabled.
 * @note If disabled, this function is a no-op.
 */
void applyWrapAroundToUe(Coordinate& ut_loc, const uint32_t num_tiers, const float isd, const bool enable) {
    if (!enable) {
        return;  // Wrap-around disabled
    }
    
    // Calculate the hexagonal network extent
    // The network spans from the center site with num_tiers rings around it
    const float deltaX = isd * std::sqrt(3.0f) / 2.0f;
    const float deltaY = isd / 2.0f;
    const uint32_t n = num_tiers;
    
    // Calculate wrap boundaries (approximate rectangular bounds for hexagonal network)
    const float x_range = 2.0f * (n + 1) * deltaX;
    const float y_range = (3.0f * n + 2.0f) * deltaY;
    
    const float x_min = -x_range / 2.0f;
    const float x_max = x_range / 2.0f;
    const float y_min = -y_range / 2.0f;
    const float y_max = y_range / 2.0f;
    
    // Apply wrap-around in X direction
    if (x_range > 0.0f) {
        while (ut_loc.x < x_min) {
            ut_loc.x += x_range;
        }
        while (ut_loc.x > x_max) {
            ut_loc.x -= x_range;
        }
    }
    
    // Apply wrap-around in Y direction
    if (y_range > 0.0f) {
        while (ut_loc.y < y_min) {
            ut_loc.y += y_range;
        }
        while (ut_loc.y > y_max) {
            ut_loc.y -= y_range;
        }
    }
}

/**
 * @brief Apply wrap-around to all UEs when enabled.
 * @param[in,out] uts UE list updated in-place (`UtParam::loc` is modified).
 * @param[in] num_tiers Wrap-around tier count.
 * @param[in] isd Inter-site distance.
 * @param[in] enable Whether wrap-around is enabled.
 * @return void. Returns early with no changes when `enable` is false.
 * @note Delegates per-UE work to `applyWrapAroundToUe(ut.loc, num_tiers, isd, enable)`.
 * @see applyWrapAroundToUe
 */
void applyWrapAroundToAllUes(std::vector<UtParam>& uts, const uint32_t num_tiers, const float isd, const bool enable) {
    if (!enable) {
        return;  // Wrap-around disabled
    }
    
    for (auto& ut : uts) {
        applyWrapAroundToUe(ut.loc, num_tiers, isd, enable);
    }
}

/**
 * @brief Update one UE location with wrap-around and distance-constraint retries.
 * @param[in,out] ut UE updated in-place.
 * @param[in] current_time Current simulation time.
 * @param[in] ref_time Reference time.
 * @param[in] site_locations Site locations used for distance checks.
 * @param[in] min_distance Minimum allowed horizontal distance to sites.
 * @param[in] num_tiers Wrap-around tier count.
 * @param[in] isd Inter-site distance.
 * @param[in] enable Whether wrap-around is enabled.
 * @param[in,out] rng Pseudo-random generator for direction resampling.
 * @throws std::invalid_argument If current_time < ref_time.
 * @note Mutates both position and velocity when redirection is needed.
 */
void updateUtLocationWithWrapAround(UtParam& ut, const float current_time, const float ref_time,
                                     const std::vector<Coordinate>& site_locations,
                                     const float min_distance,
                                     const uint32_t num_tiers, const float isd, const bool enable,
                                     std::mt19937& rng) {
    // Calculate time delta
    const float delta_t = current_time - ref_time;
    
    if (delta_t < 0.0f) {
        throw std::invalid_argument("current_time must be >= ref_time");
    }
    
    // Update position based on velocity (straight-line movement)
    ut.loc.x += ut.velocity[0] * delta_t;
    ut.loc.y += ut.velocity[1] * delta_t;
    ut.loc.z += ut.velocity[2] * delta_t;
    
    // Apply wrap-around first
    applyWrapAroundToUe(ut.loc, num_tiers, isd, enable);
    
    // Check if UE is too close to any site (using wrap-around distance check)
    static constexpr int MAX_REDIRECTION_ATTEMPTS = 100;
    int redirection_attempts{};
    
    while (isUtTooCloseToSiteWithWrapAround(ut.loc, site_locations, min_distance, num_tiers, isd, enable) &&
           redirection_attempts < MAX_REDIRECTION_ATTEMPTS) {
        // Generate new random direction with same speed
        generateRandomDirection(ut.velocity, rng);
        
        // Move UE slightly away from its current position with new direction
        const float escape_distance = min_distance * 0.1f;  // Move 10% of min_distance
        const float speed = calculateSpeed(ut.velocity);
        if (speed > 1e-6f) {
            ut.loc.x += ut.velocity[0] / speed * escape_distance;
            ut.loc.y += ut.velocity[1] / speed * escape_distance;
            ut.loc.z += ut.velocity[2] / speed * escape_distance;
        }
        
        // Apply wrap-around after escape move
        applyWrapAroundToUe(ut.loc, num_tiers, isd, enable);
        
        redirection_attempts++;
    }
    
    // If still too close after max attempts, reverse direction in horizontal plane to move away
    if (redirection_attempts >= MAX_REDIRECTION_ATTEMPTS && 
        isUtTooCloseToSiteWithWrapAround(ut.loc, site_locations, min_distance, num_tiers, isd, enable)) {
        ut.velocity[0] = -ut.velocity[0];
        ut.velocity[1] = -ut.velocity[1];
        ut.velocity[2] = ut.velocity[2];  // Keep Z velocity as is
    }
}

/**
 * @brief Batch-update UE locations with optional hexagonal wrap-around.
 * @param[in,out] uts UE list updated in-place.
 * @param[in] current_time Current simulation time.
 * @param[in] ref_time Reference time.
 * @param[in] cells Cell list used to derive unique site locations.
 * @param[in] min_distance Minimum allowed horizontal distance to sites.
 * @param[in] num_tiers Wrap-around tier count (supported: 0, 1, 2).
 * @param[in] isd Inter-site distance.
 * @param[in] enable Whether wrap-around is enabled.
 * @param[in] seed RNG seed; 0 uses current wall-clock time.
 * @throws std::invalid_argument If enabled and site-count/tier mapping is invalid.
 * @note Calls updateUtLocationWithWrapAround() once per UE.
 */
void updateAllUtLocationsWithWrapAround(std::vector<UtParam>& uts, const float current_time,
                                         const float ref_time,
                                         const std::vector<CellParam>& cells,
                                         const float min_distance,
                                         const uint32_t num_tiers, const float isd, const bool enable,
                                         const uint32_t seed) {
    // Get unique site locations
    const std::vector<Coordinate> site_locations = getUniqueSiteLocations(cells);
    
    // Validate wrap-around configuration against actual site count
    if (enable) {
        const uint32_t actual_sites = site_locations.size();
        uint32_t expected_sites = 0;
        if (num_tiers == 0) expected_sites = 1;
        else if (num_tiers == 1) expected_sites = 7;
        else if (num_tiers == 2) expected_sites = 19;
        else {
            throw std::invalid_argument(
                "Wrap-around only supports 0, 1, or 2 tiers. Got " +
                std::to_string(num_tiers)
            );
        }
        
        if (actual_sites != expected_sites) {
            throw std::invalid_argument(
                "Wrap-around site count mismatch: expected " + 
                std::to_string(expected_sites) + 
                " sites for " + std::to_string(num_tiers) + 
                " tiers, but network has " + std::to_string(actual_sites) + 
                " sites. Wrap-around only supports 1, 7, or 19 sites (0, 1, or 2 tiers)."
            );
        }
    }
    
    // Initialize random number generator
    std::mt19937 rng(seed == 0 ? static_cast<uint32_t>(std::time(nullptr)) : seed);
    
    // Update each UE with wrap-around
    for (auto& ut : uts) {
        updateUtLocationWithWrapAround(ut, current_time, ref_time, site_locations, min_distance,
                                        num_tiers, isd, enable, rng);
    }
}

