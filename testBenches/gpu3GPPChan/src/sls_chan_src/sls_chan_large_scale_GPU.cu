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
#include "sls_table.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <curand.h>
#include <cassert>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>
#include <random>

// Forward declarations of device functions
__device__ void calDistGPU(const CellParam& cellParam, const UtParam& utParam,
                          float& d_2d, float& d_3d, float& d_2d_in, float& d_2d_out,
                          float& d_3d_in, float& d_3d_out);

__device__ void calLosAngleGPU(const CellParam& cellParam, const UtParam& utParam,
                              float d_3d, float& phi_los_aod, float& phi_los_aoa,
                              float& theta_los_zod, float& theta_los_zoa);

__device__ float calLosProbGPU(Scenario scenario, float d_2d_out, float h_ut, const float force_los_prob[2], uint8_t outdoor_ind, bool is_aerial = false);

// Forward declaration of normalization kernel
__global__ void normalizeCRNGridsKernel(float** crnGrids, uint32_t totalElements, int numGrids);

__device__ float calPLGPU(const CellParam& cellParam, const UtParam& utParam, Scenario scenario,
                         float fc, bool isLos, bool optionalPlInd, curandState* state,
                         float* h_e_state, bool is_aerial = false);

__device__ float calPenetrLosGPU(Scenario scenario, uint8_t outdoor_ind, float fc,
                                float d_2d_in, float d_2d_in_o2i,
                                uint8_t o2i_building_penetr_loss_ind,
                                uint8_t o2i_car_penetr_loss_ind, curandState* state);

__device__ float calSfStdGPU(Scenario scenario, bool isLos, bool isIndoor, float fc, float d_3d, float d_2d, float h_bs, float h_ut, bool optionalPlInd, bool is_aerial = false);

__device__ float getLspAtLocationGPU(float x, float y, float maxX, float minX, float maxY, float minY,
                                   const float* crnGrid, int lspIdx, int nX, int nY);

// Forward declarations for the two-stage CRN generation (noise fill, then convolution)
__global__ void fillCRNNoiseKernel(float* tempCRN, uint32_t totalPaddedElements,
                                   curandState* curandStates, uint32_t maxCurandStates);
__global__ void convolveCRNKernel(
    const float* tempCRN, float* outputCRN,
    float maxX, float minX, float maxY, float minY,
    float correlationDist
);

// GPU kernel to calculate link parameters
__global__ void calLinkParamKernel(
    const CellParam* cellParams,
    const UtParam* utParams,
    const SystemLevelConfig* sysConfig,
    const SimConfig* simConfig,
    const CmnLinkParams* cmnLinkParams,
    const float** crnLos,
    const float** crnNlos,
    const float** crnO2i,
    float maxX, float minX, float maxY, float minY,
    uint32_t nSite, uint32_t nUT, uint8_t nSectorPerSite,
    LinkParams* linkParams,
    bool updatePLAndPenetrationLoss,
    bool updateAllLSPs,
    bool updateLosState,
    curandState* curandStates,
    const uint8_t* losOverride
)
{
    // Calculate thread index for site-UT pairs (co-sited sectors share link parameters)
    uint32_t siteIdx = blockIdx.x;
    uint32_t ueIdx = blockIdx.y * blockDim.x + threadIdx.x;
    uint32_t linkIdx = siteIdx * nUT + ueIdx;

    if (ueIdx >= nUT) return;

    // Calculate distances using the site's first sector (sector 0) for co-sited calculation
    float d_2d, d_3d, d_2d_in, d_2d_out, d_3d_in, d_3d_out;
    calDistGPU(cellParams[siteIdx * nSectorPerSite], utParams[ueIdx], 
               d_2d, d_3d, d_2d_in, d_2d_out, d_3d_in, d_3d_out);

    // Store distances in link parameters
    linkParams[linkIdx].d2d = d_2d;
    linkParams[linkIdx].d2d_in = d_2d_in;
    linkParams[linkIdx].d2d_out = d_2d_out;
    linkParams[linkIdx].d3d = d_3d;
    linkParams[linkIdx].d3d_in = d_3d_in;
    linkParams[linkIdx].d3d_out = d_3d_out;

    // Calculate LOS angles
    float phi_los_aod, phi_los_aoa, theta_los_zod, theta_los_zoa;
    calLosAngleGPU(cellParams[siteIdx * nSectorPerSite], utParams[ueIdx], d_3d,
                   phi_los_aod, phi_los_aoa, theta_los_zod, theta_los_zoa);

    // Store LOS angles in link parameters
    linkParams[linkIdx].phi_LOS_AOD = phi_los_aod;
    linkParams[linkIdx].phi_LOS_AOA = phi_los_aoa;
    linkParams[linkIdx].theta_LOS_ZOD = theta_los_zod;
    linkParams[linkIdx].theta_LOS_ZOA = theta_los_zoa;

    // Load curandState for this thread once at the beginning
    const uint32_t globalThreadId = blockIdx.x * gridDim.y * blockDim.x + 
                                   blockIdx.y * blockDim.x + 
                                   threadIdx.x;
    curandState localState = curandStates[globalThreadId];
    
    // Check if this is an aerial UE (used for LOS probability and path loss)
    bool is_aerial = (utParams[ueIdx].ue_type == UeType::AERIAL);
    
    // Calculate LOS probability and determine LOS/NLOS
    // Only regenerate LOS indicator when updateLosState is true (at start or after reset)
    // According to 3GPP TR 38.901, LOS/NLOS state should remain constant during a drop
    // For aerial UEs, use 3GPP TR 36.777 Table B-1 LOS probability
    if (updateLosState) {
        float losProb = calLosProbGPU(sysConfig->scenario, d_2d_out, utParams[ueIdx].loc.z, sysConfig->force_los_prob, utParams[ueIdx].outdoor_ind, is_aerial);
        // Preserve the random stream even when a caller overrides LOS.
        const uint8_t losDrawn = (curand_uniform(&localState) <= losProb) ? 1 : 0;
        const bool hasLosOverride =
            losOverride != nullptr && losOverride[linkIdx] != SLS_LOS_OVERRIDE_NONE;
        linkParams[linkIdx].losInd = hasLosOverride
            ? static_cast<uint8_t>(losOverride[linkIdx] != 0)
            : losDrawn;
        linkParams[linkIdx].h_e = 0.0f;  // re-arm the once-per-drop h_E draw with the LOS state
    }

    // Calculate path loss (always needed for mode 1 and 2)
    // For aerial UEs, use 3GPP TR 36.777 Table B-2 path loss models
    if (updatePLAndPenetrationLoss || updateAllLSPs) {
        float pl = calPLGPU(cellParams[siteIdx * nSectorPerSite], utParams[ueIdx], sysConfig->scenario,
                            simConfig->center_freq_hz / 1e9, linkParams[linkIdx].losInd,
                            sysConfig->optional_pl_ind != 0, &localState,
                            &linkParams[linkIdx].h_e, is_aerial);
        
        // Use pre-calculated O2I penetration loss from UE parameters
        // Per 3GPP TR 38.901 Section 7.4.3: O2I is UT-specifically generated, same for ALL BSs
        const float pl_pen = utParams[ueIdx].o2i_penetration_loss;
        
        // Add penetration loss to path loss
        pl += pl_pen;
#ifdef SLS_DEBUG_
        printf("linkIdx: %d, outdoor_ind: %d, fc: %f, d_2d_in: %f, o2i_building: %d, o2i_car: %d, pl: %f, pl_pen: %f\n", 
               linkIdx, utParams[ueIdx].outdoor_ind, simConfig->center_freq_hz / 1e9, d_2d_in,
               sysConfig->o2i_building_penetr_loss_ind, sysConfig->o2i_car_penetr_loss_ind, pl, pl_pen);
#endif
        linkParams[linkIdx].pathloss = pl;
    }

    // Generate LSPs (DS, ASD, ASA, SF, K, ZSD, ZSA)            
    // Get spatially correlated random numbers for each LSP
    float utX = utParams[ueIdx].loc.x;
    float utY = utParams[ueIdx].loc.y;
    uint8_t isLos = linkParams[linkIdx].losInd;
    uint8_t isO2I = (utParams[ueIdx].outdoor_ind == 0);  // 1 if indoor (O2I), 0 if outdoor
    
    // Determine the correct index for lgDS arrays based on priority:
    // O2I (indoor) has highest priority, then LOS, then NLOS
    uint8_t lspIdx = isO2I ? 2 : isLos;
    
    // Calculate grid dimensions (must match convolveCRNKernel output)
    // Use a reasonable default correlation distance for grid calculation (use maximum expected)
    float maxCorrDist = 120.0f;  // Maximum correlation distance from sls_table.h
    float D = 3.0f * maxCorrDist;
    int h_size = 2 * (int)D + 1;
    
    // Calculate final grid dimensions after padding and convolution (same as convolveCRNKernel;
    // the +2D padding and -(2D+1)+1 convolution shrink cancel, so any D gives the same result)
    int paddedNX = (int)roundf(maxX - minX + 1.0f + 2.0f * D);
    int paddedNY = (int)roundf(maxY - minY + 1.0f + 2.0f * D);
    int nX = paddedNX - h_size + 1;  // Final grid size after convolution
    int nY = paddedNY - h_size + 1;  // Final grid size after convolution
    
    // Get site-specific LSP values from the pre-generated CRN grids. DT is the
    // final grid when propagation delay is enabled.
    const bool includeDeltaTau = sysConfig->enable_propagation_delay != 0;
    const int nLosLsp = 7 + static_cast<int>(includeDeltaTau);
    const int nNlosLsp = 6 + static_cast<int>(includeDeltaTau);
    const int nO2iLsp = 6 + static_cast<int>(includeDeltaTau);
    const float* losGrid0 = crnLos[siteIdx * nLosLsp + 0];
    const float* losGrid1 = crnLos[siteIdx * nLosLsp + 1];
    const float* losGrid2 = crnLos[siteIdx * nLosLsp + 2];
    const float* losGrid3 = crnLos[siteIdx * nLosLsp + 3];
    const float* losGrid4 = crnLos[siteIdx * nLosLsp + 4];
    const float* losGrid5 = crnLos[siteIdx * nLosLsp + 5];
    const float* losGrid6 = crnLos[siteIdx * nLosLsp + 6];
    const float* losGridDt = includeDeltaTau ? crnLos[siteIdx * nLosLsp + 7] : nullptr;

    const float* nlosGrid0 = crnNlos[siteIdx * nNlosLsp + 0];
    const float* nlosGrid1 = crnNlos[siteIdx * nNlosLsp + 1];
    const float* nlosGrid2 = crnNlos[siteIdx * nNlosLsp + 2];
    const float* nlosGrid3 = crnNlos[siteIdx * nNlosLsp + 3];
    const float* nlosGrid4 = crnNlos[siteIdx * nNlosLsp + 4];
    const float* nlosGrid5 = crnNlos[siteIdx * nNlosLsp + 5];
    const float* nlosGridDt = includeDeltaTau ? crnNlos[siteIdx * nNlosLsp + 6] : nullptr;

    const float* o2iGrid0 = crnO2i[siteIdx * nO2iLsp + 0];
    const float* o2iGrid1 = crnO2i[siteIdx * nO2iLsp + 1];
    const float* o2iGrid2 = crnO2i[siteIdx * nO2iLsp + 2];
    const float* o2iGrid3 = crnO2i[siteIdx * nO2iLsp + 3];
    const float* o2iGrid4 = crnO2i[siteIdx * nO2iLsp + 4];
    const float* o2iGrid5 = crnO2i[siteIdx * nO2iLsp + 5];
    const float* o2iGridDt = includeDeltaTau ? crnO2i[siteIdx * nO2iLsp + 6] : nullptr;
    
    // Create array of uncorrelated variables
    float uncorrVars[LOS_MATRIX_SIZE] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    
    // Check if UE is indoor
    bool isIndoor = (utParams[ueIdx].outdoor_ind == 0);
    
    if (isIndoor) {
        // For indoor UEs, always use O2I correlation regardless of LOS/NLOS
        uncorrVars[SF_IDX] = getLspAtLocationGPU(utX, utY, maxX, minX, maxY, minY, o2iGrid0, 0, nX, nY);
        uncorrVars[K_IDX] = 0.0f;  // K-factor not applicable for O2I
        uncorrVars[DS_IDX] = getLspAtLocationGPU(utX, utY, maxX, minX, maxY, minY, o2iGrid1, 1, nX, nY);
        uncorrVars[ASD_IDX] = getLspAtLocationGPU(utX, utY, maxX, minX, maxY, minY, o2iGrid2, 2, nX, nY);
        uncorrVars[ASA_IDX] = getLspAtLocationGPU(utX, utY, maxX, minX, maxY, minY, o2iGrid3, 3, nX, nY);
        uncorrVars[ZSD_IDX] = getLspAtLocationGPU(utX, utY, maxX, minX, maxY, minY, o2iGrid4, 4, nX, nY);
        uncorrVars[ZSA_IDX] = getLspAtLocationGPU(utX, utY, maxX, minX, maxY, minY, o2iGrid5, 5, nX, nY);
    } else {
        // For outdoor UEs, use LOS/NLOS correlation as before
        uncorrVars[SF_IDX] = isLos ? getLspAtLocationGPU(utX, utY, maxX, minX, maxY, minY, losGrid0, 0, nX, nY) : 
                                    getLspAtLocationGPU(utX, utY, maxX, minX, maxY, minY, nlosGrid0, 0, nX, nY);
        uncorrVars[K_IDX] = isLos ? getLspAtLocationGPU(utX, utY, maxX, minX, maxY, minY, losGrid1, 1, nX, nY) : 0.0f;
        uncorrVars[DS_IDX] = isLos ? getLspAtLocationGPU(utX, utY, maxX, minX, maxY, minY, losGrid2, 2, nX, nY) :
                                    getLspAtLocationGPU(utX, utY, maxX, minX, maxY, minY, nlosGrid1, 1, nX, nY);
        uncorrVars[ASD_IDX] = isLos ? getLspAtLocationGPU(utX, utY, maxX, minX, maxY, minY, losGrid3, 3, nX, nY) :
                                     getLspAtLocationGPU(utX, utY, maxX, minX, maxY, minY, nlosGrid2, 2, nX, nY);
        uncorrVars[ASA_IDX] = isLos ? getLspAtLocationGPU(utX, utY, maxX, minX, maxY, minY, losGrid4, 4, nX, nY) :
                                     getLspAtLocationGPU(utX, utY, maxX, minX, maxY, minY, nlosGrid3, 3, nX, nY);
        uncorrVars[ZSD_IDX] = isLos ? getLspAtLocationGPU(utX, utY, maxX, minX, maxY, minY, losGrid5, 5, nX, nY) :
                                     getLspAtLocationGPU(utX, utY, maxX, minX, maxY, minY, nlosGrid4, 4, nX, nY);
        uncorrVars[ZSA_IDX] = isLos ? getLspAtLocationGPU(utX, utY, maxX, minX, maxY, minY, losGrid6, 6, nX, nY) :
                                     getLspAtLocationGPU(utX, utY, maxX, minX, maxY, minY, nlosGrid5, 5, nX, nY);
    }

    float rDt = 0.0f;
    if (includeDeltaTau) {
        const float* dtGrid = isIndoor ? o2iGridDt : (isLos ? losGridDt : nlosGridDt);
        rDt = getLspAtLocationGPU(utX, utY, maxX, minX, maxY, minY, dtGrid, 7, nX, nY);
    }
    
    // Perform matrix-vector multiplication to get correlated variables
    float corrVars[LOS_MATRIX_SIZE] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    
    if (isIndoor) {
        // For indoor UEs, use O2I correlation matrix (6x6, no K-factor correlation)
        for (int i = 0; i < O2I_MATRIX_SIZE; i++) {
            for (int j = 0; j <= i; j++) {  // sqrtCorrMatrix is lower triangular matrix
                // Map indices to skip K-factor (O2I matrix is 6x6, LOS is 7x7)
                // O2I order: SF, DS, ASD, ASA, ZSD, ZSA (no K)
                // LOS order: SF, K,  DS, ASD, ASA, ZSD, ZSA
                const int src_i = (i >= K_IDX) ? i + 1 : i;  // Skip K index (1) in LOS array
                const int src_j = (j >= K_IDX) ? j + 1 : j;  // Skip K index (1) in LOS array
                corrVars[src_i] += cmnLinkParams->sqrtCorrMatO2i[i * O2I_MATRIX_SIZE + j] * uncorrVars[src_j];
            }
        }
    } else if (isLos) {
        // For outdoor LOS case, use all 7 variables
        for (int i = 0; i < LOS_MATRIX_SIZE; i++) {
            for (int j = 0; j <= i; j++) {  // sqrtCorrMatrix is lower triangular matrix
                corrVars[i] += cmnLinkParams->sqrtCorrMatLos[i * LOS_MATRIX_SIZE + j] * uncorrVars[j];
            }
        }
    } else {
        // For outdoor NLOS case, skip the K-factor (index 1)
        for (int i = 0; i < NLOS_MATRIX_SIZE; i++) {
            for (int j = 0; j <= i; j++) {  // sqrtCorrMatrix is lower triangular matrix
                // Map indices to skip K-factor (NLOS matrix is 6x6, LOS is 7x7)
                // NLOS order: SF, DS, ASD, ASA, ZSD, ZSA (no K)
                // LOS order:  SF, K,  DS, ASD, ASA, ZSD, ZSA
                const int src_i = (i >= K_IDX) ? i + 1 : i;  // Skip K index (1) in LOS array
                const int src_j = (j >= K_IDX) ? j + 1 : j;  // Skip K index (1) in LOS array
                corrVars[src_i] += cmnLinkParams->sqrtCorrMatNlos[i * NLOS_MATRIX_SIZE + j] * uncorrVars[src_j];
            }
        }
        // Set K-factor to 0 for NLOS
        corrVars[K_IDX] = 0.0f;
    }
    
    float mu, sigma;
    
    // 1. Shadow Fading (SF)
#ifdef SLS_DEBUG_
    printf("linkIdx: %d, center freq: %f, SF: %f, d_3d: %f, d_2d: %f\n", linkIdx, simConfig->center_freq_hz / 1e9, corrVars[SF_IDX], d_3d, d_2d);
    printf("uncorrVars: %f, %f, %f, %f, %f, %f, %f\n", uncorrVars[SF_IDX], uncorrVars[K_IDX], uncorrVars[DS_IDX], uncorrVars[ASD_IDX], uncorrVars[ASA_IDX], uncorrVars[ZSD_IDX], uncorrVars[ZSA_IDX]);
    printf("corrVars: %f, %f, %f, %f, %f, %f, %f\n", corrVars[SF_IDX], corrVars[K_IDX], corrVars[DS_IDX], corrVars[ASD_IDX], corrVars[ASA_IDX], corrVars[ZSD_IDX], corrVars[ZSA_IDX]);
#endif
    if (updatePLAndPenetrationLoss || updateAllLSPs) {
        linkParams[linkIdx].SF = corrVars[SF_IDX] * calSfStdGPU(sysConfig->scenario, isLos, isIndoor, simConfig->center_freq_hz, d_3d, d_2d,
                                                                cellParams[siteIdx * nSectorPerSite].loc.z, utParams[ueIdx].loc.z, sysConfig->optional_pl_ind != 0, is_aerial);
    }
    
    if (updateAllLSPs) {
        // 2. Ricean K-factor (K)
        mu = cmnLinkParams->mu_K[lspIdx];
        sigma = cmnLinkParams->sigma_K[lspIdx];
        linkParams[linkIdx].K = lspIdx == 1 ? corrVars[K_IDX] * sigma + mu : 0.0f;  // Only apply K-factor for LOS
        
        // 3. Delay Spread (DS)
        mu = cmnLinkParams->mu_lgDS[lspIdx];
        sigma = cmnLinkParams->sigma_lgDS[lspIdx];
        linkParams[linkIdx].DS = powf(10.0f, corrVars[DS_IDX] * sigma + mu + 9.0f);  // add 9.0f to convert from s to ns
    
#ifdef SLS_DEBUG_
        // Debug print CRN for DS
        printf("linkIdx: %d, DS CRN - uncorr: %f, corr: %f, mu: %f, sigma: %f, DS: %e\n", 
            linkIdx, uncorrVars[DS_IDX], corrVars[DS_IDX], mu, sigma, linkParams[linkIdx].DS);
#endif
    
        // 4. Azimuth Spread of Departure (ASD)
        mu = cmnLinkParams->mu_lgASD[lspIdx];
        sigma = cmnLinkParams->sigma_lgASD[lspIdx];
        float asd_temp = powf(10.0f, corrVars[ASD_IDX] * sigma + mu);
        linkParams[linkIdx].ASD = fminf(asd_temp, 104.0f);  // Limit to 104 degrees
        
        // 5. Azimuth Spread of Arrival (ASA)
        mu = cmnLinkParams->mu_lgASA[lspIdx];
        sigma = cmnLinkParams->sigma_lgASA[lspIdx];
        float asa_temp = powf(10.0f, corrVars[ASA_IDX] * sigma + mu);
        linkParams[linkIdx].ASA = fminf(asa_temp, 104.0f);  // Limit to 104 degrees
        
        // 6. Zenith Spread of Departure (ZSD)
        // Map to actual LSP values based on scenario and LOS/NLOS
        float h_ut = utParams[ueIdx].loc.z;
        float h_bs = cellParams[siteIdx * nSectorPerSite].loc.z;
        float lgfc = cmnLinkParams->lgfc;

        switch (sysConfig->scenario) {
            case Scenario::UMa:
                if (isLos) {  // LOS
                    linkParams[linkIdx].mu_lgZSD = fmaxf(-0.5f, -2.1f * (d_2d/1000.0f) - 0.01f * (h_ut - 1.5f) + 0.75f);
                    linkParams[linkIdx].sigma_lgZSD = 0.4f;
                    linkParams[linkIdx].mu_offset_ZOD = 0.0f;
                } else {  // NLOS
                    linkParams[linkIdx].mu_lgZSD = fmaxf(-0.5f, -2.1f * (d_2d/1000.0f) - 0.01f * (h_ut - 1.5f) + 0.9f);
                    linkParams[linkIdx].sigma_lgZSD = 0.49f;
                    linkParams[linkIdx].mu_offset_ZOD = 7.66f * lgfc - 5.96f - 
                        powf(10.0f, (0.208f * lgfc - 0.782f) * log10f(fmaxf(25.0f, d_2d)) + 
                        (2.03f - 0.13f * lgfc) - 0.07f * (h_ut - 1.5f));
                }
                break;
            case Scenario::UMi:
                if (isLos) {  // LOS
                    linkParams[linkIdx].mu_lgZSD = fmaxf(-0.21f, -14.8f * (d_2d/1000.0f) - 0.01f * fabsf(h_ut - h_bs) + 0.83f);
                    linkParams[linkIdx].sigma_lgZSD = 0.35f;
                    linkParams[linkIdx].mu_offset_ZOD = 0.0f;
                } else {  // NLOS
                    linkParams[linkIdx].mu_lgZSD = fmaxf(-0.5f, -3.1f * (d_2d/1000.0f) + 0.01f * fmaxf(h_ut - h_bs, 0.0f) + 0.2f);
                    linkParams[linkIdx].sigma_lgZSD = 0.35f;
                    linkParams[linkIdx].mu_offset_ZOD = -powf(10.0f, -1.5f * log10f(fmaxf(10.0f, d_2d)) + 3.3f);
                }
                break;
            case Scenario::RMa:
                if (isLos) {  // LOS
                    linkParams[linkIdx].mu_lgZSD = fmaxf(-1.0f, -0.17f * (d_2d/1000.0f) - 0.01f * (h_ut - 1.5f) + 0.22f);
                    linkParams[linkIdx].sigma_lgZSD = 0.34f;
                    linkParams[linkIdx].mu_offset_ZOD = 0.0f;
                } else {  // NLOS
                    linkParams[linkIdx].mu_lgZSD = fmaxf(-1.0f, -0.19f * (d_2d/1000.0f) - 0.01f * (h_ut - 1.5f) + 0.28f);
                    linkParams[linkIdx].sigma_lgZSD = 0.30f;
                    linkParams[linkIdx].mu_offset_ZOD = atanf((35.0f - 3.5f)/d_2d) - atanf((35.0f - 1.5f)/d_2d);
                }
                break;
            default:
                assert(false && "Unknown scenario");
        }
        mu = linkParams[linkIdx].mu_lgZSD;
        sigma = linkParams[linkIdx].sigma_lgZSD;
        float zsd_temp = powf(10.0f, corrVars[ZSD_IDX] * sigma + mu);
        linkParams[linkIdx].ZSD = fminf(zsd_temp, 52.0f);  // Limit to 52 degrees
        
        // 7. Zenith Spread of Arrival (ZSA)
        mu = cmnLinkParams->mu_lgZSA[lspIdx];
        sigma = cmnLinkParams->sigma_lgZSA[lspIdx];
        float zsa_temp = powf(10.0f, corrVars[ZSA_IDX] * sigma + mu);
        linkParams[linkIdx].ZSA = fminf(zsa_temp, 52.0f);  // Limit to 52 degrees

        // 8. Delta Tau (Excess Delay) per 3GPP TR 38.901 Table 7.6.9-1
        if (sysConfig->enable_propagation_delay != 0) {
            if (isLos) {
                linkParams[linkIdx].delta_tau = 0.0f;
            } else {
                float mu_lg_dt, sigma_lg_dt;
                switch (sysConfig->scenario) {
                    case Scenario::UMi:  mu_lg_dt = -7.5f;  sigma_lg_dt = 0.5f;  break;
                    case Scenario::UMa:  mu_lg_dt = -7.4f;  sigma_lg_dt = 0.2f;  break;
                    case Scenario::RMa:  mu_lg_dt = -8.33f; sigma_lg_dt = 0.26f; break;
                    default:             mu_lg_dt = -7.5f;  sigma_lg_dt = 0.5f;  break;
                }
                float lg_delta_tau = mu_lg_dt + sigma_lg_dt * rDt;
                linkParams[linkIdx].delta_tau = powf(10.0f, lg_delta_tau);
            }
        } else {
            linkParams[linkIdx].delta_tau = 0.0f;
        }
    }
    
    // Store updated curandState back to global memory
    curandStates[globalThreadId] = localState;
}

// Helper function to get LSP value at a specific location
__device__ float getLspAtLocationGPU(float x, float y, float maxX, float minX, float maxY, float minY,
                                   const float* crnGrid, int lspIdx, int nX, int nY) {
#ifndef SLS_DEBUG_
    (void)lspIdx;
#endif
    if (crnGrid == nullptr || nX <= 0 || nY <= 0) {
#ifdef SLS_DEBUG_
        printf("ERROR: Invalid CRN grid for LSP %d: grid=%p, nX=%d, nY=%d\n",
               lspIdx, static_cast<const void*>(crnGrid), nX, nY);
#endif
        return 0.0f;
    }

    // Calculate the normalized position within the grid (same as CPU reference)
    constexpr float kMinDomainExtent = 1e-6f;
    const float rangeX = maxX - minX;
    const float rangeY = maxY - minY;
    float normX = (rangeX < kMinDomainExtent) ? 0.5f : (x - minX) / rangeX;
    float normY = (rangeY < kMinDomainExtent) ? 0.5f : (y - minY) / rangeY;
    
    // Clamp normalized coordinates to [0, 1]
    normX = fmaxf(0.0f, fminf(1.0f, normX));
    normY = fmaxf(0.0f, fminf(1.0f, normY));
    
    // Map to grid indices (same as CPU reference)
    float gridX = normX * (nX - 1);
    float gridY = normY * (nY - 1);
    
    // Get the four nearest grid points (same as CPU reference)
    int x0 = (int)floorf(gridX);
    int y0 = (int)floorf(gridY);
    int x1 = min(x0 + 1, nX - 1);
    int y1 = min(y0 + 1, nY - 1);

    if (x0 < 0 || y0 < 0 || x1 >= nX || y1 >= nY) {
#ifdef SLS_DEBUG_
        printf("ERROR: CRN grid coordinates out of bounds for LSP %d: "
               "x0=%d, y0=%d, x1=%d, y1=%d, nX=%d, nY=%d\n",
               lspIdx, x0, y0, x1, y1, nX, nY);
#endif
        return 0.0f;
    }
    
    int idx00 = y0 * nX + x0;
    int idx10 = y0 * nX + x1;
    int idx01 = y1 * nX + x0;
    int idx11 = y1 * nX + x1;
    const int maxIdx = nX * nY - 1;
    if (idx00 < 0 || idx10 < 0 || idx01 < 0 || idx11 < 0 ||
        idx00 > maxIdx || idx10 > maxIdx || idx01 > maxIdx || idx11 > maxIdx) {
#ifdef SLS_DEBUG_
        printf("ERROR: CRN array index out of bounds for LSP %d: "
               "idx00=%d, idx10=%d, idx01=%d, idx11=%d, max=%d\n",
               lspIdx, idx00, idx10, idx01, idx11, maxIdx);
#endif
        return 0.0f;
    }
    
    // Get the fractional parts for interpolation (same as CPU reference)
    float dx = gridX - x0;
    float dy = gridY - y0;
    
    // Perform bilinear interpolation (same as CPU reference)
    float v00 = crnGrid[idx00];
    float v10 = crnGrid[idx10];
    float v01 = crnGrid[idx01];
    float v11 = crnGrid[idx11];
    
    float v0 = v00 * (1.0f - dx) + v10 * dx;
    float v1 = v01 * (1.0f - dx) + v11 * dx;
    
    return v0 * (1.0f - dy) + v1 * dy;
}


// GPU helper functions
__device__ void calDistGPU(const CellParam& cellParam, const UtParam& utParam,
                          float& d_2d, float& d_3d, float& d_2d_in, float& d_2d_out,
                          float& d_3d_in, float& d_3d_out) {
    // Calculate total 2D distance
    d_2d = sqrtf(powf(cellParam.loc.x - utParam.loc.x, 2) + powf(cellParam.loc.y - utParam.loc.y, 2));
    
    // Use the pre-calculated indoor distance from UT parameters
    d_2d_in = utParam.d_2d_in;
    
    // Calculate outdoor 2D distance
    d_2d_out = d_2d - d_2d_in;
    
    // Calculate vertical distance
    float vertical_dist = cellParam.loc.z - utParam.loc.z;
    
    // Calculate all 3D distances
    d_3d = sqrtf(d_2d * d_2d + vertical_dist * vertical_dist);
    d_3d_in = (d_2d > 0.0f) ? d_3d * d_2d_in / d_2d : 0.0f;
    d_3d_out = d_3d - d_3d_in;
}

__device__ void calLosAngleGPU(const CellParam& cellParam, const UtParam& utParam,
                              float d_3d, float& phi_los_aod, float& phi_los_aoa,
                              float& theta_los_zod, float& theta_los_zoa) {
    float site2ut_x = utParam.loc.x - cellParam.loc.x;
    float site2ut_y = utParam.loc.y - cellParam.loc.y;
    
    // Calculate LOS AOD and AOA (azimuth angles)
    phi_los_aod = atan2f(site2ut_y, site2ut_x) * 180.0f / M_PI;  // Convert to degrees
    phi_los_aoa = phi_los_aod + 180.0f;  // AOA is opposite to AOD
    
    // Normalize angles to [-180, 180] range
    if (phi_los_aoa > 180.0f) {
        phi_los_aoa -= 360.0f;
    }
    
    // Calculate LOS ZOD and ZOA (zenith angles)
    float h_diff = cellParam.loc.z - utParam.loc.z;
    theta_los_zod = (M_PI - acosf(h_diff / d_3d)) * 180.0f / M_PI;  // Convert to degrees
    theta_los_zoa = 180.0f - theta_los_zod;  // ZOA is complementary to ZOD
}

__device__ float calLosProbGPU(Scenario scenario, float d_2d_out, float h_ut, const float force_los_prob[2], uint8_t outdoor_ind, bool is_aerial) {
    // Check if force_los_prob should be used instead of 3GPP calculations
    // force_los_prob[0] for indoor UTs, force_los_prob[1] for outdoor UEs
    float forced_prob = outdoor_ind ? force_los_prob[1] : force_los_prob[0];
    if (forced_prob >= 0.0f && forced_prob <= 1.0f) {
        return forced_prob;  // Use forced value instead of 3GPP calculation
    }
    
    // Use 3GPP LOS probability calculations
    // For aerial UEs: 3GPP TR 36.777 Table B-1
    // For terrestrial UEs: 3GPP TR 38.901 Table 7.4.2-1
    float losProb = 0.0f;
    
    if (is_aerial) {
        // Aerial UE LOS probability per 3GPP TR 36.777 Table B-1
        // P_LOS = 1 if d_2D <= d_1
        // P_LOS = d_1/d_2D + exp(-d_2D/p_1) * (1 - d_1/d_2D) if d_2D > d_1
        float d1 = 18.0f;
        float p1 = 1000.0f;
        
        switch (scenario) {
            case Scenario::RMa:
                if (h_ut <= 10.0f) {
                    // h_UT ≤ 10m: Use TR 38.901 RMa P_LOS (terrestrial formula)
                    if (d_2d_out <= 10.0f) {
                        losProb = 1.0f;
                    } else {
                        losProb = expf(-(d_2d_out - 10.0f) / 1000.0f);
                    }
                } else if (h_ut <= 40.0f) {
                    // 10m < h_UT ≤ 40m: Use aerial formula
                    d1 = fmaxf(1350.8f * log10f(h_ut) - 1602.0f, 18.0f);
                    p1 = fmaxf(15021.0f * log10f(h_ut) - 16053.0f, 1000.0f);
                    if (d_2d_out <= d1) {
                        losProb = 1.0f;
                    } else {
                        losProb = (d1 / d_2d_out) + expf(-d_2d_out / p1) * (1.0f - d1 / d_2d_out);
                    }
                } else {
                    // h_UT > 40m: 100% LOS
                    losProb = 1.0f;
                }
                break;
            case Scenario::UMa:
                if (h_ut <= 22.5f) {
                    // h_UT ≤ 22.5m: Use TR 38.901 UMa P_LOS (terrestrial formula)
                    if (d_2d_out <= 18.0f) {
                        losProb = 1.0f;
                    } else {
                        float c_prime = h_ut <= 13.0f ? 0.0f : powf((h_ut - 13.0f) / 10.0f, 1.5f);
                        losProb = ((18.0f / d_2d_out) + expf(-d_2d_out / 63.0f) * (1.0f - 18.0f / d_2d_out)) *
                                 (1.0f + c_prime * 5.0f / 4.0f * powf(d_2d_out / 100.0f, 3.0f) * expf(-d_2d_out / 150.0f));
                    }
                } else if (h_ut <= 100.0f) {
                    // 22.5m < h_UT ≤ 100m: Use aerial formula
                    d1 = fmaxf(460.0f * log10f(h_ut) - 700.0f, 18.0f);
                    p1 = 4300.0f * log10f(h_ut) - 3800.0f;
                    if (d_2d_out <= d1) {
                        losProb = 1.0f;
                    } else {
                        losProb = (d1 / d_2d_out) + expf(-d_2d_out / p1) * (1.0f - d1 / d_2d_out);
                    }
                } else {
                    // h_UT > 100m: 100% LOS
                    losProb = 1.0f;
                }
                break;
            case Scenario::UMi:
                if (h_ut <= 22.5f) {
                    // h_UT ≤ 22.5m: Use TR 38.901 UMi P_LOS (terrestrial formula)
                    if (d_2d_out <= 18.0f) {
                        losProb = 1.0f;
                    } else {
                        losProb = (18.0f / d_2d_out) + expf(-d_2d_out / 36.0f) * (1.0f - 18.0f / d_2d_out);
                    }
                } else {
                    // 22.5m < h_UT ≤ 300m: Use aerial formula
                    d1 = fmaxf(294.05f * log10f(h_ut) - 432.94f, 18.0f);
                    p1 = 233.98f * log10f(h_ut) - 0.95f;
                    if (d_2d_out <= d1) {
                        losProb = 1.0f;
                    } else {
                        losProb = (d1 / d_2d_out) + expf(-d_2d_out / p1) * (1.0f - d1 / d_2d_out);
                    }
                }
                break;
            default:
                assert(false && "Unknown scenario");
                break;
        }
    } else {
        // Terrestrial UE LOS probability per 3GPP TR 38.901 Table 7.4.2-1
        switch (scenario) {
            case Scenario::UMa:
                assert(h_ut <= 23.0f && "UE height must be less than 23m for terrestrial UMa");
                if (d_2d_out <= 18.0f) {
                    losProb = 1.0f;
                } else {
                    float c_prime = h_ut <= 13.0f ? 0.0f : powf((h_ut - 13.0f) / 10.0f, 1.5f);
                    losProb = ((18.0f / d_2d_out) + expf(-d_2d_out / 63.0f) * (1.0f - 18.0f / d_2d_out)) *
                             (1.0f + c_prime * 5.0f / 4.0f * powf(d_2d_out / 100.0f, 3.0f) * expf(-d_2d_out / 150.0f));
                }
                break;
            case Scenario::UMi:
                if (d_2d_out <= 18.0f) {
                    losProb = 1.0f;
                } else {
                    losProb = (18.0f / d_2d_out) + expf(-d_2d_out / 36.0f) * (1.0f - 18.0f / d_2d_out);
                }
                break;
            case Scenario::RMa:
                if (d_2d_out <= 10.0f) {
                    losProb = 1.0f;
                } else {
                    losProb = expf(-(d_2d_out - 10.0f) / 1000.0f);
                }
                break;
            default:
                assert(false && "Unknown scenario");
                break;
        }
    }
    return losProb;
}

// GPU version of UMa LOS path loss calculation (matches CPU implementation)
// h_e_state: persistent per-link effective environment height (Table 7.4.1-1 NOTE 1); the
// probabilistic h_E is drawn once per link and reused on every later PL evaluation
// (*h_e_state < 1 means "not drawn yet")
__device__ float calculateUMaLosPathlossGPU(float d_2d, float d_3d, float h_bs, float h_ut, float fc, curandState* state, float* h_e_state) {
    float d_2d_valid = fmaxf(d_2d, 10.0f);
    float g_d2d = d_2d_valid <= 18.0f ? 0.0f : 5.0f/4.0f * powf(d_2d_valid / 100.0f, 3.0f) * expf(-d_2d_valid / 150.0f);
    float c_d2d_hut = h_ut < 13.0f ? 0.0f : powf((h_ut - 13.0f) / 10.0f, 1.5f) * g_d2d;
    float prob_h_e = 1.0f / (1.0f + c_d2d_hut);
    
    float h_e;
    if (*h_e_state >= 1.0f) {
        h_e = *h_e_state;  // reuse the per-link draw (NOTE 1: h_E is drawn once per link)
    } else if (curand_uniform(state) <= prob_h_e) {
        h_e = 1.0f;  // With probability 1/(1+C(d2D, hUT))
        *h_e_state = h_e;
    } else {
        // Use random number for discrete uniform distribution. curand_uniform() returns
        // (0, 1] (unlike std::uniform_real_distribution's [0, 1)), so clamp the bucket
        // index: a draw of exactly 1.0 must not push h_e past max_h_e.
        float max_h_e = h_ut - 1.5f;
        int n_steps = (int)((max_h_e - 12.0f) / 3.0f) + 1;
        int step = (int)(curand_uniform(state) * n_steps);
        if (step >= n_steps) step = n_steps - 1;
        h_e = 12.0f + step * 3.0f;
        *h_e_state = h_e;
    }
    
    float d_bp_prime = 4.0f * (h_bs - h_e) * (h_ut - h_e) * fc * 10.0f/ 3.0f;  // fc is in GHz, fc*1e9/3e8 
    float pl1 = 28.0f + 22.0f * log10f(d_3d) + 20.0f * log10f(fc);
    float pl2 = 28.0f + 40.0f * log10f(d_3d) + 20.0f * log10f(fc) - 9.0f * log10f(d_bp_prime * d_bp_prime + powf(h_bs - h_ut, 2));
    
    return (d_2d_valid <= d_bp_prime) ? pl1 : pl2;
}

// GPU version of UMi LOS path loss calculation (matches CPU implementation)
__device__ float calculateUMiLosPathlossGPU(float d_2d, float d_3d, float h_bs, float h_ut, float fc) {
    float d_bp_prime = 4.0f * h_bs * h_ut * fc * 10.0f/ 3.0f;  // fc is in GHz, fc*1e9/3e8 = d_bp_prime
    float pl1 = 32.4f + 21.0f * log10f(d_3d) + 20.0f * log10f(fc);
    float pl2 = 32.4f + 40.0f * log10f(d_3d) + 20.0f * log10f(fc) - 9.5f * log10f(d_bp_prime * d_bp_prime + powf(h_bs - h_ut, 2));
    
    return (d_2d <= d_bp_prime) ? pl1 : pl2;
}

// GPU version of RMa LOS path loss calculation (matches CPU implementation)
__device__ float calculateRMaLosPathlossGPU(float d_2d, float d_3d, float h_bs, float h_ut, float fc) {
    float d_bp = 2.0f * M_PI * h_bs * h_ut * fc * 10.0f/ 3.0f;  // Breakpoint distance
    const float h = 5.0f;  // Average building height
    float pl1 = 20.0f * log10f(40.0f * M_PI * d_3d * fc / 3.0f) + 
                fminf(0.03f * powf(h, 1.72f), 10.0f) * log10f(d_3d) - 
                fminf(0.044f * powf(h, 1.72f), 14.77f) + 
                0.002f * log10f(h) * d_3d;
    float pl = d_2d <= d_bp ? pl1 : pl1 + 40.0f * log10f(d_3d / d_bp);
    
    return pl;
}

// ============================================================================
// Aerial UE Path Loss Functions (3GPP TR 36.777 Table B-2) - GPU versions
// Height-dependent formulas with applicability ranges
// ============================================================================

// Common term: 20*log10(40*π*fc/3) where fc is in GHz
// = 20*log10(40*π/3) + 20*log10(fc) ≈ 32.44 + 20*log10(fc)
__device__ float calcFreqTermGPU(float fc) {
    const float CONST_TERM = 32.44f;  // 20*log10(40*π/3)
    return CONST_TERM + 20.0f * log10f(fc);
}

// RMa-AV LOS path loss per 3GPP TR 36.777 Table B-2
// h_UT ∈ (10m, 300m], d_2D ≤ 10km:
// PL = max(23.9 - 1.8*log10(h_UT), 20) * log10(d_3D) + 20*log10(40πfc/3)
__device__ float calculateRMaAvLosPathlossGPU(float d_3d, float h_ut, float fc) {
    h_ut = fminf(fmaxf(h_ut, 10.001f), 300.0f);  // Clamp to valid range (10m, 300m]
    float n = fmaxf(23.9f - 1.8f * log10f(h_ut), 20.0f);
    return n * log10f(d_3d) + calcFreqTermGPU(fc);
}

// RMa-AV NLOS path loss per 3GPP TR 36.777 Table B-2
// h_UT ∈ (10m, 300m], d_2D ≤ 10km:
// PL = max(PL_RMa-AV-LOS, -12 + (35 - 5.3*log10(h_UT))*log10(d_3D) + 20*log10(40πfc/3))
__device__ float calculateRMaAvNlosPathlossGPU(float d_3d, float h_ut, float fc) {
    h_ut = fminf(fmaxf(h_ut, 10.001f), 300.0f);  // Clamp to valid range (10m, 300m]
    float pl_los = calculateRMaAvLosPathlossGPU(d_3d, h_ut, fc);
    float n = 35.0f - 5.3f * log10f(h_ut);
    float pl_nlos = -12.0f + n * log10f(d_3d) + calcFreqTermGPU(fc);
    return fmaxf(pl_los, pl_nlos);
}

// UMa-AV LOS path loss per 3GPP TR 36.777 Table B-2
// h_UT ∈ (22.5m, 300m], d_2D ≤ 4km:
// PL = 28.0 + 22*log10(d_3D) + 20*log10(fc)
__device__ float calculateUMaAvLosPathlossGPU(float d_3d, float fc) {
    return 28.0f + 22.0f * log10f(d_3d) + 20.0f * log10f(fc);
}

// UMa-AV NLOS path loss per 3GPP TR 36.777 Table B-2
// h_UT ∈ (10m, 100m], d_2D ≤ 4km:
// PL = -17.5 + (46 - 7*log10(h_UT))*log10(d_3D) + 20*log10(40πfc/3)
__device__ float calculateUMaAvNlosPathlossGPU(float d_3d, float h_ut, float fc) {
    h_ut = fminf(fmaxf(h_ut, 10.001f), 300.0f);  // Clamp to valid range to avoid -inf/NaN from log10f(h_ut)
    float n = 46.0f - 7.0f * log10f(h_ut);
    return -17.5f + n * log10f(d_3d) + calcFreqTermGPU(fc);
}

// UMi-AV LOS path loss per 3GPP TR 36.777 Table B-2
// h_UT ∈ (22.5m, 300m], d_2D ≤ 4km:
// PL = max{PL', 30.9 + (22.25 - 0.5*log10(h_UT))*log10(d_3D) + 20*log10(fc)}
// where PL' is free space path loss
__device__ float calculateUMiAvLosPathlossGPU(float d_3d, float h_ut, float fc) {
    h_ut = fminf(fmaxf(h_ut, 10.001f), 300.0f);  // Clamp to valid range to avoid -inf/NaN from log10f(h_ut)
    // Free space path loss: PL' = 32.4 + 20*log10(d_3D) + 20*log10(fc)
    float pl_fspl = 32.4f + 20.0f * log10f(d_3d) + 20.0f * log10f(fc);
    float n = 22.25f - 0.5f * log10f(h_ut);
    float pl_av = 30.9f + n * log10f(d_3d) + 20.0f * log10f(fc);
    return fmaxf(pl_fspl, pl_av);
}

// UMi-AV NLOS path loss per 3GPP TR 36.777 Table B-2
// h_UT ∈ (22.5m, 300m], d_2D ≤ 4km:
// PL = max{PL_UMi-AV-LOS, 32.4 + (43.2 - 7.6*log10(h_UT))*log10(d_3D) + 20*log10(fc)}
__device__ float calculateUMiAvNlosPathlossGPU(float d_3d, float h_ut, float fc) {
    h_ut = fminf(fmaxf(h_ut, 10.001f), 300.0f);  // Clamp to valid range to avoid -inf/NaN from log10f(h_ut)
    float pl_los = calculateUMiAvLosPathlossGPU(d_3d, h_ut, fc);
    float n = 43.2f - 7.6f * log10f(h_ut);
    float pl_nlos = 32.4f + n * log10f(d_3d) + 20.0f * log10f(fc);
    return fmaxf(pl_los, pl_nlos);
}

__device__ float calPLGPU(const CellParam& cellParam, const UtParam& utParam, Scenario scenario,
                         float fc, bool isLos, bool optionalPlInd, curandState* state,
                         float* h_e_state, bool is_aerial) {
    float d_3d = sqrtf((cellParam.loc.x - utParam.loc.x)*(cellParam.loc.x - utParam.loc.x) +
                       (cellParam.loc.y - utParam.loc.y)*(cellParam.loc.y - utParam.loc.y) +
                       (cellParam.loc.z - utParam.loc.z)*(cellParam.loc.z - utParam.loc.z));
    
    float d_2d = sqrtf((cellParam.loc.x - utParam.loc.x)*(cellParam.loc.x - utParam.loc.x) +
                       (cellParam.loc.y - utParam.loc.y)*(cellParam.loc.y - utParam.loc.y));
    
    float h_bs = cellParam.loc.z;
    float h_ut = utParam.loc.z;
    if (is_aerial) {
        h_ut = fminf(fmaxf((isnan(h_ut) || h_ut <= 0.0f) ? 10.001f : h_ut, 10.001f), 300.0f);
    }
    
    float pl = 0.0f;
    
    // ========================================================================
    // 3GPP TR 36.777 Table B-2: Height-dependent path loss for aerial UEs
    // Falls back to TR 38.901 for low heights (terrestrial-like behavior)
    // ========================================================================
    
    if (isLos) {
        switch (scenario) {
            case Scenario::RMa:
                // RMa-AV LOS: h_UT ≤ 10m → terrestrial; h_UT > 10m → aerial
                if (is_aerial && h_ut > 10.0f) {
                    pl = calculateRMaAvLosPathlossGPU(d_3d, h_ut, fc);
                } else {
                    pl = calculateRMaLosPathlossGPU(d_2d, d_3d, h_bs, h_ut, fc);
                }
                break;
            case Scenario::UMa:
                // UMa-AV LOS: h_UT ≤ 22.5m → terrestrial; h_UT > 22.5m → aerial
                if (is_aerial && h_ut > 22.5f) {
                    pl = calculateUMaAvLosPathlossGPU(d_3d, fc);
                } else {
                    pl = calculateUMaLosPathlossGPU(d_2d, d_3d, h_bs, h_ut, fc, state, h_e_state);
                }
                break;
            case Scenario::UMi:
                // UMi-AV LOS: h_UT ≤ 22.5m → terrestrial; h_UT > 22.5m → aerial
                if (is_aerial && h_ut > 22.5f) {
                    pl = calculateUMiAvLosPathlossGPU(d_3d, h_ut, fc);
                } else {
                    pl = calculateUMiLosPathlossGPU(d_2d, d_3d, h_bs, h_ut, fc);
                }
                break;
            default:
                break;
        }
    } else {
        // Table 7.4.1-1 defines the Optional NLOS model only for UMa/UMi/InH; RMa always
        // uses the standard max(PL_LOS, PL'_NLOS) even when optional_pl_ind is set
        if (optionalPlInd && scenario != Scenario::RMa) {
            // Optional NLOS formulas (simplified, not height-dependent)
            switch (scenario) {
                case Scenario::UMa:
                    pl = 32.4f + 20.0f * log10f(fc) + 30.0f * log10f(d_3d);
                    break;
                case Scenario::UMi:
                    pl = 32.4f + 20.0f * log10f(fc) + 31.9f * log10f(d_3d);
                    break;
                default:
                    break;
            }
        } else {
            // NLOS path loss per TR 36.777 Table B-2 (aerial) or TR 38.901 (terrestrial)
            switch (scenario) {
                case Scenario::RMa: {
                    // RMa-AV NLOS: h_UT ≤ 10m → terrestrial; h_UT > 10m → aerial
                    if (is_aerial && h_ut > 10.0f) {
                        pl = calculateRMaAvNlosPathlossGPU(d_3d, h_ut, fc);
                    } else {
                        float los_pl = calculateRMaLosPathlossGPU(d_2d, d_3d, h_bs, h_ut, fc);
                        const float W = 20.0f;
                        const float h = 5.0f;
                        float nlos_pl = 161.04f - 7.1f * log10f(W) + 7.5f * log10f(h) - 
                                      (24.37f - 3.7f * powf(h/h_bs, 2)) * log10f(h_bs) + 
                                      (43.42f - 3.1f * log10f(h_bs)) * (log10f(d_3d) - 3.0f) + 
                                      20.0f * log10f(fc) - (3.2f * powf(log10f(11.75f * h_ut), 2) - 4.97f);
                        pl = fmaxf(los_pl, nlos_pl);
                    }
                    break;
                }
                case Scenario::UMa: {
                    // UMa-AV NLOS: h_UT <= 22.5m -> terrestrial; h_UT > 22.5m -> aerial (clamped to 100m)
                    if (is_aerial && h_ut > 22.5f) {
                        pl = calculateUMaAvNlosPathlossGPU(d_3d, fminf(h_ut, 100.0f), fc);
                    } else {
                        float los_pl = calculateUMaLosPathlossGPU(d_2d, d_3d, h_bs, h_ut, fc, state, h_e_state);
                        pl = fmaxf(los_pl, 13.54f + 39.08f * log10f(d_3d) + 20.0f * log10f(fc) - 0.6f * (h_ut - 1.5f));
                    }
                    break;
                }
                case Scenario::UMi: {
                    // UMi-AV NLOS: h_UT ≤ 22.5m → terrestrial; h_UT > 22.5m → aerial
                    if (is_aerial && h_ut > 22.5f) {
                        pl = calculateUMiAvNlosPathlossGPU(d_3d, h_ut, fc);
                    } else {
                        float los_pl = calculateUMiLosPathlossGPU(d_2d, d_3d, h_bs, h_ut, fc);
                        pl = fmaxf(los_pl, 35.3f * log10f(d_3d) + 22.4f + 21.3f * log10f(fc) - 0.3f * (h_ut - 1.5f));
                    }
                    break;
                }
                default:
                    break;
            }
        }
    }
    return pl;
}

__device__ float calPenetrLosGPU(Scenario scenario, uint8_t outdoor_ind, float fc,
                                float d_2d_in, float d_2d_in_o2i,
                                uint8_t o2i_building_penetr_loss_ind,
                                uint8_t o2i_car_penetr_loss_ind, curandState* state) {
    float pl_pen = 0.0f;
    float L_glass, L_concreate, L_IRRglass;
    float pl_tw;
    
    if (!outdoor_ind) {
            switch (scenario) {
                case Scenario::UMa:
                case Scenario::UMi: {
                // Building penetration loss according to 7.4.3.1
                if (o2i_building_penetr_loss_ind == LEGACY_O2I_BUILDING_PENETR_LOSS_IND &&
                    fc * 1e9F < LEGACY_O2I_MAX_FREQUENCY_HZ) {
                    // Backward-compatible TR 36.873 model (Table 7.4.3-3).
                    {
                        float pl_tw = 20.0f;
                        float pl_in = 0.5f * d_2d_in_o2i;
                        pl_pen = pl_tw + pl_in;  // sigma_p = 0, no need to generate additional random variable
                    }
                } else {
                    // Use Table 7.4.3-2 for indicators 0-3 at every supported frequency.
                    // The host path rejects indicator 4 at >= 6 GHz before this kernel can launch.
                    switch (o2i_building_penetr_loss_ind) {
                        case 0:  // No penetration loss
                            pl_pen = 0.0f;
                            break;
                        case 1:  // Low-loss building
                            L_glass = 2.0f + 0.2f * fc;
                            L_concreate = 5.0f + 4.0f * fc;
                            pl_tw = 5.0f - 10.0f * log10f(0.3f * powf(10.0f, -0.1f * L_glass) + 0.7f * powf(10.0f, -0.1f * L_concreate));
                            pl_pen = pl_tw + 0.5f * d_2d_in + curand_normal(state) * 4.4f;
                            break;
                        case 2:  // 50% low-loss, 50% high-loss building
                            if (curand_uniform(state) < 0.5f) {
                                // Low-loss building
                                L_glass = 2.0f + 0.2f * fc;
                                L_concreate = 5.0f + 4.0f * fc;
                                pl_tw = 5.0f - 10.0f * log10f(0.3f * powf(10.0f, -0.1f * L_glass) + 0.7f * powf(10.0f, -0.1f * L_concreate));
                                pl_pen = pl_tw + 0.5f * d_2d_in + curand_normal(state) * 4.4f;
                            } else {
                                // High-loss building
                                #ifdef SLS_38901_REL18_PARAM
                                L_IRRglass = 23.0f + 0.3f * fc;   // Table 7.4.3-1 (Rel-18)
#else
                                L_IRRglass = 25.4f + 0.11f * fc;  // Table 7.4.3-1 (Rel-19, NOTE 2)
#endif
                                L_concreate = 5.0f + 4.0f * fc;    
                                pl_tw = 5.0f - 10.0f * log10f(0.7f * powf(10.0f, -0.1f * L_IRRglass) + 0.3f * powf(10.0f, -0.1f * L_concreate));
                                pl_pen = pl_tw + 0.5f * d_2d_in + curand_normal(state) * 6.5f;
                            }
                            break;
                        case 3:  // 100% high-loss building
                            #ifdef SLS_38901_REL18_PARAM
                                L_IRRglass = 23.0f + 0.3f * fc;   // Table 7.4.3-1 (Rel-18)
#else
                                L_IRRglass = 25.4f + 0.11f * fc;  // Table 7.4.3-1 (Rel-19, NOTE 2)
#endif
                            L_concreate = 5.0f + 4.0f * fc;    
                            pl_tw = 5.0f - 10.0f * log10f(0.7f * powf(10.0f, -0.1f * L_IRRglass) + 0.3f * powf(10.0f, -0.1f * L_concreate));
                            pl_pen = pl_tw + 0.5f * d_2d_in + curand_normal(state) * 6.5f;
                            break;
                        default:
                            // Unknown penetration loss index for UMa/UMi
                            break;
                    }
                }
                break;
            }
            case Scenario::RMa: {
                // Car penetration loss according to 7.4.3.2
                switch (o2i_car_penetr_loss_ind) {
                    case 0:  // No penetration loss
                        pl_pen = 0.0f;
                        break;
                    case 1:  // Low-loss building
                        L_glass = 2.0f + 0.2f * fc;
                        L_concreate = 5.0f + 4.0f * fc;
                        pl_tw = 5.0f - 10.0f * log10f(0.3f * powf(10.0f, -0.1f * L_glass) + 0.7f * powf(10.0f, -0.1f * L_concreate));
                        pl_pen = pl_tw + 0.5f * d_2d_in + curand_normal(state) * 4.4f;
                        break;
                    default:
                        // Unknown penetration loss index for RMa
                        break;
                }
                break;
            }
            default:
                // Unknown scenario
                break;
        }
    }
    else if (scenario == Scenario::RMa) {
        switch (o2i_car_penetr_loss_ind) {
            case 0:
                pl_pen = 0.0f;
                break;
            case 1:  // basic car penetration loss
                pl_pen = curand_normal(state) * 5.0f + 9.0f;
                // Note: frequency check (fc > 0.6e9f && fc <= 60e9f) should be done but omitted for GPU
                break;
            case 2:  // 50% basic, 50% metallized car penetration loss
                if (curand_uniform(state) < 0.5f) {
                    // Basic car penetration loss
                    pl_pen = curand_normal(state) * 5.0f + 9.0f;
                } else {
                    // Metallized car window penetration loss
                    pl_pen = curand_normal(state) * 20.0f + 9.0f;
                }
                // Note: frequency check (fc > 0.6e9f && fc <= 60e9f) should be done but omitted for GPU
                break;
            case 3:  // 100% metallized car window penetration loss
                pl_pen = curand_normal(state) * 20.0f + 9.0f;
                // Note: frequency check (fc > 0.6e9f && fc <= 60e9f) should be done but omitted for GPU
                break;
            default:
                // Unknown penetration loss index for RMa
                break;
        }
    }
    
    return pl_pen;
}

__device__ float calSfStdGPU(Scenario scenario, bool isLos, bool isIndoor, float fc, float d_3d, float d_2d,
                             float h_bs, float h_ut, bool optionalPlInd, bool is_aerial) {
    // O2I links use the sigma_SF of the O2I column (Table 7.5-6 / Table 7.4.3-2,-3): 7 dB for
    // UMa/UMi regardless of the outdoor-segment LOS state (RMa keeps 8 dB). O2I takes
    // precedence over the aerial branch below; the two cannot co-occur (aerial UTs are
    // outdoor by construction).
    assert(!(isIndoor && is_aerial));
    if (isIndoor && (scenario == Scenario::UMa || scenario == Scenario::UMi)) {
        return 7.0f;
    }
    if (isIndoor && scenario == Scenario::RMa) {
        return 8.0f;
    }
    if (is_aerial) {
        h_ut = fminf(fmaxf((isnan(h_ut) || h_ut <= 0.0f) ? 10.001f : h_ut, 10.001f), 300.0f);
    }
    float sf_std = 0.0f;
    if (isLos) {
        switch (scenario) {
            case Scenario::UMa:
                // 3GPP TR 36.777 Table B-3: UMa-AV LOS for h_UT > 22.5m,
                // else Table 7.4.1-1 of TR 38.901 (matches CPU calSfStd)
                if (is_aerial && h_ut > 22.5f) {
                    sf_std = 4.64f * expf(-0.0066f * h_ut);
                } else {
                    sf_std = 4.0f;  // Table 7.4.1-1 UMa LOS
                }
                break;
            case Scenario::UMi:
                // 3GPP TR 36.777 Table B-3: UMi-AV LOS for h_UT > 22.5m
                if (is_aerial && h_ut > 22.5f) {
                    sf_std = fmaxf(5.0f * expf(-0.01f * h_ut), 2.0f);
                } else {
                    sf_std = 4.0f;  // Table 7.4.1-1 UMi LOS
                }
                break;
            case Scenario::RMa: {
                // 3GPP TR 36.777 Table B-3: RMa-AV LOS for h_UT > 10m
                if (is_aerial && h_ut > 10.0f) {
                    sf_std = 4.2f * expf(-0.0046f * h_ut);
                } else {
                    float d_bp = 2 * M_PI * h_bs * h_ut * fc / 3.0e8f;  // fc in Hz, matches CPU calSfStd
                    sf_std = d_2d <= d_bp ? 4.0f : 6.0f;
                }
                break;
            }
            default:
                assert(false && "Unknown scenario");
                break;
        }
    } else {
        switch (scenario) {
            case Scenario::UMa:
                // 3GPP TR 36.777 Table B-3: UMa-AV NLOS for h_UT > 22.5m
                if (is_aerial && h_ut > 22.5f) {
                    sf_std = 6.0f;
                } else {
                    sf_std = optionalPlInd ? 7.8f : 6.0f;  // Table 7.4.1-1 UMa NLOS
                }
                break;
            case Scenario::UMi:
                // 3GPP TR 36.777 Table B-3: UMi-AV NLOS for h_UT > 22.5m
                if (is_aerial && h_ut > 22.5f) {
                    sf_std = 8.0f;
                } else {
                    sf_std = optionalPlInd ? 8.2f : 7.82f;  // Table 7.4.1-1 UMi NLOS
                }
                break;
            case Scenario::RMa:
                // 3GPP TR 36.777 Table B-3: RMa-AV NLOS for h_UT > 10m
                if (is_aerial && h_ut > 10.0f) {
                    sf_std = 6.0f;
                } else {
                    sf_std = 8.0f;
                }
                break;
            default:
                assert(false && "Unknown scenario");
                break;
        }
    }
    return sf_std;
}

template <typename Tscalar, typename Tcomplex>
void slsChan<Tscalar, Tcomplex>::initializeLargeScaleGpuKernels()
{
    CHECK_CUDAERROR(cudaGetFuncBySymbol(
        &m_calLinkParamFunction, reinterpret_cast<const void*>(calLinkParamKernel)));
}

// Host function to launch the GPU kernel
template <typename Tscalar, typename Tcomplex>
void slsChan<Tscalar, Tcomplex>::calLinkParamGPU()
{
    if (!m_crnGridsAllocated || m_crnGridSize == 0 || m_d_crnLos == nullptr ||
        m_d_crnNlos == nullptr || m_d_crnO2i == nullptr) {
        throw std::runtime_error(
            "calLinkParamGPU requires initialized, non-empty CRN grids");
    }

    // Update data on pre-allocated GPU memory
    CHECK_CURESULT(cuMemcpyHtoDAsync(
        reinterpret_cast<CUdeviceptr>(m_d_cmnLinkParams), &m_cmnLinkParams,
        sizeof(CmnLinkParams), m_strm));
    CHECK_CURESULT(cuStreamSynchronize(m_strm));

    // Launch kernel
    const uint32_t maxThreadsPerBlock = 512;
    const uint32_t threadsPerBlock = std::min(maxThreadsPerBlock, m_topology.nUT);
    const uint32_t numBlocks = (m_topology.nUT + threadsPerBlock - 1) / threadsPerBlock;
    
    dim3 blockDim(threadsPerBlock);
    // Use nSite for link parameter calculation (co-sited sectors share link parameters)
    dim3 gridDim(m_topology.nSite, numBlocks);
    
#ifdef SLS_DEBUG_
    // Debug print kernel launch parameters
    printf("DEBUG: Launching calLinkParamKernel with:\n");
    printf("  Grid: (%d, %d, %d)\n", gridDim.x, gridDim.y, gridDim.z);
    printf("  Block: (%d, %d, %d)\n", blockDim.x, blockDim.y, blockDim.z);
    printf("  nLinks: %u, seed: %u\n", m_topology.nSite * m_topology.nUT, m_randSeed);
#endif

    // Driver launches copy each argument using the kernel parameter's size;
    // keep the host object type identical to calLinkParamKernel's uint8_t.
    if (m_topology.n_sector_per_site > std::numeric_limits<uint8_t>::max()) {
        throw std::overflow_error("n_sector_per_site does not fit the kernel uint8_t parameter");
    }
    uint8_t nSectorPerSite = static_cast<uint8_t>(m_topology.n_sector_per_site);

    // A topology change invalidates an override indexed by the old link list.
    if (!m_losOverride.empty() && m_losOverride.size() != m_linkParams.size()) {
        std::fprintf(
            stderr,
            "LOS override has %zu entries but the active link list has %zu; disabling it\n",
            m_losOverride.size(), m_linkParams.size());
        clearLosOverride();
    }

    // Upload only when the host override changes. Allocation and copies happen
    // before the kernel launch and use the caller-owned stream.
    if (m_losOverrideDirty) {
        if (!m_losOverride.empty()) {
            if (m_d_losOverride == nullptr ||
                m_losOverrideCapacity != m_linkParams.size()) {
                if (m_d_losOverride != nullptr) {
                    CHECK_CURESULT(cuMemFree(
                        reinterpret_cast<CUdeviceptr>(m_d_losOverride)));
                    m_d_losOverride = nullptr;
                    m_losOverrideCapacity = 0;
                }
                CUdeviceptr losOverrideAllocation{};
                CHECK_CURESULT(cuMemAlloc(
                    &losOverrideAllocation,
                    m_linkParams.size() * sizeof(uint8_t)));
                m_d_losOverride = reinterpret_cast<uint8_t*>(losOverrideAllocation);
                m_losOverrideCapacity = m_linkParams.size();
            }
            CHECK_CURESULT(cuMemcpyHtoDAsync(
                reinterpret_cast<CUdeviceptr>(m_d_losOverride),
                m_losOverride.data(),
                m_linkParams.size() * sizeof(uint8_t),
                m_strm));
        }
        m_losOverrideDirty = false;
    }
    const uint8_t* deviceLosOverride =
        m_losOverride.empty() ? nullptr : m_d_losOverride;

    void* kernelArgs[] = {
        &m_d_cellParams, &m_d_utParams, &m_d_sysConfig, &m_d_simConfig,
        &m_d_cmnLinkParams, &m_d_crnLos, &m_d_crnNlos, &m_d_crnO2i,
        &m_maxX, &m_minX, &m_maxY, &m_minY,
        &m_topology.nSite, &m_topology.nUT, &nSectorPerSite,
        &m_d_linkParams, &m_updatePLAndPenetrationLoss, &m_updateAllLSPs,
        &m_updateLosState, &m_d_curandStates, &deviceLosOverride};
    CHECK_CURESULT(cuLaunchKernel(
        m_calLinkParamFunction,
        gridDim.x, gridDim.y, gridDim.z,
        blockDim.x, blockDim.y, blockDim.z,
        0, m_strm, kernelArgs, nullptr));
    CHECK_CURESULT(cuStreamSynchronize(m_strm));
    
    // After first call with LOS state generation, set flag to false
    // This ensures LOS state remains constant during the simulation run
    m_updateLosState = false;

    // Copy results back to host
    CHECK_CURESULT(cuMemcpyDtoHAsync(
        m_linkParams.data(), reinterpret_cast<CUdeviceptr>(m_d_linkParams),
        m_linkParams.size() * sizeof(LinkParams), m_strm));
    CHECK_CURESULT(cuStreamSynchronize(m_strm));
}

// Host function to generate CRN on GPU (following CPU reference)
// Generate Common Random Numbers for correlated LSP generation
template <typename Tscalar, typename Tcomplex>
void slsChan<Tscalar, Tcomplex>::generateCRNGPU() {
#ifdef SLS_DEBUG_
    printf("DEBUG: Starting proper CRN generation with correlation distances from sls_table.h\n");
#endif
    
    // Select appropriate correlation distances based on scenario from sls_table.h
    const CorrDist* corr_dist_los;
    const CorrDist* corr_dist_nlos;
    const CorrDist* corr_dist_o2i;
    const bool includeDeltaTau = m_sysConfig->enable_propagation_delay != 0;
    
    switch (m_sysConfig->scenario) {
        case scenario_t::UMa:
            corr_dist_los = &CORR_DIST_UMA_LOS;
            corr_dist_nlos = &CORR_DIST_UMA_NLOS;
            corr_dist_o2i = &CORR_DIST_UMA_O2I;
#ifdef SLS_DEBUG_
            printf("DEBUG: Using UMa correlation distances\n");
#endif
            break;
        case scenario_t::UMi:
            corr_dist_los = &CORR_DIST_UMI_LOS;
            corr_dist_nlos = &CORR_DIST_UMI_NLOS;
            corr_dist_o2i = &CORR_DIST_UMI_O2I;
#ifdef SLS_DEBUG_
            printf("DEBUG: Using UMi correlation distances\n");
#endif
            break;
        case scenario_t::RMa:
            corr_dist_los = &CORR_DIST_RMA_LOS;
            corr_dist_nlos = &CORR_DIST_RMA_NLOS;
            corr_dist_o2i = &CORR_DIST_RMA_O2I;
#ifdef SLS_DEBUG_
            printf("DEBUG: Using RMa correlation distances\n");
#endif
            break;
        default:
            printf("ERROR: Unknown scenario\n");
            return;
    }
    
    // Calculate CRN grid dimensions (must match what the kernel expects)
    float maxCorrDist = 120.0f;  // Maximum correlation distance from sls_table.h
    float D = 3.0f * maxCorrDist;
    int h_size = 2 * (int)D + 1;
    
    // Calculate final grid dimensions after padding and convolution (same as in kernel)
    int paddedNX = (int)roundf(m_maxX - m_minX + 1.0f + 2.0f * D);
    int paddedNY = (int)roundf(m_maxY - m_minY + 1.0f + 2.0f * D);
    int nX = paddedNX - h_size + 1;  // Final grid size after convolution
    int nY = paddedNY - h_size + 1;  // Final grid size after convolution
    m_crnGridSize = nX * nY;
    
#ifdef SLS_DEBUG_
    printf("DEBUG: Grid dimensions: %dx%d = %d elements per grid\n", nX, nY, m_crnGridSize);
#endif
    
    // Allocate correlation distance arrays
    if (m_d_corrDistLos == nullptr) {
        CUdeviceptr allocation{};
        CHECK_CURESULT(cuMemAlloc(&allocation, m_nCrnLosLsp * sizeof(float)));
        m_d_corrDistLos = reinterpret_cast<float*>(allocation);
    }
    if (m_d_corrDistNlos == nullptr) {
        CUdeviceptr allocation{};
        CHECK_CURESULT(cuMemAlloc(&allocation, m_nCrnNlosLsp * sizeof(float)));
        m_d_corrDistNlos = reinterpret_cast<float*>(allocation);
    }
    if (m_d_corrDistO2i == nullptr) {
        CUdeviceptr allocation{};
        CHECK_CURESULT(cuMemAlloc(&allocation, m_nCrnO2iLsp * sizeof(float)));
        m_d_corrDistO2i = reinterpret_cast<float*>(allocation);
    }
    
    // Set correlation distances for LOS case [SF, K, DS, ASD, ASA, ZSD, ZSA]
    std::vector<float> losCorr = {
        corr_dist_los->sf,
        corr_dist_los->k,
        corr_dist_los->ds,
        corr_dist_los->asd,
        corr_dist_los->asa,
        corr_dist_los->zsd,
        corr_dist_los->zsa
    };
    if (includeDeltaTau) {
        losCorr.push_back(corr_dist_los->dt);
    }
    
    // Set correlation distances for NLOS case [SF, DS, ASD, ASA, ZSD, ZSA] (no K)
    std::vector<float> nlosCorr = {
        corr_dist_nlos->sf,
        corr_dist_nlos->ds,
        corr_dist_nlos->asd,
        corr_dist_nlos->asa,
        corr_dist_nlos->zsd,
        corr_dist_nlos->zsa
    };
    if (includeDeltaTau) {
        nlosCorr.push_back(corr_dist_nlos->dt);
    }
    
    // Set correlation distances for O2I case [SF, DS, ASD, ASA, ZSD, ZSA] (no K)
    std::vector<float> o2iCorr = {
        corr_dist_o2i->sf,
        corr_dist_o2i->ds,
        corr_dist_o2i->asd,
        corr_dist_o2i->asa,
        corr_dist_o2i->zsd,
        corr_dist_o2i->zsa
    };
    if (includeDeltaTau) {
        o2iCorr.push_back(corr_dist_o2i->dt);
    }
    
    // Copy correlation distances to GPU memory
    CHECK_CURESULT(cuMemcpyHtoDAsync(
        reinterpret_cast<CUdeviceptr>(m_d_corrDistLos), losCorr.data(),
        losCorr.size() * sizeof(float), m_strm));
    CHECK_CURESULT(cuMemcpyHtoDAsync(
        reinterpret_cast<CUdeviceptr>(m_d_corrDistNlos), nlosCorr.data(),
        nlosCorr.size() * sizeof(float), m_strm));
    CHECK_CURESULT(cuMemcpyHtoDAsync(
        reinterpret_cast<CUdeviceptr>(m_d_corrDistO2i), o2iCorr.data(),
        o2iCorr.size() * sizeof(float), m_strm));
    
#ifdef SLS_DEBUG_
    printf("DEBUG: LOS correlation distances: SF=%.1f, K=%.1f, DS=%.1f, ASD=%.1f, ASA=%.1f, ZSD=%.1f, ZSA=%.1f\n",
           losCorr[0], losCorr[1], losCorr[2], losCorr[3], losCorr[4], losCorr[5], losCorr[6]);
    printf("DEBUG: NLOS correlation distances: SF=%.1f, DS=%.1f, ASD=%.1f, ASA=%.1f, ZSD=%.1f, ZSA=%.1f\n",
           nlosCorr[0], nlosCorr[1], nlosCorr[2], nlosCorr[3], nlosCorr[4], nlosCorr[5]);
    printf("DEBUG: O2I correlation distances: SF=%.1f, DS=%.1f, ASD=%.1f, ASA=%.1f, ZSD=%.1f, ZSA=%.1f\n",
           o2iCorr[0], o2iCorr[1], o2iCorr[2], o2iCorr[3], o2iCorr[4], o2iCorr[5]);
#endif
    
    // Allocate CRN grids - pointer arrays were already allocated in constructor
    // Use flattened indexing: [siteIdx * nLSP + lspIdx]
    const uint16_t nSite = m_topology.nSite;
    
    // Note: m_d_crnLos, m_d_crnNlos, m_d_crnO2i pointer arrays were allocated in constructor
    // Here we only need to allocate the individual grids for each site and LSP (on first call)
    
    if (!m_crnGridsAllocated) {
        // Allocate individual grids for LOS scenarios
        std::vector<float*> losGrids(nSite * m_nCrnLosLsp);
        for (uint16_t siteIdx = 0; siteIdx < nSite; siteIdx++) {
            for (int lsp = 0; lsp < m_nCrnLosLsp; lsp++) {
                int idx = siteIdx * m_nCrnLosLsp + lsp;
                CUdeviceptr allocation{};
                CHECK_CURESULT(cuMemAlloc(&allocation, m_crnGridSize * sizeof(float)));
                losGrids[idx] = reinterpret_cast<float*>(allocation);
#ifdef SLS_DEBUG_
                printf("DEBUG: Allocated LOS grid site %d LSP %d (idx %d): %p, size %d\n", 
                       siteIdx, lsp, idx, losGrids[idx], m_crnGridSize);
#endif
            }
        }
        // Copy pointers to GPU (use synchronous copy since losGrids is local and will be destroyed)
        CHECK_CURESULT(cuMemcpyHtoD(
            reinterpret_cast<CUdeviceptr>(m_d_crnLos), losGrids.data(),
            nSite * m_nCrnLosLsp * sizeof(float*)));
        
#ifdef SLS_DEBUG_
        printf("DEBUG: Successfully copied %d LOS grid pointers to device at %p\n",
               nSite * m_nCrnLosLsp, m_d_crnLos);
#endif
        
        // Allocate individual grids for NLOS scenarios
        std::vector<float*> nlosGrids(nSite * m_nCrnNlosLsp);
        for (uint16_t siteIdx = 0; siteIdx < nSite; siteIdx++) {
            for (int lsp = 0; lsp < m_nCrnNlosLsp; lsp++) {
                int idx = siteIdx * m_nCrnNlosLsp + lsp;
                CUdeviceptr allocation{};
                CHECK_CURESULT(cuMemAlloc(&allocation, m_crnGridSize * sizeof(float)));
                nlosGrids[idx] = reinterpret_cast<float*>(allocation);
#ifdef SLS_DEBUG_
                printf("DEBUG: Allocated NLOS grid site %d LSP %d (idx %d): %p, size %d\n", 
                       siteIdx, lsp, idx, nlosGrids[idx], m_crnGridSize);
#endif
            }
        }
        // Copy pointers to GPU (use synchronous copy since nlosGrids is local and will be destroyed)
        CHECK_CURESULT(cuMemcpyHtoD(
            reinterpret_cast<CUdeviceptr>(m_d_crnNlos), nlosGrids.data(),
            nSite * m_nCrnNlosLsp * sizeof(float*)));
        
        // Allocate individual grids for O2I scenarios
        std::vector<float*> o2iGrids(nSite * m_nCrnO2iLsp);
        for (uint16_t siteIdx = 0; siteIdx < nSite; siteIdx++) {
            for (int lsp = 0; lsp < m_nCrnO2iLsp; lsp++) {
                int idx = siteIdx * m_nCrnO2iLsp + lsp;
                CUdeviceptr allocation{};
                CHECK_CURESULT(cuMemAlloc(&allocation, m_crnGridSize * sizeof(float)));
                o2iGrids[idx] = reinterpret_cast<float*>(allocation);
#ifdef SLS_DEBUG_
                printf("DEBUG: Allocated O2I grid site %d LSP %d (idx %d): %p, size %d\n", 
                       siteIdx, lsp, idx, o2iGrids[idx], m_crnGridSize);
#endif
            }
        }
        // Copy pointers to GPU (use synchronous copy since o2iGrids is local and will be destroyed)
        CHECK_CURESULT(cuMemcpyHtoD(
            reinterpret_cast<CUdeviceptr>(m_d_crnO2i), o2iGrids.data(),
            nSite * m_nCrnO2iLsp * sizeof(float*)));
        
        // Mark grids as allocated
        m_crnGridsAllocated = true;
    }
    
    // Generate CRNs one by one with shared temp memory
    // Calculate temp memory size for maximum final grid (after convolution)
    float maxCorrDistTemp = 120.0f;  // Maximum correlation distance from sls_table.h  
    float D_temp = 3.0f * maxCorrDistTemp;
    int paddedNX_temp = (int)roundf(m_maxX - m_minX + 1.0f + 2.0f * D_temp);
    int paddedNY_temp = (int)roundf(m_maxY - m_minY + 1.0f + 2.0f * D_temp);
    int tempCrnSize = paddedNX_temp * paddedNY_temp;
    
    // Allocate temp GPU memory for uncorrelated noise generation
    CUdeviceptr tempCrnAllocation{};
    CHECK_CURESULT(cuMemAlloc(&tempCrnAllocation, tempCrnSize * sizeof(float)));
    float* d_tempCRN = reinterpret_cast<float*>(tempCrnAllocation);
    
    // Use fixed thread configuration matching the curandState allocation
    const int maxCrnBlocks = 128;  // Must match allocation in sls_chan.cu
    const int threadsPerBlock = 256;  // Must match allocation in sls_chan.cu
    
    int finalNX_temp = paddedNX_temp - (2 * (int)D_temp + 1) + 1;  // Final grid size after convolution
    int finalNY_temp = paddedNY_temp - (2 * (int)D_temp + 1) + 1;  // Final grid size after convolution
    
    // Calculate elements per thread dynamically (same as in sls_chan.cu)
    // Note: This calculation is performed inside the kernel as well
    uint32_t totalElements = finalNX_temp * finalNY_temp;
    const uint32_t totalThreads = maxCrnBlocks * threadsPerBlock;
    
#ifdef SLS_DEBUG_
    const uint32_t elementsPerThread = (totalElements + totalThreads - 1) / totalThreads;
    printf("DEBUG: Host CRN calculation - Elements: %u, Threads: %u, ElementsPerThread: %u\n",
           totalElements, totalThreads, elementsPerThread);
#else
    (void)totalElements;  // Suppress unused variable warning
    (void)totalThreads;   // Suppress unused variable warning
#endif
    
    // Use 1D block configuration for simplicity (128 blocks × 256 threads each)
    dim3 numBlocks(maxCrnBlocks, 1);
    dim3 threadsPerBlockDim(threadsPerBlock, 1);
    
    // Calculate shared memory size as max of filter and power array (they don't overlap in time)
    int maxL = (2 * (int)(3.0f * 120.0f) + 1);  // Max filter size = 721
    // +4 floats of slack: the compiler emits vectorized (96-bit) shared loads for the
    // filter taps, and the widened load covering the last tap over-reads up to 8 B past
    // it. The extra lanes are never consumed, but when L == maxL the tap sits flush
    // against the allocation end and the access itself is out of bounds (RMa SF grids,
    // corrDist = 120 m, flagged by compute-sanitizer). The slack keeps every widened
    // load inside the allocation.
    int maxSharedMemSize = (std::max(maxL, threadsPerBlock) + 4) * sizeof(float);
    
    // Ensure all prior allocations and copies are complete
    CHECK_CURESULT(cuStreamSynchronize(m_strm));
    
    // Copy correlation distances to host for kernel calls
    std::vector<float> losCorrelationDists(m_nCrnLosLsp);
    std::vector<float> nlosCorrelationDists(m_nCrnNlosLsp);
    std::vector<float> o2iCorrelationDists(m_nCrnO2iLsp);

    CHECK_CURESULT(cuMemcpyDtoH(
        losCorrelationDists.data(), reinterpret_cast<CUdeviceptr>(m_d_corrDistLos),
        losCorrelationDists.size() * sizeof(float)));
    CHECK_CURESULT(cuMemcpyDtoH(
        nlosCorrelationDists.data(), reinterpret_cast<CUdeviceptr>(m_d_corrDistNlos),
        nlosCorrelationDists.size() * sizeof(float)));
    CHECK_CURESULT(cuMemcpyDtoH(
        o2iCorrelationDists.data(), reinterpret_cast<CUdeviceptr>(m_d_corrDistO2i),
        o2iCorrelationDists.size() * sizeof(float)));
    
    // Copy CRN grid pointers to host for kernel calls
    std::vector<float*> losCrnGrids(nSite * m_nCrnLosLsp);
    std::vector<float*> nlosCrnGrids(nSite * m_nCrnNlosLsp);
    std::vector<float*> o2iCrnGrids(nSite * m_nCrnO2iLsp);

    CHECK_CURESULT(cuMemcpyDtoH(
        losCrnGrids.data(), reinterpret_cast<CUdeviceptr>(m_d_crnLos),
        losCrnGrids.size() * sizeof(float*)));
    CHECK_CURESULT(cuMemcpyDtoH(
        nlosCrnGrids.data(), reinterpret_cast<CUdeviceptr>(m_d_crnNlos),
        nlosCrnGrids.size() * sizeof(float*)));
    CHECK_CURESULT(cuMemcpyDtoH(
        o2iCrnGrids.data(), reinterpret_cast<CUdeviceptr>(m_d_crnO2i),
        o2iCrnGrids.size() * sizeof(float*)));

    CUfunction fillCRNNoiseFunction;
    CUfunction convolveCRNFunction;
    CHECK_CUDAERROR(cudaGetFuncBySymbol(
        &fillCRNNoiseFunction, reinterpret_cast<const void*>(fillCRNNoiseKernel)));
    CHECK_CUDAERROR(cudaGetFuncBySymbol(
        &convolveCRNFunction, reinterpret_cast<const void*>(convolveCRNKernel)));
    
    // Two-stage launch per grid: fill the padded noise buffer, then convolve. Sequential
    // launches on one stream give the device-wide ordering a block-local barrier cannot.
    auto launchCrnGeneration = [&](float* outputGrid, float corrDist) {
        float D = 3.0f * corrDist;
        uint32_t paddedNX = (uint32_t)roundf(m_maxX - m_minX + 1.0f + 2.0f * D);
        uint32_t paddedNY = (uint32_t)roundf(m_maxY - m_minY + 1.0f + 2.0f * D);
        uint32_t totalPaddedElements = paddedNX * paddedNY;
        void* fillArgs[] = {
            &d_tempCRN, &totalPaddedElements, &m_d_curandStates, &m_maxCurandStates};
        CHECK_CURESULT(cuLaunchKernel(
            fillCRNNoiseFunction,
            numBlocks.x, numBlocks.y, numBlocks.z,
            threadsPerBlockDim.x, threadsPerBlockDim.y, threadsPerBlockDim.z,
            0, m_strm, fillArgs, nullptr));

        void* convolveArgs[] = {
            &d_tempCRN, &outputGrid, &m_maxX, &m_minX, &m_maxY, &m_minY, &corrDist};
        CHECK_CURESULT(cuLaunchKernel(
            convolveCRNFunction,
            numBlocks.x, numBlocks.y, numBlocks.z,
            threadsPerBlockDim.x, threadsPerBlockDim.y, threadsPerBlockDim.z,
            maxSharedMemSize, m_strm, convolveArgs, nullptr));
    };

    // Generate LOS grids - one two-stage generation per site per LSP
    for (uint16_t siteIdx = 0; siteIdx < nSite; siteIdx++) {
        for (int lsp = 0; lsp < m_nCrnLosLsp; lsp++) {
            int idx = siteIdx * m_nCrnLosLsp + lsp;
            launchCrnGeneration(losCrnGrids[idx], losCorrelationDists[lsp]);
        }
    }
    
#ifdef SLS_DEBUG_
    printf("DEBUG: Generated LOS grids with spatial correlation (%d sites x %d LSPs = %d kernels)\n",
           nSite, m_nCrnLosLsp, nSite * m_nCrnLosLsp);
#endif
    
    // Generate NLOS grids - one two-stage generation per site per LSP
    for (uint16_t siteIdx = 0; siteIdx < nSite; siteIdx++) {
        for (int lsp = 0; lsp < m_nCrnNlosLsp; lsp++) {
            int idx = siteIdx * m_nCrnNlosLsp + lsp;
            launchCrnGeneration(nlosCrnGrids[idx], nlosCorrelationDists[lsp]);
        }
    }
    
#ifdef SLS_DEBUG_
    printf("DEBUG: Generated NLOS grids with spatial correlation (%d sites x %d LSPs = %d kernels)\n",
           nSite, m_nCrnNlosLsp, nSite * m_nCrnNlosLsp);
#endif
    
    // Generate O2I grids - one two-stage generation per site per LSP
    for (uint16_t siteIdx = 0; siteIdx < nSite; siteIdx++) {
        for (int lsp = 0; lsp < m_nCrnO2iLsp; lsp++) {
            int idx = siteIdx * m_nCrnO2iLsp + lsp;
            launchCrnGeneration(o2iCrnGrids[idx], o2iCorrelationDists[lsp]);
        }
    }
    
    // The temporary buffer is consumed by the queued convolution kernels.
    CHECK_CURESULT(cuStreamSynchronize(m_strm));

    // Free temp memory
    CHECK_CURESULT(cuMemFree(reinterpret_cast<CUdeviceptr>(d_tempCRN)));
    
#ifdef SLS_DEBUG_
    printf("DEBUG: Generated O2I grids with spatial correlation\n");
#endif
    
#ifdef SLS_DEBUG_
    printf("DEBUG: CRN generation completed successfully\n");
#endif

    // Step 2: Normalize all CRN grids using efficient single-kernel approach    
    // Shared memory size for 1024 threads (power reduction)
    const size_t sharedMemSize = 1024 * sizeof(float);
    
    // Calculate total number of grids for ALL sites
    const int totalLosGrids = nSite * m_nCrnLosLsp;
    const int totalNlosGrids = nSite * m_nCrnNlosLsp;
    const int totalO2iGrids = nSite * m_nCrnO2iLsp;
    const int totalGrids = totalLosGrids + totalNlosGrids + totalO2iGrids;
    
#ifdef SLS_DEBUG_
    printf("DEBUG: Normalizing %d total grids (%d LOS + %d NLOS + %d O2I) for %d sites\n",
           totalGrids, totalLosGrids, totalNlosGrids, totalO2iGrids, nSite);
#endif
    
    // Allocate host array for all CRN grid pointers
    std::vector<float*> allCrnGrids(totalGrids);
    
    // Copy all grid pointers into single array
    for (int i = 0; i < totalLosGrids; i++) {
        allCrnGrids[i] = losCrnGrids[i];  // LOS grids: indices 0 to (totalLosGrids-1)
    }
    for (int i = 0; i < totalNlosGrids; i++) {
        allCrnGrids[totalLosGrids + i] = nlosCrnGrids[i];  // NLOS grids
    }
    for (int i = 0; i < totalO2iGrids; i++) {
        allCrnGrids[totalLosGrids + totalNlosGrids + i] = o2iCrnGrids[i];  // O2I grids
    }
    
    // Allocate device memory for single CRN grid pointer array
    CUdeviceptr allCrnGridsAllocation{};
    CHECK_CURESULT(cuMemAlloc(&allCrnGridsAllocation, totalGrids * sizeof(float*)));
    float** d_allCrnGrids = reinterpret_cast<float**>(allCrnGridsAllocation);
    
    // Single copy operation for all grid pointers
    CHECK_CURESULT(cuMemcpyHtoD(
        reinterpret_cast<CUdeviceptr>(d_allCrnGrids), allCrnGrids.data(),
        totalGrids * sizeof(float*)));
    
    // Launch normalization kernels through the Driver API.
    CUfunction normalizeCrnFunction;
    CHECK_CUDAERROR(cudaGetFuncBySymbol(
        &normalizeCrnFunction, reinterpret_cast<const void*>(normalizeCRNGridsKernel)));
    auto launchNormalization = [&](int gridOffset, int numGrids) {
        float** gridGroup = d_allCrnGrids + gridOffset;
        void* normalizeArgs[] = {&gridGroup, &totalElements, &numGrids};
        CHECK_CURESULT(cuLaunchKernel(
            normalizeCrnFunction,
            numGrids, 1, 1,
            1024, 1, 1,
            sharedMemSize, m_strm, normalizeArgs, nullptr));
    };
    launchNormalization(0, totalLosGrids);
    launchNormalization(totalLosGrids, totalNlosGrids);
    launchNormalization(totalLosGrids + totalNlosGrids, totalO2iGrids);
    
    // The pointer array must remain alive until every normalization kernel has
    // consumed it.
    CHECK_CURESULT(cuStreamSynchronize(m_strm));

    // Clean up device memory
    CHECK_CURESULT(cuMemFree(reinterpret_cast<CUdeviceptr>(d_allCrnGrids)));
    
#ifdef SLS_DEBUG_
    printf("DEBUG: CRN normalization completed successfully\n");
#endif
}

// GPU kernel: Normalize multiple CRN grids - one block per CRN grid
__global__ void normalizeCRNGridsKernel(float** crnGrids, uint32_t totalElements, int numGrids) {
    const int blockId = blockIdx.x;  // Each block handles one CRN grid
    const int tid = threadIdx.x;     // Thread ID within block (0-1023)
    
    // Ensure we don't exceed the number of grids
    if (blockId >= numGrids) return;
    
    float* crnGrid = crnGrids[blockId];
    
    // Shared memory for power reduction (1024 floats)
    extern __shared__ float sharedPower[];
    
    // Phase 1: Calculate total power using all 1024 threads
    float localPower = 0.0f;
    
    // Each thread processes multiple elements using stride
    for (uint32_t idx = tid; idx < totalElements; idx += 1024) {
        localPower += crnGrid[idx] * crnGrid[idx];
    }
    
    // Store local power in shared memory
    sharedPower[tid] = localPower;
    __syncthreads();
    
    // Block-level reduction using shared memory
    for (int s = 512; s > 0; s >>= 1) {
        if (tid < s) {
            sharedPower[tid] += sharedPower[tid + s];
        }
        __syncthreads();
    }
    
    // Thread 0 calculates normalization factor
    __shared__ float normFactor;
    if (tid == 0) {
        const float totalPower = sharedPower[0];
        if (totalPower > 0.0f) {
            normFactor = 1.0f / sqrtf(totalPower / totalElements);
        } else {
            normFactor = 1.0f;  // Avoid division by zero
        }
    }
    __syncthreads();
    
    // Phase 2: All threads apply normalization
    for (uint32_t idx = tid; idx < totalElements; idx += 1024) {
        crnGrid[idx] *= normFactor;
    }
}

// GPU kernel: Generate single CRN with temp memory and 2D thread blocks
// Stage 1 of CRN generation: fill the padded grid with i.i.d. N(0,1) noise.
// This must be a SEPARATE kernel from the convolution: each output element's filter support
// spans noise written by other blocks, and a block-local __syncthreads() cannot order writes
// across blocks. Two sequential launches on one stream give the required device-wide barrier.
__global__ void fillCRNNoiseKernel(float* tempCRN, uint32_t totalPaddedElements,
                                   curandState* curandStates, uint32_t maxCurandStates) {
    uint32_t globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t totalThreads = gridDim.x * blockDim.x;
    uint32_t stateId = globalThreadId % maxCurandStates;
    curandState localState = curandStates[stateId];
    // Grid-stride loop: covers every padded element with no unwritten tail
    for (uint32_t idx = globalThreadId; idx < totalPaddedElements; idx += totalThreads) {
        tempCRN[idx] = curand_normal(&localState);
    }
    curandStates[stateId] = localState;
}

// Stage 2 of CRN generation: 2D separable exponential filter (7.4.4 / eq 7.4-5) applied to
// the pre-filled noise grid. Output layout matches getLspAtLocationGPU: element (ix, iy)
// lives at outputCRN[iy * finalNX + ix].
__global__ void convolveCRNKernel(
    const float* tempCRN, float* outputCRN,
    float maxX, float minX, float maxY, float minY,
    float correlationDist
) {
    int tid = threadIdx.x;  // Local thread ID within block

    // Grid dimensions (must match the host-side padded allocation for this correlationDist)
    float D = 3.0f * correlationDist;
    uint32_t paddedNX = (uint32_t)roundf(maxX - minX + 1.0f + 2.0f * D);
    uint32_t paddedNY = (uint32_t)roundf(maxY - minY + 1.0f + 2.0f * D);
    uint32_t L = (correlationDist == 0.0f) ? 1 : 2 * (uint32_t)D + 1;

    // Filter coefficients in shared memory (one thread per block fills them)
    extern __shared__ float dynamicShared[];
    float* h = dynamicShared;
    if (tid == 0) {
        if (correlationDist == 0.0f) {
            h[0] = 1.0f;
        } else {
            for (uint32_t k = 0; k < L; k++) {
                // NOTE: compute the offset in float; (k - (uint32_t)D) in unsigned arithmetic
                // wraps for k < D and silently zeroed the left half of the filter
                h[k] = expf(-fabsf((float)k - D) / correlationDist);
            }
        }
    }
    __syncthreads();  // Ensure all threads have filter ready

    // Final output dimensions after "valid" convolution
    uint32_t finalNX = paddedNX - L + 1;  // = maxX - minX + 1
    uint32_t finalNY = paddedNY - L + 1;  // = maxY - minY + 1
    uint32_t totalElements = finalNX * finalNY;
    uint32_t totalThreads = gridDim.x * blockDim.x;
    uint32_t globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;

    // Grid-stride loop over output elements
    for (uint32_t linearIdx = globalThreadId; linearIdx < totalElements; linearIdx += totalThreads) {
        uint32_t ix = linearIdx % finalNX;
        uint32_t iy = linearIdx / finalNX;

        float sum = 0.0f;
        // Separable 2D filter h(x) * h(y); tempCRN uses the same (row = y, fast index = x) layout
        for (uint32_t dj = 0; dj < L; dj++) {
            const float hj = h[dj];
            uint32_t rowBase = (iy + dj) * paddedNX + ix;
            for (uint32_t di = 0; di < L; di++) {
                sum += h[di] * hj * tempCRN[rowBase + di];
            }
        }
        outputCRN[linearIdx] = sum;
    }
    // Power normalization is handled by the separate normalizeCRNGridsKernel
}

// Explicit template instantiations
template class slsChan<float, float2>;
