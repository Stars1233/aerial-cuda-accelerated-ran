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
#include <vector>
#include <random>

// Correlation distances (defined in sls_table.h)
extern const corrDist_t corrDistUmaLos;
extern const corrDist_t corrDistUmaNlos;
extern const corrDist_t corrDistUmiLos;
extern const corrDist_t corrDistUmiNlos;
extern const corrDist_t corrDistRmaLos;
extern const corrDist_t corrDistRmaNlos;
extern const corrDist_t corrDistUmaO2i;
extern const corrDist_t corrDistUmiO2i;
extern const corrDist_t corrDistRmaO2i;

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
                         float fc, bool isLos, bool optionalPlInd, curandState* state, bool is_aerial = false);

__device__ float calPenetrLosGPU(Scenario scenario, uint8_t outdoor_ind, float fc,
                                float d_2d_in, uint8_t o2i_building_penetr_loss_ind,
                                uint8_t o2i_car_penetr_loss_ind, curandState* state);

__device__ float calSfStdGPU(Scenario scenario, bool isLos, bool isIndoor, float fc, float d_3d, float d_2d);

__device__ float getLspAtLocationGPU(float x, float y, float maxX, float minX, float maxY, float minY,
                                   const float* crnGrid, int lspIdx, int nX, int nY);

// Forward declaration for single CRN generation kernel
__global__ void generateSingleCRNKernel(
    float* tempCRN, float* outputCRN,
    float maxX, float minX, float maxY, float minY,
    float correlationDist, curandState* curandStates, uint32_t maxCurandStates
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
    curandState* curandStates
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
        linkParams[linkIdx].losInd = (curand_uniform(&localState) <= losProb) ? 1 : 0;
    }

    // Calculate path loss (always needed for mode 1 and 2)
    // For aerial UEs, use 3GPP TR 36.777 Table B-2 path loss models
    if (updatePLAndPenetrationLoss || updateAllLSPs) {
        float pl = calPLGPU(cellParams[siteIdx * nSectorPerSite], utParams[ueIdx], sysConfig->scenario,
                            simConfig->center_freq_hz / 1e9, linkParams[linkIdx].losInd, false, &localState, is_aerial);
        
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
    
    // Calculate grid dimensions (must match generateSingleCRNKernel output)
    // Use a reasonable default correlation distance for grid calculation (use maximum expected)
    float maxCorrDist = 120.0f;  // Maximum correlation distance from sls_table.h
    float D = 3.0f * maxCorrDist;
    int h_size = 2 * (int)D + 1;
    
    // Calculate final grid dimensions after padding and convolution (same as generateSingleCRNKernel)
    int paddedNX = (int)roundf(maxX - minX + 1.0f + 2.0f * D);
    int paddedNY = (int)roundf(maxY - minY + 1.0f + 2.0f * D);
    int nX = paddedNX - h_size + 1;  // Final grid size after convolution
    int nY = paddedNY - h_size + 1;  // Final grid size after convolution
    
    // Get LSP values from pre-generated CRN using getLspAtLocationGPU
    // Use site-specific CRN grids: crnLos[siteIdx * 7 + lspIdx]
    // Handle null CRN arrays (temporary fix for debugging)
    const float* losGrid0 = (crnLos != nullptr) ? crnLos[siteIdx * 7 + 0] : nullptr;
    const float* losGrid1 = (crnLos != nullptr) ? crnLos[siteIdx * 7 + 1] : nullptr;
    const float* losGrid2 = (crnLos != nullptr) ? crnLos[siteIdx * 7 + 2] : nullptr;
    const float* losGrid3 = (crnLos != nullptr) ? crnLos[siteIdx * 7 + 3] : nullptr;
    const float* losGrid4 = (crnLos != nullptr) ? crnLos[siteIdx * 7 + 4] : nullptr;
    const float* losGrid5 = (crnLos != nullptr) ? crnLos[siteIdx * 7 + 5] : nullptr;
    const float* losGrid6 = (crnLos != nullptr) ? crnLos[siteIdx * 7 + 6] : nullptr;
    
    const float* nlosGrid0 = (crnNlos != nullptr) ? crnNlos[siteIdx * 6 + 0] : nullptr;
    const float* nlosGrid1 = (crnNlos != nullptr) ? crnNlos[siteIdx * 6 + 1] : nullptr;
    const float* nlosGrid2 = (crnNlos != nullptr) ? crnNlos[siteIdx * 6 + 2] : nullptr;
    const float* nlosGrid3 = (crnNlos != nullptr) ? crnNlos[siteIdx * 6 + 3] : nullptr;
    const float* nlosGrid4 = (crnNlos != nullptr) ? crnNlos[siteIdx * 6 + 4] : nullptr;
    const float* nlosGrid5 = (crnNlos != nullptr) ? crnNlos[siteIdx * 6 + 5] : nullptr;
    
    const float* o2iGrid0 = (crnO2i != nullptr) ? crnO2i[siteIdx * 6 + 0] : nullptr;
    const float* o2iGrid1 = (crnO2i != nullptr) ? crnO2i[siteIdx * 6 + 1] : nullptr;
    const float* o2iGrid2 = (crnO2i != nullptr) ? crnO2i[siteIdx * 6 + 2] : nullptr;
    const float* o2iGrid3 = (crnO2i != nullptr) ? crnO2i[siteIdx * 6 + 3] : nullptr;
    const float* o2iGrid4 = (crnO2i != nullptr) ? crnO2i[siteIdx * 6 + 4] : nullptr;
    const float* o2iGrid5 = (crnO2i != nullptr) ? crnO2i[siteIdx * 6 + 5] : nullptr;
    
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
        linkParams[linkIdx].SF = corrVars[SF_IDX] * calSfStdGPU(sysConfig->scenario, isLos, isIndoor, simConfig->center_freq_hz, d_3d, d_2d);
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
        if (sysConfig->enable_propagation_delay == 1) {
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
                float r_DT = curand_normal(&localState);
                float lg_delta_tau = mu_lg_dt + sigma_lg_dt * r_DT;
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
    // Return dummy values if CRN grid is null (fallback for debugging)
    if (crnGrid == nullptr) {
        // Return different dummy values for each LSP index for debugging
        return (float)(lspIdx + 1) * 0.1f; // 0.1, 0.2, 0.3, ...
    }
    
    // Safety check for grid dimensions
    if (nX <= 0 || nY <= 0) {
        printf("ERROR: Invalid grid dimensions nX=%d, nY=%d for lspIdx %d\n", nX, nY, lspIdx);
        return 0.0f;
    }
    
    // Calculate the normalized position within the grid (same as CPU reference)
    float normX = (x - minX) / (maxX - minX);
    float normY = (y - minY) / (maxY - minY);
    
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
    
    // Additional bounds checking
    if (x0 < 0 || y0 < 0 || x1 >= nX || y1 >= nY) {
        printf("ERROR: Grid indices out of bounds: x0=%d, y0=%d, x1=%d, y1=%d, nX=%d, nY=%d\n", 
               x0, y0, x1, y1, nX, nY);
        return 0.0f;
    }
    
    // Check array indices before access
    int idx00 = y0 * nX + x0;
    int idx10 = y0 * nX + x1;
    int idx01 = y1 * nX + x0;
    int idx11 = y1 * nX + x1;
    int maxIdx = nX * nY - 1;
    
    if (idx00 > maxIdx || idx10 > maxIdx || idx01 > maxIdx || idx11 > maxIdx) {
        printf("ERROR: Array indices out of bounds: idx00=%d, idx10=%d, idx01=%d, idx11=%d, maxIdx=%d\n", 
               idx00, idx10, idx01, idx11, maxIdx);
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
    d_3d_in = d_3d * d_2d_in / d_2d;
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
__device__ float calculateUMaLosPathlossGPU(float d_2d, float d_3d, float h_bs, float h_ut, float fc, curandState* state) {
    float d_2d_valid = fmaxf(d_2d, 10.0f);
    float g_d2d = d_2d_valid <= 18.0f ? 0.0f : 5.0f/4.0f * powf(d_2d_valid / 100.0f, 3.0f) * expf(-d_2d_valid / 150.0f);
    float c_d2d_hut = h_ut < 13.0f ? 0.0f : powf((h_ut - 13.0f) / 10.0f, 1.5f) * g_d2d;
    float prob_h_e = 1.0f / (1.0f + c_d2d_hut);
    
    float h_e;
    float random_val = curand_uniform(state);
    if (random_val <= prob_h_e) {
        h_e = 1.0f;  // With probability 1/(1+C(d2D, hUT))
    } else {
        // Use random number for discrete uniform distribution
        float max_h_e = h_ut - 1.5f;
        int n_steps = (int)((max_h_e - 12.0f) / 3.0f) + 1;
        int step = (int)(curand_uniform(state) * n_steps);
        h_e = 12.0f + step * 3.0f;
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
                         float fc, bool isLos, bool optionalPlInd, curandState* state, bool is_aerial) {
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
                    pl = calculateUMaLosPathlossGPU(d_2d, d_3d, h_bs, h_ut, fc, state);
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
        if (optionalPlInd) {
            // Optional NLOS formulas (simplified, not height-dependent)
            switch (scenario) {
                case Scenario::UMa:
                    pl = 32.4f + 20.0f * log10f(fc) + 30.0f * log10f(d_3d);
                    break;
                case Scenario::UMi:
                    pl = 32.4f + 20.0f * log10f(fc) + 31.9f * log10f(d_3d);
                    break;
                case Scenario::RMa:
                    // RMa does not support optional pathloss model, fallback to LOS
                    pl = calculateRMaLosPathlossGPU(d_2d, d_3d, h_bs, h_ut, fc);
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
                        float los_pl = calculateUMaLosPathlossGPU(d_2d, d_3d, h_bs, h_ut, fc, state);
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
                                float d_2d_in, uint8_t o2i_building_penetr_loss_ind,
                                uint8_t o2i_car_penetr_loss_ind, curandState* state) {
    float pl_pen = 0.0f;
    float L_glass, L_concreate, L_IRRglass;
    float pl_tw;
    
    if (!outdoor_ind) {
            switch (scenario) {
                case Scenario::UMa:
                case Scenario::UMi: {
                // Building penetration loss according to 7.4.3.1
                if (fc < 6.0f) {
                    // Use Table 7.4.3-3 for frequencies below 6 GHz
                    if (o2i_building_penetr_loss_ind) {
                        float pl_tw = 20.0f;
                        float pl_in = 0.5f * 25.0f * curand_uniform(state);
                        pl_pen = pl_tw + pl_in;  // sigma_p = 0, no need to generate additional random variable
                    }
                } else {
                    // Use Table 7.4.3-2 for frequencies above 6 GHz
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
                                L_IRRglass = 23.0f + 0.3f * fc;
                                L_concreate = 5.0f + 4.0f * fc;    
                                pl_tw = 5.0f - 10.0f * log10f(0.7f * powf(10.0f, -0.1f * L_IRRglass) + 0.3f * powf(10.0f, -0.1f * L_concreate));
                                pl_pen = pl_tw + 0.5f * d_2d_in + curand_normal(state) * 6.5f;
                            }
                            break;
                        case 3:  // 100% high-loss building
                            L_IRRglass = 23.0f + 0.3f * fc;
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

__device__ float calSfStdGPU(Scenario scenario, bool isLos, bool isIndoor, float fc, float d_3d, float d_2d) {
    float sf_std = 0.0f;
    if (isLos) {
        switch (scenario) {
            case Scenario::UMa:
            case Scenario::UMi:
                sf_std = (fc < 6e9f || isIndoor) ? 7.0f : 4.0f;  // fc is in Hz
                break;
            case Scenario::RMa: {
                if (isIndoor) {
                    sf_std = 8.0f;
                    break;
                }
                // h_bs and h_ut need to be passed as parameters - using typical values for now
                float h_bs_rma = 35.0f;  // typical RMa BS height
                float h_ut_rma = 1.5f;   // typical RMa UE height
                float d_bp = 2 * M_PI * h_bs_rma * h_ut_rma * fc * 1e9f / 3.0e8f;  // Convert fc back to Hz for calculation
                sf_std = d_2d <= d_bp ? 4.0f : 6.0f;
                break;
            }
            default:
                assert(false && "Unknown scenario");
                break;
        }
    } else {
        switch (scenario) {
            case Scenario::UMa:
                sf_std = (fc < 6e9f || isIndoor) ? 7.0f : 6.0f;  // fc is in Hz
                break;
            case Scenario::UMi:
                sf_std = isIndoor ? 7.0f : 7.82f;
                break;
            case Scenario::RMa:
                sf_std = 8.0f;
                break;
            default:
                assert(false && "Unknown scenario");
                break;
        }
    }
    return sf_std;
}

// Host function to launch the GPU kernel
template <typename Tscalar, typename Tcomplex>
void slsChan<Tscalar, Tcomplex>::calLinkParamGPU()
{
    // Update data on pre-allocated GPU memory
    CHECK_CUDAERROR(cudaMemcpyAsync(m_d_cmnLinkParams, &m_cmnLinkParams, sizeof(CmnLinkParams), cudaMemcpyHostToDevice, m_strm));
    
    // get last error
    CHECK_CUDAERROR(cudaStreamSynchronize(m_strm));
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("ERROR: Common link parameter copy failed: %s\n", cudaGetErrorString(err));
    }

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
    
    calLinkParamKernel<<<gridDim, blockDim, 0, m_strm>>>(m_d_cellParams, m_d_utParams, m_d_sysConfig, m_d_simConfig,
                                               m_d_cmnLinkParams, (const float**)m_d_crnLos, (const float**)m_d_crnNlos, (const float**)m_d_crnO2i, m_maxX, m_minX,
                                               m_maxY, m_minY, m_topology.nSite, m_topology.nUT,
                                               m_topology.n_sector_per_site, m_d_linkParams, m_updatePLAndPenetrationLoss, m_updateAllLSPs, m_updateLosState, m_d_curandStates);

    // get last error
    CHECK_CUDAERROR(cudaStreamSynchronize(m_strm));
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("ERROR: Link parameter calculation kernel failed: %s\n", cudaGetErrorString(err));
    }
    
    // After first call with LOS state generation, set flag to false
    // This ensures LOS state remains constant during the simulation run
    m_updateLosState = false;

    // Copy results back to host
    CHECK_CUDAERROR(cudaMemcpyAsync(m_linkParams.data(), m_d_linkParams, m_linkParams.size() * sizeof(LinkParams), cudaMemcpyDeviceToHost, m_strm));
    CHECK_CUDAERROR(cudaStreamSynchronize(m_strm));
    
    // get last error
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("ERROR: Link parameter calculation copy back failed: %s\n", cudaGetErrorString(err));
    }
}

// Host function to generate CRN on GPU (following CPU reference)
// Generate Common Random Numbers for correlated LSP generation
template <typename Tscalar, typename Tcomplex>
void slsChan<Tscalar, Tcomplex>::generateCRNGPU() {
#ifdef SLS_DEBUG_
    printf("DEBUG: Starting proper CRN generation with correlation distances from sls_table.h\n");
#endif
    
    // Select appropriate correlation distances based on scenario from sls_table.h
    const corrDist_t* corrDistLos;
    const corrDist_t* corrDistNlos;
    const corrDist_t* corrDistO2i;
    
    switch (m_sysConfig->scenario) {
        case scenario_t::UMa:
            corrDistLos = &corrDistUmaLos;
            corrDistNlos = &corrDistUmaNlos;
            corrDistO2i = &corrDistUmaO2i;
#ifdef SLS_DEBUG_
            printf("DEBUG: Using UMa correlation distances\n");
#endif
            break;
        case scenario_t::UMi:
            corrDistLos = &corrDistUmiLos;
            corrDistNlos = &corrDistUmiNlos;
            corrDistO2i = &corrDistUmiO2i;
#ifdef SLS_DEBUG_
            printf("DEBUG: Using UMi correlation distances\n");
#endif
            break;
        case scenario_t::RMa:
            corrDistLos = &corrDistRmaLos;
            corrDistNlos = &corrDistRmaNlos;
            corrDistO2i = &corrDistRmaO2i;
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
        CHECK_CUDAERROR(cudaMalloc((void**)&m_d_corrDistLos, 7 * sizeof(float)));
    }
    if (m_d_corrDistNlos == nullptr) {
        CHECK_CUDAERROR(cudaMalloc((void**)&m_d_corrDistNlos, 6 * sizeof(float)));
    }
    if (m_d_corrDistO2i == nullptr) {
        CHECK_CUDAERROR(cudaMalloc((void**)&m_d_corrDistO2i, 6 * sizeof(float)));
    }
    
    // Set correlation distances for LOS case [SF, K, DS, ASD, ASA, ZSD, ZSA]
    float losCorr[7] = {
        corrDistLos->SF,
        corrDistLos->K,
        corrDistLos->DS,
        corrDistLos->ASD,
        corrDistLos->ASA,
        corrDistLos->ZSD,
        corrDistLos->ZSA
    };
    
    // Set correlation distances for NLOS case [SF, DS, ASD, ASA, ZSD, ZSA] (no K)
    float nlosCorr[6] = {
        corrDistNlos->SF,
        corrDistNlos->DS,
        corrDistNlos->ASD,
        corrDistNlos->ASA,
        corrDistNlos->ZSD,
        corrDistNlos->ZSA
    };
    
    // Set correlation distances for O2I case [SF, DS, ASD, ASA, ZSD, ZSA] (no K)
    float o2iCorr[6] = {
        corrDistO2i->SF,
        corrDistO2i->DS,
        corrDistO2i->ASD,
        corrDistO2i->ASA,
        corrDistO2i->ZSD,
        corrDistO2i->ZSA
    };
    
    // Copy correlation distances to GPU memory
    CHECK_CUDAERROR(cudaMemcpyAsync(m_d_corrDistLos, losCorr, 7 * sizeof(float), cudaMemcpyHostToDevice, m_strm));
    CHECK_CUDAERROR(cudaMemcpyAsync(m_d_corrDistNlos, nlosCorr, 6 * sizeof(float), cudaMemcpyHostToDevice, m_strm));
    CHECK_CUDAERROR(cudaMemcpyAsync(m_d_corrDistO2i, o2iCorr, 6 * sizeof(float), cudaMemcpyHostToDevice, m_strm));
    
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
        std::vector<float*> losGrids(nSite * 7);
        for (uint16_t siteIdx = 0; siteIdx < nSite; siteIdx++) {
            for (int lsp = 0; lsp < 7; lsp++) {
                int idx = siteIdx * 7 + lsp;
                CHECK_CUDAERROR(cudaMalloc((void**)&losGrids[idx], m_crnGridSize * sizeof(float)));
#ifdef SLS_DEBUG_
                printf("DEBUG: Allocated LOS grid site %d LSP %d (idx %d): %p, size %d\n", 
                       siteIdx, lsp, idx, losGrids[idx], m_crnGridSize);
#endif
            }
        }
        // Copy pointers to GPU (use synchronous copy since losGrids is local and will be destroyed)
        CHECK_CUDAERROR(cudaMemcpy(m_d_crnLos, losGrids.data(), nSite * 7 * sizeof(float*), cudaMemcpyHostToDevice));
        
#ifdef SLS_DEBUG_
        printf("DEBUG: Successfully copied %d LOS grid pointers to device at %p\n", nSite * 7, m_d_crnLos);
#endif
        
        // Allocate individual grids for NLOS scenarios
        std::vector<float*> nlosGrids(nSite * 6);
        for (uint16_t siteIdx = 0; siteIdx < nSite; siteIdx++) {
            for (int lsp = 0; lsp < 6; lsp++) {
                int idx = siteIdx * 6 + lsp;
                CHECK_CUDAERROR(cudaMalloc((void**)&nlosGrids[idx], m_crnGridSize * sizeof(float)));
#ifdef SLS_DEBUG_
                printf("DEBUG: Allocated NLOS grid site %d LSP %d (idx %d): %p, size %d\n", 
                       siteIdx, lsp, idx, nlosGrids[idx], m_crnGridSize);
#endif
            }
        }
        // Copy pointers to GPU (use synchronous copy since nlosGrids is local and will be destroyed)
        CHECK_CUDAERROR(cudaMemcpy(m_d_crnNlos, nlosGrids.data(), nSite * 6 * sizeof(float*), cudaMemcpyHostToDevice));
        
        // Allocate individual grids for O2I scenarios
        std::vector<float*> o2iGrids(nSite * 6);
        for (uint16_t siteIdx = 0; siteIdx < nSite; siteIdx++) {
            for (int lsp = 0; lsp < 6; lsp++) {
                int idx = siteIdx * 6 + lsp;
                CHECK_CUDAERROR(cudaMalloc((void**)&o2iGrids[idx], m_crnGridSize * sizeof(float)));
#ifdef SLS_DEBUG_
                printf("DEBUG: Allocated O2I grid site %d LSP %d (idx %d): %p, size %d\n", 
                       siteIdx, lsp, idx, o2iGrids[idx], m_crnGridSize);
#endif
            }
        }
        // Copy pointers to GPU (use synchronous copy since o2iGrids is local and will be destroyed)
        CHECK_CUDAERROR(cudaMemcpy(m_d_crnO2i, o2iGrids.data(), nSite * 6 * sizeof(float*), cudaMemcpyHostToDevice));
        
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
    float* d_tempCRN;
    CHECK_CUDAERROR(cudaMalloc((void**)&d_tempCRN, tempCrnSize * sizeof(float)));
    
    // Verify allocation was successful
    if (d_tempCRN == nullptr) {
        printf("ERROR: Failed to allocate temp CRN memory of size %d elements\n", tempCrnSize);
        return;
    }
    
    // Use fixed thread configuration matching the curandState allocation
    const int maxCrnBlocks = 128;  // Must match allocation in sls_chan.cu
    const int threadsPerBlock = 256;  // Must match allocation in sls_chan.cu
    
    int finalNX_temp = paddedNX_temp - (2 * (int)D_temp + 1) + 1;  // Final grid size after convolution
    int finalNY_temp = paddedNY_temp - (2 * (int)D_temp + 1) + 1;  // Final grid size after convolution
    
    // Calculate elements per thread dynamically (same as in sls_chan.cu)
    // Note: This calculation is performed inside the kernel as well
    int totalElements = finalNX_temp * finalNY_temp;
    int totalThreads = maxCrnBlocks * threadsPerBlock;
    
#ifdef SLS_DEBUG_
    int elementsPerThread = (totalElements + totalThreads - 1) / totalThreads;
    printf("DEBUG: Host CRN calculation - Elements: %d, Threads: %d, ElementsPerThread: %d\n",
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
    int maxSharedMemSize = std::max(maxL, threadsPerBlock) * sizeof(float);  // Max of filter or power array
    
    // Ensure all prior allocations and copies are complete
    CHECK_CUDAERROR(cudaStreamSynchronize(m_strm));
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("ERROR: Previous CUDA operation failed: %s\n", cudaGetErrorString(err));
        return;
    }
    
    // Copy correlation distances to host for kernel calls
    float losCorrelationDists[7];
    float nlosCorrelationDists[6];
    float o2iCorrelationDists[6];
    
    CHECK_CUDAERROR(cudaMemcpy(losCorrelationDists, m_d_corrDistLos, 7 * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDAERROR(cudaMemcpy(nlosCorrelationDists, m_d_corrDistNlos, 6 * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDAERROR(cudaMemcpy(o2iCorrelationDists, m_d_corrDistO2i, 6 * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Copy CRN grid pointers to host for kernel calls
    std::vector<float*> losCrnGrids(nSite * 7);
    std::vector<float*> nlosCrnGrids(nSite * 6);
    std::vector<float*> o2iCrnGrids(nSite * 6);
    
    CHECK_CUDAERROR(cudaMemcpy(losCrnGrids.data(), m_d_crnLos, nSite * 7 * sizeof(float*), cudaMemcpyDeviceToHost));
    CHECK_CUDAERROR(cudaMemcpy(nlosCrnGrids.data(), m_d_crnNlos, nSite * 6 * sizeof(float*), cudaMemcpyDeviceToHost));
    CHECK_CUDAERROR(cudaMemcpy(o2iCrnGrids.data(), m_d_crnO2i, nSite * 6 * sizeof(float*), cudaMemcpyDeviceToHost));
    
    // Generate LOS grids - one kernel call per site per LSP
    for (uint16_t siteIdx = 0; siteIdx < nSite; siteIdx++) {
        for (int lsp = 0; lsp < 7; lsp++) {
            int idx = siteIdx * 7 + lsp;
            generateSingleCRNKernel<<<numBlocks, threadsPerBlockDim, maxSharedMemSize, m_strm>>>(
                d_tempCRN, losCrnGrids[idx],
                m_maxX, m_minX, m_maxY, m_minY,
                losCorrelationDists[lsp], m_d_curandStates, m_maxCurandStates
            );
        }
    }
    
#ifdef SLS_DEBUG_
    printf("DEBUG: Generated LOS grids with spatial correlation (%d sites x 7 LSPs = %d kernels)\n", nSite, nSite * 7);
#endif
    
    // Generate NLOS grids - one kernel call per site per LSP
    for (uint16_t siteIdx = 0; siteIdx < nSite; siteIdx++) {
        for (int lsp = 0; lsp < 6; lsp++) {
            int idx = siteIdx * 6 + lsp;
            generateSingleCRNKernel<<<numBlocks, threadsPerBlockDim, maxSharedMemSize, m_strm>>>(
                d_tempCRN, nlosCrnGrids[idx],
                m_maxX, m_minX, m_maxY, m_minY,
                nlosCorrelationDists[lsp], m_d_curandStates, m_maxCurandStates
            );
        }
    }
    
#ifdef SLS_DEBUG_
    printf("DEBUG: Generated NLOS grids with spatial correlation (%d sites x 6 LSPs = %d kernels)\n", nSite, nSite * 6);
#endif
    
    // Generate O2I grids - one kernel call per site per LSP
    for (uint16_t siteIdx = 0; siteIdx < nSite; siteIdx++) {
        for (int lsp = 0; lsp < 6; lsp++) {
            int idx = siteIdx * 6 + lsp;
            generateSingleCRNKernel<<<numBlocks, threadsPerBlockDim, maxSharedMemSize, m_strm>>>(
                d_tempCRN, o2iCrnGrids[idx],
                m_maxX, m_minX, m_maxY, m_minY,
                o2iCorrelationDists[lsp], m_d_curandStates, m_maxCurandStates
            );
        }
    }
    
    // Free temp memory
    CHECK_CUDAERROR(cudaFree(d_tempCRN));
    
#ifdef SLS_DEBUG_
    printf("DEBUG: Generated O2I grids with spatial correlation\n");
#endif
    
    // Synchronize and check for errors
    CHECK_CUDAERROR(cudaStreamSynchronize(m_strm));
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("ERROR: CRN generation kernel failed: %s\n", cudaGetErrorString(err));
        return;
    }

#ifdef SLS_DEBUG_
    printf("DEBUG: CRN generation completed successfully\n");
#endif

    // Step 2: Normalize all CRN grids using efficient single-kernel approach    
    // Shared memory size for 1024 threads (power reduction)
    const size_t sharedMemSize = 1024 * sizeof(float);
    
    // Calculate total number of grids for ALL sites
    const int totalLosGrids = nSite * 7;    // nSite sites × 7 LSPs
    const int totalNlosGrids = nSite * 6;   // nSite sites × 6 LSPs
    const int totalO2iGrids = nSite * 6;    // nSite sites × 6 LSPs
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
    float** d_allCrnGrids;
    CHECK_CUDAERROR(cudaMalloc(&d_allCrnGrids, totalGrids * sizeof(float*)));
    
    // Single copy operation for all grid pointers
    CHECK_CUDAERROR(cudaMemcpy(d_allCrnGrids, allCrnGrids.data(), totalGrids * sizeof(float*), cudaMemcpyHostToDevice));
    
    // Launch normalization kernels - one block per CRN grid, 1024 threads per block
    // Normalize LOS grids (totalLosGrids blocks, 1024 threads each)
    normalizeCRNGridsKernel<<<totalLosGrids, 1024, sharedMemSize, m_strm>>>(
        d_allCrnGrids, totalElements, totalLosGrids);
    
    // Normalize NLOS grids (totalNlosGrids blocks, 1024 threads each)  
    normalizeCRNGridsKernel<<<totalNlosGrids, 1024, sharedMemSize, m_strm>>>(
        d_allCrnGrids + totalLosGrids, totalElements, totalNlosGrids);
    
    // Normalize O2I grids (totalO2iGrids blocks, 1024 threads each)
    normalizeCRNGridsKernel<<<totalO2iGrids, 1024, sharedMemSize, m_strm>>>(
        d_allCrnGrids + totalLosGrids + totalNlosGrids, totalElements, totalO2iGrids);
    
    // Clean up device memory
    CHECK_CUDAERROR(cudaFree(d_allCrnGrids));
    
    // Final synchronization
    CHECK_CUDAERROR(cudaStreamSynchronize(m_strm));
    
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
__global__ void generateSingleCRNKernel(
    float* tempCRN, float* outputCRN,
    float maxX, float minX, float maxY, float minY,
    float correlationDist, curandState* curandStates, uint32_t maxCurandStates
) {
    // 1D thread block coordinates (simplified for fixed block configuration)
    int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;  // Local thread ID within block
    
    // Calculate grid dimensions
    float D = 3.0f * correlationDist;
    uint32_t paddedNX = (uint32_t)roundf(maxX - minX + 1.0f + 2.0f * D);
    uint32_t paddedNY = (uint32_t)roundf(maxY - minY + 1.0f + 2.0f * D);
    
    // Create exponential correlation filter
    uint32_t L;
    if (correlationDist == 0.0f) {
        L = 1;
    } else {
        L = 2 * (uint32_t)D + 1;
    }
    
    // Allocate shared memory for filter coefficients and power calculation
    extern __shared__ float dynamicShared[];
    float* h = dynamicShared;  // Filter coefficients at the beginning
    
    // Generate filter coefficients in shared memory (one thread per block does this)
    if (tid == 0) {
        if (correlationDist == 0.0f) {
            h[0] = 1.0f;
        } else {
            for (uint32_t k = 0; k < L; k++) {
                h[k] = expf(-fabsf((float)(k - (uint32_t)D)) / correlationDist);
            }
        }
    }
    
    __syncthreads();  // Ensure all threads have filter ready
    
    // Calculate final output dimensions after convolution
    uint32_t finalNX = paddedNX - L + 1;  // same with maxX - minX + 1
    uint32_t finalNY = paddedNY - L + 1;  // same with maxY - minY + 1
    
    // Calculate elements per thread dynamically (must match host calculation)
    uint32_t totalElements = finalNX * finalNY;
    uint32_t totalThreads = gridDim.x * blockDim.x;  // Total threads across all blocks
    uint32_t elementsPerThread = (totalElements + totalThreads - 1) / totalThreads;
    
    // Early termination if this thread has no work to do
    if (globalThreadId * elementsPerThread >= totalElements) {
        return;  // This thread is beyond the required range
    }
    
#ifdef SLS_DEBUG_
    if (globalThreadId == 0) {  // Only print once per kernel
        printf("DEBUG: CorrDist=%.1f, Grid=%dx%d, Elements=%d, Threads=%d, ElemPerThread=%d\n",
               correlationDist, finalNX, finalNY, totalElements, totalThreads, elementsPerThread);
    }
#endif
    
    // Step 1: Generate uncorrelated noise into tempCRN using existing curand states
    int threadId = globalThreadId % maxCurandStates;
    curandState localState = curandStates[threadId];
    
    // Calculate total padded elements for tempCRN
    uint32_t totalPaddedElements = paddedNX * paddedNY;
    uint32_t paddedElementsPerThread = (totalPaddedElements + totalThreads - 1) / totalThreads;
    
    // Generate uncorrelated noise in padded space (tempCRN)
    for (uint32_t elem = 0; elem < paddedElementsPerThread; elem++) {
        uint32_t paddedLinearIdx = globalThreadId * paddedElementsPerThread + elem;
        if (paddedLinearIdx >= totalPaddedElements) break;
        
        tempCRN[paddedLinearIdx] = curand_normal(&localState);
    }
    
    // Update curand state
    curandStates[threadId] = localState;
    
    // Step 2: Clear output CRN in correlated space
    for (uint32_t elem = 0; elem < elementsPerThread; elem++) {
        uint32_t linearIdx = globalThreadId * elementsPerThread + elem;
        if (linearIdx >= totalElements) break;
        
        outputCRN[linearIdx] = 0.0f;  // Clear before accumulation
    }
    
    // Synchronize to ensure all uncorrelated noise is generated and all output is cleared
    __syncthreads();
    
    // Step 3: Apply convolution filter from tempCRN to outputCRN
    if (correlationDist == 0.0f) {
        // No correlation case: copy directly from tempCRN to outputCRN
        for (uint32_t elem = 0; elem < elementsPerThread; elem++) {
            uint32_t linearIdx = globalThreadId * elementsPerThread + elem;
            if (linearIdx >= totalElements) break;
            
            // Convert linear index to 2D coordinates in final space
            uint32_t curr_i = linearIdx / finalNY;
            uint32_t curr_j = linearIdx % finalNY;
            
            if (curr_i < finalNX && curr_j < finalNY) {
                // Map to tempCRN coordinates (same position, no offset for zero correlation)
                uint32_t tempIdx = curr_i * paddedNY + curr_j;
                if (tempIdx < totalPaddedElements) {
                    outputCRN[linearIdx] = tempCRN[tempIdx];
                }
            }
        }
    } else {
        // Apply 2D convolution filter from tempCRN to outputCRN
        for (uint32_t elem = 0; elem < elementsPerThread; elem++) {
            uint32_t linearIdx = globalThreadId * elementsPerThread + elem;
            if (linearIdx >= totalElements) break;
            
            // Convert linear index to 2D coordinates in final space
            uint32_t curr_i = linearIdx / finalNY;
            uint32_t curr_j = linearIdx % finalNY;
            
            if (curr_i < finalNX && curr_j < finalNY) {
                float sum = 0.0f;
                
                // Apply 2D convolution using separable filter h ⊗ h
                for (uint32_t di = 0; di < L; di++) {
                    for (uint32_t dj = 0; dj < L; dj++) {
                        uint32_t input_i = curr_i + di;
                        uint32_t input_j = curr_j + dj;
                        
                        if (input_i < paddedNX && input_j < paddedNY) {
                            uint32_t tempIdx = input_i * paddedNY + input_j;
                            if (tempIdx < totalPaddedElements) {
                                float weight = h[di] * h[dj];  // Separable 2D filter
                                sum += weight * tempCRN[tempIdx];
                            }
                        }
                    }
                }
                
                outputCRN[linearIdx] = sum;
            }
        }  // End of elementsPerThread loop
        
        // Note: Power normalization is now handled by separate normalizeCRNKernel
        // This allows for proper global normalization across all blocks
    }
}

// Explicit template instantiations
template class slsChan<float, float2>;