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
#include <stdexcept>
#include <string>

// Helper device function to wrap azimuth angle to [-180, 180] degrees
__device__ __forceinline__ float wrapAzimuthGPU(const float phi) {
    float wrapped = fmodf(phi + 180.0f, 360.0f);
    if (wrapped < 0.0f) {
        wrapped += 360.0f;
    }
    return wrapped - 180.0f;
}

// Helper device function to wrap zenith angle to [0, 180] degrees
__device__ __forceinline__ float wrapZenithGPU(const float theta) {
    float wrapped = fmodf(theta, 360.0f);
    if (wrapped < 0.0f) {
        wrapped += 360.0f;
    }
    // Map to [0, 180] using symmetry
    if (wrapped > 180.0f) {
        wrapped = 360.0f - wrapped;
    }
    return wrapped;
}

// Forward declarations of device functions
__device__ void genClusterDelayAndPowerGPU(float DS, const float* r_tao, bool losInd,
                                          uint16_t& nCluster, float K, const float* xi,
                                          float* delays, float* powers,
                                          uint16_t* strongest2clustersIdx,
                                          curandState* state, uint8_t outdoor_ind = 1);

__device__ void genClusterAngleGPU(uint16_t nCluster, float C_ASA, float C_ASD,
                                  float C_phi_NLOS, float c_phi_O2I, float C_theta_NLOS,
                                  float C_phi_LOS, float C_theta_LOS, float C_theta_O2I,
                                  float ASA, float ASD, float ZSA, float ZSD,
                                  float phi_LOS_AOA, float phi_LOS_AOD,
                                  float theta_LOS_ZOA, float theta_LOS_ZOD,
                                  float mu_offset_ZOD, bool losInd,
                                  uint8_t outdoor_ind, float K,
                                  const float* powers, float* phi_n_AoA,
                                  float* phi_n_AoD, float* theta_n_ZOD,
                                  float* theta_n_ZOA, curandState* state);

__device__ void genRayAngleGPU(uint16_t nCluster, uint16_t nRay,
                              float ASA, float ASD, float ZSA, float ZSD,
                              const float* phi_n_AoA, const float* phi_n_AoD,
                              const float* theta_n_ZOA, const float* theta_n_ZOD,
                              float* phi_mn_AoA, float* phi_mn_AoD,
                              float* theta_mn_ZOA, float* theta_mn_ZOD,
                              float C_ASA, float C_ASD, float C_ZSA, float mu_lgZSD,
                              const uint16_t* strongest2clustersIdx,
                              curandState* state);

// Device helper function to calculate CFR offsets based on optional dimension layout
__device__ __forceinline__ uint32_t calculateCfrOffset(uint16_t batchIdx, uint16_t ueAntIdx, uint16_t bsAntIdx, 
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

// Device helper function to calculate SC CFR offsets based on optional dimension layout
__device__ __forceinline__ uint32_t calculateScCfrOffset(uint16_t batchIdx, uint16_t ueAntIdx, uint16_t bsAntIdx, 
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

// Device function to calculate CFR from CIR
template <typename Tcomplex>
__device__ Tcomplex calCfrbyCir(float freqKHz, uint16_t cirNtaps, 
                                const float* cirNormDelayUs2Pi, 
                                const Tcomplex* cirCoeff,
                                float normalizationFactor) {
    Tcomplex cfr = {0.0f, 0.0f};
    
    // Safety check for inputs
#ifdef SLS_DEBUG_
    if (isnan(freqKHz) || isinf(freqKHz) || isnan(normalizationFactor) || isinf(normalizationFactor)) {
        printf("ERROR: Invalid input parameters: freqKHz=%f, normalizationFactor=%f\n", 
               freqKHz, normalizationFactor);
        return cfr; // Return zero CFR for invalid inputs
    }
#endif
    
    for (uint16_t tapIdx = 0; tapIdx < cirNtaps; tapIdx++) {
        // Safety checks for delay and coefficient values
        float delay = cirNormDelayUs2Pi[tapIdx];
#ifdef SLS_DEBUG_
        if (isnan(delay) || isinf(delay)) {
            printf("ERROR: Invalid delay value: %f (rayIdx=%d)\n", delay, tapIdx);
            continue; // Skip this tap
        }
#endif
        
        Tcomplex coeff = cirCoeff[tapIdx];
#ifdef SLS_DEBUG_
        if (isnan(coeff.x) || isnan(coeff.y) || isinf(coeff.x) || isinf(coeff.y)) {
            printf("ERROR: Invalid coefficient: (%f, %f) (rayIdx=%d)\n", coeff.x, coeff.y, tapIdx);
            continue; // Skip this tap
        }
#endif
        
        float phase = freqKHz * delay * 1e-3f; // kHz * 2*pi*delayUs * 1e-3 = rad
        
        // Limit phase to reasonable range to prevent numerical issues
        if (fabsf(phase) > 1e6f) {
            continue; // Skip taps with extreme phase values
        }
        
        float cosPhase = cosf(phase);
        float sinPhase = sinf(phase);
        
        // Safety check for trigonometric results
#ifdef SLS_DEBUG_
        if (isnan(cosPhase) || isnan(sinPhase)) {
            printf("ERROR: Invalid phase calculation: cos=%f, sin=%f (rayIdx=%d)\n", 
                   cosPhase, sinPhase, tapIdx);
            continue; // Skip this tap
        }
#endif
        
        // Complex multiplication: cirCoeff[tapIdx] * exp(-j * 2*pi*f*tau)
        cfr.x += coeff.x * cosPhase + coeff.y * sinPhase;
        cfr.y += coeff.y * cosPhase - coeff.x * sinPhase;
    }
    
    // Apply energy conservation normalization (Parseval's theorem)
    cfr.x *= normalizationFactor;
    cfr.y *= normalizationFactor;
    
    // Final safety check for output
#ifdef SLS_DEBUG_
    if (isnan(cfr.x) || isnan(cfr.y) || isinf(cfr.x) || isinf(cfr.y)) {
        printf("ERROR: Invalid CFR value: (%f, %f)\n", cfr.x, cfr.y);
        cfr.x = 0.0f;
        cfr.y = 0.0f;
    }
#endif
    
    return cfr;
}

// Device function to calculate field components for antenna pattern
__device__ void calculateFieldComponentsGPU(const AntPanelConfig& antPanelConfig, float theta, float phi, 
                                           float zeta, float& F_theta, float& F_phi) {
    // Normalize angles for pattern lookup
    int theta_idx = (int)roundf(theta);
    int phi_idx = (int)roundf(phi);
    float zeta_rad = zeta * M_PI / 180.0f;
    
    // Wrap theta into [0, 360] then map to [0, 180] using symmetry
    if (theta_idx < 0 || theta_idx >= 360) {
        theta_idx = theta_idx % 360;  // First modulo
        if (theta_idx < 0) {
            theta_idx += 360;  // Only add 360 if negative, avoiding second modulo
        }
    }
    theta_idx = (theta_idx > 180) ? 360 - theta_idx : theta_idx;
    
    // Handle phi: only do modulo if outside [0, 359] range
    if (phi_idx < 0 || phi_idx > 359) {
        phi_idx = phi_idx % 360;  // First modulo
        if (phi_idx < 0) {
            phi_idx += 360;  // Only add 360 if negative, avoiding second modulo
        }
    }

    // Convert from dB to linear
    // Table 7.3-1: A''(theta,phi) = -min{-(A_V + A_H), A_max}; the separable table sum
    // alone would reach -60 dB in back lobes, so re-clamp the combined attenuation to A_max
    float A_db_3D = antPanelConfig.antTheta[theta_idx] + antPanelConfig.antPhi[phi_idx];
    if (antPanelConfig.antModel == 1) {
        A_db_3D = fmaxf(A_db_3D, -SLS_ANTENNA_PATTERN_A_MAX_DB) + SLS_ANTENNA_GAIN_MAX_DBI;
    }
    float A_3D_sqrt = powf(10.0f, A_db_3D / 20.0f); // equivalent to sqrt(10^(A_db_3D/10))
    
    // Apply polarization
    F_theta = A_3D_sqrt * cosf(zeta_rad);
    F_phi = A_3D_sqrt * sinf(zeta_rad);
}

// Precompute Tx-side terms for a ray (bsAnt-specific, independent of utAntIdx).
// tx_pol[i] = sum_j( term2[i][j] * F_tx[j] ) * term5  (angles in degrees, like calculateRayCoefficientGPU)
// theta_ZOD/phi_AOD are the panel-local (eq 7.1-7/-8) angles; psi_tx is the eq 7.1-15
// polarization basis rotation of this ray for the BS panel orientation.
template <typename Tcomplex>
__device__ void computeTxRayTermsGPU(
    const AntPanelConfig& bsAntConfig, int bsAntIdx, float theta_ZOD, float phi_AOD,
    float psi_tx, float xpr, const float* randomPhase,
    Tcomplex* tx_pol)  // output: [2]
{
    const float d2r = M_PI / 180.0f;
    float theta_ZOD_rad = theta_ZOD * d2r;
    float phi_AOD_rad   = phi_AOD   * d2r;

    int M = bsAntConfig.antSize[2];
    int N = bsAntConfig.antSize[3];
    int P = bsAntConfig.antSize[4];
    int p_tx = bsAntIdx % P;
    float d_h_tx = bsAntConfig.antSpacing[2];
    float d_v_tx = bsAntConfig.antSpacing[3];
    // 38.901 Fig 7.3-1: the panel lies in the y-z plane. Column n (of N) steps along y with
    // spacing d_H; row m (of M) steps along z with spacing d_V; x (boresight) stays 0.
    float d_bar_tx_y = ((bsAntIdx / P) % N) * d_h_tx;
    float d_bar_tx_z = ((bsAntIdx / (N * P)) % M) * d_v_tx;

    float F_tx_theta, F_tx_phi;
    calculateFieldComponentsGPU(bsAntConfig, theta_ZOD, phi_AOD,
                                bsAntConfig.antPolarAngles[p_tx],
                                F_tx_theta, F_tx_phi);
    // eq 7.1-11: rotate the LCS field components into the GCS spherical basis
    {
        float c_psi = cosf(psi_tx), s_psi = sinf(psi_tx);
        float Ft = c_psi * F_tx_theta - s_psi * F_tx_phi;
        float Fp = s_psi * F_tx_theta + c_psi * F_tx_phi;
        F_tx_theta = Ft;
        F_tx_phi = Fp;
    }

    float sin_zod = sinf(theta_ZOD_rad), cos_zod = cosf(theta_ZOD_rad);
    float sin_aod = sinf(phi_AOD_rad);
    float dot_tx = sin_zod * sin_aod * d_bar_tx_y + cos_zod * d_bar_tx_z;
    Tcomplex term5 = make_cuComplex(cosf(2.0f * M_PI * dot_tx), sinf(2.0f * M_PI * dot_tx));

    float sqrt_kappa = sqrtf(xpr);
    Tcomplex t00 = make_cuComplex( cosf(randomPhase[0]),               sinf(randomPhase[0]));
    Tcomplex t01 = make_cuComplex( cosf(randomPhase[1]) / sqrt_kappa,  sinf(randomPhase[1]) / sqrt_kappa);
    Tcomplex t10 = make_cuComplex( cosf(randomPhase[2]) / sqrt_kappa,  sinf(randomPhase[2]) / sqrt_kappa);
    Tcomplex t11 = make_cuComplex( cosf(randomPhase[3]),               sinf(randomPhase[3]));

    // tx_pol[i] = sum_j term2[i][j]*F_tx[j], then * term5
    Tcomplex s0 = make_cuComplex(t00.x * F_tx_theta + t01.x * F_tx_phi,
                                 t00.y * F_tx_theta + t01.y * F_tx_phi);
    Tcomplex s1 = make_cuComplex(t10.x * F_tx_theta + t11.x * F_tx_phi,
                                 t10.y * F_tx_theta + t11.y * F_tx_phi);
    tx_pol[0] = cuCmulf(s0, term5);
    tx_pol[1] = cuCmulf(s1, term5);
}

// Compute the Rx contribution for one utAnt given precomputed tx_pol.
// theta_ZOA/phi_AOA are panel-local (eq 7.1-7/-8) angles for the pattern and array phase;
// psi_rx is the eq 7.1-15 polarization basis rotation for the UT panel orientation;
// theta_ZOA_gcs/phi_AOA_gcs are the uncorrected GCS angles, required for the Doppler term of
// eq 7.5-25 where r_rx and the (GCS) UT velocity must be expressed in the SAME frame.
template <typename Tcomplex>
__device__ Tcomplex computeRxRayContribGPU(
    const AntPanelConfig& utAntConfig, int ueAntIdx, float theta_ZOA, float phi_AOA,
    float theta_ZOA_gcs, float phi_AOA_gcs,
    float psi_rx, float currentTime, const float* utVelocityPolar, float lambda_0,
    const Tcomplex* tx_pol)  // precomputed [2]
{
    const float d2r = M_PI / 180.0f;
    float theta_ZOA_rad = theta_ZOA * d2r;
    float phi_AOA_rad   = phi_AOA   * d2r;

    int M = utAntConfig.antSize[2];
    int N = utAntConfig.antSize[3];
    int P = utAntConfig.antSize[4];
    int p_rx = ueAntIdx % P;
    float d_h_rx = utAntConfig.antSpacing[2];
    float d_v_rx = utAntConfig.antSpacing[3];
    // 38.901 Fig 7.3-1: panel in the y-z plane (column n -> y*d_H, row m -> z*d_V)
    float d_bar_rx_y = ((ueAntIdx / P) % N) * d_h_rx;
    float d_bar_rx_z = ((ueAntIdx / (N * P)) % M) * d_v_rx;

    float F_rx_theta, F_rx_phi;
    calculateFieldComponentsGPU(utAntConfig, theta_ZOA, phi_AOA,
                                utAntConfig.antPolarAngles[p_rx],
                                F_rx_theta, F_rx_phi);
    // eq 7.1-11: rotate the LCS field components into the GCS spherical basis
    {
        float c_psi = cosf(psi_rx), s_psi = sinf(psi_rx);
        float Ft = c_psi * F_rx_theta - s_psi * F_rx_phi;
        float Fp = s_psi * F_rx_theta + c_psi * F_rx_phi;
        F_rx_theta = Ft;
        F_rx_phi = Fp;
    }

    float sin_zoa = sinf(theta_ZOA_rad), cos_zoa = cosf(theta_ZOA_rad);
    float sin_aoa = sinf(phi_AOA_rad);
    float dot_rx = sin_zoa * sin_aoa * d_bar_rx_y + cos_zoa * d_bar_rx_z;
    Tcomplex term4 = make_cuComplex(cosf(2.0f * M_PI * dot_rx), sinf(2.0f * M_PI * dot_rx));

    // Doppler per eq 7.5-25: r_rx (from GCS ray angles) dot GCS velocity (vx, vy, vz)
    float theta_gcs_rad = theta_ZOA_gcs * d2r;
    float phi_gcs_rad   = phi_AOA_gcs   * d2r;
    float sin_zoa_gcs = sinf(theta_gcs_rad);
    float doppler_phase = 2.0f * M_PI
        * (sin_zoa_gcs * cosf(phi_gcs_rad) * utVelocityPolar[0]
         + sin_zoa_gcs * sinf(phi_gcs_rad) * utVelocityPolar[1]
         + cosf(theta_gcs_rad)             * utVelocityPolar[2])
        / lambda_0 * currentTime;
    Tcomplex term6 = make_cuComplex(cosf(doppler_phase), sinf(doppler_phase));

    // result = (F_rx_theta * tx_pol[0] + F_rx_phi * tx_pol[1]) * term4 * term6
    Tcomplex dot_pol = make_cuComplex(tx_pol[0].x * F_rx_theta + tx_pol[1].x * F_rx_phi,
                                     tx_pol[0].y * F_rx_theta + tx_pol[1].y * F_rx_phi);
    return cuCmulf(cuCmulf(dot_pol, term4), term6);
}

// (removed) calculateRayCoefficientGPU: dead code superseded by computeTxRayTermsGPU +
// computeRxRayContribGPU; it carried the pre-fix x-y element placement and LCS-frame Doppler.

// Device function to calculate LOS ray coefficient (GPU version of CPU calculateLOSCoefficient)
// theta_LOS_*/phi_LOS_* are panel-local (eq 7.1-7/-8) angles; psi_rx/psi_tx are the
// eq 7.1-15 polarization rotations for the UT/BS panel orientations; the *_gcs pair carries
// the uncorrected GCS arrival angles for the eq 7.5-25 Doppler term (same frame as UT velocity).
template <typename Tcomplex>
__device__ Tcomplex calculateLOSCoefficientGPU(
    const AntPanelConfig& utAntConfig, int ueAntIdx, float theta_LOS_ZOA, float phi_LOS_AOA, float psi_rx,
    const AntPanelConfig& bsAntConfig, int bsAntIdx, float theta_LOS_ZOD, float phi_LOS_AOD, float psi_tx,
    float theta_LOS_ZOA_gcs, float phi_LOS_AOA_gcs,
    float currentTime, const float* utVelocityPolar, float lambda_0, float d_3d)
{
    // Compute d_bar_rx (Rx antenna element position)
    // 38.901 Fig 7.3-1: panel in the y-z plane (column n -> y*d_H, row m -> z*d_V)
    int M = utAntConfig.antSize[2];
    int N = utAntConfig.antSize[3];
    int P = utAntConfig.antSize[4];
    int p_rx = ueAntIdx % P;
    float d_h_rx = utAntConfig.antSpacing[2];
    float d_v_rx = utAntConfig.antSpacing[3];
    float d_bar_rx[3] = {
        0.0f,
        ((ueAntIdx / P) % N) * d_h_rx,
        ((ueAntIdx / (N * P)) % M) * d_v_rx
    };

    // Convert angles to radians
    float d2pi = M_PI / 180.0f;
    float theta_LOS_ZOA_rad = theta_LOS_ZOA * d2pi;
    float phi_LOS_AOA_rad = phi_LOS_AOA * d2pi;
    float theta_LOS_ZOD_rad = theta_LOS_ZOD * d2pi;
    float phi_LOS_AOD_rad = phi_LOS_AOD * d2pi;

    // Calculate field patterns for Rx
    float F_rx_theta, F_rx_phi;
    calculateFieldComponentsGPU(utAntConfig, theta_LOS_ZOA, phi_LOS_AOA,
                               utAntConfig.antPolarAngles[p_rx],
                               F_rx_theta, F_rx_phi);
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
    float d_bar_tx[3] = {
        0.0f,
        ((bsAntIdx / P) % N) * d_h_tx,
        ((bsAntIdx / (N * P)) % M) * d_v_tx
    };
    
    // Calculate field patterns for Tx
    float F_tx_theta, F_tx_phi;
    calculateFieldComponentsGPU(bsAntConfig, theta_LOS_ZOD, phi_LOS_AOD,
                               bsAntConfig.antPolarAngles[p_tx],
                               F_tx_theta, F_tx_phi);
    // eq 7.1-11: rotate the LCS field components into the GCS spherical basis
    {
        float c_psi = cosf(psi_tx), s_psi = sinf(psi_tx);
        float Ft = c_psi * F_tx_theta - s_psi * F_tx_phi;
        float Fp = s_psi * F_tx_theta + c_psi * F_tx_phi;
        F_tx_theta = Ft;
        F_tx_phi = Fp;
    }

    // Term 1: Rx antenna field pattern
    Tcomplex term1[2] = {make_cuComplex(F_rx_theta, 0.0f), make_cuComplex(F_rx_phi, 0.0f)};

    // Term 2: LOS polarization matrix [1 0; 0 -1]
    Tcomplex term2[2][2];
    term2[0][0] = make_cuComplex(1.0f, 0.0f);
    term2[0][1] = make_cuComplex(0.0f, 0.0f);
    term2[1][0] = make_cuComplex(0.0f, 0.0f);
    term2[1][1] = make_cuComplex(-1.0f, 0.0f);

    // Term 3: Tx antenna field pattern
    Tcomplex term3[2] = {make_cuComplex(F_tx_theta, 0.0f), make_cuComplex(F_tx_phi, 0.0f)};

    // Term 4: Rx antenna position phase
    float r_head_rx[3] = {
        sinf(theta_LOS_ZOA_rad) * cosf(phi_LOS_AOA_rad),
        sinf(theta_LOS_ZOA_rad) * sinf(phi_LOS_AOA_rad),
        cosf(theta_LOS_ZOA_rad)
    };
    float dot_rx = r_head_rx[0] * d_bar_rx[0] + r_head_rx[1] * d_bar_rx[1] + r_head_rx[2] * d_bar_rx[2];
    Tcomplex term4 = make_cuComplex(cosf(2.0f * M_PI * dot_rx), sinf(2.0f * M_PI * dot_rx));

    // Term 5: Tx antenna position phase
    float r_head_tx[3] = {
        sinf(theta_LOS_ZOD_rad) * cosf(phi_LOS_AOD_rad),
        sinf(theta_LOS_ZOD_rad) * sinf(phi_LOS_AOD_rad),
        cosf(theta_LOS_ZOD_rad)
    };
    float dot_tx = r_head_tx[0] * d_bar_tx[0] + r_head_tx[1] * d_bar_tx[1] + r_head_tx[2] * d_bar_tx[2];
    Tcomplex term5 = make_cuComplex(cosf(2.0f * M_PI * dot_tx), sinf(2.0f * M_PI * dot_tx));

    // Term 6: Doppler phase per eq 7.5-25: r_rx (from GCS LOS angles) dot GCS velocity
    float theta_gcs_rad = theta_LOS_ZOA_gcs * d2pi;
    float phi_gcs_rad   = phi_LOS_AOA_gcs   * d2pi;
    float sin_zoa_gcs = sinf(theta_gcs_rad);
    float doppler_phase = 2.0f * M_PI
        * (sin_zoa_gcs * cosf(phi_gcs_rad) * utVelocityPolar[0]
         + sin_zoa_gcs * sinf(phi_gcs_rad) * utVelocityPolar[1]
         + cosf(theta_gcs_rad)             * utVelocityPolar[2])
        / lambda_0 * currentTime;
    Tcomplex term6 = make_cuComplex(cosf(doppler_phase), sinf(doppler_phase));

#ifdef SLS_DEBUG_
    // Validate final Doppler term
    if (isnan(term6.x) || isnan(term6.y)) {
        printf("ERROR: NaN in Doppler term6: (%f, %f) from phase: %f\n", 
               term6.x, term6.y, doppler_phase);
        return make_cuComplex(0.0f, 0.0f);
    }
#endif

    // Term 7: LOS path phase
    float los_path_phase = -2.0f * M_PI * d_3d / lambda_0;
    Tcomplex term7 = make_cuComplex(cosf(los_path_phase), sinf(los_path_phase));

    // Compute final result: sum over polarizations
    Tcomplex result = make_cuComplex(0.0f, 0.0f);
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            Tcomplex temp = cuCmulf(term1[i], term2[i][j]);
            temp = cuCmulf(temp, term3[j]);
            temp = cuCmulf(temp, term7);
            temp = cuCmulf(temp, term4);
            temp = cuCmulf(temp, term5);
            temp = cuCmulf(temp, term6);
            result = cuCaddf(result, temp);
        }
    }
    return result;
}

// GPU kernel to calculate cluster and ray parameters
__global__ void calClusterRayKernel(
    const CellParam* cellParams,
    const UtParam* utParams,
    const LinkParams* linkParams,
    const CmnLinkParams* cmnLinkParams,
    ClusterParams* clusterParams,
    uint32_t nSite,
    uint32_t nUT,
    uint32_t n_sector_per_site,
    curandState* curandStates)
{
    // Calculate thread index
    uint32_t siteIdx = blockIdx.x;
    uint32_t ueIdx = blockIdx.y * blockDim.x + threadIdx.x;
    uint32_t linkIdx = siteIdx * nUT + ueIdx;

    if (ueIdx >= nUT) return;

    // Load curandState for this thread once at the beginning
    const uint32_t globalThreadId = blockIdx.x * gridDim.y * blockDim.x + 
                                   blockIdx.y * blockDim.x + 
                                   threadIdx.x;
    curandState localState = curandStates[globalThreadId];

    uint8_t losInd = linkParams[linkIdx].losInd;
    
    // Add indoor status for O2I logic similar to large scale params
    uint8_t isO2I = (utParams[ueIdx].outdoor_ind == 0);  // 1 if indoor (O2I), 0 if outdoor
    
    // Calculate proper index for cmnLinkParams arrays: 2 for O2I, 1 for LOS, 0 for NLOS
    uint8_t lspIdx = isO2I ? 2 : losInd;
    
    // Set number of clusters and rays for this link
    uint16_t originalNCluster = cmnLinkParams->nCluster[lspIdx];
    clusterParams[linkIdx].nCluster = originalNCluster;  // Will be updated after filtering
    clusterParams[linkIdx].nRayPerCluster = cmnLinkParams->nRayPerCluster[lspIdx];

#ifdef SLS_DEBUG_
    // Debug link parameters before cluster generation
    if (linkIdx < 6) {
        printf("DEBUG: Link %d - Input params: DS=%.6e, losInd=%d, isO2I=%d, lspIdx=%d, K=%.3f, r_tau=%.3f, xi=%.3f\n",
               linkIdx, linkParams[linkIdx].DS, losInd, isO2I, lspIdx, linkParams[linkIdx].K,
               cmnLinkParams->r_tao[lspIdx], cmnLinkParams->xi[lspIdx]);
        if (linkParams[linkIdx].DS == 0.0f) {
            printf("DEBUG: Link %d has DS=0! This may indicate:\n", linkIdx);
            printf("  - Channel model configured for no delay spread\n");
            printf("  - Possible data initialization issue\n");
            printf("  - Or specific channel scenario (e.g., direct LOS with no multipath)\n");
        }
    }
#endif
    
    // Generate cluster delays and powers
    uint16_t finalNCluster = originalNCluster;
    genClusterDelayAndPowerGPU(linkParams[linkIdx].DS,
                              &cmnLinkParams->r_tao[lspIdx],
                              losInd,
                              finalNCluster,  // Pass by reference to get updated value
                              linkParams[linkIdx].K,
                              &cmnLinkParams->xi[lspIdx],
                              clusterParams[linkIdx].delays,
                              clusterParams[linkIdx].powers,
                              clusterParams[linkIdx].strongest2clustersIdx,
                              &localState,
                              utParams[ueIdx].outdoor_ind);
    
    // Update the cluster count after filtering
    clusterParams[linkIdx].nCluster = finalNCluster;

#ifdef SLS_DEBUG_
    // Debug print cluster delays and powers (only for link 0 to avoid concurrent printing)
    if (linkIdx == 0) {  // Only print for link 0 to avoid concurrent output
        printf("=== DEBUG: Link %d (Site %d, UE %d) ===\n", linkIdx, siteIdx, ueIdx);
        printf("  Parameters: nCluster=%d, LOS=%d, O2I=%d, lspIdx=%d, DS=%.6e, r_tau=%.3f, xi=%.3f\n", 
               clusterParams[linkIdx].nCluster, losInd, isO2I, lspIdx, linkParams[linkIdx].DS,
               cmnLinkParams->r_tao[lspIdx], cmnLinkParams->xi[lspIdx]);
        
        printf("  Delays(ns): [");
        for (uint16_t i = 0; i < clusterParams[linkIdx].nCluster && i < 10; i++) {
            printf("%.2f", clusterParams[linkIdx].delays[i]);
            if (i < clusterParams[linkIdx].nCluster - 1 && i < 9) printf(", ");
        }
        if (clusterParams[linkIdx].nCluster > 20) printf(", ...");
        printf("]\n");
        
        printf("  Powers: [");
        for (uint16_t i = 0; i < clusterParams[linkIdx].nCluster && i < 10; i++) {
            printf("%.6f", clusterParams[linkIdx].powers[i]);
            if (i < clusterParams[linkIdx].nCluster - 1 && i < 9) printf(", ");
        }
        if (clusterParams[linkIdx].nCluster > 10) printf(", ...");
        printf("]\n");
        
        // Calculate and print power statistics
        float totalPower = 0.0f;
        float maxPower = clusterParams[linkIdx].powers[0];
        float minPower = clusterParams[linkIdx].powers[0];
        for (uint16_t i = 0; i < clusterParams[linkIdx].nCluster; i++) {
            totalPower += clusterParams[linkIdx].powers[i];
            maxPower = fmaxf(maxPower, clusterParams[linkIdx].powers[i]);
            minPower = fminf(minPower, clusterParams[linkIdx].powers[i]);
        }
        printf("  Power statistics: Total=%.6f, Max=%.6f, Min=%.6f, Range=%.6f\n", 
               totalPower, maxPower, minPower, maxPower - minPower);
        
        uint16_t strongestIdx1 = clusterParams[linkIdx].strongest2clustersIdx[0];
        uint16_t strongestIdx2 = clusterParams[linkIdx].strongest2clustersIdx[1];
        printf("  Strongest clusters: [%d, %d]\n", strongestIdx1, strongestIdx2);
        printf("    Cluster %d: power=%.6f, delay=%.3f ns\n", 
               strongestIdx1, clusterParams[linkIdx].powers[strongestIdx1], 
               clusterParams[linkIdx].delays[strongestIdx1]);
        printf("    Cluster %d: power=%.6f, delay=%.3f ns\n", 
               strongestIdx2, clusterParams[linkIdx].powers[strongestIdx2], 
               clusterParams[linkIdx].delays[strongestIdx2]);
        printf("=== End Link %d ===\n\n", linkIdx);
    }
    
    // Additional concise cluster power debug for first few links
    if (linkIdx < 6) {
        printf("CLUSTER_POWERS Link %d: [", linkIdx);
        for (uint16_t i = 0; i < clusterParams[linkIdx].nCluster && i < 5; i++) {
            printf("%.4f", clusterParams[linkIdx].powers[i]);
            if (i < clusterParams[linkIdx].nCluster - 1 && i < 4) printf(", ");
        }
        if (clusterParams[linkIdx].nCluster > 5) printf(", ...");
        printf("] nCluster=%d (after filtering), strongest: [%d, %d]\n", 
               clusterParams[linkIdx].nCluster,
               clusterParams[linkIdx].strongest2clustersIdx[0], 
               clusterParams[linkIdx].strongest2clustersIdx[1]);
    }
#endif

    // Generate arrival and departure angles
    genClusterAngleGPU(clusterParams[linkIdx].nCluster,
                      cmnLinkParams->C_ASA[lspIdx],
                      cmnLinkParams->C_ASD[lspIdx],
                      cmnLinkParams->C_phi_NLOS,
                      cmnLinkParams->C_phi_O2I,
                      cmnLinkParams->C_theta_NLOS,
                      cmnLinkParams->C_phi_LOS,
                      cmnLinkParams->C_theta_LOS,
                      cmnLinkParams->C_theta_O2I,
                      linkParams[linkIdx].ASA,
                      linkParams[linkIdx].ASD,
                      linkParams[linkIdx].ZSA,
                      linkParams[linkIdx].ZSD,
                      linkParams[linkIdx].phi_LOS_AOA,
                      linkParams[linkIdx].phi_LOS_AOD,
                      linkParams[linkIdx].theta_LOS_ZOA,
                      linkParams[linkIdx].theta_LOS_ZOD,
                      linkParams[linkIdx].mu_offset_ZOD,
                      losInd,
                      utParams[ueIdx].outdoor_ind,
                      linkParams[linkIdx].K,
                      clusterParams[linkIdx].powers,
                      clusterParams[linkIdx].phi_n_AoA,
                      clusterParams[linkIdx].phi_n_AoD,
                      clusterParams[linkIdx].theta_n_ZOD,
                      clusterParams[linkIdx].theta_n_ZOA,
                      &localState);

    // Generate ray angles
    // Extract cluster spread factors based on LOS state
    float C_ASA = cmnLinkParams->C_ASA[lspIdx];
    float C_ASD = cmnLinkParams->C_ASD[lspIdx];
    float C_ZSA = cmnLinkParams->C_ZSA[lspIdx];
    genRayAngleGPU(clusterParams[linkIdx].nCluster,
                  clusterParams[linkIdx].nRayPerCluster,
                  linkParams[linkIdx].ASA,
                  linkParams[linkIdx].ASD,
                  linkParams[linkIdx].ZSA,
                  linkParams[linkIdx].ZSD,
                  clusterParams[linkIdx].phi_n_AoA,
                  clusterParams[linkIdx].phi_n_AoD,
                  clusterParams[linkIdx].theta_n_ZOA,
                  clusterParams[linkIdx].theta_n_ZOD,
                  clusterParams[linkIdx].phi_n_m_AoA,
                  clusterParams[linkIdx].phi_n_m_AoD,
                  clusterParams[linkIdx].theta_n_m_ZOA,
                  clusterParams[linkIdx].theta_n_m_ZOD,
                  C_ASA,
                  C_ASD,
                  C_ZSA,
                  linkParams[linkIdx].mu_lgZSD,  // Pass mu_lgZSD directly for per-ray calculation
                  clusterParams[linkIdx].strongest2clustersIdx,
                  &localState);

    // Generate XPR and random phases
    for (uint16_t clusterIdx = 0; clusterIdx < clusterParams[linkIdx].nCluster; clusterIdx++) {
        for (uint16_t rayIdx = 0; rayIdx < clusterParams[linkIdx].nRayPerCluster; rayIdx++) {
            // Generate XPR values
            clusterParams[linkIdx].xpr[clusterIdx * clusterParams[linkIdx].nRayPerCluster + rayIdx] = 
                std::pow(10.0f, (cmnLinkParams->mu_XPR[lspIdx] + 
                cmnLinkParams->sigma_XPR[lspIdx] * curand_normal(&localState)) / 10.0f);
        
            // Generate random phases (3GPP Step 10): drawn INDEPENDENTLY per co-sited sector,
            // stored in per-sector blocks of the (per-site) randomPhases array. Steps 1-9
            // (clusters, XPR above) stay shared per site.
            // Stored in RADIANS, uniform(-pi, pi): consumers apply cosf/sinf directly and the
            // CPU path stores radians too.
            const uint32_t rayBase = (clusterIdx * clusterParams[linkIdx].nRayPerCluster + rayIdx) * 4;
            for (uint32_t s = 0; s < n_sector_per_site && s < ClusterParams::MAX_SECTORS; ++s) {
                const uint32_t off = s * ClusterParams::PHASE_SECTOR_STRIDE + rayBase;
                clusterParams[linkIdx].randomPhases[off]     = (curand_uniform(&localState) - 0.5f) * 2.0f * M_PI;
                clusterParams[linkIdx].randomPhases[off + 1] = (curand_uniform(&localState) - 0.5f) * 2.0f * M_PI;
                clusterParams[linkIdx].randomPhases[off + 2] = (curand_uniform(&localState) - 0.5f) * 2.0f * M_PI;
                clusterParams[linkIdx].randomPhases[off + 3] = (curand_uniform(&localState) - 0.5f) * 2.0f * M_PI;
            }
        }
    }
    
    // Store updated curandState back to global memory
    curandStates[globalThreadId] = localState;
}

// GPU kernel to generate CIR
template <typename Tcomplex>
__global__ __launch_bounds__(256, 2) void generateCIRKernel(const activeLink<Tcomplex>* activeLinkParams,
                                  const CellParam* cellParams, const UtParam* utParams,
                                  const LinkParams* linkParams, const ClusterParams* clusterParams,
                                  const CmnLinkParams* cmnLinkParams, const SimConfig* simConfig,
                                  const AntPanelConfig* antPanelConfigs, const SystemLevelConfig* sysConfig,
                                  uint32_t nActiveLinks, float refTime, curandState* curandStates) {
    // Dynamic shared memory: layout [blockDim.x][nUtAnt][N_MAX_TAPS] Tcomplex
    // Sized at launch as: nCellAnt * nUtAnt * N_MAX_TAPS * sizeof(Tcomplex)
    extern __shared__ Tcomplex H_link_ue_shared[];
    uint32_t activeLinkIdx = blockIdx.x;
    uint32_t snapshotIdx = blockIdx.y;
    
    if (activeLinkIdx >= nActiveLinks) return;
    
    // Get parameters from activeLinkParams
    const activeLink<Tcomplex>& activeLink = activeLinkParams[activeLinkIdx];
    uint16_t cid = activeLink.cid;
    uint16_t uid = activeLink.uid;
    uint32_t linkIdx = activeLink.linkIdx;
    uint32_t lspReadIdx = activeLink.lspReadIdx;
    
    // Calculate O2I status for this link
    uint8_t isO2I = (utParams[uid].outdoor_ind == 0);  // 1 if indoor (O2I), 0 if outdoor
    uint8_t lspIdx = isO2I ? 2 : linkParams[lspReadIdx].losInd;
    
#ifdef SLS_DEBUG_
    // Early debug output to catch illegal memory access
    if (activeLinkIdx == 0) {
        printf("DEBUG: CIR Kernel started - activeLinkIdx=%u, cid=%u, uid=%u, lspReadIdx=%u\n",
               activeLinkIdx, cid, uid, lspReadIdx);
    }
#endif
    
    // Access CIR parameters directly from activeLinkParams
    Tcomplex* cirCoe = activeLink.cirCoe;
    uint16_t* cirNormDelay = activeLink.cirNormDelay;
    uint16_t* cirNtaps = activeLink.cirNtaps;
    
#ifdef SLS_DEBUG_
    // Check before accessing linkParams array
    if (activeLinkIdx == 0) {
        printf("DEBUG: About to access linkParams[%u]\n", lspReadIdx);
    }
#endif
    
    const LinkParams& link = linkParams[lspReadIdx];
    bool losInd = link.losInd;
    uint8_t nCluster = clusterParams[lspReadIdx].nCluster;
    uint16_t nRayPerCluster = clusterParams[lspReadIdx].nRayPerCluster;

    // Get antenna configurations and orientations
    uint32_t utAntPanelIdx = utParams[uid].antPanelIdx;
    uint32_t cellAntPanelIdx = cellParams[cid].antPanelIdx;
    AntPanelConfig utAntPanelConfig = antPanelConfigs[utAntPanelIdx];
    AntPanelConfig cellAntPanelConfig = antPanelConfigs[cellAntPanelIdx];
    uint32_t nUtAnt = utAntPanelConfig.nAnt;
    uint32_t nCellAnt = cellAntPanelConfig.nAnt;
    
    // Get antenna panel orientations (cell-specific!)
    const float* utAntPanelOrientation = utParams[uid].antPanelOrientation;
    const float* cellAntPanelOrientation = cellParams[cid].antPanelOrientation;
    
    // Convert UT velocity to polar coordinates
    float utVelocityPolar[3] = {
        utParams[uid].velocity[0],
        utParams[uid].velocity[1], 
        utParams[uid].velocity[2]
    };

    // Calculate time for this snapshot: slot duration = 1 ms / (scs / 15 kHz), split
    // evenly across the n_snapshot_per_slot intra-slot snapshots
    float timeOffset = 1e-3f * 15e3f / (simConfig->sc_spacing_hz * simConfig->n_snapshot_per_slot);
    float snapshotTime = refTime + snapshotIdx * timeOffset;
    
    // Add propagation delay if enabled
    if (sysConfig->enable_propagation_delay == 1) {
        snapshotTime += linkParams[lspReadIdx].d3d / 3.0e8f;  // d_3d / speed_of_light
    }
    // For unified layout: (total_active_links, n_snapshots, n_ut_ant, n_bs_ant, max_taps)
    // The cirCoe pointer from activeLink already points to the correct memory location
    // In external memory mode, the pointer is already positioned correctly for this link
    // In internal memory mode, we need to calculate the snapshot offset within the link
    size_t snapshotOffset = snapshotIdx * nUtAnt * nCellAnt * N_MAX_TAPS;
    
    // Check if small scale fading is disabled
    if (sysConfig->disable_small_scale_fading == 1) {
        // Small scale fading disabled: only apply path loss (fast fading = 1)
        // Each thread handles initialization of CIR elements
        int tid = threadIdx.x;
        int total_elements = nUtAnt * nCellAnt * N_MAX_TAPS;
        
        // Reset cirCoe to zero first (distributed across threads)
        for (int i = tid; i < total_elements; i += blockDim.x) {
            cirCoe[snapshotOffset + i] = make_cuComplex(0.0f, 0.0f);
        }
        
        __syncthreads();  // Ensure all elements are initialized
        
        // Apply path loss, shadowing, and antenna patterns deterministically (Phase-1)
        // For Phase-1: Apply all pattern- or array-related effects deterministically,
        // since only large-scale, non-random effects are considered
        if (sysConfig->disable_pl_shadowing != 1) {
            // The sign of the shadow fading is defined so that positive SF means more received power at UT than predicted by the path loss model
            const float pathGain = -(linkParams[lspReadIdx].pathloss - linkParams[lspReadIdx].SF);
            const float path_scale = powf(10.0f, pathGain / 20.0f);
            
            // Map the GCS LOS angles into each panel's LCS per eq 7.1-7/-8 and get the
            // eq 7.1-15 polarization rotation (cell-specific for different sectors!)
            // Orientation convention: [1] = bearing alpha, [0] = downtilt beta, [2] = roll gamma
            float theta_LOS_ZOD, phi_LOS_AOD, psi_tx;
            gcsToLcs(cellAntPanelOrientation[1], cellAntPanelOrientation[0], cellAntPanelOrientation[2],
                     linkParams[lspReadIdx].theta_LOS_ZOD, linkParams[lspReadIdx].phi_LOS_AOD,
                     theta_LOS_ZOD, phi_LOS_AOD, psi_tx);
            float theta_LOS_ZOA, phi_LOS_AOA, psi_rx;
            gcsToLcs(utAntPanelOrientation[1], utAntPanelOrientation[0] - 90.0f, utAntPanelOrientation[2],
                     linkParams[lspReadIdx].theta_LOS_ZOA, linkParams[lspReadIdx].phi_LOS_AOA,
                     theta_LOS_ZOA, phi_LOS_AOA, psi_rx);
            // Uncorrected GCS arrival angles for the Doppler term (same frame as UT velocity)
            const float theta_LOS_ZOA_gcs = linkParams[lspReadIdx].theta_LOS_ZOA;
            const float phi_LOS_AOA_gcs = linkParams[lspReadIdx].phi_LOS_AOA;
            const float d_3d = linkParams[lspReadIdx].d3d;

            // Set first tap with path loss + antenna patterns - only thread 0
            // The corrected angles above ensure different sectors get different antenna gains
            if (tid == 0) {
                for (uint16_t utAntIdx = 0; utAntIdx < nUtAnt; utAntIdx++) {
                    for (uint16_t bsAntIdx = 0; bsAntIdx < nCellAnt; bsAntIdx++) {
                        // Calculate deterministic antenna response using LOS component
                        // This includes: antenna field pattern + array response
                        const Tcomplex antennaResponse = calculateLOSCoefficientGPU<Tcomplex>(
                            utAntPanelConfig, utAntIdx, theta_LOS_ZOA, phi_LOS_AOA, psi_rx,
                            cellAntPanelConfig, bsAntIdx, theta_LOS_ZOD, phi_LOS_AOD, psi_tx,
                            theta_LOS_ZOA_gcs, phi_LOS_AOA_gcs,
                            snapshotTime, utVelocityPolar, cmnLinkParams->lambda_0, d_3d
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
            }
        } else {
            // Both path loss and small scale fading disabled: set unit channel (1+0j)
            if (tid == 0) {
                for (uint16_t utAntIdx = 0; utAntIdx < nUtAnt; utAntIdx++) {
                    for (uint16_t bsAntIdx = 0; bsAntIdx < nCellAnt; bsAntIdx++) {
                        cirCoe[snapshotOffset + (utAntIdx * nCellAnt + bsAntIdx) * N_MAX_TAPS + 0] = make_cuComplex(1.0f, 0.0f);
                    }
                }
            }
        }
        
        // Set delay and taps info for simplified channel (only thread 0 and first snapshot)
        if (tid == 0 && snapshotIdx == 0) {
            cirNormDelay[0] = 0;  // Single tap at delay 0
            for (uint16_t tap = 1; tap < N_MAX_TAPS; tap++) {
                cirNormDelay[tap] = 0;
            }
            cirNtaps[0] = 1;  // Only one tap
        }
        
        return;  // Skip the complex small scale fading calculations
    }
    
    // Robust BS-antenna parallelization via a strided loop (below): this decouples the
    // thread count from the antenna count, so nCellAnt may exceed blockDim.x (e.g. 128/256
    // massive-MIMO configs). Each thread owns ONE fixed shared-memory slice indexed by
    // threadIdx.x and reuses it across every BS antenna it processes, so total shared memory
    // is bounded by blockDim.x rather than by nCellAnt.
    Tcomplex* H_link_ue = H_link_ue_shared + threadIdx.x * nUtAnt * N_MAX_TAPS;

    // Pre-calculate K_R for LOS power adjustment
    float K_R = 0.0f;
    float losPower = 0.0f;
    if (losInd && !isO2I) {
        float K = linkParams[lspReadIdx].K;
        K_R = powf(10.0f, K / 10.0f);
        losPower = K_R / (K_R + 1.0f);
    }

    // ---- Dense tap table (per link; identical for every BS antenna and thread) ----
    // Each regular cluster occupies one dense slot; each of the two strongest clusters
    // occupies nSubCluster slots (Table 7.5-5 sub-clusters at delay offsets 0 / 1.28*c_DS /
    // 2.56*c_DS). Dense slot k holds raster tap H_tapIdx[k]; the combine step below
    // accumulates slot k into output tap denseToSparse[k], keeping cirCoe rows aligned with
    // the sorted-unique delays written to cirNormDelay.
    uint16_t H_tapIdx[N_MAX_TAPS];
    uint16_t tapCount = 0;
    for (uint16_t clusterIdx = 0; clusterIdx < nCluster && tapCount < N_MAX_TAPS; clusterIdx++) {
        bool isStrongestCluster = (clusterIdx == clusterParams[lspReadIdx].strongest2clustersIdx[0] ||
                                   clusterIdx == clusterParams[lspReadIdx].strongest2clustersIdx[1]);
        int nSub = isStrongestCluster ? cmnLinkParams->nSubCluster : 1;
        for (int subClusterIdx = 0; subClusterIdx < nSub && tapCount < N_MAX_TAPS; subClusterIdx++) {
            float clusterDelay = clusterParams[lspReadIdx].delays[clusterIdx];
            if (subClusterIdx == 1)      clusterDelay += 1.28f * cmnLinkParams->C_DS[lspIdx];
            else if (subClusterIdx == 2) clusterDelay += 2.56f * cmnLinkParams->C_DS[lspIdx];
            uint16_t tapIdx;
            if (sysConfig->enable_propagation_delay == 1) {
                // Add propagation delay (d3d/c) and excess delay delta_tau per 3GPP TR 38.901 Section 7.6.9
                // delta_tau is 0 for LOS, lognormal for NLOS (already computed and stored in linkParams)
                tapIdx = (uint16_t)roundf((clusterDelay * 1e-9f + linkParams[lspReadIdx].d3d / 3.0e8f + linkParams[lspReadIdx].delta_tau) * simConfig->sc_spacing_hz * simConfig->fft_size);
            } else {
                tapIdx = (uint16_t)roundf(clusterDelay * 1e-9f * simConfig->sc_spacing_hz * simConfig->fft_size);
            }
            H_tapIdx[tapCount++] = tapIdx;
        }
    }

    // Sorted unique output taps + dense->sparse map (cheap: tapCount <= N_MAX_TAPS = 24)
    uint16_t unique_taps[N_MAX_TAPS];
    uint16_t numUniqueTaps = 0;
    uint16_t denseToSparse[N_MAX_TAPS];
    for (uint16_t i = 0; i < tapCount; i++) {
        unique_taps[i] = H_tapIdx[i];
    }
    for (uint16_t i = 0; tapCount > 1 && i < tapCount - 1; i++) {
        for (uint16_t j = 0; j < tapCount - i - 1; j++) {
            if (unique_taps[j] > unique_taps[j + 1]) {
                uint16_t temp = unique_taps[j];
                unique_taps[j] = unique_taps[j + 1];
                unique_taps[j + 1] = temp;
            }
        }
    }
    if (tapCount > 0) {
        numUniqueTaps = 1;
        for (uint16_t i = 1; i < tapCount; i++) {
            if (unique_taps[i] != unique_taps[numUniqueTaps - 1]) {
                unique_taps[numUniqueTaps++] = unique_taps[i];
            }
        }
    }
    for (uint16_t i = 0; i < tapCount; i++) {
        denseToSparse[i] = 0;
        for (uint16_t j = 0; j < numUniqueTaps; j++) {
            if (H_tapIdx[i] == unique_taps[j]) { denseToSparse[i] = j; break; }
        }
    }

    // Strided loop over BS antennas: thread t processes antennas t, t+blockDim.x, ...
    // For nCellAnt <= blockDim.x this runs exactly once per thread (bsAntIdx == threadIdx.x),
    // i.e. identical to the one-thread-per-antenna scheme; for larger nCellAnt each thread
    // handles several antennas sequentially, reusing its shared slice. No __syncthreads() may
    // appear inside this loop (per-thread trip counts differ) — the only barrier is after it.
    for (uint16_t bsAntIdx = (uint16_t)threadIdx.x; bsAntIdx < nCellAnt; bsAntIdx += (uint16_t)blockDim.x) {
        // Initialize H_link_ue to zero for this BS antenna's thread
        for (int i = 0; i < N_MAX_TAPS * nUtAnt; i++) {
            H_link_ue[i] = make_cuComplex(0.0f, 0.0f);
        }
        uint16_t denseSlot = 0;  // walks the precomputed dense tap table in the same order

        // Process all clusters for this specific BS antenna
        for (uint16_t clusterIdx = 0; clusterIdx < nCluster; clusterIdx++) {
            // Check if this is one of the two strongest clusters for sub-clustering
            bool isStrongestCluster = (clusterIdx == clusterParams[lspReadIdx].strongest2clustersIdx[0] ||
                                    clusterIdx == clusterParams[lspReadIdx].strongest2clustersIdx[1]);

            if (isStrongestCluster) {
                // Two strongest clusters: Table 7.5-5 sub-clusters, one dense tap slot each.
                // Each ray keeps the plain eq 7.5-28 amplitude sqrt(P_n/M); the 10/20-6/20-4/20
                // sub-cluster power split arises from the ray-set sizes alone, so no extra
                // per-sub-cluster amplitude factor may be applied.
                for (int subClusterIdx = 0; subClusterIdx < cmnLinkParams->nSubCluster; subClusterIdx++) {
                    const uint16_t* rays = nullptr;
                    int nRays = 0;

                    // Use subcluster ray arrays from cmnLinkParams struct
                    if (subClusterIdx == 0) {
                        rays = cmnLinkParams->raysInSubCluster0;
                        nRays = cmnLinkParams->raysInSubClusterSizes[0];
                    } else if (subClusterIdx == 1) {
                        rays = cmnLinkParams->raysInSubCluster1;
                        nRays = cmnLinkParams->raysInSubClusterSizes[1];
                    } else if (subClusterIdx == 2) {
                        rays = cmnLinkParams->raysInSubCluster2;
                        nRays = cmnLinkParams->raysInSubClusterSizes[2];
                    }

                    // Process rays in this subcluster using defined ray indices
                    for (uint16_t rayIdx = 0; rayIdx < nRays; ++rayIdx) {
                        // Get ray parameters using specific ray indices from defined arrays
                        int rayGlobalIdx = clusterIdx * nRayPerCluster + rays[rayIdx];
                        float theta_ZOA_gcs = clusterParams[lspReadIdx].theta_n_m_ZOA[rayGlobalIdx];
                        float phi_AOA_gcs = clusterParams[lspReadIdx].phi_n_m_AoA[rayGlobalIdx];
                        // eq 7.1-7/-8/-15: map GCS ray angles into each panel's LCS
                        // (orientation: [1]=bearing alpha, [0]=downtilt beta, [2]=roll gamma)
                        float theta_ZOA, phi_AOA, psi_rx;
                        gcsToLcs(utAntPanelOrientation[1], utAntPanelOrientation[0] - 90.0f, utAntPanelOrientation[2],
                                 theta_ZOA_gcs, phi_AOA_gcs, theta_ZOA, phi_AOA, psi_rx);
                        float theta_ZOD, phi_AOD, psi_tx;
                        gcsToLcs(cellAntPanelOrientation[1], cellAntPanelOrientation[0], cellAntPanelOrientation[2],
                                 clusterParams[lspReadIdx].theta_n_m_ZOD[rayGlobalIdx],
                                 clusterParams[lspReadIdx].phi_n_m_AoD[rayGlobalIdx],
                                 theta_ZOD, phi_AOD, psi_tx);
                        float xpr = clusterParams[lspReadIdx].xpr[rayGlobalIdx];
                        const float* randomPhase = &clusterParams[lspReadIdx].randomPhases[((sysConfig->n_sector_per_site > 0 ? cid % sysConfig->n_sector_per_site : 0) % ClusterParams::MAX_SECTORS) * ClusterParams::PHASE_SECTOR_STRIDE + rayGlobalIdx * 4];

                        // Precompute Tx-side terms once per (ray, bsAnt), reuse across utAnts
                        Tcomplex tx_pol[2];
                        computeTxRayTermsGPU<Tcomplex>(
                            cellAntPanelConfig, bsAntIdx, theta_ZOD, phi_AOD,
                            psi_tx, xpr, randomPhase, tx_pol);

                        float clusterPower = clusterParams[lspReadIdx].powers[clusterIdx];
                        if (losInd && !isO2I && clusterIdx == 0) {
                            clusterPower -= losPower;
                            clusterPower = fmaxf(clusterPower, 0.0f);
                        }
                        float power = sqrtf(clusterPower / nRayPerCluster);

                        for (uint16_t utAntIdx = 0; utAntIdx < nUtAnt; utAntIdx++) {
                            Tcomplex rayCoeff = computeRxRayContribGPU<Tcomplex>(
                                utAntPanelConfig, utAntIdx, theta_ZOA, phi_AOA,
                                theta_ZOA_gcs, phi_AOA_gcs,
                                psi_rx, snapshotTime, utVelocityPolar,
                                cmnLinkParams->lambda_0, tx_pol);
                            int hlinkIndex = utAntIdx * N_MAX_TAPS + denseSlot;
                            H_link_ue[hlinkIndex] = cuCaddf(H_link_ue[hlinkIndex],
                                make_cuComplex(rayCoeff.x * power, rayCoeff.y * power));
                        }
                    }
                    denseSlot++;  // one dense tap slot per sub-cluster
                }
            }
            else {
                // Process regular cluster (one dense tap slot)
                for (uint16_t rayIdx = 0; rayIdx < nRayPerCluster; rayIdx++) {
                    int rayGlobalIdx = clusterIdx * nRayPerCluster + rayIdx;
                    float theta_ZOA_gcs = clusterParams[lspReadIdx].theta_n_m_ZOA[rayGlobalIdx];
                    float phi_AOA_gcs = clusterParams[lspReadIdx].phi_n_m_AoA[rayGlobalIdx];
                    // eq 7.1-7/-8/-15: map GCS ray angles into each panel's LCS
                    // (orientation: [1]=bearing alpha, [0]=downtilt beta, [2]=roll gamma)
                    float theta_ZOA, phi_AOA, psi_rx;
                    gcsToLcs(utAntPanelOrientation[1], utAntPanelOrientation[0] - 90.0f, utAntPanelOrientation[2],
                             theta_ZOA_gcs, phi_AOA_gcs, theta_ZOA, phi_AOA, psi_rx);
                    float theta_ZOD, phi_AOD, psi_tx;
                    gcsToLcs(cellAntPanelOrientation[1], cellAntPanelOrientation[0], cellAntPanelOrientation[2],
                             clusterParams[lspReadIdx].theta_n_m_ZOD[rayGlobalIdx],
                             clusterParams[lspReadIdx].phi_n_m_AoD[rayGlobalIdx],
                             theta_ZOD, phi_AOD, psi_tx);
                    float xpr = clusterParams[lspReadIdx].xpr[rayGlobalIdx];
                    const float* randomPhase = &clusterParams[lspReadIdx].randomPhases[((sysConfig->n_sector_per_site > 0 ? cid % sysConfig->n_sector_per_site : 0) % ClusterParams::MAX_SECTORS) * ClusterParams::PHASE_SECTOR_STRIDE + rayGlobalIdx * 4];

                    // Precompute Tx-side terms once per (ray, bsAnt)
                    Tcomplex tx_pol[2];
                    computeTxRayTermsGPU<Tcomplex>(
                        cellAntPanelConfig, bsAntIdx, theta_ZOD, phi_AOD,
                        psi_tx, xpr, randomPhase, tx_pol);

                    float clusterPower = clusterParams[lspReadIdx].powers[clusterIdx];
                    if (losInd && !isO2I && clusterIdx == 0) {
                        clusterPower -= losPower;
                        clusterPower = fmaxf(clusterPower, 0.0f);
                    }
                    float power = sqrtf(clusterPower / nRayPerCluster);

                    for (uint16_t utAntIdx = 0; utAntIdx < nUtAnt; utAntIdx++) {
                        Tcomplex rayCoeff = computeRxRayContribGPU<Tcomplex>(
                            utAntPanelConfig, utAntIdx, theta_ZOA, phi_AOA,
                            theta_ZOA_gcs, phi_AOA_gcs,
                            psi_rx, snapshotTime, utVelocityPolar,
                            cmnLinkParams->lambda_0, tx_pol);
                        int hlinkIndex = utAntIdx * N_MAX_TAPS + denseSlot;
                        H_link_ue[hlinkIndex] = cuCaddf(H_link_ue[hlinkIndex],
                            make_cuComplex(rayCoeff.x * power, rayCoeff.y * power));
                    }
                }
                denseSlot++;
            }
        }

        // Handle LOS case if present (like CPU reference)
        if (losInd && !isO2I) {
            float K = linkParams[lspReadIdx].K;
            float K_R = powf(10.0f, K / 10.0f);
            float los_scale = sqrtf(K_R / (K_R + 1.0f));
            // NLOS already been scaled by subtracting K/(K+1) from first cluster power
            
            // eq 7.1-7/-8/-15: map GCS LOS angles into each panel's LCS
            // (orientation: [1]=bearing alpha, [0]=downtilt beta, [2]=roll gamma)
            float theta_LOS_ZOA, phi_LOS_AOA, psi_rx_los;
            gcsToLcs(utAntPanelOrientation[1], utAntPanelOrientation[0] - 90.0f, utAntPanelOrientation[2],
                     linkParams[lspReadIdx].theta_LOS_ZOA, linkParams[lspReadIdx].phi_LOS_AOA,
                     theta_LOS_ZOA, phi_LOS_AOA, psi_rx_los);
            float theta_LOS_ZOD, phi_LOS_AOD, psi_tx_los;
            gcsToLcs(cellAntPanelOrientation[1], cellAntPanelOrientation[0], cellAntPanelOrientation[2],
                     linkParams[lspReadIdx].theta_LOS_ZOD, linkParams[lspReadIdx].phi_LOS_AOD,
                     theta_LOS_ZOD, phi_LOS_AOD, psi_tx_los);
            // Uncorrected GCS arrival angles for the Doppler term (same frame as UT velocity)
            float theta_LOS_ZOA_gcs = linkParams[lspReadIdx].theta_LOS_ZOA;
            float phi_LOS_AOA_gcs = linkParams[lspReadIdx].phi_LOS_AOA;
            float d_3d = linkParams[lspReadIdx].d3d;

            // Calculate LOS component for each antenna pair
            for (uint16_t utAntIdx = 0; utAntIdx < nUtAnt; utAntIdx++) {
                // Calculate LOS coefficient with proper antenna orientations
                Tcomplex H_LOS = calculateLOSCoefficientGPU<Tcomplex>(
                    utAntPanelConfig, utAntIdx, theta_LOS_ZOA, phi_LOS_AOA, psi_rx_los,
                    cellAntPanelConfig, bsAntIdx, theta_LOS_ZOD, phi_LOS_AOD, psi_tx_los,
                    theta_LOS_ZOA_gcs, phi_LOS_AOA_gcs,
                    snapshotTime, utVelocityPolar, cmnLinkParams->lambda_0, d_3d
                );
                
                // Combine LOS and NLOS components at first tap (like CPU reference)
                int tap0Idx = utAntIdx * N_MAX_TAPS + 0;  // only UE antenna dimension
                Tcomplex nlos_component = H_link_ue[tap0Idx];
                
#ifdef SLS_DEBUG_
                // NaN detection for LOS and NLOS components
                if (isnan(H_LOS.x) || isnan(H_LOS.y) || 
                    isinf(H_LOS.x) || isinf(H_LOS.y)) {
                    printf("ERROR: Invalid LOS component - Real: %f, Imag: %f (activeLinkIdx=%u, utAnt=%u, bsAnt=%u)\n",
                           H_LOS.x, H_LOS.y, activeLinkIdx, utAntIdx, bsAntIdx);
                    printf("  K_R: %f, d_3d: %f\n", K_R, d_3d);
                }
                if (isnan(nlos_component.x) || isnan(nlos_component.y) || 
                    isinf(nlos_component.x) || isinf(nlos_component.y)) {
                    printf("ERROR: Invalid NLOS component - Real: %f, Imag: %f (activeLinkIdx=%u, utAnt=%u, bsAnt=%u)\n",
                           nlos_component.x, nlos_component.y, activeLinkIdx, utAntIdx, bsAntIdx);
                }
#endif
                
                // Combine LOS and NLOS (NLOS already has correct power from cluster processing)
                Tcomplex combined = make_cuComplex(
                    los_scale * H_LOS.x + nlos_component.x,
                    los_scale * H_LOS.y + nlos_component.y
                );
                
#ifdef SLS_DEBUG_
                // NaN detection for final combined coefficient
                if (isnan(combined.x) || isnan(combined.y) || 
                    isinf(combined.x) || isinf(combined.y)) {
                    printf("ERROR: Invalid combined LOS+NLOS coefficient - Real: %f, Imag: %f (activeLinkIdx=%u, utAnt=%u, bsAnt=%u)\n",
                           combined.x, combined.y, activeLinkIdx, utAntIdx, bsAntIdx);
                    printf("  K_R: %f, los_scale: %f\n", K_R, los_scale);
                }
#endif
                H_link_ue[tap0Idx] = combined;
            }
        }

        // Apply large scale fading (like CPU reference) if not disabled
        if (sysConfig->disable_pl_shadowing != 1) {
            // The sign of the shadow fading is defined so that positive SF means more received power at UT than predicted by the path loss model
            float pathGain = -(linkParams[lspReadIdx].pathloss - linkParams[lspReadIdx].SF);
            float path_scale = powf(10.0f, pathGain / 20.0f);
            
            // Apply path scaling to current BS antenna's coefficients
            for (uint16_t utAntIdx = 0; utAntIdx < nUtAnt; utAntIdx++) {
                for (uint16_t tap = 0; tap < N_MAX_TAPS; tap++) {
                    int hlinkIndex = utAntIdx * N_MAX_TAPS + tap;
                    H_link_ue[hlinkIndex] = make_cuComplex(
                        H_link_ue[hlinkIndex].x * path_scale,
                        H_link_ue[hlinkIndex].y * path_scale);
                }
            }
        }

        // Step 3: Initialize cirCoe to zero and combine H_link coefficients (like CPU reference)
        // Access cirCoe directly from activeLinkParams - it already points to the correct memory location
        Tcomplex* cirCoeBase = &cirCoe[snapshotOffset];
        
        // Initialize cirCoe to zero
        for (uint16_t utAntIdx = 0; utAntIdx < nUtAnt; utAntIdx++) {
            for (uint16_t tap = 0; tap < N_MAX_TAPS; tap++) {
                cirCoeBase[(utAntIdx * nCellAnt + bsAntIdx) * N_MAX_TAPS + tap] = make_cuComplex(0.0f, 0.0f);
            }
        }
    
        // Step 4: Combine H_link dense slots into cirCoe at their sparse (sorted-unique) tap
        // positions so cirCoe rows stay aligned with cirNormDelay; coefficients whose raster
        // taps collide accumulate into the same sparse slot
        for (uint16_t denseIdx = 0; denseIdx < tapCount; denseIdx++) {
            uint16_t sparseIdx = denseToSparse[denseIdx];
            for (uint16_t utAntIdx = 0; utAntIdx < nUtAnt; utAntIdx++) {
                int hlinkSrcIdx = utAntIdx * N_MAX_TAPS + denseIdx;  // Source from H_link_ue
                int cirCoeDstIdx = (utAntIdx * nCellAnt + bsAntIdx) * N_MAX_TAPS + sparseIdx;  // Destination in full layout
                cirCoeBase[cirCoeDstIdx] = cuCaddf(cirCoeBase[cirCoeDstIdx], H_link_ue[hlinkSrcIdx]);
            }
        }
    } // End of strided BS-antenna loop (bounded shared memory; supports nCellAnt > blockDim.x)

    __syncthreads();

    // Write the sparse tap metadata (per-link, identical across threads; delays are
    // per-link, not per-snapshot, so only thread 0 of snapshot 0 writes)
    if (threadIdx.x == 0 && snapshotIdx == 0) {
        for (uint16_t i = 0; i < numUniqueTaps; i++) {
            cirNormDelay[i] = unique_taps[i];
        }
        for (uint16_t i = numUniqueTaps; i < N_MAX_TAPS; i++) {
            cirNormDelay[i] = 0;
        }
        cirNtaps[0] = numUniqueTaps;
    }
}

// Host function to launch the GPU kernels
template <typename Tscalar, typename Tcomplex>
void slsChan<Tscalar, Tcomplex>::calClusterRayGPU()
{
#ifdef SLS_DEBUG_
    // Debug host-side link parameters before GPU transfer
    printf("DEBUG: Host-side link parameters before GPU transfer:\n");
    for (size_t i = 0; i < std::min((size_t)6, m_linkParams.size()); i++) {
        printf("  Link %zu: DS=%.6e, losInd=%d, K=%.3f\n", 
               i, m_linkParams[i].DS, m_linkParams[i].losInd, m_linkParams[i].K);
    }
#endif
    // sync stream
    CHECK_CURESULT(cuStreamSynchronize(m_strm));
    
    // Launch kernel with adjusted dimensions (reduced max to avoid "too many resources" error)
    const uint32_t maxThreadsPerBlock = 256;  // Reduced from 1024 to avoid GPU resource limits
    const uint32_t threadsPerBlock = std::min(maxThreadsPerBlock, m_topology.nUT);
    const uint32_t numBlocks = (m_topology.nUT + threadsPerBlock - 1) / threadsPerBlock;
    
    dim3 blockDim(threadsPerBlock);
    dim3 gridDim(m_topology.nSite, numBlocks);

    void* kernelArgs[] = {
        &m_d_cellParams, &m_d_utParams, &m_d_linkParams, &m_d_cmnLinkParams,
        &m_d_clusterParams, &m_topology.nSite, &m_topology.nUT,
        &m_topology.n_sector_per_site, &m_d_curandStates};
    CUresult err = cuLaunchKernel(
        m_calClusterRayFunction,
        gridDim.x, gridDim.y, gridDim.z,
        blockDim.x, blockDim.y, blockDim.z,
        0, m_strm, kernelArgs, nullptr);

    if (err != CUDA_SUCCESS) {
        const char* errorString = nullptr;
        cuGetErrorString(err, &errorString);
        printf("ERROR: Cluster ray generation failed: %s\n", errorString);
        return;
    }

    // Copy results back to host
    CHECK_CURESULT(cuMemcpyDtoHAsync(
        m_clusterParams.data(), reinterpret_cast<CUdeviceptr>(m_d_clusterParams),
        m_clusterParams.size() * sizeof(ClusterParams), m_strm));
    CHECK_CURESULT(cuStreamSynchronize(m_strm));
}

template <typename Tscalar, typename Tcomplex>
void slsChan<Tscalar, Tcomplex>::generateCIRGPU()
{
    // Debug: Check activeLink pointers before copying to GPU
#ifdef SLS_DEBUG_
    printf("DEBUG: Checking activeLink pointers before GPU copy:\n");
    printf("  Total active links: %zu\n", m_activeLinkParams.size());
    for (size_t i = 0; i < std::min((size_t)6, m_activeLinkParams.size()); i++) {
        printf("  Link %zu: cirCoe=%p, cirNormDelay=%p, cirNtaps=%p\n", 
               i, m_activeLinkParams[i].cirCoe, m_activeLinkParams[i].cirNormDelay, m_activeLinkParams[i].cirNtaps);
        printf("    linkIdx=%u, cid=%u, uid=%u, lspReadIdx=%u\n",
               m_activeLinkParams[i].linkIdx, m_activeLinkParams[i].cid, m_activeLinkParams[i].uid, m_activeLinkParams[i].lspReadIdx);
    }
    if (m_simConfig->internal_memory_mode == 0) {
        printf("  Memory mode: External\n");
    } else if (m_simConfig->internal_memory_mode == 1){
        printf("  Memory mode: Internal memory for CIR and external memory for CFR\n");
    } else if (m_simConfig->internal_memory_mode == 2){
        printf("  Memory mode: External memory for CIR/CFR\n");
    } else {
        printf("  Memory mode: Invalid\n");
    }
    printf("  Per-cell allocation with m_d_activeLinkParams=%p\n", m_d_activeLinkParams);
#endif

    // CRITICAL: Ensure activeLinkParams copy to GPU completes before kernel launch
    CHECK_CURESULT(cuStreamSynchronize(m_strm));

    // Reset CIR buffers to zero before generation to prevent stale data corruption
    if (m_simConfig->internal_memory_mode >= 1 && m_cirCoe != nullptr) {
        size_t cirCoeSize = m_lastAllocatedLinks * 
                           m_simConfig->n_snapshot_per_slot *
                           m_cmnLinkParams.nUeAnt * 
                           m_cmnLinkParams.nBsAnt * 
                           N_MAX_TAPS * sizeof(Tcomplex);
        CHECK_CURESULT(cuMemsetD8Async(
            reinterpret_cast<CUdeviceptr>(m_cirCoe), 0, cirCoeSize, m_strm));
        
        size_t cirNormDelaySize = m_lastAllocatedLinks * N_MAX_TAPS * sizeof(uint16_t);
        CHECK_CURESULT(cuMemsetD8Async(
            reinterpret_cast<CUdeviceptr>(m_cirNormDelay), 0, cirNormDelaySize, m_strm));
        
        size_t cirNtapsSize = m_lastAllocatedLinks * sizeof(uint16_t);
        CHECK_CURESULT(cuMemsetD8Async(
            reinterpret_cast<CUdeviceptr>(m_cirNtaps), 0, cirNtapsSize, m_strm));
        
#ifdef SLS_DEBUG_
        printf("DEBUG: Reset CIR buffers - cirCoe: %zu bytes, delays: %zu bytes, ntaps: %zu bytes\n",
               cirCoeSize, cirNormDelaySize, cirNtapsSize);
#endif
        
        // Ensure buffer reset completes before kernel launch
        CHECK_CURESULT(cuStreamSynchronize(m_strm));
    }

    uint32_t numActiveLinks = static_cast<uint32_t>(m_activeLinkParams.size());
    if (numActiveLinks == 0) {
        return;
    }

    // Launch kernel: parallelize the BS-antenna loop across threads, capped at
    // CIR_MAX_THREADS. Antennas beyond the cap are covered by the kernel's strided loop,
    // so nBsAnt > CIR_MAX_THREADS (massive MIMO) still launches and shared-memory usage
    // stays bounded by the cap rather than scaling with nBsAnt.
    // NOTE: CIR_MAX_THREADS must not exceed the __launch_bounds__(256, ...) on generateCIRKernel.
    constexpr uint32_t CIR_MAX_THREADS = 256;
    uint32_t threadsPerBlock = m_cmnLinkParams.nBsAnt;
    if (threadsPerBlock > CIR_MAX_THREADS) threadsPerBlock = CIR_MAX_THREADS;
    if (threadsPerBlock == 0) threadsPerBlock = 1;
    // Shared memory: each thread owns one H_link_ue slice = nUtAnt * N_MAX_TAPS * sizeof(Tcomplex)
    const size_t perThreadSharedBytes = (size_t)m_cmnLinkParams.nUeAnt * N_MAX_TAPS * sizeof(Tcomplex);

    // Cap the block so the launch fits the achieved shared-memory budget; the kernel's
    // strided BS-antenna loop covers the antennas a smaller block no longer owns 1:1.
    const uint32_t maxThreadsBySharedMem =
        static_cast<uint32_t>(static_cast<size_t>(m_cirSharedBudget) / perThreadSharedBytes);
    if (maxThreadsBySharedMem == 0) {
        fprintf(stderr,
                "ERROR: generateCIRKernel H_link_ue slice (%zu bytes/thread, nUeAnt=%u) exceeds "
                "the dynamic shared-memory budget (%d bytes); not launching\n",
                perThreadSharedBytes, m_cmnLinkParams.nUeAnt, m_cirSharedBudget);
        throw std::runtime_error(
            "generateCIRKernel per-thread shared-memory requirement exceeds the device budget");
    } else if (threadsPerBlock > maxThreadsBySharedMem) {
        threadsPerBlock = maxThreadsBySharedMem;
    }
    const size_t sharedMemBytes = (size_t)threadsPerBlock * perThreadSharedBytes;

    dim3 blockDim(threadsPerBlock);
    dim3 gridDim(numActiveLinks, m_simConfig->n_snapshot_per_slot);

    void* kernelArgs[] = {
        &m_d_activeLinkParams, &m_d_cellParams, &m_d_utParams, &m_d_linkParams,
        &m_d_clusterParams, &m_d_cmnLinkParams, &m_d_simConfig, &m_d_antPanelConfigs,
        &m_d_sysConfig, &numActiveLinks, &m_refTime, &m_d_curandStates};
    const CUresult kernelError = cuLaunchKernel(
        m_generateCirFunction,
        gridDim.x, gridDim.y, gridDim.z,
        blockDim.x, blockDim.y, blockDim.z,
        sharedMemBytes, m_strm, kernelArgs, nullptr);
    if (kernelError != CUDA_SUCCESS) {
        const char* errorString = nullptr;
        cuGetErrorString(kernelError, &errorString);
        throw std::runtime_error(
            std::string("generateCIRKernel launch failed: ") +
            (errorString != nullptr ? errorString : "unknown CUDA driver error"));
    }

#ifdef SLS_DEBUG_
    // Wait for kernel completion and check for runtime errors
    CUresult syncError = cuCtxSynchronize();
    if (syncError != CUDA_SUCCESS) {
        const char* errorString = nullptr;
        cuGetErrorString(syncError, &errorString);
        printf("ERROR: generateCIRKernel execution failed: %s\n", errorString);
    } else {
        printf("DEBUG: generateCIRKernel completed successfully\n");
        printf("DEBUG: Grid dim: (%d, %d), Block dim: (%d, %d, %d)\n", 
               gridDim.x, gridDim.y, blockDim.x, blockDim.y, blockDim.z);
        printf("DEBUG: Active links: %zu, snapshots: %d\n", 
               m_activeLinkParams.size(), m_simConfig->n_snapshot_per_slot);
    }
    
    // Host-side NaN detection for generated CIR coefficients
    if (m_simConfig->internal_memory_mode >= 1 && m_activeLinkParams.size() > 0) {
        const int sampleSize = std::min(16, (int)(m_cmnLinkParams.nUeAnt * m_cmnLinkParams.nBsAnt * N_MAX_TAPS));
        Tcomplex* hostSample = new Tcomplex[sampleSize];
        
        if (m_activeLinkParams[0].cirCoe != nullptr) {
            CHECK_CURESULT(cuMemcpyDtoHAsync(
                hostSample, reinterpret_cast<CUdeviceptr>(m_activeLinkParams[0].cirCoe),
                sampleSize * sizeof(Tcomplex), m_strm));
            CHECK_CURESULT(cuStreamSynchronize(m_strm));
            
            int nanCount = 0;
            int infCount = 0;
            for (int i = 0; i < sampleSize; i++) {
                if (isnan(hostSample[i].x) || isnan(hostSample[i].y)) {
                    nanCount++;
                    printf("WARNING: NaN detected in CIR coefficient [%d]: (%.4e, %.4e)\n", 
                           i, hostSample[i].x, hostSample[i].y);
                }
                if (isinf(hostSample[i].x) || isinf(hostSample[i].y)) {
                    infCount++;
                    printf("WARNING: Infinity detected in CIR coefficient [%d]: (%.4e, %.4e)\n", 
                           i, hostSample[i].x, hostSample[i].y);
                }
            }
            
            if (nanCount > 0 || infCount > 0) {
                printf("ERROR: Invalid CIR coefficients detected - NaN: %d, Inf: %d out of %d samples\n", 
                       nanCount, infCount, sampleSize);
            } else {
                printf("DEBUG: CIR coefficient validation passed - all %d samples are valid\n", sampleSize);
            }
        }
        
        delete[] hostSample;
    }
#endif

#ifdef SLS_DEBUG_
    // Copy sample m_cirCoe data from GPU to host for debugging
    if (m_activeLinkParams.size() > 0 && m_simConfig->n_snapshot_per_slot > 0) {
        // Debug: Print memory layout information
        printf("DEBUG: Memory layout info:\n");
        printf("  nUeAnt: %d, nBsAnt: %d, N_MAX_TAPS: %d\n", 
               m_cmnLinkParams.nUeAnt, m_cmnLinkParams.nBsAnt, N_MAX_TAPS);
        printf("  n_snapshot_per_slot: %d\n", m_simConfig->n_snapshot_per_slot);
        
        // Calculate the correct offset for first active link, first snapshot
        uint32_t nUtAnt = m_cmnLinkParams.nUeAnt;
        uint32_t nCellAnt = m_cmnLinkParams.nBsAnt;
        uint32_t offsetPerLink = m_simConfig->n_snapshot_per_slot * nUtAnt * nCellAnt * N_MAX_TAPS;
        uint32_t offsetPerSnapshot = nUtAnt * nCellAnt * N_MAX_TAPS;
        
        printf("  Offset per link: %d, offset per snapshot: %d\n", offsetPerLink, offsetPerSnapshot);
        
        // Copy first few CIR coefficients from correct location
        const int sampleSize = std::min(8, (int)(nUtAnt * nCellAnt * N_MAX_TAPS));
        Tcomplex* hostSample = new Tcomplex[sampleSize];
        
        // Copy from the beginning of first link, first snapshot
        if (m_activeLinkParams[0].cirCoe != nullptr) {
            CHECK_CURESULT(cuMemcpyDtoHAsync(
                hostSample, reinterpret_cast<CUdeviceptr>(m_activeLinkParams[0].cirCoe),
                sampleSize * sizeof(Tcomplex), m_strm));
            CHECK_CURESULT(cuStreamSynchronize(m_strm));
        } else {
            memset(hostSample, 0, sampleSize * sizeof(Tcomplex));
        }
        
        printf("DEBUG: Sample m_cirCoe values (first %d elements, link 0, snapshot 0):\n", sampleSize);
        for (int i = 0; i < sampleSize; i++) {
            printf("  Element %d: (%.4e, %.4e)\n", i, hostSample[i].x, hostSample[i].y);
        }
        
        // Debug print n_taps and normalized delay values
        if (!m_activeLinkParams.empty()) {
            // Additional sync to ensure all kernel writes are visible
            CHECK_CURESULT(cuCtxSynchronize());
            
            const int maxLinksToShow = std::min(6, (int)m_activeLinkParams.size());
            const int maxTapsToShow = 8;
            
            printf("DEBUG: n_taps and normalized delay values:\n");
            printf("DEBUG: Per-cell allocation, total_active_links=%zu, showing_first=%d\n", 
                   m_activeLinkParams.size(), maxLinksToShow);
            for (size_t idx = 0; idx < m_activeLinkParams.size(); idx++) {
                printf("DEBUG: activeLink[%zu] cirNtaps ptr=%p, cirNormDelay ptr=%p\n", 
                       idx, (void*)m_activeLinkParams[idx].cirNtaps, (void*)m_activeLinkParams[idx].cirNormDelay);
            }
            
            // Read n_taps directly from activeLink pointers
            uint16_t* hostNtaps = new uint16_t[m_activeLinkParams.size()];
            uint16_t* hostNormDelay = new uint16_t[m_activeLinkParams.size() * N_MAX_TAPS];
            
            for (size_t idx = 0; idx < m_activeLinkParams.size(); idx++) {
                if (m_activeLinkParams[idx].cirNtaps) {
                    CHECK_CURESULT(cuMemcpyDtoHAsync(
                        &hostNtaps[idx],
                        reinterpret_cast<CUdeviceptr>(m_activeLinkParams[idx].cirNtaps),
                        sizeof(uint16_t), m_strm));
                    CHECK_CURESULT(cuStreamSynchronize(m_strm));
                } else {
                    hostNtaps[idx] = 0;
                }
                
                if (m_activeLinkParams[idx].cirNormDelay) {
                    CHECK_CURESULT(cuMemcpyDtoHAsync(
                        &hostNormDelay[idx * N_MAX_TAPS],
                        reinterpret_cast<CUdeviceptr>(m_activeLinkParams[idx].cirNormDelay),
                        N_MAX_TAPS * sizeof(uint16_t), m_strm));
                    CHECK_CURESULT(cuStreamSynchronize(m_strm));
                } else {
                    memset(&hostNormDelay[idx * N_MAX_TAPS], 0, N_MAX_TAPS * sizeof(uint16_t));
                }
            }
            for (int linkIdx = 0; linkIdx < maxLinksToShow; linkIdx++) {
                for (int snapIdx = 0; snapIdx < m_simConfig->n_snapshot_per_slot; snapIdx++) {
                    // Note: Both n_taps and delays are per-link, same for all snapshots
                    uint16_t ntaps = hostNtaps[linkIdx];
                    
                    printf("  Link %d, Snapshot %d: n_taps = %d\n", linkIdx, snapIdx, ntaps);
                    
                    if (ntaps > 0) {
                        int delayBaseIdx = linkIdx * N_MAX_TAPS;
                        int tapsToShow = std::min((int)ntaps, maxTapsToShow);
                        printf("    Normalized delays: [");
                        for (int tapIdx = 0; tapIdx < tapsToShow; tapIdx++) {
                            printf("%d", hostNormDelay[delayBaseIdx + tapIdx]);
                            if (tapIdx < tapsToShow - 1) printf(", ");
                        }
                        if (ntaps > maxTapsToShow) {
                            printf(", ... (%d more)", ntaps - maxTapsToShow);
                        }
                        printf("]\n");
                    } else {
                        printf("    No taps for this link/snapshot\n");
                    }
                }
            }
            
            delete[] hostNtaps;
            delete[] hostNormDelay;
        } else {
            printf("DEBUG: m_cirNtaps or m_cirNormDelay is null, cannot print debug info\n");
        }
        
        delete[] hostSample;
    }
#endif

    // No need to free temporary memory - using pre-allocated member variables
}

    // GPU kernel to generate CFR - Mode 1 (PRBG level only)
template <typename Tcomplex>
__global__ void generateCFRKernel_runMode1(const activeLink<Tcomplex>* activeLinkParams,
                                          const CellParam* cellParams, const UtParam* utParams,
                                          const LinkParams* linkParams, const ClusterParams* clusterParams,
                                          const CmnLinkParams* cmnLinkParams, const SimConfig* simConfig,
                                          const AntPanelConfig* antPanelConfigs, const SystemLevelConfig* sysConfig,
                                          uint32_t nActiveLinks, float refTime, float cfrNormalizationFactor, curandState* curandStates) {
    // GRID(nActiveLinks, m_scaleUeAntFreqChan * m_nBatch, m_scaleBsAntFreqChan)
    // BLOCK(N_Prbg, nBsAnt/m_scaleBsAntFreqChan, nUeAnt/m_scaleUeAntFreqChan)
    
    uint32_t activeLinkIdx = blockIdx.x;
    if (activeLinkIdx >= nActiveLinks) return;
    
    const activeLink<Tcomplex>& activeLink = activeLinkParams[activeLinkIdx];
    uint16_t N_sc = simConfig->fft_size;
    uint16_t N_Prbg = simConfig->n_prbg;
    uint16_t N_sc_Prbg = (uint16_t)ceilf((float)(simConfig->n_prb * 12) / simConfig->n_prbg);
    uint16_t N_sc_over_2 = N_sc >> 1;
    uint8_t freqConvertType = simConfig->freq_convert_type;
    uint8_t scSampling = simConfig->sc_sampling;
    bool optionalCfrDim = (simConfig->optional_cfr_dim == 1);
    
    // Get antenna configurations from actual UT and Cell parameters  
    uint16_t cid = activeLink.cid;
    uint16_t uid = activeLink.uid;
    uint32_t utAntPanelIdx = utParams[uid].antPanelIdx;
    uint32_t cellAntPanelIdx = cellParams[cid].antPanelIdx;
    AntPanelConfig utAntPanelConfig = antPanelConfigs[utAntPanelIdx];
    AntPanelConfig cellAntPanelConfig = antPanelConfigs[cellAntPanelIdx];
    uint32_t nUtAnt = utAntPanelConfig.nAnt;
    uint32_t nCellAnt = cellAntPanelConfig.nAnt;
    
    // Thread and block indices
    uint16_t batchIdx = blockIdx.y;
    
    // Shared memory for CIR
    extern __shared__ char shareData[];
    Tcomplex* s_timeChanLocal = reinterpret_cast<Tcomplex*>(shareData);
    __shared__ float s_cirNormDelayUs2Pi[N_MAX_TAPS];
    
    // Read CIR normalization delays (only once per block)
    uint16_t cirNtaps = activeLink.cirNtaps[0];  // Note: cirNtaps is per-link, not per-snapshot
    if (threadIdx.x == 0) {
        for (uint16_t copyIdx = 0; copyIdx < cirNtaps; copyIdx++) {
            // Convert normalized delay to 2*pi*delay in microseconds
            // Note: cirNormDelay is per-link, not per-snapshot
            float delayUs = activeLink.cirNormDelay[copyIdx] * 1e6f / 
                           (simConfig->sc_spacing_hz * simConfig->fft_size);
            s_cirNormDelayUs2Pi[copyIdx] = 2.0f * M_PI * delayUs;
        }
    }
    __syncthreads();
    
    // Iterate over all antenna combinations
    for (uint16_t ueAntIdx = 0; ueAntIdx < nUtAnt; ueAntIdx++) {
        for (uint16_t bsAntIdx = 0; bsAntIdx < nCellAnt; bsAntIdx++) {
            // Calculate CIR offset for this antenna combination within this link's data
            // activeLink.cirCoe already points to this link's data, so just need antenna + snapshot offset
            size_t cirOffset = batchIdx * nUtAnt * nCellAnt * N_MAX_TAPS + (ueAntIdx * nCellAnt + bsAntIdx) * N_MAX_TAPS;
            
            // Copy CIR coefficients to shared memory with CFO rotation
            for (uint32_t copyIdx = threadIdx.x; copyIdx < cirNtaps; copyIdx += blockDim.x) {
#ifdef SLS_DEBUG_
                // Bounds check to prevent out-of-bounds access
                if (copyIdx >= N_MAX_TAPS) {
                    printf("ERROR: copyIdx %u >= N_MAX_TAPS %d for cirNtaps %u\n", 
                           copyIdx, N_MAX_TAPS, cirNtaps);
                    break;
                }
#endif
                
                // Placeholder CFO rotation - to be implemented later
                Tcomplex cfrRotationTotal = {1.0f, 0.0f}; // Identity rotation for now
                
                // activeLink.cirCoe already points to this link's data
                Tcomplex tmpCopyCir = activeLink.cirCoe[cirOffset + copyIdx];
                
#ifdef SLS_DEBUG_
                // Additional safety check for the retrieved coefficient
                if (isnan(tmpCopyCir.x) || isnan(tmpCopyCir.y) || isinf(tmpCopyCir.x) || isinf(tmpCopyCir.y)) {
                    printf("ERROR: Invalid CIR coefficient at copyIdx=%u, cirOffset=%zu: (%f, %f)\n", 
                           copyIdx, cirOffset, tmpCopyCir.x, tmpCopyCir.y);
                    tmpCopyCir = make_cuComplex(0.0f, 0.0f); // Use zero instead of NaN
                }
#endif
                
                s_timeChanLocal[copyIdx].x = tmpCopyCir.x * cfrRotationTotal.x - tmpCopyCir.y * cfrRotationTotal.y;
                s_timeChanLocal[copyIdx].y = tmpCopyCir.x * cfrRotationTotal.y + tmpCopyCir.y * cfrRotationTotal.x;
            }
            __syncthreads();

            // A block is capped at 256 threads. Iterate in block-sized tiles so
            // configurations with more than 256 PRBGs still produce every output.
            // The barriers stay outside this loop so inactive lanes in the final
            // partial tile cannot skip a block-wide synchronization point.
            for (uint32_t prbgLinearIdx = threadIdx.x;
                 prbgLinearIdx < N_Prbg;
                 prbgLinearIdx += blockDim.x) {
                const uint16_t prbgIdx = static_cast<uint16_t>(prbgLinearIdx);

                // Calculate offsets for this antenna combination
                uint32_t prbg_offset = calculateCfrOffset(batchIdx, ueAntIdx, bsAntIdx, prbgIdx, nUtAnt, nCellAnt, N_Prbg, optionalCfrDim);
                uint16_t localScOffset = prbgIdx * N_sc_Prbg;
            
                // Calculate CFR on frequency
                Tcomplex cfrOnFreqKHz = {0.0f, 0.0f};
                Tcomplex tempSum = {0.0f, 0.0f};
            
                switch(freqConvertType) {
                case 0: // use first SC for CFR on the Prbg
                {
                    float freqKHz = (localScOffset - N_sc_over_2) * simConfig->sc_spacing_hz * 1e-3f;
                    cfrOnFreqKHz = calCfrbyCir(freqKHz, cirNtaps, s_cirNormDelayUs2Pi, s_timeChanLocal, cfrNormalizationFactor);
                    if (activeLink.freqChanPrbg != nullptr) {
                        activeLink.freqChanPrbg[prbg_offset] = cfrOnFreqKHz;
                    }
                    break;
                }
                
                case 1: // use center SC for CFR on the Prbg
                {
                    uint16_t N_sc_current_Prbg = (prbgIdx < N_Prbg - 1) ? N_sc_Prbg : (N_sc - (N_Prbg - 1) * N_sc_Prbg);
                    float freqKHz = (localScOffset + N_sc_current_Prbg/2 - N_sc_over_2) * simConfig->sc_spacing_hz * 1e-3f;
                    cfrOnFreqKHz = calCfrbyCir(freqKHz, cirNtaps, s_cirNormDelayUs2Pi, s_timeChanLocal, cfrNormalizationFactor);
                    if (activeLink.freqChanPrbg != nullptr) {
                        activeLink.freqChanPrbg[prbg_offset] = cfrOnFreqKHz;
                    }
                    break;
                }
                
                case 2: // use last SC for CFR on the Prbg
                {
                    uint16_t N_sc_current_Prbg = (prbgIdx < N_Prbg - 1) ? N_sc_Prbg : (N_sc - (N_Prbg - 1) * N_sc_Prbg);
                    float freqKHz = (localScOffset + N_sc_current_Prbg - 1 - N_sc_over_2) * simConfig->sc_spacing_hz * 1e-3f;
                    cfrOnFreqKHz = calCfrbyCir(freqKHz, cirNtaps, s_cirNormDelayUs2Pi, s_timeChanLocal, cfrNormalizationFactor);
                    if (activeLink.freqChanPrbg != nullptr) {
                        activeLink.freqChanPrbg[prbg_offset] = cfrOnFreqKHz;
                    }
                    break;
                }
                
                case 3: // use average SC for CFR on the Prbg
                {
                    uint16_t N_sc_current_Prbg = (prbgIdx < N_Prbg - 1) ? N_sc_Prbg : (N_sc - (N_Prbg - 1) * N_sc_Prbg);
                    uint16_t sample_count = (N_sc_current_Prbg + scSampling - 1) / scSampling;
                    float inverseNScPrbg = 1.0f / sample_count;
                    tempSum = {0.0f, 0.0f};
                    for (uint16_t scInPrbgIdx = 0; scInPrbgIdx < N_sc_current_Prbg; scInPrbgIdx += scSampling) {
                        float freqKHz = (localScOffset + scInPrbgIdx - N_sc_over_2) * simConfig->sc_spacing_hz * 1e-3f;
                        cfrOnFreqKHz = calCfrbyCir(freqKHz, cirNtaps, s_cirNormDelayUs2Pi, s_timeChanLocal, cfrNormalizationFactor);
                        tempSum.x += cfrOnFreqKHz.x;
                        tempSum.y += cfrOnFreqKHz.y;
                    }
                    if (activeLink.freqChanPrbg != nullptr) {
                        activeLink.freqChanPrbg[prbg_offset].x = tempSum.x * inverseNScPrbg;
                        activeLink.freqChanPrbg[prbg_offset].y = tempSum.y * inverseNScPrbg;
                    }
                    break;
                }
                
                case 4: // use average SC with frequency ramping removal
                {
                    uint16_t N_sc_current_Prbg = (prbgIdx < N_Prbg - 1) ? N_sc_Prbg : (N_sc - (N_Prbg - 1) * N_sc_Prbg);
                    uint16_t sample_count = (N_sc_current_Prbg + scSampling - 1) / scSampling;
                    float inverseNScPrbg = 1.0f / sample_count;
                    tempSum = {0.0f, 0.0f};
                    float centerFreqKHz = (localScOffset + N_sc_current_Prbg/2 - N_sc_over_2) * simConfig->sc_spacing_hz * 1e-3f;
                    for (uint16_t scInPrbgIdx = 0; scInPrbgIdx < N_sc_current_Prbg; scInPrbgIdx += scSampling) {
                        cfrOnFreqKHz = calCfrbyCir(centerFreqKHz, cirNtaps, s_cirNormDelayUs2Pi, s_timeChanLocal, cfrNormalizationFactor);
                        tempSum.x += cfrOnFreqKHz.x;
                        tempSum.y += cfrOnFreqKHz.y;
                    }
                    if (activeLink.freqChanPrbg != nullptr) {
                        activeLink.freqChanPrbg[prbg_offset].x = tempSum.x * inverseNScPrbg;
                        activeLink.freqChanPrbg[prbg_offset].y = tempSum.y * inverseNScPrbg;
                    }
                    break;
                }
                
                default:
                    if (activeLink.freqChanPrbg != nullptr) {
                        activeLink.freqChanPrbg[prbg_offset] = make_cuComplex(0.0f, 0.0f);
                    }
#ifdef SLS_DEBUG_
                    printf("Error: Invalid freqConvertType %d!\n", freqConvertType);
#endif
                    break;
                }
#ifdef SLS_DEBUG_
                // NaN detection for CFR coefficients
                if (isnan(cfrOnFreqKHz.x) || isnan(cfrOnFreqKHz.y) ||
                    isinf(cfrOnFreqKHz.x) || isinf(cfrOnFreqKHz.y)) {
                    printf("ERROR: Invalid CFR detected in case 0 - Real: %f, Imag: %f (activeLinkIdx=%u, ueAnt=%u, bsAnt=%u, prbgIdx=%u)\n",
                        cfrOnFreqKHz.x, cfrOnFreqKHz.y, activeLinkIdx, ueAntIdx, bsAntIdx, prbgIdx);
                    cfrOnFreqKHz = make_cuComplex(0.0f, 0.0f); // Use zero instead of NaN
                }
#endif
            }
            __syncthreads();
        }
    }
}

    // GPU kernel to generate CFR - Mode 2&3 (SC level, and optionally PRBG level)
// Mode 2: SC CFR only (freqChanPrbg == nullptr)
// Mode 3: SC CFR + PRBG CFR (freqChanPrbg != nullptr)
template <typename Tcomplex>
__global__ void generateCFRKernel_runMode23(const activeLink<Tcomplex>* activeLinkParams,
                                           const CellParam* cellParams, const UtParam* utParams,
                                           const LinkParams* linkParams, const ClusterParams* clusterParams,
                                           const CmnLinkParams* cmnLinkParams, const SimConfig* simConfig,
                                           const AntPanelConfig* antPanelConfigs, const SystemLevelConfig* sysConfig,
                                           uint32_t nActiveLinks, float refTime, float cfrNormalizationFactor, curandState* curandStates) {
    // GRID(nActiveLinks, m_scaleUeAntFreqChan * m_nBatch, m_scaleBsAntFreqChan)
    // BLOCK(N_Prbg, nBsAnt/m_scaleBsAntFreqChan, nUeAnt/m_scaleUeAntFreqChan)
    
    uint32_t activeLinkIdx = blockIdx.x;
    if (activeLinkIdx >= nActiveLinks) return;
    
    const activeLink<Tcomplex>& activeLink = activeLinkParams[activeLinkIdx];
    uint16_t N_Prbg = simConfig->n_prbg;
    uint16_t N_sc = simConfig->n_prb * 12;
    uint16_t N_sc_Prbg = (uint16_t)ceilf((float)(N_sc) / N_Prbg);
    uint16_t N_sc_last_Prbg = N_sc - (N_Prbg - 1) * N_sc_Prbg;  // Use allocated size, not fft_size
    uint16_t N_sc_over_2 = N_sc >> 1;
    uint8_t freqConvertType = simConfig->freq_convert_type;
    uint8_t scSampling = simConfig->sc_sampling;
    bool optionalCfrDim = (simConfig->optional_cfr_dim == 1);
    
    // Get antenna configurations from actual UT and Cell parameters  
    uint16_t cid = activeLink.cid;
    uint16_t uid = activeLink.uid;
    uint32_t utAntPanelIdx = utParams[uid].antPanelIdx;
    uint32_t cellAntPanelIdx = cellParams[cid].antPanelIdx;
    AntPanelConfig utAntPanelConfig = antPanelConfigs[utAntPanelIdx];
    AntPanelConfig cellAntPanelConfig = antPanelConfigs[cellAntPanelIdx];
    uint32_t nUtAnt = utAntPanelConfig.nAnt;
    uint32_t nCellAnt = cellAntPanelConfig.nAnt;
    
    // Thread and block indices
    uint16_t batchIdx = blockIdx.y;
    
    // Shared memory for CIR
    extern __shared__ char shareData[];
    Tcomplex* s_timeChanLocal = reinterpret_cast<Tcomplex*>(shareData);
    __shared__ float s_cirNormDelayUs2Pi[N_MAX_TAPS];
    
    // Read CIR normalization delays (only once per block)
    uint16_t cirNtaps = activeLink.cirNtaps[0];  // Note: cirNtaps is per-link, not per-snapshot
    if (threadIdx.x == 0) {
        for (uint16_t copyIdx = 0; copyIdx < cirNtaps; copyIdx++) {
            // Convert normalized delay to 2*pi*delay in microseconds
            // Note: cirNormDelay is per-link, not per-snapshot
            float delayUs = activeLink.cirNormDelay[copyIdx] * 1e6f / 
                           (simConfig->sc_spacing_hz * simConfig->fft_size);
            s_cirNormDelayUs2Pi[copyIdx] = 2.0f * M_PI * delayUs;
        }
    }
    __syncthreads();
    
    // Iterate over all antenna combinations
    for (uint16_t ueAntIdx = 0; ueAntIdx < nUtAnt; ueAntIdx++) {
        for (uint16_t bsAntIdx = 0; bsAntIdx < nCellAnt; bsAntIdx++) {
            // Calculate CIR offset for this antenna combination within this link's data
            // activeLink.cirCoe already points to this link's data, so just need antenna + snapshot offset
            size_t cirOffset = batchIdx * nUtAnt * nCellAnt * N_MAX_TAPS + (ueAntIdx * nCellAnt + bsAntIdx) * N_MAX_TAPS;
            
            // Copy CIR coefficients to shared memory with CFO rotation
            for (uint32_t copyIdx = threadIdx.x; copyIdx < cirNtaps; copyIdx += blockDim.x) {
#ifdef SLS_DEBUG_
                // Bounds check to prevent out-of-bounds access
                if (copyIdx >= N_MAX_TAPS) {
                    printf("ERROR: copyIdx %u >= N_MAX_TAPS %d for cirNtaps %u\n", 
                           copyIdx, N_MAX_TAPS, cirNtaps);
                    break;
                }
#endif
                
                // Placeholder CFO rotation - to be implemented later
                Tcomplex cfrRotationTotal = {1.0f, 0.0f}; // Identity rotation for now
                
                // activeLink.cirCoe already points to this link's data
                Tcomplex tmpCopyCir = activeLink.cirCoe[cirOffset + copyIdx];
                
                // Additional safety check for the retrieved coefficient
#ifdef SLS_DEBUG_
                if (isnan(tmpCopyCir.x) || isnan(tmpCopyCir.y) || isinf(tmpCopyCir.x) || isinf(tmpCopyCir.y)) {
                    printf("ERROR: Invalid CIR coefficient at copyIdx=%u, cirOffset=%zu: (%f, %f)\n", 
                           copyIdx, cirOffset, tmpCopyCir.x, tmpCopyCir.y);
                    tmpCopyCir = make_cuComplex(0.0f, 0.0f); // Use zero instead of NaN
                }
#endif
                
                s_timeChanLocal[copyIdx].x = tmpCopyCir.x * cfrRotationTotal.x - tmpCopyCir.y * cfrRotationTotal.y;
                s_timeChanLocal[copyIdx].y = tmpCopyCir.x * cfrRotationTotal.y + tmpCopyCir.y * cfrRotationTotal.x;
            }
            __syncthreads();

            // Cover all PRBGs even when N_Prbg exceeds the 256-thread block cap.
            // No barrier is allowed inside this uneven per-thread loop.
            for (uint32_t prbgLinearIdx = threadIdx.x;
                 prbgLinearIdx < N_Prbg;
                 prbgLinearIdx += blockDim.x) {
                const uint16_t prbgIdx = static_cast<uint16_t>(prbgLinearIdx);

                // Calculate offsets for this antenna combination
                uint32_t prbg_offset = calculateCfrOffset(batchIdx, ueAntIdx, bsAntIdx, prbgIdx, nUtAnt, nCellAnt, N_Prbg, optionalCfrDim);
                uint16_t localScOffset = prbgIdx * N_sc_Prbg;
                // CFR pointers are already per-link, so no need to include activeLinkIdx again
                uint32_t sc_start_offset = calculateScCfrOffset(batchIdx, ueAntIdx, bsAntIdx, localScOffset, nUtAnt, nCellAnt, N_sc, optionalCfrDim);
                // Element stride between consecutive SCs in freqChanSc: the optional layout is
                // [.., nSc, nUtAnt, nBsAnt] so neighboring SCs sit nUtAnt*nBsAnt apart, while
                // the default layout keeps SC innermost (stride 1)
                const uint32_t scStride = optionalCfrDim ? nUtAnt * nCellAnt : 1u;
            
                // Calculate CFR on all SCs and save to GPU global memory
                Tcomplex cfrOnFreqKHz = {0.0f, 0.0f};
                Tcomplex tempSum = {0.0f, 0.0f};
                uint16_t N_sc_current_Prbg = (prbgIdx < N_Prbg - 1) ? N_sc_Prbg : N_sc_last_Prbg;
                const uint16_t sample_count =
                    (N_sc_current_Prbg + scSampling - 1) / scSampling;
                const float inverseNScPrbg = 1.0f / sample_count;
            
                for (uint16_t scInPrbgIdx = 0;
                     scInPrbgIdx < N_sc_current_Prbg;
                     scInPrbgIdx += scSampling) {
                    const float freqKHz =
                        (localScOffset + scInPrbgIdx - N_sc_over_2) *
                        simConfig->sc_spacing_hz * 1e-3f;
                    cfrOnFreqKHz = calCfrbyCir(
                        freqKHz, cirNtaps, s_cirNormDelayUs2Pi, s_timeChanLocal,
                        cfrNormalizationFactor);

#ifdef SLS_DEBUG_
                    // NaN detection for SC CFR coefficients (runMode23)
                    if (isnan(cfrOnFreqKHz.x) || isnan(cfrOnFreqKHz.y) ||
                        isinf(cfrOnFreqKHz.x) || isinf(cfrOnFreqKHz.y)) {
                        printf("ERROR: Invalid SC CFR detected in runMode23 - Real: %f, Imag: %f (activeLinkIdx=%u, ueAnt=%u, bsAnt=%u, scIdx=%u)\n",
                               cfrOnFreqKHz.x, cfrOnFreqKHz.y, activeLinkIdx,
                               ueAntIdx, bsAntIdx, scInPrbgIdx);
                        cfrOnFreqKHz = make_cuComplex(0.0f, 0.0f);
                    }
#endif
                    if (activeLink.freqChanSc != nullptr) {
                        activeLink.freqChanSc[
                            sc_start_offset + scInPrbgIdx * scStride] = cfrOnFreqKHz;
                    }

                    // Only accumulate for PRBG conversion if PRBG output is needed.
                    if (activeLink.freqChanPrbg != nullptr &&
                        freqConvertType == 3) {
                        tempSum.x += cfrOnFreqKHz.x;
                        tempSum.y += cfrOnFreqKHz.y;
                    }
                }

                // Convert SC CFR to PRBG CFR when run mode 3 requests it.
                if (activeLink.freqChanPrbg != nullptr) {
                    switch (freqConvertType) {
                        case 0:
                            if (activeLink.freqChanSc != nullptr) {
                                activeLink.freqChanPrbg[prbg_offset] =
                                    activeLink.freqChanSc[sc_start_offset];
                            } else {
                                const float freqKHz =
                                    (localScOffset - N_sc_over_2) *
                                    simConfig->sc_spacing_hz * 1e-3f;
                                activeLink.freqChanPrbg[prbg_offset] = calCfrbyCir(
                                    freqKHz, cirNtaps, s_cirNormDelayUs2Pi,
                                    s_timeChanLocal, cfrNormalizationFactor);
                            }
                            break;

                        case 1:
                            if ((N_sc_current_Prbg / 2) % scSampling == 0 &&
                                activeLink.freqChanSc != nullptr) {
                                activeLink.freqChanPrbg[prbg_offset] =
                                    activeLink.freqChanSc[
                                        sc_start_offset +
                                        (N_sc_current_Prbg / 2) * scStride];
                            } else {
                                float freqKHz =
                                    (localScOffset + N_sc_current_Prbg / 2 - N_sc_over_2) *
                                    simConfig->sc_spacing_hz * 1e-3f;
                                cfrOnFreqKHz = calCfrbyCir(
                                    freqKHz, cirNtaps, s_cirNormDelayUs2Pi,
                                    s_timeChanLocal, cfrNormalizationFactor);
                                activeLink.freqChanPrbg[prbg_offset] = cfrOnFreqKHz;
                            }
                            break;

                        case 2:
                            if ((N_sc_current_Prbg - 1) % scSampling == 0 &&
                                activeLink.freqChanSc != nullptr) {
                                activeLink.freqChanPrbg[prbg_offset] =
                                    activeLink.freqChanSc[
                                        sc_start_offset +
                                        (N_sc_current_Prbg - 1) * scStride];
                            } else {
                                float freqKHz =
                                    (localScOffset + N_sc_current_Prbg - 1 - N_sc_over_2) *
                                    simConfig->sc_spacing_hz * 1e-3f;
                                cfrOnFreqKHz = calCfrbyCir(
                                    freqKHz, cirNtaps, s_cirNormDelayUs2Pi,
                                    s_timeChanLocal, cfrNormalizationFactor);
                                activeLink.freqChanPrbg[prbg_offset] = cfrOnFreqKHz;
                            }
                            break;

                        case 3:
                            activeLink.freqChanPrbg[prbg_offset].x =
                                tempSum.x * inverseNScPrbg;
                            activeLink.freqChanPrbg[prbg_offset].y =
                                tempSum.y * inverseNScPrbg;
                            break;

                        case 4: {
                            const float freqKHz =
                                (localScOffset + N_sc_current_Prbg / 2 - N_sc_over_2) *
                                simConfig->sc_spacing_hz * 1e-3f;
                            activeLink.freqChanPrbg[prbg_offset] = calCfrbyCir(
                                freqKHz, cirNtaps, s_cirNormDelayUs2Pi,
                                s_timeChanLocal, cfrNormalizationFactor);
                            break;
                        }

                        default:
                            activeLink.freqChanPrbg[prbg_offset] =
                                make_cuComplex(0.0f, 0.0f);
#ifdef SLS_DEBUG_
                            printf("Error: Invalid freqConvertType %d!\n", freqConvertType);
#endif
                            break;
                    }

#ifdef SLS_DEBUG_
                    if (isnan(activeLink.freqChanPrbg[prbg_offset].x) || isnan(activeLink.freqChanPrbg[prbg_offset].y) || 
                        isinf(activeLink.freqChanPrbg[prbg_offset].x) || isinf(activeLink.freqChanPrbg[prbg_offset].y)) {
                        printf("ERROR: Invalid PRBG CFR detected in runMode23 - Real: %f, Imag: %f (activeLinkIdx=%u, ueAnt=%u, bsAnt=%u, prbgIdx=%u)\n",
                               activeLink.freqChanPrbg[prbg_offset].x, activeLink.freqChanPrbg[prbg_offset].y, activeLinkIdx, ueAntIdx, bsAntIdx, prbgIdx);
                        activeLink.freqChanPrbg[prbg_offset] = make_cuComplex(0.0f, 0.0f); // Use zero instead of NaN
                    }
#endif
                }
            }
            __syncthreads();
        }
    }
}

// GPU kernel to generate CFR - Mode 4 (All N_FFT subcarriers)
template <typename Tcomplex>
__global__ void generateCFRKernel_runMode4(const activeLink<Tcomplex>* activeLinkParams,
                                          const CellParam* cellParams, const UtParam* utParams,
                                          const LinkParams* linkParams, const ClusterParams* clusterParams,
                                          const CmnLinkParams* cmnLinkParams, const SimConfig* simConfig,
                                          const AntPanelConfig* antPanelConfigs, const SystemLevelConfig* sysConfig,
                                          uint32_t nActiveLinks, float refTime, float cfrNormalizationFactor, curandState* curandStates) {
    // GRID(nActiveLinks, nSnapshots, 1)
    // BLOCK(min(N_FFT, 1024), 1, 1) - iterate over antennas instead of using thread dimensions
    
    uint32_t activeLinkIdx = blockIdx.x;
    if (activeLinkIdx >= nActiveLinks) return;
    
    const activeLink<Tcomplex>& activeLink = activeLinkParams[activeLinkIdx];
    uint16_t N_fft = simConfig->fft_size;
    uint16_t N_fft_over_2 = N_fft >> 1;
    bool optionalCfrDim = (simConfig->optional_cfr_dim == 1);
    
    // Get antenna configurations from actual UT and Cell parameters
    uint16_t cid = activeLink.cid;
    uint16_t uid = activeLink.uid;
    uint32_t utAntPanelIdx = utParams[uid].antPanelIdx;
    uint32_t cellAntPanelIdx = cellParams[cid].antPanelIdx;
    AntPanelConfig utAntPanelConfig = antPanelConfigs[utAntPanelIdx];
    AntPanelConfig cellAntPanelConfig = antPanelConfigs[cellAntPanelIdx];
    uint32_t nUtAnt = utAntPanelConfig.nAnt;
    uint32_t nCellAnt = cellAntPanelConfig.nAnt;
    
    // Debug antenna config (only from first thread to avoid spam)
#ifdef SLS_DEBUG_
    if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0) {
        printf("KERNEL DEBUG: Run Mode 4 - nUtAnt=%u, nCellAnt=%u\n", nUtAnt, nCellAnt);
        printf("KERNEL DEBUG: uid=%u, cid=%u\n", uid, cid);
        printf("KERNEL DEBUG: utAntPanelIdx=%u, cellAntPanelIdx=%u\n", utAntPanelIdx, cellAntPanelIdx);
        printf("KERNEL DEBUG: utAntPanelConfig.nAnt=%u, cellAntPanelConfig.nAnt=%u\n", 
               utAntPanelConfig.nAnt, cellAntPanelConfig.nAnt);
        printf("KERNEL DEBUG: blockDim=(%u,%u,%u), gridDim=(%u,%u,%u)\n", 
               blockDim.x, blockDim.y, blockDim.z, gridDim.x, gridDim.y, gridDim.z);
    }
#endif
    
    // Thread and block indices
    uint16_t batchIdx = blockIdx.y;
    
    // Shared memory for CIR
    extern __shared__ char shareData[];
    Tcomplex* s_timeChanLocal = reinterpret_cast<Tcomplex*>(shareData);
    __shared__ float s_cirNormDelayUs2Pi[N_MAX_TAPS];
    
    // Read CIR normalization delays (only once per block)
    uint16_t cirNtaps = activeLink.cirNtaps[0];  // Note: cirNtaps is per-link, not per-snapshot
    if (threadIdx.x == 0) {
        for (uint16_t copyIdx = 0; copyIdx < cirNtaps; copyIdx++) {
            // Convert normalized delay to 2*pi*delay in microseconds
            // Note: cirNormDelay is per-link, not per-snapshot
            float delayUs = activeLink.cirNormDelay[copyIdx] * 1e6f / 
                           (simConfig->sc_spacing_hz * simConfig->fft_size);
            s_cirNormDelayUs2Pi[copyIdx] = 2.0f * M_PI * delayUs;
        }
    }
    __syncthreads();
    
    // Iterate over all antenna combinations (instead of using thread dimensions)
    for (uint16_t ueAntIdx = 0; ueAntIdx < nUtAnt; ueAntIdx++) {
        for (uint16_t bsAntIdx = 0; bsAntIdx < nCellAnt; bsAntIdx++) {
            
            // Copy CIR coefficients to shared memory with CFO rotation
            // activeLink.cirCoe already points to this link's data, so just need antenna + snapshot offset
            size_t cirOffset = batchIdx * nUtAnt * nCellAnt * N_MAX_TAPS + (ueAntIdx * nCellAnt + bsAntIdx) * N_MAX_TAPS;
            
            // Load CIR for this antenna combination
            for (uint16_t copyIdx = threadIdx.x; copyIdx < cirNtaps; copyIdx += blockDim.x) {
#ifdef SLS_DEBUG_
                // Bounds check to prevent out-of-bounds access
                if (copyIdx >= N_MAX_TAPS) {
                    printf("ERROR: copyIdx %u >= N_MAX_TAPS %d for cirNtaps %u\n", 
                           copyIdx, N_MAX_TAPS, cirNtaps);
                    break;
                }
#endif
                
                // Placeholder CFO rotation - to be implemented later
                Tcomplex cfrRotationTotal = {1.0f, 0.0f}; // Identity rotation for now
                
                // activeLink.cirCoe already points to this link's data
                Tcomplex tmpCopyCir = activeLink.cirCoe[cirOffset + copyIdx];
                s_timeChanLocal[copyIdx].x = tmpCopyCir.x * cfrRotationTotal.x - tmpCopyCir.y * cfrRotationTotal.y;
                s_timeChanLocal[copyIdx].y = tmpCopyCir.x * cfrRotationTotal.y + tmpCopyCir.y * cfrRotationTotal.x;
            }
            __syncthreads();
            
            // Generate CFR for multiple subcarriers per thread to cover all N_FFT subcarriers
            uint16_t subcarriersPerThread = (N_fft + blockDim.x - 1) / blockDim.x;  // Ceiling division
            for (uint16_t scStep = 0; scStep < subcarriersPerThread; scStep++) {
                uint16_t scIdx = threadIdx.x + scStep * blockDim.x;
                
                if (scIdx < N_fft && activeLink.freqChanSc != nullptr) {
                    // activeLink.freqChanSc already points to this link's data, just need antenna + subcarrier offset
                    uint32_t sc_offset = calculateScCfrOffset(batchIdx, ueAntIdx, bsAntIdx, scIdx, nUtAnt, nCellAnt, N_fft, optionalCfrDim);
                    
#ifdef SLS_DEBUG_
                    // Bounds checking to prevent illegal memory access
                    uint32_t maxOffset = simConfig->n_snapshot_per_slot * nUtAnt * nCellAnt * N_fft;
                    
                    if (sc_offset >= maxOffset) {
                        printf("ERROR: CFR SC offset %u >= maxOffset %u (batch=%u, ueAnt=%u, bsAnt=%u, sc=%u)\n", 
                               sc_offset, maxOffset, batchIdx, ueAntIdx, bsAntIdx, scIdx);
                        return; // Skip this write to prevent illegal access
                    }
                    
                    // Debug first few writes for the first link
                    if (activeLinkIdx == 0 && scIdx < 3) {
                        printf("DEBUG: Link %u CFR SC write: offset=%u, freqChanSc=%p, maxOffset=%u\n", 
                               activeLinkIdx, sc_offset, (void*)activeLink.freqChanSc, maxOffset);
                    }
#endif
                    
                    float freqKHz = (scIdx - N_fft_over_2) * simConfig->sc_spacing_hz * 1e-3f;
                    Tcomplex cfrOnFreqKHz = calCfrbyCir(freqKHz, cirNtaps, s_cirNormDelayUs2Pi, s_timeChanLocal, cfrNormalizationFactor);
                    
#ifdef SLS_DEBUG_
                    // NaN detection for SC CFR coefficients (runMode4)
                    if (isnan(cfrOnFreqKHz.x) || isnan(cfrOnFreqKHz.y) || 
                        isinf(cfrOnFreqKHz.x) || isinf(cfrOnFreqKHz.y)) {
                        printf("ERROR: Invalid SC CFR detected in runMode4 - Real: %f, Imag: %f (activeLinkIdx=%u, ueAnt=%u, bsAnt=%u, scIdx=%u)\n",
                               cfrOnFreqKHz.x, cfrOnFreqKHz.y, activeLinkIdx, ueAntIdx, bsAntIdx, scIdx);
                        cfrOnFreqKHz = make_cuComplex(0.0f, 0.0f); // Use zero instead of NaN
                    }
#endif
                    
                    activeLink.freqChanSc[sc_offset] = cfrOnFreqKHz;
                }
            }
            __syncthreads();
        }
    }
    
    // NOTE: PRBG level CFR is not needed for run mode 4 since we have full N_FFT resolution
}

// implementation for generateCFRGPU
template <typename Tscalar, typename Tcomplex>
void slsChan<Tscalar, Tcomplex>::initializeSmallScaleGpuKernels()
{
    CHECK_CUDAERROR(cudaGetFuncBySymbol(
        &m_calClusterRayFunction,
        reinterpret_cast<const void*>(calClusterRayKernel)));
    CHECK_CUDAERROR(cudaGetFuncBySymbol(
        &m_generateCirFunction,
        reinterpret_cast<const void*>(generateCIRKernel<Tcomplex>)));
    CHECK_CUDAERROR(cudaGetFuncBySymbol(
        &m_generateCfrRunMode1Function,
        reinterpret_cast<const void*>(generateCFRKernel_runMode1<Tcomplex>)));
    CHECK_CUDAERROR(cudaGetFuncBySymbol(
        &m_generateCfrRunMode23Function,
        reinterpret_cast<const void*>(generateCFRKernel_runMode23<Tcomplex>)));
    CHECK_CUDAERROR(cudaGetFuncBySymbol(
        &m_generateCfrRunMode4Function,
        reinterpret_cast<const void*>(generateCFRKernel_runMode4<Tcomplex>)));

    constexpr int defaultSharedLimit = 48 * 1024;
    CUdevice device;
    int maxOptin = 0;
    CHECK_CURESULT(cuCtxGetDevice(&device));
    CHECK_CURESULT(cuDeviceGetAttribute(
        &maxOptin, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, device));
    m_cirSharedBudget = defaultSharedLimit;
    if (maxOptin > defaultSharedLimit) {
        const CUresult attrErr = cuFuncSetAttribute(
            m_generateCirFunction, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, maxOptin);
        if (attrErr == CUDA_SUCCESS) {
            m_cirSharedBudget = maxOptin;
        } else {
            const char* errorString = nullptr;
            cuGetErrorString(attrErr, &errorString);
            printf("WARNING: could not raise generateCIRKernel dynamic shared mem to %d bytes: %s\n",
                   maxOptin, errorString != nullptr ? errorString : "unknown CUDA error");
        }
    }
}

template <typename Tscalar, typename Tcomplex>
void slsChan<Tscalar, Tcomplex>::generateCFRGPU()
{
#ifdef SLS_DEBUG_
    printf("DEBUG: generateCFRGPU called, run_mode=%d, active_links=%zu\n", 
           m_simConfig->run_mode, m_activeLinkParams.size());
#endif
    
    if (m_activeLinkParams.empty()) {
#ifdef SLS_DEBUG_
        printf("DEBUG: No active links, skipping CFR generation\n");
#endif
        return;
    }
    
    // Check if we need SC level or PRBG level CFR
    bool needScLevel = false;
    bool needPrbgLevel = false;
    
    for (const auto& activeLink : m_activeLinkParams) {
        if (activeLink.freqChanSc != nullptr) needScLevel = true;
        if (activeLink.freqChanPrbg != nullptr) needPrbgLevel = true;
    }
    
    if (!needScLevel && !needPrbgLevel) {
        return; // Nothing to compute
    }
    
    // Get antenna configurations
    uint32_t nUtAnt = (*m_antPanelConfig)[0].nAnt;
    uint32_t nCellAnt = (*m_antPanelConfig)[1].nAnt;
    
    // Calculate safe block dimensions (reduced to avoid GPU resource limits)
    const uint32_t maxThreadsPerBlock = 256;
    const uint32_t maxSharedMemPerBlock = 48 * 1024; // 48KB typical limit
    
    // For other modes, keep the first dimension but ensure total threads <= 256
    uint32_t threadsPerBlockX;
    if (m_simConfig->run_mode == 4) {
        threadsPerBlockX = std::min((uint32_t)m_simConfig->fft_size, maxThreadsPerBlock);
    } else {
        threadsPerBlockX = std::min((uint32_t)m_simConfig->n_prbg, maxThreadsPerBlock);
    }
    uint32_t threadsPerBlockY = 1u; // Use 1 for Y dimension  
    uint32_t threadsPerBlockZ = 1u; // Use 1 for Z dimension
    
    // Total threads per block is now guaranteed <= 256
    uint32_t totalThreadsPerBlock = threadsPerBlockX * threadsPerBlockY * threadsPerBlockZ;
    
    dim3 blockDim(threadsPerBlockX, threadsPerBlockY, threadsPerBlockZ);
    dim3 gridDim(m_activeLinkParams.size(), m_simConfig->n_snapshot_per_slot, 1);
    
    // Calculate shared memory size and ensure it fits
    size_t sharedMemSize = N_MAX_TAPS * threadsPerBlockY * threadsPerBlockZ * sizeof(Tcomplex);
    if (sharedMemSize > maxSharedMemPerBlock) {
        // Reduce shared memory usage by limiting the working threads
        sharedMemSize = maxSharedMemPerBlock / 2; // Use half the available shared memory
        printf("WARNING: Reducing shared memory usage to %zu bytes\n", sharedMemSize);
    }

    // Validate configuration before launch
    if (totalThreadsPerBlock == 0 || totalThreadsPerBlock > maxThreadsPerBlock) {
        throw std::invalid_argument(
            "Invalid CFR block configuration: " + std::to_string(totalThreadsPerBlock) +
            " threads per block");
    }
    if (gridDim.x == 0 || gridDim.y == 0) {
        throw std::invalid_argument("CFR launch grid must be nonzero");
    }
    
    // CFR normalization: Channel models should NOT apply FFT energy normalization
    // The CIR taps are already power-normalized (cluster powers + path loss)
    // Applying sqrt(1/N_fft) would incorrectly reduce channel power by N_fft
    // For 3GPP channel models, CFR should represent actual channel gain, not FFT-normalized energy
    float cfrNormalizationFactor = 1.0f;  // No normalization for channel frequency response
#ifdef SLS_DEBUG_
    printf("DEBUG: CFR normalization factor: %f (no FFT energy normalization for channel models)\n", 
            cfrNormalizationFactor);
#endif

    // Choose kernel based on run mode
#ifdef SLS_DEBUG_
    printf("DEBUG: Checking run mode: %d\n", m_simConfig->run_mode);
    
    // Debug: Check activeLink CFR pointers
    printf("DEBUG: First few activeLink CFR pointers:\n");
    for (size_t i = 0; i < std::min((size_t)3, m_activeLinkParams.size()); i++) {
        printf("  Link %zu: freqChanSc=%p, freqChanPrbg=%p\n", i, 
               (void*)m_activeLinkParams[i].freqChanSc, 
               (void*)m_activeLinkParams[i].freqChanPrbg);
    }
#endif
    CUfunction generateCFRFunction{};
    dim3 launchGridDim = gridDim;
    dim3 launchBlockDim = blockDim;
    size_t launchSharedMemSize = sharedMemSize;

    if (m_simConfig->run_mode == 1) {
        // Mode 1: CIR and CFR on PRBG only
        generateCFRFunction = m_generateCfrRunMode1Function;
    } else if (m_simConfig->run_mode == 2 || m_simConfig->run_mode == 3) {
        // Mode 2: CIR and CFR on SC (n_prb*12 subcarriers) only
        // Mode 3: CIR and CFR on SC (n_prb*12 subcarriers) and PRBG
        generateCFRFunction = m_generateCfrRunMode23Function;
    } else if (m_simConfig->run_mode == 4) {
        // Mode 4: CIR and CFR on all N_FFT subcarriers
#ifdef SLS_DEBUG_
        printf("DEBUG: Antenna counts - nCellAnt=%u, nUtAnt=%u\n", nCellAnt, nUtAnt);
#endif
        
        // For CIR to CFR conversion, use min(N_fft, 1024) threads with (1,1) dimensions
        // and iterate over antenna indices to avoid exceeding 1024 threads per block
        uint32_t threadsPerBlockX_fft = std::min((uint32_t)m_simConfig->fft_size, 1024u);
        
#ifdef SLS_DEBUG_
        printf("DEBUG: Block dimensions - X=%u, Y=%u, Z=%u\n", 
               threadsPerBlockX_fft, 1u, 1u);
#endif
        
        launchBlockDim = dim3(threadsPerBlockX_fft, 1, 1);
        // Grid dimensions: (nActiveLinks, nSnapshots, 1). No need for antenna
        // combinations in the grid because the kernel iterates over antennas.
        launchGridDim = dim3(m_activeLinkParams.size(), m_simConfig->n_snapshot_per_slot, 1);
        launchSharedMemSize = N_MAX_TAPS * sizeof(Tcomplex) + N_MAX_TAPS * sizeof(float);
        
        // Always print Run Mode 4 configuration (not just in debug mode)
#ifdef SLS_DEBUG_
        printf("=== RUN MODE 4 KERNEL CONFIG ===\n");
        printf("  Block dim: (%d, %d, %d) = %d total threads\n", 
               launchBlockDim.x, launchBlockDim.y, launchBlockDim.z,
               launchBlockDim.x * launchBlockDim.y * launchBlockDim.z);
        printf("  Grid dim: (%d, %d, %d)\n", launchGridDim.x, launchGridDim.y, launchGridDim.z);
        printf("  N_FFT: %d subcarriers (NOT n_prbg=%d)\n", m_simConfig->fft_size, m_simConfig->n_prbg);
        printf("  Active links: %zu\n", m_activeLinkParams.size());
        printf("  Processing FULL N_FFT resolution with antenna iteration\n");
        printf("  === HOST DEBUG: Antenna Panel Configs ===\n");
        for (size_t i = 0; i < m_antPanelConfig->size(); i++) {
            printf("  Panel[%zu]: nAnt=%u\n", i, (*m_antPanelConfig)[i].nAnt);
        }
        printf("=== END RUN MODE 4 CONFIG ===\n");
#endif
        generateCFRFunction = m_generateCfrRunMode4Function;
    } else {
        throw std::invalid_argument(
            "Unsupported CFR run mode: " + std::to_string(m_simConfig->run_mode));
    }

    uint32_t numActiveLinks = static_cast<uint32_t>(m_activeLinkParams.size());
    void* kernelArgs[] = {
        &m_d_activeLinkParams, &m_d_cellParams, &m_d_utParams,
        &m_d_linkParams, &m_d_clusterParams, &m_d_cmnLinkParams,
        &m_d_simConfig, &m_d_antPanelConfigs, &m_d_sysConfig,
        &numActiveLinks, &m_refTime, &cfrNormalizationFactor, &m_d_curandStates};
    CHECK_CURESULT(cuLaunchKernel(
        generateCFRFunction,
        launchGridDim.x, launchGridDim.y, launchGridDim.z,
        launchBlockDim.x, launchBlockDim.y, launchBlockDim.z,
        launchSharedMemSize, m_strm, kernelArgs, nullptr));

#ifdef SLS_DEBUG_
    // Synchronization is debug-only so the normal path preserves asynchronous
    // stream semantics. Stream synchronization surfaces execution failures
    // without waiting for unrelated work in the CUDA context.
    CHECK_CURESULT(cuStreamSynchronize(m_strm));
    printf("DEBUG: generateCFRKernel completed successfully\n");
#endif
}

// Implementation of missing GPU device functions

__device__ void genClusterDelayAndPowerGPU(float DS, const float* r_tao, bool losInd,
                                          uint16_t& nCluster, float K, const float* xi,
                                          float* delays, float* powers,
                                          uint16_t* strongest2clustersIdx,
                                          curandState* state, uint8_t outdoor_ind) {
    // Generate cluster delays using exponential distribution
    float r_tau = *r_tao;
    
#ifdef SLS_DEBUG_
    // Debug input parameters (only from first few threads to avoid spam)
    if (blockIdx.x == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
        printf("DEBUG: genClusterDelayAndPowerGPU inputs: DS=%.6e, r_tau=%.3f, xi=%.3f, nCluster=%d, losInd=%d, outdoor_ind=%d, K=%.3f\n",
               DS, r_tau, *xi, nCluster, losInd, outdoor_ind, K);
        if (DS == 0.0f) {
            printf("DEBUG: DS=0 detected! All cluster delays will be set to 0.0\n");
        }
    }
#endif
    
    // Generate cluster delays for ALL clusters using exponential distribution
    // Formula: tao_prime_n(n) = -r_tao*DS*log(rand(1)) for all n
    for (uint16_t n = 0; n < nCluster; n++) {
        if (DS > 0.0f) {
            float uniform_rand = curand_uniform(state);
            // Ensure uniform_rand is not too close to 0 to avoid log(0)
            uniform_rand = fmaxf(uniform_rand, 1e-10f);
            delays[n] = -r_tau * DS * logf(uniform_rand);
        } else {
            // When DS=0, all clusters have the same delay (zero)
            delays[n] = 0.0f;
        }
    }
    
    // Sort delays in ascending order: tao_n = sort(tao_prime_n)
    for (uint16_t i = 0; i < nCluster - 1; i++) {
        for (uint16_t j = 0; j < nCluster - i - 1; j++) {
            if (delays[j] > delays[j + 1]) {
                float temp = delays[j];
                delays[j] = delays[j + 1];
                delays[j + 1] = temp;
            }
        }
    }
    
    // Normalize delays to start from zero: tao_n = tao_n - tao_n(1)
    float minDelay = delays[0];
    for (uint16_t n = 0; n < nCluster; n++) {
        delays[n] -= minDelay;
    }

#ifdef SLS_DEBUG_
    // Debug delay generation process (following standard algorithm steps)
    if (blockIdx.x == 0 && threadIdx.x == 0 && threadIdx.y == 0 && DS > 0.0f) {
        printf("DEBUG: Cluster delay generation process:\n");
        printf("  Step 1 - Generated tao_prime_n (unsorted, first 5): ");
        // Note: delays are already sorted at this point, but we show the concept
        for (uint16_t i = 0; i < nCluster && i < 5; i++) {
            printf("%.3f ", delays[i] + minDelay); // Show original before normalization
        }
        printf("\n");
        printf("  Step 2 - After sort(tao_prime_n), first 5: ");
        for (uint16_t i = 0; i < nCluster && i < 5; i++) {
            printf("%.3f ", delays[i] + minDelay); // Show sorted before normalization
        }
        printf("\n");
        printf("  Step 3 - After tao_n = tao_n - tao_n(1), first 5: ");
        for (uint16_t i = 0; i < nCluster && i < 5; i++) {
            printf("%.6f ", delays[i]); // Show final delays (possibly C_tau scaled)
        }
        printf("\n");
        printf("  minDelay (tao_n(1)) was %.6f ns\n", minDelay);
        if (losInd && outdoor_ind) {
            printf("  Step 4 - C_tau scaling applied for LOS outdoor scenario\n");
        }
    }
#endif

    // Generate cluster powers using exponential decay
    // NOTE: eq 7.5-5 powers must be computed from the UNSCALED delays — per Step 5 the
    // C_tau-scaled LOS delays "are not to be used in cluster power generation"; the C_tau
    // scaling is therefore applied AFTER this power loop.
    float totalPower = 0.0f;
    float xi_val = *xi;
    
#ifdef SLS_DEBUG_
    if (blockIdx.x == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
        printf("DEBUG: Generating cluster powers with xi=%.3f, DS=%.6e, r_tau=%.3f\n", 
               xi_val, DS, r_tau);
    }
#endif
    
    for (uint16_t n = 0; n < nCluster; n++) {
        // Calculate exponential decay term
        float decay_term = (DS > 0.0f) ? expf(-delays[n] * (r_tau - 1.0f) / (r_tau * DS)) : 1.0f;
        
        // Calculate shadow fading term with clipping to avoid extreme values
        float normal_rand = curand_normal(state);
        
        // Safety check for random number generation
#ifdef SLS_DEBUG_
        if (isnan(normal_rand) || isinf(normal_rand)) {
            printf("ERROR: Invalid normal random number generated: %f (cluster=%d)\n", normal_rand, n);
            return;
        }
#endif
        float shadow_term = powf(10.0f, - xi_val * normal_rand / 10.0f);
#ifdef SLS_DEBUG_
        if (isnan(shadow_term) || isinf(shadow_term) || shadow_term <= 0.0f) {
            printf("ERROR: Invalid shadow fading term: %f (cluster=%d, xi=%.3f, normal_rand=%.3f)\n", 
                   shadow_term, n, xi_val, normal_rand);
            return;
        }
#endif
        
        powers[n] = decay_term * shadow_term;
        
        // Safety check for individual power values
#ifdef SLS_DEBUG_
        if (isnan(powers[n]) || isinf(powers[n]) || powers[n] < 0.0f) {
            printf("ERROR: Invalid individual power value: %f (cluster=%d)\n", powers[n], n);
            return;
        }
#endif
        
        totalPower += powers[n];

#ifdef SLS_DEBUG_
        if (blockIdx.x == 0 && threadIdx.x == 0 && threadIdx.y == 0 && n < 8) {
            printf("  Cluster %d: delay=%.3f ns, decay_term=%.6f, shadow_term=%.6f, power=%.6f\n",
                   n, delays[n], decay_term, shadow_term, powers[n]);
        }
#endif
    }

    // Apply C_tau scaling for LOS outdoor scenarios (eq 7.5-4), AFTER eq 7.5-5 so the
    // powers above were computed from the unscaled delays; the scaled delays are what
    // Steps 11-12 consume as CIR tap delays
    if (losInd && outdoor_ind) {
        // Calculate C_tau factor: C_tau = 0.7705 - 0.0433*K + 0.0002*K^2 + 0.000017*K^3
        float K2 = K * K;
        float K3 = K2 * K;
        float C_tau = 0.7705f - 0.0433f * K + 0.0002f * K2 + 0.000017f * K3;

        // Scale delays: tao_n = (tao_n - min_delay) / C_tau
        // Note: min_delay is already subtracted above, so just divide by C_tau
        for (uint16_t n = 0; n < nCluster; n++) {
            delays[n] /= C_tau;
        }

#ifdef SLS_DEBUG_
        if (blockIdx.x == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
            printf("  LOS outdoor: Applied C_tau scaling: K=%.3f -> C_tau=%.6f\n", K, C_tau);
            printf("  First 3 scaled delays: ");
            for (uint16_t i = 0; i < nCluster && i < 3; i++) {
                printf("%.6f ", delays[i]);
            }
            printf("\n");
        }
#endif
    }

#ifdef SLS_DEBUG_
    if (blockIdx.x == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
        printf("  Total power before normalization: %.6f\n", totalPower);
        printf("  Raw powers: [");
        for (uint16_t i = 0; i < nCluster && i < 8; i++) {
            printf("%.6f", powers[i]);
            if (i < nCluster - 1 && i < 7) printf(", ");
        }
        if (nCluster > 8) printf(", ...");
        printf("]\n");
    }
#endif
    
    // Normalize powers (avoid division by zero and NaN)
    if (totalPower > 1e-20f && !isnan(totalPower) && !isinf(totalPower)) {
        for (uint16_t n = 0; n < nCluster; n++) {
            powers[n] /= totalPower;
#ifdef SLS_DEBUG_
            if (isnan(powers[n]) || isinf(powers[n]) || powers[n] < 0.0f) {
                printf("ERROR: Invalid normalized power value: %f (cluster=%d, totalPower=%f)\n", 
                       powers[n], n, totalPower);
                return;
            }
#endif
        }
    } else {
#ifdef SLS_DEBUG_
        printf("ERROR: Invalid total power for normalization: %f\n", totalPower);
#endif
        return;
    }
    
#ifdef SLS_DEBUG_
    if (blockIdx.x == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
        printf("  Normalized powers: [");
        for (uint16_t i = 0; i < nCluster && i < 8; i++) {
            printf("%.6f", powers[i]);
            if (i < nCluster - 1 && i < 7) printf(", ");
        }
        if (nCluster > 8) printf(", ...");
        printf("]\n");
        
        // Verify normalization
        float sum = 0.0f;
        for (uint16_t i = 0; i < nCluster; i++) {
            sum += powers[i];
        }
        printf("  Power sum after normalization: %.6f (should be ~1.0)\n", sum);
    }
#endif
    
    // Find max power for threshold calculation (AFTER LOS K-factor per 3GPP)
    float maxPower = powers[0];
    for (uint16_t i = 1; i < nCluster; i++) {
        if (powers[i] > maxPower) {
            maxPower = powers[i];
        }
    }
    const float powerThreshold = maxPower * powf(10.0f, -25.0f / 10.0f);
    
#ifdef SLS_DEBUG_
    if (blockIdx.x == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
        printf("  Power filtering: maxPower=%.6f, threshold=%.6f (-25dB)\n", 
               maxPower, powerThreshold);
        printf("  Power distribution before filtering:\n");
        for (uint16_t i = 0; i < nCluster; i++) {
            printf("    Cluster %d: power=%.6f, delay=%.3f ns%s\n", 
                   i, powers[i], delays[i], 
                   (powers[i] >= powerThreshold) ? " (will keep)" : " (will discard)");
        }
    }
#endif
    
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
#ifdef SLS_DEBUG_
        else if (blockIdx.x == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
            printf("  Discarding cluster %d: power=%.6f < threshold=%.6f\n", 
                   i, powers[i], powerThreshold);
        }
#endif
    }
    
    // Update nCluster to reflect valid clusters only
    nCluster = validClusterCount;
    
#ifdef SLS_DEBUG_
    if (blockIdx.x == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
        printf("  After filtering: %d clusters remain\n", nCluster);
        printf("  Filtered powers: [");
        for (uint16_t i = 0; i < nCluster && i < 8; i++) {
            printf("%.6f", powers[i]);
            if (i < nCluster - 1 && i < 7) printf(", ");
        }
        if (nCluster > 8) printf(", ...");
        printf("]\n");
    }
#endif
    
    // Now find the two strongest clusters among the filtered clusters
    uint16_t maxIdx1{}, maxIdx2{};
    if (nCluster >= 2) {
        maxIdx1 = 0;
        maxIdx2 = 1;
        if (powers[1] > powers[0]) {
            maxIdx1 = 1;
            maxIdx2 = 0;
        }
        
        for (uint16_t n = 2; n < nCluster; n++) {
            if (powers[n] > powers[maxIdx1]) {
                maxIdx2 = maxIdx1;
                maxIdx1 = n;
            } else if (powers[n] > powers[maxIdx2]) {
                maxIdx2 = n;
            }
        }
    } else if (nCluster == 1) {
        maxIdx1 = 0;
        maxIdx2 = 0;
    }
    
    strongest2clustersIdx[0] = maxIdx1;
    strongest2clustersIdx[1] = maxIdx2;
    
#ifdef SLS_DEBUG_
    if (blockIdx.x == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
        printf("  After filtering: nCluster reduced to %d\n", nCluster);
        if (nCluster > 0) {
            printf("  Strongest clusters: [%d, %d] with powers [%.6f, %.6f]\n", 
                   maxIdx1, maxIdx2, 
                   powers[maxIdx1], (nCluster > 1) ? powers[maxIdx2] : 0.0f);
            printf("  Final power distribution:\n");
            for (uint16_t i = 0; i < nCluster; i++) {
                printf("    Cluster %d: power=%.6f, delay=%.3f ns%s\n", 
                       i, powers[i], delays[i], 
                       (i == maxIdx1 || i == maxIdx2) ? " (strongest)" : "");
            }
        }
    }
#endif
    
    if (losInd && outdoor_ind) {
        // Apply Ricean K-factor for the outdoor LOS case (eq 7.5-8)
        // All clusters multiplied by 1/(K+1), then first cluster gets K/(K+1) added.
        // O2I links are excluded: Table 7.5-6 sets K = N/A for O2I, and Step 11 adds no
        // specular ray for them (consistent with the losInd && !isO2I gate there).
        float K_linear = powf(10.0f, K / 10.0f);
        float scaling = 1.0f / (1.0f + K_linear);
        
    #ifdef SLS_DEBUG_
        if (blockIdx.x == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
            printf("DEBUG: Applying Ricean K-factor: K_dB=%.3f, K_linear=%.6f, scaling=%.6f\n", 
                K, K_linear, scaling);
            printf("  Powers before K-factor: [");
            for (uint16_t i = 0; i < nCluster && i < 8; i++) {
                printf("%.6f", powers[i]);
                if (i < nCluster - 1 && i < 7) printf(", ");
            }
            if (nCluster > 8) printf(", ...");
            printf("]\n");
        }
    #endif
        
        for (uint16_t n = 1; n < nCluster; n++) {  // Skip first cluster (LOS)
            powers[n] *= scaling;
        }
        powers[0] = K_linear / (1.0f + K_linear) + powers[0] * scaling;
        
    #ifdef SLS_DEBUG_
        if (blockIdx.x == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
            printf("  Powers after K-factor: [");
            for (uint16_t i = 0; i < nCluster && i < 8; i++) {
                printf("%.6f", powers[i]);
                if (i < nCluster - 1 && i < 7) printf(", ");
            }
            if (nCluster > 8) printf(", ...");
            printf("]\n");
        }
    #endif
    }

#ifdef SLS_DEBUG_
    // Validate results for NaN/Inf values
    if (blockIdx.x == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
        bool hasNaN = false;
        for (uint16_t n = 0; n < nCluster; n++) {
            if (isnan(delays[n]) || isinf(delays[n])) {
                printf("DEBUG: Invalid delay[%d] = %f\n", n, delays[n]);
                hasNaN = true;
            }
            if (isnan(powers[n]) || isinf(powers[n])) {
                printf("DEBUG: Invalid power[%d] = %f\n", n, powers[n]);
                hasNaN = true;
            }
        }
        if (hasNaN) {
            printf("DEBUG: NaN/Inf detected in cluster generation!\n");
        }
    }
#endif
}

__device__ void genClusterAngleGPU(uint16_t nCluster, float C_ASA, float C_ASD,
                                  float C_phi_NLOS, float c_phi_O2I, float C_theta_NLOS,
                                  float C_phi_LOS, float C_theta_LOS, float C_theta_O2I,
                                  float ASA, float ASD, float ZSA, float ZSD,
                                  float phi_LOS_AOA, float phi_LOS_AOD,
                                  float theta_LOS_ZOA, float theta_LOS_ZOD,
                                  float mu_offset_ZOD, bool losInd,
                                  uint8_t outdoor_ind, float K,
                                  const float* powers, float* phi_n_AoA,
                                  float* phi_n_AoD, float* theta_n_ZOD,
                                  float* theta_n_ZOA, curandState* state) {
    
#ifdef SLS_DEBUG_
    // Debug input parameters (only from first thread to avoid spam)
    if (blockIdx.x == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
        printf("DEBUG: genClusterAngleGPU inputs:\n");
        printf("  nCluster=%d, losInd=%d, outdoor_ind=%d, K=%.3f\n", 
               nCluster, losInd, outdoor_ind, K);
        printf("  C_ASA=%.3f, C_ASD=%.3f, C_phi_NLOS=%.3f, c_phi_O2I=%.3f, C_theta_NLOS=%.3f\n", 
               C_ASA, C_ASD, C_phi_NLOS, c_phi_O2I, C_theta_NLOS);
        printf("  C_phi_LOS=%.3f, C_theta_LOS=%.3f, C_theta_O2I=%.3f\n", 
               C_phi_LOS, C_theta_LOS, C_theta_O2I);
        printf("  ASA=%.3f, ASD=%.3f, ZSA=%.3f, ZSD=%.3f\n", 
               ASA, ASD, ZSA, ZSD);
        printf("  LOS angles: phi_AOA=%.3f, phi_AOD=%.3f, theta_ZOA=%.3f, theta_ZOD=%.3f\n", 
               phi_LOS_AOA, phi_LOS_AOD, theta_LOS_ZOA, theta_LOS_ZOD);
        printf("  mu_offset_ZOD=%.3f\n", mu_offset_ZOD);
        
        // Show first few cluster powers
        printf("  Cluster powers: [");
        for (uint16_t i = 0; i < nCluster && i < 8; i++) {
            printf("%.6f", powers[i]);
            if (i < nCluster - 1 && i < 7) printf(", ");
        }
        if (nCluster > 8) printf(", ...");
        printf("]\n");
    }
#endif
    
    // Calculate C_phi and C_theta
    float C_phi, C_theta;
    if (outdoor_ind == 0) { // indoor UE, O2I
        C_phi = c_phi_O2I;
        C_theta = C_theta_O2I;
    } else {
        if (losInd) { // outdoor LOS
            const float scalingFactor_phi = (1.1035f - 0.028f * K - 0.002f * K * K + 0.0001f * K * K * K);
            const float scalingFactor_theta = (1.3086f + 0.0339f * K - 0.0077f * K * K + 0.0002f * K * K * K);
            C_phi = C_phi_LOS * scalingFactor_phi;
            C_theta = C_theta_LOS * scalingFactor_theta;
        } else { // outdoor NLOS
            C_phi = C_phi_NLOS;
            C_theta = C_theta_NLOS;
        }
    }

#ifdef SLS_DEBUG_
    if (blockIdx.x == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
        printf("  Calculated scaling factors: C_phi=%.6f, C_theta=%.6f\n", C_phi, C_theta);
        if (losInd && outdoor_ind) {
            printf("    LOS case: C_phi adjusted from %.6f using K=%.3f\n", C_phi_LOS, K);
            printf("    LOS case: C_theta adjusted from %.6f using K=%.3f\n", C_theta_LOS, K);
        } else {
            printf("    NLOS case: Using original C_phi=%.6f, C_theta=%.6f\n", C_phi_NLOS, C_theta_NLOS);
        }
    }
#endif

    // Find the maximum power of the clusters
    float max_p_n = powers[0];
    for (uint16_t n = 1; n < nCluster; n++) {
        if (powers[n] > max_p_n) {
            max_p_n = powers[n];
        }
    }

    // Safety check for max power
    if (max_p_n <= 0.0f || isnan(max_p_n)) {
#ifdef SLS_DEBUG_
        printf("ERROR: Invalid max_p_n detected: %f\n", max_p_n);
#endif
        return;
    }

#ifdef SLS_DEBUG_
    if (blockIdx.x == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
        printf("  Maximum cluster power: %.6f\n", max_p_n);
    }
#endif

    // Generate AOA (Azimuth of Arrival)
    for (uint16_t n = 0; n < nCluster; n++) {
        float Xn = (curand_uniform(state) < 0.5f) ? 1.0f : -1.0f;
        float Yn = ASA / 7.0f * curand_normal(state);
        
        // Safety check for power ratio to prevent log(0) and sqrt(negative)
        float power_ratio = powers[n] / max_p_n;
#ifdef SLS_DEBUG_
        if (power_ratio <= 0.0f) {
            printf("ERROR: Invalid power ratio: %f (powers[%d]=%f, max_p_n=%f)\n", 
                   power_ratio, n, powers[n], max_p_n);
            return;
        }
#endif
        power_ratio = fminf(power_ratio, 1.0f);   // Ensure ratio <= 1
        
        float log_term = -logf(power_ratio);
#ifdef SLS_DEBUG_
        if (isnan(log_term) || isinf(log_term) || log_term < 0.0f) {
            printf("ERROR: Invalid log term: %f (power_ratio=%f)\n", log_term, power_ratio);
            return;
        }
#endif
        
        float phi_prime_AOA = 2.0f * (ASA / 1.4f) * sqrtf(log_term) / fmaxf(C_phi, 1e-6f);
        phi_n_AoA[n] = Xn * phi_prime_AOA + Yn + phi_LOS_AOA;
        
#ifdef SLS_DEBUG_
        if (blockIdx.x == 0 && threadIdx.x == 0 && threadIdx.y == 0 && n < 5) {
            printf("  AOA Cluster %d: Xn=%.1f, Yn=%.3f, phi_prime=%.3f, phi_n_AoA=%.3f\n", 
                   n, Xn, Yn, phi_prime_AOA, phi_n_AoA[n]);
        }
#endif
    }
    
    if (losInd && outdoor_ind) {
        // eq 7.5-12: re-center so the FIRST cluster coincides with the LOS direction
        const float aoaShift = phi_n_AoA[0] - phi_LOS_AOA;
        for (uint16_t n = 0; n < nCluster; n++) {
            phi_n_AoA[n] -= aoaShift;
        }
#ifdef SLS_DEBUG_
        if (blockIdx.x == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
            printf("  LOS override: phi_n_AoA[0] set to %.3f\n", phi_LOS_AOA);
        }
#endif
    }
    
    // Generate AOD (Azimuth of Departure)
    for (uint16_t n = 0; n < nCluster; n++) {
        float Xn = (curand_uniform(state) < 0.5f) ? 1.0f : -1.0f;
        float Yn = ASD / 7.0f * curand_normal(state);
        
        // Safety check for power ratio to prevent log(0) and sqrt(negative)
        float power_ratio = powers[n] / max_p_n;
#ifdef SLS_DEBUG_
        if (power_ratio <= 0.0f) {
            printf("ERROR: Invalid power ratio: %f (powers[%d]=%f, max_p_n=%f)\n", 
                   power_ratio, n, powers[n], max_p_n);
            return;
        }
#endif
        power_ratio = fminf(power_ratio, 1.0f);   // Ensure ratio <= 1
        
        float log_term = -logf(power_ratio);
#ifdef SLS_DEBUG_
        if (isnan(log_term) || isinf(log_term) || log_term < 0.0f) {
            printf("ERROR: Invalid log term: %f (power_ratio=%f)\n", log_term, power_ratio);
            return;
        }
#endif
        
        float phi_prime_AOD = 2.0f * (ASD / 1.4f) * sqrtf(log_term) / fmaxf(C_phi, 1e-6f);
        phi_n_AoD[n] = Xn * phi_prime_AOD + Yn + phi_LOS_AOD;
        
#ifdef SLS_DEBUG_
        if (blockIdx.x == 0 && threadIdx.x == 0 && threadIdx.y == 0 && n < 5) {
            printf("  AOD Cluster %d: Xn=%.1f, Yn=%.3f, phi_prime=%.3f, phi_n_AoD=%.3f\n", 
                   n, Xn, Yn, phi_prime_AOD, phi_n_AoD[n]);
        }
#endif
    }
    
    if (losInd && outdoor_ind) {
        // eq 7.5-12: re-center so the FIRST cluster coincides with the LOS direction
        const float aodShift = phi_n_AoD[0] - phi_LOS_AOD;
        for (uint16_t n = 0; n < nCluster; n++) {
            phi_n_AoD[n] -= aodShift;
        }
#ifdef SLS_DEBUG_
        if (blockIdx.x == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
            printf("  LOS override: phi_n_AoD[0] set to %.3f\n", phi_LOS_AOD);
        }
#endif
    }
    
    // Generate ZOA (Zenith of Arrival)
    for (uint16_t n = 0; n < nCluster; n++) {
        float Xn = (curand_uniform(state) < 0.5f) ? 1.0f : -1.0f;
        float Yn = ZSA / 7.0f * curand_normal(state);
        
        // Safety check for power ratio to prevent log(0)
        float power_ratio = powers[n] / max_p_n;
        power_ratio = fmaxf(power_ratio, 1e-10f); // Prevent log(0)
        power_ratio = fminf(power_ratio, 1.0f);   // Ensure ratio <= 1
        
        float log_term = logf(power_ratio);
#ifdef SLS_DEBUG_
        if (isnan(log_term) || isinf(log_term)) {
            printf("ERROR: Invalid log term: %f (power_ratio=%f)\n", log_term, power_ratio);
            return;
        }
#endif
        
        float theta_prime_ZOA = -ZSA * log_term / fmaxf(C_theta, 1e-6f);
        float theta_bar_ZOA = outdoor_ind ? theta_LOS_ZOA : 90.0f;
        theta_n_ZOA[n] = Xn * theta_prime_ZOA + Yn + theta_bar_ZOA;
        
#ifdef SLS_DEBUG_
        if (blockIdx.x == 0 && threadIdx.x == 0 && threadIdx.y == 0 && n < 5) {
            printf("  ZOA Cluster %d: Xn=%.1f, Yn=%.3f, theta_prime=%.3f, theta_bar=%.3f, theta_n_ZOA=%.3f\n", 
                   n, Xn, Yn, theta_prime_ZOA, theta_bar_ZOA, theta_n_ZOA[n]);
        }
#endif
    }
    
    if (losInd && outdoor_ind) {
        // eq 7.5-17: re-center so the FIRST cluster coincides with the LOS direction
        const float zoaShift = theta_n_ZOA[0] - theta_LOS_ZOA;
        for (uint16_t n = 0; n < nCluster; n++) {
            theta_n_ZOA[n] -= zoaShift;
        }
#ifdef SLS_DEBUG_
        if (blockIdx.x == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
            printf("  LOS override: theta_n_ZOA[0] set to %.3f\n", theta_LOS_ZOA);
        }
#endif
    }
    
    // Generate ZOD (Zenith of Departure)
    for (uint16_t n = 0; n < nCluster; n++) {
        float Xn = (curand_uniform(state) < 0.5f) ? 1.0f : -1.0f;
        float Yn = ZSD / 7.0f * curand_normal(state);
        
        // Safety check for power ratio to prevent log(0)
        float power_ratio = powers[n] / max_p_n;
        power_ratio = fmaxf(power_ratio, 1e-10f); // Prevent log(0)
        power_ratio = fminf(power_ratio, 1.0f);   // Ensure ratio <= 1
        
        float log_term = logf(power_ratio);
#ifdef SLS_DEBUG_
        if (isnan(log_term) || isinf(log_term)) {
            printf("ERROR: Invalid log term: %f (power_ratio=%f)\n", log_term, power_ratio);
            return;
        }
#endif
        
        float theta_prime_ZOD = -ZSD * log_term / fmaxf(C_theta, 1e-6f);
        theta_n_ZOD[n] = Xn * theta_prime_ZOD + Yn + theta_LOS_ZOD + mu_offset_ZOD;
        
#ifdef SLS_DEBUG_
        if (blockIdx.x == 0 && threadIdx.x == 0 && threadIdx.y == 0 && n < 5) {
            printf("  ZOD Cluster %d: Xn=%.1f, Yn=%.3f, theta_prime=%.3f, theta_n_ZOD=%.3f\n", 
                   n, Xn, Yn, theta_prime_ZOD, theta_n_ZOD[n]);
        }
#endif
    }
    
    if (losInd && outdoor_ind) {
        // eq 7.5-19 with the 7.5-17 LOS substitution: anchor the FIRST cluster
        const float zodShift = theta_n_ZOD[0] - theta_LOS_ZOD - mu_offset_ZOD;
        for (uint16_t n = 0; n < nCluster; n++) {
            theta_n_ZOD[n] -= zodShift;
        }
#ifdef SLS_DEBUG_
        if (blockIdx.x == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
            printf("  LOS override: theta_n_ZOD[0] set to %.3f\n", theta_LOS_ZOD + mu_offset_ZOD);
        }
#endif
    }

#ifdef SLS_DEBUG_
    // Debug final angle arrays (summary)
    if (blockIdx.x == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
        printf("  Final cluster angles summary (first 5):\n");
        for (uint16_t n = 0; n < nCluster && n < 5; n++) {
            printf("    Cluster %d: AOA=%.2f°, AOD=%.2f°, ZOA=%.2f°, ZOD=%.2f°%s\n", 
                   n, phi_n_AoA[n], phi_n_AoD[n], theta_n_ZOA[n], theta_n_ZOD[n],
                   (losInd && outdoor_ind && n == 0) ? " (LOS)" : "");
        }
        if (nCluster > 5) {
            printf("    ... and %d more clusters\n", nCluster - 5);
        }
        printf("=== End genClusterAngleGPU ===\n\n");
    }
#endif
}

// Shuffle a 20-entry offset-index permutation within each 3GPP TR 38.901 Table 7.5-5
// sub-cluster ray set ({0-7,18,19}, {8-11,16,17}, {12-15}), so idx restricted to each set
// is a permutation of that set. Step 8 requires this for the two strongest clusters.
__device__ inline void shuffleWithinSubClustersGPU(uint8_t* idx, curandState* state)
{
    const uint8_t subRays[3][10] = {
        {0, 1, 2, 3, 4, 5, 6, 7, 18, 19},
        {8, 9, 10, 11, 16, 17, 0, 0, 0, 0},
        {12, 13, 14, 15, 0, 0, 0, 0, 0, 0}
    };
    const uint8_t subSizes[3] = {10, 6, 4};
    for (int s = 0; s < 3; s++) {
        // Fisher-Yates over the set's positions; curand_uniform returns (0, 1], so clamp
        // the scaled draw (uniformity loss ~2^-24), same convention as the full-cluster loop
        for (int i = subSizes[s] - 1; i > 0; i--) {
            uint8_t j = (uint8_t)(curand_uniform(state) * (i + 1));
            if (j > i) j = (uint8_t)i;
            uint8_t a = subRays[s][i];
            uint8_t b = subRays[s][j];
            uint8_t tmp = idx[a];
            idx[a] = idx[b];
            idx[b] = tmp;
        }
    }
}

__device__ void genRayAngleGPU(uint16_t nCluster, uint16_t nRay,
                              float ASA, float ASD, float ZSA, float ZSD,
                              const float* phi_n_AoA, const float* phi_n_AoD,
                              const float* theta_n_ZOA, const float* theta_n_ZOD,
                              float* phi_mn_AoA, float* phi_mn_AoD,
                              float* theta_mn_ZOA, float* theta_mn_ZOD,
                              float C_ASA, float C_ASD, float C_ZSA, float mu_lgZSD,
                              const uint16_t* strongest2clustersIdx,
                              curandState* state) {

#ifdef SLS_DEBUG_
    // Debug input parameters (only from first thread to avoid spam)
    if (blockIdx.x == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
        printf("DEBUG: genRayAngleGPU inputs:\n");
        printf("  nCluster=%d, nRay=%d\n", nCluster, nRay);
        printf("  ASA=%.3f, ASD=%.3f, ZSA=%.3f, ZSD=%.3f\n", ASA, ASD, ZSA, ZSD);
        printf("  Ray spread factors: C_ASA=%.3f, C_ASD=%.3f, C_ZSA=%.3f, mu_lgZSD=%.3f\n", 
               C_ASA, C_ASD, C_ZSA, mu_lgZSD);
    }
#endif
    
    // Standardized ray offset angles (3GPP specifications - constant for all scenarios)
    const float rayOffsets[20] = {
        0.0447f, -0.0447f, 0.1413f, -0.1413f, 0.2492f, -0.2492f, 0.3715f, -0.3715f,
        0.5129f, -0.5129f, 0.6797f, -0.6797f, 0.8844f, -0.8844f, 1.1481f, -1.1481f,
        1.5195f, -1.5195f, 2.1551f, -2.1551f
    };

#ifdef SLS_DEBUG_
    if (blockIdx.x == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
        printf("  Ray offset angles (const): [");
        for (int i = 0; i < 10; i++) {
            printf("%.4f", rayOffsets[i]);
            if (i < 9) printf(", ");
        }
        printf(", ...]\n");
    }
#endif
    
    // For each cluster
    for (uint16_t n = 0; n < nCluster; n++) {
        // Generate random permutations for each angle type (like MATLAB randperm)
        uint8_t idxASA[20], idxASD[20], idxZSA[20], idxZSD[20];
        
        // Initialize permutation arrays
        for (uint16_t i = 0; i < nRay; i++) {
            idxASA[i] = i;
            idxASD[i] = i;
            idxZSA[i] = i;
            idxZSD[i] = i;
        }
        
        // Apply Fisher-Yates shuffle for each angle type. Step 8: for the two strongest
        // clusters (which Step 11 splits into Table 7.5-5 delay sub-clusters by ray index)
        // the coupling must stay within each sub-cluster ray set; same equality test as
        // generateCIRKernel so coupling scope always matches the actual split.
        bool splitCluster = (nRay == 20) &&
                            (n == strongest2clustersIdx[0] || n == strongest2clustersIdx[1]);
        if (splitCluster) {
            shuffleWithinSubClustersGPU(idxASA, state);
            shuffleWithinSubClustersGPU(idxASD, state);
            shuffleWithinSubClustersGPU(idxZSA, state);
            shuffleWithinSubClustersGPU(idxZSD, state);
        } else {
        // curand_uniform returns (0, 1], so the scaled draw can hit i+1; clamp to i to stay
        // in-bounds (uniformity loss is ~2^-24)
        for (uint16_t i = nRay - 1; i > 0; i--) {
            // ASA permutation
            uint8_t j_ASA = (uint8_t)(curand_uniform(state) * (i + 1));
            if (j_ASA > i) j_ASA = i;
            uint8_t temp_ASA = idxASA[i];
            idxASA[i] = idxASA[j_ASA];
            idxASA[j_ASA] = temp_ASA;

            // ASD permutation
            uint8_t j_ASD = (uint8_t)(curand_uniform(state) * (i + 1));
            if (j_ASD > i) j_ASD = i;
            uint8_t temp_ASD = idxASD[i];
            idxASD[i] = idxASD[j_ASD];
            idxASD[j_ASD] = temp_ASD;

            // ZSA permutation
            uint8_t j_ZSA = (uint8_t)(curand_uniform(state) * (i + 1));
            if (j_ZSA > i) j_ZSA = i;
            uint8_t temp_ZSA = idxZSA[i];
            idxZSA[i] = idxZSA[j_ZSA];
            idxZSA[j_ZSA] = temp_ZSA;

            // ZSD permutation
            uint8_t j_ZSD = (uint8_t)(curand_uniform(state) * (i + 1));
            if (j_ZSD > i) j_ZSD = i;
            uint8_t temp_ZSD = idxZSD[i];
            idxZSD[i] = idxZSD[j_ZSD];
            idxZSD[j_ZSD] = temp_ZSD;
        }
        }
        
        // For each ray in the cluster
        for (uint16_t m = 0; m < nRay; m++) {
            uint16_t rayIdx = n * nRay + m;
            uint8_t offsetIdx_ASA = idxASA[m];
            uint8_t offsetIdx_ASD = idxASD[m];
            uint8_t offsetIdx_ZSA = idxZSA[m];
            uint8_t offsetIdx_ZSD = idxZSD[m];
            
            // Generate AOA (Azimuth of Arrival)
            phi_mn_AoA[rayIdx] = phi_n_AoA[n] + C_ASA * rayOffsets[offsetIdx_ASA];
            
            // Generate AOD (Azimuth of Departure)
            phi_mn_AoD[rayIdx] = phi_n_AoD[n] + C_ASD * rayOffsets[offsetIdx_ASD];
            
            // Generate ZOA (Zenith of Arrival) with angle wrapping
            float temp_ZOA = theta_n_ZOA[n] + C_ZSA * rayOffsets[offsetIdx_ZSA];
            // Normalize to [0°, 360°) range only if needed to avoid expensive fmodf
            if (temp_ZOA < 0.0f || temp_ZOA >= 360.0f) {
                temp_ZOA = fmodf(temp_ZOA, 360.0f);
                if (temp_ZOA < 0.0f) {
                    temp_ZOA += 360.0f;
                }
            }
            // Apply zenith angle reflection for [0°, 180°] range
            theta_mn_ZOA[rayIdx] = (temp_ZOA > 180.0f) ? 360.0f - temp_ZOA : temp_ZOA;
            
            // Generate ZOD (Zenith of Departure) following MATLAB reference: (3/8)*10^mu_lgZSD * RayOffsetAngles
            const float zsd_scale = (3.0f/8.0f) * powf(10.0f, mu_lgZSD);
            float temp_ZOD = theta_n_ZOD[n] + zsd_scale * rayOffsets[offsetIdx_ZSD];
            // Normalize to [0°, 360°) range only if needed to avoid expensive fmodf
            if (temp_ZOD < 0.0f || temp_ZOD >= 360.0f) {
                temp_ZOD = fmodf(temp_ZOD, 360.0f);
                if (temp_ZOD < 0.0f) {
                    temp_ZOD += 360.0f;
                }
            }
            // Apply zenith angle reflection for [0°, 180°] range
            theta_mn_ZOD[rayIdx] = (temp_ZOD > 180.0f) ? 360.0f - temp_ZOD : temp_ZOD;

#ifdef SLS_DEBUG_
            // Debug first few rays of first few clusters
            if (blockIdx.x == 0 && threadIdx.x == 0 && threadIdx.y == 0 && n < 3 && m < 3) {
                printf("  Ray[%d][%d] (idx=%d): offsets[ASA:%d,ASD:%d,ZSA:%d,ZSD:%d]=[%.4f,%.4f,%.4f,%.4f] -> AOA=%.3f°, AOD=%.3f°, ZOA=%.3f°, ZOD=%.3f°\n", 
                       n, m, rayIdx, offsetIdx_ASA, offsetIdx_ASD, offsetIdx_ZSA, offsetIdx_ZSD,
                       rayOffsets[offsetIdx_ASA], rayOffsets[offsetIdx_ASD], rayOffsets[offsetIdx_ZSA], rayOffsets[offsetIdx_ZSD],
                       phi_mn_AoA[rayIdx], phi_mn_AoD[rayIdx], 
                       theta_mn_ZOA[rayIdx], theta_mn_ZOD[rayIdx]);
            }
#endif
        }
    }

#ifdef SLS_DEBUG_
    // Debug summary
    if (blockIdx.x == 0 && threadIdx.x == 0 && threadIdx.y == 0) {
        printf("  Generated %d total rays (%d clusters x %d rays)\n", nCluster * nRay, nCluster, nRay);
        printf("=== End genRayAngleGPU ===\n\n");
    }
#endif
}

// Explicit template instantiations
template class slsChan<float, float2>;
