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

#include "chanModelsApi.hpp"
#include "sls_chan_src/sls_chan.cuh"
#include "tdl_chan_src/tdl_chan.cuh"
#include "cdl_chan_src/cdl_chan.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>  // For __half and __half2
#include <memory>
#include <vector>
#include <random>


template <typename Tscalar, typename Tcomplex>
statisChanModel<Tscalar, Tcomplex>::statisChanModel(
    const SimConfig* sim_config,
    const SystemLevelConfig* system_level_config,
    const LinkLevelConfig* link_level_config,
    const ExternalConfig* external_config,
    uint32_t randSeed,
    cudaStream_t strm) {
    
    // Direct pointer assignment
    m_sim_config = sim_config;
    m_system_level_config = system_level_config;
    m_link_level_config = link_level_config;
    m_external_config = external_config;
    m_rand_seed = randSeed;

    // Handle CUDA stream
    if (strm == nullptr) {
        // Create a new stream if none provided
        cudaError_t err = cudaStreamCreate(&m_strm);
        if (err != cudaSuccess) {
            throw ChannelModelError("Failed to create CUDA stream: " + std::string(cudaGetErrorString(err)));
        }
        m_owns_stream = true;
    } else {
        // Use provided stream
        m_strm = strm;
        m_owns_stream = false;
    }

    if (m_sim_config->link_sim_ind == 0) {
        // Initialize system level channel models based on configuration
        m_sls_chan = std::make_unique<slsChan<Tscalar, Tcomplex>>(
            m_sim_config, 
            m_system_level_config, 
            m_external_config, 
            m_rand_seed, 
            m_strm);
        // No need to call setup() separately since we pass external_config to constructor
    }
    else {
        // TODO: add full support for TDL/CDL channel models later
        if (m_link_level_config->fast_fading_type == 1) {  // TDL
            m_tdl_chan_cfg = std::make_unique<tdlConfig_t>();
            
            // Populate TDL channel configuration parameters
            // Basic TDL parameters from link_level_config
            m_tdl_chan_cfg->useSimplifiedPdp = true;  // Default to simplified PDP
            m_tdl_chan_cfg->delayProfile = m_link_level_config->delay_profile;
            m_tdl_chan_cfg->delaySpread = m_link_level_config->delay_spread;
            m_tdl_chan_cfg->maxDopplerShift = std::sqrt(
                m_link_level_config->velocity[0] * m_link_level_config->velocity[0] +
                m_link_level_config->velocity[1] * m_link_level_config->velocity[1] +
                m_link_level_config->velocity[2] * m_link_level_config->velocity[2]
            ) * m_sim_config->center_freq_hz / 3e8;  // Calculate max Doppler from velocity
            
            // Simulation parameters from sim_config
            if (m_sim_config->fft_size <= 0 || m_sim_config->sc_spacing_hz <= 0.0f) {
                throw ChannelModelError(
                    "Invalid sampling config for TDL: fft_size and sc_spacing_hz must be positive. "
                    "Got fft_size=" + std::to_string(m_sim_config->fft_size) +
                    ", sc_spacing_hz=" + std::to_string(m_sim_config->sc_spacing_hz));
            }
            m_tdl_chan_cfg->f_samp = m_sim_config->fft_size * m_sim_config->sc_spacing_hz;  // Sampling frequency = N_FFT * SCS
            m_tdl_chan_cfg->nCell = 1;  // Default for link-level simulation
            m_tdl_chan_cfg->nUe = 1;    // Default for link-level simulation
            
            // Antenna configuration from external_config
            // Initialize with default values first
            m_tdl_chan_cfg->nBsAnt = 4;  // Default BS antenna count
            m_tdl_chan_cfg->nUeAnt = 4;  // Default UE antenna count
            
            // Overwrite with actual values if available
            if (m_external_config->ant_panel_config.size() > 0) {
                m_tdl_chan_cfg->nBsAnt = m_external_config->ant_panel_config[0].nAnt;
            }
            if (m_external_config->ant_panel_config.size() > 1) {
                m_tdl_chan_cfg->nUeAnt = m_external_config->ant_panel_config[1].nAnt;
            }
            
            // Channel update and processing parameters
            m_tdl_chan_cfg->fBatch = 15e3;  // Update rate
            m_tdl_chan_cfg->numPath = m_link_level_config->num_ray;
            m_tdl_chan_cfg->cfoHz = m_link_level_config->cfo_hz;
            m_tdl_chan_cfg->delay = m_link_level_config->delay;
            
            // Signal processing parameters
            m_tdl_chan_cfg->sigLenPerAnt = m_sim_config->fft_size;  // Use FFT size as signal length, can be set to different value if needed
            m_tdl_chan_cfg->N_sc = m_sim_config->n_prb * 12;  // Total subcarriers
            m_tdl_chan_cfg->N_sc_Prbg = m_sim_config->n_prbg * 12;  // Subcarriers per PRB group
            m_tdl_chan_cfg->scSpacingHz = m_sim_config->sc_spacing_hz;
            m_tdl_chan_cfg->freqConvertType = static_cast<uint8_t>(m_sim_config->freq_convert_type);
            m_tdl_chan_cfg->scSampling = static_cast<uint8_t>(m_sim_config->sc_sampling);
            m_tdl_chan_cfg->runMode = static_cast<uint8_t>(m_sim_config->run_mode);
            m_tdl_chan_cfg->procSigFreq = static_cast<uint8_t>(m_sim_config->proc_sig_freq);
            m_tdl_chan_cfg->saveAntPairSample = 0;  // Default disabled
            
            // Initialize batch length vector - empty means use fBatch
            if (m_sim_config->n_snapshot_per_slot == 1) {
                m_tdl_chan_cfg->batchLen = {1};
            } else { // assuming 14 OFDM symbols per slot, mu=1, N_FFT=4096; First CP has 352 samples, other CP has 288 samples
                m_tdl_chan_cfg->batchLen = {352+4096, 288+4096, 288+4096, 288+4096, 288+4096, 288+4096, 288+4096, 288+4096, 288+4096, 288+4096, 288+4096, 288+4096, 288+4096, 288+4096};
            }
            m_tdl_chan_cfg->txSigIn = m_sim_config->tx_sig_in;  // Will be set when needed
            
            // Legacy TDL-specific parameters - these are not actually part of tdlConfig_t
            // but are internal to the TDL implementation
            
            m_tdl_chan = std::make_unique<tdlChan<Tscalar, Tcomplex>>(m_tdl_chan_cfg.get(), m_rand_seed, m_strm);
        } 
        else if (m_link_level_config->fast_fading_type == 2) {  // CDL
            m_cdl_chan_cfg = std::make_unique<cdlConfig_t>();
            
            // Populate CDL channel configuration parameters
            // Basic CDL parameters from link_level_config
            m_cdl_chan_cfg->delayProfile = m_link_level_config->delay_profile;
            m_cdl_chan_cfg->delaySpread = m_link_level_config->delay_spread;
            m_cdl_chan_cfg->maxDopplerShift = std::sqrt(
                m_link_level_config->velocity[0] * m_link_level_config->velocity[0] +
                m_link_level_config->velocity[1] * m_link_level_config->velocity[1] +
                m_link_level_config->velocity[2] * m_link_level_config->velocity[2]
            ) * m_sim_config->center_freq_hz / 3e8;  // Calculate max Doppler from velocity
            
            // Simulation parameters from sim_config
            if (m_sim_config->fft_size <= 0 || m_sim_config->sc_spacing_hz <= 0.0f) {
                throw ChannelModelError(
                    "Invalid sampling config for CDL: fft_size and sc_spacing_hz must be positive. "
                    "Got fft_size=" + std::to_string(m_sim_config->fft_size) +
                    ", sc_spacing_hz=" + std::to_string(m_sim_config->sc_spacing_hz));
            }
            m_cdl_chan_cfg->f_samp = m_sim_config->fft_size * m_sim_config->sc_spacing_hz;  // Sampling frequency = N_FFT * SCS
            m_cdl_chan_cfg->nCell = 1;  // Default for link-level simulation
            m_cdl_chan_cfg->nUe = 1;    // Default for link-level simulation
            
            // Antenna configuration from external_config
            if (m_external_config->ant_panel_config.size() >= 2) {
                const auto& bs_panel = m_external_config->ant_panel_config[0];
                const auto& ue_panel = m_external_config->ant_panel_config[1];

                // BS Antenna Configuration
                if (bs_panel.antSize[0] * bs_panel.antSize[1] * bs_panel.antSize[2] * 
                    bs_panel.antSize[3] * bs_panel.antSize[4] == bs_panel.nAnt) {
                    // Use proper vector assignment for bsAntSize
                    m_cdl_chan_cfg->bsAntSize.assign({
                        static_cast<uint16_t>(bs_panel.antSize[0]),  // M_g
                        static_cast<uint16_t>(bs_panel.antSize[1]),  // N_g
                        static_cast<uint16_t>(bs_panel.antSize[2]),  // M
                        static_cast<uint16_t>(bs_panel.antSize[3]),  // N
                        static_cast<uint16_t>(bs_panel.antSize[4])   // P
                    });

                    // Use proper vector assignment for bsAntSpacing
                    m_cdl_chan_cfg->bsAntSpacing.assign({
                        bs_panel.antSpacing[0],
                        bs_panel.antSpacing[1],
                        bs_panel.antSpacing[2],
                        bs_panel.antSpacing[3]
                    });

                    // Use proper vector assignment for bsAntPolarAngles
                    m_cdl_chan_cfg->bsAntPolarAngles.assign({
                        static_cast<float>(bs_panel.antPolarAngles[0]),
                        static_cast<float>(bs_panel.antPolarAngles[1])
                    });

                    m_cdl_chan_cfg->bsAntPattern = static_cast<uint8_t>(bs_panel.antModel);
                }

                // UE Antenna Configuration
                if (ue_panel.antSize[0] * ue_panel.antSize[1] * ue_panel.antSize[2] * 
                    ue_panel.antSize[3] * ue_panel.antSize[4] == ue_panel.nAnt) {
                    // Use proper vector assignment for ueAntSize
                    m_cdl_chan_cfg->ueAntSize.assign({
                        static_cast<uint16_t>(ue_panel.antSize[0]),  // M_g
                        static_cast<uint16_t>(ue_panel.antSize[1]),  // N_g
                        static_cast<uint16_t>(ue_panel.antSize[2]),  // M
                        static_cast<uint16_t>(ue_panel.antSize[3]),  // N
                        static_cast<uint16_t>(ue_panel.antSize[4])   // P
                    });

                    // Use proper vector assignment for ueAntSpacing
                    m_cdl_chan_cfg->ueAntSpacing.assign({
                        ue_panel.antSpacing[0],
                        ue_panel.antSpacing[1],
                        ue_panel.antSpacing[2],
                        ue_panel.antSpacing[3]
                    });

                    // Use proper vector assignment for ueAntPolarAngles
                    m_cdl_chan_cfg->ueAntPolarAngles.assign({
                        static_cast<float>(ue_panel.antPolarAngles[0]),
                        static_cast<float>(ue_panel.antPolarAngles[1])
                    });

                    m_cdl_chan_cfg->ueAntPattern = static_cast<uint8_t>(ue_panel.antModel);
                }
            } else {
                // Default BS antenna configuration
                m_cdl_chan_cfg->bsAntSize.assign({1, 1, 1, 2, 2});
                m_cdl_chan_cfg->bsAntSpacing.assign({1.0f, 1.0f, 0.5f, 0.5f});
                m_cdl_chan_cfg->bsAntPolarAngles.assign({45.0f, -45.0f});
                m_cdl_chan_cfg->bsAntPattern = 1;
                
                // Default UE antenna configuration
                m_cdl_chan_cfg->ueAntSize.assign({1, 1, 2, 2, 1});
                m_cdl_chan_cfg->ueAntSpacing.assign({1.0f, 1.0f, 0.5f, 0.5f});
                m_cdl_chan_cfg->ueAntPolarAngles.assign({0.0f, 90.0f});
                m_cdl_chan_cfg->ueAntPattern = 0;
            }
            
            // Movement direction - map from velocity vector
            float velocity_magnitude = std::sqrt(
                m_link_level_config->velocity[0] * m_link_level_config->velocity[0] +
                m_link_level_config->velocity[1] * m_link_level_config->velocity[1]
            );
            if (velocity_magnitude > 0.0f) {
                float azimuth = std::atan2(m_link_level_config->velocity[1], 
                                         m_link_level_config->velocity[0]) * 180.0f / M_PI;
                if (azimuth < 0.0f) azimuth += 360.0f;
                m_cdl_chan_cfg->vDirection = {azimuth, 0.0f};  // Azimuth angle, zenith = 0
            } else {
                m_cdl_chan_cfg->vDirection = {90.0f, 0.0f};  // Default moving direction
            }
            
            // Channel update and processing parameters
            m_cdl_chan_cfg->fBatch = 15e3;  // Update rate
            m_cdl_chan_cfg->numRay = m_link_level_config->num_ray;
            m_cdl_chan_cfg->cfoHz = m_link_level_config->cfo_hz;
            m_cdl_chan_cfg->delay = m_link_level_config->delay;
            
            // Signal processing parameters
            m_cdl_chan_cfg->sigLenPerAnt = m_sim_config->fft_size;  // Use FFT size as signal length, can be set to different value if needed
            m_cdl_chan_cfg->N_sc = m_sim_config->n_prb * 12;  // Total subcarriers
            m_cdl_chan_cfg->N_sc_Prbg = m_sim_config->n_prbg * 12;  // Subcarriers per PRB group
            m_cdl_chan_cfg->scSpacingHz = m_sim_config->sc_spacing_hz;
            m_cdl_chan_cfg->freqConvertType = static_cast<uint8_t>(m_sim_config->freq_convert_type);
            m_cdl_chan_cfg->scSampling = static_cast<uint8_t>(m_sim_config->sc_sampling);
            m_cdl_chan_cfg->runMode = static_cast<uint8_t>(m_sim_config->run_mode);
            m_cdl_chan_cfg->procSigFreq = static_cast<uint8_t>(m_sim_config->proc_sig_freq);
            m_cdl_chan_cfg->saveAntPairSample = 0;  // Default disabled
            
            // Initialize batch length vector - empty means use fBatch
            if (m_sim_config->n_snapshot_per_slot == 1) {
                m_cdl_chan_cfg->batchLen = {1};
            } else { // assuming 14 OFDM symbols per slot, mu=1, N_FFT=4096; First CP has 352 samples, other CP has 288 samples
                m_cdl_chan_cfg->batchLen = {352+4096, 288+4096, 288+4096, 288+4096, 288+4096, 288+4096, 288+4096, 288+4096, 288+4096, 288+4096, 288+4096, 288+4096, 288+4096, 288+4096};
            }
            m_cdl_chan_cfg->txSigIn = m_sim_config->tx_sig_in;  // Will be set when needed
            
            m_cdl_chan = std::make_unique<cdlChan<Tscalar, Tcomplex>>(m_cdl_chan_cfg.get(), m_rand_seed, m_strm);
        }
        else {
            throw ChannelModelError("Invalid fast fading type: " + std::to_string(m_link_level_config->fast_fading_type) + 
                                  ". Expected 1 (TDL) or 2 (CDL).");
        }
    }
}

template <typename Tscalar, typename Tcomplex>
void statisChanModel<Tscalar, Tcomplex>::run(
    const float refTime,
    const uint8_t continuous_fading,
    const std::vector<uint16_t>& activeCell,
    const std::vector<std::vector<uint16_t>>& activeUt,
    const std::vector<Coordinate>& utNewLoc,
    const std::vector<float3>& utNewVelocity,
    const std::vector<Tcomplex*>& cir_coe,
    const std::vector<uint16_t*>& cir_norm_delay,
    const std::vector<uint16_t*>& cir_n_taps,
    const std::vector<Tcomplex*>& cfr_sc,
    const std::vector<Tcomplex*>& cfr_prbg) {
    
    if (m_sim_config->link_sim_ind == 0) {  // System-level simulation
        // Pass per-cell vectors directly to SLS channel model
        m_sls_chan->run(refTime, continuous_fading, activeCell, activeUt, utNewLoc, utNewVelocity, 
                       cir_coe, cir_norm_delay, cir_n_taps, cfr_sc, cfr_prbg);
        
    } else if (m_sim_config->link_sim_ind == 1) {  // Link-level simulation
        if (m_link_level_config->fast_fading_type == 1) {  // TDL
            m_tdl_chan->run(0.0f, 0.0f, 0);
            // Get TDL channel responses
            // Implementation depends on your data structure
        } else if (m_link_level_config->fast_fading_type == 2) {  // CDL
            m_cdl_chan->run(0.0f, 0.0f, 0);
            // Get CDL channel responses
            // Implementation depends on your data structure
        }
        else {
            throw ChannelModelError("Invalid fast fading type: " + std::to_string(m_link_level_config->fast_fading_type) + 
                                  ". Expected 1 (TDL) or 2 (CDL).");
        }
    }
    else {
        throw ChannelModelError("Invalid link simulation indicator: " + std::to_string(m_sim_config->link_sim_ind) + 
                              ". Expected 0 (System-level) or 1 (Link-level).");
    }
}

template <typename Tscalar, typename Tcomplex>
void statisChanModel<Tscalar, Tcomplex>::run(
    const float refTime0,
    const uint8_t continuous_fading,
    const uint8_t enableSwapTxRx,
    const uint8_t txColumnMajorInd) {
    
    // Link-level simulation
    // Note: continuous_fading parameter is handled at the statisChanModel level
    // but underlying TDL/CDL channels don't support this parameter directly
    if (m_link_level_config->fast_fading_type == 1) {  // TDL
        if (m_tdl_chan) {
            m_tdl_chan->run(refTime0, enableSwapTxRx, txColumnMajorInd);
        }
    } else if (m_link_level_config->fast_fading_type == 2) {  // CDL
        if (m_cdl_chan) {
            m_cdl_chan->run(refTime0, enableSwapTxRx, txColumnMajorInd);
        }
    }
}

template <typename Tscalar, typename Tcomplex>
void statisChanModel<Tscalar, Tcomplex>::reset() {
    if (m_tdl_chan) m_tdl_chan->reset();
    if (m_cdl_chan) m_cdl_chan->reset();
    if (m_sls_chan) m_sls_chan->reset();
}

template <typename Tscalar, typename Tcomplex>
void statisChanModel<Tscalar, Tcomplex>::dump_los_nlos_stats(float* lost_nlos_stats) {
    if (m_sls_chan && lost_nlos_stats) {
        m_sls_chan->dump_los_nlos_stats(lost_nlos_stats);
    }
}

template <typename Tscalar, typename Tcomplex>
void statisChanModel<Tscalar, Tcomplex>::dump_pl_sf_stats(
    float* pl_sf,
    const std::vector<uint16_t>& activeCell,
    const std::vector<uint16_t>& activeUt) {
    if (m_sls_chan && pl_sf) {
        m_sls_chan->dump_pl_sf_stats(pl_sf, activeCell, activeUt);
    }
}

template <typename Tscalar, typename Tcomplex>
void statisChanModel<Tscalar, Tcomplex>::dump_pl_sf_ant_gain_stats(
    float* pl_sf_ant_gain,
    const std::vector<uint16_t>& activeCell,
    const std::vector<uint16_t>& activeUt) {
    if (m_sls_chan && pl_sf_ant_gain) {
        m_sls_chan->dump_pl_sf_ant_gain_stats(pl_sf_ant_gain, activeCell, activeUt);
    }
}

template <typename Tscalar, typename Tcomplex>
void statisChanModel<Tscalar, Tcomplex>::dump_topology_to_yaml(const std::string& filename) {
    if (m_sls_chan) {
        m_sls_chan->dumpTopologyToYaml(filename);
    }
}

template <typename Tscalar, typename Tcomplex>
void statisChanModel<Tscalar, Tcomplex>::saveSlsChanToH5File(std::string_view filenameEnding) {
    if (m_sls_chan) {
        m_sls_chan->saveSlsChanToH5File(filenameEnding);
    }
}

// Explicit template instantiations are in the header file 