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

#include "gpu3gppchanApi.hpp"
#include "sls_chan_src/sls_chan.cuh"
#include "tdl_chan_src/tdl_chan.cuh"
#include "cdl_chan_src/cdl_chan.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>  // For __half and __half2
#include <limits>
#include <memory>
#include <vector>
#include <random>

namespace {

constexpr int64_t K_SUBCARRIERS_PER_PRB{12};

class CuStreamGuard final {
public:
    CuStreamGuard() = default;
    CuStreamGuard(const CuStreamGuard&) = delete;
    CuStreamGuard& operator=(const CuStreamGuard&) = delete;

    ~CuStreamGuard()
    {
        if (stream_ != nullptr)
        {
            (void)cuStreamDestroy(stream_);
        }
    }

    void reset(CUstream stream) noexcept { stream_ = stream; }
    void release() noexcept { stream_ = nullptr; }

private:
    CUstream stream_{};
};

class CuPrimaryContextGuard final {
public:
    CuPrimaryContextGuard() = default;
    CuPrimaryContextGuard(const CuPrimaryContextGuard&) = delete;
    CuPrimaryContextGuard& operator=(const CuPrimaryContextGuard&) = delete;

    ~CuPrimaryContextGuard()
    {
        if (retained_)
        {
            CUcontext current_context{};
            if (cuCtxGetCurrent(&current_context) == CUDA_SUCCESS &&
                current_context == context_)
            {
                (void)cuCtxSetCurrent(nullptr);
            }
            (void)cuDevicePrimaryCtxRelease(device_);
        }
    }

    void reset(CUdevice device, CUcontext context) noexcept
    {
        device_ = device;
        context_ = context;
        retained_ = true;
    }
    void release() noexcept { retained_ = false; }

private:
    CUdevice device_{};
    CUcontext context_{};
    bool retained_{};
};

void throwOnCuError(CUresult status, const char* operation)
{
    if (status == CUDA_SUCCESS)
    {
        return;
    }
    const char* error_string = nullptr;
    (void)cuGetErrorString(status, &error_string);
    throw ChannelModelError(
        std::string(operation) + ": " +
        (error_string != nullptr ? error_string : "unknown CUDA error"));
}

} // namespace


template <typename Tscalar, typename Tcomplex>
statisChanModel<Tscalar, Tcomplex>::statisChanModel(
    const SimConfig* sim_config,
    const SystemLevelConfig* system_level_config,
    const LinkLevelConfig* link_level_config,
    const ExternalConfig* external_config,
    uint32_t randSeed,
    cudaStream_t strm) {
    CuPrimaryContextGuard primary_context_guard;
    // Declare the stream guard after the context guard so constructor-failure
    // unwinding destroys a context-owned stream before releasing that context.
    CuStreamGuard owned_stream_guard;

    if (sim_config == nullptr) {
        throw ChannelModelError("sim_config must not be null");
    }
    if (sim_config->link_sim_ind != 0 && sim_config->link_sim_ind != 1) {
        throw ChannelModelError("link_sim_ind must be 0 (system-level) or 1 (link-level)");
    }
    if (sim_config->link_sim_ind == 0 && system_level_config == nullptr) {
        throw ChannelModelError("system_level_config is required for system-level simulation");
    }
    if (sim_config->link_sim_ind == 1 && link_level_config == nullptr) {
        throw ChannelModelError("link_level_config is required for link-level simulation");
    }
    if (sim_config->cpu_only_mode < 0 || sim_config->cpu_only_mode > 1) {
        throw ChannelModelError("cpu_only_mode must be 0 or 1");
    }
    if (sim_config->internal_memory_mode < 0 || sim_config->internal_memory_mode > 2) {
        throw ChannelModelError("internal_memory_mode must be in [0, 2]");
    }
    if (sim_config->run_mode < 0 || sim_config->run_mode > 4) {
        throw ChannelModelError("run_mode must be in [0, 4]");
    }
    if (sim_config->freq_convert_type < 0 || sim_config->freq_convert_type > 4) {
        throw ChannelModelError("freq_convert_type must be in [0, 4]");
    }
    if (sim_config->n_snapshot_per_slot <= 0 || sim_config->n_prb <= 0 ||
        sim_config->n_prbg <= 0 || sim_config->fft_size <= 0 ||
        sim_config->sc_sampling <= 0) {
        throw ChannelModelError(
            "n_snapshot_per_slot, n_prb, n_prbg, fft_size, and sc_sampling must be positive");
    }
    const int64_t total_subcarriers =
        static_cast<int64_t>(sim_config->n_prb) * K_SUBCARRIERS_PER_PRB;
    if (total_subcarriers > std::numeric_limits<uint16_t>::max()) {
        throw ChannelModelError("n_prb produces more subcarriers than the channel supports");
    }
    if (sim_config->n_prbg > sim_config->n_prb) {
        throw ChannelModelError("n_prbg must be in [1, n_prb]");
    }
    
    // Direct pointer assignment
    m_sim_config = sim_config;
    m_system_level_config = system_level_config;
    m_link_level_config = link_level_config;
    m_external_config = external_config;
    m_rand_seed = randSeed;

    // A CPU-only system-level model must not initialize CUDA as a side effect.
    const bool cudaRequired = sim_config->link_sim_ind == 1 || sim_config->cpu_only_mode == 0;

    // Handle CUDA stream
    if (!cudaRequired) {
        m_strm = nullptr;
        m_owns_stream = false;
    } else if (strm == nullptr) {
        // Unlike the Runtime API, cuStreamCreate does not initialize a context.
        // Make the device-0 primary context current when the caller did not
        // already supply one, and retain it for the model lifetime.
        throwOnCuError(cuInit(0), "Failed to initialize the CUDA Driver API");
        CUcontext current_context{};
        throwOnCuError(cuCtxGetCurrent(&current_context), "Failed to query the current CUDA context");
        if (current_context == nullptr) {
            CUdevice device{};
            throwOnCuError(cuDeviceGet(&device, 0), "Failed to get CUDA device 0");
            throwOnCuError(
                cuDevicePrimaryCtxRetain(&current_context, device),
                "Failed to retain the CUDA primary context");
            primary_context_guard.reset(device, current_context);
            throwOnCuError(
                cuCtxSetCurrent(current_context),
                "Failed to make the CUDA primary context current");
            m_owned_primary_context = current_context;
            m_retained_primary_device = device;
            m_owns_primary_context = true;
        }

        // Preserve legacy-default-stream ordering for callers that do not
        // provide a stream. This simulator is not a deadline-path component,
        // so it must not silently opt callers into a spinning wait policy.
        CUstream driver_stream{};
        const CUresult status = cuStreamCreate(&driver_stream, CU_STREAM_DEFAULT);
        throwOnCuError(status, "Failed to create CUDA stream");
        owned_stream_guard.reset(driver_stream);
        m_strm = driver_stream;
        m_owns_stream = true;
    } else {
        // Driver-API allocations and launches still require an initialized,
        // current context even when the stream came from the Runtime API.
        throwOnCuError(cuInit(0), "Failed to initialize the CUDA Driver API");
        CUcontext current_context{};
        throwOnCuError(cuCtxGetCurrent(&current_context), "Failed to query the current CUDA context");
        if (current_context == nullptr) {
            throw ChannelModelError(
                "A caller-provided CUDA stream requires a current CUDA context on the constructing thread");
        }
        m_strm = strm;
        m_owns_stream = false;
    }

    if (m_sim_config->link_sim_ind == 0) {
        // Initialize system level channel models based on configuration
        m_sls_chan.reset(createSlsChan<Tscalar, Tcomplex>(
            m_sim_config, 
            m_system_level_config, 
            m_external_config, 
            m_rand_seed, 
            m_strm));
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
            if (m_external_config != nullptr &&
                !m_external_config->ant_panel_config.empty()) {
                m_tdl_chan_cfg->nBsAnt = m_external_config->ant_panel_config[0].nAnt;
            }
            if (m_external_config != nullptr &&
                m_external_config->ant_panel_config.size() > 1) {
                m_tdl_chan_cfg->nUeAnt = m_external_config->ant_panel_config[1].nAnt;
            }
            
            // Channel update and processing parameters
            m_tdl_chan_cfg->fBatch = 15e3;  // Update rate
            m_tdl_chan_cfg->numPath = m_link_level_config->num_ray;
            m_tdl_chan_cfg->cfoHz = m_link_level_config->cfo_hz;
            m_tdl_chan_cfg->delay = m_link_level_config->delay;
            
            // Signal processing parameters
            m_tdl_chan_cfg->sigLenPerAnt = m_sim_config->fft_size;  // Use FFT size as signal length, can be set to different value if needed
            m_tdl_chan_cfg->N_sc = static_cast<uint16_t>(
                static_cast<int64_t>(m_sim_config->n_prb) * K_SUBCARRIERS_PER_PRB);
            m_tdl_chan_cfg->N_sc_Prbg = static_cast<uint16_t>(
                static_cast<int64_t>(m_sim_config->n_prbg) * K_SUBCARRIERS_PER_PRB);
            m_tdl_chan_cfg->scSpacingHz = m_sim_config->sc_spacing_hz;
            m_tdl_chan_cfg->freqConvertType = static_cast<uint8_t>(m_sim_config->freq_convert_type);
            m_tdl_chan_cfg->scSampling = static_cast<uint8_t>(m_sim_config->sc_sampling);
            m_tdl_chan_cfg->runMode = static_cast<uint8_t>(m_sim_config->run_mode);
            m_tdl_chan_cfg->procSigFreq = static_cast<uint8_t>(m_sim_config->proc_sig_freq);
            m_tdl_chan_cfg->saveAntPairSample = 0;  // Default disabled
            
            // Initialize batch length vector - empty means use fBatch
            if (m_sim_config->n_snapshot_per_slot == 1) {
                m_tdl_chan_cfg->batchLen.assign(1, 1U);
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
            if (m_external_config != nullptr &&
                m_external_config->ant_panel_config.size() >= 2) {
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
            m_cdl_chan_cfg->N_sc = static_cast<uint16_t>(
                static_cast<int64_t>(m_sim_config->n_prb) * K_SUBCARRIERS_PER_PRB);
            m_cdl_chan_cfg->N_sc_Prbg = static_cast<uint16_t>(
                static_cast<int64_t>(m_sim_config->n_prbg) * K_SUBCARRIERS_PER_PRB);
            m_cdl_chan_cfg->scSpacingHz = m_sim_config->sc_spacing_hz;
            m_cdl_chan_cfg->freqConvertType = static_cast<uint8_t>(m_sim_config->freq_convert_type);
            m_cdl_chan_cfg->scSampling = static_cast<uint8_t>(m_sim_config->sc_sampling);
            m_cdl_chan_cfg->runMode = static_cast<uint8_t>(m_sim_config->run_mode);
            m_cdl_chan_cfg->procSigFreq = static_cast<uint8_t>(m_sim_config->proc_sig_freq);
            m_cdl_chan_cfg->saveAntPairSample = 0;  // Default disabled
            
            // Initialize batch length vector - empty means use fBatch
            if (m_sim_config->n_snapshot_per_slot == 1) {
                m_cdl_chan_cfg->batchLen.assign(1, 1U);
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

    // Construction succeeded; the model destructor now owns this stream.
    owned_stream_guard.release();
    primary_context_guard.release();
}

template <typename Tscalar, typename Tcomplex>
statisChanModel<Tscalar, Tcomplex>::~statisChanModel()
{
    // Channel implementations may use the stream while releasing their CUDA
    // allocations, so destroy them before destroying an internally-owned stream.
    m_sls_chan.reset();
    m_tdl_chan.reset();
    m_cdl_chan.reset();

    if (m_owns_stream && m_strm != nullptr)
    {
        const CUresult status = cuStreamDestroy(m_strm);
        if (status != CUDA_SUCCESS)
        {
            const char* error_string = nullptr;
            cuGetErrorString(status, &error_string);
            fprintf(stderr,
                    "Warning: statisChanModel cuStreamDestroy failed: %s\n",
                    error_string != nullptr ? error_string : "unknown CUDA error");
        }
    }
    if (m_owns_primary_context)
    {
        CUcontext current_context{};
        if (cuCtxGetCurrent(&current_context) == CUDA_SUCCESS &&
            current_context == m_owned_primary_context)
        {
            (void)cuCtxSetCurrent(nullptr);
        }
        const CUresult status = cuDevicePrimaryCtxRelease(m_retained_primary_device);
        if (status != CUDA_SUCCESS)
        {
            const char* error_string = nullptr;
            (void)cuGetErrorString(status, &error_string);
            fprintf(stderr,
                    "Warning: statisChanModel primary-context release failed: %s\n",
                    error_string != nullptr ? error_string : "unknown CUDA error");
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

template class statisChanModel<float, cuComplex>;

StatisChanModelFloat*
createStatisChanModelFloat(const SimConfig* sim_config,
                           const SystemLevelConfig* system_level_config,
                           const LinkLevelConfig* link_level_config,
                           const ExternalConfig* external_config,
                           uint32_t randSeed,
                           cudaStream_t strm)
{
    return new StatisChanModelFloat(sim_config,
                                    system_level_config,
                                    link_level_config,
                                    external_config,
                                    randSeed,
                                    strm);
}

void
destroyStatisChanModelFloat(StatisChanModelFloat* model) noexcept
{
    delete model;
}

uint32_t
getStatisChanModelEffectiveMaxTaps(const StatisChanModelFloat* model) noexcept
{
    return model != nullptr ? model->getEffectiveMaxTaps() : 0;
}

uint32_t
getStatisChanModelEffectiveNBsAnt(const StatisChanModelFloat* model) noexcept
{
    return model != nullptr ? model->getEffectiveNBsAnt() : 0;
}

uint32_t
getStatisChanModelEffectiveNUeAnt(const StatisChanModelFloat* model) noexcept
{
    return model != nullptr ? model->getEffectiveNUeAnt() : 0;
}

uint32_t
getStatisChanModelNumSiteUtLinks(const StatisChanModelFloat* model) noexcept
{
    return model != nullptr ? model->getNumSiteUtLinks() : 0;
}

bool
setStatisChanModelLosOverride(StatisChanModelFloat* model,
                              const uint8_t* losInd,
                              size_t nLinks)
{
    return model != nullptr && model->setLosOverride(losInd, nLinks);
}

const LinkParams*
getStatisChanModelLinkParamsHost(const StatisChanModelFloat* model) noexcept
{
    return model != nullptr ? model->getLinkParamsHost() : nullptr;
}

const ClusterParams*
getStatisChanModelClusterParamsHost(const StatisChanModelFloat* model) noexcept
{
    return model != nullptr ? model->getClusterParamsHost() : nullptr;
}
