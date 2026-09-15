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

#ifndef GPU3GPPCHAN_API_HPP
#define GPU3GPPCHAN_API_HPP

#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include <string>
#include <memory>
#include "gpu3gppchanDataset.hpp"
#include "tdl_chan_src/tdl_chan.cuh"
#include "cdl_chan_src/cdl_chan.cuh"
#include "sls_chan_src/sls_chan.cuh"
#include "config_reader.hpp"

/**
 * @brief Main statistical channel model class
 * 
 * Supports multiple channel models:
 * - Link-level: TDL (Tapped Delay Line), CDL (Clustered Delay Line)
 * - System-level: 3GPP TR 38.901 models (UMa, UMi, RMa)
 * - ISAC: Integrated Sensing and Communications (3GPP TR 38.901 Section 7.9)
 * 
 * ISAC Features (via SystemLevelConfig.isac_type):
 * - Type 0: Communication only (traditional channel model)
 * - Type 1: Single-cell TRP monostatic sensing (BS acts as TX and RX for
 *   sensing). The current native path requires CPU-only execution.
 * - Type 2: Reserved; bistatic sensing is rejected until endpoint-specific
 *   output routing is implemented.
 * 
 * Sensing targets are configured via ExternalConfig.st_config with:
 * - UAV sensing targets; small/large is controlled by st_size_ind. Other
 *   SensingTargetType enum values are reserved and rejected by this model.
 * - 2 RCS models (Model 1: deterministic, Model 2: angular dependent)
 * - Single-SPST UAV calibration support
 * 
 * @see StParam for sensing target parameters
 * @see SpstParam for scattering point parameters
 * @see SystemLevelConfig.isac_type for ISAC mode selection
 */
template <typename Tscalar, typename Tcomplex>
class statisChanModel {
public:
    /**
     * @brief Constructor for statistical channel model
     * 
     * @param sim_config Simulation configuration (frequency, bandwidth, FFT, etc.)
     * @param system_level_config System-level configuration (scenario, sites, UTs, ISAC)
     * @param link_level_config Link-level configuration (TDL/CDL parameters)
     * @param external_config External configuration (cells, UTs, antenna panels, sensing targets)
     * @param randSeed Random seed for reproducible simulations
     * @param strm CUDA stream for GPU operations (nullptr creates internal stream)
     */
    statisChanModel(const SimConfig* sim_config,
                const SystemLevelConfig* system_level_config,
                const LinkLevelConfig* link_level_config,
                const ExternalConfig* external_config,
                uint32_t randSeed,
                cudaStream_t strm = nullptr);

    ~statisChanModel();

    // Delete copy constructor and assignment operator
    statisChanModel(const statisChanModel&) = delete;
    statisChanModel& operator=(const statisChanModel&) = delete;

    // for system level simulation
    void run(const float refTime = 0.0f,
             const uint8_t continuous_fading = 1,
             const std::vector<uint16_t>& activeCell = {},
             const std::vector<std::vector<uint16_t>>& activeUt = {},
             const std::vector<Coordinate>& utNewLoc = {},
             const std::vector<float3>& utNewVelocity = {},
             const std::vector<Tcomplex*>& cir_coe = {},
             const std::vector<uint16_t*>& cir_norm_delay = {},
             const std::vector<uint16_t*>& cir_n_taps = {},
             const std::vector<Tcomplex*>& cfr_sc = {},
             const std::vector<Tcomplex*>& cfr_prbg = {});

    // for link level simulation
    void run(const float refTime0 = 0.0f,
             const uint8_t continuous_fading = 1,
             const uint8_t enableSwapTxRx = 0,
             const uint8_t txColumnMajorInd = 0);

    void reset();
    void dump_los_nlos_stats(float* lost_nlos_stats = nullptr);
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
     * antGain is per antenna element only (no array gain); downstream may add array/beamforming gain.
     * @param pl_sf_ant_gain Pointer to array for storing gain stats (required)
     * @param activeCell Vector of active cell IDs (optional, empty vector dumps all cells)
     * @param activeUt Vector of active UT IDs (optional, empty vector dumps all UEs)
     */
    void dump_pl_sf_ant_gain_stats(float* pl_sf_ant_gain,
                                   const std::vector<uint16_t>& activeCell = {},
                                   const std::vector<uint16_t>& activeUt = {});

    void dump_topology_to_yaml(const std::string& filename);
    
    /**
     * Save SLS channel data to H5 file for debugging
     * 
     * @param filenameEnding Optional string to append to filename
     */
    void saveSlsChanToH5File(std::string_view filenameEnding = "");

    /** @brief CIR coefficients for cellIdx (GPU or CPU pointer depending on mode) */
    [[nodiscard]] Tcomplex* getCirCoe(uint32_t cellIdx = 0) {
        return m_sls_chan ? m_sls_chan->getCirCoe(cellIdx) : nullptr;
    }
    /** @brief CIR normalized delay indices for cellIdx */
    [[nodiscard]] uint16_t* getCirIndex(uint32_t cellIdx = 0) {
        return m_sls_chan ? m_sls_chan->getCirIndex(cellIdx) : nullptr;
    }
    /** @brief Number of non-zero CIR taps per link for cellIdx */
    [[nodiscard]] uint16_t* getCirNtaps(uint32_t cellIdx = 0) {
        return m_sls_chan ? m_sls_chan->getCirNtaps(cellIdx) : nullptr;
    }
    /**
     * Return host-side per-link large-scale parameters.
     *
     * @return Model-owned array, or `nullptr` when no SLS model exists.
     * @note Valid after `run()` and until the next run or model destruction.
     */
    [[nodiscard]] const LinkParams* getLinkParamsHost() const {
        return m_sls_chan ? m_sls_chan->getLinkParamsHost() : nullptr;
    }
    /**
     * Return host-side per-link cluster parameters.
     *
     * @return Model-owned array, or `nullptr` when no SLS model exists.
     * @note Valid after `run()` and until the next run or model destruction.
     */
    [[nodiscard]] const ClusterParams* getClusterParamsHost() const {
        return m_sls_chan ? m_sls_chan->getClusterParamsHost() : nullptr;
    }
    /**
     * Return number of site-UT links represented by snapshots.
     *
     * @return Link count, or zero when no SLS model exists.
     */
    [[nodiscard]] uint32_t getNumSiteUtLinks() const {
        return m_sls_chan ? m_sls_chan->getNumLinks() : 0;
    }
    /**
     * Override LOS state per site-UT link for next regeneration.
     *
     * @param[in] losInd Values 0=NLOS, 1=LOS, or 255=model draw. A null
     *     pointer with zero links clears all overrides.
     * @param[in] nLinks Number of entries in `losInd`.
     * @return `true` when override was accepted or cleared; otherwise `false`.
     */
    [[nodiscard]] bool setLosOverride(const uint8_t* losInd, size_t nLinks) {
        return m_sls_chan && m_sls_chan->setLosOverride(losInd, nLinks);
    }
    /** @brief CFR on PRBG for cellIdx (nullptr if run_mode does not include PRBG) */
    [[nodiscard]] Tcomplex* getFreqChanPrbg(uint32_t cellIdx = 0) {
        return m_sls_chan ? m_sls_chan->getFreqChanPrbg(cellIdx) : nullptr;
    }
    /** @brief CFR on subcarriers for cellIdx (nullptr if run_mode does not include SC) */
    [[nodiscard]] Tcomplex* getFreqChanSc(uint32_t cellIdx = 0) {
        return m_sls_chan ? m_sls_chan->getFreqChanSc(cellIdx) : nullptr;
    }
    /** @brief Effective max CIR taps (may be > N_MAX_TAPS for ISAC) */
    [[nodiscard]] uint32_t getEffectiveMaxTaps() const {
        return m_sls_chan ? m_sls_chan->getEffectiveMaxTaps() : 0;
    }
    /** @brief Number of active links after last run() */
    [[nodiscard]] uint32_t getNumActiveLinks() const {
        return m_sls_chan ? m_sls_chan->getNumActiveLinks() : 0;
    }
    /** @brief Number of BS antennas */
    [[nodiscard]] uint16_t getNBsAnt() const {
        return m_sls_chan ? m_sls_chan->getNBsAnt() : 0;
    }
    /** @brief Number of UE antennas */
    [[nodiscard]] uint16_t getNUeAnt() const {
        return m_sls_chan ? m_sls_chan->getNUeAnt() : 0;
    }
    /** @brief Effective BS antenna count, including monostatic sensing. */
    [[nodiscard]] uint32_t getEffectiveNBsAnt() const {
        return m_sls_chan ? m_sls_chan->getEffectiveNBsAnt() : 0;
    }
    /** @brief Effective UE/RX antenna count, including monostatic sensing. */
    [[nodiscard]] uint32_t getEffectiveNUeAnt() const {
        return m_sls_chan ? m_sls_chan->getEffectiveNUeAnt() : 0;
    }
    
private:
    struct SlsChanDeleter final {
        void operator()(slsChan<Tscalar, Tcomplex>* channel) const noexcept {
            destroySlsChan(channel);
        }
    };

    const SimConfig* m_sim_config;
    const SystemLevelConfig* m_system_level_config;
    const LinkLevelConfig* m_link_level_config;
    const ExternalConfig* m_external_config;
    uint32_t m_rand_seed;
    cudaStream_t m_strm = nullptr;
    bool m_owns_stream = false;  // Flag to track if we created the stream
    CUcontext m_owned_primary_context{};
    CUdevice m_retained_primary_device{};
    bool m_owns_primary_context = false;
    
    // link level channel models, TDL and CDL
    // TODO: add AWGN channel model
    // TDL channel model
    std::unique_ptr<tdlConfig_t> m_tdl_chan_cfg;
    std::unique_ptr<tdlChan<Tscalar, Tcomplex>> m_tdl_chan;
    // CDL channel model
    std::unique_ptr<cdlConfig_t> m_cdl_chan_cfg;
    std::unique_ptr<cdlChan<Tscalar, Tcomplex>> m_cdl_chan;
    // system level channel models
    // support UMa, UMi, RMa
    std::unique_ptr<slsChan<Tscalar, Tcomplex>, SlsChanDeleter> m_sls_chan;
};

// The float/cuComplex specialization is instantiated by libgpu3gppchan.
// Consumers must not instantiate it again: doing so emits inline ownership
// methods into the consumer DSO and can mix two independently compiled class
// layouts at runtime.
extern template class statisChanModel<float, cuComplex>;
// template class statisChanModel<__half, __half2>;  // Disabled for now

using StatisChanModelFloat = statisChanModel<float, cuComplex>;

/**
 * ABI-boundary helpers for integrations that load libgpu3gppchan as a shared
 * library. Keep allocation, destruction, and inline member access in the
 * library that owns the concrete model implementation.
 */
[[nodiscard]] StatisChanModelFloat* createStatisChanModelFloat(
    const SimConfig* sim_config,
    const SystemLevelConfig* system_level_config,
    const LinkLevelConfig* link_level_config,
    const ExternalConfig* external_config,
    uint32_t randSeed,
    cudaStream_t strm = nullptr);

void destroyStatisChanModelFloat(StatisChanModelFloat* model) noexcept;

/**
 * @brief Query the effective CIR tap capacity, including ISAC expansion.
 * @param model Channel-model instance, or `nullptr`.
 * @return Required tap capacity, or zero when `model` is `nullptr`.
 */
[[nodiscard]] uint32_t getStatisChanModelEffectiveMaxTaps(
    const StatisChanModelFloat* model) noexcept;

/**
 * @brief Query the effective base-station antenna count.
 * @param model Channel-model instance, or `nullptr`.
 * @return Effective base-station antenna count, or zero for `nullptr`.
 */
[[nodiscard]] uint32_t getStatisChanModelEffectiveNBsAnt(
    const StatisChanModelFloat* model) noexcept;

/**
 * @brief Query the effective user-equipment antenna count.
 * @param model Channel-model instance, or `nullptr`.
 * @return Effective user-equipment antenna count, or zero for `nullptr`.
 */
[[nodiscard]] uint32_t getStatisChanModelEffectiveNUeAnt(
    const StatisChanModelFloat* model) noexcept;

/**
 * Return number of site-UT links represented by model snapshots.
 *
 * @param[in] model Channel-model instance, or `nullptr`.
 * @return Link count, or zero when `model` is `nullptr`.
 */
[[nodiscard]] uint32_t getStatisChanModelNumSiteUtLinks(
    const StatisChanModelFloat* model) noexcept;

/**
 * Override per-link LOS state inside libgpu3gppchan.
 *
 * @param[in,out] model Channel-model instance.
 * @param[in] losInd Values 0=NLOS, 1=LOS, or 255=model draw. A null pointer
 *     with zero links clears all overrides.
 * @param[in] nLinks Number of entries in `losInd`.
 * @return `true` when override was accepted or cleared; otherwise `false`.
 */
[[nodiscard]] bool setStatisChanModelLosOverride(
    StatisChanModelFloat* model, const uint8_t* losInd, size_t nLinks);

/**
 * Return host-side per-link large-scale parameters.
 *
 * @param[in] model Channel-model instance, or `nullptr`.
 * @return Model-owned array, or `nullptr` when unavailable.
 * @note Valid after `run()` and until the next run or model destruction.
 */
[[nodiscard]] const LinkParams* getStatisChanModelLinkParamsHost(
    const StatisChanModelFloat* model) noexcept;

/**
 * Return host-side per-link cluster parameters.
 *
 * @param[in] model Channel-model instance, or `nullptr`.
 * @return Model-owned array, or `nullptr` when unavailable.
 * @note Valid after `run()` and until the next run or model destruction.
 */
[[nodiscard]] const ClusterParams* getStatisChanModelClusterParamsHost(
    const StatisChanModelFloat* model) noexcept;

// Custom exception class for channel model errors
class ChannelModelError : public std::runtime_error {
public:
    explicit ChannelModelError(const std::string& message) 
        : std::runtime_error("Channel Model Error: " + message) {}
};

#endif // GPU3GPPCHAN_API_HPP
