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

#include "../../src/api.h"
#include "../../src/cumac.h"
#include "h5TvCreate.h"
#include "h5TvLoad.h"
#include <array>
#include <yaml-cpp/yaml.h>
#include <unordered_set>  // for std::unordered_set
#include "gpu3gppchanApi.hpp"

/** @brief Selects CPU, GPU, or both for PHY abstraction (PDSCH rate / SINR path). */
enum class PhyExecTarget : uint8_t {
    CPU = 0,  //!< CPU-only reference updateDataRatePdschCpu (only mode implemented in this testbench)
    GPU = 1,  //!< GPU path updateDataRatePdschGpu — @b not implemented; rejected at CLI startup
    Both = 2  //!< Run GPU then CPU — @b not implemented; rejected at CLI startup
};

/** @return @c true if @p target is supported; currently only @c PhyExecTarget::CPU. */
inline bool isPhyExecTargetSupported(PhyExecTarget target)
{
    return target == PhyExecTarget::CPU;
}

struct netDataMmimo {
    uint8_t     scenarioUma{};           //!< Simulation scenario: 0 UMi, 1 UMa
    float       carrierFreq{};          //!< Carrier frequency (GHz)
    float       bsHeight{};             //!< BS antenna height (m)
    float       ueHeight{};             //!< UE antenna height (m)
    float       bsAntDownTilt{};        //!< BS antenna downtilt (deg)
    float       GEmax{};                //!< Max directional gain of an antenna element
    float       cellRadius{};           //!< Cell radius (m)
    float       bsTxPowerDbm{};         //!< BS transmit power (dBm)
    float       ueTxPowerDbm{};         //!< UE transmit power (dBm)
    float       bsTxPower{};            //!< BS transmit power (W)
    float       ueTxPower{};            //!< UE transmit power (W)
    float       bsTxPowerPerPrg{};      //!< BS transmit power per PRG (W)
    float       ueTxPowerPerPrg{};      //!< UE transmit power per PRG (W)
    uint16_t    numCell{};              //!< Number of cells (max 21 supported)
    float       sfStd{};                //!< Shadow fading standard deviation (dB)
    float       noiseVarDbm{};          //!< Noise variance (dBm)
    float       noiseVar{};             //!< Noise variance (W)
    float       rho{};                  //!< Coupling-loss scaling factor rho
    float       rhoPrime{};             //!< Coupling-loss scaling factor rho'
    float       minD2Bs{};              //!< Minimum UE–BS 2D distance (m)
    float       sectorOrien[3]{};       //!< Per-sector azimuth orientation (deg), three sectors
    float       sqrChanEstNmse{};       //!< Squared channel estimation NMSE (linear)
    float       maxCombinedPostEqSinrdB{}; //!< maximum combined post-equalizer SNR in dB
    std::vector<std::vector<float>>   bsPos{}; //!< base station positions
    std::vector<float>                bsOrien{}; //!< base station orientations
    std::vector<std::vector<float>>   uePos{}; //!< UE positions
    std::vector<float>                chanGainDB{}; //!< DL channel gain in dB
    std::vector<float>                chanPlSfDB{}; //!< combined pathloss+shadowfading in dB for DL
    float*                            chanGainDBGpu{}; //!< GPU pointer to DL channel gains
    uint16_t    numThrdBlk{};           //!< CUDA thread blocks for channel kernel
    uint16_t    numThrdPerBlk{};        //!< CUDA threads per block for channel kernel
    curandState_t* states{};           //!< Device curand states for channel generation

    // Explicit default ctor: required for std::make_unique<netDataMmimo>() once copy/move are deleted.
    netDataMmimo() = default;

    // Frees device allocations from detSimParams(); must run exactly once per owning object.
    ~netDataMmimo() {
        if (chanGainDBGpu != nullptr) CUDA_CHECK_ERR(cudaFree(chanGainDBGpu));
        if (states != nullptr) CUDA_CHECK_ERR(cudaFree(states));
    }

    // chanGainDBGpu and states are raw owning pointers; shallow copy/move would double-free in ~netDataMmimo().
    netDataMmimo(const netDataMmimo&) = delete;
    netDataMmimo& operator=(const netDataMmimo&) = delete;
    netDataMmimo(netDataMmimo&&) = delete;
    netDataMmimo& operator=(netDataMmimo&&) = delete;
};

class mMimoNetwork {
public:
    /**
     * @brief Construct the multi-cell MIMO network and initialize simulation state.
     *
     * Loads network configuration from the given file, sets up API structures, channel modeling,
     * and the simulation environment (including PHY parameters and optional SLS channel model)
     * used for scheduling and PHY abstraction runs.
     *
     * @param[in] configFilePath Path to a YAML or HDF5 configuration file for the scheduler and channel setup.
     * @param[in] strm CUDA stream handle for asynchronous GPU work (default stream 0).
     * @param[in] seed_override If @p >= 0, overrides the random seed from the configuration; if @p -1 (default),
     *                          the seed is taken from the configuration file.
     */
    mMimoNetwork(const std::string& configFilePath, cudaStream_t strm = 0, int seed_override = -1);
    ~mMimoNetwork();
    
    // API structures
    std::unique_ptr<cumac::cumacCellGrpUeStatus> cellGrpUeStatusGpu;
    std::unique_ptr<cumac::cumacSchdSol> schdSolGpu;
    std::unique_ptr<cumac::cumacCellGrpPrms> cellGrpPrmsGpu;

    // get() functions
    unsigned    getSeed() const { return m_seed; }
    uint8_t     getDL() const { return m_DL; }
    uint16_t    getNCell() const { return m_nCell; }
    /**
     * @brief Fast-fading / channel model selection flag from configuration.
     * @return @c m_fadingType: @c 0 uses internal Rayleigh fading; @c 1 uses the statistic link-level (SLS) channel model.
     */
    [[nodiscard]] uint8_t getFadingType() const { return m_fadingType; }
    /**
     * @brief Configured count of active UEs per cell used for scheduling and channel state sizing.
     * @return @c m_nActiveUePerCell.
     */
    [[nodiscard]] uint16_t getNActiveUePerCell() const { return m_nActiveUePerCell; }
    uint8_t     getHarqEnabled() const { return m_harqEnabled; }
    uint8_t     getUeGrpMode() const { return m_ueGrpMode; }
    
    // channel modeling
    void genNetTopology();
    void genLSFading();
    void genFadingChannGpu(int slotIdx);
    
    /**
     * Setup channel modeling based on configuration
     * 
     * Initializes the appropriate channel model based on the fading type:
     * - fading_type = 0: Uses internal Rayleigh fading (genChan64TrKernel)
     * - fading_type = 1: Uses SLS channel model (mapSlsToMmimoKernel) with embedded configuration
     * 
     * For SLS channel model, this function creates the statisChanModel instance
     * using the embedded configuration parsed from the YAML file.
     */
    void setupChannel();
    
    // SLS channel modeling
    void initSlsChannel(const std::string& slsConfigPath);
    void genSlsChannelData(int slotIdx);

    /**
     * @brief PHY-layer PDSCH throughput / SINR abstraction for one simulation slot.
     *
     * Dispatches to updateDataRatePdschCpu(), updateDataRatePdschGpu(), or both depending on
     * @p gpuInd. Prepares per-UE uniform random draws used in TB error modeling before the update.
     *
     * @note Only @c PhyExecTarget::CPU is implemented in this testbench. @c GPU and @c Both throw
     *       at runtime if selected; @c main.cpp rejects non-CPU targets at CLI startup.
     *
     * @param[in] gpuInd Where to run the PHY update: @c PhyExecTarget::CPU, @c GPU, or @c Both.
     * @param[in] slotIdx Zero-based simulation slot index.
     * @param[in] saveSlotLog When true, detailed per-slot logs are filled for optional HDF5 export.
     */
    void phyAbstract(PhyExecTarget gpuInd, const uint16_t slotIdx, const bool saveSlotLog);

    /**
     * @brief CPU reference implementation for downlink PDSCH data rate and PHY metrics.
     *
     * Requires downlink (@c m_DL==1), managed memory, and type-1 PRBG allocation per current constraints.
     *
     * @param[in] slotIdx Slot index being evaluated.
     * @param[in] saveSlotLog When true, populate extended per-slot logging buffers for this slot.
     */
    void updateDataRatePdschCpu(const uint16_t slotIdx, const bool saveSlotLog);

    /**
     * @brief GPU implementation for downlink PDSCH data rate and PHY metrics.
     *
     * @note @b Not implemented. Throws @c std::runtime_error if called. Use @c PhyExecTarget::CPU.
     *
     * @param[in] slotIdx Slot index being evaluated.
     * @param[in] saveSlotLog When true, populate extended per-slot logging buffers for this slot.
     */
    void updateDataRatePdschGpu(const uint16_t slotIdx, const bool saveSlotLog);

    // validate scheduling solution
    void validateSchedSol();

    /**
     * @brief Allocates per-slot HDF5 export buffers (call only when per-slot logging is enabled).
     *
     * Sizes per-UE and per-cell vectors along the time axis for the full run. Memory scales roughly
     * as O(active UEs × PRGs × layers × totSimuSlots) for the largest tensors; use CLI -l only when
     * HDF5 export is needed. Default runs must not call this (see main.cpp saveSlotLog / -l).
     *
     * @param[in] totSimuSlots Number of simulation slots (must be in (0, kMaxTotSimuSlotsPerSlotLog]).
     * @return None.
     */
    void initSimuRecords(int totSimuSlots);

    // per-UE log to record simulation results
    std::vector<std::vector<int>> perUEperSlotMcs;
    // log to record per UE per slot MCS selection
    // denote uIdx = 0, 1, 2, ..., m_nActiveUe - 1 as the index of active UEs
    // denote slotIdx = 0, 1, 2, ..., totSimuSlots (input from main.cpp) - 1 as the index of slots
    // perUEperSlotMcs[uIdx][slotIdx] = -1, 0, 1, 2, ..., 27
    // -1: not scheduled or have no PRBG allocation
    // 0: MCS 0
    // 1: MCS 1
    // 2: MCS 2
    // ...
    // 27: MCS 27
    
    std::vector<std::vector<int>> perUEperSlotLayerSel;
    // log to record per UE per slot layer selection
    // denote uIdx = 0, 1, 2, ..., m_nActiveUe - 1 as the index of active UEs
    // denote slotIdx = 0, 1, 2, ..., totSimuSlots (input from main.cpp) - 1 as the index of slots
    // perUEperSlotLayerSel[uIdx][slotIdx] = -1, 0, 1, 2, ..., m_nMaxLayerPerUeMuDl 
    // -1: not scheduled or have no PRBG allocation
    // 0: no layer selected
    // 1: layer 1 selected
    // 2: layer 2 selected

    std::vector<std::vector<std::vector<std::vector<float>>>> perUEperRbgperLayerperSlotRawSinr;
    // log to record per UE per RBG per layer per slot post-Eq SINR before EESM combining
    // denote uIdx = 0, 1, 2, ..., m_nActiveUe - 1 as the index of active UEs
    // denote RbgIdx = 0, 1, 2, ..., m_nPrbGrp - 1 as the index of RBGs
    // denote LayerIdx = 0, 1, 2, ..., m_nMaxLayerPerUeMuDl - 1 as the index of layers
    // denote slotIdx = 0, 1, 2, ..., totSimuSlots (input from main.cpp) - 1 as the index of slots
    // perUEperRbgperLayerperSlotRawSinr[uIdx][RbgIdx][LayerIdx][slotIdx] indicates the raw post-Eq SINR (in dB scale) of the uIdx-th UE, in the RbgIdx-th Rbg, in the LayerIdx-th layer in the slotIdx-th slot
    // perUEperRbgperLayerperSlotRawSinr[uIdx][RbgIdx][LayerIdx][slotIdx] = -inf: not scheduled on that RBG / layer or have no PRBG allocation

    std::vector<std::vector<float>> perUEperSlotAvgSinr;
    // log to record per UE per slot average post-Eq SINR from EESM combining
    // denote uIdx = 0, 1, 2, ..., m_nActiveUe - 1 as the index of active UEs
    // denote slotIdx = 0, 1, 2, ..., totSimuSlots (input from main.cpp) - 1 as the index of slots
    // perUEperSlotAvgSinr[uIdx][slotIdx] indicates the average SINR (in dB scale) of the uIdx-th UE in the slotIdx-th slot
    // perUEperSlotAvgSinr[uIdx][slotIdx] = -inf: not scheduled or have no PRBG allocation

    std::vector<std::vector<float>> perUEperSlotServingCellPathLossAndSF;
    // log to record per UE per slot serving cell path loss and shadow fading
    // denote uIdx = 0, 1, 2, ..., m_nActiveUe - 1 as the index of active UEs
    // denote slotIdx = 0, 1, 2, ..., totSimuSlots (input from main.cpp) - 1 as the index of slots
    // perUEperSlotServingCellPathLossAndSF[uIdx][slotIdx] indicates the path loss and shadow fading (in dB) of the uIdx-th UE in the slotIdx-th slot from the serving link

    std::vector<std::vector<std::vector<float>>> perUEperCellperSlotAllCellsPathLossAndSF;
    // log to record per UE per cell per slot path loss and shadow fading from all cells (including serving cell)
    // denote uIdx = 0, 1, 2, ..., m_nActiveUe - 1 as the index of active UEs
    // denote cIdx = 0, 1, 2, ..., m_nCell - 1 as the index of cells
    // denote slotIdx = 0, 1, 2, ..., totSimuSlots (input from main.cpp) - 1 as the index of slots
    // perUEperCellperSlotAllCellsPathLossAndSF[uIdx][cIdx][slotIdx] indicates the path loss and shadow fading (in dB) of the uIdx-th UE in the slotIdx-th slot from the cIdx-th cell

    std::vector<std::vector<float>> perUEperSlotServingCellChannelGain;
    // log to record per UE per slot channel gain from serving cell
    // denote uIdx = 0, 1, 2, ..., m_nActiveUe - 1 as the index of active UEs
    // denote slotIdx = 0, 1, 2, ..., totSimuSlots (input from main.cpp) - 1 as the index of slots
    // perUEperSlotServingCellChannelGain[uIdx][slotIdx] indicates the channel gain (in dB) of the uIdx-th UE in the slotIdx-th slot from the serving link

    std::vector<std::vector<std::vector<float>>> perUEperCellperSlotAllCellsChannelGain;
    // log to record per UE per cell per slot channel gain from all cells (including serving cell)
    // denote uIdx = 0, 1, 2, ..., m_nActiveUe - 1 as the index of active UEs
    // denote cIdx = 0, 1, 2, ..., m_nCell - 1 as the index of cells
    // denote slotIdx = 0, 1, 2, ..., totSimuSlots (input from main.cpp) - 1 as the index of slots
    // perUEperCellperSlotAllCellsChannelGain[uIdx][cIdx][slotIdx] indicates the channel gain (in dB) of the uIdx-th UE in the slotIdx-th slot from the cIdx-th cell

    std::vector<std::vector<std::vector<float>>> perUEperRbgperSlotGeometrySir;
    // log to record per UE per RBG per slot geometry SIR
    // denote uIdx = 0, 1, 2, ..., m_nActiveUe - 1 as the index of active UEs
    // denote rbgIdx = 0, 1, 2, ..., m_nPrbGrp - 1 as the index of RBGs
    // denote slotIdx = 0, 1, 2, ..., totSimuSlots (input from main.cpp) - 1 as the index of slots
    // perUEperRbgperSlotGeometrySir[uIdx][rbgIdx][slotIdx] indicates the geometry SIR (in dB) of the uIdx-th UE on the rbgIdx-th RBG in the slotIdx-th slot
    // perUEperRbgperSlotGeometrySir[uIdx][rbgIdx][slotIdx] = -inf: not scheduled on that RBG or have no PRBG allocation

    std::vector<std::vector<std::vector<float>>> perUEperRbgperSlotGeometrySnr;
    // log to record per UE per RBG per slot geometry SNR
    // denote uIdx = 0, 1, 2, ..., m_nActiveUe - 1 as the index of active UEs
    // denote rbgIdx = 0, 1, 2, ..., m_nPrbGrp - 1 as the index of RBGs
    // denote slotIdx = 0, 1, 2, ..., totSimuSlots (input from main.cpp) - 1 as the index of slots
    // perUEperRbgperSlotGeometrySnr[uIdx][rbgIdx][slotIdx] indicates the geometry SNR (in dB) of the uIdx-th UE on the rbgIdx-th RBG in the slotIdx-th slot
    // perUEperRbgperSlotGeometrySnr[uIdx][rbgIdx][slotIdx] = -inf: not scheduled on that RBG or have no PRBG allocation

    std::vector<std::vector<std::vector<float>>> perUEperRbgperSlotGeometrySinr;
    // log to record per UE per RBG per slot geometry SINR
    // denote uIdx = 0, 1, 2, ..., m_nActiveUe - 1 as the index of active UEs
    // denote rbgIdx = 0, 1, 2, ..., m_nPrbGrp - 1 as the index of RBGs
    // denote slotIdx = 0, 1, 2, ..., totSimuSlots (input from main.cpp) - 1 as the index of slots
    // perUEperRbgperSlotGeometrySinr[uIdx][rbgIdx][slotIdx] indicates the geometry SINR (in dB) of the uIdx-th UE on the rbgIdx-th RBG in the slotIdx-th slot
    // perUEperRbgperSlotGeometrySinr[uIdx][rbgIdx][slotIdx] = -inf: not scheduled on that RBG or have no PRBG allocation

    std::vector<std::vector<std::vector<float>>> perUEperRbgperSlotRawPreEqSir;
    // log to record per UE per Rbg per slot raw (without EESM combining) pre-Eq Signal-to-Interference-Ratio (SIR)
    // denote uIdx = 0, 1, 2, ..., m_nActiveUe - 1 as the index of active UEs
    // denote rbgIdx = 0, 1, 2, ..., m_nPrbGrp - 1 as the index of RBGs
    // denote slotIdx = 0, 1, 2, ..., totSimuSlots (input from main.cpp) - 1 as the index of slots
    // perUEperRbgperSlotRawPreEqSir[uIdx][rbgIdx][slotIdx] indicates the raw pre-Eq SIR (in dB scale) of the uIdx-th UE, in the rbgIdx-th Rbg in the slotIdx-th slot
    // perUEperRbgperSlotRawPreEqSir[uIdx][rbgIdx][slotIdx] = -inf: not scheduled or have no PRBG allocation

    std::vector<std::vector<std::vector<float>>> perUEperRbgperSlotRawPreEqSnr;
    // log to record per UE per Rbg per slot raw (without EESM combining) pre-Eq Signal-to-Noise-Ratio (SNR)
    // denote uIdx = 0, 1, 2, ..., m_nActiveUe - 1 as the index of active UEs
    // denote rbgIdx = 0, 1, 2, ..., m_nPrbGrp - 1 as the index of RBGs
    // denote slotIdx = 0, 1, 2, ..., totSimuSlots (input from main.cpp) - 1 as the index of slots
    // perUEperRbgperSlotRawPreEqSnr[uIdx][rbgIdx][slotIdx] indicates the raw pre-Eq SNR (in dB scale) of the uIdx-th UE, in the rbgIdx-th Rbg in the slotIdx-th slot
    // perUEperRbgperSlotRawPreEqSnr[uIdx][rbgIdx][slotIdx] = -inf: not scheduled or have no PRBG allocation

    std::vector<std::vector<std::vector<float>>> perUEperRbgperSlotRawPreEqSinr;
    // log to record per UE per Rbg per slot raw (without EESM combining) pre-Eq Signal-to-Interference-and-Noise-Ratio (SINR)
    // denote uIdx = 0, 1, 2, ..., m_nActiveUe - 1 as the index of active UEs
    // denote rbgIdx = 0, 1, 2, ..., m_nPrbGrp - 1 as the index of RBGs
    // denote slotIdx = 0, 1, 2, ..., totSimuSlots (input from main.cpp) - 1 as the index of slots
    // perUEperRbgperSlotRawPreEqSinr[uIdx][rbgIdx][slotIdx] indicates the raw pre-Eq SINR (in dB scale) of the uIdx-th UE, in the rbgIdx-th Rbg in the slotIdx-th slot
    // perUEperRbgperSlotRawPreEqSinr[uIdx][rbgIdx][slotIdx] = -inf: not scheduled or have no PRBG allocation

    std::vector<std::vector<float>> perUEperSlotBler;
    // log to record per UE per slot BLER
    // denote uIdx = 0, 1, 2, ..., m_nActiveUe - 1 as the index of active UEs
    // denote slotIdx = 0, 1, 2, ..., totSimuSlots (input from main.cpp) - 1 as the index of slots
    // perUEperSlotBler[uIdx][slotIdx] indicates the BLER of the uIdx-th UE in the slotIdx-th slot
    // perUEperSlotBler[uIdx][slotIdx] = -inf: not scheduled or have no PRBG allocation

    std::vector<std::vector<int>> perUEperSlotTbErr;
    // log to record per UE per slot tbErr
    // denote uIdx = 0, 1, 2, ..., m_nActiveUe - 1 as the index of active UEs
    // denote slotIdx = 0, 1, 2, ..., totSimuSlots (input from main.cpp) - 1 as the index of slots
    // perUEperSlotTbErr[uIdx][slotIdx] = -1, 0, 1
    // -1: not scheduled or have no PRBG allocation
    // 0: successful transmission
    // 1: transmission error

    std::vector<std::vector<float>> perUEperSlotInsRate;
    // log to record per UE per slot instantaneous rate
    // denote uIdx = 0, 1, 2, ..., m_nActiveUe - 1 as the index of active UEs
    // denote slotIdx = 0, 1, 2, ..., totSimuSlots (input from main.cpp) - 1 as the index of slots
    // perUEperSlotInsRate[uIdx][slotIdx] indicates the instantaneous rate (in bits/s/Hz) of the uIdx-th UE in the slotIdx-th slot

    std::vector<std::vector<float>> perUEperSlotAvgRate;
    // log to record per UE per slot average rate
    // denote uIdx = 0, 1, 2, ..., m_nActiveUe - 1 as the index of active UEs
    // denote slotIdx = 0, 1, 2, ..., totSimuSlots (input from main.cpp) - 1 as the index of slots
    // perUEperSlotAvgRate[uIdx][slotIdx] indicates the average rate (in bits/s/Hz) of the uIdx-th UE in the slotIdx-th slot

    // per-cell log to record simulation results
    std::vector<std::vector<int>> perCellperSlotNumScheUEs;
    // log to record per cell per slot number of scheduled UEs with non-zero PRBG allocation
    // denote cIdx = 0, 1, 2, ..., m_nCell - 1 as the index of cells
    // denote slotIdx = 0, 1, 2, ..., totSimuSlots (input from main.cpp) - 1 as the index of slots
    // perCellperSlotNumScheUEs[cIdx][slotIdx] is the number of UEs associated with the cIdx-th cell that are scheduled for downlink transmission and have non-zero PRBG allocation in the slotIdx-th slot

    std::vector<std::vector<float>> perCellperSlotTbErr;
    // log to record per cell per slot average tbErr
    // denote cIdx = 0, 1, 2, ..., m_nCell - 1 as the index of cells
    // denote slotIdx = 0, 1, 2, ..., totSimuSlots (input from main.cpp) - 1 as the index of slots
    // perCellperSlotTbErr[cIdx][slotIdx] is the average BLER for all scheduled UEs (with non-zero PRBG allocation) for the cIdx-th cell in the slotIdx-th slot
    // if no UE from the cIdx-th cell is scheduled in the slotIdx-th slot, then perCellperSlotTbErr[cIdx][slotIdx] = -1.0f

    std::vector<std::vector<float>> perCellperSlotInsRate;
    // log to record per cell per slot instantaneous rate
    // denote cIdx = 0, 1, 2, ..., m_nCell - 1 as the index of cells
    // denote slotIdx = 0, 1, 2, ..., totSimuSlots (input from main.cpp) - 1 as the index of slots
    // perCellperSlotInsRate[cIdx][slotIdx] is the average instantaneous rate (in bps) for all scheduled UEs (with non-zero PRBG allocation) for the cIdx-th cell in the slotIdx-th slot

    std::vector<std::vector<std::vector<int>>> perCellperGrpperSlotNumScheLayers;
    // log to record per cell per PRG per slot number of scheduled layers
    // denote cIdx = 0, 1, 2, ..., m_nCell - 1 as the index of cells
    // denote rbgIdx = 0, 1, 2, ..., m_nPrbGrp - 1 as the index of PRG
    // denote slotIdx = 0, 1, 2, ..., totSimuSlots (input from main.cpp) - 1 as the index of slots
    // perCellperGrpperSlotNumScheLayers[cIdx][rbgIdx][slotIdx] is the number of total scheduled layers for downlink transmission for the cIdx-th cell, from the rbgIdx-th PRG in the slotIdx-th slot

private:
    // simulation configuration
    std::unique_ptr<netDataMmimo> netData;
    
    // CPU matrix operation algorithms
    std::unique_ptr<cumac::cpuMatAlg> matAlg;
    
    // SLS channel model
    std::unique_ptr<statisChanModel<float, cuComplex>> m_slsChannelModel;
    SystemLevelConfig m_sysConfig;
    LinkLevelConfig m_linkConfig;
    SimConfig m_simConfig;
    ExternalConfig m_extConfig;
    bool m_useSlsChannel = false;

    // Channel configuration
    uint8_t m_fadingType = 0; // 0: internal Rayleigh fading, 1: using statistic channel model (SLS)

    // PF scheduling parameters
    float m_pfAvgRateUpd{0.001f};

    /** Maximum totSimuSlots allowed for per-slot HDF5 logging (initSimuRecords / -l). */
    static constexpr uint16_t kMaxTotSimuSlotsPerSlotLog = 2048;
    bool m_perSlotLogExportEnabled{false}; //!< Set by initSimuRecords(); required before saveSlotLog writes.

    // randomness
    unsigned    m_seed{0}; // randomness seed
    std::default_random_engine randomEngine; // random number generation engine
    std::uniform_real_distribution<float> uniformRealDist;
    std::unique_ptr<float[]> floatRandomArr;

    // parameters and buffers
    cudaStream_t m_strm{};
    uint8_t m_fullBufferTraffic{1};
    uint8_t m_riBasedLayerSelSu{};

    uint8_t m_harqEnabled{}; // indicator for whether to enable HARQ re-transmission
    uint8_t m_DL; // indicator for DL/UL: 1 for DL, 0 for UL
    uint8_t m_ueGrpMode{}; // MU-MIMO UE grouping mode, 0: dynamic UE grouping per TTI, 1: flag-triggered UE grouping (controlled by the muUeGrpTrigger flag in cumacCellGrpPrms)
    uint8_t m_muGrpUpdate{0}; // trigger for performing MU-MIMO UE grouping in the current TTI, 0: not triggering UE grouping in the current TTI, 1: triggering UE grouping in the current TTI
    uint16_t m_nCell{};  // total number of cells
    std::vector<uint16_t> m_activeCellIds;  // Active cell IDs for SLS generation; empty means all [0..m_nCell-1]
    std::vector<uint16_t> m_utDropCellIds;  // Optional UT drop-cell IDs for SLS UE dropping; empty means use active cells
    std::vector<uint32_t> m_dumpChanSlots;  // Slot indices for SLS H5 dump; only used when fading_type == 1
    uint8_t m_semiStatFreqAlloc{}; // indication for whether or not to enable semi-static subband allocation for SU UEs/MU UEGs
    uint16_t m_numUeForGrpPerCell{}; // number of UEs considered for MU-MIMO UE grouping per TTI per cell 
    uint8_t m_numUeSchdPerCellTTI{}; // total number of SU-MIMO UEs and MU-MIMO UE groups scheduled per TTI per cell 
    uint16_t m_nMaxActUePerCell{}; // maximum number of active UEs per cell. 
    uint8_t m_nMaxUePerGrpUl{}; // maximum number of UEs per UEG for UL
    uint8_t m_nMaxUePerGrpDl{}; // maximium number of UEs per UEG for DL
    uint8_t m_nMaxLayerPerGrpUl{}; // maximium number of layers per UEG for UL
    uint8_t m_nMaxLayerPerGrpDl{}; // maximium number of layers per UEG for DL
    uint8_t m_nMaxLayerPerUeSuUl{}; // maximium number of layers per UE for SU-MIMO UL
    uint8_t m_nMaxLayerPerUeSuDl{}; // maximium number of layers per UE for SU-MIMO DL
    uint8_t m_nMaxLayerPerUeMuUl{}; // maximium number of layers per UE for MU-MIMO UL
    uint8_t m_nMaxLayerPerUeMuDl{}; // maximium number of layers per UE for MU-MIMO DL  
    uint8_t m_nMaxUegPerCellDl{}; // maximum number of UEGs per cell for DL 
    uint8_t m_nMaxUegPerCellUl{}; // maximum number of UEGs per cell for UL
    uint16_t m_nActiveUePerCell{}; // number of active UEs per cell
    uint16_t m_nActiveUe{};
    uint16_t m_nPrbGrp{}; // the number of PRGs that can be allocated for the current TTI, excluding the PRGs that need to be reserved for HARQ re-tx's
    uint8_t m_nBsAnt{}; // Each RU’s number of TX & RX antenna ports. Value: 64
    uint8_t m_nUeAnt{}; // Each active UE’s number of TX & RX antenna ports. Value: 2, 4
    uint16_t m_nPrbPerGrp{}; // the number of PRBs per PRG.
    uint16_t m_scs{}; // subcarrier spacing in Hz.
    float m_W{}; // Frequency bandwidth (Hz) of a PRG.
    float m_zfCoeff{}; // Scalar coefficient used for regularizing the zero-forcing beamformer.
    float m_betaCoeff{};
    float m_chanCorrThr{}; // threshold on the channel vector correlation value for UE grouping
    float m_srsSnrThr{};
    float m_muCoeff{}; // Coefficient for prioritizing UEs selected for MU-MIMO transmissions.
    uint8_t m_bfPowAllocScheme{}; // power allocation scheme for beamforming weights computation
    uint8_t m_allocType{}; // PRB allocation type. Currently only support 1: consecutive type-1 allocation.
    float m_muGrpSrsSnrMaxGap{}; // maximum gap among the SRS SNRs of UEs in the same MU-MIMO UEG
    float m_muGrpSrsSnrSplitThr{}; // threshold to split the SRS SNR range for grouping UEs for MU-MIMO separately
    uint8_t m_mcsSelLutType{}; // MCS selection look-up table type
    uint8_t m_mcsSelCqi{}; // CQI-based MCS selection
    float m_mcsSelSinrCapThr{25.99}; // SINR capping threshold for MCS selection
    float m_slotDuration{}; //!< OFDM slot duration in seconds.
    float m_targetMaxBlerMcs0{0.3f}; //!< Target BLER for MCS 0 used when deriving the minimum SINR floor.
    float m_minSinrDb{}; //!< Minimum SINR in dB consistent with @c m_targetMaxBlerMcs0 (from BLER LUT in the constructor).
    float m_maxCombinedPostEqSinrdB{50.0f}; //!< Symmetric bound (dB) on combined post-EQ SINR after EESM so exponentials stay finite under double-precision limits.
    float m_zeroInterfSirDb{100.0f}; //!< Fallback SIR in dB when no interference power was accumulated (avoids division by zero in SIR metrics).

    // simulation parameters
    float m_chanEstNmseDB{}; // channel estimation error NMSE in dB

    uint8_t m_useManagedMemFlag{1}; // 0: explicit CPU/GPU buffers; 1: CUDA managed memory
    uint8_t m_pdschNrOfDataSymb{8}; // number of data symbols in PDSCH
    static constexpr std::array<float, 28> eesm_beta_values = {
        // QPSK (MCS 0-4)
        1.6f, 1.63f, 1.67f, 1.73f, 1.79f, 
        // 16-QAM (MCS 5-10)
        4.27f, 4.7100f, 5.16f, 5.66f, 6.16f, 6.5f, 
        // 64-QAM (MCS 11-19)
        10.97f, 12.92f, 14.96f, 17.06f, 19.3300f, 21.85f, 24.51f, 27.14f, 29.94f, 
        // 256-QAM (MCS 20-27)
        56.48f, 65.00f, 78.58f, 92.48f, 106.27f, 118.74f, 126.36f, 132.54f
    };

    // CPU data buffers
    std::vector<cuComplex*> m_srsEstChanPtrArr;
    std::vector<int32_t*> m_srsUeMapPtrArr;
    std::vector<uint16_t*> m_sortedUeListPtrArr;
    std::unique_ptr<cumac::multiCellMuGrpList> m_muGrpListPtr;
    std::vector<uint8_t> m_cellAssocActUe;
    std::vector<float> m_avgRatesActUe;
    std::vector<int8_t> m_newDataActUe;
    std::vector<int8_t> m_tbErrLast;
    std::vector<int8_t> m_riActUe;
    std::vector<int8_t> m_cqiActUe;
    std::vector<float> m_wbSinr;
    std::vector<float> m_srsWbSnr;
    std::vector<float> m_beamformGainLastTx;
    std::vector<float> m_beamformGainCurrTx;
    std::vector<float> m_bfGainPrgCurrTx;

    std::vector<cuComplex> m_prdMatCpu;
    std::vector<int16_t> m_allocSolCpu;
    std::vector<uint8_t> m_layerSelSolCpu;
    std::vector<int16_t> m_mcsSelSolCpu;
    std::vector<uint16_t> m_ueOrderInGrpCpu;
    std::vector<uint16_t> m_setSchdUePerCellTTICpu;
    std::vector<uint8_t> m_nSCIDCpu;

    // CPU pointers pointing to CPU memory for the generated channel 
    // genChanCpu is not populated and used with (m_useManagedMemFlag == 1)
    std::vector<std::vector<cuComplex>> genChanCpu;

    // CPU pointers pointing to GPU / CUDA managed memory for generated chanenl
    std::vector<cuComplex*> m_genChanPtrArr;

    // GPU pointers pointing to GPU / CUDA managed memory for generated chanenl
    cuComplex**  genChanGpu = nullptr;
    
    // private functions
    [[nodiscard]] int loadConfigYaml(const std::string& configFilePath); // Load configuration from YAML file
    [[nodiscard]] int loadConfigHdf5(const std::string& configFilePath); // Load configuration from HDF5 file
    /**
     * Read channel configuration from YAML node
     * 
     * Parses the channel_config section from YAML configuration and sets
     * appropriate member variables for channel modeling setup. When SLS channel
     * model is selected, uses ConfigReader::readConfigFromYamlNode to parse
     * embedded SLS configuration (system_level, link_level, simulation, antenna_panels).
     * 
     * @param[in] channelConfigNode YAML node containing channel configuration parameters
     */
    void readChannelConfig(const YAML::Node& channelConfigNode);
    
    /**
     * Parse embedded SLS configuration from YAML node, reading only fields that exist
     * 
     * @param[in] config YAML node containing the channel configuration
     */
    void parseEmbeddedSlsConfig(const YAML::Node& config);
    
    void setupApiStructs();
    void destroyApiStructs();
    void copySolutionToCpu();
    void copyGenChanToCpu();
    
    // simulation functions
    void detSimParams();

    // helper functions
    bool hasExtension(const std::string& filename, const std::string& ext);
    bool isYamlFile(const std::string& path);
    bool isHdf5File(const std::string& path);

    // BLER lookup (MCS vs SINR/BLER CSV)
    std::string m_blerLutPath; //!< Path to the CSV BLER table passed to loadBlerTable() (MCS, SNR/BLER lists, SNR at 0.1 BLER).
    std::vector<std::vector<std::vector<float>>> m_blerTable; //!< Per-MCS rows @c {SINR dB, BLER, SNR@0.1BLER}; filled by loadBlerTable().

    /**
     * @brief Parses the BLER CSV at @p blerLutPath and populates @c m_blerTable for MCS 0-27.
     *
     * @param[in] blerLutPath Filesystem path to the comma-separated lookup table.
     */
    void loadBlerTable(const std::string& blerLutPath);

    /**
     * @brief Interpolates block error rate (BLER) from the loaded LUT for a given average SINR and MCS.
     *
     * @param[in] avgSinrDB Average post-EQ SINR in dB consistent with the LUT reference SNR axis.
     * @param[in] MCS Modulation and coding scheme index in @c [0, 27].
     * @return BLER in @c [0, 1] from linear interpolation; saturates to @c 1.0 below the lowest tabulated SNR and @c 0.0 above the highest.
     */
    [[nodiscard]] float calcBler(float avgSinrDB, int MCS);

    /**
     * @brief Computes NR-style PDSCH/PUSCH transport block size (TBS) from PRBs, symbols, layers, rate, and QAM order.
     *
     * @param[in] rbSize Number of allocated PRBs.
     * @param[in] nDataSymb Number of data OFDM symbols in the slot.
     * @param[in] nrOfLayers Number of MIMO layers used for the transmission.
     * @param[in] codeRate Effective channel code rate (fractional, e.g. @c cumac::mcsTable_codeRate / 1024).
     * @param[in] qam QAM order in bits per symbol (e.g. @c cumac::mcsTable_qamOrder).
     * @return Selected TBS in bits after quantization against @c cumac::TBS_table.
     */
    [[nodiscard]] uint32_t determineTbsPxsch(int rbSize, int nDataSymb, int nrOfLayers, float codeRate, int qam);

    /**
     * @brief Returns the SINR in dB on the MCS-0 BLER curve that matches @p targetMaxBlerMcs0 (linear interpolation).
     *
     * @param[in] targetMaxBlerMcs0 Target BLER at MCS 0 in @c [0.0, 1.0] (typically @c m_targetMaxBlerMcs0).
     * @return Minimum average SINR in dB associated with that BLER; uses tabular endpoints when @p targetMaxBlerMcs0 lies outside the inner range.
     */
    [[nodiscard]] float calcSinrMinDb(const float targetMaxBlerMcs0);
    
};

__global__ void init_curand(unsigned int t_seed, int id_offset, curandState *state);

__global__ void addChannelEstErrorKernel(const cuComplex* inputChan,
                                               cuComplex* outputChan,
                                               const int totalElements,
                                               const float sqrChanEstNmse,
                                               curandState_t* states);

__global__ void genChan64TrKernel(cuComplex**       genChanGpu, 
                                  cuComplex**       srsEstChan,
                                  int32_t**         srsUeMap,
                                  float*            srsWbSnrGpu,
                                  float*            chanGainDBGpu,
                                  const int         nPrbGrp, 
                                  const int         numCell, 
                                  const int         nActiveUe,
                                  const int         numBsAnt,
                                  const int         numUeAnt,
                                  const float       rho,
                                  const float       rhoPrime,
                                  const float       sqrChanEstNmse,
                                  const float       ueTxPowerPerPrg,
                                  const float       noiseVar,
                                  const int         slotIdx,
                                  curandState_t*    states);
