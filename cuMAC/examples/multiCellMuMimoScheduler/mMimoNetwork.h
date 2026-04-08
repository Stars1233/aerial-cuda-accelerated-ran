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
#include <yaml-cpp/yaml.h>
#include <unordered_set>  // for std::unordered_set
#include "chanModelsApi.hpp"

struct netDataMmimo {
    uint8_t     scenarioUma{};  // indication for simulation scenario: 0 - UMi, 1 - UMa
    float       carrierFreq{}; // carrier frequency in unit of GHz
    float       bsHeight{}; // BS antenna height in unit of meters
    float       ueHeight{}; // UE antenna height in unit of meters
    float       bsAntDownTilt{}; // antenna downtilting degree
    float       GEmax{}; // maximum directional gain of an antenna element 
    float       cellRadius{}; // cell radius in unit of meters
    float       bsTxPowerDbm{}; // BS transmit power in unit of dBm
    float       ueTxPowerDbm{}; // UE transmit power in unit of dBm
    float       bsTxPower{}; // BS transmit power in unit of W
    float       ueTxPower{}; // UE transmit power in unit of W
    float       bsTxPowerPerPrg{}; // BS transmit power per PRG in unit of W
    float       ueTxPowerPerPrg{}; // UE transmit power per PRG in unit of W
    uint16_t    numCell{}; // totel number of cells. Currently support a maximum number of 21 cells
    float       sfStd{}; // shadow fading STD in dB
    float       noiseVarDbm{}; // noise variance in dBm
    float       noiseVar{}; // noise variance in W
    float       rho{};
    float       rhoPrime{};
    float       minD2Bs{};
    float       sectorOrien[3]{};
    float       sqrChanEstNmse;
        
    std::vector<std::vector<float>>   bsPos;
    std::vector<std::vector<float>>   uePos;
    std::vector<float>                chanGainDB; // for DL
    float*                            chanGainDBGpu = nullptr; // for DL

    // for channel generation CUDA kernel
    uint16_t    numThrdBlk{};
    uint16_t    numThrdPerBlk{};
    curandState_t* states = nullptr;

    // destructor
    ~netDataMmimo() {
        if (chanGainDBGpu != nullptr) CUDA_CHECK_ERR(cudaFree(chanGainDBGpu));
        if (states != nullptr) CUDA_CHECK_ERR(cudaFree(states));
    }
};

class mMimoNetwork {
public:
    mMimoNetwork(const std::string& configFilePath, cudaStream_t strm = 0);
    ~mMimoNetwork();
    
    // API structures
    std::unique_ptr<cumac::cumacCellGrpUeStatus> cellGrpUeStatusGpu;
    std::unique_ptr<cumac::cumacSchdSol> schdSolGpu;
    std::unique_ptr<cumac::cumacCellGrpPrms> cellGrpPrmsGpu;

    // get() functions
    unsigned    getSeed() const { return m_seed; }
    uint8_t     getDL() const { return m_DL; }
    uint16_t    getNCell() const { return m_nCell; }
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

    // PHY layer abstraction
    void phyAbstract(int slotIdx); // wrapper function for PHY layer abstraction
    void updateDataRatePdschGpu(int slotIdx);

    // validate scheduling solution
    void validateSchedSol();

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

    // randomness
    unsigned    m_seed{0}; // randomness seed
    std::default_random_engine randomEngine; // random number generation engine
    std::uniform_real_distribution<float> uniformRealDist;

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

    // simulation parameters
    float m_chanEstNmseDB{}; // channel estimation error NMSE in dB

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

    std::vector<std::vector<cuComplex>> genChanCpu;

    // GPU data buffers
    cuComplex**  genChanGpu = nullptr;
    std::vector<cuComplex*> m_genChanPtrArr;

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
