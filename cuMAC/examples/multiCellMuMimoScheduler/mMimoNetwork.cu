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

#include "mMimoNetwork.h"

// Mathematical constants for channel estimation error calculations
static constexpr float INV_SQRT_2 = 0.7071067811865476f; // 1/sqrt(2) for Gaussian noise scaling

mMimoNetwork::mMimoNetwork(const std::string& configFilePath, cudaStream_t strm)
{
    if (isYamlFile(configFilePath)) {
        // Load configuration from YAML file
        if (loadConfigYaml(configFilePath) != 0)
        {
            std::cerr << "Error loading configuration from config.yaml" << std::endl;
            throw std::runtime_error("Failed to load configuration");
        }
    } else if (isHdf5File(configFilePath)) {
        // Load configuration from HDF5 file
        if (loadConfigHdf5(configFilePath) != 0)
        {
            std::cerr << "Error loading configuration from HDF5 file" << std::endl;
            throw std::runtime_error("Failed to load configuration");
        }
    } else {
        std::cerr << "Error: invalid configuration file" << std::endl;
        throw std::runtime_error("Invalid configuration file");
    }

    randomEngine = std::default_random_engine(m_seed);
    uniformRealDist = std::uniform_real_distribution<float>(0.0,1.0);

    m_strm = strm;

    cellGrpUeStatusGpu = std::make_unique<cumac::cumacCellGrpUeStatus>();
    schdSolGpu         = std::make_unique<cumac::cumacSchdSol>();
    cellGrpPrmsGpu     = std::make_unique<cumac::cumacCellGrpPrms>();

    setupApiStructs();

    netData = std::make_unique<netDataMmimo>();
    detSimParams();

    setupChannel();
}

mMimoNetwork::~mMimoNetwork()
{
    /* Destruction */
    destroyApiStructs();
    for (int i = 0; i < m_nCell; i++) {
        CUDA_CHECK_ERR(cudaFree(m_genChanPtrArr[i]));
    }
    CUDA_CHECK_ERR(cudaFree(genChanGpu));
}

void mMimoNetwork::detSimParams()
{
    netData->scenarioUma = 1; // UMa
    netData->carrierFreq = 5.0; // carrier frequency in unit of GHz
    netData->sfStd = 3.0; // shadow fading STD in dB
    float noiseFigure  = 10.0; // dB
    netData->noiseVarDbm = -174.0 + noiseFigure + 10.0*log10(m_W);
    netData->noiseVar = pow(10.0, (netData->noiseVarDbm - 30.0)/10.0); // W
    netData->bsHeight = 25;
    netData->ueHeight = 1.5;
    netData->bsAntDownTilt = 102.0; // degree
    netData->GEmax = 9.0; // dBi
    netData->cellRadius = 500.0;
    netData->bsTxPowerDbm = 49.0; // dBm, 79.4328 W
    netData->ueTxPowerDbm = 23.0; // dBm, 0.2 W
    netData->bsTxPower = pow(10.0, (netData->bsTxPowerDbm - 30.0)/10.0); // W
    netData->ueTxPower = pow(10.0, (netData->ueTxPowerDbm - 30.0)/10.0); // W
    netData->bsTxPowerPerPrg = netData->bsTxPower/m_nPrbGrp; // W
    netData->ueTxPowerPerPrg = netData->ueTxPower/m_nPrbGrp;
    netData->numCell = m_nCell; // assume all cells are coordinated cells
    netData->sectorOrien[0] = M_PI/3.0;
    netData->sectorOrien[1] = M_PI;
    netData->sectorOrien[2] = M_PI*5.0/3.0;
    netData->minD2Bs = 30.0;
    netData->rho = 0.9938; // for 5.0 GHz carrier frequency, 0.5 ms time slot duration (30000 Hz SCS), 3 m/s UE moving speed
    netData->rhoPrime = 0.0786; // sqrt(1-rho^2)*sqrt(0.5)
    netData->sqrChanEstNmse = pow(10.0, m_chanEstNmseDB/20.0); // square root of channel estimation error NMSE

    netData->bsPos.resize(netData->numCell);
    netData->chanGainDB.resize(netData->numCell*m_nActiveUe);
    CUDA_CHECK_ERR(cudaMalloc((void **)&netData->chanGainDBGpu, netData->numCell*m_nActiveUe*sizeof(float)));

    netData->uePos.resize(m_nActiveUe);
    for (int cIdx = 0; cIdx < netData->numCell; cIdx++) {
        netData->bsPos[cIdx].resize(3);
        for (int uIdx = 0; uIdx < m_nActiveUePerCell; uIdx++) {
            netData->uePos[cIdx*m_nActiveUePerCell + uIdx].resize(3);
        }
    }

    netData->numThrdBlk = m_nCell*m_nPrbGrp;
    netData->numThrdPerBlk = m_nBsAnt*m_nUeAnt*floor(1024.0/static_cast<float>(m_nBsAnt*m_nUeAnt));

    CUDA_CHECK_ERR(cudaMalloc((void **)&netData->states, netData->numThrdBlk*netData->numThrdPerBlk*sizeof(curandState_t)));
     
    init_curand<<<netData->numThrdBlk, netData->numThrdPerBlk, 0, m_strm>>>(time(NULL), 0, netData->states);
}

int mMimoNetwork::loadConfigYaml(const std::string& configFilePath)
{
    try
    {
        // Load YAML file using yaml-cpp
        YAML::Node root = YAML::LoadFile(configFilePath);
        
        // Load parameters from YAML
        m_seed = root["seed"].as<unsigned>();
        m_DL = root["DL"].as<uint8_t>();
        m_harqEnabled = root["harqEnabled"].as<uint8_t>();
        m_ueGrpMode = root["ueGrpMode"].as<uint8_t>();
        m_semiStatFreqAlloc = root["semiStatFreqAlloc"].as<uint8_t>();
        m_nActiveUePerCell = root["nActiveUePerCell"].as<uint16_t>();
        m_numUeForGrpPerCell = root["numUeForGrpPerCell"].as<uint16_t>();
        if (m_nActiveUePerCell != m_numUeForGrpPerCell) {
            std::cerr << "Error: parameters nActiveUePerCell and numUeForGrpPerCell must be the same" << std::endl;
        }
        m_numUeSchdPerCellTTI = root["numUeSchdPerCellTTI"].as<uint8_t>();
        m_nMaxActUePerCell = root["nMaxActUePerCell"].as<uint16_t>();
        m_nMaxUePerGrpUl = root["nMaxUePerGrpUl"].as<uint8_t>();
        m_nMaxUePerGrpDl = root["nMaxUePerGrpDl"].as<uint8_t>();
        m_nMaxLayerPerGrpUl = root["nMaxLayerPerGrpUl"].as<uint8_t>();
        m_nMaxLayerPerGrpDl = root["nMaxLayerPerGrpDl"].as<uint8_t>();
        m_nMaxLayerPerUeSuUl = root["nMaxLayerPerUeSuUl"].as<uint8_t>();
        m_nMaxLayerPerUeSuDl = root["nMaxLayerPerUeSuDl"].as<uint8_t>();
        m_nMaxLayerPerUeMuUl = root["nMaxLayerPerUeMuUl"].as<uint8_t>();
        m_nMaxLayerPerUeMuDl = root["nMaxLayerPerUeMuDl"].as<uint8_t>();
        m_nMaxUegPerCellDl = root["nMaxUegPerCellDl"].as<uint8_t>();
        m_nMaxUegPerCellUl = root["nMaxUegPerCellUl"].as<uint8_t>();
        m_nCell = root["nCell"].as<uint16_t>();
        m_activeCellIds.clear();
        if (root["activeCellIds"]) {
            const YAML::Node activeCellIdsNode = root["activeCellIds"];
            std::unordered_set<uint16_t> seenCell;
            if (activeCellIdsNode.IsNull()) {
                // Treat empty/null as [] -> use all cells [0..nCell-1].
            } else if (activeCellIdsNode.IsScalar()) {
                const uint16_t cid = activeCellIdsNode.as<uint16_t>();
                seenCell.insert(cid);
                m_activeCellIds.push_back(cid);
            } else if (activeCellIdsNode.IsSequence()) {
                for (std::size_t idx = 0; idx < activeCellIdsNode.size(); ++idx) {
                    const uint16_t cid = activeCellIdsNode[idx].as<uint16_t>();
                    if (seenCell.insert(cid).second) {
                        m_activeCellIds.push_back(cid);
                    }
                }
            } else {
                throw std::runtime_error("activeCellIds must be null, a single number, or a sequence");
            }
        }
        m_nActiveUe = m_nActiveUePerCell*m_nCell;
        m_nPrbGrp = root["nPrbGrp"].as<uint16_t>();
        m_nBsAnt = root["nBsAnt"].as<uint8_t>();
        m_nUeAnt = root["nUeAnt"].as<uint8_t>();
        m_nPrbPerGrp = root["nPrbPerGrp"].as<uint16_t>();
        m_scs = root["scs"].as<uint16_t>();
        m_W = static_cast<float>(12.0*m_nPrbPerGrp*m_scs);
        m_zfCoeff = root["zfCoeff"].as<float>();
        m_betaCoeff = root["betaCoeff"].as<float>();
        m_chanCorrThr = root["chanCorrThr"].as<float>();
        m_srsSnrThr = root["srsSnrThr"].as<float>();
        m_muCoeff = root["muCoeff"].as<float>();
        m_muGrpSrsSnrMaxGap = root["muGrpSrsSnrMaxGap"].as<float>();
        m_muGrpSrsSnrSplitThr = root["muGrpSrsSnrSplitThr"].as<float>();
        m_allocType = root["allocType"].as<uint8_t>();
        m_bfPowAllocScheme = root["bfPowAllocScheme"].as<uint8_t>();
        m_riBasedLayerSelSu = root["riBasedLayerSelSu"].as<uint8_t>();
        m_mcsSelCqi = root["mcsSelCqi"].as<uint8_t>();
        m_fullBufferTraffic = root["fullBufferTraffic"].as<uint8_t>();
        m_mcsSelLutType = root["mcsSelLutType"].as<uint8_t>();

        m_chanEstNmseDB = root["chanEstNmseDB"].as<float>();
        
        // Load channel configuration
        if (root["channel_config"]) {
            readChannelConfig(root["channel_config"]);
        } else {
            m_fadingType = 0; // Default to internal Rayleigh fading
        }
        
        printf("Loaded config parameters\n");

        return 0; // Success
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error loading configuration from " << configFilePath << ": " << e.what() << std::endl;
        return -1; // Failure
    }
}

int mMimoNetwork::loadConfigHdf5(const std::string& configFilePath)
{
    try
    {
        cumac::cumacSchedulerParam data;

        // open TV H5 file
        H5::H5File file(configFilePath, H5F_ACC_RDONLY);

        // Open the dataset
        H5::DataSet dataset = file.openDataSet("cumacSchedulerParam");

        // Get the compound data type
        H5::CompType compoundType = dataset.getCompType();

        // Read the data from the dataset
        dataset.read(&data, compoundType);
        
        // Load parameters from HDF5 file
        m_DL = 1;
        m_harqEnabled = 1;
        m_ueGrpMode = 0;
        m_semiStatFreqAlloc = data.semiStatFreqAlloc;
        m_nActiveUe = data.nActiveUe;
        m_numUeForGrpPerCell = data.numUeForGrpPerCell;
        m_numUeSchdPerCellTTI = data.numUeSchdPerCellTTI;
        m_nMaxActUePerCell = data.nMaxActUePerCell;
        m_nMaxUePerGrpUl = data.nMaxUePerGrpUl;
        m_nMaxUePerGrpDl = data.nMaxUePerGrpDl;
        m_nMaxLayerPerGrpUl = data.nMaxLayerPerGrpUl;
        m_nMaxLayerPerGrpDl = data.nMaxLayerPerGrpDl;
        m_nMaxLayerPerUeSuUl = data.nMaxLayerPerUeSuUl;
        m_nMaxLayerPerUeSuDl = data.nMaxLayerPerUeSuDl;
        m_nMaxLayerPerUeMuUl = data.nMaxLayerPerUeMuUl;
        m_nMaxLayerPerUeMuDl = data.nMaxLayerPerUeMuDl;
        m_nMaxUegPerCellDl = data.nMaxUegPerCellDl;
        m_nMaxUegPerCellUl = data.nMaxUegPerCellUl;
        m_nCell = data.nCell;
        m_nActiveUePerCell = data.nActiveUe/m_nCell;
        m_nPrbGrp = data.nPrbGrp;
        m_nBsAnt = data.nBsAnt;
        m_nUeAnt = data.nUeAnt;
        m_W = data.W;
        m_zfCoeff = data.zfCoeff;
        m_betaCoeff = data.betaCoeff;
        m_chanCorrThr = data.chanCorrThr;
        m_srsSnrThr = data.srsSnrThr;
        m_muCoeff = data.muCoeff;
        m_muGrpSrsSnrMaxGap = data.muGrpSrsSnrMaxGap;
        m_muGrpSrsSnrSplitThr = data.muGrpSrsSnrSplitThr;
        m_allocType = data.allocType;
        m_bfPowAllocScheme = data.bfPowAllocScheme;
        m_mcsSelCqi = 0;
        m_mcsSelLutType = data.mcsSelLutType;
        m_mcsSelSinrCapThr = data.mcsSelSinrCapThr;

        printf("Loaded config parameters\n");

        return 0; // Success
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error loading configuration from " << configFilePath << ": " << e.what() << std::endl;
        return -1; // Failure
    }
}

void mMimoNetwork::readChannelConfig(const YAML::Node& channelConfigNode)
{
    // Read fading type
    m_fadingType = channelConfigNode["fading_type"].as<uint8_t>();
    m_dumpChanSlots.clear();
    if (channelConfigNode["dump_chan_slots"]) {
        const YAML::Node dumpSlotsNode = channelConfigNode["dump_chan_slots"];
        if (dumpSlotsNode.IsNull()) {
            // Treat empty/null as [].
        } else if (dumpSlotsNode.IsScalar()) {
            const int slotIdx = dumpSlotsNode.as<int>();
            if (slotIdx < 0) {
                throw std::runtime_error("channel_config.dump_chan_slots must contain non-negative slot indices");
            }
            m_dumpChanSlots.push_back(static_cast<uint32_t>(slotIdx));
        } else if (dumpSlotsNode.IsSequence()) {
            for (std::size_t i = 0; i < dumpSlotsNode.size(); ++i) {
                const int slotIdx = dumpSlotsNode[i].as<int>();
                if (slotIdx < 0) {
                    throw std::runtime_error("channel_config.dump_chan_slots must contain non-negative slot indices");
                }
                m_dumpChanSlots.push_back(static_cast<uint32_t>(slotIdx));
            }
        } else {
            throw std::runtime_error("channel_config.dump_chan_slots must be null, a single number, or a sequence");
        }
    }
    if (m_fadingType != 1 && !m_dumpChanSlots.empty()) {
        std::printf("Warning: channel_config.dump_chan_slots is ignored when fading_type != 1\n");
        m_dumpChanSlots.clear();
    }
    
    // If using SLS channel model, parse the embedded SLS configuration
    if (m_fadingType == 1) {
        // Parse SLS configuration from the embedded YAML node (only fields that exist)
        parseEmbeddedSlsConfig(channelConfigNode);
    }
    printf("Successfully loaded channel configuration: fading_type = %u\n", m_fadingType);
    
    // System-level parameters derived from cuMAC config
    constexpr uint8_t nSectorPerSite = 3;  // assume three sectors per site
    m_sysConfig.n_sector_per_site = nSectorPerSite;
    m_sysConfig.n_ut = m_nActiveUePerCell * m_nCell;
    m_sysConfig.n_ut_drop_cells = 0;
    if (m_nCell > SystemLevelConfig::kMaxUtDropCells) {
        throw std::runtime_error("nCell exceeds maximum supported ut_drop_cells capacity");
    }

    const uint32_t maxSupportedCellId = SystemLevelConfig::kMaxUtDropCells;
    std::vector<uint16_t> activeCellPool;
    if (m_activeCellIds.empty()) {
        activeCellPool.reserve(m_nCell);
        for (uint32_t cid = 0; cid < m_nCell; ++cid) {
            activeCellPool.push_back(static_cast<uint16_t>(cid));
        }
        m_sysConfig.n_site = (m_nCell + nSectorPerSite - 1) / nSectorPerSite;
    } else {
        uint16_t maxActiveCellId = 0;
        if (m_activeCellIds.size() != m_nCell) {
            throw std::runtime_error("activeCellIds length (" + std::to_string(m_activeCellIds.size()) +
                                     ") must equal nCell (" + std::to_string(m_nCell) + ")");
        }
        activeCellPool.reserve(m_activeCellIds.size());
        for (const uint16_t cid : m_activeCellIds) {
            if (cid >= maxSupportedCellId) {
                throw std::runtime_error("activeCellIds contains cell id " + std::to_string(cid) +
                                         " which exceeds the maximum supported (" + std::to_string(maxSupportedCellId - 1) + ")");
            }
            activeCellPool.push_back(cid);
            maxActiveCellId = std::max(maxActiveCellId, cid);
        }
        m_sysConfig.n_site = (maxActiveCellId + 1 + nSectorPerSite - 1) / nSectorPerSite;  // maxActiveCellId is 0-indexed
    }

    if (m_utDropCellIds.empty()) {
        for (const uint16_t cid : activeCellPool) {
            m_sysConfig.ut_drop_cells[m_sysConfig.n_ut_drop_cells++] = cid;
        }
    } else {
        std::unordered_set<uint16_t> activeCellSet(activeCellPool.begin(), activeCellPool.end());
        for (const uint16_t cid : m_utDropCellIds) {
            if (cid >= maxSupportedCellId) {
                throw std::runtime_error("ut_drop_cells contains cell id " + std::to_string(cid) +
                                         " which exceeds the maximum supported (" + std::to_string(maxSupportedCellId - 1) + ")");
            }
            if (!activeCellSet.count(cid)) {
                throw std::runtime_error("ut_drop_cells contains cell id " + std::to_string(cid) +
                                         " that is not in activeCellIds");
            }
            if (m_sysConfig.n_ut_drop_cells >= SystemLevelConfig::kMaxUtDropCells) {
                throw std::runtime_error("ut_drop_cells exceeds max supported entries (" + std::to_string(SystemLevelConfig::kMaxUtDropCells) + ")");
            }
            m_sysConfig.ut_drop_cells[m_sysConfig.n_ut_drop_cells++] = cid;
        }
    }
    
    // Some simulation parameters derived from cuMAC config  
    m_simConfig.sc_spacing_hz = m_scs;
    m_simConfig.n_prb = m_nPrbGrp * m_nPrbPerGrp;
    m_simConfig.n_prbg = m_nPrbGrp;
}

/**
 * Parse embedded SLS configuration from YAML node, reading only fields that exist
 * 
 * @param[in] config YAML node containing the channel configuration
 */
void mMimoNetwork::parseEmbeddedSlsConfig(const YAML::Node& config)
{
    try {
        // Read System Level Configuration (only fields that exist in our config)
        if (config["system_level"]) {
            const YAML::Node& sl = config["system_level"];
            
            // Convert string to Scenario enum
            if (sl["scenario"]) {
                std::string scenario_str = sl["scenario"].as<std::string>();
                if (scenario_str == "UMa") {
                    m_sysConfig.scenario = Scenario::UMa;
                } else if (scenario_str == "UMi") {
                    m_sysConfig.scenario = Scenario::UMi;
                } else if (scenario_str == "RMa") {
                    m_sysConfig.scenario = Scenario::RMa;
                } else if (scenario_str == "Indoor") {
                    m_sysConfig.scenario = Scenario::Indoor;
                } else if (scenario_str == "InF") {
                    m_sysConfig.scenario = Scenario::InF;
                } else if (scenario_str == "SMa") {
                    m_sysConfig.scenario = Scenario::SMa;
                } else {
                    throw std::runtime_error("Invalid scenario: " + scenario_str);
                }
            }
            
            // Read only the fields that exist in our simplified config
            if (sl["isd"]) m_sysConfig.isd = sl["isd"].as<float>();
            if (sl["ut_drop_option"]) {
                const uint8_t utDropOption = sl["ut_drop_option"].as<std::uint8_t>();
                if (utDropOption > 2) {
                    throw std::runtime_error("Invalid ut_drop_option in channel_config.system_level: must be 0, 1, or 2");
                }
                m_sysConfig.ut_drop_option = utDropOption;
            }
            // m_utDropCellIds is populated here; clear before parsing to avoid stale state.
            if (sl["ut_drop_cells"]) {
                m_utDropCellIds.clear();
                const YAML::Node utDropCellsNode = sl["ut_drop_cells"];
                std::unordered_set<uint16_t> seenCell;
                if (utDropCellsNode.IsNull()) {
                    // Treat empty/null as [] -> use active cells.
                } else if (utDropCellsNode.IsScalar()) {
                    const uint16_t cid = utDropCellsNode.as<uint16_t>();
                    seenCell.insert(cid);
                    m_utDropCellIds.push_back(cid);
                } else if (utDropCellsNode.IsSequence()) {
                    for (std::size_t idx = 0; idx < utDropCellsNode.size(); ++idx) {
                        const uint16_t cid = utDropCellsNode[idx].as<uint16_t>();
                        if (seenCell.insert(cid).second) {
                            m_utDropCellIds.push_back(cid);
                        }
                    }
                } else {
                    throw std::runtime_error("ut_drop_cells must be null, a single number, or a sequence");
                }
            }
            if (sl["ut_cell_2d_dist"] && sl["ut_cell_2d_dist"].IsSequence()) {
                if (sl["ut_cell_2d_dist"].size() != 2) {
                    throw std::runtime_error("ut_cell_2d_dist must contain exactly 2 values: [min, max]");
                }
                m_sysConfig.ut_cell_2d_dist[0] = sl["ut_cell_2d_dist"][0].as<float>();
                m_sysConfig.ut_cell_2d_dist[1] = sl["ut_cell_2d_dist"][1].as<float>();
            }
            if (sl["optional_pl_ind"]) m_sysConfig.optional_pl_ind = sl["optional_pl_ind"].as<std::uint8_t>();
            if (sl["o2i_building_penetr_loss_ind"]) m_sysConfig.o2i_building_penetr_loss_ind = sl["o2i_building_penetr_loss_ind"].as<std::uint8_t>();
            if (sl["o2i_car_penetr_loss_ind"]) m_sysConfig.o2i_car_penetr_loss_ind = sl["o2i_car_penetr_loss_ind"].as<std::uint8_t>();
            if (sl["enable_near_field_effect"]) m_sysConfig.enable_near_field_effect = sl["enable_near_field_effect"].as<std::uint8_t>();
            if (sl["enable_non_stationarity"]) m_sysConfig.enable_non_stationarity = sl["enable_non_stationarity"].as<std::uint8_t>();
            if (sl["force_indoor_ratio"]) m_sysConfig.force_indoor_ratio = sl["force_indoor_ratio"].as<float>();
            if (sl["force_los_prob"] && sl["force_los_prob"].size() >= 2) {
                m_sysConfig.force_los_prob[0] = sl["force_los_prob"][0].as<float>();
                m_sysConfig.force_los_prob[1] = sl["force_los_prob"][1].as<float>();
            }
            if (sl["force_ut_speed"] && sl["force_ut_speed"].size() >= 2) {
                m_sysConfig.force_ut_speed[0] = sl["force_ut_speed"][0].as<float>();
                m_sysConfig.force_ut_speed[1] = sl["force_ut_speed"][1].as<float>();
            }
            if (sl["disable_pl_shadowing"]) m_sysConfig.disable_pl_shadowing = sl["disable_pl_shadowing"].as<std::uint8_t>();
            if (sl["enable_per_tti_lsp"]) m_sysConfig.enable_per_tti_lsp = sl["enable_per_tti_lsp"].as<std::uint8_t>();
            if (sl["enable_propagation_delay"]) m_sysConfig.enable_propagation_delay = sl["enable_propagation_delay"].as<std::uint8_t>();
        }

        // Read Link Level Configuration (only fields that exist)
        if (config["link_level"]) {
            const YAML::Node& ll = config["link_level"];
            
            if (ll["fast_fading_type"]) m_linkConfig.fast_fading_type = ll["fast_fading_type"].as<int>();
            if (ll["delay_profile"]) {
                std::string delay_str = ll["delay_profile"].as<std::string>();
                m_linkConfig.delay_profile = delay_str.empty() ? 'A' : delay_str[0];
            }
            if (ll["delay_spread"]) m_linkConfig.delay_spread = ll["delay_spread"].as<float>();
            if (ll["velocity"] && ll["velocity"].size() >= 3) {
                m_linkConfig.velocity[0] = ll["velocity"][0].as<float>();
                m_linkConfig.velocity[1] = ll["velocity"][1].as<float>();
                m_linkConfig.velocity[2] = ll["velocity"][2].as<float>();
            }
            if (ll["num_ray"]) m_linkConfig.num_ray = ll["num_ray"].as<int>();
            if (ll["cfo_hz"]) m_linkConfig.cfo_hz = ll["cfo_hz"].as<float>();
            if (ll["delay"]) m_linkConfig.delay = ll["delay"].as<float>();
        }

        // Read Simulation Configuration (only fields that exist)
        if (config["simulation"]) {
            const YAML::Node& tc = config["simulation"];
            
            if (tc["link_sim_ind"]) m_simConfig.link_sim_ind = tc["link_sim_ind"].as<int>();
            if (tc["center_freq_hz"]) m_simConfig.center_freq_hz = tc["center_freq_hz"].as<float>();
            if (tc["bandwidth_hz"]) m_simConfig.bandwidth_hz = tc["bandwidth_hz"].as<float>();
            if (tc["fft_size"]) m_simConfig.fft_size = tc["fft_size"].as<int>();
            if (tc["n_snapshot_per_slot"]) m_simConfig.n_snapshot_per_slot = tc["n_snapshot_per_slot"].as<int>();
            if (tc["run_mode"]) m_simConfig.run_mode = tc["run_mode"].as<int>();
            if (tc["internal_memory_mode"]) m_simConfig.internal_memory_mode = tc["internal_memory_mode"].as<int>();
            if (tc["freq_convert_type"]) m_simConfig.freq_convert_type = tc["freq_convert_type"].as<int>();
            if (tc["sc_sampling"]) m_simConfig.sc_sampling = tc["sc_sampling"].as<int>();
            if (tc["proc_sig_freq"]) m_simConfig.proc_sig_freq = tc["proc_sig_freq"].as<int>();
            if (tc["optional_cfr_dim"]) m_simConfig.optional_cfr_dim = tc["optional_cfr_dim"].as<int>();
            if (tc["cpu_only_mode"]) m_simConfig.cpu_only_mode = tc["cpu_only_mode"].as<int>();
        }

        // TODO: add support for cpu_only_mode in cuMAC later
        if (m_simConfig.cpu_only_mode == 1) {
            throw std::runtime_error("Error: CPU only mode is not supported in cuMAC yet");
        }

        // Read Antenna Panel Configurations (needed since cuMAC only specifies total antenna counts)
        if (config["antenna_panels"]) {
            const YAML::Node& ap = config["antenna_panels"];
            m_extConfig.ant_panel_config.clear();

            // Read panel_0 (BS panel)
            if (ap["panel_0"]) {
                AntPanelConfig panel_0;
                const YAML::Node& p0 = ap["panel_0"];
                panel_0.nAnt = p0["n_ant"].as<uint32_t>();
                panel_0.antModel = p0["ant_model"].as<int>();
                
                // Read ant_size array
                const YAML::Node& ant_size = p0["ant_size"];
                if (ant_size.size() != 5) {
                    throw std::runtime_error("ant_size must have exactly 5 elements");
                }
                for (int i = 0; i < 5; ++i) {
                    panel_0.antSize[i] = ant_size[i].as<uint32_t>();
                }
                
                // Read ant_spacing array
                const YAML::Node& ant_spacing = p0["ant_spacing"];
                if (ant_spacing.size() != 4) {
                    throw std::runtime_error("ant_spacing must have exactly 4 elements");
                }
                for (int i = 0; i < 4; ++i) {
                    panel_0.antSpacing[i] = ant_spacing[i].as<float>();
                }
                
                // Read ant_polar_angles array
                const YAML::Node& ant_polar_angles = p0["ant_polar_angles"];
                if (ant_polar_angles.size() != 2) {
                    throw std::runtime_error("ant_polar_angles must have exactly 2 elements");
                }
                for (int i = 0; i < 2; ++i) {
                    panel_0.antPolarAngles[i] = ant_polar_angles[i].as<float>();
                }
                
                m_extConfig.ant_panel_config.push_back(panel_0);
            }

            // Read panel_1 (UE panel)
            if (ap["panel_1"]) {
                AntPanelConfig panel_1;
                const YAML::Node& p1 = ap["panel_1"];
                panel_1.nAnt = p1["n_ant"].as<uint32_t>();
                panel_1.antModel = p1["ant_model"].as<int>();
                
                // Read ant_size array
                const YAML::Node& ant_size = p1["ant_size"];
                if (ant_size.size() != 5) {
                    throw std::runtime_error("ant_size must have exactly 5 elements");
                }
                for (int i = 0; i < 5; ++i) {
                    panel_1.antSize[i] = ant_size[i].as<uint32_t>();
                }
                
                // Read ant_spacing array
                const YAML::Node& ant_spacing = p1["ant_spacing"];
                if (ant_spacing.size() != 4) {
                    throw std::runtime_error("ant_spacing must have exactly 4 elements");
                }
                for (int i = 0; i < 4; ++i) {
                    panel_1.antSpacing[i] = ant_spacing[i].as<float>();
                }
                
                // Read ant_polar_angles array
                const YAML::Node& ant_polar_angles = p1["ant_polar_angles"];
                if (ant_polar_angles.size() != 2) {
                    throw std::runtime_error("ant_polar_angles must have exactly 2 elements");
                }
                for (int i = 0; i < 2; ++i) {
                    panel_1.antPolarAngles[i] = ant_polar_angles[i].as<float>();
                }
                
                m_extConfig.ant_panel_config.push_back(panel_1);
            }
        }
        
    } catch (const YAML::Exception& e) {
        throw std::runtime_error("Error parsing embedded SLS configuration: " + std::string(e.what()));
    }
}

void mMimoNetwork::setupApiStructs()
{
    // setup CPU buffers
    m_prdMatCpu.resize(m_nCell*m_nPrbGrp*m_nBsAnt*cumac::maxNumLayerPerGrpDL_);
    m_allocSolCpu.resize(2*m_nActiveUe);
    m_mcsSelSolCpu.resize(m_nActiveUe);
    m_layerSelSolCpu.resize(m_nActiveUe);
    m_ueOrderInGrpCpu.resize(m_nActiveUe);
    m_setSchdUePerCellTTICpu.resize(m_nCell*m_numUeForGrpPerCell);
    m_nSCIDCpu.resize(m_nActiveUe);
    
    m_cellAssocActUe.resize(m_nCell*m_nActiveUe);
    std::fill(m_cellAssocActUe.begin(), m_cellAssocActUe.end(), 0);
    m_srsEstChanPtrArr.resize(m_nCell);
    m_srsUeMapPtrArr.resize(m_nCell);
    m_sortedUeListPtrArr.resize(m_nCell);
    for (int i = 0; i < m_nCell; i++) {
        CUDA_CHECK_ERR(cudaMalloc((void **)&m_srsEstChanPtrArr[i], m_nCell*m_numUeForGrpPerCell*m_nPrbGrp*m_nUeAnt*m_nBsAnt*sizeof(cuComplex)));
        CUDA_CHECK_ERR(cudaMalloc((void **)&m_srsUeMapPtrArr[i], m_nActiveUe*sizeof(int32_t)));
        CUDA_CHECK_ERR(cudaMalloc((void **)&m_sortedUeListPtrArr[i], m_nMaxActUePerCell*sizeof(uint16_t)));

        for (int uIdx = 0; uIdx < m_nActiveUePerCell; uIdx++) {
            m_cellAssocActUe[i*m_nActiveUe + i*m_nActiveUePerCell + uIdx] = 1;
        }
    }
    m_avgRatesActUe.resize(m_nActiveUe);
    std::fill(m_avgRatesActUe.begin(), m_avgRatesActUe.end(), 1.0);
    m_newDataActUe.resize(m_nActiveUe);
    std::fill(m_newDataActUe.begin(), m_newDataActUe.end(), 1);
    m_tbErrLast.resize(m_nActiveUe);
    std::fill(m_tbErrLast.begin(), m_tbErrLast.end(), 0);
    if (m_riBasedLayerSelSu == 1) {
        m_riActUe.resize(m_nActiveUe);
        std::fill(m_riActUe.begin(), m_riActUe.end(), 4);
    }
    if (m_mcsSelCqi == 1) { 
        m_cqiActUe.resize(m_nActiveUe);
        std::fill(m_cqiActUe.begin(), m_cqiActUe.end(), 0);
    } 
    m_wbSinr.resize(m_nActiveUe*m_nUeAnt);
    std::fill(m_wbSinr.begin(), m_wbSinr.end(), 0);
    for (int i = 0; i < m_nActiveUe; i++) {
        if (m_mcsSelLutType == 0) {
            m_wbSinr[i*m_nUeAnt] = 0.3491; // db2pow(-4.57)
        } else {
            m_wbSinr[i*m_nUeAnt] = 0.3715; // db2pow(-4.3)
        }
    }
    m_srsWbSnr.resize(m_nActiveUe);
    std::fill(m_srsWbSnr.begin(), m_srsWbSnr.end(), 0);
    m_beamformGainLastTx.resize(m_nActiveUe);
    std::fill(m_beamformGainLastTx.begin(), m_beamformGainLastTx.end(), -100.0);
    m_beamformGainCurrTx.resize(m_nActiveUe);
    std::fill(m_beamformGainCurrTx.begin(), m_beamformGainCurrTx.end(), -100.0);
    m_bfGainPrgCurrTx.resize(m_nActiveUe*m_nPrbGrp);
    std::fill(m_bfGainPrgCurrTx.begin(), m_bfGainPrgCurrTx.end(), -100.0);

    m_muGrpListPtr = std::make_unique<cumac::multiCellMuGrpList>();

    // setup CPU & GPU buffers for generated channels
    genChanCpu.resize(m_nCell);
    CUDA_CHECK_ERR(cudaMalloc((void **)&genChanGpu, m_nCell*sizeof(cuComplex*)));
    m_genChanPtrArr.resize(m_nCell);
    for (int i = 0; i < m_nCell; i++) {
        CUDA_CHECK_ERR(cudaMalloc((void **)&m_genChanPtrArr[i], m_nActiveUe*m_nPrbGrp*m_nUeAnt*m_nBsAnt*sizeof(cuComplex)));
        genChanCpu[i].resize(m_nActiveUe*m_nPrbGrp*m_nUeAnt*m_nBsAnt);
    }
    CUDA_CHECK_ERR(cudaMemcpyAsync(genChanGpu, m_genChanPtrArr.data(), m_nCell*sizeof(cuComplex*), cudaMemcpyHostToDevice, m_strm));

    // setup API structures
    cellGrpPrmsGpu->dlSchInd = m_DL;
    cellGrpPrmsGpu->harqEnabledInd = m_harqEnabled;
    cellGrpPrmsGpu->ueGrpMode = m_ueGrpMode;
    cellGrpPrmsGpu->muGrpUpdate = m_muGrpUpdate;
    cellGrpPrmsGpu->nCell = m_nCell;
    cellGrpPrmsGpu->semiStatFreqAlloc = m_semiStatFreqAlloc;
    cellGrpPrmsGpu->nActiveUe = m_nActiveUe;
    cellGrpPrmsGpu->numUeForGrpPerCell = m_numUeForGrpPerCell;
    cellGrpPrmsGpu->numUeSchdPerCellTTI = m_numUeSchdPerCellTTI;
    cellGrpPrmsGpu->nMaxActUePerCell = m_nMaxActUePerCell;
    cellGrpPrmsGpu->nMaxUePerGrpUl = m_nMaxUePerGrpUl;
    cellGrpPrmsGpu->nMaxUePerGrpDl = m_nMaxUePerGrpDl;
    cellGrpPrmsGpu->nMaxLayerPerGrpUl = m_nMaxLayerPerGrpUl;
    cellGrpPrmsGpu->nMaxLayerPerGrpDl = m_nMaxLayerPerGrpDl;
    cellGrpPrmsGpu->nMaxLayerPerUeSuUl = m_nMaxLayerPerUeSuUl;
    cellGrpPrmsGpu->nMaxLayerPerUeSuDl = m_nMaxLayerPerUeSuDl;
    cellGrpPrmsGpu->nMaxLayerPerUeMuUl = m_nMaxLayerPerUeMuUl;
    cellGrpPrmsGpu->nMaxLayerPerUeMuDl = m_nMaxLayerPerUeMuDl;
    cellGrpPrmsGpu->nMaxUegPerCellDl = m_nMaxUegPerCellDl;
    cellGrpPrmsGpu->nMaxUegPerCellUl = m_nMaxUegPerCellUl;
    cellGrpPrmsGpu->nPrbGrp = m_nPrbGrp;
    cellGrpPrmsGpu->nBsAnt = m_nBsAnt;
    cellGrpPrmsGpu->nUeAnt = m_nUeAnt;
    cellGrpPrmsGpu->W = m_W;
    cellGrpPrmsGpu->zfCoeff = m_zfCoeff;
    cellGrpPrmsGpu->betaCoeff = m_betaCoeff;
    cellGrpPrmsGpu->chanCorrThr = m_chanCorrThr;
    cellGrpPrmsGpu->srsSnrThr = m_srsSnrThr;
    cellGrpPrmsGpu->muCoeff = m_muCoeff;
    cellGrpPrmsGpu->muGrpSrsSnrMaxGap = m_muGrpSrsSnrMaxGap;
    cellGrpPrmsGpu->muGrpSrsSnrSplitThr = m_muGrpSrsSnrSplitThr;
    cellGrpPrmsGpu->allocType = m_allocType;
    cellGrpPrmsGpu->bfPowAllocScheme = m_bfPowAllocScheme;
    cellGrpPrmsGpu->mcsSelCqi = m_mcsSelCqi;
    cellGrpPrmsGpu->mcsSelLutType = m_mcsSelLutType;
    cellGrpPrmsGpu->mcsSelSinrCapThr = m_mcsSelSinrCapThr;
    
    CUDA_CHECK_ERR(cudaMalloc((void **)&m_muGrpListPtr->numUeInGrp, cumac::maxNumCoorCells_*cumac::maxNumUegPerCell_*sizeof(uint16_t)));
    CUDA_CHECK_ERR(cudaMalloc((void **)&m_muGrpListPtr->ueId, cumac::maxNumCoorCells_*cumac::maxNumUegPerCell_*cumac::maxNumLayerPerGrpDL_*sizeof(uint16_t)));
    CUDA_CHECK_ERR(cudaMalloc((void **)&m_muGrpListPtr->subbandId, cumac::maxNumCoorCells_*cumac::maxNumUegPerCell_*sizeof(int16_t)));
    CUDA_CHECK_ERR(cudaMalloc((void **)&cellGrpPrmsGpu->cellAssocActUe, m_nCell*m_nActiveUe*sizeof(uint8_t)));
    CUDA_CHECK_ERR(cudaMemcpyAsync(cellGrpPrmsGpu->cellAssocActUe, m_cellAssocActUe.data(), m_nCell*m_nActiveUe*sizeof(uint8_t), cudaMemcpyHostToDevice, m_strm));
    CUDA_CHECK_ERR(cudaMalloc((void **)&cellGrpPrmsGpu->wbSinr, m_nActiveUe*m_nUeAnt*sizeof(float)));
    CUDA_CHECK_ERR(cudaMemcpyAsync(cellGrpPrmsGpu->wbSinr, m_wbSinr.data(), m_nActiveUe*m_nUeAnt*sizeof(float), cudaMemcpyHostToDevice, m_strm));
    CUDA_CHECK_ERR(cudaMalloc((void **)&cellGrpPrmsGpu->prdMat, m_nCell*m_nPrbGrp*m_nBsAnt*cumac::maxNumLayerPerGrpDL_*sizeof(cuComplex)));
    CUDA_CHECK_ERR(cudaMalloc((void **)&cellGrpUeStatusGpu->bfGainPrgCurrTx, m_nActiveUe*m_nPrbGrp*sizeof(float)));
    CUDA_CHECK_ERR(cudaMemcpyAsync(cellGrpUeStatusGpu->bfGainPrgCurrTx, m_bfGainPrgCurrTx.data(), m_nActiveUe*m_nPrbGrp*sizeof(float), cudaMemcpyHostToDevice, m_strm));
    CUDA_CHECK_ERR(cudaMalloc((void **)&cellGrpUeStatusGpu->beamformGainCurrTx, m_nActiveUe*sizeof(float)));
    CUDA_CHECK_ERR(cudaMemcpyAsync(cellGrpUeStatusGpu->beamformGainCurrTx, m_beamformGainCurrTx.data(), m_nActiveUe*sizeof(float), cudaMemcpyHostToDevice, m_strm));
    CUDA_CHECK_ERR(cudaMalloc((void **)&cellGrpUeStatusGpu->beamformGainLastTx, m_nActiveUe*sizeof(float)));
    CUDA_CHECK_ERR(cudaMemcpyAsync(cellGrpUeStatusGpu->beamformGainLastTx, m_beamformGainLastTx.data(), m_nActiveUe*sizeof(float), cudaMemcpyHostToDevice, m_strm));
    CUDA_CHECK_ERR(cudaMalloc((void **)&schdSolGpu->ueOrderInGrp, m_nActiveUe*sizeof(uint16_t)));
    CUDA_CHECK_ERR(cudaMalloc((void **)&schdSolGpu->layerSelSol, m_nActiveUe*sizeof(uint8_t)));
    CUDA_CHECK_ERR(cudaMalloc((void **)&schdSolGpu->allocSol, 2*m_nActiveUe*sizeof(int16_t)));
    CUDA_CHECK_ERR(cudaMalloc((void **)&schdSolGpu->mcsSelSol, m_nActiveUe*sizeof(int16_t)));
    CUDA_CHECK_ERR(cudaMalloc((void **)&schdSolGpu->setSchdUePerCellTTI, m_nCell*m_numUeForGrpPerCell*sizeof(uint16_t)));
    CUDA_CHECK_ERR(cudaMalloc((void **)&cellGrpPrmsGpu->srsEstChan, m_nCell*sizeof(cuComplex*)));
    CUDA_CHECK_ERR(cudaMemcpyAsync(cellGrpPrmsGpu->srsEstChan, m_srsEstChanPtrArr.data(), m_nCell*sizeof(cuComplex*), cudaMemcpyHostToDevice, m_strm));
    CUDA_CHECK_ERR(cudaMalloc((void **)&cellGrpPrmsGpu->srsUeMap, m_nCell*sizeof(int32_t*)));
    CUDA_CHECK_ERR(cudaMemcpyAsync(cellGrpPrmsGpu->srsUeMap, m_srsUeMapPtrArr.data(), m_nCell*sizeof(int32_t*), cudaMemcpyHostToDevice, m_strm));
    CUDA_CHECK_ERR(cudaMalloc((void **)&schdSolGpu->muGrpList, sizeof(cumac::multiCellMuGrpList)));
    CUDA_CHECK_ERR(cudaMemcpyAsync(schdSolGpu->muGrpList, m_muGrpListPtr.get(), sizeof(cumac::multiCellMuGrpList), cudaMemcpyHostToDevice, m_strm));
    CUDA_CHECK_ERR(cudaMalloc((void **)&cellGrpPrmsGpu->srsWbSnr, m_nActiveUe*sizeof(float)));
    CUDA_CHECK_ERR(cudaMemcpyAsync(cellGrpPrmsGpu->srsWbSnr, m_srsWbSnr.data(), m_nActiveUe*sizeof(float), cudaMemcpyHostToDevice, m_strm));
    CUDA_CHECK_ERR(cudaMalloc((void **)&cellGrpPrmsGpu->blerTargetActUe, m_nActiveUe*sizeof(float)));
    std::vector<float> blerTargetActUe(m_nActiveUe, 0.1);
    CUDA_CHECK_ERR(cudaMemcpyAsync(cellGrpPrmsGpu->blerTargetActUe, blerTargetActUe.data(), sizeof(float)*m_nActiveUe, cudaMemcpyHostToDevice, m_strm));
    if (m_riBasedLayerSelSu == 1) {
        CUDA_CHECK_ERR(cudaMalloc((void **)&cellGrpUeStatusGpu->riActUe, m_nActiveUe*sizeof(int8_t)));
        CUDA_CHECK_ERR(cudaMemcpyAsync(cellGrpUeStatusGpu->riActUe, m_riActUe.data(), m_nActiveUe*sizeof(int8_t), cudaMemcpyHostToDevice, m_strm));
    } else {
        cellGrpUeStatusGpu->riActUe = nullptr;
    }
    if (m_mcsSelCqi == 1) { 
        CUDA_CHECK_ERR(cudaMalloc((void **)&cellGrpUeStatusGpu->cqiActUe, m_nActiveUe*sizeof(int8_t)));
        CUDA_CHECK_ERR(cudaMemcpyAsync(cellGrpUeStatusGpu->cqiActUe, m_cqiActUe.data(), m_nActiveUe*sizeof(int8_t), cudaMemcpyHostToDevice, m_strm));
    } else {
        cellGrpUeStatusGpu->cqiActUe = nullptr;
    }
    CUDA_CHECK_ERR(cudaMalloc((void **)&cellGrpUeStatusGpu->avgRatesActUe, m_nActiveUe*sizeof(float)));
    CUDA_CHECK_ERR(cudaMemcpyAsync(cellGrpUeStatusGpu->avgRatesActUe, m_avgRatesActUe.data(), m_nActiveUe*sizeof(float), cudaMemcpyHostToDevice, m_strm));
    CUDA_CHECK_ERR(cudaMalloc((void **)&cellGrpUeStatusGpu->allocSolLastTx, 2*m_nActiveUe*sizeof(int16_t)));
    CUDA_CHECK_ERR(cudaMalloc((void **)&cellGrpUeStatusGpu->layerSelSolLastTx, m_nActiveUe*sizeof(uint8_t)));
    CUDA_CHECK_ERR(cudaMalloc((void **)&cellGrpUeStatusGpu->mcsSelSolLastTx, m_nActiveUe*sizeof(int16_t)));
    if (m_fullBufferTraffic == 1) {
        cellGrpUeStatusGpu->bufferSize = nullptr;
    } else {
        CUDA_CHECK_ERR(cudaMalloc((void **)&cellGrpUeStatusGpu->bufferSize, m_nActiveUe*sizeof(uint32_t)));
    }
    CUDA_CHECK_ERR(cudaMalloc((void **)&cellGrpUeStatusGpu->lastSchdSlotActUe, m_nActiveUe*sizeof(uint32_t)));
    CUDA_CHECK_ERR(cudaMalloc((void **)&cellGrpPrmsGpu->currSlotIdxPerCell, m_nCell*sizeof(uint32_t)));
    CUDA_CHECK_ERR(cudaMalloc((void **)&cellGrpUeStatusGpu->newDataActUe, m_nActiveUe*sizeof(int8_t)));
    CUDA_CHECK_ERR(cudaMemcpyAsync(cellGrpUeStatusGpu->newDataActUe, m_newDataActUe.data(), m_nActiveUe*sizeof(int8_t), cudaMemcpyHostToDevice, m_strm));
    CUDA_CHECK_ERR(cudaMalloc((void **)&cellGrpUeStatusGpu->tbErrLast, m_nActiveUe*sizeof(int8_t)));
    CUDA_CHECK_ERR(cudaMemcpyAsync(cellGrpUeStatusGpu->tbErrLast, m_tbErrLast.data(), m_nActiveUe*sizeof(int8_t), cudaMemcpyHostToDevice, m_strm));
    CUDA_CHECK_ERR(cudaMalloc((void **)&schdSolGpu->sortedUeList, m_nCell*sizeof(uint16_t*)));
    CUDA_CHECK_ERR(cudaMemcpyAsync(schdSolGpu->sortedUeList, m_sortedUeListPtrArr.data(), m_nCell*sizeof(uint16_t*), cudaMemcpyHostToDevice, m_strm));
    CUDA_CHECK_ERR(cudaMalloc((void **)&schdSolGpu->nSCID, m_nActiveUe*sizeof(uint8_t)));
    CUDA_CHECK_ERR(cudaMalloc((void **)&schdSolGpu->muMimoInd, m_nActiveUe*sizeof(uint8_t)));
}

void mMimoNetwork::destroyApiStructs()
{
    for (int i = 0; i < m_nCell; i++) {
        CUDA_CHECK_ERR(cudaFree(m_srsEstChanPtrArr[i]));
        CUDA_CHECK_ERR(cudaFree(m_srsUeMapPtrArr[i]));
        CUDA_CHECK_ERR(cudaFree(m_sortedUeListPtrArr[i]));
    }
    CUDA_CHECK_ERR(cudaFree(m_muGrpListPtr->numUeInGrp));
    CUDA_CHECK_ERR(cudaFree(m_muGrpListPtr->ueId));
    CUDA_CHECK_ERR(cudaFree(m_muGrpListPtr->subbandId));
    CUDA_CHECK_ERR(cudaFree(cellGrpPrmsGpu->cellAssocActUe));
    CUDA_CHECK_ERR(cudaFree(cellGrpPrmsGpu->wbSinr));
    CUDA_CHECK_ERR(cudaFree(cellGrpPrmsGpu->prdMat));
    CUDA_CHECK_ERR(cudaFree(cellGrpUeStatusGpu->bfGainPrgCurrTx));
    CUDA_CHECK_ERR(cudaFree(cellGrpUeStatusGpu->beamformGainCurrTx));
    CUDA_CHECK_ERR(cudaFree(cellGrpUeStatusGpu->beamformGainLastTx));
    CUDA_CHECK_ERR(cudaFree(schdSolGpu->ueOrderInGrp));
    CUDA_CHECK_ERR(cudaFree(schdSolGpu->layerSelSol));
    CUDA_CHECK_ERR(cudaFree(schdSolGpu->allocSol));
    CUDA_CHECK_ERR(cudaFree(schdSolGpu->mcsSelSol));
    CUDA_CHECK_ERR(cudaFree(schdSolGpu->setSchdUePerCellTTI));
    CUDA_CHECK_ERR(cudaFree(cellGrpPrmsGpu->srsEstChan));
    CUDA_CHECK_ERR(cudaFree(cellGrpPrmsGpu->srsUeMap));
    CUDA_CHECK_ERR(cudaFree(schdSolGpu->muGrpList));
    CUDA_CHECK_ERR(cudaFree(cellGrpPrmsGpu->srsWbSnr));
    CUDA_CHECK_ERR(cudaFree(cellGrpPrmsGpu->blerTargetActUe));
    if (m_riBasedLayerSelSu == 1) {
        CUDA_CHECK_ERR(cudaFree(cellGrpUeStatusGpu->riActUe));
    }
    if (m_mcsSelCqi == 1) {
        CUDA_CHECK_ERR(cudaFree(cellGrpUeStatusGpu->cqiActUe));
    }
    CUDA_CHECK_ERR(cudaFree(cellGrpUeStatusGpu->avgRatesActUe));
    CUDA_CHECK_ERR(cudaFree(cellGrpUeStatusGpu->allocSolLastTx));
    CUDA_CHECK_ERR(cudaFree(cellGrpUeStatusGpu->layerSelSolLastTx));
    CUDA_CHECK_ERR(cudaFree(cellGrpUeStatusGpu->mcsSelSolLastTx));
    CUDA_CHECK_ERR(cudaFree(cellGrpUeStatusGpu->bufferSize));
    CUDA_CHECK_ERR(cudaFree(cellGrpUeStatusGpu->lastSchdSlotActUe));
    CUDA_CHECK_ERR(cudaFree(cellGrpPrmsGpu->currSlotIdxPerCell));
    CUDA_CHECK_ERR(cudaFree(cellGrpUeStatusGpu->newDataActUe));
    CUDA_CHECK_ERR(cudaFree(cellGrpUeStatusGpu->tbErrLast));
    CUDA_CHECK_ERR(cudaFree(schdSolGpu->sortedUeList));
    CUDA_CHECK_ERR(cudaFree(schdSolGpu->nSCID));
    CUDA_CHECK_ERR(cudaFree(schdSolGpu->muMimoInd));
}

// GPU kernel to add channel estimation error to channel data
// Grid dimension: numCells, Block dimension: min(256/1024, nActiveUe)
// Each block handles one cell, each thread iterates through all UEs (nActiveUe total) for that cell
__global__ void addChannelEstErrorKernel(cuComplex** inputChanPtrArr,
                                         cuComplex** outputChanPtrArr,
                                         const float sqrChanEstNmse,
                                         const float* lsFadingGainDB,
                                         const int nActiveUe,
                                         const int elementsPerLink,
                                         curandState_t* states)
{
    const int cellIdx = blockIdx.x;  // Each block handles one cell
    const int threadId = threadIdx.x;  // Thread within the block
    const int blockSize = blockDim.x;
    
    // Get input and output channel pointers for this cell
    const cuComplex* inputChan = inputChanPtrArr[cellIdx];
    cuComplex* outputChan = outputChanPtrArr[cellIdx];
    
    if (inputChan && outputChan && states && lsFadingGainDB) {
        // Each thread iterates through UEs with stride
        for (int ueIdx = threadId; ueIdx < nActiveUe; ueIdx += blockSize) {
            // Get LS fading gain for this UE from this cell (negative value in dB)
            const float lsGainDB = lsFadingGainDB[cellIdx * nActiveUe + ueIdx]; // TODO: review when selected links are generated
            
            // Convert dB to linear scale factor for error scaling
            const float lsGainLinearSqrt = powf(10.0f, lsGainDB * 0.05f);  // 0.05 = 1/20
            const float errorScaling = sqrChanEstNmse * lsGainLinearSqrt;
            
            // Process all channel elements for this UE
            const int startIdx = ueIdx * elementsPerLink;
            const int endIdx = startIdx + elementsPerLink;
            
            // Use global thread ID for random state indexing
            const int globalThreadId = cellIdx * blockSize + threadId;
            
            for (int elemIdx = startIdx; elemIdx < endIdx; ++elemIdx) {
                // Copy perfect channel data
                const cuComplex perfectChan = inputChan[elemIdx];
                
                // Add LS-fading-scaled channel estimation error
                const float errorReal = INV_SQRT_2 * errorScaling * curand_normal(&states[globalThreadId]);
                const float errorImag = INV_SQRT_2 * errorScaling * curand_normal(&states[globalThreadId]);
                
                outputChan[elemIdx].x = perfectChan.x + errorReal;
                outputChan[elemIdx].y = perfectChan.y + errorImag;
            }
        }
    }
}

__global__ void updateUeMapKernel(int32_t** srsUeMap,
                                  const int numCells,
                                  const int nActiveUe)
{
    // each thread block handles one cell
    const int cellIdx = blockIdx.x;
    const int threadId = threadIdx.x;
    const int blockSize = blockDim.x;
    
    if (cellIdx < numCells && srsUeMap) {
        for (int ueIdx = threadId; ueIdx < nActiveUe; ueIdx += blockSize) {
            srsUeMap[cellIdx][ueIdx] = ueIdx;
        }
    }
}

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
                                  curandState_t*    states)
{
    int globalThdIdx = blockIdx.x*blockDim.x + threadIdx.x;
    int prgIdx = blockIdx.x/numCell;
    int cIdx = blockIdx.x%numCell; 
    int nBsUeAntPrd = numBsAnt*numUeAnt;
    int nPrgBsUeAntPrd = nPrbGrp*nBsUeAntPrd;
    int uIdx = threadIdx.x/nBsUeAntPrd;
    int eIdx = threadIdx.x - uIdx*nBsUeAntPrd;
    
    if (prgIdx == 0) {
        for (int idx = threadIdx.x; idx < nActiveUe; idx += blockDim.x) {
            srsUeMap[cIdx][idx] = idx;
        }
    }
    __syncthreads(); 

    int numUePerRnd = blockDim.x/nBsUeAntPrd;
    int numRound = ceil(static_cast<float>(nActiveUe)/numUePerRnd);
    
    float genChan_real;
    float genChan_imag;
    float channEstErr_real;
    float channEstErr_imag;
    for (int rndIdx = 0; rndIdx < numRound; rndIdx++) {
        int currUeIdx = uIdx + rndIdx*numUePerRnd; 
        if (currUeIdx < nActiveUe) {
            int currChannIdx = currUeIdx*nPrgBsUeAntPrd + prgIdx*nBsUeAntPrd + eIdx;
            float amplDb = chanGainDBGpu[cIdx*nActiveUe + currUeIdx];
            float sqrtAmpl = pow(10.0, (amplDb * 0.05f)); // 0.05 = 1/20

            if (slotIdx > 0) {
                genChan_real = rho*genChanGpu[cIdx][currChannIdx].x + rhoPrime*sqrtAmpl*curand_normal(&states[globalThdIdx]);
                genChan_imag = rho*genChanGpu[cIdx][currChannIdx].y + rhoPrime*sqrtAmpl*curand_normal(&states[globalThdIdx]);
            } else {
                genChan_real = sqrtAmpl*INV_SQRT_2*curand_normal(&states[globalThdIdx]);
                genChan_imag = sqrtAmpl*INV_SQRT_2*curand_normal(&states[globalThdIdx]);
            }

            channEstErr_real = INV_SQRT_2*sqrtAmpl*sqrChanEstNmse*curand_normal(&states[globalThdIdx]);
            channEstErr_imag = INV_SQRT_2*sqrtAmpl*sqrChanEstNmse*curand_normal(&states[globalThdIdx]);
            
            genChanGpu[cIdx][currChannIdx].x = genChan_real;
            genChanGpu[cIdx][currChannIdx].y = genChan_imag;

            srsEstChan[cIdx][currChannIdx].x = genChan_real + channEstErr_real;
            srsEstChan[cIdx][currChannIdx].y = genChan_imag + channEstErr_imag;
        }
    }
    //if (threadIdx.x == 0) {
    //    printf("blockIdx.x = %d, cIdx = %d, prgIdx = %d\n", blockIdx.x, cIdx, prgIdx);
    //}
}

void mMimoNetwork::genNetTopology()
{
    float Angle = M_PI/6.0;
    int bsIdx = 0;

    for (int celli = 0; celli < 7; celli++) {
        float coorX = 0;
        float coorY = 0;
        if (celli > 0) {
            coorX = sin(Angle)*2.0*netData->cellRadius;
            coorY = cos(Angle)*2.0*netData->cellRadius;
            Angle += M_PI/3.0;
        }

        for (int secIdx = 0; secIdx < 3; secIdx++) {
            if (bsIdx < netData->numCell) {
                netData->bsPos[bsIdx][0] = coorX;
                netData->bsPos[bsIdx][1] = coorY;
                netData->bsPos[bsIdx][2] = netData->bsHeight;

                for (int uIdx = 0; uIdx < m_nActiveUePerCell; uIdx++) {
                    float randomAngle = 2.0*M_PI*uniformRealDist(randomEngine)/3.0 - M_PI/3.0; // centered at 0 degree
                    randomAngle += netData->sectorOrien[secIdx];
                    float randomDistance = (netData->cellRadius - netData->minD2Bs)*uniformRealDist(randomEngine) + netData->minD2Bs;

                    netData->uePos[bsIdx*m_nActiveUePerCell+uIdx][0] = cos(randomAngle)*randomDistance+netData->bsPos[bsIdx][0];
                    netData->uePos[bsIdx*m_nActiveUePerCell+uIdx][1] = sin(randomAngle)*randomDistance+netData->bsPos[bsIdx][1];
                    netData->uePos[bsIdx*m_nActiveUePerCell+uIdx][2] = netData->ueHeight;
                }
                bsIdx++;
            } else
                break;
        }

        if (bsIdx == netData->numCell)
            break;
    }
}

void mMimoNetwork::genLSFading()
{
    std::lognormal_distribution<float> ln_distribution(0.0, netData->sfStd);

    // for testing
    std::vector<std::vector<float>> snrDBAssoc(netData->numCell);

    for (int cIdx = 0; cIdx < netData->numCell; cIdx++) { // loop through all cells
        snrDBAssoc[cIdx].resize(m_nActiveUePerCell); 
        int sectorIdx = cIdx % 3;
        float sectorOrien = netData->sectorOrien[sectorIdx];
        for (int uIdx = 0; uIdx < m_nActiveUe; uIdx++) { // loop through all active UEs
            float distanceBsUe_2D = sqrt(pow(netData->bsPos[cIdx][0] - netData->uePos[uIdx][0], 2.0)+
                                    pow(netData->bsPos[cIdx][1] - netData->uePos[uIdx][1], 2.0));
            float deltaH = netData->bsPos[cIdx][2] - netData->uePos[uIdx][2];
            float distanceBsUe_3D = sqrt(pow(distanceBsUe_2D, 2.0) + pow(deltaH, 2.0));

            // antenna attenuation
            float theta = 180.0 - atan(distanceBsUe_2D/deltaH)*180.0/M_PI;
            float phi = atan2(netData->uePos[uIdx][1] - netData->bsPos[cIdx][1], 
                        netData->uePos[uIdx][0] - netData->bsPos[cIdx][0]);
            phi = phi>=0?phi:(2.0*M_PI+phi);
            float degreeGap = phi - sectorOrien;
            if (degreeGap > M_PI) {
                degreeGap = 2.0*M_PI - degreeGap;
            } else if (degreeGap < -M_PI) {
                degreeGap = 2.0*M_PI + degreeGap;
            }
            degreeGap = degreeGap*180.0/M_PI;

            float antAttenVerti = -std::min(12.0*pow((theta-netData->bsAntDownTilt)/65.0, 2.0), 30.0);
            float antAttenHoriz = -std::min(12.0*pow(degreeGap/65.0, 2.0), 30.0);
            float antAtten = -std::min(-(antAttenVerti+antAttenHoriz), float(30.0));
            float antGain = netData->GEmax + antAtten;
            // pathloss + shadow fading
            float PL = 32.4+20.0*log10(netData->carrierFreq)+30.0*log10(distanceBsUe_3D);
            float SF=10.0*log10(ln_distribution(randomEngine));

            netData->chanGainDB[cIdx*m_nActiveUe + uIdx] = antGain - PL + SF;

            if (floor(uIdx/m_nActiveUePerCell) == cIdx) {
                snrDBAssoc[cIdx][uIdx%m_nActiveUePerCell] = 10.0*log10(netData->bsTxPowerPerPrg/netData->noiseVar) + netData->chanGainDB[cIdx*m_nActiveUe + uIdx];
            }
        }
    }

    CUDA_CHECK_ERR(cudaMemcpyAsync(netData->chanGainDBGpu, netData->chanGainDB.data(), netData->numCell*m_nActiveUe*sizeof(float), cudaMemcpyHostToDevice, m_strm));

    std::ofstream file;
    file.open("snr.txt", std::fstream::out);

    file << "snrDBAssoc = [";
    for (int cIdx = 0; cIdx < netData->numCell; cIdx++) {
        for (int uIdx = 0; uIdx < m_nActiveUePerCell; uIdx++) {
            if ((cIdx*m_nActiveUePerCell + uIdx)<(m_nActiveUe - 1)) {
                file << snrDBAssoc[cIdx][uIdx] << " ";
            } else {
                file << snrDBAssoc[cIdx][uIdx] << "];\n\n";
            }
        }
    }

    file.close();
}

void mMimoNetwork::setupChannel()
{
    if (m_fadingType == 0) {
        // Use internal Rayleigh fading - no additional setup needed
        m_useSlsChannel = false;
        genNetTopology();
        genLSFading();
        printf("Setup channel: Using internal Rayleigh fading (genChan64TrKernel)\n");
    } else if (m_fadingType == 1) {
        // Use SLS channel model with embedded configuration
        m_useSlsChannel = true;
        printf("Setup channel: Using SLS channel model with embedded configuration\n");
        
        // Initialize SLS channel model directly with the parsed configuration
        try {
            // Validate and sync antenna configuration between cuMAC and SLS
            if (m_extConfig.ant_panel_config.size() >= 2) {
                const uint32_t slsBsAnts = m_extConfig.ant_panel_config[0].nAnt;
                const uint32_t slsUeAnts = m_extConfig.ant_panel_config[1].nAnt;
                
                if (slsBsAnts != m_nBsAnt) {
                    std::string errorMsg = "BS antenna configuration mismatch - cuMAC: " + std::to_string(m_nBsAnt) + 
                                          ", SLS config: " + std::to_string(slsBsAnts) + 
                                          ". Please ensure antenna configurations are consistent.";
                    throw std::runtime_error(errorMsg);
                }
                if (slsUeAnts != m_nUeAnt) {
                    std::string errorMsg = "UE antenna configuration mismatch - cuMAC: " + std::to_string(m_nUeAnt) + 
                                          ", SLS config: " + std::to_string(slsUeAnts) + 
                                          ". Please ensure antenna configurations are consistent.";
                    throw std::runtime_error(errorMsg);
                }
            }
            
            // Create SLS channel model with the embedded configuration
            m_slsChannelModel = std::make_unique<statisChanModel<float, cuComplex>>(
                &m_simConfig, &m_sysConfig, &m_linkConfig, &m_extConfig, m_seed, m_strm);
            
            printf("SLS channel model initialized successfully with embedded configuration\n");
        } catch (const std::exception& e) {
            std::cerr << "Error initializing SLS channel model: " << e.what() << std::endl;
            printf("Falling back to internal Rayleigh fading\n");
            m_useSlsChannel = false;
        }
    } else {
        printf("Warning: Unknown fading type %u, defaulting to internal Rayleigh fading\n", m_fadingType);
        m_useSlsChannel = false;
    }
}

void mMimoNetwork::genFadingChannGpu(int slotIdx)
{
    if (m_useSlsChannel) {
        genSlsChannelData(slotIdx);
    } else {
        genChan64TrKernel<<<netData->numThrdBlk, netData->numThrdPerBlk, 0, m_strm>>>(genChanGpu, cellGrpPrmsGpu->srsEstChan, cellGrpPrmsGpu->srsUeMap, cellGrpPrmsGpu->srsWbSnr, netData->chanGainDBGpu, m_nPrbGrp, m_nCell, m_nActiveUe, m_nBsAnt, m_nUeAnt, netData->rho, netData->rhoPrime, netData->sqrChanEstNmse, netData->ueTxPowerPerPrg, netData->noiseVar, slotIdx, netData->states);
        CUDA_CHECK_ERR(cudaGetLastError()); // Check kernel launch success
    }
}

void mMimoNetwork::genSlsChannelData(int slotIdx)
{
    try {
        // Run SLS channel model to generate realistic channel data
        // Time progression for continuous fading
        float refTime = slotIdx * 15.0f / static_cast<float>(m_scs);  // slot duration based on subcarrier spacing
        uint8_t continuous_fading = 1;
        
        // Run SLS model with external CFR buffers
        // This will populate our mMIMO channel buffers directly
        std::vector<uint16_t> activeCells(netData->numCell);
        std::vector<std::vector<uint16_t>> activeUts(netData->numCell);

        if (m_activeCellIds.empty()) {
            for (uint16_t cellIdx = 0; cellIdx < netData->numCell; cellIdx++) {
                activeCells[cellIdx] = cellIdx;
                activeUts[cellIdx].resize(m_nActiveUe);
                for (uint16_t ueIdx = 0; ueIdx < m_nActiveUe; ueIdx++) {
                    activeUts[cellIdx][ueIdx] = ueIdx;
                }
            }
        } else {
            for (uint16_t cellIdx = 0; cellIdx < netData->numCell; cellIdx++) {
                activeCells[cellIdx] = m_activeCellIds[cellIdx];
                activeUts[cellIdx].resize(m_nActiveUe);
                for (uint16_t ueIdx = 0; ueIdx < m_nActiveUe; ueIdx++) {
                    activeUts[cellIdx][ueIdx] = ueIdx;
                }
            }
        }

        // Collect all unique UEs that have at least one active link across all cells
        std::set<uint16_t> uniqueActiveUts;
        for (uint32_t cellIdx = 0; cellIdx < netData->numCell; cellIdx++) {
            for (uint16_t ueIdx : activeUts[cellIdx]) {
                uniqueActiveUts.insert(ueIdx);
            }
        }
        
        // Convert set to vector for API compatibility
        std::vector<uint16_t> fullActiveUts(uniqueActiveUts.begin(), uniqueActiveUts.end());
        
        // run SLS channel model
        m_slsChannelModel->run(refTime, continuous_fading, activeCells, activeUts, 
                              {}, {}, {}, {}, {}, {}, m_genChanPtrArr);
        
        // dump SLS channel data to H5 file
        if (std::find(m_dumpChanSlots.begin(), m_dumpChanSlots.end(), static_cast<uint32_t>(slotIdx)) != m_dumpChanSlots.end()) {
            m_slsChannelModel->saveSlsChanToH5File("_cuMAC_slot" + std::to_string(slotIdx));
        }
        
        // dump pathloss, shadowing and antenna gain for channel estimation error (aligned with chanGainDB = antGain - PL + SF)
        m_slsChannelModel->dump_pl_sf_ant_gain_stats(netData->chanGainDB.data(), activeCells, fullActiveUts);
        CUDA_CHECK_ERR(cudaMemcpyAsync(netData->chanGainDBGpu, netData->chanGainDB.data(), netData->numCell*m_nActiveUe*sizeof(float), cudaMemcpyHostToDevice, m_strm));

        // Apply channel estimation error to the generated channel data
        // Launch configuration: grid = numCells, block = 256 threads
        const int elementsPerLink = m_nPrbGrp * m_nBsAnt * m_nUeAnt;
        const int blockSize = std::min((uint16_t)256, m_nActiveUe);
        const dim3 gridSize(netData->numCell);  // One block per cell
        dim3 blockDim(blockSize);        
        if (blockSize * netData->numCell > netData->numThrdBlk*netData->numThrdPerBlk) { // number of curand states is smaller than the number of threads
            blockDim.x = (netData->numThrdBlk*netData->numThrdPerBlk + blockSize - 1) / blockSize; // adjust block size to fit the number of curand states
        }
        
        addChannelEstErrorKernel<<<gridSize, blockDim, 0, m_strm>>>(
            genChanGpu, cellGrpPrmsGpu->srsEstChan,
            netData->sqrChanEstNmse, netData->chanGainDBGpu,
            m_nActiveUe, elementsPerLink,
            netData->states);
        CUDA_CHECK_ERR(cudaGetLastError()); // Check kernel launch success
        
        // Update UE mapping once for all cells
        const size_t totalUeMapElements = netData->numCell * m_nActiveUe;
        dim3 ueMapBlockSize(256);
        dim3 ueMapGridSize((totalUeMapElements + ueMapBlockSize.x - 1) / ueMapBlockSize.x);
        
        updateUeMapKernel<<<ueMapGridSize, ueMapBlockSize, 0, m_strm>>>(
            cellGrpPrmsGpu->srsUeMap, netData->numCell, m_nActiveUe);
        CUDA_CHECK_ERR(cudaGetLastError()); // Check kernel launch success
        
        // Synchronize to ensure all kernels complete
        CUDA_CHECK_ERR(cudaStreamSynchronize(m_strm));
        
    } catch (const std::exception& e) {
        std::cerr << "Error generating SLS channel data: " << e.what() << std::endl;
        exit(1);
    }
}

void mMimoNetwork::phyAbstract(int slotIdx)
{

}

void mMimoNetwork::updateDataRatePdschGpu(int slotIdx)
{
    //for (int cIdx = 0; cIdx < m_nCell; cIdx++) {
    //    CUDA_CHECK_ERR(cudaMemcpy(genChanCpu)

}

void mMimoNetwork::copySolutionToCpu()
{
    CUDA_CHECK_ERR(cudaMemcpy(m_mcsSelSolCpu.data(), schdSolGpu->mcsSelSol, m_nActiveUe*sizeof(int16_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK_ERR(cudaMemcpy(m_layerSelSolCpu.data(), schdSolGpu->layerSelSol, m_nActiveUe*sizeof(uint8_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK_ERR(cudaMemcpy(m_ueOrderInGrpCpu.data(), schdSolGpu->ueOrderInGrp, m_nActiveUe*sizeof(uint16_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK_ERR(cudaMemcpy(m_setSchdUePerCellTTICpu.data(), schdSolGpu->setSchdUePerCellTTI, m_nCell*m_numUeForGrpPerCell*sizeof(uint16_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK_ERR(cudaMemcpy(m_allocSolCpu.data(), schdSolGpu->allocSol, 2*m_nActiveUe*sizeof(int16_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK_ERR(cudaMemcpy(m_nSCIDCpu.data(), schdSolGpu->nSCID, m_nActiveUe*sizeof(uint8_t), cudaMemcpyDeviceToHost));
}

void mMimoNetwork::copyGenChanToCpu()
{
    for (int cIdx = 0; cIdx < m_nCell; cIdx++) {
        
    }

}

void mMimoNetwork::validateSchedSol()
{
    copySolutionToCpu();    

    for (int cIdx = 0; cIdx < m_nCell; cIdx++) {
        std::vector<std::vector<uint16_t>> schdUegs;
        std::vector<std::vector<int16_t>> uegAllocPrg;

        std::vector<std::vector<int16_t>> uegMcsSel;
        std::vector<std::vector<uint8_t>> uegLayerSel;
        std::vector<std::vector<uint8_t>> uegNscid;

        for (int uIdx = 0; uIdx < m_numUeForGrpPerCell; uIdx++) {
            uint16_t schdUeIdx = m_setSchdUePerCellTTICpu[cIdx*m_numUeForGrpPerCell + uIdx];
            if (schdUeIdx != 0xFFFF) {
                int16_t startPrg = m_allocSolCpu[2*schdUeIdx];
                int16_t endPrg = m_allocSolCpu[2*schdUeIdx + 1] - 1;
                int16_t mcsSel = m_mcsSelSolCpu[schdUeIdx];
                uint8_t layerSel = m_layerSelSolCpu[schdUeIdx];
                uint8_t nscid = m_nSCIDCpu[schdUeIdx];
                
                bool found = false;
                for (int i = 0; i < schdUegs.size(); i++) {
                    if (uegAllocPrg[i][0] == startPrg) {
                        found = true;
                        schdUegs[i].push_back(schdUeIdx);
                        uegMcsSel[i].push_back(mcsSel);
                        uegLayerSel[i].push_back(layerSel);
                        uegNscid[i].push_back(nscid);
                        break;
                    }
                }
                if (!found) {
                    schdUegs.push_back({schdUeIdx});
                    uegAllocPrg.push_back({startPrg, endPrg});
                    uegMcsSel.push_back({mcsSel});
                    uegLayerSel.push_back({layerSel});
                    uegNscid.push_back({nscid});
                }
            }
        }

        printf("Cell #%d: %d UE groups scheduled\n", cIdx, schdUegs.size());
        for (int i = 0; i < schdUegs.size(); i++) {
            printf("    UE group #%d: %d UEs, startPrg = %d, endPrg = %d, total number of layers = %d\n", i, schdUegs[i].size(), uegAllocPrg[i][0], uegAllocPrg[i][1], std::accumulate(uegLayerSel[i].begin(), uegLayerSel[i].end(), 0));
            printf("        UE IDs: ");
            for (int j = 0; j < schdUegs[i].size(); j++) {
                printf("%d ", schdUegs[i][j]);
            }
            printf("\n");
            printf("        MCS: ");
            for (int j = 0; j < uegMcsSel[i].size(); j++) {
                printf("%d ", uegMcsSel[i][j]);
            }
            printf("\n");
            printf("        Number of layers: ");
            for (int j = 0; j < uegLayerSel[i].size(); j++) {
                printf("%d ", uegLayerSel[i][j]);
            }
            printf("\n");
            printf("        nSCID: ");
            for (int j = 0; j < uegNscid[i].size(); j++) {
                printf("%d ", uegNscid[i][j]);
            }
            printf("\n");
        }
    }
}

__global__ void init_curand(unsigned int t_seed, int id_offset, curandState *state)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	curand_init(t_seed, idx + id_offset, 0, &state[idx]);
}

bool mMimoNetwork::hasExtension(const std::string& filename, const std::string& ext) {
    if (filename.length() < ext.length()) return false;
    return std::equal(ext.rbegin(), ext.rend(), filename.rbegin(),
                      [](char a, char b) { return tolower(a) == tolower(b); });
}

bool mMimoNetwork::isYamlFile(const std::string& path) {
    return hasExtension(path, ".yaml") || hasExtension(path, ".yml");
}

bool mMimoNetwork::isHdf5File(const std::string& path) {
    return hasExtension(path, ".h5") || hasExtension(path, ".hdf5");
}