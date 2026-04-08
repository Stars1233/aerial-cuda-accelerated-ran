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

#ifndef CONFIG_READER_HPP
#define CONFIG_READER_HPP

#include "chanModelsApi.hpp"
#include <yaml-cpp/yaml.h>
#include <string>
#include <vector>
#include <stdexcept>
#include <cstdint>
#include <unordered_set>

class ConfigReader {
public:
    // read full yaml config file
    static void readConfig(const std::string& config_file,
                          SystemLevelConfig& system_config,
                          LinkLevelConfig& link_config,
                          SimConfig& sim_config,
                          ExternalConfig& external_config) {
        try {
            YAML::Node config = YAML::LoadFile(config_file);
            readConfigFromYamlNode(config, system_config, link_config, sim_config, external_config);
        } catch (const std::exception& e) {
            throw std::runtime_error("Error parsing YAML file: " + std::string(e.what()));
        }
    }

    // easy to embed channel model config in a long yaml file
    static void readConfigFromYamlNode(const YAML::Node& config,
                                  SystemLevelConfig& system_config,
                                  LinkLevelConfig& link_config,
                                  SimConfig& sim_config,
                                  ExternalConfig& external_config) {
        try {
            // Read System Level Configuration
            if (config["system_level"]) {
                const YAML::Node& sl = config["system_level"];
                // Convert string to Scenario enum
                std::string scenario_str = sl["scenario"].as<std::string>();
                if (scenario_str == "UMa") {
                    system_config.scenario = Scenario::UMa;
                } else if (scenario_str == "UMi") {
                    system_config.scenario = Scenario::UMi;
                } else if (scenario_str == "RMa") {
                    system_config.scenario = Scenario::RMa;
                } else if (scenario_str == "Indoor") {
                    system_config.scenario = Scenario::Indoor;
                } else if (scenario_str == "InF") {
                    system_config.scenario = Scenario::InF;
                } else if (scenario_str == "SMa") {
                    system_config.scenario = Scenario::SMa;
                } else {
                    throw std::runtime_error("Invalid scenario: " + scenario_str);
                }
                system_config.isd = sl["isd"].as<float>();
                system_config.n_site = sl["n_site"].as<std::uint32_t>();
                system_config.n_sector_per_site = sl["n_sector_per_site"].as<std::uint8_t>();
                system_config.n_ut = sl["n_ut"].as<std::uint32_t>();
                if (sl["ut_drop_option"]) {
                    system_config.ut_drop_option = sl["ut_drop_option"].as<std::uint8_t>();
                }
                if (sl["ut_cell_2d_dist"] && sl["ut_cell_2d_dist"].IsSequence()) {
                    if (sl["ut_cell_2d_dist"].size() != 2) {
                        throw std::runtime_error("ut_cell_2d_dist must contain exactly 2 values: [min, max]");
                    }
                    system_config.ut_cell_2d_dist[0] = sl["ut_cell_2d_dist"][0].as<float>();
                    system_config.ut_cell_2d_dist[1] = sl["ut_cell_2d_dist"][1].as<float>();
                }
                system_config.n_ut_drop_cells = 0;
                const YAML::Node utDropCells = sl["ut_drop_cells"];
                if (utDropCells) {
                    const uint32_t maxDropCells = static_cast<uint32_t>(sizeof(system_config.ut_drop_cells) / sizeof(system_config.ut_drop_cells[0]));
                    const uint32_t nSector = system_config.n_site * system_config.n_sector_per_site;
                    std::unordered_set<uint32_t> seenCell;
                    auto appendCid = [&](uint32_t cid) {
                        if (cid >= nSector) {
                            throw std::runtime_error("ut_drop_cells contains invalid cell id " + std::to_string(cid) +
                                                     " (must be < n_site * n_sector_per_site = " + std::to_string(nSector) + ")");
                        }
                        if (!seenCell.insert(cid).second) {
                            return;  // duplicate, skip
                        }
                        if (system_config.n_ut_drop_cells >= maxDropCells) {
                            throw std::runtime_error("ut_drop_cells exceeds max supported entries (" + std::to_string(maxDropCells) + ")");
                        }
                        system_config.ut_drop_cells[system_config.n_ut_drop_cells++] = cid;
                    };

                    if (utDropCells.IsNull()) {
                        // Treat null as empty list: allow all cells.
                    } else if (utDropCells.IsScalar()) {
                        // Single number is treated as one-element list.
                        appendCid(utDropCells.as<std::uint32_t>());
                    } else if (utDropCells.IsSequence()) {
                        if (utDropCells.size() > nSector) {
                            throw std::runtime_error("ut_drop_cells contains more entries (" + std::to_string(utDropCells.size()) +
                                                     ") than total cells (" + std::to_string(nSector) + ")");
                        }
                        for (std::size_t i = 0; i < utDropCells.size(); ++i) {
                            appendCid(utDropCells[i].as<std::uint32_t>());
                        }
                    } else {
                        throw std::runtime_error("ut_drop_cells must be null, a single number, or a sequence");
                    }
                }
                system_config.optional_pl_ind = sl["optional_pl_ind"].as<std::uint8_t>();
                system_config.o2i_building_penetr_loss_ind = sl["o2i_building_penetr_loss_ind"].as<std::uint8_t>();
                system_config.o2i_car_penetr_loss_ind = sl["o2i_car_penetr_loss_ind"].as<std::uint8_t>();
                system_config.enable_near_field_effect = sl["enable_near_field_effect"].as<std::uint8_t>();
                system_config.enable_non_stationarity = sl["enable_non_stationarity"].as<std::uint8_t>();
                system_config.force_los_prob[0] = sl["force_los_prob"][0].as<float>();
                system_config.force_los_prob[1] = sl["force_los_prob"][1].as<float>();
                system_config.force_indoor_ratio = sl["force_indoor_ratio"].as<float>();
                system_config.force_ut_speed[0] = sl["force_ut_speed"][0].as<float>();
                system_config.force_ut_speed[1] = sl["force_ut_speed"][1].as<float>();
                system_config.disable_pl_shadowing = sl["disable_pl_shadowing"].as<std::uint8_t>();
                system_config.disable_small_scale_fading = sl["disable_small_scale_fading"].as<std::uint8_t>();
                system_config.enable_per_tti_lsp = sl["enable_per_tti_lsp"].as<std::uint8_t>();
                system_config.enable_propagation_delay = sl["enable_propagation_delay"].as<uint8_t>();
                
                // ISAC Configuration (optional, defaults to communication-only mode)
                if (sl["isac_type"]) {
                    system_config.isac_type = sl["isac_type"].as<std::uint8_t>();
                }
                if (sl["n_st"]) {
                    system_config.n_st = sl["n_st"].as<std::uint32_t>();
                }
                if (sl["st_target_type"]) {
                    const std::uint8_t target_type_raw = sl["st_target_type"].as<std::uint8_t>();
                    constexpr std::uint8_t max_target_type =
                        static_cast<std::uint8_t>(SensingTargetType::HAZARD);
                    if (target_type_raw > max_target_type) {
                        throw std::runtime_error(
                            "Invalid st_target_type value " + std::to_string(target_type_raw) +
                            ". Valid range is [0, " + std::to_string(max_target_type) + "].");
                    }
                    system_config.st_target_type = static_cast<SensingTargetType>(target_type_raw);
                }
                if (sl["st_rcs_model"]) {
                    system_config.st_rcs_model = sl["st_rcs_model"].as<std::uint8_t>();
                }
                if (sl["st_horizontal_speed"]) {
                    const YAML::Node& speedNode = sl["st_horizontal_speed"];
                    if (speedNode.IsSequence()) {
                        if (speedNode.size() != 2) {
                            throw std::runtime_error("st_horizontal_speed must have exactly 2 elements: [min, max]");
                        }
                        system_config.st_horizontal_speed[0] = speedNode[0].as<float>();
                        system_config.st_horizontal_speed[1] = speedNode[1].as<float>();
                    } else {
                        // Backward compatible: scalar means fixed speed
                        const float v = speedNode.as<float>();
                        system_config.st_horizontal_speed[0] = v;
                        system_config.st_horizontal_speed[1] = v;
                    }
                }
                if (sl["st_vertical_velocity"]) {
                    system_config.st_vertical_velocity = sl["st_vertical_velocity"].as<float>();
                }
                if (sl["st_min_dist_from_tx_rx"]) {
                    system_config.st_min_dist_from_tx_rx = sl["st_min_dist_from_tx_rx"].as<float>();
                }
                if (sl["st_minimum_distance"]) {
                    system_config.st_minimum_distance = sl["st_minimum_distance"].as<float>();
                }
                if (sl["st_size_ind"]) {
                    system_config.st_size_ind = sl["st_size_ind"].as<std::uint8_t>();
                }
                if (sl["st_distribution_option"]) {
                    auto dist_opt = sl["st_distribution_option"].as<std::vector<int>>();
                    if (dist_opt.size() < 2) {
                        throw std::runtime_error(
                            "st_distribution_option must contain at least 2 elements, but got " +
                            std::to_string(dist_opt.size()));
                    }
                    system_config.st_distribution_option[0] = static_cast<std::uint8_t>(dist_opt[0]);
                    system_config.st_distribution_option[1] = static_cast<std::uint8_t>(dist_opt[1]);
                }
                // ST height range for vertical Option B
                // New key: st_height: [min, max] (or scalar for fixed height)
                // Legacy key supported: st_fixed_height
                auto parse_height_range = [&](const YAML::Node& heightNode) {
                    if (heightNode.IsSequence()) {
                        if (heightNode.size() != 2) {
                            throw std::runtime_error("st_height must have exactly 2 elements: [min, max]");
                        }
                        system_config.st_height[0] = heightNode[0].as<float>();
                        system_config.st_height[1] = heightNode[1].as<float>();
                    } else {
                        const float h = heightNode.as<float>();
                        system_config.st_height[0] = h;
                        system_config.st_height[1] = h;
                    }
                };
                if (sl["st_height"]) {
                    parse_height_range(sl["st_height"]);
                } else if (sl["st_fixed_height"]) {
                    // Backward compatible: old name
                    parse_height_range(sl["st_fixed_height"]);
                }
                if (sl["st_drop_radius"]) {
                    system_config.st_drop_radius = sl["st_drop_radius"].as<float>();
                }
                if (sl["st_override_k_db"]) {
                    const auto& node = sl["st_override_k_db"];
                    if (node.IsNull()) {
                        system_config.st_override_k_db = std::numeric_limits<float>::quiet_NaN();
                    } else {
                        system_config.st_override_k_db = node.as<float>();
                    }
                }
                if (sl["path_drop_threshold_db"]) {
                    system_config.path_drop_threshold_db = sl["path_drop_threshold_db"].as<float>();
                }
                if (sl["isac_disable_background"]) {
                    system_config.isac_disable_background = sl["isac_disable_background"].as<std::uint8_t>();
                }
                if (sl["isac_disable_target"]) {
                    system_config.isac_disable_target = sl["isac_disable_target"].as<std::uint8_t>();
                }
                
                // Aerial UE Configuration (optional)
                if (sl["aerial_ue_fast_fading_alt"]) {
                    system_config.aerial_ue_fast_fading_alt = sl["aerial_ue_fast_fading_alt"].as<std::uint8_t>();
                }
                if (sl["aerial_ue_height_min"]) {
                    system_config.aerial_ue_height_min = sl["aerial_ue_height_min"].as<float>();
                }
                if (sl["aerial_ue_height_max"]) {
                    system_config.aerial_ue_height_max = sl["aerial_ue_height_max"].as<float>();
                }
            }

            // Read Link Level Configuration
            if (config["link_level"]) {
                const YAML::Node& ll = config["link_level"];
                link_config.fast_fading_type = ll["fast_fading_type"].as<int>();
                std::string delay_str = ll["delay_profile"].as<std::string>();
                link_config.delay_profile = delay_str.empty() ? 'A' : delay_str[0];
                link_config.delay_spread = ll["delay_spread"].as<float>();
                link_config.velocity[0] = ll["velocity"][0].as<float>();
                link_config.velocity[1] = ll["velocity"][1].as<float>();
                link_config.velocity[2] = ll["velocity"][2].as<float>();
                link_config.num_ray = ll["num_ray"].as<int>();
                link_config.cfo_hz = ll["cfo_hz"].as<float>();
                link_config.delay = ll["delay"].as<float>();
            }

            // Read Simulation Configuration
            if (config["simulation"]) {
                const YAML::Node& tc = config["simulation"];
                sim_config.link_sim_ind = tc["link_sim_ind"].as<int>();
                sim_config.center_freq_hz = tc["center_freq_hz"].as<float>();
                sim_config.bandwidth_hz = tc["bandwidth_hz"].as<float>();
                sim_config.sc_spacing_hz = tc["sc_spacing_hz"].as<float>();
                sim_config.fft_size = tc["fft_size"].as<int>();
                sim_config.n_prb = tc["n_prb"].as<int>();
                sim_config.n_prbg = tc["n_prbg"].as<int>();
                sim_config.n_snapshot_per_slot = tc["n_snapshot_per_slot"].as<int>();
                sim_config.run_mode = tc["run_mode"].as<int>();
                sim_config.internal_memory_mode = tc["internal_memory_mode"].as<int>();
                sim_config.freq_convert_type = tc["freq_convert_type"].as<int>();
                sim_config.sc_sampling = tc["sc_sampling"].as<int>();
                sim_config.proc_sig_freq = tc["proc_sig_freq"].as<int>();
                sim_config.optional_cfr_dim = tc["optional_cfr_dim"].as<int>();
                sim_config.cpu_only_mode = tc["cpu_only_mode"].as<int>();
                if (tc["h5_dump_level"]) {
                    sim_config.h5_dump_level = tc["h5_dump_level"].as<int>();
                }
            }

            // Read Antenna Panel Configurations
            if (config["antenna_panels"]) {
                const YAML::Node& ap = config["antenna_panels"];
                external_config.ant_panel_config.clear();

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
                    
                    // Sanity check: verify nAnt equals product of antSize elements
                    uint32_t antSizeProduct = panel_0.antSize[0] * panel_0.antSize[1] * panel_0.antSize[2] * 
                                             panel_0.antSize[3] * panel_0.antSize[4];
                    if (panel_0.nAnt != antSizeProduct) {
                        throw std::runtime_error("panel_0.nAnt (" + std::to_string(panel_0.nAnt) + 
                                                ") does not match product of antSize elements (" + 
                                                std::to_string(antSizeProduct) + ")");
                    }
                    
                    // Read ant_spacing array
                    const YAML::Node& ant_spacing = p0["ant_spacing"];
                    if (ant_spacing.size() != 4) {
                        throw std::runtime_error("ant_spacing must have exactly 4 elements");
                    }
                    for (int i = 0; i < 4; ++i) {
                        panel_0.antSpacing[i] = ant_spacing[i].as<float>();
                    }

                    // Read ant_theta array
                    const YAML::Node& ant_theta = p0["ant_theta"];
                    if (panel_0.antModel == 2) {
                        if (ant_theta.size() != 181) {
                            throw std::runtime_error("ant_theta must have exactly 181 elements when ant_model=2");
                        }
                        for (int i = 0; i < 181; ++i) {
                            panel_0.antTheta[i] = ant_theta[i].as<float>();
                        }
                    }

                    // Read ant_phi array
                    const YAML::Node& ant_phi = p0["ant_phi"];
                    if (panel_0.antModel == 2) {
                        if (ant_phi.size() != 360) {
                            throw std::runtime_error("ant_phi must have exactly 360 elements when ant_model=2");
                        }
                        for (int i = 0; i < 360; ++i) {
                            panel_0.antPhi[i] = ant_phi[i].as<float>();
                        }
                    }
                    
                    // Read ant_polar_angles array
                    const YAML::Node& ant_polar_angles = p0["ant_polar_angles"];
                    if (ant_polar_angles.size() != 2) {
                        throw std::runtime_error("ant_polar_angles must have exactly 2 elements");
                    }
                    for (int i = 0; i < 2; ++i) {
                        panel_0.antPolarAngles[i] = ant_polar_angles[i].as<float>();
                    }
                    
                    external_config.ant_panel_config.push_back(panel_0);
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
                    
                    // Sanity check: verify nAnt equals product of antSize elements
                    uint32_t antSizeProduct = panel_1.antSize[0] * panel_1.antSize[1] * panel_1.antSize[2] * 
                                             panel_1.antSize[3] * panel_1.antSize[4];
                    if (panel_1.nAnt != antSizeProduct) {
                        throw std::runtime_error("panel_1.nAnt (" + std::to_string(panel_1.nAnt) + 
                                                ") does not match product of antSize elements (" + 
                                                std::to_string(antSizeProduct) + ")");
                    }
                    
                    // Read ant_spacing array
                    const YAML::Node& ant_spacing = p1["ant_spacing"];
                    if (ant_spacing.size() != 4) {
                        throw std::runtime_error("ant_spacing must have exactly 4 elements");
                    }
                    for (int i = 0; i < 4; ++i) {
                        panel_1.antSpacing[i] = ant_spacing[i].as<float>();
                    }

                    // Read ant_theta array
                    const YAML::Node& ant_theta = p1["ant_theta"];
                    if (panel_1.antModel == 2) {
                        if (ant_theta.size() != 180) {
                            throw std::runtime_error("ant_theta must have exactly 180 elements when ant_model=2");
                        }
                        for (int i = 0; i < 180; ++i) {
                            panel_1.antTheta[i] = ant_theta[i].as<float>();
                        }
                    }

                    // Read ant_phi array
                    const YAML::Node& ant_phi = p1["ant_phi"];
                    if (panel_1.antModel == 2) {
                        if (ant_phi.size() != 360) {
                            throw std::runtime_error("ant_phi must have exactly 360 elements when ant_model=2");
                        }
                        for (int i = 0; i < 360; ++i) {
                            panel_1.antPhi[i] = ant_phi[i].as<float>();
                        }
                    }
                    
                    // Read ant_polar_angles array
                    const YAML::Node& ant_polar_angles = p1["ant_polar_angles"];
                    if (ant_polar_angles.size() != 2) {
                        throw std::runtime_error("ant_polar_angles must have exactly 2 elements");
                    }
                    for (int i = 0; i < 2; ++i) {
                        panel_1.antPolarAngles[i] = ant_polar_angles[i].as<float>();
                    }
                    
                    external_config.ant_panel_config.push_back(panel_1);
                }
            }

            // Read UT Configurations (optional)
            if (config["external_config"] && config["external_config"]["ut_config"]) {
                const YAML::Node& ut_configs = config["external_config"]["ut_config"];
                external_config.ut_config.clear();
                
                // Support both array format and individual UT format
                if (ut_configs.IsSequence()) {
                    // Array format: ut_config: [ut1, ut2, ...]
                    for (const auto& ut_node : ut_configs) {
                        UtParamCfg ut_param;
                        
                        if (ut_node["uid"]) ut_param.uid = ut_node["uid"].as<uint32_t>();
                        if (ut_node["outdoor_ind"]) ut_param.outdoor_ind = ut_node["outdoor_ind"].as<uint8_t>();
                        if (ut_node["ant_panel_idx"]) ut_param.antPanelIdx = ut_node["ant_panel_idx"].as<uint32_t>();
                        
                        // Read location
                        if (ut_node["loc"]) {
                            const YAML::Node& loc = ut_node["loc"];
                            if (loc.size() >= 3) {
                                ut_param.loc.x = loc[0].as<float>();
                                ut_param.loc.y = loc[1].as<float>();
                                ut_param.loc.z = loc[2].as<float>();
                            }
                        }
                        
                        // Read antenna panel orientation
                        if (ut_node["ant_panel_orientation"]) {
                            const YAML::Node& orientation = ut_node["ant_panel_orientation"];
                            if (orientation.size() >= 3) {
                                ut_param.antPanelOrientation[0] = orientation[0].as<float>();
                                ut_param.antPanelOrientation[1] = orientation[1].as<float>();
                                ut_param.antPanelOrientation[2] = orientation[2].as<float>();
                            }
                        }
                        
                        // Read velocity
                        if (ut_node["velocity"]) {
                            const YAML::Node& velocity = ut_node["velocity"];
                            if (velocity.size() >= 3) {
                                ut_param.velocity[0] = velocity[0].as<float>();
                                ut_param.velocity[1] = velocity[1].as<float>();
                                ut_param.velocity[2] = velocity[2].as<float>();
                            }
                        }
                        
                        external_config.ut_config.push_back(ut_param);
                    }
                } else {
                    // Individual UT format: ut_config: { ut_1: {...}, ut_2: {...} }
                    for (const auto& ut_pair : ut_configs) {
                        const YAML::Node& ut_node = ut_pair.second;
                        UtParamCfg ut_param;
                        
                        if (ut_node["uid"]) ut_param.uid = ut_node["uid"].as<uint32_t>();
                        if (ut_node["outdoor_ind"]) ut_param.outdoor_ind = ut_node["outdoor_ind"].as<uint8_t>();
                        if (ut_node["ant_panel_idx"]) ut_param.antPanelIdx = ut_node["ant_panel_idx"].as<uint32_t>();
                        
                        // Read location
                        if (ut_node["loc"]) {
                            const YAML::Node& loc = ut_node["loc"];
                            if (loc.size() >= 3) {
                                ut_param.loc.x = loc[0].as<float>();
                                ut_param.loc.y = loc[1].as<float>();
                                ut_param.loc.z = loc[2].as<float>();
                            }
                        }
                        
                        // Read antenna panel orientation
                        if (ut_node["ant_panel_orientation"]) {
                            const YAML::Node& orientation = ut_node["ant_panel_orientation"];
                            if (orientation.size() >= 3) {
                                ut_param.antPanelOrientation[0] = orientation[0].as<float>();
                                ut_param.antPanelOrientation[1] = orientation[1].as<float>();
                                ut_param.antPanelOrientation[2] = orientation[2].as<float>();
                            }
                        }
                        
                        // Read velocity
                        if (ut_node["velocity"]) {
                            const YAML::Node& velocity = ut_node["velocity"];
                            if (velocity.size() >= 3) {
                                ut_param.velocity[0] = velocity[0].as<float>();
                                ut_param.velocity[1] = velocity[1].as<float>();
                                ut_param.velocity[2] = velocity[2].as<float>();
                            }
                        }
                        
                        external_config.ut_config.push_back(ut_param);
                    }
                }
            }

            // Read Cell Configurations (optional)
            if (config["external_config"] && config["external_config"]["cell_config"]) {
                const YAML::Node& cell_configs = config["external_config"]["cell_config"];
                external_config.cell_config.clear();
                
                // Support both array format and individual cell format
                if (cell_configs.IsSequence()) {
                    // Array format: cell_config: [cell1, cell2, ...]
                    for (const auto& cell_node : cell_configs) {
                        CellParam cell_param;
                        
                        if (cell_node["cid"]) cell_param.cid = cell_node["cid"].as<uint32_t>();
                        if (cell_node["site_id"]) cell_param.siteId = cell_node["site_id"].as<uint32_t>();
                        if (cell_node["ant_panel_idx"]) cell_param.antPanelIdx = cell_node["ant_panel_idx"].as<uint32_t>();
                        
                        // Read location
                        if (cell_node["loc"]) {
                            const YAML::Node& loc = cell_node["loc"];
                            if (loc.size() >= 3) {
                                cell_param.loc.x = loc[0].as<float>();
                                cell_param.loc.y = loc[1].as<float>();
                                cell_param.loc.z = loc[2].as<float>();
                            }
                        }
                        
                        // Read antenna panel orientation
                        if (cell_node["ant_panel_orientation"]) {
                            const YAML::Node& orientation = cell_node["ant_panel_orientation"];
                            if (orientation.size() >= 3) {
                                cell_param.antPanelOrientation[0] = orientation[0].as<float>();
                                cell_param.antPanelOrientation[1] = orientation[1].as<float>();
                                cell_param.antPanelOrientation[2] = orientation[2].as<float>();
                            }
                        }
                        
                        external_config.cell_config.push_back(cell_param);
                    }
                } else {
                    // Individual cell format: cell_config: { cell_1: {...}, cell_2: {...} }
                    for (const auto& cell_pair : cell_configs) {
                        const YAML::Node& cell_node = cell_pair.second;
                        CellParam cell_param;
                        
                        if (cell_node["cid"]) cell_param.cid = cell_node["cid"].as<uint32_t>();
                        if (cell_node["site_id"]) cell_param.siteId = cell_node["site_id"].as<uint32_t>();
                        if (cell_node["ant_panel_idx"]) cell_param.antPanelIdx = cell_node["ant_panel_idx"].as<uint32_t>();
                        
                        // Read location
                        if (cell_node["loc"]) {
                            const YAML::Node& loc = cell_node["loc"];
                            if (loc.size() >= 3) {
                                cell_param.loc.x = loc[0].as<float>();
                                cell_param.loc.y = loc[1].as<float>();
                                cell_param.loc.z = loc[2].as<float>();
                            }
                        }
                        
                        // Read antenna panel orientation
                        if (cell_node["ant_panel_orientation"]) {
                            const YAML::Node& orientation = cell_node["ant_panel_orientation"];
                            if (orientation.size() >= 3) {
                                cell_param.antPanelOrientation[0] = orientation[0].as<float>();
                                cell_param.antPanelOrientation[1] = orientation[1].as<float>();
                                cell_param.antPanelOrientation[2] = orientation[2].as<float>();
                            }
                        }
                        
                        external_config.cell_config.push_back(cell_param);
                    }
                }
            }
        }
        catch (const YAML::Exception& e) {
            throw std::runtime_error("Error parsing YAML file: " + std::string(e.what()));
        }
    }
};

#endif // CONFIG_READER_HPP 