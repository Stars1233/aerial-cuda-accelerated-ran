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

#ifndef ANTENNA_CONFIG_READER_HPP
#define ANTENNA_CONFIG_READER_HPP

#include "chanModelsDataset.hpp"
#include <yaml-cpp/yaml.h>
#include <string>
#include <vector>
#include <stdexcept>
#include <regex>
#include <algorithm>

class AntennaConfigReader {
private:
    static AntPanelConfig parsePanelConfig(const YAML::Node& panel_node) {
        AntPanelConfig panel;
        
        panel.nAnt = panel_node["nAnt"].as<uint16_t>();
        
        // Read antSize array
        const YAML::Node& antSize = panel_node["antSize"];
        if (antSize.size() != 5) {
            throw std::runtime_error("antSize must have exactly 5 elements");
        }
        for (int i = 0; i < 5; ++i) {
            panel.antSize[i] = antSize[i].as<uint16_t>();
        }
        
        // Read antSpacing array
        const YAML::Node& antSpacing = panel_node["antSpacing"];
        if (antSpacing.size() != 4) {
            throw std::runtime_error("antSpacing must have exactly 4 elements");
        }
        for (int i = 0; i < 4; ++i) {
            panel.antSpacing[i] = antSpacing[i].as<float>();
        }

        // Read antModel first to determine if antTheta and antPhi arrays are needed
        panel.antModel = panel_node["antModel"].as<int>();
        
        // Validate antModel value
        if (panel.antModel > 2) {
            throw std::runtime_error("antModel must be 0, 1, or 2");
        }
        
        // Only read and validate antTheta and antPhi arrays if antModel == 2 (direct pattern)
        if (panel.antModel == 2) {
            // Read antTheta array
            const YAML::Node& antTheta = panel_node["antTheta"];
            if (antTheta.size() != 181) {
                throw std::runtime_error("antTheta must have exactly 181 elements");
            }
            for (int i = 0; i < 181; ++i) {
                panel.antTheta[i] = antTheta[i].as<float>();
            }

            // Read antPhi array
            const YAML::Node& antPhi = panel_node["antPhi"];
            if (antPhi.size() != 360) {
                throw std::runtime_error("antPhi must have exactly 360 elements");
            }
            for (int i = 0; i < 360; ++i) {
                panel.antPhi[i] = antPhi[i].as<float>();
            }
        } else {
            // Zero-initialize antTheta and antPhi arrays when antModel is not 2
            std::fill_n(panel.antTheta, 181, 0.0f);
            std::fill_n(panel.antPhi, 360, 0.0f);
        }
        
        // Read antPolarAngles array
        const YAML::Node& antPolarAngles = panel_node["antPolarAngles"];
        if (antPolarAngles.size() != 2) {
            throw std::runtime_error("antPolarAngles must have exactly 2 elements");
        }
        for (int i = 0; i < 2; ++i) {
            panel.antPolarAngles[i] = antPolarAngles[i].as<float>();
        }
        return panel;
    }

public:
    static std::vector<AntPanelConfig> readConfig(const std::string& config_file) {
        try {
            YAML::Node config = YAML::LoadFile(config_file);
            std::vector<AntPanelConfig> panels;
            
            // Regular expression to match panel keys
            std::regex panel_pattern("^panel_\\d+$");
            
            // Check if antenna_panels node exists (preferred structure)
            YAML::Node panels_node;
            if (config["antenna_panels"]) {
                panels_node = config["antenna_panels"];
            } else {
                // Fall back to root level for backward compatibility
                panels_node = config;
            }
            
            // Create pairs of (panel_index, panel_config) to maintain proper ordering
            std::vector<std::pair<int, AntPanelConfig>> indexed_panels;
            
            // Iterate through all nodes in the panels_node
            for (const auto& node : panels_node) {
                const std::string& key = node.first.as<std::string>();
                
                // Check if the key matches the panel pattern
                if (std::regex_match(key, panel_pattern)) {
                    // Extract panel index from key (e.g., "panel_0" -> 0)
                    int panel_index = std::stoi(key.substr(6)); // Skip "panel_" prefix
                    indexed_panels.emplace_back(panel_index, parsePanelConfig(node.second));
                }
            }
            
            // Sort panels by their actual index to maintain consistent order
            std::sort(indexed_panels.begin(), indexed_panels.end(), 
                [](const std::pair<int, AntPanelConfig>& a, const std::pair<int, AntPanelConfig>& b) {
                    return a.first < b.first;  // Sort by panel index
                });
            
            // Extract just the panel configurations in sorted order
            for (const auto& indexed_panel : indexed_panels) {
                panels.push_back(indexed_panel.second);
            }
            
            return panels;
        }
        catch (const YAML::Exception& e) {
            throw std::runtime_error("Error parsing YAML file: " + std::string(e.what()));
        }
    }
};

#endif // ANTENNA_CONFIG_READER_HPP 