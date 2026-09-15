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

#include <iostream> // temp, remove

#include "yaml-cpp/yaml.h"

#include "ch_est_config_loader.hpp"

namespace {
    auto parse(const YAML::Node& config, const std::string& section) {
        const auto &data = config[section];
        std::vector<trt_engine::TrtParams> out;
        for(const auto& elem : data) {
            cuphyTrtTensorPrms_t prms{};
            auto name = elem["name"].as<std::string>();
            prms.name = name.c_str();
            prms.dataType = static_cast<decltype(prms.dataType)>(elem["dataType"].as<int>());
            auto vecDims = elem["dims"].as<std::vector<int>>();
            prms.nDims = vecDims.size();
            std::size_t idx{};
            for (auto e: vecDims) {
                prms.dims[idx++] = e;
            }
            out.emplace_back(prms);
        }
        return out;
    }
} // anonymous NS

namespace ch_est {

    /**
     * const char*      name;
     * cuphyDataType_t  dataType;
     * uint8_t          nDims;
     * int              dims[5];
     * @param filename filename to load from
     * @return true if successful, false otherwise
     */
    cuphyStatus_t YAMLLoader::load(const std::string& filename) {
        YAML::Node config;
        try {
            config = YAML::LoadFile(filename);
        } catch (const YAML::BadFile& badFile) {
            NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "Cannot load file: {} err:  {}", filename, badFile.what());
            return CUPHY_STATUS_INVALID_ARGUMENT;
        }
        try {
            m_inputs = parse(config, "inputs");
        } catch (const std::exception& ex) {
            NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "Failed to parse inputs data err: {} ", ex.what());
            return CUPHY_STATUS_INVALID_ARGUMENT;
        }
        try {
            m_outputs = parse(config, "outputs");
        } catch (const std::exception& ex) {
            NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "Failed to parse outputs data err: {}", ex.what());
            return CUPHY_STATUS_INVALID_ARGUMENT;
        }
        try {
            m_modelFilename = config["file"].as<std::string>();
        } catch (const std::exception& ex) {
            NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "Failed to parse filename data err: {}", ex.what());
            return CUPHY_STATUS_INVALID_ARGUMENT;
        }
        try {
            m_maxBatchSize = config["max_batch_size"].as<std::size_t>();
        } catch (const std::exception& ex) {
            NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "Failed to parse max_batch_size data err: {}", ex.what());
            return CUPHY_STATUS_INVALID_ARGUMENT;
        }
        return CUPHY_STATUS_SUCCESS;
    }

} // namespace ch_est
