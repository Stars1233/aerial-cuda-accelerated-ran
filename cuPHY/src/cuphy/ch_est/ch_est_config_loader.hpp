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

#ifndef CUPHY_CH_EST_CONFIG_LOADER_HPP
#define CUPHY_CH_EST_CONFIG_LOADER_HPP

#include <string>
#include <vector>

#include <gsl-lite/gsl-lite.hpp>

#include "cuphy.h"
#include "trt_engine/trt_engine.hpp"

namespace ch_est {
/**
 * @brief FileLoader defines an interface to load configuration files for ch_est.
 * Configuration files are used by the channel estimate factory functions to create
 * and instantiate the correct implementation for a specific use case.
 */
class FileLoader {
public:
    FileLoader() = default;
    virtual ~FileLoader() = default;
    FileLoader(const FileLoader& fileLoader) = default;
    FileLoader& operator=(const FileLoader& fileLoader) = default;
    FileLoader(FileLoader&& fileLoader) = default;
    FileLoader& operator=(FileLoader&& fileLoader) = default;

    /**
     * @brief load filename with configurations.
     * @param filename filename to load
     * @return cuphyStatus_t as SUCCESS or Failure
     */
    [[nodiscard]] virtual cuphyStatus_t load(const std::string& filename) = 0;
};

/**
 * @brief YAML file loader concrete implementation.
 */
class YAMLLoader final : public FileLoader {
public:
    /**
     * @brief load filename with configuration
     * @param filename filename to load from
     * @return cuphyStatus_t SUCCESS or Failure
     */
    [[nodiscard]] cuphyStatus_t load(const std::string &filename) final;

    /**
     * @brief get trt_engine::TrtParams with Tensor Input Parameters
     * @return span<> of the TrtParams, for inputs
     */
    [[nodiscard]] gsl_lite::span<const trt_engine::TrtParams> getInputs() const noexcept { return m_inputs; }

    /**
     * @brief get trt_engine::TrtParams with Tensor Output Parameters
     * @return span<> of the TrtParams, for outputs
     */
    [[nodiscard]] gsl_lite::span<const trt_engine::TrtParams> getOutputs() const noexcept { return m_outputs; }

    /**
     * @brief Get the model filename
     * @return view of the string with the model's filename
     */
    [[nodiscard]] std::string_view getModelFilename() const noexcept { return m_modelFilename; }

    /**
     * @brief return the max batch size
     * @return Max batch size value.
     */
    [[nodiscard]] auto getMaxBatchSize() const noexcept { return m_maxBatchSize; }
private:
    std::string m_modelFilename;
    std::size_t m_maxBatchSize{};
    std::vector<trt_engine::TrtParams> m_inputs;
    std::vector<trt_engine::TrtParams> m_outputs;
};

} // namespace ch_est

#endif //CUPHY_CH_EST_CONFIG_LOADER_HPP
