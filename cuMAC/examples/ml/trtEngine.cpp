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

#include <algorithm>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include "trtEngine.h"

namespace cumac_ml {

void trtLogger::log(Severity severity, const char *msg) noexcept {
    // Only log warnings or more important.
    if (severity <= Severity::kWARNING) {
        std::cout << msg << std::endl;
    }
}


trtEngine::trtEngine(const char* modelPath,
                     const bool parseFromOnnx,
                     const uint32_t maxBatchSize,
                     const std::vector<trtTensorPrms_t>& inputTensorPrms,
                     const std::vector<trtTensorPrms_t>& outputTensorPrms):
m_maxBatchSize(maxBatchSize),
m_inputTensorPrms(inputTensorPrms),
m_outputTensorPrms(outputTensorPrms),
m_numInputs(inputTensorPrms.size()),
m_numOutputs(outputTensorPrms.size())
{
    if(parseFromOnnx)
        buildFromOnnx(modelPath);
    else
        buildFromTrt(modelPath);
}


void trtEngine::buildFromTrt(const char* trtModelPath)
{
    std::ifstream engineFile;
    try {
        engineFile.open(trtModelPath, std::ios::binary);
        engineFile.exceptions(std::ifstream::failbit);
    }
    catch(std::ifstream::failure e) {
        std::cerr << "\nModel file " << trtModelPath << " not found!" << std::endl;
        exit(1);
    }

    engineFile.seekg(0, engineFile.end);
    long int fsize = engineFile.tellg();
    engineFile.seekg(0, engineFile.beg);
    std::vector<char> engineData(fsize);
    engineFile.read(engineData.data(), fsize);

    m_runtime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(m_logger));
    m_engine = std::unique_ptr<nvinfer1::ICudaEngine>(m_runtime->deserializeCudaEngine(engineData.data(), fsize));
    m_context = std::unique_ptr<nvinfer1::IExecutionContext>(m_engine->createExecutionContext());
}


void trtEngine::buildFromOnnx(const char* onnxModelPath)
{
    // Create our engine builder.
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(m_logger));

    // Define an explicit batch size and then create the network.
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(0));

    // Create a builder config.
    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());

    // Add two optimization profiles.
    nvinfer1::IOptimizationProfile* profile = builder->createOptimizationProfile();
    for(const auto& inputTensorPrm : m_inputTensorPrms) {

        nvinfer1::Dims dims;
        toNvInferDims(inputTensorPrm.dims, dims);

        profile->setDimensions(inputTensorPrm.name.c_str(), nvinfer1::OptProfileSelector::kOPT, dims);

        dims.d[0] = 1;
        profile->setDimensions(inputTensorPrm.name.c_str(), nvinfer1::OptProfileSelector::kMIN, dims);

        dims.d[0] = m_maxBatchSize;
        profile->setDimensions(inputTensorPrm.name.c_str(), nvinfer1::OptProfileSelector::kMAX, dims);
    }
    config->addOptimizationProfile(profile);

    // Create a parser for reading the ONNX file and parse it.
    auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, m_logger));
    auto parsed = parser->parseFromFile(onnxModelPath, static_cast<int>(nvinfer1::ILogger::Severity::kWARNING));

    std::unique_ptr<nvinfer1::IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
    m_runtime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(m_logger));
    m_engine = std::unique_ptr<nvinfer1::ICudaEngine>(m_runtime->deserializeCudaEngine(plan->data(), plan->size()));
    m_context = std::unique_ptr<nvinfer1::IExecutionContext>(m_engine->createExecutionContext());
}


bool trtEngine::setup(const std::vector<void*>& inputDeviceBuf,
                      const std::vector<void*>& outputDeviceBuf,
                      const uint32_t batchSize)
{
    // Optional value - if not given, use maximum batch size.
    uint32_t currentBatchSize = m_maxBatchSize;
    if(batchSize) {
        currentBatchSize = batchSize;
    }

    bool status;

    // Set correct batch size everywhere.
    for(auto& inputTensorPrm : m_inputTensorPrms) {
        inputTensorPrm.dims[0] = currentBatchSize;

        nvinfer1::Dims dims;
        toNvInferDims(inputTensorPrm.dims, dims);
        status = m_context->setInputShape(inputTensorPrm.name.c_str(), dims);
        if(!status) {
            std::cerr << "Failed to set input tensor shape for tensor " << inputTensorPrm.name << "!" << std::endl;
            return false;
        }
    }

    for(auto& outputTensorPrm : m_outputTensorPrms) {
        outputTensorPrm.dims[0] = currentBatchSize;
    }

    // Set input and output tensor addresses.
    for(int i = 0; i < m_numInputs; i++) {
        std::string inputName = m_inputTensorPrms[i].name;
        status = m_context->setTensorAddress(inputName.c_str(), inputDeviceBuf[i]);
        if(!status) {
            std::cerr << "Failed to set input tensor address for tensor " << inputName << "!" << std::endl;
            return false;
        }
    }
    for(int i = 0; i < m_numOutputs; i++) {
        std::string outputName = m_outputTensorPrms[i].name;
        status = m_context->setTensorAddress(outputName.c_str(), outputDeviceBuf[i]);
        if(!status) {
            std::cerr << "Failed to set output tensor address for tensor " << outputName << "!" << std::endl;
            return false;
        }
    }

    if (!m_context->allInputDimensionsSpecified()) {
        return false;
    }

    return true;
}


void trtEngine::toNvInferDims(const std::vector<int>& shape, nvinfer1::Dims& dims)
{
    dims.nbDims = shape.size();
    std::copy(shape.begin(), shape.end(), dims.d);
}


bool trtEngine::run(cudaStream_t cuStream)
{
    return m_context->enqueueV3(cuStream);
}

} // namespace cumac_ml