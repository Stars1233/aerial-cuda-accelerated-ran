/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "cumacAiranSubcontext.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <stdexcept>

namespace cumac_ml {

namespace {

// Throwing CUDA error check (kept local so this TU does not need cumac.h just
// for its CUDA_CHECK_ERR macro, matching trtEngine.cpp).
inline void cudaCheckAiran(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("cumacAiranSubcontext CUDA error (") + what +
                                 "): " + cudaGetErrorString(err));
    }
}

}  // namespace


cumacAiranSubcontext::cumacAiranSubcontext(const std::string& tvFilename,
                                           uint8_t            in_GPU,
                                           uint8_t            in_halfPrecision,
                                           cudaStream_t       strm)
{
    m_GPU           = in_GPU;
    m_halfPrecision = in_halfPrecision;

    airanTvConfig      cfg;
    std::vector<float> hostInput;
    loadAiranTv(tvFilename, cfg, hostInput);

    init(cfg, hostInput, strm);
}


cumacAiranSubcontext::cumacAiranSubcontext(const airanTvConfig&      cfg,
                                           const std::vector<float>& hostInput,
                                           uint8_t                   in_GPU,
                                           uint8_t                   in_halfPrecision,
                                           cudaStream_t              strm)
{
    m_GPU           = in_GPU;
    m_halfPrecision = in_halfPrecision;

    init(cfg, hostInput, strm);
}


void cumacAiranSubcontext::init(const airanTvConfig&      cfg,
                                const std::vector<float>& hostInput,
                                cudaStream_t              strm)
{
    if (m_GPU != 1) {
        throw std::runtime_error(
            "cumacAiranSubcontext: only GPU inference is supported (in_GPU must be 1)");
    }
    if (cfg.modelPath.empty()) {
        throw std::runtime_error("cumacAiranSubcontext: TV has no model path");
    }
    if (cfg.obsDim <= 0 || cfg.actionDim <= 0 || cfg.batchSize <= 0) {
        throw std::runtime_error("cumacAiranSubcontext: invalid TV dimensions");
    }

    m_modelPath         = cfg.modelPath;
    m_inputName         = cfg.inputName.empty() ? "obs" : cfg.inputName;
    m_outputName        = cfg.outputName.empty() ? "logits" : cfg.outputName;
    m_obsDim            = cfg.obsDim;
    m_actionDim         = cfg.actionDim;
    m_batchSize         = cfg.batchSize;
    m_maxBatchSize      = (cfg.maxBatchSize >= cfg.batchSize) ? cfg.maxBatchSize : cfg.batchSize;
    m_copyOutputEachRun = cfg.copyOutputEachRun;

    // Assemble the engine build/run configuration. in_halfPrecision forces FP16;
    // otherwise the TV's stored precision is honored.
    trtEngineConfig engCfg;
    engCfg.modelPath                = m_modelPath;
    engCfg.precision                = (m_halfPrecision != 0 || cfg.precision != 0)
                                          ? trtPrecision::kFP16
                                          : trtPrecision::kFP32;
    engCfg.useCudaGraph             = (cfg.useCudaGraph != 0);
    engCfg.builderOptimizationLevel = cfg.builderOptLevel;
    engCfg.enableEngineCache        = true;
    engCfg.forceRebuild             = false;
    engCfg.inputName                = m_inputName;
    engCfg.outputName               = m_outputName;
    if (!cfg.ptToOnnxScript.empty()) {
        engCfg.ptToOnnxScript = cfg.ptToOnnxScript;
    }
    engCfg.verbose = false;  // keep the subcontext quiet inside the test bench

    const std::vector<trtTensorPrms_t> inputTensorPrms  = {{m_inputName, {m_maxBatchSize, m_obsDim}}};
    const std::vector<trtTensorPrms_t> outputTensorPrms = {{m_outputName, {m_maxBatchSize, m_actionDim}}};

    m_engine = std::make_unique<trtEngine>(engCfg, static_cast<uint32_t>(m_maxBatchSize),
                                           inputTensorPrms, outputTensorPrms);

    // Allocate host (pinned) staging + device I/O buffers, sized for max batch.
    const size_t inElems  = static_cast<size_t>(m_maxBatchSize) * static_cast<size_t>(m_obsDim);
    const size_t outElems = static_cast<size_t>(m_maxBatchSize) * static_cast<size_t>(m_actionDim);

    cudaCheckAiran(cudaMallocHost((void**)&m_hInput, inElems * sizeof(float)), "cudaMallocHost(hInput)");
    cudaCheckAiran(cudaMallocHost((void**)&m_hOutput, outElems * sizeof(float)), "cudaMallocHost(hOutput)");
    cudaCheckAiran(cudaMalloc((void**)&m_dInput, inElems * sizeof(float)), "cudaMalloc(dInput)");
    cudaCheckAiran(cudaMalloc((void**)&m_dOutput, outElems * sizeof(float)), "cudaMalloc(dOutput)");

    // Seed the host input from the TV (zero-fill if none was provided).
    std::memset(m_hInput, 0, inElems * sizeof(float));
    if (!hostInput.empty()) {
        const size_t n = std::min(inElems, hostInput.size());
        std::memcpy(m_hInput, hostInput.data(), n * sizeof(float));
    }
    std::memset(m_hOutput, 0, outElems * sizeof(float));

    cudaCheckAiran(cudaMemcpyAsync(m_dInput, m_hInput, inElems * sizeof(float),
                                   cudaMemcpyHostToDevice, strm),
                   "cudaMemcpyAsync(dInput init)");

    m_inputBuffers  = {(void*)m_dInput};
    m_outputBuffers = {(void*)m_dOutput};
}


cumacAiranSubcontext::~cumacAiranSubcontext()
{
    // Engine must be torn down before the buffers it may reference.
    m_engine.reset();
    if (m_dInput)  cudaFree(m_dInput);
    if (m_dOutput) cudaFree(m_dOutput);
    if (m_hInput)  cudaFreeHost(m_hInput);
    if (m_hOutput) cudaFreeHost(m_hOutput);
}


void cumacAiranSubcontext::setup(const std::string& tvFilename, cudaStream_t strm)
{
    airanTvConfig      cfg;
    std::vector<float> hostInput;
    loadAiranTv(tvFilename, cfg, hostInput);

    // The engine is fixed at construction time; a setup() TV may only refresh
    // the input observations, not change the model geometry.
    if (cfg.obsDim != m_obsDim || cfg.batchSize != m_batchSize ||
        cfg.maxBatchSize != m_maxBatchSize) {
        throw std::runtime_error(
            "cumacAiranSubcontext::setup: TV input geometry differs from the built engine");
    }

    const size_t inElems = static_cast<size_t>(m_maxBatchSize) * static_cast<size_t>(m_obsDim);
    if (!hostInput.empty()) {
        const size_t n = std::min(inElems, hostInput.size());
        std::memcpy(m_hInput, hostInput.data(), n * sizeof(float));
    }
    cudaCheckAiran(cudaMemcpyAsync(m_dInput, m_hInput, inElems * sizeof(float),
                                   cudaMemcpyHostToDevice, strm),
                   "cudaMemcpyAsync(dInput setup)");

    setup(strm);
}


void cumacAiranSubcontext::setup(cudaStream_t /*strm*/)
{
    if (!m_engine->setup(m_inputBuffers, m_outputBuffers, static_cast<uint32_t>(m_batchSize))) {
        throw std::runtime_error("cumacAiranSubcontext::setup: trtEngine setup() failed");
    }
    m_isSetup = true;
}


void cumacAiranSubcontext::run(cudaStream_t strm)
{
    if (!m_isSetup) {
        throw std::runtime_error("cumacAiranSubcontext::run: setup() must be called before run()");
    }
    if (!m_engine->run(strm)) {
        throw std::runtime_error("cumacAiranSubcontext::run: trtEngine run() failed");
    }
    if (m_copyOutputEachRun) {
        copyOutputToHost(strm);
    }
}


void cumacAiranSubcontext::copyOutputToHost(cudaStream_t strm)
{
    const size_t outElems = static_cast<size_t>(m_batchSize) * static_cast<size_t>(m_actionDim);
    cudaCheckAiran(cudaMemcpyAsync(m_hOutput, m_dOutput, outElems * sizeof(float),
                                   cudaMemcpyDeviceToHost, strm),
                   "cudaMemcpyAsync(hOutput)");
}


bool cumacAiranSubcontext::usesCudaGraph() const
{
    return m_engine ? m_engine->usesCudaGraph() : false;
}


const char* cumacAiranSubcontext::precisionName() const
{
    return m_engine ? m_engine->precisionName() : "n/a";
}


void cumacAiranSubcontext::debugLog()
{
    printf("********************************************\n");
    printf("** cuMAC AI-RAN inference subcontext:\n\n");
    printf("model:           %s\n", m_modelPath.c_str());
    printf("precision:       %s\n", precisionName());
    printf("CUDA graph:      %s\n", usesCudaGraph() ? "on" : "off");
    printf("input  '%s':  [%d, %d]\n", m_inputName.c_str(), m_batchSize, m_obsDim);
    printf("output '%s':  [%d, %d]\n", m_outputName.c_str(), m_batchSize, m_actionDim);
    printf("maxBatchSize:    %d\n", m_maxBatchSize);
    printf("copyOutputEachRun: %d\n", m_copyOutputEachRun);
    printf("********************************************\n");
}

}  // namespace cumac_ml
