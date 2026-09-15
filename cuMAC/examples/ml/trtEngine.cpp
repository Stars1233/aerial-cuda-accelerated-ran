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
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include "trtEngine.h"

namespace cumac_ml {

namespace {

// Throwing CUDA error check (avoids pulling the heavy cumac.h into this TU just
// for its CUDA_CHECK_ERR macro). Build/capture failures abort with a clear message.
inline void cudaCheck(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error (") + what + "): " + cudaGetErrorString(err));
    }
}

// Lower-cased filesystem extension (including the leading dot), e.g. ".onnx".
std::string lowerExtension(const std::string& path) {
    std::string ext = std::filesystem::path(path).extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return ext;
}

// Quote a path/argument for a POSIX shell command (handles spaces safely).
std::string shellQuote(const std::string& s) {
    std::string out = "'";
    for (char c : s) {
        if (c == '\'') {
            out += "'\\''";  // close quote, escaped quote, reopen quote
        } else {
            out += c;
        }
    }
    out += "'";
    return out;
}

// True if file 'a' is at least as recent as file 'b' (i.e. cache 'a' is up to date
// w.r.t. source 'b'). Returns false if either timestamp cannot be read.
bool isAtLeastAsRecent(const std::filesystem::path& a, const std::filesystem::path& b) {
    std::error_code ec1, ec2;
    auto ta = std::filesystem::last_write_time(a, ec1);
    auto tb = std::filesystem::last_write_time(b, ec2);
    if (ec1 || ec2) {
        return false;
    }
    return ta >= tb;
}

}  // namespace


void trtLogger::log(Severity severity, const char *msg) noexcept {
    // Print everything at or above the configured reportable severity
    // (kWARNING by default; raised to kINFO during a verbose build so the
    // fused engine layers are visible).
    if (severity <= m_reportableSeverity) {
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
    // Legacy path: preserve the original behavior (FP32, no workspace limit, no
    // engine caching, no CUDA Graph, default builder optimization level, and the
    // original warning-only build logging). The config only records the model
    // path here.
    m_config.modelPath                 = modelPath ? modelPath : "";
    m_config.useCudaGraph              = false;
    m_config.builderOptimizationLevel  = -1;     // leave the TensorRT default untouched
    m_config.verbose                   = false;  // keep the original kWARNING-only logging

    if(parseFromOnnx)
        buildFromOnnx(modelPath);
    else
        buildFromTrt(modelPath);
}


trtEngine::trtEngine(const trtEngineConfig& config,
                     const uint32_t maxBatchSize,
                     const std::vector<trtTensorPrms_t>& inputTensorPrms,
                     const std::vector<trtTensorPrms_t>& outputTensorPrms):
m_maxBatchSize(maxBatchSize),
m_inputTensorPrms(inputTensorPrms),
m_outputTensorPrms(outputTensorPrms),
m_numInputs(inputTensorPrms.size()),
m_numOutputs(outputTensorPrms.size()),
m_config(config)
{
    buildFromConfig();
}


trtEngine::~trtEngine()
{
    destroyCudaGraph();
}


void trtEngine::buildFromConfig()
{
    namespace fs = std::filesystem;

    if (m_config.modelPath.empty()) {
        throw std::runtime_error("trtEngine: model path is empty");
    }
    if (!fs::exists(m_config.modelPath)) {
        throw std::runtime_error("trtEngine: model file not found: " + m_config.modelPath);
    }

    const std::string ext = lowerExtension(m_config.modelPath);

    // Already-serialized TensorRT engine: deserialize directly.
    if (ext == ".engine" || ext == ".trt" || ext == ".plan") {
        if (m_config.verbose) {
            std::cout << "trtEngine: loading serialized engine " << m_config.modelPath << std::endl;
        }
        buildFromTrt(m_config.modelPath.c_str());
        return;
    }

    // Resolve the ONNX source: convert from .pt if needed, otherwise use as-is.
    std::string onnxPath = m_config.modelPath;
    if (ext == ".pt") {
        onnxPath = ensureOnnxFromPt(m_config.modelPath);
    } else if (ext != ".onnx") {
        throw std::runtime_error("trtEngine: unsupported model format '" + ext +
                                 "' (expected .pt, .onnx, or .engine/.trt/.plan)");
    }

    const bool   fp16          = (m_config.precision == trtPrecision::kFP16);
    const size_t workspaceBytes = m_config.workspaceMiB ? (m_config.workspaceMiB << 20) : 0;

    // Try the cached serialized engine first (fast path: no ONNX parse / .pt conversion).
    std::string enginePath;
    if (m_config.enableEngineCache) {
        enginePath = resolveEnginePath(m_config.modelPath);
        if (!m_config.forceRebuild && fs::exists(enginePath) && isAtLeastAsRecent(enginePath, onnxPath)) {
            if (m_config.verbose) {
                std::cout << "trtEngine: loading cached engine " << enginePath << std::endl;
            }
            buildFromTrt(enginePath.c_str());
            return;
        }
    }

    if (m_config.verbose) {
        std::cout << "trtEngine: building TensorRT engine from " << onnxPath
                  << " (precision=" << precisionName()
                  << ", workspace=" << m_config.workspaceMiB << " MiB)" << std::endl;
    }
    buildFromOnnx(onnxPath.c_str(), fp16, workspaceBytes, enginePath);
}


std::string trtEngine::ensureOnnxFromPt(const std::string& ptPath)
{
    namespace fs = std::filesystem;

    fs::path src(ptPath);
    fs::path dir = m_config.engineCacheDir.empty() ? src.parent_path()
                                                   : fs::path(m_config.engineCacheDir);
    std::string stem = src.stem().string();
    if (m_config.includeObsNormalizer) {
        stem += ".obsnorm";
    }
    fs::path onnxPath = dir / (stem + ".onnx");

    if (!m_config.forceRebuild && fs::exists(onnxPath) && isAtLeastAsRecent(onnxPath, src)) {
        if (m_config.verbose) {
            std::cout << "trtEngine: using cached ONNX " << onnxPath.string() << std::endl;
        }
        return onnxPath.string();
    }

    if (!fs::exists(m_config.ptToOnnxScript)) {
        throw std::runtime_error("trtEngine: .pt -> ONNX exporter script not found: " +
                                 m_config.ptToOnnxScript +
                                 "\n  Set 'conversion.script' in infConfig.yaml to the path of pt_to_onnx.py.");
    }

    std::error_code ec;
    fs::create_directories(dir, ec);

    std::string cmd = shellQuote(m_config.pythonExecutable) + " " +
                      shellQuote(m_config.ptToOnnxScript) +
                      " --checkpoint " + shellQuote(ptPath) +
                      " --output " + shellQuote(onnxPath.string()) +
                      " --input-name " + shellQuote(m_config.inputName) +
                      " --output-name " + shellQuote(m_config.outputName);
    if (m_config.includeObsNormalizer) {
        cmd += " --include-obs-normalizer";
    }

    if (m_config.verbose) {
        std::cout << "trtEngine: converting PyTorch checkpoint to ONNX:\n  " << cmd << std::endl;
    }

    // This example uses a shell to run the optional exporter. shellQuote() protects
    // the arguments from shell interpretation, but the shell still resolves a
    // non-absolute executable through PATH. Therefore pythonExecutable must be an
    // absolute path or come from validated, trusted configuration. Production code
    // that accepts untrusted configuration should use fork()/exec() or a subprocess
    // library instead of std::system().
    int rc = std::system(cmd.c_str());
    if (rc != 0 || !fs::exists(onnxPath)) {
        throw std::runtime_error(
            "trtEngine: .pt -> ONNX conversion failed (exit code " + std::to_string(rc) + ").\n"
            "  Ensure '" + m_config.pythonExecutable + "' is available and has PyTorch installed,\n"
            "  or pre-convert the checkpoint and point 'model.path' at the resulting .onnx/.engine.");
    }

    return onnxPath.string();
}


std::string trtEngine::resolveEnginePath(const std::string& sourcePath) const
{
    namespace fs = std::filesystem;

    fs::path src(sourcePath);
    fs::path dir = m_config.engineCacheDir.empty() ? src.parent_path()
                                                   : fs::path(m_config.engineCacheDir);
    std::string stem = src.stem().string();
    if (m_config.includeObsNormalizer && lowerExtension(sourcePath) == ".pt") {
        stem += ".obsnorm";
    }
    const std::string prec = (m_config.precision == trtPrecision::kFP16) ? "fp16" : "fp32";
    std::string name = stem + "." + prec + ".b" + std::to_string(m_maxBatchSize);
    // Encode the optimization level so engines built at different levels are
    // cached separately (avoids silently loading a stale engine after a change).
    if (m_config.builderOptimizationLevel >= 0) {
        name += ".opt" + std::to_string(m_config.builderOptimizationLevel);
    }
    name += ".engine";
    return (dir / name).string();
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
    if (!m_engine) {
        throw std::runtime_error(std::string("trtEngine: failed to deserialize engine ") + trtModelPath);
    }
    m_context = std::unique_ptr<nvinfer1::IExecutionContext>(m_engine->createExecutionContext());
    if (!m_context) {
        throw std::runtime_error("trtEngine: failed to create execution context");
    }
}


void trtEngine::buildFromOnnx(const char* onnxModelPath,
                              bool fp16,
                              size_t workspaceBytes,
                              const std::string& serializeEnginePath)
{
    // Surface TensorRT's own build messages (including the final fused engine
    // layer list) when verbose, so the number of kernels TensorRT generates is
    // visible in the build log alongside our own progress prints.
    if (m_config.verbose) {
        m_logger.setReportableSeverity(nvinfer1::ILogger::Severity::kINFO);
    }

    // Create our engine builder.
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(m_logger));

    // Define an explicit batch size and then create the network.
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(0));

    // Create a builder config.
    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());

    // Optional workspace memory pool limit.
    if (workspaceBytes > 0) {
        config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, workspaceBytes);
    }

    // Builder optimization level. Higher levels let TensorRT search longer for
    // faster tactics and fuse more layers into fewer kernels - the main latency
    // lever for small, launch-overhead-bound networks. Default TRT level is 3
    // (max 5); a negative config value leaves the TensorRT default untouched.
    if (m_config.builderOptimizationLevel >= 0) {
        config->setBuilderOptimizationLevel(m_config.builderOptimizationLevel);
        if (m_config.verbose) {
            std::cout << "trtEngine: builder optimization level "
                      << m_config.builderOptimizationLevel << std::endl;
        }
    }

    // Enable FP16 if requested and supported by the platform.
    //
    // platformHasFastFp16() and BuilderFlag::kFP16 are marked TRT_DEPRECATED in
    // recent TensorRT (superseded by strongly-typed networks, where precision is
    // carried by the ONNX). We intentionally keep the weakly-typed network here so
    // the engine retains FP32 input/output tensors (matching the FP32 host/device
    // buffers used by this benchmark and by the cuMAC DRL integration) while still
    // letting TensorRT pick FP16 kernels internally. Silence just this deprecation
    // so the project-wide -Werror=deprecated-declarations does not break the build.
    if (fp16) {
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
        if (builder->platformHasFastFp16()) {
            config->setFlag(nvinfer1::BuilderFlag::kFP16);
        } else {
            std::cout << "trtEngine: FP16 requested but not supported on this platform; building FP32."
                      << std::endl;
        }
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
    }

    // Add an optimization profile with a dynamic batch dimension.
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
    if (!parsed) {
        std::string errMsg = std::string("trtEngine: failed to parse ONNX file ") + onnxModelPath;
        for (int i = 0; i < parser->getNbErrors(); i++) {
            errMsg += std::string("\n  ") + parser->getError(i)->desc();
        }
        throw std::runtime_error(errMsg);
    }

    std::unique_ptr<nvinfer1::IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
    if (!plan) {
        throw std::runtime_error(std::string("trtEngine: failed to build engine from ") + onnxModelPath);
    }

    // Cache the serialized engine for fast subsequent loads.
    if (!serializeEnginePath.empty()) {
        std::error_code ec;
        std::filesystem::create_directories(std::filesystem::path(serializeEnginePath).parent_path(), ec);
        std::ofstream out(serializeEnginePath, std::ios::binary);
        if (out) {
            out.write(reinterpret_cast<const char*>(plan->data()), plan->size());
            out.close();
            if (m_config.verbose) {
                std::cout << "trtEngine: cached engine to " << serializeEnginePath << std::endl;
            }
        } else {
            std::cout << "trtEngine: WARNING could not write engine cache to " << serializeEnginePath
                      << " (continuing without caching)" << std::endl;
        }
    }

    m_runtime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(m_logger));
    m_engine = std::unique_ptr<nvinfer1::ICudaEngine>(m_runtime->deserializeCudaEngine(plan->data(), plan->size()));
    if (!m_engine) {
        throw std::runtime_error("trtEngine: failed to deserialize freshly built engine");
    }
    m_context = std::unique_ptr<nvinfer1::IExecutionContext>(m_engine->createExecutionContext());
    if (!m_context) {
        throw std::runtime_error("trtEngine: failed to create execution context");
    }
}


bool trtEngine::setup(const std::vector<void*>& inputDeviceBuf,
                      const std::vector<void*>& outputDeviceBuf,
                      const uint32_t batchSize)
{
    // Tensor addresses / shapes are about to change, so any previously captured
    // CUDA Graph is stale and must be recaptured on the next run().
    destroyCudaGraph();

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
    if (!m_config.useCudaGraph) {
        return m_context->enqueueV3(cuStream);
    }

    // Lazily capture the enqueue into a CUDA Graph on the first run after setup().
    if (!m_graphCaptured) {
        // One un-captured enqueue first so any lazy initialization/allocation that
        // TensorRT performs on the first inference happens outside graph capture.
        if (!m_context->enqueueV3(cuStream)) {
            return false;
        }
        cudaCheck(cudaStreamSynchronize(cuStream), "cudaStreamSynchronize(pre-capture)");

        cudaCheck(cudaStreamBeginCapture(cuStream, cudaStreamCaptureModeThreadLocal),
                  "cudaStreamBeginCapture");
        const bool enqueued = m_context->enqueueV3(cuStream);
        cudaCheck(cudaStreamEndCapture(cuStream, &m_graph), "cudaStreamEndCapture");
        if (!enqueued) {
            destroyCudaGraph();
            return false;
        }
        cudaCheck(cudaGraphInstantiateWithFlags(&m_graphExec, m_graph, 0), "cudaGraphInstantiateWithFlags");
        m_graphCaptured = true;
    }

    cudaCheck(cudaGraphLaunch(m_graphExec, cuStream), "cudaGraphLaunch");
    return true;
}


void trtEngine::destroyCudaGraph()
{
    if (m_graphExec) {
        cudaGraphExecDestroy(m_graphExec);
        m_graphExec = nullptr;
    }
    if (m_graph) {
        cudaGraphDestroy(m_graph);
        m_graph = nullptr;
    }
    m_graphCaptured = false;
}

} // namespace cumac_ml
