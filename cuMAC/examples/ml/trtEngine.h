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

#include <memory>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <NvInfer.h>

namespace cumac_ml {

// Class to extend TensorRT logger that's needed with TensorRT.
class trtLogger : public nvinfer1::ILogger {
public:
    // Lowest-importance severity that gets printed. Defaults to kWARNING to keep
    // normal runs quiet; bump to kINFO/kVERBOSE to surface TensorRT's own layer
    // fusion and engine layer information during an engine build.
    void setReportableSeverity(Severity severity) noexcept { m_reportableSeverity = severity; }

private:
    void log(Severity severity, const char* msg) noexcept override;
    Severity m_reportableSeverity = Severity::kWARNING;
};


// Tensor parameters passed to the TRT engine constructor.
typedef struct trtTensorPrms {
    std::string      name;  // Tensor name that must match a tensor in the given TRT engine file.
    std::vector<int> dims;  // Tensor dimensions. Batch size should be included,
                            // but can be set to whatever number and set dynamically during setup.
                            // If the batch size is not given during setup, it is set to the maximum
                            // batch size given during init.
} trtTensorPrms_t;


// Inference compute precision used when building a TensorRT engine.
enum class trtPrecision {
    kFP32,  // Full precision (default).
    kFP16   // Half precision (enabled only when the platform supports fast FP16).
};


// Configuration for building/running a trtEngine. Controls the model source, the
// build precision, CUDA Graph usage, engine caching, and (for PyTorch .pt inputs)
// the .pt -> ONNX conversion. All of these can affect inference latency, so they
// are exposed here and surfaced through infConfig.yaml for the latency benchmark.
struct trtEngineConfig {
    // Path to the model. The format is auto-detected from the extension:
    //   .pt                 -> PyTorch checkpoint, converted to ONNX (see below), then built.
    //   .onnx               -> ONNX graph, parsed and built into a TensorRT engine.
    //   .engine/.trt/.plan  -> already-serialized TensorRT engine, deserialized directly.
    std::string  modelPath;

    // Build precision (FP32 or FP16). Only affects .pt/.onnx builds.
    trtPrecision precision            = trtPrecision::kFP32;

    // Capture the inference enqueue once and replay it with a CUDA Graph in run().
    // Greatly reduces per-call launch overhead for small/low-latency networks.
    bool         useCudaGraph         = false;

    // TensorRT builder workspace memory pool limit in MiB (0 = do not set a limit).
    size_t       workspaceMiB         = 1024;

    // TensorRT builder optimization level. Higher levels let the builder search
    // longer for faster tactics and more aggressive layer fusions. The default
    // TensorRT level is 3; the maximum is 5. For tiny, launch-bound MLPs (where
    // latency is dominated by the number of kernel launches rather than math)
    // this is the main lever for getting TensorRT to fuse more layers into
    // fewer kernels. A negative value leaves the TensorRT default untouched.
    // Only affects .pt/.onnx builds (not the deserialization of a cached engine).
    int32_t      builderOptimizationLevel = 5;

    // Serialize the built engine to disk and reuse it on subsequent runs (fast path,
    // skips ONNX parsing and the .pt conversion entirely).
    bool         enableEngineCache    = true;

    // Directory for the cached ONNX/engine artifacts. Empty => alongside the model file.
    std::string  engineCacheDir;

    // Force re-conversion (.pt -> ONNX) and engine rebuild even if a cache exists.
    bool         forceRebuild         = false;

    // --- .pt -> ONNX conversion (only used when modelPath is a .pt) ---
    std::string  pythonExecutable     = "python3";    // Python interpreter (must have PyTorch).
    std::string  ptToOnnxScript       = "pt_to_onnx.py";  // Path to the bundled exporter script.
    bool         includeObsNormalizer = false;        // Fold the running obs normalizer into the graph.

    // I/O tensor names. Must match the ONNX graph. For .pt conversion these names are
    // passed through to the exporter so they always match the produced ONNX.
    std::string  inputName            = "obs";
    std::string  outputName           = "logits";

    // Emit progress/build messages.
    bool         verbose              = true;
};


// A generic TRT engine wrapper supporting multiple inputs.
class trtEngine {
public:

    // TRT engine constructor (legacy: explicit ONNX vs serialized-engine selection).
    trtEngine(const char* modelPath,
              const bool parseFromOnnx,
              const uint32_t maxBatchSize,
              const std::vector<trtTensorPrms_t>& inputTensorPrms,
              const std::vector<trtTensorPrms_t>& outputTensorPrms);
    // modelPath - The path to the model file, this can be either a TRT engine file converted from
    //             ONNX (using trtexec), or an ONNX file directly in which case it gets parsed here.
    // parseFromOnnx - Indicate that the model file above is in ONNX format (as opposed to TRT).
    // maxBatchSize - Maximum batch size if batch size is dynamic. The actual batch size if batch size is fixed.
    // inputTensorPrms - Input tensor parameters.
    // outputTensorPrms - Output tensor parameters.

    // TRT engine constructor (config-driven: supports .pt/.onnx/.engine, FP16, CUDA Graph, caching).
    trtEngine(const trtEngineConfig& config,
              const uint32_t maxBatchSize,
              const std::vector<trtTensorPrms_t>& inputTensorPrms,
              const std::vector<trtTensorPrms_t>& outputTensorPrms);
    // config - Engine build/run configuration (model source, precision, CUDA Graph, caching, .pt conversion).
    // maxBatchSize - Maximum batch size for the dynamic batch optimization profile.
    // inputTensorPrms - Input tensor parameters (names must match the model graph).
    // outputTensorPrms - Output tensor parameters (names must match the model graph).

    ~trtEngine();
    trtEngine(trtEngine const&)            = delete;
    trtEngine& operator=(trtEngine const&) = delete;

    // Setup input/output tensor buffer addresses. Set actual batch size.
    bool setup(const std::vector<void*>& inputBuffers,
               const std::vector<void*>& outputBuffers,
               const uint32_t batchSize = 0);
    // inputBuffers - Device memory buffers for the input tensors. The buffers need to be in the same order as inputTensorPrms.
    // outputBuffers - Device memory buffers for the output tensors. The buffers need to be in the same order as outputTensorPrms.
    // batchSize - Batch size for this inference run. Can be omitted, in which case maxBatchSize is used (use for fixed batch size).

    // Run inference. When CUDA Graph is enabled (config), the enqueue is captured on the
    // first call (after setup) and replayed thereafter; otherwise enqueueV3 is used directly.
    bool run(cudaStream_t cuStream);

    // Whether CUDA Graph replay is enabled for this engine.
    bool usesCudaGraph() const { return m_config.useCudaGraph; }

    // Human-readable build precision ("fp16"/"fp32").
    const char* precisionName() const {
        return m_config.precision == trtPrecision::kFP16 ? "fp16" : "fp32";
    }

private:
    // Builder functions, build from TRT engine file or from ONNX file.
    void buildFromTrt(const char* trtModelPath);
    void buildFromOnnx(const char* onnxModelPath,
                       bool fp16 = false,
                       size_t workspaceBytes = 0,
                       const std::string& serializeEnginePath = "");

    // Config-driven build dispatch (.pt/.onnx/.engine + caching).
    void buildFromConfig();

    // Convert a PyTorch .pt checkpoint to ONNX (via the bundled exporter), with caching.
    // Returns the path to the resulting ONNX file.
    std::string ensureOnnxFromPt(const std::string& ptPath);

    // Resolve the cached engine path for the current model/precision/batch size.
    std::string resolveEnginePath(const std::string& sourcePath) const;

    // Tear down a previously captured CUDA Graph (if any).
    void destroyCudaGraph();

    // Helper to convert tensor shape to nvinfer format.
    void toNvInferDims(const std::vector<int>& shape, nvinfer1::Dims& dims);

    // Maximum batch size.
    uint32_t m_maxBatchSize;

    // Model inputs and outputs.
    std::vector<trtTensorPrms_t> m_inputTensorPrms;
    std::vector<trtTensorPrms_t> m_outputTensorPrms;
    int m_numInputs;
    int m_numOutputs;

    // Build/run configuration (defaults are used by the legacy constructor).
    trtEngineConfig m_config;

    // TensorRT components.
    std::unique_ptr<nvinfer1::IRuntime> m_runtime = nullptr;
    std::unique_ptr<nvinfer1::ICudaEngine> m_engine = nullptr;
    std::unique_ptr<nvinfer1::IExecutionContext> m_context = nullptr;

    // CUDA Graph state (captured lazily on the first run() after setup()).
    bool            m_graphCaptured = false;
    cudaGraph_t     m_graph         = nullptr;
    cudaGraphExec_t m_graphExec     = nullptr;

    trtLogger m_logger;
};

}  // namespace cumac_ml
