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
#include <NvInfer.h>

namespace cumac_ml {

// Class to extend TensorRT logger that's needed with TensorRT.
class trtLogger : public nvinfer1::ILogger {
    void log (Severity severity, const char* msg) noexcept override;
};


// Tensor parameters passed to the TRT engine constructor.
typedef struct trtTensorPrms {
    std::string      name;  // Tensor name that must match a tensor in the given TRT engine file.
    std::vector<int> dims;  // Tensor dimensions. Batch size should be included,
                            // but can be set to whatever number and set dynamically during setup.
                            // If the batch size is not given during setup, it is set to the maximum
                            // batch size given during init.
} trtTensorPrms_t;


// A generic TRT engine wrapper supporting multiple inputs.
class trtEngine {
public:

    // TRT engine constructor.
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

    ~trtEngine()                           = default;
    trtEngine(trtEngine const&)            = delete;
    trtEngine& operator=(trtEngine const&) = delete;

    // Setup input/output tensor buffer addresses. Set actual batch size.
    bool setup(const std::vector<void*>& inputBuffers,
               const std::vector<void*>& outputBuffers,
               const uint32_t batchSize = 0);
    // inputBuffers - Device memory buffers for the input tensors. The buffers need to be in the same order as inputTensorPrms.
    // outputBuffers - Device memory buffers for the output tensors. The buffers need to be in the same order as outputTensorPrms.
    // batchSize - Batch size for this inference run. Can be omitted, in which case maxBatchSize is used (use for fixed batch size).

    // Run inference.
    bool run(cudaStream_t cuStream);

private:
    // Builder functions, build from TRT engine file or from ONNX file.
    void buildFromTrt(const char* trtModelPath);
    void buildFromOnnx(const char* onnxModelPath);

    // Helper to convert tensor shape to nvinfer format.
    void toNvInferDims(const std::vector<int>& shape, nvinfer1::Dims& dims);

    // Maximum batch size.
    uint32_t m_maxBatchSize;

    // Model inputs and outputs.
    std::vector<trtTensorPrms_t> m_inputTensorPrms;
    std::vector<trtTensorPrms_t> m_outputTensorPrms;
    int m_numInputs;
    int m_numOutputs;

    // TensorRT components.
    std::unique_ptr<nvinfer1::IRuntime> m_runtime = nullptr;
    std::unique_ptr<nvinfer1::ICudaEngine> m_engine = nullptr;
    std::unique_ptr<nvinfer1::IExecutionContext> m_context = nullptr;

    trtLogger m_logger;
};

}  // namespace cumac_ml
