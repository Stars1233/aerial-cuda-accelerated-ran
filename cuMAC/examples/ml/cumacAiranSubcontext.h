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

#pragma once

#include <memory>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "airanTvUtil.h"
#include "trtEngine.h"

namespace cumac_ml {

// cumacAiranSubcontext - an AI-RAN model-inference subcontext.
//
// This is the inference-side analogue of cumac::cumacSubcontext: it owns one
// trtEngine model-inference workload (its TensorRT engine plus the host/device
// I/O buffers) and exposes the same constructor / setup() / run() lifecycle so
// it can be driven by the cubb_gpu_test_bench cuMAC test worker on a dedicated
// CUDA stream / (green) context. Running it alongside the cuPHY workers is what
// exercises GPU sharing between cuPHY and cuMAC trtEngine inference.
//
// Like cumac::cumacSubcontext, the object is configured entirely from an HDF5
// test vector (see airanTvUtil.h) so a worker only needs a file name. A
// config-struct constructor is also provided for callers that already hold the
// parsed config (e.g. to avoid temp files).
//
// All methods are stream-ordered and require external synchronization, matching
// cumac::cumacSubcontext (the test worker synchronizes the stream).
class cumacAiranSubcontext {
public:
    // Construct from an HDF5 AI-RAN TV file.
    // tvFilename       : AI-RAN TV (see cumac_ml::createAiranTv).
    // in_GPU           : 1 - run on GPU (only supported mode; 0 is rejected).
    // in_halfPrecision : 0 - build FP32 engine, 1 - request FP16 engine
    //                    (overrides the precision stored in the TV).
    // strm             : CUDA stream used for the initial input upload.
    // Requires external synchronization after construction.
    cumacAiranSubcontext(const std::string& tvFilename,
                         uint8_t            in_GPU,
                         uint8_t            in_halfPrecision,
                         cudaStream_t       strm);

    // Construct directly from a parsed config + host input buffer (no file I/O).
    // hostInput may be empty, in which case a zero-filled input is used until
    // setup() supplies data.
    cumacAiranSubcontext(const airanTvConfig&      cfg,
                         const std::vector<float>& hostInput,
                         uint8_t                   in_GPU,
                         uint8_t                   in_halfPrecision,
                         cudaStream_t              strm);

    ~cumacAiranSubcontext();
    cumacAiranSubcontext(const cumacAiranSubcontext&)            = delete;
    cumacAiranSubcontext& operator=(const cumacAiranSubcontext&) = delete;

    // (Re)load the inference input from a TV file, upload it to the device, and
    // bind the engine's I/O tensors for the active batch size. Mirrors
    // cumac::cumacSubcontext::setup() which reloads its TV each call.
    // Requires external synchronization.
    void setup(const std::string& tvFilename, cudaStream_t strm);

    // Bind the engine I/O tensors using the already-resident input (no reload).
    // Requires external synchronization.
    void setup(cudaStream_t strm);

    // Run one inference. When the TV requested copyOutputEachRun, the output is
    // also copied device->host (stream-ordered). Requires external synchronization.
    void run(cudaStream_t strm);

    // Copy the latest inference output device->host on the given stream. The
    // caller must synchronize the stream before reading getOutputHost().
    void copyOutputToHost(cudaStream_t strm);

    // Print a short summary of the configured inference workload.
    void debugLog();

    // --- read-only accessors (for testing / evaluation) ---
    const float* getOutputHost() const { return m_hOutput; }
    const float* getInputHost() const { return m_hInput; }
    int  batchSize() const { return m_batchSize; }
    int  obsDim() const { return m_obsDim; }
    int  actionDim() const { return m_actionDim; }
    int  outputElems() const { return m_batchSize * m_actionDim; }
    bool usesCudaGraph() const;
    const char* precisionName() const;

private:
    // Shared construction helper (engine build + buffer allocation + first upload).
    void init(const airanTvConfig& cfg, const std::vector<float>& hostInput, cudaStream_t strm);

    // Resolved configuration.
    std::string m_modelPath;
    std::string m_inputName;
    std::string m_outputName;
    int     m_obsDim            = 0;
    int     m_actionDim         = 0;
    int     m_batchSize         = 0;
    int     m_maxBatchSize      = 0;
    uint8_t m_GPU               = 1;
    uint8_t m_halfPrecision     = 0;
    uint8_t m_copyOutputEachRun = 0;

    // TensorRT engine wrapper.
    std::unique_ptr<trtEngine> m_engine;

    // Host (pinned) staging buffers and device I/O buffers (sized for maxBatch).
    float* m_hInput  = nullptr;
    float* m_hOutput = nullptr;
    float* m_dInput  = nullptr;
    float* m_dOutput = nullptr;
    std::vector<void*> m_inputBuffers;
    std::vector<void*> m_outputBuffers;

    // True once the engine I/O tensors have been bound via setup().
    bool m_isSetup = false;
};

}  // namespace cumac_ml
