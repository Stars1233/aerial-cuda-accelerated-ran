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

// Test-vector (TV) creation/loading helpers for the cuMAC AI-RAN inference
// subcontext (cumac_ml::cumacAiranSubcontext).
//
// An AI-RAN TV is a small HDF5 file that fully describes one trtEngine
// inference workload so that the subcontext can be (re)built and driven without
// any out-of-band configuration. It mirrors the way the cuMAC scheduler
// subcontext is driven by an H5 TV (see cumac/src/cumacSubcontext + the
// h5TvCreate/h5TvLoad tools): a compound parameter dataset plus the numeric
// input tensor data, all in one file.
//
// File layout (datasets):
//   "airanInferenceParam" : scalar compound (dims/batch/precision/graph flags)
//   "modelPath"           : scalar variable-length string (path to .onnx/.engine/.pt)
//   "inputName"           : scalar variable-length string (graph input tensor name)
//   "outputName"          : scalar variable-length string (graph output tensor name)
//   "ptToOnnxScript"      : scalar variable-length string (optional .pt->ONNX exporter)
//   "obs"                 : float[maxBatchSize * obsDim] inference input observations
//
// These helpers intentionally avoid pulling in the TensorRT headers so the TV
// can be created by tools (e.g. the test bench) that do not link TensorRT.

#include <cstdint>
#include <string>
#include <vector>

namespace cumac_ml {

// Parameters used to author an AI-RAN inference TV (createAiranTv()).
struct airanTvParams {
    // Model to run. Format auto-detected from the extension by trtEngine:
    //   .pt  -> converted to ONNX (needs ptToOnnxScript) -> built
    //   .onnx -> parsed and built
    //   .engine/.trt/.plan -> deserialized directly
    std::string modelPath;

    // ONNX graph I/O tensor names (must match the model graph).
    std::string inputName  = "obs";
    std::string outputName = "logits";

    // Optional path to the bundled pt_to_onnx.py exporter (only used for .pt models).
    std::string ptToOnnxScript;

    // Inference shapes: input feature dim, output dim, and batch sizes.
    int32_t obsDim       = 487;  // input feature dimension (number of features per sample)
    int32_t actionDim    = 28;   // output dimension (e.g. MCS-offset logits)
    int32_t batchSize    = 16;   // number of samples per forward pass
    int32_t maxBatchSize = 16;   // max batch for the dynamic optimization profile

    // Build/run knobs (all affect inference latency).
    uint8_t precision         = 0;  // 0 = fp32, 1 = fp16
    uint8_t useCudaGraph      = 1;  // capture + replay the enqueue with a CUDA Graph
    int32_t builderOptLevel   = 5;  // TensorRT builder optimization level (<0 = TRT default)
    uint8_t copyOutputEachRun = 0;  // subcontext copies output D2H after each run() when set

    // Synthetic-observation RNG seed. Used only when 'obs' is left empty: the
    // input tensor is then filled with deterministic N(0,1) values so the TV is
    // fully reproducible without external data.
    uint64_t seed = 12345;

    // Explicit input observations (size maxBatchSize * obsDim). When empty the
    // data is generated from 'seed'.
    std::vector<float> obs;
};

// Scalar configuration recovered from an AI-RAN TV (loadAiranTv()).
struct airanTvConfig {
    std::string modelPath;
    std::string inputName;
    std::string outputName;
    std::string ptToOnnxScript;

    int32_t obsDim       = 0;
    int32_t actionDim    = 0;
    int32_t batchSize    = 0;
    int32_t maxBatchSize = 0;

    uint8_t precision         = 0;
    uint8_t useCudaGraph      = 0;
    int32_t builderOptLevel   = 5;
    uint8_t copyOutputEachRun = 0;
};

// Fill a vector with deterministic N(0,1) observations of length 'count'.
std::vector<float> makeSyntheticObs(uint64_t seed, size_t count);

// Author an AI-RAN inference TV at 'filename' (HDF5, truncated if it exists).
// Throws std::runtime_error / H5::Exception on failure.
void createAiranTv(const std::string& filename, const airanTvParams& params);

// Load an AI-RAN TV: fills 'cfg' from the compound + string datasets and resizes
// 'hostInput' to maxBatchSize*obsDim with the stored observations.
// Throws std::runtime_error / H5::Exception on failure.
void loadAiranTv(const std::string& filename, airanTvConfig& cfg, std::vector<float>& hostInput);

// Convenience: build TV-authoring parameters from an existing infConfig.yaml
// (reuses cumac_ml::loadInfConfig). The model path / script paths are resolved
// relative to the config file, exactly like the trtEngine latency benchmark.
airanTvParams airanTvParamsFromInfConfig(const std::string& infConfigPath);

}  // namespace cumac_ml
