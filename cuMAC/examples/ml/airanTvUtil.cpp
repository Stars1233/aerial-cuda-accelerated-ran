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

#include "airanTvUtil.h"

#include <cstddef>
#include <random>
#include <stdexcept>

#include <H5Cpp.h>

#include "infConfig.h"  // cumac_ml::loadInfConfig (header-only; does not link TensorRT)

namespace cumac_ml {

namespace {

// Plain-old-data mirror of the scalar AI-RAN parameters stored as the HDF5
// compound dataset "airanInferenceParam". The in-memory compound type below is
// built with HOFFSET of this struct for both write and read, so the mapping is
// always correct regardless of the file's member ordering.
struct AiranParamPod {
    int32_t obsDim;
    int32_t actionDim;
    int32_t batchSize;
    int32_t maxBatchSize;
    int32_t builderOptLevel;
    uint8_t precision;
    uint8_t useCudaGraph;
    uint8_t copyOutputEachRun;
};

const char* kParamDataset      = "airanInferenceParam";
const char* kObsDataset        = "obs";
const char* kModelPathDataset  = "modelPath";
const char* kInputNameDataset  = "inputName";
const char* kOutputNameDataset = "outputName";
const char* kPtScriptDataset   = "ptToOnnxScript";

// Build the in-memory compound type describing AiranParamPod.
H5::CompType makeAiranCompType() {
    H5::CompType ct(sizeof(AiranParamPod));
    ct.insertMember("obsDim", HOFFSET(AiranParamPod, obsDim), H5::PredType::NATIVE_INT32);
    ct.insertMember("actionDim", HOFFSET(AiranParamPod, actionDim), H5::PredType::NATIVE_INT32);
    ct.insertMember("batchSize", HOFFSET(AiranParamPod, batchSize), H5::PredType::NATIVE_INT32);
    ct.insertMember("maxBatchSize", HOFFSET(AiranParamPod, maxBatchSize), H5::PredType::NATIVE_INT32);
    ct.insertMember("builderOptLevel", HOFFSET(AiranParamPod, builderOptLevel), H5::PredType::NATIVE_INT32);
    ct.insertMember("precision", HOFFSET(AiranParamPod, precision), H5::PredType::NATIVE_UINT8);
    ct.insertMember("useCudaGraph", HOFFSET(AiranParamPod, useCudaGraph), H5::PredType::NATIVE_UINT8);
    ct.insertMember("copyOutputEachRun", HOFFSET(AiranParamPod, copyOutputEachRun), H5::PredType::NATIVE_UINT8);
    return ct;
}

// Write a single variable-length string as a scalar dataset.
void writeStringDataset(H5::H5File& file, const char* name, const std::string& value) {
    H5::StrType strType(H5::PredType::C_S1, H5T_VARIABLE);
    H5::DataSpace scalarSpace(H5S_SCALAR);
    H5::DataSet ds = file.createDataSet(name, strType, scalarSpace);
    ds.write(value, strType);
}

// Read a scalar variable-length string dataset (empty string if absent). Uses
// try/catch rather than nameExists() to stay portable across HDF5 versions.
std::string readStringDataset(H5::H5File& file, const char* name) {
    try {
        H5::Exception::dontPrint();
        H5::DataSet ds      = file.openDataSet(name);
        H5::StrType strType = ds.getStrType();
        std::string out;
        ds.read(out, strType);
        return out;
    } catch (const H5::Exception&) {
        return std::string();
    }
}

}  // namespace


std::vector<float> makeSyntheticObs(uint64_t seed, size_t count) {
    std::vector<float> data(count);
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (size_t i = 0; i < count; ++i) {
        data[i] = dist(rng);
    }
    return data;
}


void createAiranTv(const std::string& filename, const airanTvParams& params) {
    if (params.modelPath.empty()) {
        throw std::runtime_error("createAiranTv: modelPath is required");
    }
    if (params.obsDim <= 0 || params.actionDim <= 0) {
        throw std::runtime_error("createAiranTv: obsDim and actionDim must be positive");
    }
    if (params.batchSize <= 0) {
        throw std::runtime_error("createAiranTv: batchSize must be positive");
    }

    const int32_t maxBatch = (params.maxBatchSize >= params.batchSize) ? params.maxBatchSize
                                                                       : params.batchSize;
    const size_t  obsCount = static_cast<size_t>(maxBatch) * static_cast<size_t>(params.obsDim);

    // Resolve the observation tensor: explicit data if provided (and correctly
    // sized), otherwise deterministic synthetic data from the seed.
    std::vector<float> obs;
    if (!params.obs.empty()) {
        if (params.obs.size() != obsCount) {
            throw std::runtime_error("createAiranTv: provided obs size does not match maxBatchSize*obsDim");
        }
        obs = params.obs;
    } else {
        obs = makeSyntheticObs(params.seed, obsCount);
    }

    H5::H5File file(filename, H5F_ACC_TRUNC);

    // Scalar compound parameters.
    AiranParamPod pod{};
    pod.obsDim            = params.obsDim;
    pod.actionDim         = params.actionDim;
    pod.batchSize         = params.batchSize;
    pod.maxBatchSize      = maxBatch;
    pod.builderOptLevel   = params.builderOptLevel;
    pod.precision         = params.precision;
    pod.useCudaGraph      = params.useCudaGraph;
    pod.copyOutputEachRun = params.copyOutputEachRun;

    H5::CompType compType = makeAiranCompType();
    H5::DataSet paramDs   = file.createDataSet(kParamDataset, compType, H5::DataSpace());
    paramDs.write(&pod, compType);

    // String datasets.
    writeStringDataset(file, kModelPathDataset, params.modelPath);
    writeStringDataset(file, kInputNameDataset, params.inputName);
    writeStringDataset(file, kOutputNameDataset, params.outputName);
    writeStringDataset(file, kPtScriptDataset, params.ptToOnnxScript);

    // Observation tensor [maxBatchSize, obsDim].
    hsize_t dims[2] = {static_cast<hsize_t>(maxBatch), static_cast<hsize_t>(params.obsDim)};
    H5::DataSpace obsSpace(2, dims);
    H5::DataSet   obsDs = file.createDataSet(kObsDataset, H5::PredType::NATIVE_FLOAT, obsSpace);
    obsDs.write(obs.data(), H5::PredType::NATIVE_FLOAT);
}


void loadAiranTv(const std::string& filename, airanTvConfig& cfg, std::vector<float>& hostInput) {
    H5::H5File file(filename, H5F_ACC_RDONLY);

    // Scalar compound parameters.
    AiranParamPod pod{};
    H5::CompType compType = makeAiranCompType();
    H5::DataSet  paramDs  = file.openDataSet(kParamDataset);
    paramDs.read(&pod, compType);

    cfg.obsDim            = pod.obsDim;
    cfg.actionDim         = pod.actionDim;
    cfg.batchSize         = pod.batchSize;
    cfg.maxBatchSize      = (pod.maxBatchSize >= pod.batchSize) ? pod.maxBatchSize : pod.batchSize;
    cfg.builderOptLevel   = pod.builderOptLevel;
    cfg.precision         = pod.precision;
    cfg.useCudaGraph      = pod.useCudaGraph;
    cfg.copyOutputEachRun = pod.copyOutputEachRun;

    // String datasets.
    cfg.modelPath      = readStringDataset(file, kModelPathDataset);
    cfg.inputName      = readStringDataset(file, kInputNameDataset);
    cfg.outputName     = readStringDataset(file, kOutputNameDataset);
    cfg.ptToOnnxScript = readStringDataset(file, kPtScriptDataset);
    if (cfg.inputName.empty())  cfg.inputName  = "obs";
    if (cfg.outputName.empty()) cfg.outputName = "logits";

    // Observation tensor (flattened to maxBatchSize*obsDim).
    const size_t obsCount = static_cast<size_t>(cfg.maxBatchSize) * static_cast<size_t>(cfg.obsDim);
    hostInput.assign(obsCount, 0.0f);
    H5::DataSet obsDs = file.openDataSet(kObsDataset);
    obsDs.read(hostInput.data(), H5::PredType::NATIVE_FLOAT);
}


airanTvParams airanTvParamsFromInfConfig(const std::string& infConfigPath) {
    infConfig cfg = loadInfConfig(infConfigPath);

    airanTvParams p;
    p.modelPath        = cfg.engine.modelPath;
    p.inputName        = cfg.engine.inputName;
    p.outputName       = cfg.engine.outputName;
    p.ptToOnnxScript   = cfg.engine.ptToOnnxScript;
    p.obsDim           = cfg.obsDim;
    p.actionDim        = cfg.actionDim;
    p.batchSize        = cfg.batchSize;
    p.maxBatchSize     = cfg.maxBatchSize;
    p.precision        = (cfg.engine.precision == trtPrecision::kFP16) ? 1 : 0;
    p.useCudaGraph     = cfg.engine.useCudaGraph ? 1 : 0;
    p.builderOptLevel  = cfg.engine.builderOptimizationLevel;
    p.copyOutputEachRun = 0;
    return p;
}

}  // namespace cumac_ml
