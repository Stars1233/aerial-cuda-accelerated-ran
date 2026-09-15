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

// cuMAC TensorRT inference engine real-time latency benchmark.
//
// Loads a model (PyTorch .pt checkpoint, ONNX, or serialized TensorRT engine)
// described by infConfig.yaml, builds a TensorRT engine (FP32/FP16, optional
// CUDA Graph, engine caching), and measures single-shot inference latency under
// the configured options. The .pt path is converted to ONNX once via the
// bundled pt_to_onnx.py exporter and then cached as a TensorRT engine.

#include "cumac.h"
#include "trtEngine.h"
#include "infConfig.h"

#include <chrono>
#include <cmath>
#include <iomanip>
#include <numeric>
#include <random>
#include <vector>

namespace {

void usage() {
    printf("cuMAC TRT engine inference latency benchmark [options]\n");
    printf("  Options:\n");
    printf("  -c  Path to infConfig.yaml (default: ./infConfig.yaml, then alongside the binary)\n");
    printf("  -g  GPU device index (overrides runtime.gpuId in the config)\n");
    printf("  -h  Show this help\n");
    printf("Example: './trtEngine -c infConfig.yaml -g 0'\n");
}

// Locate the configuration file: explicit CLI path, then current directory,
// then next to the executable, then the compiled-in source default.
std::string findConfigPath(const std::string& cliPath) {
    namespace fs = std::filesystem;
    if (!cliPath.empty()) {
        return cliPath;
    }
    if (fs::exists("infConfig.yaml")) {
        return "infConfig.yaml";
    }
    std::error_code ec;
    fs::path exe = fs::read_symlink("/proc/self/exe", ec);
    if (!ec) {
        fs::path candidate = exe.parent_path() / "infConfig.yaml";
        if (fs::exists(candidate)) {
            return candidate.string();
        }
    }
#ifdef CUMAC_ML_DIR
    {
        fs::path candidate = fs::path(CUMAC_ML_DIR) / "infConfig.yaml";
        if (fs::exists(candidate)) {
            return candidate.lexically_normal().string();
        }
    }
#endif
    return "";
}

// Linear-interpolated percentile of an already-sorted vector.
double percentile(const std::vector<double>& sorted, double p) {
    if (sorted.empty()) {
        return 0.0;
    }
    const double idx = (p / 100.0) * static_cast<double>(sorted.size() - 1);
    const size_t lo = static_cast<size_t>(std::floor(idx));
    const size_t hi = static_cast<size_t>(std::ceil(idx));
    const double frac = idx - static_cast<double>(lo);
    return sorted[lo] * (1.0 - frac) + sorted[hi] * frac;
}

}  // namespace


int main(int argc, char* argv[]) {
try {
    std::string cliConfigPath;
    int gpuOverride = -1;

    int iArg = 1;
    while (iArg < argc) {
        if ('-' == argv[iArg][0]) {
            switch (argv[iArg][1]) {
                case 'c':
                    if (++iArg >= argc) {
                        fprintf(stderr, "ERROR: No config file path given.\n");
                        return 1;
                    }
                    cliConfigPath.assign(argv[iArg++]);
                    break;
                case 'g':
                    if (++iArg >= argc || 1 != sscanf(argv[iArg], "%i", &gpuOverride)) {
                        fprintf(stderr, "ERROR: Invalid GPU index.\n");
                        return 1;
                    }
                    iArg++;
                    break;
                case 'h':
                    usage();
                    return 0;
                default:
                    fprintf(stderr, "ERROR: Unknown option: %s\n", argv[iArg]);
                    usage();
                    return 1;
            }
        } else {
            fprintf(stderr, "ERROR: Invalid command line argument: %s\n", argv[iArg]);
            return 1;
        }
    }

    const std::string configPath = findConfigPath(cliConfigPath);
    if (configPath.empty()) {
        fprintf(stderr, "ERROR: Could not find infConfig.yaml. Pass one with -c <path>.\n");
        usage();
        return 1;
    }

    std::cout << "=========================================================" << std::endl;
    std::cout << "cuMAC TRT engine inference latency benchmark" << std::endl;
    std::cout << "=========================================================" << std::endl;
    std::cout << "Config file: " << configPath << std::endl;

    cumac_ml::infConfig cfg = cumac_ml::loadInfConfig(configPath);
    if (gpuOverride >= 0) {
        cfg.gpuId = gpuOverride;
    }

    int nGPUs = 0;
    CUDA_CHECK_ERR(cudaGetDeviceCount(&nGPUs));
    if (cfg.gpuId < 0 || cfg.gpuId >= nGPUs) {
        fprintf(stderr, "ERROR: Invalid GPU index %d (have %d GPUs).\n", cfg.gpuId, nGPUs);
        return 1;
    }
    CUDA_CHECK_ERR(cudaSetDevice(cfg.gpuId));

    cudaDeviceProp prop;
    CUDA_CHECK_ERR(cudaGetDeviceProperties(&prop, cfg.gpuId));

    std::cout << "GPU device:  " << cfg.gpuId << " (" << prop.name << ", " << prop.multiProcessorCount
              << " SMs)" << std::endl;
    std::cout << "Model:       " << cfg.engine.modelPath << std::endl;
    std::cout << "Precision:   " << (cfg.engine.precision == cumac_ml::trtPrecision::kFP16 ? "fp16" : "fp32")
              << "   CUDA graph: " << (cfg.engine.useCudaGraph ? "on" : "off") << std::endl;
    std::cout << "Shapes:      input '" << cfg.engine.inputName << "' [" << cfg.batchSize << ", " << cfg.obsDim
              << "]  ->  output '" << cfg.engine.outputName << "' [" << cfg.batchSize << ", " << cfg.actionDim
              << "]" << std::endl;
    std::cout << "Iterations:  warmup=" << cfg.warmupIters << "  timed=" << cfg.timingIters
              << "  copyInputEachIter=" << (cfg.copyInputEachIter ? "true" : "false") << std::endl;
    std::cout << "---------------------------------------------------------" << std::endl;

    cudaStream_t stream;
    CUDA_CHECK_ERR(cudaStreamCreate(&stream));

    // Build the engine (this may convert a .pt to ONNX and/or build the TRT engine).
    const std::vector<cumac_ml::trtTensorPrms_t> inputTensorPrms = {
        {cfg.engine.inputName, {cfg.maxBatchSize, cfg.obsDim}}};
    const std::vector<cumac_ml::trtTensorPrms_t> outputTensorPrms = {
        {cfg.engine.outputName, {cfg.maxBatchSize, cfg.actionDim}}};

    auto buildStart = std::chrono::steady_clock::now();
    std::unique_ptr<cumac_ml::trtEngine> engine = std::make_unique<cumac_ml::trtEngine>(
        cfg.engine, static_cast<uint32_t>(cfg.maxBatchSize), inputTensorPrms, outputTensorPrms);
    double buildMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - buildStart).count();
    std::cout << "Engine ready in " << std::fixed << std::setprecision(1) << buildMs << " ms" << std::endl;

    // Allocate host (pinned) and device I/O buffers, sized for the max batch.
    const size_t inElems = static_cast<size_t>(cfg.maxBatchSize) * cfg.obsDim;
    const size_t outElems = static_cast<size_t>(cfg.maxBatchSize) * cfg.actionDim;
    const size_t inBytesBatch = static_cast<size_t>(cfg.batchSize) * cfg.obsDim * sizeof(float);

    float* hInput = nullptr;
    float* hOutput = nullptr;
    float* dInput = nullptr;
    float* dOutput = nullptr;
    CUDA_CHECK_ERR(cudaMallocHost((void**)&hInput, inElems * sizeof(float)));
    CUDA_CHECK_ERR(cudaMallocHost((void**)&hOutput, outElems * sizeof(float)));
    CUDA_CHECK_ERR(cudaMalloc((void**)&dInput, inElems * sizeof(float)));
    CUDA_CHECK_ERR(cudaMalloc((void**)&dOutput, outElems * sizeof(float)));

    // Deterministic pseudo-random observations.
    std::mt19937 rng(12345);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (size_t i = 0; i < inElems; i++) {
        hInput[i] = dist(rng);
    }
    CUDA_CHECK_ERR(cudaMemcpyAsync(dInput, hInput, inElems * sizeof(float), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK_ERR(cudaStreamSynchronize(stream));

    std::vector<void*> inputBuffers = {(void*)dInput};
    std::vector<void*> outputBuffers = {(void*)dOutput};
    if (!engine->setup(inputBuffers, outputBuffers, static_cast<uint32_t>(cfg.batchSize))) {
        throw std::runtime_error("trtEngine setup() failed");
    }

    // Warmup (also triggers CUDA Graph capture on the first run when enabled).
    for (int i = 0; i < cfg.warmupIters; i++) {
        if (cfg.copyInputEachIter) {
            CUDA_CHECK_ERR(cudaMemcpyAsync(dInput, hInput, inBytesBatch, cudaMemcpyHostToDevice, stream));
        }
        if (!engine->run(stream)) {
            throw std::runtime_error("trtEngine run() failed during warmup");
        }
    }
    CUDA_CHECK_ERR(cudaStreamSynchronize(stream));

    cudaEvent_t evStart, evEnd;
    CUDA_CHECK_ERR(cudaEventCreate(&evStart));
    CUDA_CHECK_ERR(cudaEventCreate(&evEnd));

    // (1) Bulk measurement: average GPU time per inference over the whole loop
    // (no per-iteration synchronization), matching steady-state throughput.
    CUDA_CHECK_ERR(cudaEventRecord(evStart, stream));
    for (int i = 0; i < cfg.timingIters; i++) {
        if (cfg.copyInputEachIter) {
            CUDA_CHECK_ERR(cudaMemcpyAsync(dInput, hInput, inBytesBatch, cudaMemcpyHostToDevice, stream));
        }
        engine->run(stream);
    }
    CUDA_CHECK_ERR(cudaEventRecord(evEnd, stream));
    CUDA_CHECK_ERR(cudaEventSynchronize(evEnd));
    float bulkMs = 0.0f;
    CUDA_CHECK_ERR(cudaEventElapsedTime(&bulkMs, evStart, evEnd));
    const double bulkAvgUs = static_cast<double>(bulkMs) * 1000.0 / std::max(1, cfg.timingIters);

    // (2) Per-call measurement: GPU time of each individual inference (one event
    // pair + sync per call) to expose the latency distribution.
    std::vector<double> perCallUs;
    perCallUs.reserve(cfg.timingIters);
    for (int i = 0; i < cfg.timingIters; i++) {
        CUDA_CHECK_ERR(cudaEventRecord(evStart, stream));
        if (cfg.copyInputEachIter) {
            CUDA_CHECK_ERR(cudaMemcpyAsync(dInput, hInput, inBytesBatch, cudaMemcpyHostToDevice, stream));
        }
        engine->run(stream);
        CUDA_CHECK_ERR(cudaEventRecord(evEnd, stream));
        CUDA_CHECK_ERR(cudaEventSynchronize(evEnd));
        float ms = 0.0f;
        CUDA_CHECK_ERR(cudaEventElapsedTime(&ms, evStart, evEnd));
        perCallUs.push_back(static_cast<double>(ms) * 1000.0);
    }
    std::sort(perCallUs.begin(), perCallUs.end());
    const double meanUs = std::accumulate(perCallUs.begin(), perCallUs.end(), 0.0) /
                          static_cast<double>(std::max<size_t>(1, perCallUs.size()));

    // Sanity check: copy outputs back and report the argmax of the first row.
    CUDA_CHECK_ERR(cudaMemcpyAsync(hOutput, dOutput, outElems * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK_ERR(cudaStreamSynchronize(stream));
    int argMax = 0;
    float maxLogit = hOutput[0];
    for (int j = 1; j < cfg.actionDim; j++) {
        if (hOutput[j] > maxLogit) {
            maxLogit = hOutput[j];
            argMax = j;
        }
    }

    std::cout << "---------------------------------------------------------" << std::endl;
    std::cout << "Inference latency (" << (cfg.engine.precision == cumac_ml::trtPrecision::kFP16 ? "fp16" : "fp32")
              << ", CUDA graph " << (engine->usesCudaGraph() ? "on" : "off") << ", batch " << cfg.batchSize << ")"
              << std::endl;
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "  Compute-only avg (bulk):   " << std::setw(8) << bulkAvgUs << " us/inference" << std::endl;
    std::cout << "  Per-call mean:             " << std::setw(8) << meanUs << " us" << std::endl;
    std::cout << "  Per-call min:              " << std::setw(8) << perCallUs.front() << " us" << std::endl;
    std::cout << "  Per-call p50:              " << std::setw(8) << percentile(perCallUs, 50.0) << " us" << std::endl;
    std::cout << "  Per-call p90:              " << std::setw(8) << percentile(perCallUs, 90.0) << " us" << std::endl;
    std::cout << "  Per-call p99:              " << std::setw(8) << percentile(perCallUs, 99.0) << " us" << std::endl;
    std::cout << "  Per-call max:              " << std::setw(8) << perCallUs.back() << " us" << std::endl;
    std::cout << std::setprecision(1);
    std::cout << "  Throughput (bulk):         " << std::setw(8) << (1.0e6 / std::max(1e-9, bulkAvgUs))
              << " inferences/s" << std::endl;
    std::cout << "  Sanity: row 0 argmax MCS offset = " << argMax << " (logit " << std::setprecision(4) << maxLogit
              << ")" << std::endl;
    std::cout << "=========================================================" << std::endl;

    CUDA_CHECK_ERR(cudaEventDestroy(evStart));
    CUDA_CHECK_ERR(cudaEventDestroy(evEnd));
    CUDA_CHECK_ERR(cudaFreeHost(hInput));
    CUDA_CHECK_ERR(cudaFreeHost(hOutput));
    CUDA_CHECK_ERR(cudaFree(dInput));
    CUDA_CHECK_ERR(cudaFree(dOutput));
    CUDA_CHECK_ERR(cudaStreamDestroy(stream));

    std::cout << "\033[1;32mPASSED!\033[0m" << std::endl;
    return 0;
} catch (const std::exception& e) {
    std::cerr << "ERROR: " << e.what() << std::endl;
    return 1;
}
}
