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

// Unit test for the cuMAC AI-RAN inference subcontext (cumacAiranSubcontext).
//
// This stand-alone test drives cumacAiranSubcontext through the *exact* lifecycle
// the cubb_gpu_test_bench cuMAC test worker uses for its scheduler pipes:
//
//   macInit  -> construct one subcontext ("pipe") per MAC slot from an H5 TV
//   macSetup -> setup() each pipe (bind engine I/O for the active batch)
//   macRun   -> for each slot: record start event, run() on the stream, record end
//   eval     -> per-slot GPU timing from a common start event
//
// It builds a small AI-RAN TV from an infConfig.yaml (same model the trtEngine
// latency benchmark uses), runs the pipes on a single CUDA stream across many
// timing iterations, validates the inference output, and reports per-inference
// latency. Passing here means the subcontext is ready to be plugged into
// cubb_gpu_test_bench for cuPHY + cuMAC trtEngine GPU-sharing tests.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "airanTvUtil.h"
#include "cumacAiranSubcontext.h"
#include "infConfig.h"

namespace {

void usage() {
    printf("cuMAC AI-RAN inference subcontext unit test [options]\n");
    printf("  Options:\n");
    printf("  -c  Path to infConfig.yaml (default: ./infConfig.yaml, then next to the binary)\n");
    printf("  -t  Path to write the generated AI-RAN TV (default: ./airan_unit_tv.h5)\n");
    printf("  -g  GPU device index (overrides runtime.gpuId in the config)\n");
    printf("  -s  Number of MAC slots / inference pipes to create (default: 4)\n");
    printf("  -n  Number of timing iterations (default: from config latency.timingIters)\n");
    printf("  -h  Show this help\n");
    printf("Example: './cumacAiranSubcontext -c infConfig.yaml -g 0 -s 4'\n");
}

// Locate infConfig.yaml: explicit CLI path, then CWD, then next to the binary,
// then the compiled-in source default (mirrors trtEngine/main.cpp).
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
    const double idx  = (p / 100.0) * static_cast<double>(sorted.size() - 1);
    const size_t lo   = static_cast<size_t>(std::floor(idx));
    const size_t hi   = static_cast<size_t>(std::ceil(idx));
    const double frac = idx - static_cast<double>(lo);
    return sorted[lo] * (1.0 - frac) + sorted[hi] * frac;
}

}  // namespace


int main(int argc, char* argv[]) {
try {
    std::string cliConfigPath;
    std::string tvPath = "airan_unit_tv.h5";
    int gpuOverride    = -1;
    int nSlots         = 4;
    int timingOverride = -1;

    int iArg = 1;
    while (iArg < argc) {
        if ('-' == argv[iArg][0]) {
            switch (argv[iArg][1]) {
                case 'c':
                    if (++iArg >= argc) { fprintf(stderr, "ERROR: No config path given.\n"); return 1; }
                    cliConfigPath.assign(argv[iArg++]);
                    break;
                case 't':
                    if (++iArg >= argc) { fprintf(stderr, "ERROR: No TV path given.\n"); return 1; }
                    tvPath.assign(argv[iArg++]);
                    break;
                case 'g':
                    if (++iArg >= argc || 1 != sscanf(argv[iArg], "%i", &gpuOverride)) {
                        fprintf(stderr, "ERROR: Invalid GPU index.\n"); return 1;
                    }
                    iArg++;
                    break;
                case 's':
                    if (++iArg >= argc || 1 != sscanf(argv[iArg], "%i", &nSlots) || nSlots < 1) {
                        fprintf(stderr, "ERROR: Invalid number of slots.\n"); return 1;
                    }
                    iArg++;
                    break;
                case 'n':
                    if (++iArg >= argc || 1 != sscanf(argv[iArg], "%i", &timingOverride)) {
                        fprintf(stderr, "ERROR: Invalid number of iterations.\n"); return 1;
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
    std::cout << "cuMAC AI-RAN inference subcontext unit test" << std::endl;
    std::cout << "=========================================================" << std::endl;
    std::cout << "Config file: " << configPath << std::endl;

    cumac_ml::infConfig cfg = cumac_ml::loadInfConfig(configPath);
    if (gpuOverride >= 0) {
        cfg.gpuId = gpuOverride;
    }
    const int warmupIters = cfg.warmupIters;
    const int timingIters = (timingOverride >= 0) ? timingOverride : cfg.timingIters;

    int nGPUs = 0;
    if (cudaGetDeviceCount(&nGPUs) != cudaSuccess || nGPUs <= 0) {
        fprintf(stderr, "ERROR: No CUDA devices available.\n");
        return 1;
    }
    if (cfg.gpuId < 0 || cfg.gpuId >= nGPUs) {
        fprintf(stderr, "ERROR: Invalid GPU index %d (have %d GPUs).\n", cfg.gpuId, nGPUs);
        return 1;
    }
    if (cudaSetDevice(cfg.gpuId) != cudaSuccess) {
        fprintf(stderr, "ERROR: cudaSetDevice(%d) failed.\n", cfg.gpuId);
        return 1;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, cfg.gpuId);
    std::cout << "GPU device:  " << cfg.gpuId << " (" << prop.name << ", " << prop.multiProcessorCount
              << " SMs)" << std::endl;

    // -------------------------------------------------------------------------
    // Author an AI-RAN TV from the resolved inference config, exactly as a test
    // bench would when it needs a self-contained inference workload description.
    cumac_ml::airanTvParams tvParams = cumac_ml::airanTvParamsFromInfConfig(configPath);
    cumac_ml::createAiranTv(tvPath, tvParams);
    std::cout << "AI-RAN TV:   " << tvPath << std::endl;
    std::cout << "Model:       " << tvParams.modelPath << std::endl;
    std::cout << "Shapes:      input '" << tvParams.inputName << "' [" << tvParams.batchSize << ", "
              << tvParams.obsDim << "]  ->  output '" << tvParams.outputName << "' ["
              << tvParams.batchSize << ", " << tvParams.actionDim << "]" << std::endl;
    std::cout << "Slots/pipes: " << nSlots << "   warmup=" << warmupIters
              << "   timed=" << timingIters << std::endl;
    std::cout << "---------------------------------------------------------" << std::endl;

    cudaStream_t stream;
    if (cudaStreamCreate(&stream) != cudaSuccess) {
        fprintf(stderr, "ERROR: cudaStreamCreate failed.\n");
        return 1;
    }

    // -------------------------------------------------------------------------
    // macInit analogue: create one subcontext ("pipe") per MAC slot from the TV.
    auto buildStart = std::chrono::steady_clock::now();
    std::vector<std::unique_ptr<cumac_ml::cumacAiranSubcontext>> pipes;
    pipes.reserve(nSlots);
    for (int s = 0; s < nSlots; ++s) {
        pipes.emplace_back(std::make_unique<cumac_ml::cumacAiranSubcontext>(
            tvPath, /*GPU=*/1, static_cast<uint8_t>(tvParams.precision), stream));
    }
    if (cudaStreamSynchronize(stream) != cudaSuccess) {
        throw std::runtime_error("cudaStreamSynchronize after pipe construction failed");
    }
    double buildMs =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - buildStart).count();
    pipes.front()->debugLog();
    std::cout << nSlots << " inference pipe(s) ready in " << std::fixed << std::setprecision(1)
              << buildMs << " ms" << std::endl;

    // -------------------------------------------------------------------------
    // macSetup analogue: bind each pipe's engine I/O tensors.
    for (auto& pipe : pipes) {
        pipe->setup(tvPath, stream);
    }
    if (cudaStreamSynchronize(stream) != cudaSuccess) {
        throw std::runtime_error("cudaStreamSynchronize after setup failed");
    }

    // Warmup (also triggers CUDA Graph capture on the first run when enabled).
    for (int i = 0; i < warmupIters; ++i) {
        for (auto& pipe : pipes) {
            pipe->run(stream);
        }
    }
    if (cudaStreamSynchronize(stream) != cudaSuccess) {
        throw std::runtime_error("cudaStreamSynchronize after warmup failed");
    }

    cudaEvent_t evStart, evEnd;
    cudaEventCreate(&evStart);
    cudaEventCreate(&evEnd);

    // (1) Bulk measurement: average GPU time per inference over the whole loop
    // (one start/run/end per slot per iteration, no per-call sync).
    cudaEventRecord(evStart, stream);
    for (int i = 0; i < timingIters; ++i) {
        for (auto& pipe : pipes) {
            pipe->run(stream);
        }
    }
    cudaEventRecord(evEnd, stream);
    cudaEventSynchronize(evEnd);
    float bulkMs = 0.0f;
    cudaEventElapsedTime(&bulkMs, evStart, evEnd);
    const long totalInferences = static_cast<long>(timingIters) * static_cast<long>(nSlots);
    const double bulkAvgUs = static_cast<double>(bulkMs) * 1000.0 / std::max(1L, totalInferences);

    // (2) Per-call measurement: GPU time of each individual inference (one event
    // pair + sync per call) to expose the latency distribution -- the same
    // per-slot timing the test bench records around each cuMAC pipe run.
    std::vector<double> perCallUs;
    perCallUs.reserve(static_cast<size_t>(timingIters) * pipes.size());
    for (int i = 0; i < timingIters; ++i) {
        for (auto& pipe : pipes) {
            cudaEventRecord(evStart, stream);
            pipe->run(stream);
            cudaEventRecord(evEnd, stream);
            cudaEventSynchronize(evEnd);
            float ms = 0.0f;
            cudaEventElapsedTime(&ms, evStart, evEnd);
            perCallUs.push_back(static_cast<double>(ms) * 1000.0);
        }
    }
    std::sort(perCallUs.begin(), perCallUs.end());
    const double meanUs = std::accumulate(perCallUs.begin(), perCallUs.end(), 0.0) /
                          static_cast<double>(std::max<size_t>(1, perCallUs.size()));

    // -------------------------------------------------------------------------
    // Validate the inference output: copy back from the first pipe and check the
    // logits are finite, then report the argmax of the first sample.
    pipes.front()->copyOutputToHost(stream);
    if (cudaStreamSynchronize(stream) != cudaSuccess) {
        throw std::runtime_error("cudaStreamSynchronize before output check failed");
    }
    const float* out = pipes.front()->getOutputHost();
    const int    outElems = pipes.front()->outputElems();
    bool outputOk = (outElems > 0);
    for (int j = 0; j < outElems; ++j) {
        if (!std::isfinite(out[j])) {
            outputOk = false;
            fprintf(stderr, "ERROR: non-finite logit at index %d\n", j);
            break;
        }
    }
    int   argMax   = 0;
    float maxLogit = (outElems > 0) ? out[0] : 0.0f;
    for (int j = 1; j < pipes.front()->actionDim(); ++j) {
        if (out[j] > maxLogit) {
            maxLogit = out[j];
            argMax   = j;
        }
    }

    std::cout << "---------------------------------------------------------" << std::endl;
    std::cout << "Inference latency (" << pipes.front()->precisionName() << ", CUDA graph "
              << (pipes.front()->usesCudaGraph() ? "on" : "off") << ", batch "
              << pipes.front()->batchSize() << ", " << nSlots << " pipe(s))" << std::endl;
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
    std::cout << "  Sanity: sample 0 argmax = " << argMax << " (logit " << std::setprecision(4)
              << maxLogit << ")" << std::endl;
    std::cout << "=========================================================" << std::endl;

    cudaEventDestroy(evStart);
    cudaEventDestroy(evEnd);
    pipes.clear();
    cudaStreamDestroy(stream);

    if (!outputOk) {
        std::cout << "\033[1;31mFAILED!\033[0m" << std::endl;
        return 1;
    }

    std::cout << "\033[1;32mPASSED!\033[0m" << std::endl;
    return 0;
} catch (const std::exception& e) {
    std::cerr << "ERROR: " << e.what() << std::endl;
    return 1;
}
}
