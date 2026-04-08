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

#include <iostream>
#include <vector>
#include <string>
#include <cstdio>
#include <chrono>
#include <cstdint>
#include <unistd.h>  // For getopt
#include <yaml-cpp/yaml.h>
#include "chanModelsApi.hpp"
#include "config_reader.hpp"

struct TestBenchConfig {
    std::uint32_t nTti;
    uint8_t continuous_fading;
    bool enableHalfPrecision;
    bool enableSwapTxRx;
    std::uint16_t rand_seed;            //Random seed for simulation
};

TestBenchConfig readTestBenchConfig(const std::string& configFile) {
    TestBenchConfig config;
    try {
        YAML::Node yamlConfig = YAML::LoadFile(configFile);
        
        // Read test bench specific config
        if (yamlConfig["test_bench"]) {
            config.nTti = yamlConfig["test_bench"]["n_tti"].template as<std::uint32_t>();
            config.continuous_fading = yamlConfig["test_bench"]["continuous_fading"].template as<uint8_t>();
            config.enableHalfPrecision = yamlConfig["test_bench"]["enable_half_precision"].template as<bool>();
            config.enableSwapTxRx = yamlConfig["test_bench"]["enable_swap_tx_rx"].template as<bool>();
            

            config.rand_seed = yamlConfig["test_bench"]["rand_seed"].template as<std::uint16_t>(0);
            
            // Validate n_tti - must be at least 1
            if (config.nTti < 1) {
                std::fprintf(stderr, "WARNING: n_tti=%u is less than 1. Setting to minimum value of 1.\n", config.nTti);
                config.nTti = 1;
            }
        } else {
            // Default values if not found in config
            config.nTti = 1;
            config.continuous_fading = 1;
            config.enableHalfPrecision = false;
            config.enableSwapTxRx = false;
            config.rand_seed = 0;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error reading config file: " << e.what() << std::endl;
        throw;
    }
    return config;
}

template<typename Tscalar, typename Tcomplex>
void test_SLS(const std::string& config_file, const std::string& file_ending = "_test", bool enable_debug = false) {
    // Read configurations
    SystemLevelConfig system_config;
    LinkLevelConfig link_config;
    SimConfig sim_config;
    ExternalConfig external_config;
    TestBenchConfig tb_config = readTestBenchConfig(config_file);
    
    // Read main configuration
    ConfigReader::readConfig(config_file, system_config, link_config, sim_config, external_config);
    
    // Print configuration information
    std::printf("Configuration:\n");
    std::printf("  Scenario: %s\n", 
                system_config.scenario == Scenario::UMa ? "UMa" :
                system_config.scenario == Scenario::UMi ? "UMi" :
                system_config.scenario == Scenario::RMa ? "RMa" : "Other");
    std::printf("  Sites: %u, Sectors/site: %u, UTs: %u\n", 
                system_config.n_site, system_config.n_sector_per_site, system_config.n_ut);
    std::printf("  Internal memory mode: %u (0=external CIR/CFR, 1=internal CIR/external CFR, 2=internal CIR/CFR)\n", 
                sim_config.internal_memory_mode);
    std::printf("  Run mode: %u\n", sim_config.run_mode);
    std::printf("  Number of TTIs: %u\n", tb_config.nTti);
    
    // Print ISAC configuration if enabled
    if (system_config.isac_type > 0) {
        std::printf("  ISAC Mode: %s (type=%u)\n", 
                    system_config.isac_type == 1 ? "MONOSTATIC" : "BISTATIC",
                    system_config.isac_type);
        std::printf("    Sensing Targets: %u\n", system_config.n_st);
        std::printf("    Target Type: %s (type=%u)\n",
                    system_config.st_target_type == SensingTargetType::UAV ? "UAV" :
                    system_config.st_target_type == SensingTargetType::AUTOMOTIVE ? "AUTOMOTIVE" :
                    system_config.st_target_type == SensingTargetType::HUMAN ? "HUMAN" :
                    system_config.st_target_type == SensingTargetType::AGV ? "AGV" : 
                    system_config.st_target_type == SensingTargetType::HAZARD ? "HAZARD" : "UNKNOWN",
                    static_cast<unsigned int>(system_config.st_target_type));
        std::printf("    RCS Model: %u (%s)\n", 
                    system_config.st_rcs_model,
                    system_config.st_rcs_model == 1 ? "deterministic monostatic" : "angular dependent");
        std::printf("    ST Horizontal Speed Range: [%.1f, %.1f] m/s\n",
                    system_config.st_horizontal_speed[0], system_config.st_horizontal_speed[1]);
    } else {
        std::printf("  ISAC Mode: DISABLED (communication only)\n");
    }
    
    if (enable_debug) {
        std::printf("  Debug output: ENABLED (via -d flag)\n");
    }
    // Create CUDA stream only if not in CPU-only mode (to avoid GPU memory allocation)
    cudaStream_t cuMainStrm = nullptr;
    if (sim_config.cpu_only_mode == 0) {
        cudaStreamCreate(&cuMainStrm);
    }

    // Create SLS channel instance
    slsChan<Tscalar, Tcomplex>* slsChanTest = new slsChan<Tscalar, Tcomplex>(&sim_config, &system_config, &external_config, tb_config.rand_seed, cuMainStrm);

    // Print GPU memory usage
    float gpuMemUse = slsChanTest->getGpuMemUseMB();
    std::printf("GPU memory usage: %.2f MB\n", gpuMemUse);

    // Use "all links" mode with empty vectors (now properly supported)
    // Alternative: explicit per-cell mode (commented out)
    /*
    // Configure active cells and UTs for internal memory mode
    // Based on configuration: n_site=1, n_sector_per_site=3, n_ut=2
    // This creates 3 cells (0,1,2) and 2 UTs (0,1)
    std::vector<uint16_t> activeCell = {0, 1, 2}; // All 3 cells active
    std::vector<std::vector<uint16_t>> activeUt = {
        {0, 1}, // Both UTs active for cell 0
        {0, 1}, // Both UTs active for cell 1
        {0, 1}  // Both UTs active for cell 2
    };
    */

    // Calculate TTI length and simulation interval
    float ttiLen = 0.001 / (sim_config.sc_spacing_hz / 15e3); // TTI length

    // Time measurement setup
    const bool isCpuOnly = (sim_config.cpu_only_mode == 1);
    cudaEvent_t start, stop;
    if (!isCpuOnly)
    {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }
    std::vector<float> elapsedTimeCudaEvtVec(tb_config.nTti);
    std::vector<float> elapsedTimeCpuClockVec(tb_config.nTti);

    // Warm up GPU kernels
    slsChanTest->run(0, tb_config.continuous_fading);

    // Main simulation loop
    for(int ttiIdx = 0; ttiIdx < tb_config.nTti; ttiIdx++) {
        auto startTime = std::chrono::high_resolution_clock::now();

        if (!isCpuOnly)
        {
            cudaEventRecord(start, cuMainStrm);
        }
        
        try {
            // Run channel model with error handling, all links
            slsChanTest->run(ttiIdx * ttiLen, tb_config.continuous_fading);
            // Alternative: explicit active links mode
            // Set active cells and UTs dynamically based on configuration
            // std::vector<uint16_t> activeCell = {0};

            // // Generate active UTs from 0 to n_ut-1
            // std::vector<uint16_t> utList{};
            // utList.reserve(system_config.n_ut);
            // for (uint16_t utIdx = 0; utIdx < system_config.n_ut; ++utIdx) {
            //     utList.push_back(utIdx);
            // }
            // std::vector<std::vector<uint16_t>> activeUt = {utList};
            // slsChanTest->run(ttiIdx * ttiLen, tb_config.continuous_fading, activeCell, activeUt);
            
            // Ensure GPU operations complete before continuing (GPU mode)
            if (!isCpuOnly)
            {
                cudaEventRecord(stop, cuMainStrm);
                cudaEventSynchronize(stop);
            }
            
            // Access CFR data if debug mode is enabled via -d flag
            if (enable_debug && sim_config.run_mode >= 2 && ttiIdx == 0) {
                // Example: Access CFR SC data for first cell
                cuComplex* cfrScData = slsChanTest->getFreqChanSc(0); // Cell 0
                if (cfrScData != nullptr) {
                    std::printf("TTI %d: CFR SC data accessible (pointer: %p)\n", ttiIdx, (void*)cfrScData);
                    
                    // Optional: Copy first few values to host for inspection
                    const int sampleSize = 4;
                    cuComplex hostSamples[sampleSize];
                    if (isCpuOnly) {
                        // In CPU-only mode, pointer is host memory
                        std::memcpy(hostSamples, cfrScData, sampleSize * sizeof(cuComplex));
                    } else {
                        // In GPU mode, copy from device
                        cudaMemcpy(hostSamples, cfrScData, sampleSize * sizeof(cuComplex), cudaMemcpyDeviceToHost);
                    }
                    std::printf("  Sample CFR values: ");
                    for (int i = 0; i < sampleSize; i++) {
                        std::printf("(%.3e, %.3e) ", hostSamples[i].x, hostSamples[i].y);
                    }
                    std::printf("\n");
                } else {
                    std::printf("TTI %d: Internal CFR SC data not available (check memory mode)\n", ttiIdx);
                }
            }
            
        } catch (const std::exception& e) {
            std::fprintf(stderr, "Error in TTI %d: %s\n", ttiIdx, e.what());
            break;
        } catch (...) {
            std::fprintf(stderr, "Unknown error in TTI %d\n", ttiIdx);
            break;
        }
        
        printf("TTI %d: channel model run completed\n", ttiIdx);
        
        // Save H5 file at specific TTIs (only if debug mode is enabled via -d flag)
        if (enable_debug && (ttiIdx == 0 || ttiIdx == int(tb_config.nTti/2) || ttiIdx == tb_config.nTti-1)) {
            const std::string h5_filename = "_TTI" + std::to_string(ttiIdx) + file_ending;
            slsChanTest->saveSlsChanToH5File(h5_filename);
        }
        
        if (!isCpuOnly)
        {
            cudaEventElapsedTime(&elapsedTimeCudaEvtVec[ttiIdx], start, stop);
        }

        auto endTime = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float, std::milli> duration = endTime - startTime;
        elapsedTimeCpuClockVec[ttiIdx] = duration.count();
    }

    // Calculate and print statistics
    float avgCudaTime = 0.0f;
    float avgCpuTime = 0.0f;
    for(int i = 0; i < tb_config.nTti; i++) {
        if (!isCpuOnly) { avgCudaTime += elapsedTimeCudaEvtVec[i]; }
        avgCpuTime += elapsedTimeCpuClockVec[i];
    }
    if (!isCpuOnly) { avgCudaTime /= tb_config.nTti; }
    avgCpuTime /= tb_config.nTti;

    if (!isCpuOnly)
    {
        std::printf("Average CUDA event time: %.2f ms\n", avgCudaTime);
    }
    std::printf("Average CPU clock time: %.2f ms\n", avgCpuTime);

    // Cleanup
    delete slsChanTest;
    if (!isCpuOnly)
    {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaStreamDestroy(cuMainStrm);
    }
}

int main(int argc, char** argv) {
    std::string config_file;
    std::string file_ending = "";  // Default: empty (no H5 dump)
    bool dump_h5 = false;
    
    int opt;
    while ((opt = getopt(argc, argv, "dh")) != -1) {
        switch (opt) {
            case 'd':
                {
                    dump_h5 = true;
                    // Check if the next argument exists and is not another flag
                    if (optind < argc && argv[optind][0] != '-') {
                        // -d followed by an argument: use that argument
                        const std::string arg = std::string(argv[optind]);
                        // Avoid double underscore if user already provided underscore prefix
                        file_ending = (arg[0] != '_') ? "_" + arg : arg;
                        optind++;  // Consume this argument
                    } else {
                        // -d with no argument: use default
                        file_ending = "_test";
                    }
                }
                break;
            case 'h':
                std::cout << "Usage: " << argv[0] << " [-d [file_ending]] <config_file>" << std::endl;
                std::cout << "Options:" << std::endl;
                std::cout << "  -d [file_ending]  Enable debug mode: dump H5 files and show CFR data" << std::endl;
                std::cout << "                    If -d not provided: no H5 files dumped, no debug output" << std::endl;
                std::cout << "                    If -d with no arg: files dumped as TTI<n>_test" << std::endl;
                std::cout << "                    If -d with arg: files dumped as TTI<n>_<arg>" << std::endl;
                std::cout << "  -h                Show this help message" << std::endl;
                return 0;
            default:
                std::cerr << "Usage: " << argv[0] << " [-d [file_ending]] <config_file>" << std::endl;
                return 1;
        }
    }
    
    // Check if config file is provided
    if (optind >= argc) {
        std::cerr << "Error: Config file is required" << std::endl;
        std::cerr << "Usage: " << argv[0] << " [-d file_ending] <config_file>" << std::endl;
        return 1;
    }
    
    config_file = argv[optind];
    
    std::cout << "Using config file: " << config_file << std::endl;
    if (dump_h5) {
        std::cout << "H5 file dumping: ENABLED with ending: " << file_ending << std::endl;
    }
    
    test_SLS<float, cuComplex>(config_file, file_ending, dump_h5);
    return 0;
} 