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

#include "cuphy.h"
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include "hdf5hpp.hpp"
#include "cuphy.hpp"
#include "datasets.hpp"
#include "srs_rx.hpp"
#include "cuphy_channels.hpp"  // Add this include for bfw_tx
#include "bfw_tx.hpp"  // for bfw_tx pipeline
#include <fmt/format.h>
#include <unordered_map>
#include <unordered_set>
#include <CLI/CLI.hpp>

using Clock = std::chrono::high_resolution_clock;
using TimePoint = std::chrono::time_point<Clock>;
template <typename T, typename unit>
using duration = std::chrono::duration<T, unit>;

int main(int argc, char* argv[]) {
    int returnValue = 0;
    char nvlog_yaml_file[1024];
    std::string relative_path = std::string("../../../../").append(NVLOG_DEFAULT_CONFIG_FILE);
    std::string log_name = "srs_bfw.log";
    nv_get_absolute_path(nvlog_yaml_file, relative_path.c_str());
    pthread_t log_thread_id = -1;
    bool     useGreenCtxs   = false;
    uint32_t SMsPerGreenCtx = 0;

    try {
        // Setup CLI11 for command line argument parsing
        CLI::App app{"cuPHY SRS BFW Example"};
        app.footer("Notes:\n"
                   "    1. If only SRS or BFW is specified, the example behaves identical to standalone pipelines\n"
                   "    2. Supports multiple SRS input files and single BFW pipeline configuration\n"
                   "    3. When both pipelines are used, assumes all channel estimates needed by BFW\n"
                   "       are available from the SRS pipeline unless --rnti-warning is specified\n"
                   "    4. Multiple SRS files allow processing different SRS TVs for the same BFW operation\n"
                   "\n"
                   "Examples:\n"
                   "    Single SRS:     --srs file1.h5\n" 
                   "    Multiple SRS:   --srs file1.h5 file2.h5\n"
                   "    Multiple SRS:   --srs file1.h5 --srs file2.h5");
        
        // Command line arguments
        std::vector<std::string> srs_input_filenames{};
        std::string bfw_input_filename{};
        std::string output_filename{};
        uint64_t procModeBmsk{SRS_PROC_MODE_FULL_SLOT}; // default stream mode
        int32_t totalIters{1};
        bool rnti_not_found_as_warning{false};

        // Add CLI options
        app.add_option("--srs", srs_input_filenames, "SRS Input HDF5 filename(s) - supports multiple formats:\n"
                      "  --srs file1.h5 file2.h5 (space-separated)\n"
                      "  --srs file1.h5 --srs file2.h5 (repeated option)")
            ->check(CLI::ExistingFile)
            ->expected(1, -1);  // Accept 1 or more values
        
        app.add_option("--bfw", bfw_input_filename, "BFW Input HDF5 filename")
            ->check(CLI::ExistingFile);
        
        app.add_option("-m,--mode", procModeBmsk, "SRS proc mode: streams(0x0), graphs (0x1)")
            ->default_val(0x0);
        
        app.add_option("-r,--iterations", totalIters, "Number of run iterations")
            ->default_val(1)
            ->check(CLI::PositiveNumber);
        
        app.add_option("-o,--output", output_filename, "Write pipeline tensors and debug data to HDF5 files");
        
        app.add_flag("--rnti-warning", rnti_not_found_as_warning, 
            "Treat RNTI not found as warning instead of error (default: error)");

        app.add_option("--G", SMsPerGreenCtx, "Use green contexts with specified SM count per context")
            ->check(CLI::PositiveNumber)
            ->each([&useGreenCtxs](const std::string&) { useGreenCtxs = true; });

        // Custom validator to ensure at least one of SRS or BFW is specified
        app.callback([&]() {
            if (srs_input_filenames.empty() && bfw_input_filename.empty()) {
                throw CLI::ValidationError("At least one of --srs or --bfw must be specified");
            }
        });

        // Parse command line arguments
        try {
            app.parse(argc, argv);
        } catch (const CLI::ParseError &e) {
            return app.exit(e);
        }

        std::vector<std::string> srsInputFilenameVec, bfwInputFilenameVec;

        // Determine which pipelines to run
        bool runSrs = !srs_input_filenames.empty();
        bool runBfw = !bfw_input_filename.empty();

        if (runSrs) {
            srsInputFilenameVec = srs_input_filenames;  // Use all SRS input files
            NVLOGC_FMT(NVLOG_SRS, "Will process {} SRS input file(s)", srsInputFilenameVec.size());
            for (const auto& filename : srsInputFilenameVec) {
                NVLOGC_FMT(NVLOG_SRS, "  SRS input: {}", filename);
            }
        }
        if (runBfw) {
            bfwInputFilenameVec.push_back(bfw_input_filename);
        }

        // Initialize CUDA
        const int gpuId = 0; // select GPU device 0
        CUDA_CHECK(cudaSetDevice(gpuId));
        CUdevice current_device;
        CU_CHECK(cuDeviceGet(&current_device, gpuId));

#if CUDA_VERSION >= 12040
        CUdevResource initial_device_GPU_resources = {};
        CUdevResourceType default_resource_type = CU_DEV_RESOURCE_TYPE_SM;
        CUdevResource split_result[2] = {{}, {}};
        cuphy::cudaGreenContext srs_bfw_green_ctx;
        unsigned int split_groups = 1;

        if(useGreenCtxs)
        {
            // Best to ensure that MPS service is not running
            int mpsEnabled = 0;
            CU_CHECK(cuDeviceGetAttribute(&mpsEnabled, CU_DEVICE_ATTRIBUTE_MPS_ENABLED, current_device));
            if (mpsEnabled == 1) {
                NVLOGE_FMT(NVLOG_TAG_BASE_CUPHY, AERIAL_CUPHY_EVENT, "MPS is enabled. Heads-up that currently using green contexts with MPS enabled can have unintended side effects. Will run regardless.");
            } else {
                NVLOGC_FMT(NVLOG_TAG_BASE_CUPHY, "MPS service is not running.");
            }

            // Check SMsPerGreenCtxs value is in valid range
            int32_t gpuMaxSmCount = 0;
            CU_CHECK(cuDeviceGetAttribute(&gpuMaxSmCount, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, current_device));
            if (SMsPerGreenCtx > static_cast<uint32_t>(gpuMaxSmCount))
            {
                NVLOGE_FMT(NVLOG_TAG_BASE_CUPHY, AERIAL_CUPHY_EVENT, "ERROR: Invalid --G argument {}. It is greater than {} (GPU's max SMs).", SMsPerGreenCtx, gpuMaxSmCount);
                return 1;
            }

            CU_CHECK(cuDeviceGetDevResource(current_device, &initial_device_GPU_resources, default_resource_type));
            CU_CHECK(cuDevSmResourceSplitByCount(&split_result[0], &split_groups, &initial_device_GPU_resources, &split_result[1], 0, SMsPerGreenCtx));
            srs_bfw_green_ctx.create(gpuId, &split_result[0]);
            srs_bfw_green_ctx.bind();
            NVLOGC_FMT(NVLOG_SRS, "SRS_BFW green context will have access to {} SMs ({} SMs requested).", srs_bfw_green_ctx.getSmCount(), SMsPerGreenCtx);
        }
#endif

        cuphy::stream cuStrmMain(cudaStreamNonBlocking);
        cudaStream_t cuStrm = cuStrmMain.handle();

        log_thread_id = nvlog_fmtlog_init(nvlog_yaml_file, log_name.c_str(), NULL);
        nvlog_fmtlog_thread_init();

        // Initialize timing measurements
        std::vector<cuphy::event_timer> srsEvtTimers(totalIters);
        std::vector<cuphy::event_timer> bfwEvtTimers(totalIters);

        // After parsing arguments, before creating pipelines:
        std::string srs_debug_filename, bfw_debug_filename;
        std::unique_ptr<hdf5hpp::hdf5_file> srsDebugFile;
        std::unique_ptr<hdf5hpp::hdf5_file> bfwDebugFile;

        if(!output_filename.empty()) {
            // Create debug filenames based on input files
            srs_debug_filename = "srsDebug_" + output_filename;  // assuming output_filename is the pending filename
            bfw_debug_filename = "bfwDebug_" + output_filename;  // assuming output_filename is the pending filename
            
            // Create debug files
            srsDebugFile.reset(new hdf5hpp::hdf5_file(hdf5hpp::hdf5_file::create(srs_debug_filename.c_str())));
            bfwDebugFile.reset(new hdf5hpp::hdf5_file(hdf5hpp::hdf5_file::create(bfw_debug_filename.c_str())));
            
            NVLOGC_FMT(NVLOG_SRS, "Will write SRS debug output to: {}", srs_debug_filename);
            NVLOGC_FMT(NVLOG_BFW, "Will write BFW debug output to: {}", bfw_debug_filename);
        }

        cuphySrsRxHndl_t srsRxHndl = nullptr;
        std::unique_ptr<class srsStaticApiDataset> srsStaticApiDataset;
        std::unique_ptr<class srsDynApiDataset> srsDynApiDataset;
        std::unique_ptr<class srsEvalDataset> srsEvalDataset;
        std::unordered_map<uint16_t, uint16_t> srsRntiUeIdxMap;
        
        std::unique_ptr<class bfwStaticApiDataset> bfwStaticApiDataset;
        std::unique_ptr<class bfwDynApiDataset> bfwDynApiDataset;
        std::unique_ptr<class bfwEvalDataset> bfwEvalDataset;
        std::unique_ptr<cuphy::bfw_tx> bfwTxPipeline;

        std::unordered_set<uint16_t> processedBfwIdx;

        // Initialize SRS if needed
        if (runSrs) {
            srsStaticApiDataset.reset(new class srsStaticApiDataset(srsInputFilenameVec, cuStrm, srs_debug_filename));
            srsDynApiDataset.reset(new class srsDynApiDataset(srsInputFilenameVec, cuStrm, procModeBmsk));
            srsEvalDataset.reset(new class srsEvalDataset(srsInputFilenameVec, cuStrm));
            cuStrmMain.synchronize();

            // Create a map of RNTI to SRS UE index
            NVLOGC_FMT(NVLOG_SRS, "Creating RNTI to UE index mapping for {} total UEs from {} SRS file(s)", 
                srsDynApiDataset->ueSrsPrmVec.size(), srsInputFilenameVec.size());
            
            for(uint16_t i = 0; i < srsDynApiDataset->ueSrsPrmVec.size(); i++) {
                const uint16_t rnti = srsDynApiDataset->ueSrsPrmVec[i].RNTI;
                srsRntiUeIdxMap[rnti] = i;
                NVLOGD_FMT(NVLOG_SRS, "  UE[{}]: RNTI={}", i, rnti);
            }
            NVLOGC_FMT(NVLOG_SRS, "Created RNTI mapping for {} unique RNTIs", srsRntiUeIdxMap.size());

            cuphyStatus_t statusCreate = cuphyCreateSrsRx(&srsRxHndl, &srsStaticApiDataset->srsStatPrms, cuStrm);
            if(CUPHY_STATUS_SUCCESS != statusCreate) throw cuphy::cuphy_exception(statusCreate);
        }

        // Initialize BFW if needed
        if (runBfw) {
            bfwStaticApiDataset.reset(new class bfwStaticApiDataset(bfwInputFilenameVec, cuStrm, bfw_debug_filename));
            bfwDynApiDataset.reset(new class bfwDynApiDataset(bfwInputFilenameVec, cuStrm, procModeBmsk));
            bfwEvalDataset.reset(new class bfwEvalDataset(bfwInputFilenameVec, cuStrm));
            bfwTxPipeline.reset(new cuphy::bfw_tx(bfwStaticApiDataset->bfwStatPrms, cuStrm));
        }

        for(int iterIdx = 0; iterIdx < totalIters; iterIdx++) {
            // Run SRS pipeline if enabled
            if (runSrs) {
                cuphySetupSrsRx(srsRxHndl, &srsDynApiDataset->srsDynPrm, nullptr);

                srsEvtTimers[iterIdx].record_begin(cuStrm);
                cuphyRunSrsRx(srsRxHndl, procModeBmsk);
                srsEvtTimers[iterIdx].record_end(cuStrm);
                cuStrmMain.synchronize();

                // Write SRS debug data if output is enabled
                if(!output_filename.empty()) {
                    cuphyStatus_t statusDebugWrite = cuphyWriteDbgBufSynchSrs(srsRxHndl, cuStrm);
                    cuStrmMain.synchronize();
                    if(CUPHY_STATUS_SUCCESS != statusDebugWrite) throw cuphy::cuphy_exception(statusDebugWrite);
                }

                // Evaluate SRS results
                srsEvalDataset->evalSrsRx(srsDynApiDataset->srsDynPrm, srsDynApiDataset->tSrsChEstVec, 
                    srsDynApiDataset->dataOut.pRbSnrBuffer, srsDynApiDataset->dataOut.pSrsReports, cuStrm);
            }

            // Run BFW pipeline if enabled
            if (runBfw) {
                // Only try to copy SRS data if both pipelines are running
                if (runSrs) {
                    const uint16_t nSrsUes = srsDynApiDataset->tSrsChEstVec.size();
                    const uint16_t nBfwUeGrps = bfwDynApiDataset->bfwUeGrpPrmVec.size();
                    const uint16_t nBfwLayers = bfwDynApiDataset->bfwLayerPrmVec.size();

                    NVLOGC_FMT(NVLOG_BFW, "Total SRS UEs: {}, BFW UE Groups: {}, BFW Layers: {}", nSrsUes, nBfwUeGrps, nBfwLayers);

                    // Process channel estimates for each BFW layer
                    for(int layerIdx = 0; layerIdx < nBfwLayers; layerIdx++) {
                        uint16_t bfwIdx = bfwDynApiDataset->bfwLayerPrmVec[layerIdx].chEstInfoBufIdx;  // BFW UE idx
                        if (processedBfwIdx.find(bfwIdx) != processedBfwIdx.end()) {
                            NVLOGD_FMT(NVLOG_BFW, "Skipping already processed BFW UE idx: {} for layer[{}]", bfwIdx, layerIdx);
                            continue;
                        }
                        processedBfwIdx.insert(bfwIdx);

                        uint16_t srsIdx;  // SRS UE idx
                        if (srsRntiUeIdxMap.find(bfwIdx + 1) != srsRntiUeIdxMap.end()) {
                            srsIdx = srsRntiUeIdxMap[bfwIdx + 1];  // Get SRS index using RNTI=chEstInfoBufIdx+1
                        } else {
                            if (rnti_not_found_as_warning) {
                                NVLOGW_FMT(NVLOG_BFW, "RNTI {} not found in srsRntiUeIdxMap", bfwIdx + 1);
                                continue;
                            } else {
                                NVLOGE_FMT(NVLOG_BFW, AERIAL_CUPHY_EVENT, "RNTI {} not found in srsRntiUeIdxMap", bfwIdx + 1);
                                throw std::runtime_error(fmt::format("RNTI {} not found in srsRntiUeIdxMap", bfwIdx + 1));
                            }
                        }

                        NVLOGC_FMT(NVLOG_BFW, "Processing BFW UE[{}] for layer[{}], RNTI={}, using SRS UE idx={}", 
                            bfwIdx, layerIdx, bfwIdx + 1, srsIdx);

                        // get SRS and BFW channel estimates buffer
                        auto& srsChEst = srsDynApiDataset->tSrsChEstVec[srsIdx];
                        auto& bfwChEst = bfwDynApiDataset->tSrsChEstVec[bfwIdx];

                        // Get and print all dimensions
                        const uint32_t dim0_srs = srsChEst.desc().get_dim(0);  // 136 PRBGs
                        const uint32_t dim1_srs = srsChEst.desc().get_dim(1);  // 64 RxAnts
                        const uint32_t dim2_srs = srsChEst.desc().get_dim(2);  // 2 Layers
                        
                        const uint32_t dim0_bfw = bfwChEst.desc().get_dim(0);  // 137 PRBGs
                        const uint32_t dim1_bfw = bfwChEst.desc().get_dim(1);  // 64 RxAnts
                        const uint32_t dim2_bfw = bfwChEst.desc().get_dim(2);  // 2 Layers

                        NVLOGD_FMT(NVLOG_BFW, "SRS Tensor dimensions [{}][{}][{}] PRBGs x RxAnts x Layers", 
                            dim0_srs, dim1_srs, dim2_srs);
                        NVLOGD_FMT(NVLOG_BFW, "BFW Tensor dimensions [{}][{}][{}] PRBGs x RxAnts x Layers", 
                            dim0_bfw, dim1_bfw, dim2_bfw);

                        // Use CUDA's built-in complex type
                        using complex_half = __half2;  // CUDA's built-in complex half type

                        // For each RxAnt and Layer combination
                        for (uint32_t rx = 0; rx < dim1_srs; rx++) {
                            for (uint32_t layer = 0; layer < dim2_srs; layer++) {
                                // Calculate offsets using strides
                                const size_t srs_offset = rx * srsChEst.layout().strides()[1] + 
                                                        layer * srsChEst.layout().strides()[2];
                                const size_t bfw_offset = rx * bfwChEst.layout().strides()[1] + 
                                                        layer * bfwChEst.layout().strides()[2];

                                // Direct D2D copy of continuous PRB line
                                cudaError_t status = cudaMemcpy(
                                    static_cast<char*>(bfwChEst.addr()) + bfw_offset * sizeof(__half2),
                                    static_cast<char*>(srsChEst.addr()) + srs_offset * sizeof(__half2),
                                    std::min(dim0_srs, dim0_bfw) * sizeof(__half2),  // NOTE: SRS always uses 136 PRBGs for chEst buffer, BFW uses actual PRBGs for chEst buffer
                                    cudaMemcpyDeviceToDevice);
                                if (status != cudaSuccess) {
                                    NVLOGE_FMT(NVLOG_BFW, AERIAL_CUPHY_EVENT, 
                                        "D2D copy failed for RxAnt={}, Layer={}: {}", 
                                        rx, layer, cudaGetErrorString(status));
                                    continue;
                                }

                                // Copy the last PRB if needed
                                if (dim0_bfw > dim0_srs) {
                                    status = cudaMemcpy(
                                        static_cast<char*>(bfwChEst.addr()) + 
                                            (bfw_offset + dim0_srs) * sizeof(__half2),
                                        static_cast<char*>(bfwChEst.addr()) + 
                                            (bfw_offset + dim0_srs - 1) * sizeof(__half2),
                                        sizeof(__half2),
                                        cudaMemcpyDeviceToDevice);
                                    if (status != cudaSuccess) {
                                        NVLOGE_FMT(NVLOG_BFW, AERIAL_CUPHY_EVENT, 
                                            "D2D copy for last PRB failed for RxAnt={}, Layer={}: {}", 
                                            rx, layer, cudaGetErrorString(status));
                                        continue;
                                    }
                                    NVLOGD_FMT(NVLOG_BFW, "Copied last PRB for RxAnt={}, Layer={}", 
                                        rx, layer);
                                }

                                // Debug: Copy first few values to host only if needed
                                if (rx == 0 && layer == 0) {
                                    __half2 debugBuffer[4];
                                    status = cudaMemcpy(
                                        debugBuffer,
                                        static_cast<char*>(bfwChEst.addr()) + bfw_offset * sizeof(__half2),
                                        4 * sizeof(__half2),
                                        cudaMemcpyDeviceToHost);
                                    if (status == cudaSuccess) {
                                        for (int i = 0; i < 4; i++) {
                                            float real = __half2float(debugBuffer[i].x);
                                            float imag = __half2float(debugBuffer[i].y);
                                            NVLOGD_FMT(NVLOG_BFW, 
                                                "rx={}, layer={}, prb={}, val={:f} + {:f}j",
                                                rx, layer, i, real, imag);
                                        }
                                    }
                                }

                                NVLOGD_FMT(NVLOG_BFW, "Completed copy for RxAnt={}, Layer={}", 
                                    rx, layer);
                            }
                        }
                        NVLOGD_FMT(NVLOG_BFW, "Completed copying all channel estimates of BFW UE[{}] for layer[{}]", bfwIdx, layerIdx);
                    }
                    cuStrmMain.synchronize();
                }
                
                bfwTxPipeline->setup(bfwDynApiDataset->bfwDynPrms);

                bfwEvtTimers[iterIdx].record_begin(cuStrm);
                bfwTxPipeline->run(procModeBmsk);
                bfwEvtTimers[iterIdx].record_end(cuStrm);
                cuStrmMain.synchronize();

                // Write BFW debug data if output is enabled
                if(!output_filename.empty()) {
                    bfwTxPipeline->writeDbgSynch(cuStrm);
                }

                // Evaluate BFW results
                bfwEvalDataset->bfwEvalCoefs(*bfwStaticApiDataset, *bfwDynApiDataset, cuStrm, 30.0f, true);
            }
        }

        // Calculate and print timing statistics
        if (runSrs) {
            float totalSrsTime = 0.0f;
            for(int i = 0; i < totalIters; i++) {
                srsEvtTimers[i].synchronize();
                totalSrsTime += srsEvtTimers[i].elapsed_time_ms();
            }
            NVLOGC_FMT(NVLOG_SRS, "Average SRS processing time (ms): {:.4f}", totalSrsTime/totalIters);
        }

        if (runBfw) {
            float totalBfwTime = 0.0f;
            for(int i = 0; i < totalIters; i++) {
                bfwEvtTimers[i].synchronize();
                totalBfwTime += bfwEvtTimers[i].elapsed_time_ms();
            }
            NVLOGC_FMT(NVLOG_BFW, "Average BFW processing time (ms): {:.4f}", totalBfwTime/totalIters);
        }

        // Cleanup
        if (srsRxHndl) {
            cuphyDestroySrsRx(srsRxHndl);
        }

    } catch(std::exception& e) {
        NVLOGE_FMT(NVLOG_SRS, AERIAL_CUPHY_EVENT, "EXCEPTION: {}", e.what());
        returnValue = 1;
    } catch(...) {
        NVLOGE_FMT(NVLOG_SRS, AERIAL_CUPHY_EVENT, "UNKNOWN EXCEPTION");
        returnValue = 2;
    }
    
    nvlog_fmtlog_close(log_thread_id);
    return returnValue;
} 