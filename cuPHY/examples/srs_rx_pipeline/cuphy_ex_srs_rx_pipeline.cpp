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
#include <cstdio>
#include <cstdlib>
#include <string>
#include "hdf5hpp.hpp"
#include "cuphy.hpp"
#include "datasets.hpp"
#include "test_config.hpp"
#include "srs_rx.hpp"

#include <dirent.h> // opendir, readdir

#include "CLI/CLI.hpp"

using Clock     = std::chrono::high_resolution_clock;
using TimePoint = std::chrono::time_point<Clock>;
template <typename T, typename unit>
    using duration = std::chrono::duration<T, unit>;

////////////////////////////////////////////////////////////////////////
// main()
int main(int argc, char* argv[])
{
    int returnValue = 0;
    char nvlog_yaml_file[1024];
    // Relative path from binary to default nvlog_config.yaml
    std::string relative_path = std::string("../../../../").append(NVLOG_DEFAULT_CONFIG_FILE);
    std::string log_name = "srs_rx.log";
    nv_get_absolute_path(nvlog_yaml_file, relative_path.c_str());
    pthread_t log_thread_id = -1;
    try
    {
        //------------------------------------------------------------------
        // Parse command line arguments
        CLI::App app{"cuphy_ex_srs_rx_pipeline"};
        std::string              inputFilename, outputFilename;
        std::vector<std::vector<std::string>> inputFilenameVec;
        uint64_t procModeBmsk   = SRS_PROC_MODE_FULL_SLOT; // default stream mode
        int32_t  totalIters     = 1000;
        uint32_t                 SMsPerGreenCtx = 0;
        bool                     skipEval       = false;
        std::ignore = app.add_option("-i", inputFilename, "Input HDF5 or YAML filename")->required();
        std::ignore = app.add_option("-m", procModeBmsk, "SRS processing mode: streams(0x0), graphs (0x1) (default = 0x0)");
        std::ignore = app.add_option("-r", totalIters, "# of iterations, Number of run iterations to run (default = 1000)");
        std::ignore = app.add_option("-o", outputFilename, "Output HDFS debug file");
        std::ignore = app.add_option("--G", SMsPerGreenCtx, "Use green contexts (GC) with specified SM count per context (default = 0; GC disabled)");
        std::ignore = app.add_flag("--skip-eval", skipEval, "Skip reference evaluation (for TVs without a valid golden, e.g. live captures)");
        CLI11_PARSE(app, argc, argv)

        if (totalIters <= 0) {
            const auto err = fmt::format("Invalid number of run iterations: {}, need to be >0", totalIters);
            NVLOGE_FMT(NVLOG_SRS, AERIAL_CUPHY_EVENT, "{}", err);
            return app.exit({"Parse error", err});
        }
        if (procModeBmsk != SRS_PROC_MODE_FULL_SLOT and procModeBmsk != SRS_PROC_MODE_FULL_SLOT_GRAPHS) {
            const auto err = fmt::format("Invalid processing mode (0x{:x})", procModeBmsk);
            NVLOGE_FMT(NVLOG_SRS, AERIAL_CUPHY_EVENT, "{}", err);
            return app.exit({"Parse error", err});
        }

        //------------------------------------------------------------------
        // input files
        std::string inFileExtn = inputFilename.substr(inputFilename.find_last_of(".") + 1);
        if(inFileExtn == "yaml")
        {
            // yaml parsing
            cuphy::test_config testCfg(inputFilename.c_str());
            int                nCells           = testCfg.num_cells();
            int                nSlots           = testCfg.num_slots();
            const std::string  channelName = "SRS";

            // Resize inputFilenameVec to have nSlots slots
            inputFilenameVec.resize(nSlots);

            try
            {
                for(size_t idxSlot = 0; idxSlot < nSlots; idxSlot++)
                {
                    for(int idxCell = 0; idxCell < nCells; idxCell++)
                    {
                        auto fname = testCfg.slots()[idxSlot].at(channelName)[idxCell];
                        inputFilenameVec[idxSlot].push_back(fname);
                    }
                }
            }
            catch(...)
            {
                throw std::runtime_error("SRS channel name not found in the input file");
            }
            // Verify each slot has the correct number of files
            for (size_t idxSlot = 0; idxSlot < nSlots; idxSlot++)
            {
                if (inputFilenameVec[idxSlot].size() != static_cast<size_t>(nCells)) {
                    throw std::runtime_error(fmt::format(
                        "Slot {} expected {} SRS files, got {}", idxSlot, nCells,
                        inputFilenameVec[idxSlot].size()));
                }
            }
        }
        else
        {
            // Single file case - create one slot with one file
            inputFilenameVec.resize(1);
            inputFilenameVec[0].push_back(inputFilename);
        }
        //------------------------------------------------------------------

        int gpuId = 0; // select GPU device 0
        CUDA_CHECK(cudaSetDevice(gpuId));
        CUdevice current_device;
        CU_CHECK(cuDeviceGet(&current_device, gpuId));

#if CUDA_VERSION >= 12040
        CUdevResource initial_device_GPU_resources = {};
        CUdevResourceType default_resource_type = CU_DEV_RESOURCE_TYPE_SM; // other alternative is CU_DEV_RESOURCE_TYPE_INVALID
        CUdevResource split_result[2] = {{}, {}};
        cuphy::cudaGreenContext srs_green_ctx;
        unsigned int split_groups = 1;
        // ---------------------------------------------------------------
        if(SMsPerGreenCtx > 0)
        {
            // Best to ensure that MPS service is not running
            int mpsEnabled = 0;
            CU_CHECK(cuDeviceGetAttribute(&mpsEnabled, CU_DEVICE_ATTRIBUTE_MPS_ENABLED, current_device));
            if (mpsEnabled == 1) {
                NVLOGE_FMT(NVLOG_TAG_BASE_CUPHY, AERIAL_CUPHY_EVENT,  "MPS is enabled. Heads-up that currently using green contexts with MPS enabled can have unintended side effects. Will run regardless.");
                //exit(1);
            } else {
                NVLOGC_FMT(NVLOG_TAG_BASE_CUPHY, "MPS service is not running.");
            }

            //Check SMsPerGreenCtxs value is in valid range
            int32_t gpuMaxSmCount = 0;
            CU_CHECK(cuDeviceGetAttribute(&gpuMaxSmCount, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, current_device));
            if (SMsPerGreenCtx > (uint32_t)gpuMaxSmCount)
            {
                NVLOGE_FMT(NVLOG_TAG_BASE_CUPHY, AERIAL_CUPHY_EVENT,  "ERROR: Invalid --G argument {}. It is greater than {} (GPU's max SMs).", SMsPerGreenCtx, gpuMaxSmCount);
                exit(1);
            }

            CU_CHECK(cuDeviceGetDevResource(current_device, &initial_device_GPU_resources, default_resource_type));
            CU_CHECK(cuDevSmResourceSplitByCount(&split_result[0], &split_groups, &initial_device_GPU_resources, &split_result[1], 0, SMsPerGreenCtx));
            srs_green_ctx.create(gpuId, &split_result[0]);
            srs_green_ctx.bind();
            NVLOGC_FMT(NVLOG_SRS, "SRS green context will have access to {} SMs ({} SMs requested).", srs_green_ctx.getSmCount(), SMsPerGreenCtx);
        }
#endif

        // ---------------------------------------------------------------

        log_thread_id = nvlog_fmtlog_init(nvlog_yaml_file, log_name.c_str(),NULL);
        nvlog_fmtlog_thread_init();
        if(procModeBmsk == SRS_PROC_MODE_FULL_SLOT_GRAPHS)
        {
            NVLOGI_FMT(NVLOG_SRS, "CUDA graph mode");
        } else {
            NVLOGI_FMT(NVLOG_SRS, "CUDA stream mode");
        }
        // CUDA Timers
        cuphy::event_timer evtTmrSetup;
        cuphy::event_timer evtTmrRun;
        // CPU Start/Stop times
        TimePoint timePtStartSetup, timePtStopSetup;
        TimePoint timePtStartRun, timePtStopRun;

        duration<float, std::micro> elpasedTimeDurationUs;
        typedef enum _elapsedTypes
        {
            ELAPSED_CPU_SETUP = 0,
            ELAPSED_EVT_SETUP = 1,
            ELAPSED_CPU_RUN   = 2,
            ELAPSED_EVT_RUN   = 3,
            ELAPSED_TYPES_MAX
        } elapsedTypes_t;
        std::array<std::vector<float>,ELAPSED_TYPES_MAX> m_elapsedTimes;
        for(auto& timeVec : m_elapsedTimes)
        {
            timeVec.resize(totalIters);
        }

        //-----------------------------------------------------------------
        // Open debug file

        std::unique_ptr<hdf5hpp::hdf5_file> debugFile;
        if(!outputFilename.empty())
        {
            debugFile.reset(new hdf5hpp::hdf5_file(hdf5hpp::hdf5_file::create(outputFilename.c_str())));
        }

        //------------------------------------------------------------------
        // Load API parameters
        cuphy::stream cuStrmMain(cudaStreamNonBlocking);
        cudaStream_t  cuStrm           = cuStrmMain.handle();

        // Get number of slots for processing
        int nSlots = inputFilenameVec.size();
        
        // Loop over each slot
        for(int slotIdx = 0; slotIdx < nSlots; slotIdx++)
        {
            NVLOGC_FMT(NVLOG_SRS, "Processing slot {} with {} files", slotIdx, inputFilenameVec[slotIdx].size());
            
            srsStaticApiDataset  srsStaticApiDataset(inputFilenameVec[slotIdx], cuStrm, outputFilename);
            srsDynApiDataset     srsDynApiDataset(inputFilenameVec[slotIdx],    cuStrm, procModeBmsk);
            srsEvalDataset       srsEvalDataset(inputFilenameVec[slotIdx],      cuStrm);
            cuStrmMain.synchronize(); // synch to ensure data copied

            //------------------------------------------------------------------
            // Create srs receiver object

            cuphySrsRxHndl_t srsRxHndl;
            
            cuphyStatus_t statusCreate = cuphyCreateSrsRx(&srsRxHndl, &srsStaticApiDataset.srsStatPrms, cuStrm);
            if(CUPHY_STATUS_SUCCESS != statusCreate) throw cuphy::cuphy_exception(statusCreate);


            for(int iterIdx=0; iterIdx<totalIters; iterIdx++)
            {
                
                auto& elapsedTimeUsSetup      = m_elapsedTimes[ELAPSED_CPU_SETUP][iterIdx];
                auto& elapsedTimeUsRun        = m_elapsedTimes[ELAPSED_CPU_RUN][iterIdx];
                auto& elapsedEvtTimeUsSetup   = m_elapsedTimes[ELAPSED_EVT_SETUP][iterIdx];
                auto& elapsedEvtTimeUsRun     = m_elapsedTimes[ELAPSED_EVT_RUN][iterIdx];

                //------------------------------------------------------------------
                // Setup srs receiver object

                cuphySrsBatchPrmHndl_t const batchPrmHndl = nullptr;  // batchPrms currently un-used

                // Record GPU & CPU time before setup
                evtTmrSetup.record_begin(cuStrm);
                timePtStartSetup = Clock::now();
                cuphyStatus_t statusSetup = cuphySetupSrsRx(srsRxHndl, &srsDynApiDataset.srsDynPrm, batchPrmHndl);
                // Record GPU & CPU time after setup
                timePtStopSetup = Clock::now();
                evtTmrSetup.record_end(cuStrm);

                if(CUPHY_STATUS_SUCCESS != statusSetup) throw cuphy::cuphy_exception(statusSetup);

                //------------------------------------------------------------------
                // Run srs receiver object


                // Record GPU & CPU time before run
                evtTmrRun.record_begin(cuStrm);
                timePtStartRun = Clock::now();
                cuphyStatus_t statusRun = cuphyRunSrsRx(srsRxHndl, procModeBmsk);
                // Record GPU & CPU time after run
                timePtStopRun = Clock::now();
                evtTmrRun.record_end(cuStrm);

                if(CUPHY_STATUS_SUCCESS != statusRun) throw cuphy::cuphy_exception(statusRun);

                //------------------------------------------------------------------

                // Process timing data
                evtTmrSetup.synchronize();
                evtTmrRun.synchronize();
                cuStrmMain.synchronize();

                elpasedTimeDurationUs = timePtStopSetup - timePtStartSetup;
                elapsedTimeUsSetup    = elpasedTimeDurationUs.count();
                elpasedTimeDurationUs = timePtStopRun - timePtStartRun;
                elapsedTimeUsRun      = elpasedTimeDurationUs.count();
                elapsedEvtTimeUsSetup = evtTmrSetup.elapsed_time_ms()*1000;
                elapsedEvtTimeUsRun   = evtTmrRun.elapsed_time_ms()*1000;
            }

            //------------------------------------------------------------------
            // Evaluate results    

            cudaStreamSynchronize(cuStrm);
            if(!skipEval)
            {
                srsEvalDataset.evalSrsRx(srsDynApiDataset.srsDynPrm, srsDynApiDataset.tSrsChEstVec, srsDynApiDataset.dataOut.pRbSnrBuffer, srsDynApiDataset.dataOut.pSrsReports, cuStrm);
            }
            else
            {
                // No golden reference (e.g. live capture): report chEst NaN%/max|H|
                // directly so numeric issues (e.g. FP16 overflow) can be spotted
                // without eval. Allocate hHest with align_tight so it is contiguous
                // regardless of the source layout's strides (convert() is stride-aware),
                // which makes the linear p[2*i] traversal below valid by construction.
                for(size_t ue = 0; ue < srsDynApiDataset.tSrsChEstVec.size(); ++ue)
                {
                    cuphy::typed_tensor<CUPHY_C_32F, cuphy::pinned_alloc> hHest(srsDynApiDataset.tSrsChEstVec[ue].layout(), cuphy::tensor_flags::align_tight);
                    hHest.convert(srsDynApiDataset.tSrsChEstVec[ue], cuStrm);
                    cudaStreamSynchronize(cuStrm);

                    const auto& dims = hHest.layout().dimensions();
                    size_t nElem = 1;
                    for(int d = 0; d < hHest.layout().rank(); ++d) { nElem *= static_cast<size_t>(dims[d]); }
                    const float* p     = reinterpret_cast<const float*>(hHest.addr());
                    size_t nBad = 0;
                    float  maxMag = 0.f;
                    for(size_t i = 0; i < nElem; ++i)
                    {
                        const float re = p[2*i], im = p[2*i + 1];
                        if(std::isnan(re) || std::isnan(im) || std::isinf(re) || std::isinf(im)) { ++nBad; continue; }
                        const float mag = std::sqrt(re*re + im*im);
                        if(mag > maxMag) maxMag = mag;
                    }
                    printf("[skip-eval] chEst UE %zu: elems %zu, NaN/Inf %zu (%.2f%%), max|H| %.3f\n",
                           ue, nElem, nBad, nElem ? 100.0 * double(nBad) / double(nElem) : 0.0, maxMag);
                }
            }
            
            //------------------------------------------------------------------
            // Write debug output

            if(!outputFilename.empty())
            {
                cuphyStatus_t statusDebugWrite = cuphyWriteDbgBufSynchSrs(srsRxHndl, cuStrm);
                cuStrmMain.synchronize();
                if(CUPHY_STATUS_SUCCESS != statusDebugWrite) throw cuphy::cuphy_exception(statusDebugWrite);
            }

            // Timing Debug
            float avgElapsedTimesUs[ELAPSED_TYPES_MAX];
            float minElapsedTimesUs[ELAPSED_TYPES_MAX];
            float maxElapsedTimesUs[ELAPSED_TYPES_MAX];
            for(int i=0;i<ELAPSED_TYPES_MAX;i++)
            {
                const auto minmax_pair = std::minmax_element(std::begin(m_elapsedTimes[i]),std::end(m_elapsedTimes[i]));
                float mean = std::accumulate(std::begin(m_elapsedTimes[i]),std::end(m_elapsedTimes[i]),0.0)/m_elapsedTimes[i].size();
                avgElapsedTimesUs[i] = mean;
                minElapsedTimesUs[i] = *minmax_pair.first;
                maxElapsedTimesUs[i] = *minmax_pair.second;

            }

            NVLOGC_FMT(NVLOG_SRS,"Slot {} timing results {}, format: avg (min, max) ", slotIdx,
                procModeBmsk == SRS_PROC_MODE_FULL_SLOT ? "in stream mode" : "in graph mode");

            NVLOGC_FMT(NVLOG_SRS,"{} Pipeline[{:02d}]: Slot {} - GPU Time usec (using CUDA events, over {:04d} runs): Run {: 9.4f} ({: 9.4f}, {: 9.4f}) Setup {: 9.4f} ({: 9.4f}, {: 9.4f}) Total {: 9.4f}",
                   "SrsRx",
                   slotIdx,
                   slotIdx,
                   m_elapsedTimes[0].size(),
                   avgElapsedTimesUs[ELAPSED_EVT_RUN],
                   minElapsedTimesUs[ELAPSED_EVT_RUN],
                   maxElapsedTimesUs[ELAPSED_EVT_RUN],
                   avgElapsedTimesUs[ELAPSED_EVT_SETUP],
                   minElapsedTimesUs[ELAPSED_EVT_SETUP],
                   maxElapsedTimesUs[ELAPSED_EVT_SETUP],
                   avgElapsedTimesUs[ELAPSED_EVT_RUN] + avgElapsedTimesUs[ELAPSED_EVT_SETUP]);

            NVLOGC_FMT(NVLOG_SRS,"{} Pipeline[{:02d}]: Slot {} - CPU Time usec (using wall clock,  over {:04d} runs): Run {: 9.4f} ({: 9.4f}, {: 9.4f}) Setup {: 9.4f} ({: 9.4f}, {: 9.4f}) Total {: 9.4f}",
                   "SrsRx",
                   slotIdx,
                   slotIdx,
                   m_elapsedTimes[0].size(),
                   avgElapsedTimesUs[ELAPSED_CPU_RUN],
                   minElapsedTimesUs[ELAPSED_CPU_RUN],
                   maxElapsedTimesUs[ELAPSED_CPU_RUN],
                   avgElapsedTimesUs[ELAPSED_CPU_SETUP],
                   minElapsedTimesUs[ELAPSED_CPU_SETUP],
                   maxElapsedTimesUs[ELAPSED_CPU_SETUP],
                   avgElapsedTimesUs[ELAPSED_CPU_RUN] + avgElapsedTimesUs[ELAPSED_CPU_SETUP]);

            // --------------------------------------------------------------------
            // cleanup

            cuphyStatus_t statusDestroy = cuphyDestroySrsRx(srsRxHndl);
            if(CUPHY_STATUS_SUCCESS != statusDestroy) throw cuphy::cuphy_exception(statusDestroy);
            
        } // End slot loop

    }
    catch(std::exception& e)
    {
        NVLOGE_FMT(NVLOG_SRS, AERIAL_CUPHY_EVENT,  "EXCEPTION: {}", e.what());
        returnValue = 1;
    }
    catch(...)
    {
        NVLOGE_FMT(NVLOG_SRS, AERIAL_CUPHY_EVENT,  "UNKNOWN EXCEPTION");
        returnValue = 2;
    }
    nvlog_fmtlog_close(log_thread_id);
    return returnValue;
}
