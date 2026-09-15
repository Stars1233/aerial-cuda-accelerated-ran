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

#include "cuphy_api.h"
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include "hdf5hpp.hpp"
#include "cuphy_hdf5.hpp"
#include "cuphy.hpp"
#include "datasets.hpp"
#include "test_config.hpp"
#include "nvlog.hpp"

//#define CUPHY_MEMTRACE // uncomment to exercise in cuPHY local run
#ifdef CUPHY_MEMTRACE
#include "memtrace.h"
#endif

#include <cstring>
#include <iostream>
#include <fstream>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <unistd.h> // for getcwd()
#include <dirent.h> // opendir, readdir
#include <errno.h>
#include <sys/stat.h> // for mkdir

using Clock     = std::chrono::high_resolution_clock;
using TimePoint = std::chrono::time_point<Clock>;
template <typename T, typename unit>
    using duration = std::chrono::duration<T, unit>;


////////////////////////////////////////////////////////////////////////
// usage()
void usage(char* argv[])
{
    printf("%s [options]\n", argv[0]);
    printf("  Options:\n");
    printf("    -i  input_filename         Input yaml file for slot/cell config or HDF5 file for single cell example\n");
    printf("    -l  log_filename           filename to save log output\n");
    printf("    -m  processing mode        PUCCH proc mode: streams (0x0), graphs (0x1)\n");
    printf("    -o  output_filename        Output HDFS debug file\n");
    printf("    -r  num                    Number of iterations to run\n");
    printf("    --G SM count               Use green contexts with specified SM count per context.\n");
    printf("    -s, --skip-polar           Alias for --processing-mode 1\n");
    printf("    --processing-mode N        PUCCH backend skip mode: 0=full, 1=skip polar, 2=skip backend (front-end only)\n");
    printf("    --frontend-ref-tol F       Per-UCI NRMSE tol for mode-2 LLR compare (default 0.01 = 1%% NRMSE)\n");
}

////////////////////////////////////////////////////////////////////////
// main()
int main(int argc, char* argv[])
{
    int returnValue = 0;
    char nvlog_yaml_file[1024];
    // Relative path from binary to default nvlog_config.yaml
    std::string relative_path = std::string("../../../../").append(NVLOG_DEFAULT_CONFIG_FILE);
    std::string log_name = "pucch.log";
    nv_get_absolute_path(nvlog_yaml_file, relative_path.c_str());
    pthread_t log_thread_id = -1;
    bool     useGreenCtxs       = false;
    uint32_t SMsPerGreenCtx     = 0;

    try
    {
        //------------------------------------------------------------------
        // Parse command line arguments
        int         iArg = 1;
        std::string inputFilename  = std::string();
        std::string outputFilename = std::string();
        int         processingMode = 0; // 0=full, 1=skip polar, 2=skip backend
        float       frontEndRefTol = 0.01f;
        uint64_t    procModeBmsk   = PUCCH_PROC_MODE_FULL_SLOT;
        int         totalIters     = 1;

        while(iArg < argc)
        {
            if('-' == argv[iArg][0])
            {
                // Handle long options ("--...") before falling through to single-char switches
                // so we can match by full name rather than a single character after '-'.
                if(argv[iArg][1] == '-')
                {
                    std::string longOpt(argv[iArg] + 2);
                    if(longOpt == "skip-polar")
                    {
                        processingMode = 1;
                        ++iArg;
                        continue;
                    }
                    if(longOpt == "processing-mode")
                    {
                        if(++iArg >= argc)
                        {
                            NVLOGE_FMT(NVLOG_PUCCH, AERIAL_CUPHY_EVENT, "ERROR: --processing-mode requires an argument (0, 1, or 2)");
                            exit(1);
                        }
                        const std::string token(argv[iArg]);
                        int parsedMode = 0;
                        {
                            size_t pos = 0;
                            try
                            {
                                parsedMode = std::stoi(token, &pos);
                            }
                            catch(const std::invalid_argument&)
                            {
                                NVLOGE_FMT(NVLOG_PUCCH, AERIAL_CUPHY_EVENT, "ERROR: --processing-mode must be an integer 0, 1, or 2 (got '{}')", token);
                                exit(1);
                            }
                            catch(const std::out_of_range&)
                            {
                                NVLOGE_FMT(NVLOG_PUCCH, AERIAL_CUPHY_EVENT, "ERROR: --processing-mode value out of range (got '{}')", token);
                                exit(1);
                            }
                            if(pos != token.size())
                            {
                                NVLOGE_FMT(NVLOG_PUCCH, AERIAL_CUPHY_EVENT, "ERROR: --processing-mode has trailing characters (got '{}')", token);
                                exit(1);
                            }
                        }
                        if(parsedMode < 0 || parsedMode > 2)
                        {
                            NVLOGE_FMT(NVLOG_PUCCH, AERIAL_CUPHY_EVENT, "ERROR: --processing-mode must be 0, 1, or 2 (got {})", parsedMode);
                            exit(1);
                        }
                        processingMode = parsedMode;
                        ++iArg;
                        continue;
                    }
                    if(longOpt == "frontend-ref-tol")
                    {
                        if(++iArg >= argc)
                        {
                            NVLOGE_FMT(NVLOG_PUCCH, AERIAL_CUPHY_EVENT, "ERROR: --frontend-ref-tol requires a float argument");
                            exit(1);
                        }
                        const std::string token(argv[iArg]);
                        float parsedTol = 0.0f;
                        {
                            size_t pos = 0;
                            try
                            {
                                parsedTol = std::stof(token, &pos);
                            }
                            catch(const std::invalid_argument&)
                            {
                                NVLOGE_FMT(NVLOG_PUCCH, AERIAL_CUPHY_EVENT,
                                           "ERROR: --frontend-ref-tol must be a non-negative finite float (got '{}')",
                                           token);
                                exit(1);
                            }
                            catch(const std::out_of_range&)
                            {
                                NVLOGE_FMT(NVLOG_PUCCH, AERIAL_CUPHY_EVENT,
                                           "ERROR: --frontend-ref-tol value out of range (got '{}')",
                                           token);
                                exit(1);
                            }
                            if(pos != token.size())
                            {
                                NVLOGE_FMT(NVLOG_PUCCH, AERIAL_CUPHY_EVENT,
                                           "ERROR: --frontend-ref-tol has trailing characters (got '{}')",
                                           token);
                                exit(1);
                            }
                        }
                        if(!std::isfinite(parsedTol) || (parsedTol < 0.0f))
                        {
                            NVLOGE_FMT(NVLOG_PUCCH, AERIAL_CUPHY_EVENT,
                                       "ERROR: --frontend-ref-tol must be a non-negative finite float (got {})",
                                       token);
                            exit(1);
                        }
                        frontEndRefTol = parsedTol;
                        ++iArg;
                        continue;
                    }
                    if(longOpt == "G")
                    {
                        useGreenCtxs = true;
                        if(((++iArg >= argc) || (1 != sscanf(argv[iArg], "%i", &SMsPerGreenCtx))) || (SMsPerGreenCtx == 0))
                        {
                            NVLOGE_FMT(NVLOG_PUCCH, AERIAL_CUPHY_EVENT, "ERROR: Invalid or missing useGreenCtxs argument (--G)");
                            exit(1);
                        }
                        ++iArg;
                        continue;
                    }
                    NVLOGE_FMT(NVLOG_PUCCH, AERIAL_CUPHY_EVENT, "ERROR: Unknown option: {}", argv[iArg]);
                    usage(argv);
                    exit(1);
                }

                switch(argv[iArg][1])
                {
                    case 'i':
                        if(++iArg >= argc)
                        {
                            throw std::invalid_argument("No valid filename provided");
                        }
                        inputFilename.assign(argv[iArg++]);
                        break;
                    case 'l':
                        if(++iArg < argc)
                        {
                            log_name.assign(argv[iArg++]);
                        }
                        break;
                    case 'o':
                        if(++iArg < argc)
                        {
                            outputFilename.assign(argv[iArg++]);
                        }
                        break;
                    case 'm':
                        if((++iArg >= argc) || (1 != sscanf(argv[iArg], "%lu", &procModeBmsk)) || ((procModeBmsk != PUCCH_PROC_MODE_FULL_SLOT) && (procModeBmsk != PUCCH_PROC_MODE_FULL_SLOT_GRAPHS)))
                        {
                            NVLOGE_FMT(NVLOG_PUCCH, AERIAL_CUPHY_EVENT,  "ERROR: Invalid processing mode ({})", procModeBmsk);
                            exit(1);
                        }
                        ++iArg;
                        break;
                    case 'r':
                        if(++iArg < argc)
                        {
                            totalIters = std::stoi(argv[iArg++]);
                        }
                        break;
                    case 's':
                        // short alias for --skip-polar (mode 1)
                        processingMode = 1;
                        ++iArg;
                        break;
                    default:
                        usage(argv);
                        throw std::invalid_argument(fmt::format("Unknown option: {}", argv[iArg]));
                }
            }
            else
            {
                throw std::invalid_argument(fmt::format("Invalid command line argument: {}", argv[iArg]));
            }
        }
        if(inputFilename.empty())
        {
            usage(argv);
            throw std::invalid_argument("No valid filename provided");
        }
        log_thread_id = nvlog_fmtlog_init(nvlog_yaml_file, log_name.c_str(),NULL);
        nvlog_fmtlog_thread_init();
        if(procModeBmsk == PUCCH_PROC_MODE_FULL_SLOT_GRAPHS)
        {
            NVLOGI_FMT(NVLOG_PUCCH, "CUDA graph enabled!");
        } else {
            NVLOGI_FMT(NVLOG_PUCCH, "CUDA stream mode");
        }

        int gpuId = 0; // select GPU device 0
        CUDA_CHECK(cudaSetDevice(gpuId));
        CUdevice current_device;
        CU_CHECK(cuDeviceGet(&current_device, gpuId));

#if CUDA_VERSION >= 12040
        CUdevResource initial_device_GPU_resources = {};
        CUdevResourceType default_resource_type = CU_DEV_RESOURCE_TYPE_SM; // other alternative is CU_DEV_RESOURCE_TYPE_INVALID
        CUdevResource split_result[2] = {{}, {}};
        cuphy::cudaGreenContext pucch_green_ctx;
        unsigned int split_groups = 1;

        if(useGreenCtxs)
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
            pucch_green_ctx.create(gpuId, &split_result[0]);
            pucch_green_ctx.bind();
            NVLOGC_FMT(NVLOG_PUCCH, "PUCCH green context will have access to {} SMs ({} SMs requested).", pucch_green_ctx.getSmCount(), SMsPerGreenCtx);
        }
#endif

        // Initialize debug instrumentation
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
        // input files
        std::vector<std::string> inputFileNameVec;
        std::string              inFileExtn = inputFilename.substr(inputFilename.find_last_of(".") + 1);
        if(inFileExtn == "yaml")
        {
            // yaml parsing
            cuphy::test_config testCfg(inputFilename.c_str());
            int                nCells           = testCfg.num_cells();
            int                nSlots           = testCfg.num_slots();
            const std::string  pucchChannelName = "PUCCH";

            try
            {
                for(size_t idxSlot = 0; idxSlot < nSlots; idxSlot++)
                {
                    for(int idxCell = 0; idxCell < nCells; idxCell++)
                    {
                        auto fname = testCfg.slots()[idxSlot].at(pucchChannelName)[idxCell];
                        inputFileNameVec.emplace_back(fname);
                    }
                }
            }
            catch(...)
            {
                throw std::runtime_error("PUCCH channel name not found in the input file");
            }
            assert(inputFileNameVec.size() == nCells);
        }
        else
        {
            inputFileNameVec.emplace_back(inputFilename);
        }

        //------------------------------------------------------------------
        // Load API parameters

        cuphy::stream cuStrmMain;
        cudaStream_t  cuStrm           = cuStrmMain.handle();

        pucchStaticApiDataset  statPucchApiDataset(inputFileNameVec, cuStrm, outputFilename);
        pucchDynApiDataset     dynPucchApiDataset (inputFileNameVec, cuStrm, procModeBmsk);
        EvalPucchDataset       evalPucchDataset   (inputFileNameVec, cuStrm);
        cuStrmMain.synchronize(); // synch to ensure data copied

        cuphyPucchDynPrms_t&  pucchDynPrm   = dynPucchApiDataset.pucchDynPrm;
        cuphyPucchStatPrms_t& pucchStatPrms =  statPucchApiDataset.pucchStatPrms;

        // Honor --processing-mode by flipping the static param before PucchRx creation.
        // Mode 1 wires post-polar descriptors from the dataset layer; mode 2 keeps
        // front-end reference checking in the example/dataset layer.
        const char* refH5Path = statPucchApiDataset.refH5Path.empty() ? nullptr
                                                                      : statPucchApiDataset.refH5Path.c_str();
        pucchStatPrms.pipelineData.pPostPolarData = nullptr;
        if(processingMode == 1)
        {
            pucchStatPrms.pipelineMode = PUCCH_PIPELINE_SKIP_POLAR;
            pucchStatPrms.pipelineData.pPostPolarData = (refH5Path != nullptr && refH5Path[0] != '\0')
                                                        ? dynPucchApiDataset.enablePostPolarData(refH5Path, cuStrm)
                                                        : nullptr;
            NVLOGI_FMT(NVLOG_PUCCH, "PUCCH pipelineMode = SKIP_POLAR via mode 1; post-polar refs sourced from input TV.");
        }
        else if(processingMode == 2)
        {
            pucchStatPrms.pipelineMode = PUCCH_PIPELINE_SKIP_BACKEND;
            if(refH5Path != nullptr)
            {
                dynPucchApiDataset.enableFrontEndLlrOutput(refH5Path, cuStrm);
            }
            NVLOGI_FMT(NVLOG_PUCCH, "PUCCH pipelineMode = SKIP_BACKEND via mode 2; front-end LLR compare vs TV (NRMSE tol={}).", frontEndRefTol);
        }
        else
        {
            // Explicitly force RUN so --processing-mode 0 overrides any prior value the
            // TV/YAML dataset constructor may have set (rather than silently inheriting it).
            pucchStatPrms.pipelineMode = PUCCH_PIPELINE_FULL;
            NVLOGI_FMT(NVLOG_PUCCH, "PUCCH pipelineMode = FULL (mode 0).");
        }

        //------------------------------------------------------------------
        // allocate output buffers

        size_t MAX_N_F234_UCI  = CUPHY_PUCCH_F2_MAX_UCI + CUPHY_PUCCH_F3_MAX_UCI;
        
        //------------------------------------------------------------------
        // Finish setting dynamic parameters

        dynPucchApiDataset.pucchDynPrm.cuStream                       = cuStrm; // save stream in dynamic parameters
        dynPucchApiDataset.pucchDynPrm.cpuCopyOn                      = 1;      // option to copy uci output to CPU immediately after run
        //------------------------------------------------------------------
        // Create pucch reciever object

        cuphyPucchRxHndl_t pucchRxHndl;
        
        cuphyStatus_t statusCreate = cuphyCreatePucchRx(&pucchRxHndl, &pucchStatPrms, cuStrm);
        if(CUPHY_STATUS_SUCCESS != statusCreate) throw cuphy::cuphy_exception(statusCreate);
#ifdef CUPHY_MEMTRACE
        memtrace_set_config(MI_MEMTRACE_CONFIG_ENABLE | MI_MEMTRACE_CONFIG_EXIT_AFTER_BACKTRACE);
#endif

        for(int iterIdx=0; iterIdx<totalIters;iterIdx++)
        {
            
            auto& elapsedTimeUsSetup      = m_elapsedTimes[ELAPSED_CPU_SETUP][iterIdx];
            auto& elapsedTimeUsRun        = m_elapsedTimes[ELAPSED_CPU_RUN][iterIdx];
            auto& elapsedEvtTimeUsSetup   = m_elapsedTimes[ELAPSED_EVT_SETUP][iterIdx];
            auto& elapsedEvtTimeUsRun     = m_elapsedTimes[ELAPSED_EVT_RUN][iterIdx];

            //------------------------------------------------------------------
            // Setup pucch reciever object

            cuphyPucchBatchPrmHndl_t const batchPrmHndl = nullptr;  // batchPrms currently un-used
#ifdef CUPHY_MEMTRACE
            memtrace_set_config(MI_MEMTRACE_CONFIG_ENABLE | MI_MEMTRACE_CONFIG_EXIT_AFTER_BACKTRACE);
#endif

            // Record GPU & CPU time before setup
            evtTmrSetup.record_begin(cuStrm);
            timePtStartSetup = Clock::now();
            cuphyStatus_t statusSetup  = cuphySetupPucchRx(pucchRxHndl, &pucchDynPrm, batchPrmHndl);
            // Record GPU & CPU time after setup
            timePtStopSetup = Clock::now();
            evtTmrSetup.record_end(cuStrm);

            if(CUPHY_STATUS_SUCCESS != statusSetup) throw cuphy::cuphy_exception(statusSetup);

            //------------------------------------------------------------------
            // Run pucch reciever object

            uint64_t procModeBmsk = 0; // procModeBmsk currently un-used

            // Record GPU & CPU time before run
            evtTmrRun.record_begin(cuStrm);
            timePtStartRun = Clock::now();
            cuphyStatus_t statusRun = cuphyRunPucchRx(pucchRxHndl, procModeBmsk);
            // Record GPU & CPU time after run
            timePtStopRun = Clock::now();
            evtTmrRun.record_end(cuStrm);
            
            if(CUPHY_STATUS_SUCCESS != statusRun) throw cuphy::cuphy_exception(statusRun);

            //------------------------------------------------------------------

            // Process timing data
            evtTmrSetup.synchronize();
            evtTmrRun.synchronize();
            cuStrmMain.synchronize();

            // Compare cuphy UCI output to reference
#ifdef CUPHY_MEMTRACE
            memtrace_set_config(0); // Disable for evalPucchRxPipeline() call; not in the critical path
#endif
            if (iterIdx == 0)
            {
                if(processingMode == 2)
                {
                    // Mode 2 disables the UCI seg parser; final UCI bits are garbage. Verify
                    // by NRMSE-comparing the front-end LLR buffers against the TV reference.
                    const char* refPath = refH5Path;
                    if(refPath == nullptr)
                    {
                        // Mode 2's only verification is the LLR compare. If the ref path is null,
                        // fail visibly rather than reporting a false-green PASS for an unverified run.
                        NVLOGE_FMT(NVLOG_PUCCH, AERIAL_CUPHY_EVENT, "Mode 2 requested but input TV ref path is null; cannot verify");
                        returnValue = 4;
                    }
                    else
                    {
                        cuphyStatus_t cmpStatus = comparePucchFrontEndRefForBackendSkip(pucchDynPrm.pDataOut, refPath, frontEndRefTol, cuStrm);
                        if(cmpStatus != CUPHY_STATUS_SUCCESS)
                        {
                            NVLOGE_FMT(NVLOG_PUCCH, AERIAL_CUPHY_EVENT, "Mode 2 front-end LLR compare FAILED (status={})", static_cast<int>(cmpStatus));
                            returnValue = 3;
                        }
                        else
                        {
                            NVLOGC_FMT(NVLOG_PUCCH, "Mode 2 front-end LLR compare PASSED");
                        }
                    }
                }
                else
                {
                    evalPucchDataset.evalPucchRxPipeline(pucchDynPrm);
                }
            }

            elpasedTimeDurationUs = timePtStopSetup - timePtStartSetup;
            elapsedTimeUsSetup    = elpasedTimeDurationUs.count();
            elpasedTimeDurationUs = timePtStopRun - timePtStartRun;
            elapsedTimeUsRun      = elpasedTimeDurationUs.count();
            elapsedEvtTimeUsSetup = evtTmrSetup.elapsed_time_ms()*1000;
            elapsedEvtTimeUsRun   = evtTmrRun.elapsed_time_ms()*1000;
        }
#ifdef CUPHY_MEMTRACE
        memtrace_set_config(0); // disable memory allocation tracing beyond this point
#endif
        //------------------------------------------------------------------
        // Save debug output

        if(!outputFilename.empty())
        {
            cuphyStatus_t statusDebugWrite = cuphyWriteDbgBufSynchPucch(pucchRxHndl, cuStrm);
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

        NVLOGC_FMT(NVLOG_PUCCH,"Timing results {}, format: avg (min, max) ",
            procModeBmsk == PUCCH_PROC_MODE_FULL_SLOT ? "in stream mode" : "in graph mode");
        
        NVLOGC_FMT(NVLOG_PUCCH,"{} Pipeline[{:02d}]: Metric - GPU Time usec (using CUDA events, over {:04d} runs): Run {: 9.4f} ({: 9.4f}, {: 9.4f}) Setup {: 9.4f} ({: 9.4f}, {: 9.4f}) Total {: 9.4f}",
               "PucchRx",
               0,
               m_elapsedTimes[0].size(),
               avgElapsedTimesUs[ELAPSED_EVT_RUN],
               minElapsedTimesUs[ELAPSED_EVT_RUN],
               maxElapsedTimesUs[ELAPSED_EVT_RUN],
               avgElapsedTimesUs[ELAPSED_EVT_SETUP],
               minElapsedTimesUs[ELAPSED_EVT_SETUP],
               maxElapsedTimesUs[ELAPSED_EVT_SETUP],
               avgElapsedTimesUs[ELAPSED_EVT_RUN] + avgElapsedTimesUs[ELAPSED_EVT_SETUP]);

        NVLOGC_FMT(NVLOG_PUCCH,"{} Pipeline[{:02d}]: Metric - CPU Time usec (using wall clock,  over {:04d} runs): Run {: 9.4f} ({: 9.4f}, {: 9.4f}) Setup {: 9.4f} ({: 9.4f}, {: 9.4f}) Total {: 9.4f}",
               "PucchRx",
               0,
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

        cuphyStatus_t statusDestroy = cuphyDestroyPucchRx(pucchRxHndl);
        if(CUPHY_STATUS_SUCCESS != statusDestroy) throw cuphy::cuphy_exception(statusDestroy);

    }
    catch(std::exception& e)
    {
        if(log_thread_id < 0)
        {
            log_thread_id = nvlog_fmtlog_init(nvlog_yaml_file, log_name.c_str(),NULL);
            nvlog_fmtlog_thread_init();
        }
        NVLOGE_FMT(NVLOG_PUCCH, AERIAL_CUPHY_EVENT, "EXCEPTION: {}", e.what());
        returnValue = 1;
    }
    catch(...)
    {
        if(log_thread_id < 0)
        {
            log_thread_id = nvlog_fmtlog_init(nvlog_yaml_file, log_name.c_str(),NULL);
            nvlog_fmtlog_thread_init();
        }
        NVLOGE_FMT(NVLOG_PUCCH, AERIAL_CUPHY_EVENT, "UNKNOWN EXCEPTION");
        returnValue = 2;
    }
    nvlog_fmtlog_close(log_thread_id);
    return returnValue;
}
