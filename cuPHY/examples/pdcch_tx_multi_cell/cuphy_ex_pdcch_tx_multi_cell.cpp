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

#include "util.hpp"
#include "pdcch_tx.hpp"
#include "cuphy_channels.hpp"
#include "datasets.hpp"
#include <list>
#include <fstream>
#include "test_config.hpp"

#define PDCCH_STREAM_PRIORITY 0
#define PRINT_PDCCH_CONFIG 0 // Set to 1 for debugging purposes
#define REV_DCI_ORDER 0 // Set to 1 to test different DCI order layout

using namespace cuphy;
using Clock     = std::chrono::high_resolution_clock;
using TimePoint = std::chrono::time_point<Clock>;
template <typename T, typename unit>
    using duration = std::chrono::duration<T, unit>;

/**
 *  @brief Print usage information for the DL pipeline example.
 */
void usage()
{
    printf("  Options:\n");
    printf("    -h                          Display usage information\n");
    printf("    -i  input_filename          Input yaml file for slot/cell config \n");
    printf("    -r  # of iterations         Number of run iterations to run\n");
    printf("    -d  # of microseconds       Delay kernel duration in us\n");
    printf("    -k                          Enable reference check. Compare GPU output with test vector (first iteration only).\n");
    printf("    -m  proc_mode               Processing mode: streams(0), graphs (1)\n");
    printf("    -g                          Execute all cells in a slot on the same PDCCH object with batching too.\n");
    printf("    -s  setup_mode              0 (default) - setup is not timed; 1 - time setup only; no run is run; 2 - time both setup and run, back to back.\n");
    printf("    -F  GPU kernel sel          Selection GPU kernels for the PDCCH pipline.\n");
    printf("    -X  GPU delay duration      Set the delay duration in us for the PDCCH pipline (default: 0).\n");
    printf("    --G SM count                Use green contexts with specified SM count per context.\n");
}

int main(int argc, char* argv[])
{
    int returnValue = 0;

    char nvlog_yaml_file[1024];
    // Relative path from binary to default nvlog_config.yaml
    std::string relative_path = std::string("../../../../").append(NVLOG_DEFAULT_CONFIG_FILE);
    nv_get_absolute_path(nvlog_yaml_file, relative_path.c_str());
    pthread_t log_thread_id = nvlog_fmtlog_init(nvlog_yaml_file, "pdcch_tx_multicell.log",NULL);
    nvlog_fmtlog_thread_init();

    //------------------------------------------------------------------
    // Parse command line arguments
    int         iArg = 1;
    std::string inputFileName;
    uint32_t    num_iterations   = 1;
    bool        ref_check_pdcch  = false;
    uint64_t    procModeBmsk     = 0;
    uint32_t    delayUs          = 10000;
    bool        group_cells      = false; // Group cells and if possible, also process all cells in a single kernel per component.
    int         time_setup_mode  = 0;     // default mode: only time run, not setup
    std::string setup_modes[3]   = {"GPU-run only", "GPU-setup only", "GPU-setup-and-run"};
    bool     useGreenCtxs        = false;
    uint32_t SMsPerGreenCtx      = 0;
    uint8_t  kernelSelMode       = PDCCH_ALL;
    uint32_t delayUsInPipeline = 0;

    while(iArg < argc)
    {
        if('-' == argv[iArg][0])
        {
            switch(argv[iArg][1])
            {
            case 'h':
                usage();
                exit(0);
                break;
            case 'i':
                if(++iArg >= argc)
                {
                    NVLOGE_FMT(NVLOG_PDCCH, AERIAL_CUPHY_EVENT,  "ERROR: No filename provided.");
                }
                inputFileName.assign(argv[iArg++]);
                break;
            case 'r':
                if((++iArg >= argc) || (1 != sscanf(argv[iArg], "%i", &num_iterations)) || ((num_iterations <= 0)))
                {
                    NVLOGF_FMT(NVLOG_PDCCH, AERIAL_CUPHY_EVENT,  "ERROR: Invalid number of iterations");
                }
                ++iArg;
                break;
            case 'd':
                    delayUs = std::stoi(argv[++iArg]);
                    ++iArg;
                    break;
            case 'g':
                group_cells = true;
                ++iArg;
                break;
            case 'k':
                ref_check_pdcch = true;
                ++iArg;
                break;
            case 'm':
                if((++iArg >= argc) || (1 != sscanf(argv[iArg], "%lu", &procModeBmsk)) || ((PDCCH_PROC_MODE_STREAMS != procModeBmsk) && (PDCCH_PROC_MODE_GRAPHS != procModeBmsk)))
                {
                    NVLOGF_FMT(NVLOG_PDCCH, AERIAL_CUPHY_EVENT,  "ERROR: Invalid process mode");
                }
                ++iArg;
                break;
            case 's':
                if((++iArg >= argc) || (1 != sscanf(argv[iArg], "%i", &time_setup_mode)) || ((time_setup_mode < 0)) || ((time_setup_mode > 2)))
                {
                    NVLOGF_FMT(NVLOG_PDCCH, AERIAL_CUPHY_EVENT,  "ERROR: Invalid process mode");
                }
                ++iArg;
                break;
            case 'F':
                if((++iArg >= argc) || (1 != sscanf(argv[iArg], "%hhu", &kernelSelMode)) || (kernelSelMode >= PDCCH_MAX_KERNEL_SEL_MODES))
                {
                    NVLOGE_FMT(NVLOG_PDCCH, AERIAL_CUPHY_EVENT,  "ERROR: unsupported GPU kernel selection mode");
                    exit(1);
                }
                ++iArg;
                break;
            case 'X':
                if((++iArg >= argc) || (1 != sscanf(argv[iArg], "%u", &delayUsInPipeline)) || delayUsInPipeline > 5000)
                {
                    NVLOGE_FMT(NVLOG_PDCCH, AERIAL_CUPHY_EVENT,  "ERROR: too large delay duration");
                    exit(1);
                }
                ++iArg;
                break;
            case '-':
                 switch(argv[iArg][2])
                 {
                    case 'G':
                        useGreenCtxs = true;
                        if(((++iArg >= argc) || (1 != sscanf(argv[iArg], "%i", &SMsPerGreenCtx))) || (SMsPerGreenCtx == 0))
                        {
                            // Will later check that SMsPerGreenCtx does not exceed the SMs of the GPU. This will also capture if a negative number was provided.
                            NVLOGE_FMT(NVLOG_PDCCH, AERIAL_CUPHY_EVENT,  "ERROR: Invalid or missing useGreenCtxs argument (--G)");
                            exit(1);
                        }
                        ++iArg;
                        break;
                    default:
                        NVLOGE_FMT(NVLOG_PDCCH, AERIAL_CUPHY_EVENT,  "ERROR: Unknown option: {}", argv[iArg]);
                        usage();
                        exit(1);
                        break;
                 }
                 break;
            default:
                NVLOGE_FMT(NVLOG_PDCCH, AERIAL_CUPHY_EVENT,  "ERROR: Unknown option: {}", argv[iArg]);
                usage();
                exit(1);
                break;
            }
        }
        else
        {
            NVLOGF_FMT(NVLOG_PDCCH, AERIAL_CUPHY_EVENT,  "ERROR: Invalid command line argument: {}", argv[iArg]);
        }
    }
    if(inputFileName.empty())
    {
        usage();
        exit(1);
    }

    int gpuId = 0; // select GPU device 0
    CUDA_CHECK(cudaSetDevice(gpuId));
    CUdevice current_device;
    CU_CHECK(cuDeviceGet(&current_device, gpuId));

#if CUDA_VERSION >= 12040
    CUdevResource initial_device_GPU_resources = {};
    CUdevResourceType default_resource_type = CU_DEV_RESOURCE_TYPE_SM; // other alternative is CU_DEV_RESOURCE_TYPE_INVALID
    CUdevResource split_result[2] = {{}, {}};
    cuphy::cudaGreenContext pdcch_green_ctx;
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
        pdcch_green_ctx.create(gpuId, &split_result[0]);
        pdcch_green_ctx.bind();
        NVLOGC_FMT(NVLOG_PDCCH, "PDCCH green context will have access to {} SMs ({} SMs requested).", pdcch_green_ctx.getSmCount(), SMsPerGreenCtx);
    }
#endif


    cuphy::test_config testCfg(inputFileName.c_str());
    const std:: string channelName = "PDCCH";
    testCfg.print_channel(channelName); // print only PDCCH channel related parts of the input YAML file
    //testCfg.print(); // print entire YAML file contents, incl. other channels not run with this example
    int num_cells = testCfg.num_cells(); // The same number of cells is present across all slots.
    int num_slots = testCfg.num_slots();

    //NVLOGC_FMT(NVLOG_PDCCH, "PDCCH multi-cell with {} cells and {} slots",  num_cells, num_slots);

    std::vector<std::vector<cuphy::pdcch_tx>>       m_pdcchTxPipes;
    std::vector<std::vector<pdcchStaticApiDataset>> m_pdcchTxStaticApiDataSets;
    std::vector<std::vector<pdcchDynApiDataset>>    m_pdcchTxDynamicApiDataSets;

    std::vector<stream>        streams;
    cudaEvent_t                start_streams_event;
    std::vector<cudaEvent_t> stop_streams_events(num_cells);
    const int num_pdcch_objects = group_cells ? 1 : num_cells;

    m_pdcchTxPipes.resize(num_slots);
    m_pdcchTxStaticApiDataSets.resize(num_slots);
    m_pdcchTxDynamicApiDataSets.resize(num_slots);

    for(int idxSlot = 0; idxSlot < num_slots; idxSlot++) {
        int cells_in_slot = testCfg.slots()[idxSlot].at(channelName).size();
        if (cells_in_slot != num_cells) {
            NVLOGF_FMT(NVLOG_PDCCH, AERIAL_CUPHY_EVENT, "Slot {} error: expected {} cells but got {}", idxSlot, num_cells, cells_in_slot);
        }
        // Reminder num_pdcch_object is the number of cells, if group_cells is false, or 1 otherwise
        m_pdcchTxStaticApiDataSets[idxSlot].reserve(num_pdcch_objects);
        m_pdcchTxDynamicApiDataSets[idxSlot].reserve(num_pdcch_objects);

        // Loop over all cells (not num_pdcch_objects) to populate the static and dynamic datasets
        for(int i = 0; i < num_cells; i += 1)
        {
            std::string tv_filename = testCfg.slots()[idxSlot].at(channelName)[i];
            if((idxSlot == 0) && (i < num_pdcch_objects)) {
                streams.emplace_back(cudaStreamNonBlocking, PDCCH_STREAM_PRIORITY);
            }
            if (group_cells) {
                if (i == 0) {
                    m_pdcchTxStaticApiDataSets[idxSlot].emplace_back(num_cells);
                    m_pdcchTxStaticApiDataSets[idxSlot][i].pdcchStatPrms.kernelSelOption = kernelSelMode;
                    m_pdcchTxStaticApiDataSets[idxSlot][i].pdcchStatPrms.delayUs = delayUsInPipeline;
                    m_pdcchTxDynamicApiDataSets[idxSlot].emplace_back(tv_filename, m_pdcchTxStaticApiDataSets[idxSlot][i].pdcchStatPrms.nMaxCellsPerSlot, streams[i].handle(), procModeBmsk);
                    m_pdcchTxPipes[idxSlot].emplace_back(m_pdcchTxStaticApiDataSets[idxSlot][i].pdcchStatPrms);
                } else {
                    // Nothing to cumulative update for the static parameters.
                    // Update the dyanmic parameters
                    m_pdcchTxDynamicApiDataSets[idxSlot][0].cumulativeUpdate(tv_filename, streams[0].handle());
                }
            } else {
                m_pdcchTxStaticApiDataSets[idxSlot].emplace_back();
                m_pdcchTxStaticApiDataSets[idxSlot][i].pdcchStatPrms.kernelSelOption = kernelSelMode;
                m_pdcchTxStaticApiDataSets[idxSlot][i].pdcchStatPrms.delayUs = delayUsInPipeline;
                m_pdcchTxDynamicApiDataSets[idxSlot].emplace_back(tv_filename, m_pdcchTxStaticApiDataSets[idxSlot][i].pdcchStatPrms.nMaxCellsPerSlot, streams[i].handle(), procModeBmsk);
                m_pdcchTxPipes[idxSlot].emplace_back(m_pdcchTxStaticApiDataSets[idxSlot][i].pdcchStatPrms);
            }
        }
    }

#if PRINT_PDCCH_CONFIG
    // Print dynamic dataset params for each PDCCH object. For dbg. purposes
    for(int idxSlot = 0; idxSlot < num_slots; idxSlot++) {
        for(int i = 0; i < num_pdcch_objects; i += 1) {
            m_pdcchTxPipes[idxSlot][i].handle()->pipeline->printPdcchConfig(m_pdcchTxDynamicApiDataSets[idxSlot][i].pdcch_dyn_params);
        }
    }
#endif

#if REV_DCI_ORDER
    // rev. order and print dynamic dataset params for each PDCCH object. For dbg. purposes
    for(int idxSlot = 0; idxSlot < num_slots; idxSlot++) {
        for(int i = 0; i < num_pdcch_objects; i += 1) {
            m_pdcchTxDynamicApiDataSets[idxSlot][i].printPayload();
            m_pdcchTxDynamicApiDataSets[idxSlot][i].revDciOrder();
            m_pdcchTxDynamicApiDataSets[idxSlot][i].printPayload();
            //m_pdcchTxPipes[idxSlot][i].handle()->pipeline->printPdcchConfig(m_pdcchTxDynamicApiDataSets[idxSlot][i].pdcch_dyn_params);
        }
    }
#endif

    NVLOGC_FMT(NVLOG_PDCCH, "");
    NVLOGC_FMT(NVLOG_PDCCH, "Timing {} PDCCH DL pipeline(s). ", num_cells);
    if (group_cells)
    {
        NVLOGC_FMT(NVLOG_PDCCH, "Grouping all cells in a slot.");
    } else {
        NVLOGC_FMT(NVLOG_PDCCH, "");
    }
    if (time_setup_mode== 0) {
        NVLOGC_FMT(NVLOG_PDCCH, "- NB: Allocations, setup processing not included.");
        NVLOGC_FMT(NVLOG_PDCCH, "");
    }

    /* PdcchTx::Run for all num_cells pipelines will be timed on streams[0].handle() CUDA stream (note, that is NOT stream 0).
    Have that stream wait for all other streams to complete their work too. */

    TimePoint timePtStartSetup, timePtStopSetup;
    TimePoint timePtStartRun, timePtStopRun;
    duration<float, std::micro> elapsedTimeDurationUs;
    typedef enum _elapsedTypes
    {
        ELAPSED_CPU_SETUP = 0,
        ELAPSED_CPU_RUN   = 1,
        ELAPSED_TYPES_MAX
    } elapsedTypes_t;
    std::array<std::vector<float>,ELAPSED_TYPES_MAX> m_elapsedTimes;
    for(auto& timeVec : m_elapsedTimes)
    {
        timeVec.resize(num_iterations, 0.0f);
    }

    for(int i = 0; i < num_pdcch_objects; i++)
    {
        CUDA_CHECK(cudaEventCreateWithFlags(&stop_streams_events[i], cudaEventDisableTiming));
    }
    CUDA_CHECK(cudaEventCreateWithFlags(&start_streams_event, cudaEventDisableTiming));

    float total_time_slot[num_slots]; // Total time of a slot divided by number of iterations (num_iterations)
    float total_time_single_cell_slot[num_slots][num_pdcch_objects];
    //NVLOGC_FMT(NVLOG_PDCCH, "num slots {}, num_pdcch_objects {}", num_slots, num_pdcch_objects);

    for(int idxSlot = 0; idxSlot < num_slots; idxSlot++)
    {
        float                    total_time = 0;
        std::vector<float>       total_time_single_cell(num_pdcch_objects, 0);
        std::vector<event_timer> cuphy_timer_single_cell(num_pdcch_objects);

        if (time_setup_mode == 0){ // In this mode, setup is not timed
            for(int i = 0; i < num_pdcch_objects; i++)
            {
                m_pdcchTxPipes[idxSlot][i].setup(m_pdcchTxDynamicApiDataSets[idxSlot][i].pdcch_dyn_params);
            }
        }

        // Need to reset in case of multiple slots, as I sum up over number of objects
        for(auto& timeVec : m_elapsedTimes)
        {
            std::fill(timeVec.begin(), timeVec.end(), 0.0f);
        }

        gpu_us_delay(delayUs, 0, streams[0].handle());
        CUDA_CHECK(cudaEventRecord(start_streams_event, streams[0].handle()));

        for(int iter = 0; iter < num_iterations; iter++)
        {
            auto& elapsedTimeUsSetup      = m_elapsedTimes[ELAPSED_CPU_SETUP][iter];
            auto& elapsedTimeUsRun        = m_elapsedTimes[ELAPSED_CPU_RUN][iter];

            event_timer cuphy_timer;
            cuphy_timer.record_begin(streams[0].handle());

            for(int i = 0; i < num_pdcch_objects; i++)
            {
                cudaStream_t strm_handle = streams[i].handle();

                if(i != 0)
                {
                    CUDA_CHECK(cudaStreamWaitEvent(streams[i].handle(), start_streams_event, 0));
                }

                cuphy_timer_single_cell[i].record_begin(streams[i].handle());
                if (time_setup_mode != 0) { // If time_setup_mode is 1 or 2, it is timed
                    timePtStartSetup = Clock::now();
                    m_pdcchTxPipes[idxSlot][i].setup(m_pdcchTxDynamicApiDataSets[idxSlot][i].pdcch_dyn_params);
                    timePtStopSetup = Clock::now();
                }
                if (time_setup_mode != 1)
                {
                    timePtStartRun = Clock::now();
                    m_pdcchTxPipes[idxSlot][i].run(0 /*pdcch_proc_mode*/);
                    timePtStopRun = Clock::now();
                }
                cuphy_timer_single_cell[i].record_end(streams[i].handle());

                /* Record a stop event on all streams but the streams[0].handle() stream. */
                /* Have streams[0].handle() stream wait for all other streams to complete their work before stopping the timer. */
                if(i != 0)
                {
                    CUDA_CHECK(cudaEventRecord(stop_streams_events[i], strm_handle));
                    CUDA_CHECK(cudaStreamWaitEvent(streams[0].handle(), stop_streams_events[i], 0));
                }
                elapsedTimeDurationUs = timePtStopSetup - timePtStartSetup;
                elapsedTimeUsSetup    += elapsedTimeDurationUs.count(); // accumulate for all objects (e.g., if running without -g on a single thread)
                elapsedTimeDurationUs = timePtStopRun - timePtStartRun;
                elapsedTimeUsRun      += elapsedTimeDurationUs.count(); // accumulate fo all objects
            }

            cuphy_timer.record_end(streams[0].handle());
            cuphy_timer.synchronize();
            total_time += cuphy_timer.elapsed_time_ms();

            for(int i = 0; i < num_pdcch_objects; i++)
            {
                cuphy_timer_single_cell[i].synchronize(); // To be safe
                total_time_single_cell[i] += cuphy_timer_single_cell[i].elapsed_time_ms();
            }

            if (ref_check_pdcch && (iter == 0)) {
               for(int i = 0; i < num_pdcch_objects; i++)
               {
                  int errors = m_pdcchTxDynamicApiDataSets[idxSlot][i].refCheck(true);
                  if (errors != 0) {
                      exit(1);
                  }
               }
            }
            gpu_us_delay(delayUs, 0, streams[0].handle()); // 10ms delay kernel. Can update/comment out.
            CUDA_CHECK(cudaEventRecord(start_streams_event, streams[0].handle()));
        }
        total_time_slot[idxSlot] = total_time / num_iterations;

        for(int i = 0; i < num_pdcch_objects; i++)
        {
            total_time_single_cell_slot[idxSlot][i] = total_time_single_cell[i] / num_iterations;
        }

        float avgElapsedTimesUs[ELAPSED_TYPES_MAX];
        float minElapsedTimesUs[ELAPSED_TYPES_MAX];
        float maxElapsedTimesUs[ELAPSED_TYPES_MAX];
        for(int i = 0; i < ELAPSED_TYPES_MAX; i++)
        {
            const auto minmax_pair = std::minmax_element(std::begin(m_elapsedTimes[i]), std::end(m_elapsedTimes[i]));
            float mean = std::accumulate(std::begin(m_elapsedTimes[i]), std::end(m_elapsedTimes[i]), 0.0)/m_elapsedTimes[i].size();
            avgElapsedTimesUs[i] = mean;
            minElapsedTimesUs[i] = *minmax_pair.first;
            maxElapsedTimesUs[i] = *minmax_pair.second;
        }

        NVLOGC_FMT(NVLOG_PDCCH, "Slot # {}, PDCCH pipeline(s) CPU Time: {:.2f} us total (summed over {} PDCCH objects, avg. over {} iterations)",
                   idxSlot, avgElapsedTimesUs[ELAPSED_CPU_RUN] + avgElapsedTimesUs[ELAPSED_CPU_SETUP], num_pdcch_objects, num_iterations);

        NVLOGC_FMT(NVLOG_PDCCH, "--> Slot # {}: CPU-Run {:.2f} us (min {:.2f} us, max {:.2f} us), CPU-Setup {:.2f} us (min {:.2f} us, max {:.2f} us)", idxSlot,
                   avgElapsedTimesUs[ELAPSED_CPU_RUN], minElapsedTimesUs[ELAPSED_CPU_RUN], maxElapsedTimesUs[ELAPSED_CPU_RUN],
                   avgElapsedTimesUs[ELAPSED_CPU_SETUP], minElapsedTimesUs[ELAPSED_CPU_SETUP], maxElapsedTimesUs[ELAPSED_CPU_SETUP]);

    } // end of slots

    for(int idxSlot = 0; idxSlot < num_slots; idxSlot++)
    {
        NVLOGC_FMT(NVLOG_PDCCH, "Slot # {}, PDCCH pipeline(s) {}: {:.2f} us (avg. over {} iterations) in {} mode.", idxSlot, setup_modes[time_setup_mode].c_str(), total_time_slot[idxSlot] * 1000, num_iterations, (procModeBmsk == 0) ? "Stream" : "Graphs");
        for(int i = 0; i < num_pdcch_objects; i++)
        {
            if (group_cells) {
                NVLOGC_FMT(NVLOG_PDCCH, "--> PDCCH object # {} with {} cells: {:.2f} us (avg over {} iterations)", i, num_cells, total_time_single_cell_slot[idxSlot][i] * 1000, num_iterations);
            } else {
                NVLOGC_FMT(NVLOG_PDCCH, "--> Cell # {} : {:.2f} us (avg over {} iterations)", i, total_time_single_cell_slot[idxSlot][i] * 1000, num_iterations);
            }
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    nvlog_fmtlog_close(log_thread_id);

    return 0;
}
