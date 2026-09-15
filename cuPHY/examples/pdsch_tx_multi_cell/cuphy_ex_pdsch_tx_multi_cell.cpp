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
#include "pdsch_tx.hpp"
#include "cuphy_channels.hpp"
#include "datasets.hpp"
#include <optional>
#include <list>
#include <fstream>
#include "test_config.hpp"

#define _READ_TB_CRC_ 0 // Can set to 1, if needed. Not passed through cli right now. Shall we extend it?
#define PRINT_PDSCH_CONFIG 0 // Set to 1 for dbg purposes

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
    printf("    -i  input_filename          Input file (yaml for slot/cell config or single h5 file)\n");
    printf("    -r  # of iterations         Number of run iterations to run\n");
    printf("    -d  # of microseconds       Delay kernel duration in us\n");
    printf("    -k                          Enable reference check. Compare GPU output with test vector.\n");
    printf("    -m  process mode            streams(0), graphs (1).\n");
    printf("    -g                          Execute all cells in a slot on the same PDSCH object.\n"); // Reminder: they should all have identical static parameters.
    printf("    -s  setup_mode              0 (default) - setup is not timed; 1 - time setup only; no run is run; 2 - time both setup and run, back to back.\n");
    printf("    -c  cpu_id                  cpu_id used for CPU affinity setting.\n");
    printf("    -p  priority                Thread priority.\n");
    printf("    -a  alignment               Byte alignment between TBs for the same cell. Default is 1-byte alignment\n");
    printf("    -b                          Use asynchronous batched memcpy when copying the PDSCH input buffers.\n");
    printf("    -t                          PDSCH TB input buffers located on the device, instead of the host.\n");
    printf("    --G SM count                Use green contexts with specified SM count per context.\n");
    printf("    --P pipeline mode           0 (default) - full processing; 1 - AAS processing (single-cell only); 2 - post-FEC processing; 3 - post-FEC and RM scrambling processing.\n");
    printf("    --D delay(usec)             Delay in microseconds for kernel modeling FEC processing\n");
}

int main(int argc, char* argv[])
{
    int returnValue = 0;

    char nvlog_yaml_file[1024];
    // Relative path from binary to default nvlog_config.yaml
    std::string relative_path = std::string("../../../../").append(NVLOG_DEFAULT_CONFIG_FILE);
    nv_get_absolute_path(nvlog_yaml_file, relative_path.c_str());
    pthread_t log_thread_id = nvlog_fmtlog_init(nvlog_yaml_file, "pdsch_tx_multicell.log",NULL);
    nvlog_fmtlog_thread_init();

    //------------------------------------------------------------------
    // Parse command line arguments
    int         iArg = 1;
    std::string inputFileName;
    uint32_t    num_iterations      = 1;
    bool        ref_check_pdsch     = false;
    bool        use_batched_memcpy  = false;
    int         cfg_process_mode    = 0;
    int         cfg_priority        = 0;
    int         cfg_cpu_id          = -1;
    uint32_t    delayUs             = 10000;
    bool        group_cells            = false; // Group cells in the same cell-group and if possible process them all in a single kernel per component.
    int         time_setup_mode        = 0; // default mode: only time run, not setup
    std::string setup_modes[3] = {"GPU-run only", "GPU-setup only", "GPU-setup-and-run"};
    int forced_TB_byte_alignment = 1; // 1 byte alignment by default
    bool     useGreenCtxs       = false;
    uint32_t SMsPerGreenCtx     = 0;
    bool     pdsch_TB_input_on_GPU = false;
    bool     read_TB_CRC           = (_READ_TB_CRC_ == 1); //false by default
    int      cfg_pipeline_mode_index = 0; // default is full slot processing
    constexpr int PDSCH_PROCESSING_MODES = 4;
    std::string pipeline_modes_str[PDSCH_PROCESSING_MODES] = {"PDSCH_FULL_PROCESSING", "PDSCH_AAS_PROCESSING", "PDSCH_POST_FEC_PROCESSING", "PDSCH_POST_FEC_RM_SCRAMBLING_PROCESSING"};
    cuphyPdschPipelineMode_t pipeline_modes[PDSCH_PROCESSING_MODES] = {cuphyPdschPipelineMode_t::PDSCH_FULL_PROCESSING,
                                                                       cuphyPdschPipelineMode_t::PDSCH_AAS_PROCESSING,
                                                                       cuphyPdschPipelineMode_t::PDSCH_POST_FEC_PROCESSING,
                                                                       cuphyPdschPipelineMode_t::PDSCH_POST_FEC_RM_SCRAMBLING_PROCESSING};
    cuphyPdschPipelineMode_t pdsch_pipeline_mode = pipeline_modes[cfg_pipeline_mode_index]; // to be updated later
    uint32_t fec_delay_usec = 0;

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
                    NVLOGF_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT,  "ERROR: No filename provided.");
                }
                inputFileName.assign(argv[iArg++]);
                break;
            case 'r':
                if((++iArg >= argc) || (1 != sscanf(argv[iArg], "%i", &num_iterations)) || ((num_iterations <= 0)))
                {
                    NVLOGF_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT,  "ERROR: Invalid number of iterations");
                }
                ++iArg;
                break;
            case 'd':
                    delayUs = std::stoi(argv[++iArg]);
                    ++iArg;
                    break;
            case 'g':
                group_cells = true; // Group all cells in a slot and run them on a single PdschTx channel.
                ++iArg;
                break;
            case 'k':
                ref_check_pdsch = true;
                ++iArg;
                break;
            case 'b':
                use_batched_memcpy = true;
                ++iArg;
                break;
            case 't':
                pdsch_TB_input_on_GPU = true;
                ++iArg;
                break;
            case 's':
                if((++iArg >= argc) || (1 != sscanf(argv[iArg], "%i", &time_setup_mode)) || ((time_setup_mode < 0)) || ((time_setup_mode > 2)))
                {
                    NVLOGF_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT,  "ERROR: Invalid process mode");
                }
                ++iArg;
                break;
            case 'm':
                if((++iArg >= argc) || (1 != sscanf(argv[iArg], "%i", &cfg_process_mode)) || ((cfg_process_mode < 0)) || ((cfg_process_mode > 1)))
                {
                    NVLOGF_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT,  "ERROR: Invalid process mode");
                }
                ++iArg;
                break;
            case 'a':
                if((++iArg >= argc) || (1 != sscanf(argv[iArg], "%i", &forced_TB_byte_alignment)) || ((forced_TB_byte_alignment <= 0)) || ((forced_TB_byte_alignment > 32) || ((forced_TB_byte_alignment & (forced_TB_byte_alignment -1)) != 0)))
                {
                    NVLOGF_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT,  "ERROR: Invalid TB byte alignment {}. Supported values are 1, 2, 4, 8, 16, 32", forced_TB_byte_alignment);
                }
                ++iArg;
                break;
            case 'c':
                if((++iArg >= argc) || (1 != sscanf(argv[iArg], "%i", &cfg_cpu_id)) || ((cfg_cpu_id < 0)))
                {
                    NVLOGF_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT,  "ERROR: Invalid cpu_id");
                }
                ++iArg;
                break;
            case 'p':
                if((++iArg >= argc) || (1 != sscanf(argv[iArg], "%i", &cfg_priority)) || ((cfg_priority <= 0)) || ((cfg_priority >99)))
                {
                    NVLOGF_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT,  "ERROR: Invalid thread priority");
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
                            NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT,  "ERROR: Invalid or missing useGreenCtxs argument (--G)");
                            exit(1);
                        }
                        ++iArg;
                        break;
                    case 'P':
                        if((++iArg >= argc) || (1 != sscanf(argv[iArg], "%i", &cfg_pipeline_mode_index)))
                        {
                            NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT,  "ERROR: Invalid or missing pipeline mode argument (--P)");
                            exit(1);
                        }
                        ++iArg;
                        break;
                    case 'D':
                    {
                        int parsed_fec_delay_usec = 0;
                        if(((++iArg >= argc) || (1 != sscanf(argv[iArg], "%i", &parsed_fec_delay_usec))) || (parsed_fec_delay_usec < 0))
                        {
                            NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT,  "ERROR: Invalid or missing fec delay in usec argument (--D)");
                            exit(1);
                        }
                        fec_delay_usec = static_cast<uint32_t>(parsed_fec_delay_usec);
                        ++iArg;
                        break;
                    }
                    default:
                        NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT,  "ERROR: Unknown option: {}", argv[iArg]);
                        usage();
                        exit(1);
                        break;
                 }
                 break;
            default:
                NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT,  "ERROR: Unknown option: {}", argv[iArg]);
                usage();
                exit(1);
                break;
            }
        }
        else
        {
            NVLOGF_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT,  "ERROR: Invalid command line argument: {}", argv[iArg]);
        }
    }
    if(inputFileName.empty())
    {
        usage();
        exit(1);
    }

    const int gpuId = 0; // select GPU device 0
    CUDA_CHECK(cudaSetDevice(gpuId));
    CUdevice current_device;
    CU_CHECK(cuDeviceGet(&current_device, gpuId));

    CUmoduleLoadingMode mode{};
    CUresult status = cuModuleGetLoadingMode(&mode);
    if (status != CUDA_SUCCESS) NVLOGC_FMT(NVLOG_PDSCH, "cuModuleGetLoading returned {}", status);
    NVLOGC_FMT(NVLOG_PDSCH, "mode {} (reminder EAGER_LOADING is {} while lazy is {})", mode, CU_MODULE_EAGER_LOADING, CU_MODULE_LAZY_LOADING);

    if ((cfg_pipeline_mode_index < 0) || (cfg_pipeline_mode_index >= PDSCH_PROCESSING_MODES))
    {
        NVLOGE_FMT(NVLOG_TAG_BASE_CUPHY, AERIAL_CUPHY_EVENT,  "ERROR: Invalid --P argument {}. Needs to be in [0, {}] range.", cfg_pipeline_mode_index, PDSCH_PROCESSING_MODES - 1);
        exit(1);
    }
    else
    {
        if ((group_cells) && (cfg_pipeline_mode_index == 1))
        {
            NVLOGE_FMT(NVLOG_TAG_BASE_CUPHY, AERIAL_CUPHY_EVENT,  "ERROR: --P 1 not supported with -g");
            exit(1);
        }
    }
    pdsch_pipeline_mode = pipeline_modes[cfg_pipeline_mode_index];

#if CUDA_VERSION >= 12040
    CUdevResource initial_device_GPU_resources = {};
    CUdevResourceType default_resource_type = CU_DEV_RESOURCE_TYPE_SM; // other alternative is CU_DEV_RESOURCE_TYPE_INVALID
    unsigned int split_groups = 1;
    CUdevResource split_result[2] = {{}, {}};
    cuphy::cudaGreenContext pdsch_green_ctx;

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
        pdsch_green_ctx.create(gpuId, &split_result[0]);
        pdsch_green_ctx.bind();
        NVLOGC_FMT(NVLOG_PDSCH, "PDSCH green context will have access to {} SMs ({} SMs requested).", pdsch_green_ctx.getSmCount(), SMsPerGreenCtx);
    }
#endif

    // Determine input filename type (.yaml or h5)
    std::string inputFileExtn = inputFileName.substr(inputFileName.find_last_of(".") + 1);
    const std::string channelName = "PDSCH";
    bool is_yaml = (inputFileExtn == "yaml");
    std::optional<cuphy::test_config> testCfg;

    if(is_yaml)
    {
        testCfg.emplace(inputFileName.c_str());
        testCfg->print_channel(channelName); // print only PDSCH channel related parts of the input YAML file
        //testCfg->print(); // print entire YAML file contents, incl. other channels not run with this example
    }
    int num_cells = is_yaml ? testCfg->num_cells() : 1; // The same number of cells is present across all slots in case of a yaml
    int num_slots = is_yaml ? testCfg->num_slots() : 1;

    NVLOGC_FMT(NVLOG_PDSCH, "PDSCH multi-cell with {} cells and {} slots",  num_cells, num_slots);

    bool graphs_mode = (cfg_process_mode >= 1);
    std::string graphs_streams_mode_string = (graphs_mode) ? "Graphs" : "Streams";
    bool identical_LDPC_configs = true; // A runtime check resets LDPC configs to non-identical if they are not.
    cuphyPdschProcMode_t pdsch_proc_mode = (graphs_mode) ? PDSCH_PROC_MODE_GRAPHS : PDSCH_PROC_MODE_NO_GRAPHS;
    pdsch_proc_mode = static_cast<cuphyPdschProcMode_t>((uint32_t) pdsch_proc_mode | (uint32_t) PDSCH_INTER_CELL_BATCHING); // not needed; inter cell batching is applied regardless of this flag


    // The Downlink pipeline includes: (a) CRC, (b) LDPC  encoder, (c) Rate-Matching,
    // (d) Modulation Mapper and (e) DMRS components.

    std::vector<std::vector<cuphy::pdsch_tx>>       m_pdschTxPipes;
    std::vector<std::vector<pdschStaticApiDataset>> m_pdschTxStaticApiDatasets;
    std::vector<std::vector<pdschDynApiDataset>>    m_pdschTxDynamicApiDatasets;

    std::vector<stream>        streams;
    cudaEvent_t                start_streams_event;
    std::vector<cudaEvent_t> stop_streams_events(num_cells);
    const int num_pdsch_objects = group_cells ? 1 : num_cells;

    m_pdschTxPipes.resize(num_slots);
    m_pdschTxStaticApiDatasets.resize(num_slots);
    m_pdschTxDynamicApiDatasets.resize(num_slots);

    for(int idxSlot = 0; idxSlot < num_slots; idxSlot++) {
        int cells_in_slot = is_yaml ? testCfg->slots()[idxSlot].at(channelName).size() : 1;
        if (cells_in_slot != num_cells) {
            NVLOGF_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT, "Slot {} error: expected {} cells but got {}", idxSlot, num_cells, cells_in_slot);
        }
        // Reminder num_pdsch_object is the number of cells, if group_cells is false, or 1 otherwise
        m_pdschTxStaticApiDatasets[idxSlot].reserve(num_pdsch_objects);
        m_pdschTxDynamicApiDatasets[idxSlot].reserve(num_pdsch_objects);

        // Loop over all cells (not num_pdsch_objects) to populate the static and dynamic datasets
        // NB: current implementation of PDSCH static/dynamic datasets implicitly assumes that the static dataset is populated first and then the dynamic, because it relies on some max values
        // from static. Will thus follow this format to initially limit scope of changes.
        for(int i = 0; i < num_cells; i += 1)
        {
            std::string tv_filename = is_yaml ? testCfg->slots()[idxSlot].at(channelName)[i] : inputFileName;
            if((idxSlot == 0) && (i < num_pdsch_objects)) {
                streams.emplace_back(cudaStreamNonBlocking, PDSCH_STREAM_PRIORITY);
            }

            if (group_cells) {
                if (i == 0) {
                    m_pdschTxStaticApiDatasets[idxSlot].emplace_back(tv_filename, "", ref_check_pdsch, identical_LDPC_configs, PDSCH_STREAM_PRIORITY, num_cells, 0 /*maxNCbsPerTb*/, 0 /*maxNTbs */, 0 /*maxNPRbs*/, use_batched_memcpy, pdsch_pipeline_mode, read_TB_CRC, fec_delay_usec);
                } else {
                    // Update static parameters
                    m_pdschTxStaticApiDatasets[idxSlot][0].cumulativeUpdate(tv_filename, "", ref_check_pdsch, identical_LDPC_configs);
                }
            } else {
                m_pdschTxStaticApiDatasets[idxSlot].emplace_back(tv_filename, "", ref_check_pdsch, identical_LDPC_configs, PDSCH_STREAM_PRIORITY, 1 /*single cell per pipeline*/,  0 /*maxNCbsPerTb*/, 0 /*maxNTbs */, 0 /*maxNPRbs*/, use_batched_memcpy, pdsch_pipeline_mode, read_TB_CRC, fec_delay_usec);
                m_pdschTxDynamicApiDatasets[idxSlot].emplace_back(tv_filename, m_pdschTxStaticApiDatasets[idxSlot][i].pdschStatPrms.nMaxCellsPerSlot, streams[i].handle(), pdsch_proc_mode, m_pdschTxStaticApiDatasets[idxSlot][i].pdschStatPrms, pdsch_TB_input_on_GPU, forced_TB_byte_alignment, m_pdschTxStaticApiDatasets[idxSlot][i].getEmax());
                m_pdschTxPipes[idxSlot].emplace_back(m_pdschTxStaticApiDatasets[idxSlot][i].pdschStatPrms);
            }
        }

        // Dynamic now given NB above
        if (group_cells) {
            for(int i = 0; i < num_cells; i += 1)
            {
                std::string tv_filename = is_yaml ? testCfg->slots()[idxSlot].at(channelName)[i] : inputFileName;
                //printf("group_cells slot %d, cell %d has tvname %s\n", idxSlot, i, tv_filename.c_str());

                if(i == 0) {
                    m_pdschTxPipes[idxSlot].emplace_back(m_pdschTxStaticApiDatasets[idxSlot][i].pdschStatPrms);
                    m_pdschTxDynamicApiDatasets[idxSlot].emplace_back(tv_filename, m_pdschTxStaticApiDatasets[idxSlot][i].pdschStatPrms.nMaxCellsPerSlot, streams[i].handle(), pdsch_proc_mode, m_pdschTxStaticApiDatasets[idxSlot][i].pdschStatPrms, pdsch_TB_input_on_GPU, forced_TB_byte_alignment, m_pdschTxStaticApiDatasets[idxSlot][i].getEmax());
                } else {

                    // Update the dynamic parameters
                    m_pdschTxDynamicApiDatasets[idxSlot][0].cumulativeUpdate(tv_filename, streams[0].handle(), pdsch_proc_mode);
                }
            }
        }

    }



#if PRINT_PDSCH_CONFIG
    // Print static/dynamic dataset params for each PDSCH object. For dbg. purposes
    for(int idxSlot = 0; idxSlot < num_slots; idxSlot++) {
        NVLOGC_FMT(NVLOG_PDSCH, "======================== Slot {} ===============================", idxSlot);
        for(int i = 0; i < num_pdsch_objects; i += 1) {
            NVLOGC_FMT(NVLOG_PDSCH, "======================== PDSCH channel object {} ===============================", i);
            m_pdschTxStaticApiDatasets[idxSlot][i].print();
            cuphy::print_pdsch_dynamic(&m_pdschTxDynamicApiDatasets[idxSlot][i].pdsch_dyn_params);
            //Could also print cell group params as follows
            m_pdschTxDynamicApiDatasets[idxSlot][i].print();
            NVLOGC_FMT(NVLOG_PDSCH, "================================================================================");
        }
        NVLOGC_FMT(NVLOG_PDSCH, "================================================================================");
    }
#endif

    if (cfg_cpu_id >=0)
    {
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(cfg_cpu_id, &cpuset);
        int ret = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
        if (ret)
        {
            NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT, "pthread_setaffinity_np error: {}", ret);
	    return -1;
        }
    }
    if (cfg_priority > 0)
    {
        struct sched_param params;
        params.__sched_priority = cfg_priority;
        int ret = pthread_setschedparam(pthread_self(), SCHED_FIFO, &params);
        if (ret != 0)
        {
            NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT, "pthread_setschedparam error: {}", ret);
            return -1;
        }
    }

    NVLOGC_FMT(NVLOG_PDSCH, "");
    NVLOGC_FMT(NVLOG_PDSCH, "Timing {} PDSCH DL pipeline(s). ", num_cells);
    if (group_cells)
    {
        NVLOGC_FMT(NVLOG_PDSCH, "Grouping all cells in a slot.");
    } else {
        NVLOGC_FMT(NVLOG_PDSCH, "");
    }
    if (time_setup_mode== 0) {
        NVLOGC_FMT(NVLOG_PDSCH, "- NB: Allocations, setup processing not included.");
        NVLOGC_FMT(NVLOG_PDSCH, "");
    }


    /* PpdschTx::Run for all num_cells pipelines will be timed on streams[0].handle() CUDA stream (note, that is NOT stream 0).
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

    for(int i = 0; i < num_pdsch_objects; i++)
    {
        CUDA_CHECK(cudaEventCreateWithFlags(&stop_streams_events[i], cudaEventDisableTiming));
    }
    CUDA_CHECK(cudaEventCreateWithFlags(&start_streams_event, cudaEventDisableTiming));

    float total_time_slot[num_slots]; // Total time of a slot divided by number of iterations (num_iterations)
    float total_time_single_cell_slot[num_slots][num_pdsch_objects];
    //NVLOGC_FMT(NVLOG_PDSCH, "num slots {}, num_pdsch_objects {}", num_slots, num_pdsch_objects);

#if 0
    if(ref_check_pdsch) {
        for(int idxSlot = 0; idxSlot < num_slots; idxSlot++)
        {
            for(int i = 0; i < num_pdsch_objects; i++)
            {
                m_pdschTxPipes[idxSlot][i].setup(m_pdschTxDynamicApiDatasets[idxSlot][i].pdsch_dyn_params, nullptr);
                m_pdschTxPipes[idxSlot][i].run(static_cast<uint64_t>(pdsch_proc_mode));

                if (group_cells)
                {
                    updateRefCheckMultipleCells(m_pdschTxPipes[idxSlot][i].handle() , false);
                } else {
                    updateRefCheck(m_pdschTxPipes[idxSlot][i].handle() , false);
                }
            }
        }
    }
#endif
    try {

    for(int idxSlot = 0; idxSlot < num_slots; idxSlot++)
    {
        float                    total_time = 0;
        std::vector<float>       total_time_single_cell(num_pdsch_objects, 0);
        std::vector<event_timer> cuphy_timer_single_cell(num_pdsch_objects);

        if (time_setup_mode == 0){ // In this mode, setup is not timed
            for(int i = 0; i < num_pdsch_objects; i++)
            {
                m_pdschTxPipes[idxSlot][i].setup(m_pdschTxDynamicApiDatasets[idxSlot][i].pdsch_dyn_params, nullptr);
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

            for(int i = 0; i < num_pdsch_objects; i++)
            {
                cudaStream_t strm_handle = streams[i].handle();

                if(i != 0)
                {
                    CUDA_CHECK(cudaStreamWaitEvent(streams[i].handle(), start_streams_event, 0));
                }

                cuphy_timer_single_cell[i].record_begin(streams[i].handle());
                if (time_setup_mode != 0) { // If time_setup_mode is 1 or 2, it is timed
                    timePtStartSetup = Clock::now();
                    m_pdschTxPipes[idxSlot][i].setup(m_pdschTxDynamicApiDatasets[idxSlot][i].pdsch_dyn_params, nullptr);
                    timePtStopSetup = Clock::now();
                }
                if (time_setup_mode != 1)
                {
                    timePtStartRun = Clock::now();
                    m_pdschTxPipes[idxSlot][i].run(static_cast<uint64_t>(pdsch_proc_mode));
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
                if(time_setup_mode != 0)
                {
                    elapsedTimeDurationUs = timePtStopSetup - timePtStartSetup;
                    elapsedTimeUsSetup    += elapsedTimeDurationUs.count(); // accumulate for all objects (e.g., if running without -g on a single thread)
                }
                if(time_setup_mode != 1)
                {
                    elapsedTimeDurationUs = timePtStopRun - timePtStartRun;
                    elapsedTimeUsRun      += elapsedTimeDurationUs.count(); // accumulate fo all objects
                }
            }

            cuphy_timer.record_end(streams[0].handle());
            cuphy_timer.synchronize();
            total_time += cuphy_timer.elapsed_time_ms();

            for(int i = 0; i < num_pdsch_objects; i++)
            {
                cuphy_timer_single_cell[i].synchronize(); // To be safe
                total_time_single_cell[i] += cuphy_timer_single_cell[i].elapsed_time_ms();
            }

            // Currently ref. check for PDSCH happens as part of PDSCH GPU run if the flag is set, which will affect (pollute) timing measurements.
            // It is recommended you only use timing measurements without ref. check enabled.
            // Could potentially consider adding ref. check support to the PDSCH dataset and to this code here or not include the first iteration.
            if (ref_check_pdsch && (iter == 0)) {
               for(int i = 0; i < num_pdsch_objects; i++)
               {
                   // Disable ref-check for subsequent iterations for that object. Ref. check won't work properly in case of multiple iterations when only run is replayed
                   if (group_cells)
                   {
                         updateRefCheckMultipleCells(m_pdschTxPipes[idxSlot][i].handle() , false);
                   }
                   else
                   {
                         updateRefCheck(m_pdschTxPipes[idxSlot][i].handle() , false);
                   }
               }
            }
            gpu_us_delay(delayUs, 0, streams[0].handle()); // 10ms delay kernel. Can update/comment out.
            CUDA_CHECK(cudaEventRecord(start_streams_event, streams[0].handle()));
        }
        total_time_slot[idxSlot] = total_time / num_iterations;

        for(int i = 0; i < num_pdsch_objects; i++)
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

        NVLOGC_FMT(NVLOG_PDSCH, "Slot # {}, PDSCH pipeline(s) CPU Time: {:.2f} us total (summed over {} PDSCH objects, avg. over {} iterations)",
                   idxSlot, avgElapsedTimesUs[ELAPSED_CPU_RUN] + avgElapsedTimesUs[ELAPSED_CPU_SETUP], num_pdsch_objects, num_iterations);

        NVLOGC_FMT(NVLOG_PDSCH, "--> Slot # {}: CPU-Run {:.2f} us (min {:.2f} us, max {:.2f} us), CPU-Setup {:.2f} us (min {:.2f} us, max {:.2f} us)", idxSlot,
                   avgElapsedTimesUs[ELAPSED_CPU_RUN], minElapsedTimesUs[ELAPSED_CPU_RUN], maxElapsedTimesUs[ELAPSED_CPU_RUN],
                   avgElapsedTimesUs[ELAPSED_CPU_SETUP], minElapsedTimesUs[ELAPSED_CPU_SETUP], maxElapsedTimesUs[ELAPSED_CPU_SETUP]);

    } // end of slots

    for(int idxSlot = 0; idxSlot < num_slots; idxSlot++)
    {
        NVLOGC_FMT(NVLOG_PDSCH, "Slot # {}, PDSCH pipeline(s) {}: {:.2f} us (avg. over {} iterations) in {} and {} mode.", idxSlot, setup_modes[time_setup_mode].c_str(), total_time_slot[idxSlot] * 1000, num_iterations, (graphs_mode == 0) ? "Stream" : "Graphs", pipeline_modes_str[cfg_pipeline_mode_index]);
        for(int i = 0; i < num_pdsch_objects; i++)
        {
            if (group_cells) {
                NVLOGC_FMT(NVLOG_PDSCH, "--> PDSCH object # {} with {} cells: {:.2f} us (avg over {} iterations)", i, num_cells, total_time_single_cell_slot[idxSlot][i] * 1000, num_iterations);
            } else {
                NVLOGC_FMT(NVLOG_PDSCH, "--> Cell # {} : {:.2f} us (avg over {} iterations)", i, total_time_single_cell_slot[idxSlot][i] * 1000, num_iterations);
            }
        }
    }

    } // end of try (TODO will indent block in a subsequent change)
    catch(std::exception& e)
    {
        NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT, "EXCEPTION: {}", e.what());
        returnValue = 1;
    }
    catch(...)
    {
        NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT, "UNKNOWN EXCEPTION");
        returnValue = 2;
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    nvlog_fmtlog_close(log_thread_id);
    return returnValue;

}
