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
#include "cuphy.hpp"
#include "cuphy_internal.h"
#include "cuphy_hdf5.hpp"
#include "hdf5hpp.hpp"
#include "util.hpp"
#include "datasets.hpp"

#include <cstdlib>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

#include "memtrace.h"
#include "nvlog.hpp"

using namespace std;
using namespace cuphy;
using namespace hdf5hpp;


void usage() {
    printf("  embed_pdcch_tf_signal [options]\n");
    printf("  Options:\n");
    printf("    -h                          Display usage information\n");
    printf("    -i  input_filename          Input HDF5 filename\n");
    printf("    -r  # of iterations         Number of iterations to run\n");
    printf("    -m  proc_mode               Processing mode: streams(0), graphs (1)\n");
}

int main(int argc, char* argv[])
{
    int iArg = 1;
    std::string inputFileName;
    uint64_t procModeBmsk   = 0;
    int      num_iterations = 1;
    char nvlog_yaml_file[1024];
    uint8_t  kernelSelMode = PDCCH_ALL;
    uint32_t delayUs = 0;
    // Relative path from binary to default nvlog_config.yaml
    std::string relative_path = std::string("../../../../").append(NVLOG_DEFAULT_CONFIG_FILE);
    nv_get_absolute_path(nvlog_yaml_file, relative_path.c_str());
    pthread_t log_thread_id = nvlog_fmtlog_init(nvlog_yaml_file, "pdcch_tx.log",NULL);
    nvlog_fmtlog_thread_init();

    while(iArg < argc)
    {
        if('-' == argv[iArg][0])
        {
            switch(argv[iArg][1])
            {
            case 'h':
                usage();
                exit(0);
            case 'i':
                if(++iArg >= argc)
                {
                    NVLOGE_FMT(NVLOG_PDCCH, AERIAL_CUPHY_EVENT, "ERROR: No filename provided.");
                }
                inputFileName.assign(argv[iArg++]);
                break;
            case 'r':
                if((++iArg >= argc) || (1 != sscanf(argv[iArg], "%i", &num_iterations)) || ((num_iterations <= 0)))
                {
                    NVLOGF_FMT(NVLOG_PDCCH, AERIAL_CUPHY_EVENT, "ERROR: Invalid number of iterations");
                }
                ++iArg;
                break;
            case 'm':
                if((++iArg >= argc) || (1 != sscanf(argv[iArg], "%lu", &procModeBmsk)) || ((PDCCH_PROC_MODE_STREAMS != procModeBmsk) && (PDCCH_PROC_MODE_GRAPHS != procModeBmsk)))
                {
                    NVLOGF_FMT(NVLOG_PDCCH, AERIAL_CUPHY_EVENT, "ERROR: Invalid processing mode ({:#x})", procModeBmsk);
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
                if((++iArg >= argc) || (1 != sscanf(argv[iArg], "%u", &delayUs)) || delayUs > 5000)
                {
                    NVLOGE_FMT(NVLOG_PDCCH, AERIAL_CUPHY_EVENT,  "ERROR: too large delay duration");
                    exit(1);
                }
                ++iArg;
                break;
            default:
                NVLOGE_FMT(NVLOG_PDCCH, AERIAL_CUPHY_EVENT, "ERROR: Unknown option: {}", argv[iArg]);
                usage();
                exit(1);
            }
        }
        else
        {
            NVLOGF_FMT(NVLOG_PDCCH, AERIAL_CUPHY_EVENT, "ERROR: Invalid command line argument: {}", argv[iArg]);
        }
    }
    if(inputFileName.empty())
    {
        usage();
        exit(1);
    }

    cuphyStatus_t status = CUPHY_STATUS_SUCCESS;

    // Create CUDA stream
    stream strm(cudaStreamNonBlocking);
    cudaStream_t strm_handle = strm.handle();

    // Allocate PDCCH handle
    std::unique_ptr<cuphyPdcchTxHndl_t> pdcch_handle = std::make_unique<cuphyPdcchTxHndl_t>();

    pdcchStaticApiDataset static_dataset;
    static_dataset.pdcchStatPrms.kernelSelOption = kernelSelMode;
    static_dataset.pdcchStatPrms.delayUs = delayUs;
    pdcchDynApiDataset    dynamic_dataset(inputFileName, 1 /* number of cells*/, strm_handle, procModeBmsk);

    status = cuphyCreatePdcchTx(pdcch_handle.get(), &(static_dataset.pdcchStatPrms));
    if(status != CUPHY_STATUS_SUCCESS)
    {
        NVLOGF_FMT(NVLOG_PDCCH, AERIAL_CUPHY_EVENT, "Error! cuphyCreatePdcchTx(): {}", cuphyGetErrorString(status));
    }
    event_timer cuphy_timer;

    // Enable dynamic memory allocation tracing in real-time code path; only applicable when running with LD_PRELOAD=<PATH to libmimalloc.so>
    memtrace_set_config(MI_MEMTRACE_CONFIG_ENABLE | MI_MEMTRACE_CONFIG_EXIT_AFTER_BACKTRACE);

    status = cuphySetupPdcchTx(*pdcch_handle, &(dynamic_dataset.pdcch_dyn_params));
    if(status != CUPHY_STATUS_SUCCESS)
    {
        NVLOGF_FMT(NVLOG_PDCCH, AERIAL_CUPHY_EVENT, "Error! cuphySetupPdcchTx(): {}", cuphyGetErrorString(status));
    }

    strm.synchronize();

    // Time PDCCH Run. NB: CRC, scrambling sequence generation, async copies etc are not counted!
    cuphy_timer.record_begin(strm_handle);

    for (int iter = 0; iter < num_iterations; iter++) {
        // generate PDCCH QAM and DMRS, and map them to subcarriers
        status = cuphyRunPdcchTx(*pdcch_handle, 0);
    }

    if (status != CUPHY_STATUS_SUCCESS) {
        NVLOGF_FMT(NVLOG_PDCCH, AERIAL_CUPHY_EVENT, "Error! cuphyRunPdcchTx(): {}", cuphyGetErrorString(status));
    }
    memtrace_set_config(0); // disable memory allocation tracing beyond this point

    cuphy_timer.record_end(strm_handle);
    cuphy_timer.synchronize();
    float time1 = cuphy_timer.elapsed_time_ms();
    time1 /= num_iterations;
    NVLOGC_FMT(NVLOG_PDCCH, "PDCCH TX pipeline: {:.2f} us (avg. over {} iterations), Run() in {} mode", time1 * 1000, num_iterations, (procModeBmsk == 0) ? "Stream": "Graphs");
    NVLOGC_FMT(NVLOG_PDCCH, "- NB: Allocations/CRC computation/Scrambling sequence generation/mem. copies not included!");

    int rslt = (dynamic_dataset.refCheck(true) == 0) ? 0 : 1;

    // Cleanup
    status = cuphyDestroyPdcchTx(*pdcch_handle);
    if(status != CUPHY_STATUS_SUCCESS)
    {
        NVLOGF_FMT(NVLOG_PDCCH, AERIAL_CUPHY_EVENT, "Error! cuphyDestroyPdcchTx(): {}", cuphyGetErrorString(status));
    }

    nvlog_fmtlog_close(log_thread_id);

    return rslt;
}
