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

#include "pfmSortTest.h"
#include "nvlog_fmt.hpp"

/////////////////////////////////////////////////////////////////////////
// usage()
void usage() {
    printf("cuMAC PFM sorting test with [Arguments]\n");
    printf("Arguments:\n");
    printf("  -g  [GPU device index (default 0)]\n");
    printf("  -c  [Configuration file name (default config.yaml)]\n");
    printf("  -t  [Indication for creating HDF5 TV file(s): 0 - not creating, 1 - create a single TV at the last slot, 2 - create TVs for all slots (default 0)]\n");
    printf("  -v  [Verification mode - runs unit test and exits]\n");
    printf("  -h  [Print help usage]\n");
}

int main(int argc, char* argv[]) 
{
    int iArg = 1;

    std::string configFileName;
    uint8_t createHdf5TvInd = 0;
    uint16_t deviceIdx = 0;
    bool verifyMode = false;

    while(iArg < argc) {
        if('-' == argv[iArg][0]) {
            switch(argv[iArg][1]) {
                case 'g': // set GPU device index
                    if(++iArg >= argc) {
                        fprintf(stderr, "ERROR: No GPU device index given.\n");
                        exit(1);
                    } else {
                        deviceIdx = static_cast<uint16_t>(atoi(argv[iArg++]));
                    }
                    break;
                case 'c': // set configuration file name
                    if(++iArg >= argc) {
                        fprintf(stderr, "ERROR: No configuration file name given.\n");
                        exit(1);
                    } else {
                        configFileName.assign(argv[iArg++]);
                    }
                    break;
                case 't': // set indication for creating an HDF5 TV file
                    if(++iArg >= argc) {
                        fprintf(stderr, "ERROR: No indication for creating an HDF5 TV file given.\n");
                        exit(1);
                    } else {
                        createHdf5TvInd = static_cast<uint8_t>(atoi(argv[iArg++]));
                        if (createHdf5TvInd != 0 && createHdf5TvInd != 1 && createHdf5TvInd != 2) {
                            fprintf(stderr, "ERROR: Invalid indication for creating an HDF5 TV file.\n");
                            exit(1);
                        }
                    }
                    break;
                case 'v': // enable verification mode
                    verifyMode = true;
                    iArg++;
                    break;
                case 'h': // print help usage
                    usage();
                    exit(0);
                    break;
                default:
                    fprintf(stderr, "ERROR: Unknown option: %s\n", argv[iArg]);
                    usage();
                    exit(1);
                    break;
            }
        } else {
            fprintf(stderr, "ERROR: Invalid command line argument: %s\n", argv[iArg]);
            exit(1);
        }
    }

    if (configFileName.size() == 0) {
        // default configuration file path/name from build directory
        configFileName = CONFIG_PFMSORT_CONFIG_YAML;
    }

    // Get full path for config file using relative path resolution
    char config_yaml_path[MAX_PATH_LEN];
    get_full_path_file(config_yaml_path, NULL, configFileName.c_str(), CONFIG_CUBB_ROOT_DIR_RELATIVE_NUM);
    printf("config_yaml=%s\n", config_yaml_path);
    std::string configFilePathStr = std::string(config_yaml_path);

    // set GPU device with fallback mechanism
    int deviceCount{};
    CUDA_CHECK_ERR(cudaGetDeviceCount(&deviceCount));

    if (static_cast<int>(deviceIdx) >= deviceCount) {
        printf("WARNING: Requested GPU device %u exceeds available device count (%d). Falling back to GPU device 0.\n", 
            deviceIdx, deviceCount);
        deviceIdx = 0;
    }
    
    CUDA_CHECK_ERR(cudaSetDevice(deviceIdx));
    printf("cuMAC PFM sorting test: Running on GPU device %d (total devices: %d)\n", 
           deviceIdx, deviceCount);

    if (createHdf5TvInd > 0) {
        printf("Creating HDF5 TV file\n");
    } else {
        printf("Not creating HDF5 TV file\n");
    }

    // create stream
    cudaStream_t cuStrmMain;
    CUDA_CHECK_ERR(cudaStreamCreate(&cuStrmMain));

    // create PFM data manager
    std::unique_ptr<pfm_data_manage_t> pfmDataManage;
    pfmDataManage = std::make_unique<pfm_data_manage_t>(configFilePathStr, cuStrmMain);

    // Verification mode: load and validate TV then exit
    if (verifyMode) {
        printf("Verification mode: Running unit test with cell_num=%d, slot_num=%d\n", pfmDataManage->get_num_cell(), pfmDataManage->get_num_slot());
        int ret = pfmDataManage->unit_test(pfmDataManage->get_num_cell(), pfmDataManage->get_num_slot());
        if (ret != 0) {
            printf("Result: PFM sorting test failed.\n");
            exit(1);
        }
        printf("Result: PFM sorting test passed.\n");
        exit(0);
    }

    // setup randomness seed
    srand(pfmDataManage->get_seed());

    // create PFM sorting object
    std::unique_ptr<cumac::pfmSort> pfmSort;
    pfmSort = std::make_unique<cumac::pfmSort>();

    uint32_t num_slot = pfmDataManage->get_num_slot();

    // verify the number of H5 TVs to be created if the test mode is -t 2
    if (createHdf5TvInd == 2) {
        if (num_slot > pfmDataManage->get_max_num_h5_tv_created()) {
            // printf("ERROR: the number of H5 TVs to be created is bigger than the maximum allowed value configured in the config.yaml file.\n");
            //exit(1);
            num_slot = pfmDataManage->get_max_num_h5_tv_created();
        }
    }

    for (uint32_t slot_idx = 0; slot_idx < num_slot; slot_idx++) {
        pfmDataManage->prepare_pfm_data();
        
        pfmSort->setup(pfmDataManage->get_pfm_sort_task());
        
        pfmSort->run(pfmDataManage->get_pfm_output_host_ptr());
        pfmDataManage->sync_stream();

        pfmDataManage->cpu_pfm_sort();

        bool is_valid = pfmDataManage->validate_pfm_output();
        if (!is_valid) {
            printf("Result: PFM sorting test failed.\n");
            exit(1);
        }

        if ((createHdf5TvInd == 1 && slot_idx == num_slot-1) || (createHdf5TvInd == 2)) {
            std::string saveTvName = pfmDataManage->pfm_save_tv_H5();
            printf("Saved TV to %s\n", saveTvName.c_str());

            bool is_valid_tv = pfmDataManage->pfm_validate_tv_h5(saveTvName);
            if (!is_valid_tv) {
                printf("PFM TV validation failed.\n");
                exit(1);
            }

            printf("PFM TV validation passed.\n");
        }

        pfmDataManage->increase_slot_idx();
    }

    printf("Result: PFM sorting test passed.\n");
    exit(0);
}