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

#include "multiCellMuMimoScheduler.h"
#include "mMimoNetwork.h"

/////////////////////////////////////////////////////////////////////////
// usage()
void usage() {
    printf("cuMAC 64T64R MU-MIMO scheduler pipeline test with [Arguments]\n");
    printf("Arguments:\n");
    printf("  -g  [GPU device index (default 0)]\n");
    printf("  -i  [cuMAC_HDF5_TV_file]\n");
    printf("  -a  [Indication for AODT testing: 0 - not for AODT testing, 1 - for AODT testing (default 0)]\n");
    printf("  -c  [Configuration file name (default config.yaml)]\n");
    printf("  -h  [Print help usage]\n");
}

int main(int argc, char* argv[]) 
{
    int iArg = 1;

    std::string inputFileName;
    std::string configFileName;
    uint16_t deviceIdx = 0;

    // AODT testing indication
    uint8_t aodtTest = 0;

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
                case 'i': // input channel file name
                    if(++iArg >= argc) {
                        fprintf(stderr, "ERROR: No input file name given.\n");
                        exit(1);
                    } else {
                        inputFileName.assign(argv[iArg++]);
                    }
                    break;
                case 'a': // set AODT test indication
                    if((++iArg >= argc) || (1 != sscanf(argv[iArg], "%hhi", &aodtTest)) || (aodtTest != 0 && aodtTest != 1 )) {
                        fprintf(stderr, "ERROR: Unsupported AODT test indication.\n");
                        exit(1);
                    }
                    ++iArg;
                    break; 
                case 'c': // set configuration file name
                    if(++iArg >= argc) {
                        fprintf(stderr, "ERROR: No configuration file name given.\n");
                        exit(1);
                    } else {
                        configFileName.assign(argv[iArg++]);
                    }
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
        configFileName = "./cuMAC/examples/multiCellMuMimoScheduler/config.yaml";
    }

    // set GPU device with fallback mechanism
    int deviceCount{};
    CUDA_CHECK_ERR(cudaGetDeviceCount(&deviceCount));
    
    unsigned my_dev = gpuDeviceIdx;
    if (static_cast<int>(gpuDeviceIdx) >= deviceCount) {
        printf("WARNING: Requested GPU device %u exceeds available device count (%d). Falling back to GPU device 0.\n", 
               gpuDeviceIdx, deviceCount);
        my_dev = 0;
    }
    
    CUDA_CHECK_ERR(cudaSetDevice(my_dev));
    printf("cuMAC 64T64R MU-MIMO scheduler pipeline test: Running on GPU device %d (total devices: %d)\n", 
           my_dev, deviceCount);

    // create stream
    cudaStream_t cuStrmMain;
    CUDA_CHECK_ERR(cudaStreamCreate(&cuStrmMain));

    // create network 
    std::unique_ptr<mMimoNetwork> mMimoNet;
    
    if (inputFileName.size() == 0) { // TV not provided
        mMimoNet = std::make_unique<mMimoNetwork>(configFileName, cuStrmMain);
    } else {
        mMimoNet = std::make_unique<mMimoNetwork>(inputFileName, cuStrmMain);
    }

    // setup randomness seed
    srand(mMimoNet->getSeed());

    // MU-MIMO UE sorting
    auto mcUeSortGpu = std::make_unique<cumac::multiCellMuUeSort>(mMimoNet->cellGrpPrmsGpu.get());

    // MU-MIMO UE grouping
    auto mcUeGrpGpu = std::make_unique<cumac::multiCellMuUeGrp>(mMimoNet->cellGrpPrmsGpu.get());

    // beamforming
    auto beamformGpu = std::make_unique<cumac::multiCellBeamform>(mMimoNet->cellGrpPrmsGpu.get());

    // GPU MCS selection
    auto mcsSelGpu = std::make_unique<cumac::mcsSelectionLUT>(mMimoNet->cellGrpPrmsGpu.get(), cuStrmMain);

    if (inputFileName.size() > 0) { // TV provided
        loadFromH5(inputFileName, mMimoNet->cellGrpUeStatusGpu.get(), mMimoNet->cellGrpPrmsGpu.get(), mMimoNet->schdSolGpu.get());
        preProcessInput(mMimoNet->cellGrpUeStatusGpu.get(), mMimoNet->cellGrpPrmsGpu.get(), mMimoNet->schdSolGpu.get());
    }

    if (inputFileName.size() == 0) { // TV not provided
        mMimoNet->genFadingChannGpu(0);
    }

    // setup modules
    mcUeSortGpu->setup(mMimoNet->cellGrpUeStatusGpu.get(), mMimoNet->schdSolGpu.get(), mMimoNet->cellGrpPrmsGpu.get(), cuStrmMain);
    std::cout<<"UE sorting setup executed"<<std::endl;

    mcUeGrpGpu->setup(mMimoNet->cellGrpUeStatusGpu.get(), mMimoNet->schdSolGpu.get(), mMimoNet->cellGrpPrmsGpu.get(), cuStrmMain);
    std::cout<<"UE grouping setup executed"<<std::endl;

    beamformGpu->setup(mMimoNet->cellGrpUeStatusGpu.get(), mMimoNet->schdSolGpu.get(), mMimoNet->cellGrpPrmsGpu.get(), cuStrmMain);
    std::cout<<"Beamforming setup executed"<<std::endl;

    mcsSelGpu->setup(mMimoNet->cellGrpUeStatusGpu.get(), mMimoNet->schdSolGpu.get(), mMimoNet->cellGrpPrmsGpu.get(), cuStrmMain);
    std::cout<<"MCS selection setup executed"<<std::endl;

    // run modules
    mcUeSortGpu->run(cuStrmMain);
    std::cout<<"UE sorting run executed"<<std::endl;

    mcUeGrpGpu->run(cuStrmMain);
    std::cout<<"UE grouping run executed"<<std::endl;

    beamformGpu->run(cuStrmMain);
    std::cout<<"Beamforming run executed"<<std::endl;  

    mcsSelGpu->run(cuStrmMain);
    std::cout<<"MCS selection run executed"<<std::endl;

    CUDA_CHECK_ERR(cudaStreamSynchronize(cuStrmMain));
 
    std::string saveTvName = "TV_cumac_result_64T64R_" + std::to_string(mMimoNet->getNCell()) +"PC_" + (mMimoNet->getDL() == 1 ? "DL" : "UL") + ".h5";

    saveToH5(saveTvName,
             mMimoNet->cellGrpUeStatusGpu.get(),
             mMimoNet->cellGrpPrmsGpu.get(),
             mMimoNet->schdSolGpu.get());
    
    if (inputFileName.size() == 0) { // no TV provided
        mMimoNet->validateSchedSol();
        printf("Summary - cuMAC multi-cell MU-MIMO scheduler simulation test: PASS\n");
        return 0;
    } else {
        bool solCheckPass = compareCpuGpuAllocSol(saveTvName, inputFileName);
        if (solCheckPass) {
            printf("Summary - cuMAC multi-cell MU-MIMO scheduler solution check: PASS\n");
        } else {
            printf("Summary - cuMAC multi-cell MU-MIMO scheduler solution check: FAIL\n");
        }
        return !solCheckPass;
    }
}
   