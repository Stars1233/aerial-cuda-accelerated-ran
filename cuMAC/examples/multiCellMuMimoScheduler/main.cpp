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
    printf("\n");
    printf("NOTE: PDSCH PHY abstraction (MMSE-IRC SINR, BLER, TBS) runs on CPU only.\n");
    printf("      Scheduling, channel generation, and beamforming still use the GPU.\n");
    printf("      -p values 1 (GPU) and 2 (Both) are rejected at startup.\n");
    printf("\n");
    printf("Arguments:\n");
    printf("  -i  [cuMAC_HDF5_TV_file]\n");
    printf("  -a  [Indication for AODT testing: 0 - not for AODT testing, 1 - for AODT testing (default 0)]\n");
    printf("  -c  [Configuration file name (default config.yaml)]\n");
    printf("  -t  [Number of simulation slots (default 1)]\n");
    printf("  -s  [Random seed (default: value from config file)]\n");
    printf("  -l  [Enable per-slot log recording (default: disabled)]\n");
    printf("  -p  [PHY abstraction target: 0=CPU (default, only supported), 1=GPU (not implemented), 2=Both (not implemented)]\n");
    printf("  -h  [Print help usage]\n");
}

int main(int argc, char* argv[])
{
    cumac::loadParameters();

    int iArg = 1;

    std::string inputFileName;
    std::string configFileName;
    int deviceIdx = gpuDeviceIdx;

    // AODT testing indication
    uint8_t aodtTest = 0;

    // simulation and PHY abstraction parameters
    PhyExecTarget gpuInd = PhyExecTarget::CPU;
    uint16_t totSimuSlots = 1;
    bool saveSlotLog = false;
    int randomSeed = -1;


    while(iArg < argc) {
        if('-' == argv[iArg][0]) {
            switch(argv[iArg][1]) {
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
                case 't': // set simulation slot number
                    if(++iArg >= argc) {
                        fprintf(stderr, "ERROR: No simulation slot number given.\n");
                        exit(1);
                    } else {
                        totSimuSlots = static_cast<uint16_t>(atoi(argv[iArg++]));
                        if (totSimuSlots == 0) {
                            fprintf(stderr, "ERROR: Simulation slot count must be > 0.\n");
                            exit(1);
                        }
                    }
                    break;
                case 's': // set random seed
                    if(++iArg >= argc) {
                        fprintf(stderr, "ERROR: No random seed given.\n");
                        exit(1);
                    } else {
                        randomSeed = atoi(argv[iArg++]);
                    }
                    break;
                case 'l': // enable slot log recording
                    saveSlotLog = true;
                    ++iArg;
                    break;
                case 'p': // PHY abstraction execution target (PDSCH BLER/TBS path; CPU-only today)
                    if (++iArg >= argc) {
                        fprintf(stderr, "ERROR: No PHY abstraction target given.\n");
                        exit(1);
                    } else {
                        const int phyTarget = atoi(argv[iArg++]);
                        if (phyTarget < 0 || phyTarget > 2) {
                            fprintf(stderr, "ERROR: PHY abstraction target must be 0 (CPU), 1 (GPU), or 2 (Both).\n");
                            exit(1);
                        }
                        gpuInd = static_cast<PhyExecTarget>(phyTarget);
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

    if (!isPhyExecTargetSupported(gpuInd)) {
        fprintf(stderr,
                "ERROR: PHY abstraction target %d is not implemented.\n"
                "       Only CPU (0) is supported for PDSCH BLER/TBS (updateDataRatePdschCpu).\n"
                "       GPU (1) and Both (2) will be added in a future release.\n"
                "       Scheduling, channel generation, and beamforming still use CUDA device 0.\n",
                static_cast<int>(gpuInd));
        exit(1);
    }

    if (configFileName.size() == 0) {
        // default configuration file path/name from build directory
        configFileName = "./cuMAC/examples/multiCellMuMimoScheduler/config.yaml";
    }

    // set GPU device with fallback mechanism
    int deviceCount{};
    CUDA_CHECK_ERR(cudaGetDeviceCount(&deviceCount));
    
    unsigned my_dev = static_cast<unsigned>(deviceIdx);
    if (deviceIdx < 0 || deviceIdx >= deviceCount) {
        printf("WARNING: Requested GPU device %d exceeds available device count (%d). Falling back to GPU device 0.\n",
               deviceIdx, deviceCount);
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
        mMimoNet = std::make_unique<mMimoNetwork>(configFileName, cuStrmMain, randomSeed);
    } else {
        mMimoNet = std::make_unique<mMimoNetwork>(inputFileName, cuStrmMain, randomSeed);
    }

    srand(mMimoNet->getSeed());

    // initialize simulation record log
    if (saveSlotLog){
        mMimoNet->initSimuRecords(totSimuSlots);
    }
    
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

    std::cout<<"\nSimulation started!\n"<<std::endl;
    printf("PHY abstraction (PDSCH MMSE-IRC / BLER / TBS): CPU only (PhyExecTarget::CPU)\n\n");

    for (uint16_t slotIdx = 0; slotIdx < totSimuSlots; slotIdx++) {
        std::cout<<"\nsimulation started for slot: "<<slotIdx<<std::endl;

        // Synchronize stream to ensure GPU operations complete before CPU accesses managed memory
        CUDA_CHECK_ERR(cudaStreamSynchronize(cuStrmMain));

        // generate fading channel
        if (inputFileName.size() == 0) { // TV not provided
            mMimoNet->genFadingChannGpu(slotIdx);
        }

        // setup modules
        mcUeSortGpu->setup(mMimoNet->cellGrpUeStatusGpu.get(), mMimoNet->schdSolGpu.get(), mMimoNet->cellGrpPrmsGpu.get(), cuStrmMain);

        mcUeGrpGpu->setup(mMimoNet->cellGrpUeStatusGpu.get(), mMimoNet->schdSolGpu.get(), mMimoNet->cellGrpPrmsGpu.get(), cuStrmMain);

        beamformGpu->setup(mMimoNet->cellGrpUeStatusGpu.get(), mMimoNet->schdSolGpu.get(), mMimoNet->cellGrpPrmsGpu.get(), cuStrmMain);

        mcsSelGpu->setup(mMimoNet->cellGrpUeStatusGpu.get(), mMimoNet->schdSolGpu.get(), mMimoNet->cellGrpPrmsGpu.get(), cuStrmMain);

        // run modules
        mcUeSortGpu->run(cuStrmMain);

        mcUeGrpGpu->run(cuStrmMain);

        beamformGpu->run(cuStrmMain);

        mcsSelGpu->run(cuStrmMain);

        // Synchronize stream to ensure GPU operations complete before CPU accesses managed memory
        CUDA_CHECK_ERR(cudaStreamSynchronize(cuStrmMain));

        // PHY layer processing
        mMimoNet->phyAbstract(gpuInd, slotIdx, saveSlotLog);
    }

    CUDA_CHECK_ERR(cudaStreamSynchronize(cuStrmMain));

    std::cout<<"\nSimulation finished!\n"<<std::endl;
    std::string saveTvName = "TV_cumac_result_64T64R_" + std::to_string(mMimoNet->getNCell()) +"PC_" + (mMimoNet->getDL() == 1 ? "DL" : "UL") + ".h5";
    // std::string saveTvName = "TV_cumac_result_64T64R_" + std::to_string(mMimoNet->getFadingType()) +"fading_" + std::to_string(mMimoNet->getNCell()) +"PC_" + std::to_string(mMimoNet->getNActiveUePerCell()) + "ActiveUePerCell_" + std::to_string(totSimuSlots) + "slots_seed" + std::to_string(mMimoNet->getSeed()) + ".h5";

    if (saveSlotLog){
        saveToH5_perSlotLog(saveTvName,
                            mMimoNet->cellGrpUeStatusGpu.get(),
                            mMimoNet->cellGrpPrmsGpu.get(),
                            mMimoNet->schdSolGpu.get(),
                            mMimoNet->perUEperSlotMcs,
                            mMimoNet->perUEperSlotLayerSel,
                            mMimoNet->perUEperSlotAvgSinr,
                            mMimoNet->perUEperRbgperSlotGeometrySinr,
                            mMimoNet->perUEperSlotServingCellChannelGain,
                            mMimoNet->perUEperCellperSlotAllCellsChannelGain,
                            mMimoNet->perUEperSlotServingCellPathLossAndSF,
                            mMimoNet->perUEperCellperSlotAllCellsPathLossAndSF,
                            mMimoNet->perUEperRbgperSlotGeometrySir,
                            mMimoNet->perUEperRbgperSlotGeometrySnr,
                            mMimoNet->perUEperRbgperSlotRawPreEqSinr,
                            mMimoNet->perUEperRbgperSlotRawPreEqSir,
                            mMimoNet->perUEperRbgperSlotRawPreEqSnr,
                            mMimoNet->perUEperRbgperLayerperSlotRawSinr,
                            mMimoNet->perUEperSlotTbErr,
                            mMimoNet->perUEperSlotBler,
                            mMimoNet->perUEperSlotInsRate,
                            mMimoNet->perUEperSlotAvgRate,
                            mMimoNet->perCellperSlotNumScheUEs,
                            mMimoNet->perCellperSlotTbErr,
                            mMimoNet->perCellperSlotInsRate,
                            mMimoNet->perCellperGrpperSlotNumScheLayers);
    } else {
        saveToH5(saveTvName,
            mMimoNet->cellGrpUeStatusGpu.get(),
            mMimoNet->cellGrpPrmsGpu.get(),
            mMimoNet->schdSolGpu.get());
    }

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
   