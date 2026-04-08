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

#include "fading_chan.cuh"
#include "chanModelsCommon.h"

/**
 * @brief display usage of fading channel test
 * 
 */
void usage() {
    std::cout << "Fading channel link-level test [options]" << std::endl;
    std::cout << "  Arguments:" << std::endl;
    printf("  -i  [input TV file name to read carrier pars, channel pars, freq rx samples]\n");
    printf("  -m  [fading mode: 0 for AWGN, 1 for TDL, 2 for CDL (default 1)]\n");
    printf("  -s  [random seed (default 0)]\n");
    printf("  -n  [number of TTIs (default 10)]\n");
    printf("  -t  [PHY channel type, 0 - PUSCH, 1 - PUCCH, 2 - PRACH (default 1)]\n");
    printf("  -p  [enable half precision (default disable)]\n");
    printf("  -d  [enable debug to print sample channel and save TDL coes in H5 file (default disable)]\n");
    printf("  -h  display usage information \n");
    std::cout << "Example (TDL, seed 0, 10 iterations, using FP32): ./fading_chan_ex -i <path to TV> " << std::endl;
    std::cout << "Example (TDL, seed 0, 10 iterations, using FP16): ./fading_chan_ex -i <path to TV> -p " << std::endl;
    std::cout << "Example (AWGN, seed 0, 10 iterations, using FP16): ./fading_chan_ex -i <path to TV> -m 0 -p " << std::endl;
    std::cout << "Example (TDL, seed 1, 10 iterations, using FP32, PRACH): ./fading_chan_ex -i <path to TV> -s 1 -t 2" << std::endl;
}

/**
 * @brief main test function for fading channel
 * 
 * @tparam Tcomplex Template for complex number, cuPHY tensor type and the scalar type will be automatically decided
 * @param inputFileName input TV file name to read carrier pars, channel pars, freq rx samples
 * @param nTti number of iteration to measure time
 * @param fadingMode fading mode: 0 for AWGN, 1 for TDL
 * @param randSeed random seed, used for TDL and generating noise
 * @param phyChanType 0 - PUSCH, 1 - PUCCH, 2 - PRACH
 * @param enableHalfPrecision true for use FP16, false for use FP32
 * @param debugInd true for use FP16, false for use FP32
 */
template<typename Tcomplex>
void test_fadeChan(std::string inputFileName, int nTti = 10, uint8_t fadingMode = 1, uint16_t randSeed = 0, uint8_t phyChanType = 1, bool & enableHalfPrecision = 0, bool & debugInd = 0)
{
    cudaStream_t cuMainStrm;
    cudaStreamCreate(&cuMainStrm);
    // Input file
    hdf5hpp::hdf5_file inputFile = hdf5hpp::hdf5_file::open(inputFileName.c_str());
    /*------------------------- Create buffer --------------------------------*/
    hdf5hpp::hdf5_dataset dset_carrier  = inputFile.open_dataset("carrier_pars");
    hdf5hpp::hdf5_dataset_elem dset_elem = dset_carrier[0];
    uint16_t N_sc                   = dset_elem["N_sc"].as<uint16_t>();
    uint16_t N_bsLayer              = dset_elem["N_bsAnt"].as<uint16_t>();
    uint16_t N_ueLayer              = dset_elem["N_ueAnt"].as<uint16_t>();
    uint16_t N_symbol_slot          = dset_elem["N_symbol_slot"].as<uint16_t>();
    Tcomplex* freqTxGpuPtr = nullptr;
    Tcomplex* freqRxGpuPtr = nullptr;
    size_t txBytes = sizeof(Tcomplex) * (size_t)N_sc * N_symbol_slot * N_bsLayer;
    size_t rxBytes = sizeof(Tcomplex) * (size_t)N_sc * N_symbol_slot * N_ueLayer;
    CHECK_CUDAERROR(cudaMalloc(&freqTxGpuPtr, txBytes));
    CHECK_CUDAERROR(cudaMalloc(&freqRxGpuPtr, rxBytes));

    fadingChan<Tcomplex> * fadeChanPtr = new fadingChan<Tcomplex>(freqTxGpuPtr, freqRxGpuPtr, cuMainStrm, fadingMode, randSeed, phyChanType);

    fadeChanPtr -> setup(inputFile, 1/*enableSwapTxRx*/);  // assuming UL

    /*------------------------ measure time ------------------------*/
    float elapsedTime;
    std::vector<float> elapsedTimeVec; // save run time per TTI
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float TTIlen = 0.0005f; // length of each TTI, assuming numerology 1
    float targetSNR = 10.0f; // target SNR in dB

    for(int TTIIdx=0; TTIIdx<nTti; TTIIdx++)
    {
        if(debugInd)
        {
            printf("Running TTI %d \n", TTIIdx);
        }
        cudaEventRecord(start, cuMainStrm);

        fadeChanPtr -> run(TTIlen * TTIIdx, targetSNR);

        cudaEventRecord(stop, cuMainStrm);
        CHECK_CUDAERROR(cudaEventSynchronize(stop));
        cudaStreamSynchronize(cuMainStrm);
        cudaEventElapsedTime(&elapsedTime, start, stop);
        elapsedTimeVec.push_back(elapsedTime);

        // optional test: to check SNR per antenna
        // report average SNR over all antennas & save SNRs to "SNR.txt" file during savefadingChanToH5File() if called;
        fadeChanPtr -> calSnr(13, 0, 240);
    }

    if(debugInd)
    {
        // print running time info
        printf("Total fading channel process cost: %f milisecond (avg over %ld runs) \n", std::reduce(elapsedTimeVec.begin(), elapsedTimeVec.end())/float(elapsedTimeVec.size()), elapsedTimeVec.size());

        for(auto x: elapsedTimeVec)
        {
        printf("%f, ", x);
        }
        printf("\n");
    }

    /*-----------------Save to Rx freq to file -------------------*/
    fadeChanPtr -> savefadingChanToH5File();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("Fail: CUDA error: %s\n", cudaGetErrorString(err));
    }
    else
    {
        printf("Success: fading channel runs without errors\n");
    }

    cudaFree(freqTxGpuPtr);
    cudaFree(freqRxGpuPtr);
    delete fadeChanPtr;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(cuMainStrm);
}

int main(int argc, char* argv[])
{      
    std::string inputFileName; // input TV name
    uint8_t fadingMode = 1; // fading mode: 0 for AWGN, 1 for TDL
    uint16_t randSeed = 0; // random seeds, used for TDL and generating noise
    int nTti = 10;  // number of TTIs
    uint8_t phyChanType = 1; // 0 - PUSCH, 1 - PUCCH, 2 - PRACH
    bool enableHalfPrecision = false; // true for use FP16, false for use FP32
    bool debugInd   = false; // true for debug info, false for no debug info
    // arguments parser
    // inputFilename fadingMode randSeed nTti phyChanType enableHalfPrecision debugInd
    int iArg = 1;
    // read options from CLI
    while(iArg < argc) {
      if('-' == argv[iArg][0]) {
        switch(argv[iArg][1]) {
            case 'i':
                if(++iArg >= argc)
                {
                    fprintf(stderr, "ERROR: Invalid input files \n");
                    exit(1);
                }
                inputFileName.assign(argv[iArg++]);
                break;      
            case 'm': // fading mode
                if((++iArg >= argc) || (1 != sscanf(argv[iArg], "%hhu", &fadingMode)) || (fadingMode < 0) || (fadingMode > 2))
                {
                fprintf(stderr, "ERROR: Invalid input fading mode.\n");
                exit(1);
                } 
                ++iArg;
                break;
            case 's': // random seed 
                if((++iArg >= argc) || (1 != sscanf(argv[iArg], "%hu", &randSeed)) || (randSeed < 0))
                {
                    fprintf(stderr, "ERROR: Invalid input random seed.\n");
                    exit(1);
                } 
                ++iArg;
                break;
            case 'n': // number of TTI
                if((++iArg >= argc) || (1 != sscanf(argv[iArg], "%u", &nTti)) || (nTti < 0))
                {
                fprintf(stderr, "ERROR: Invalid number of TTIs.\n");
                exit(1);
                } 
                ++iArg;
                break;
            case 'p': // enable half precision
                enableHalfPrecision = true;
                ++iArg;
                break;
            case 'd': // enable debug
                debugInd = true;
                ++iArg;
                break;
            case 't': // PHY channel type
                if((++iArg >= argc) || (1 != sscanf(argv[iArg], "%hhu", &phyChanType)) || (phyChanType < 0) || (phyChanType > 2))
                {
                fprintf(stderr, "ERROR: Invalid physical channel type.\n");
                exit(1);
                } 
                ++iArg;
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
        }
        else
        {
            fprintf(stderr, "ERROR: Invalid command line argument: %s\n", argv[iArg]);
            usage();
            exit(1);
        }
    }

    if(inputFileName.empty())
    {
        fprintf(stderr, "ERROR: No input TV was given \n");
        usage();
        exit(1);
    }
    // start testing based on TV, iter, and precision 
    std::string fadingModeStr = (fadingMode == 0 ? "AWGN" : (fadingMode == 1 ? "TDL" : "CDL"));
    if(enableHalfPrecision)
    {
        printf("FadingChan test: Using 16 bits precision, %s, random seed = %d, %d iterations \n", fadingModeStr.c_str(), randSeed, nTti);
        test_fadeChan<__half2>(inputFileName, nTti, fadingMode, randSeed, phyChanType, enableHalfPrecision, debugInd);        
    }
    else
    {
        printf("FadingChan test: Using 32 bits precision, %s, random seed = %d, %d iterations \n", fadingModeStr.c_str(), randSeed, nTti);
        test_fadeChan<cuComplex>(inputFileName, nTti, fadingMode, randSeed, phyChanType, enableHalfPrecision, debugInd);
    }

    return 0;
}
