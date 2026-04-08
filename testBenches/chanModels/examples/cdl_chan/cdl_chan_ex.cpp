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

#include "cdl_chan.cuh"
#include "chanModelsCommon.h"
#include <cassert>
#include <type_traits>
#include <thread>
#include "hdf5hpp.hpp"
#include <H5Cpp.h>

/**
 * @brief test CDL channel
 * 
 * @tparam Tscalar 
 * @tparam Tcomplex 
 * @param cdlCfg // CDL config struct
 * @param nTti // number of TTI to run
 * @param enableHalfPrecision half precision in cdl channel coe and processing tx signals
 * @param enableSwapTxRx 0: DL case; 1: UL case
 * @param debugInd define whether sample CDL channel data is printed out and saved to h5 file
 */
template<typename Tscalar, typename Tcomplex>
void test_CDL(cdlConfig_t * cdlCfg, uint32_t nTti, bool & enableHalfPrecision, uint8_t & enableSwapTxRx, bool & debugInd)
{
    cudaStream_t cuMainStrm;
    cudaStreamCreate(&cuMainStrm);

    curandGenerator_t curandGen = nullptr;
    uint32_t txSigSize = 0;

    if(cdlCfg -> sigLenPerAnt) // randomly generate test input signal
    {
        uint16_t nBsAnt = std::accumulate(cdlCfg -> bsAntSize.begin(), cdlCfg -> bsAntSize.end(), 1U, std::multiplies<uint32_t>());
        uint16_t nUeAnt = std::accumulate(cdlCfg -> ueAntSize.begin(), cdlCfg -> ueAntSize.end(), 1U, std::multiplies<uint32_t>());
        txSigSize = (cdlCfg -> nCell) * (cdlCfg -> nUe) * (cdlCfg -> sigLenPerAnt) * (enableSwapTxRx ? nUeAnt : nBsAnt);
        ASSERT(txSigSize <= 2147483647u, "size of tx time signal for all links must be smaller than 2147483647u");

        CHECK_CUDAERROR(cudaMalloc((void**) &(cdlCfg -> txSigIn), sizeof(Tcomplex) * txSigSize));
        CHECK_CURANDERROR(curandCreateGenerator(&curandGen, CURAND_RNG_PSEUDO_DEFAULT));
        CHECK_CURANDERROR(curandSetStream(curandGen, cuMainStrm));
        printf("CDL channel test: tx time signals use %.2f MB GPU memory. \n", sizeof(Tcomplex) * txSigSize / 1024.0f / 1024.0f);
    }
    else
    {
        cdlCfg -> txSigIn = nullptr;
    }
    
    /*---------------    Below tests CDL channel class       --------------------*/
    uint16_t randSeed = 0; // time(nullptr)
    cdlChan<Tscalar, Tcomplex> * cdlChanTest = new cdlChan<Tscalar, Tcomplex>(cdlCfg, randSeed, cuMainStrm);
    cdlChanTest -> printGpuMemUseMB();
    float ttiLen = 0.001 / (cdlCfg -> scSpacingHz / 15e3); // TTI length
    float intervalUs = ttiLen * 1e6 * 1e2;  // simulation interval between cdl runs, should be large than each cdl run

    // for time measurement
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    std::vector<float> elapsedTimeCudaEvtVec(nTti);
    std::vector<float> elapsedTimeCpuClockVec(nTti);

    if(cdlCfg -> sigLenPerAnt)
    {
        if constexpr (std::is_same_v<Tcomplex, cuComplex>) {
            CHECK_CURANDERROR(curandGenerateNormal(curandGen, (float*)(cdlCfg -> txSigIn), txSigSize * 2, 0.0f, sqrtf(0.5f)));
        } else {
            float* d_tmpBuf = nullptr;
            CHECK_CUDAERROR(cudaMalloc((void**)&d_tmpBuf, txSigSize * 2 * sizeof(float)));
            CHECK_CURANDERROR(curandGenerateNormal(curandGen, d_tmpBuf, txSigSize * 2, 0.0f, sqrtf(0.5f)));
            std::vector<float> h_floats(txSigSize * 2);
            CHECK_CUDAERROR(cudaMemcpy(h_floats.data(), d_tmpBuf, txSigSize * 2 * sizeof(float), cudaMemcpyDeviceToHost));
            cudaFree(d_tmpBuf);
            std::vector<__half> h_halves(txSigSize * 2);
            for (uint32_t i = 0; i < txSigSize * 2; ++i) {
                h_halves[i] = __float2half(h_floats[i]);
            }
            CHECK_CUDAERROR(cudaMemcpy(cdlCfg -> txSigIn, h_halves.data(), txSigSize * 2 * sizeof(__half), cudaMemcpyHostToDevice));
        }
    }

    // warm up GPU kernels
    cdlChanTest -> run(0, enableSwapTxRx); // use current TTI time as reference time

    for(int ttiIdx=0; ttiIdx<nTti; ttiIdx++)
    {
        // Get the current time
        auto startTime = std::chrono::high_resolution_clock::now();

        cudaEventRecord(start, cuMainStrm);
        cdlChanTest -> run(ttiIdx * ttiLen, enableSwapTxRx); // use current TTI time as reference time
        cudaEventRecord(stop, cuMainStrm);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTimeCudaEvtVec[ttiIdx], start, stop);

        if(debugInd)
        {
          printf("TTI Idx = %d \n", ttiIdx);
          /*---------------    Below are optional save to file       --------------------*/
          if(ttiIdx == 0 || ttiIdx == nTti/2 || ttiIdx == nTti-1)
          {
              std::string padFileNameEnding = "_TTI" + std::to_string(ttiIdx);
              cdlChanTest -> saveCdlChanToH5File(padFileNameEnding);
          }
        }

        // Calculate the elapsed time since the start of the loop
        auto endTime = std::chrono::high_resolution_clock::now();
        elapsedTimeCpuClockVec[ttiIdx] = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count();

        // Calculate the remaining time to sleep
        int remainingTimeToSleep = intervalUs - static_cast<int>(elapsedTimeCpuClockVec[ttiIdx]);

        // If the elapsed time is less than intervalUs, sleep for the remaining time
        if (remainingTimeToSleep > 0) {
            std::this_thread::sleep_for(std::chrono::microseconds(remainingTimeToSleep));
        }
    }

    printf("CDL channel test done:\n");
    
    // Calculate min, max, and average elapsed times (ignoring zero elements)
    float minTimeCudaEvt = std::numeric_limits<float>::max();
    float maxTimeCudaEvt = std::numeric_limits<float>::min();
    float totalTime = 0.0f;
    for (int itrIdx = 0; itrIdx < nTti; itrIdx++) 
    {
        minTimeCudaEvt = std::min(minTimeCudaEvt, elapsedTimeCudaEvtVec[itrIdx]);
        maxTimeCudaEvt = std::max(maxTimeCudaEvt, elapsedTimeCudaEvtVec[itrIdx]);
        totalTime += elapsedTimeCudaEvtVec[itrIdx];
    }
    float avgTimeCudaEvt = totalTime / nTti;
    printf("Timing in ms using cudaEvent (avg over %d iterations): avg %4.4f (min %4.4f, max %4.4f) \n", nTti, avgTimeCudaEvt, minTimeCudaEvt, maxTimeCudaEvt);

    float minTimeCpuClock = std::numeric_limits<float>::max();
    float maxTimeCpuClock = std::numeric_limits<float>::min();
    totalTime = 0.0f;
    for (int itrIdx = 0; itrIdx < nTti; itrIdx++) 
    {
        float tmpElapsedTimeCpuClock = elapsedTimeCpuClockVec[itrIdx] / 1000.0f;
        minTimeCpuClock = std::min(minTimeCpuClock, tmpElapsedTimeCpuClock);
        maxTimeCpuClock = std::max(maxTimeCpuClock, tmpElapsedTimeCpuClock);
        totalTime += tmpElapsedTimeCpuClock;
    }
    float avgTimeCpuClock = totalTime / nTti;

    printf("Timing in ms using CPU clock (avg over %d iterations): avg %4.4f (min %4.4f, max %4.4f) \n", nTti, avgTimeCpuClock, minTimeCpuClock, maxTimeCpuClock);

    if(debugInd)
    {
      /*---------------    Below are optinonal printout message       --------------------*/
      cdlChanTest -> printTimeChan();
      if(cdlCfg -> sigLenPerAnt)
      {
          cdlChanTest -> printSig();
      }
      
      if(cdlCfg -> runMode == 1)
      {
          cdlChanTest -> printFreqPrbgChan();
      }
      else if(cdlCfg -> runMode == 2)
      {
          cdlChanTest -> printFreqScChan();
          cdlChanTest -> printFreqPrbgChan();
      }
    }
    if(curandGen) curandDestroyGenerator(curandGen);
    cudaFree(cdlCfg -> txSigIn);
    delete cdlChanTest;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(cuMainStrm);
}

// printout usage message
void usage() {
    printf("CDL channel test with [Arguments]\n");
    printf("  Arguments:\n");
    printf("  -c  [number of cells (default 1)]\n");
    printf("  -u  [number of Ues  (default 1)]\n");
    printf("  -t  [tx antennas config (default 1 2 2 1 1)]\n");
    printf("  -r  [rx antennas config (default 2 2 1 1 1)]\n");
    printf("  -l  [number of time sample per antenna (default 8192)]\n");
    printf("  -n  [number of TTIs  (default 10)]\n");
    printf("  -m  [test mode, 0: time domain channel; \n");
    printf("                  1: time and frequency domain channel Prbg; \n");
    printf("                  2: time and frequency domain channel on Sc and Prbg; \n");
    printf("            Processing samples depends on -l (default 1)]\n");
    printf("  -o  [enable channel for 14 OFDM symbols, one channel per OFDM symbol (default disable)]\n");
    printf("  -s  [enable PDP in 38.901 (default 38.141)]\n");
    printf("  -f  [type of convert CFR on SC to PRBG, 0: use first SC; 1: use middle SC; 2: using last SC: 3: using average over SCs (default 1)]\n");
    printf("  -w  [5G numerology 0~3, (default 1)]\n");
    printf("  -g  [sc sampling rate when average over CFR on sc to prbg (default 1)]\n");
    printf("  -p  [enable half precision (default disable)]\n");
    printf("  -x  [choose processing tx sample in time (0) or frequency domain (1) (default 0)]\n");
    printf("  -d  [enable debug to print sample channel and save CDL coes in H5 file (default disable)]\n");
    printf("  -s  [enable swap tx/rx in processing t (default disable)]\n");
    printf("  -h  display usage information \n");
    printf("Example (1 cells, 1 UEs, 4T4R, 4096 samples per antenna, 100 TTIs, time/freq channel): './cdl_chan_ex -l 4096 -n 100 -m 1'\n");
    printf("Example (9 cells, 450 UEs, 4T4R, 0 samples per antenna, 100 TTIs, time/freq channel): './cdl_chan_ex -c 9 -u 450 -l 4096 -n 100 -m 1'\n");
}

int main(int argc, char* argv[])
{
    int iArg = 1;

    // parameters can be changed
    uint16_t nCell   = 1;
    uint16_t nUe     = 1;
    std::vector<uint16_t> bsAntSize = {1,1,1,2,2};  // {M_g, N_g, M, N, P}
    uint16_t nBsAnt  = 4;
    std::vector<uint16_t> ueAntSize = {1,1,2,2,1};  // {M_g, N_g, M, N, P}
    uint16_t nUeAnt  = 4;
    uint32_t sigLenPerAnt = 8192;
    uint32_t nTti    = 10;
    uint8_t  runMode = 1;
    uint8_t  enableOfdmSymChan = false;
    uint8_t  freqConvertType = 1;
    uint8_t  mu = 1;
    uint8_t  scSampling = 1;
    uint8_t  procSigFreq = 0;
    bool enableHalfPrecision = false;
    bool debugInd    = false;
    uint8_t enableSwapTxRx = 0;
    // read options from CLI
    while(iArg < argc) {
      if('-' == argv[iArg][0]) {
        switch(argv[iArg][1]) {
          case 'c': // number of cells
            if((++iArg >= argc) || (1 != sscanf(argv[iArg], "%hu", &nCell)) || (nCell < 0))
            {
              fprintf(stderr, "ERROR: Invalid number of cells.\n");
              exit(1);
            } 
            ++iArg;
            break;          
          case 'u': // number of UEs
            if((++iArg >= argc) || (1 != sscanf(argv[iArg], "%hu", &nUe)) || (nUe < 0))
            {
              fprintf(stderr, "ERROR: Invalid number of UEs.\n");
              exit(1);
            } 
            ++iArg;
            break;
          case 't': // bs antenna settings
            // Parse up to 5 values following -t
            for (int count = 0; count < 5 && (iArg+1 < argc); ++count) 
            {
              uint16_t value;
              if (1 != sscanf(argv[iArg+1], "%hu", &value) || value < 0) {
                  break;
              }
              bsAntSize[count] = value;
              ++iArg;
            }
            ++iArg;
            break;
          case 'r': // ue antenna settings
            // Parse up to 5 values following -r
            for (int count = 0; count < 5 && (iArg+1 < argc); ++count)
            {
              uint16_t value;
              if (1 != sscanf(argv[iArg+1], "%hu", &value) || value < 0) {
                  break;
              }
              ueAntSize[count] = value;
              ++iArg;
            }
            ++iArg;
            break;
          case 'l': // input length of time domain samples per antenna
            if((++iArg >= argc) || (1 != sscanf(argv[iArg], "%u", &sigLenPerAnt)) || (sigLenPerAnt < 0) )
            {
              fprintf(stderr, "ERROR: Invalid length of time domain samples per antenna.\n");
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
          case 'm': // runMode of CDL
            if((++iArg >= argc) || (1 != sscanf(argv[iArg], "%hhu", &runMode)) || (runMode < 0) || (runMode > 2))
            {
              fprintf(stderr, "ERROR: Invalid run mode.\n");
              exit(1);
            } 
            ++iArg;
            break;
          case 'f': // choose SC to PRBG CFR conversion type
            if((++iArg >= argc) || (1 != sscanf(argv[iArg], "%hhu", &freqConvertType)) || (freqConvertType < 0) || (freqConvertType > 4))
            {
              fprintf(stderr, "ERROR: Invalid conversion type from SC to PRBG CFR.\n");
              exit(1);
            } 
            ++iArg;
            break;
          case 'w': // 5G numerology mu
            if((++iArg >= argc) || (1 != sscanf(argv[iArg], "%hhu", &mu)) || (mu < 0) || (mu > 3))
            {
              fprintf(stderr, "ERROR: Invalid 5G numerology.\n");
              exit(1);
            } 
            ++iArg;
            break;
          case 'g': // sc sampling rate
            if((++iArg >= argc) || (1 != sscanf(argv[iArg], "%hhu", &scSampling)) || (scSampling < 1))
            {
              fprintf(stderr, "ERROR: Invalid Sc sampling rate.\n");
              exit(1);
            } 
            ++iArg;
            break;
          case 'p': // enable half precision
            enableHalfPrecision = true;
            ++iArg;
            break;
          case 's': // enable swap tx rx, DL by default
            enableSwapTxRx = true;
            ++iArg;
            break;
          case 'x': // process tx samples in time or freq domain
            if((++iArg >= argc) || (1 != sscanf(argv[iArg], "%hhu", &procSigFreq)) || (procSigFreq < 0) || (procSigFreq > 2))
            {
              fprintf(stderr, "ERROR: Invalid tx sample processing mode.\n");
              exit(1);
            } 
            ++iArg;
            break;
          case 'o': // enable enable channel for 14 OFDM symbols, one channel per OFDM symbol
            enableOfdmSymChan = true;
            ++iArg;
            break;
          case 'd': // enable debug
            debugInd = true;
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
      else {
        fprintf(stderr, "ERROR: Invalid command line argument: %s\n", argv[iArg]);
        exit(1);
      }
    }
    nBsAnt = std::accumulate(bsAntSize.begin(), bsAntSize.end(), 1U, std::multiplies<uint32_t>());
    nUeAnt = std::accumulate(ueAntSize.begin(), ueAntSize.end(), 1U, std::multiplies<uint32_t>());

    printf("CDL channel test: %d cells, %d UEs, %d ([%d, %d, %d, %d, %d]) nBsAnt, %d ([%d, %d, %d, %d, %d]) nUeAnt. \n", nCell, nUe, nBsAnt, bsAntSize[0], bsAntSize[1], bsAntSize[2], bsAntSize[3], bsAntSize[4], nUeAnt, ueAntSize[0], ueAntSize[1], ueAntSize[2], ueAntSize[3], ueAntSize[4]);
    switch(runMode)
    {
        case 0:
            if(sigLenPerAnt)
            {
                printf("CDL channel test: time channel coes and process signals with %d samples per antenna.\n", sigLenPerAnt);
            }
            else
            {
                printf("CDL channel test: time channel coes.\n");
            }
            break;
        case 1:
            if(sigLenPerAnt)
            {
                printf("CDL channel test: runMode %d, freqConvertType %d, time channel coes, frequency channel coes on Prbg, and process signals with %d samples per antenna.\n", runMode, freqConvertType, sigLenPerAnt);
            }
            else
            {
                printf("CDL channel test: runMode %d, freqConvertType %d, time channel coes and frequency channel coes on Prbg.\n", runMode, freqConvertType);
            }
            break;
        case 2:
            if(sigLenPerAnt)
            {
                printf("CDL channel test: runMode %d, freqConvertType %d, time channel coes, frequency channel coes on Sc and Prbg, and process signals with %d samples per antenna.\n", runMode, freqConvertType, sigLenPerAnt);
            }
            else
            {
                printf("CDL channel test: runMode %d, freqConvertType %d, time channel coes and frequency channel coes on Sc and Prbg.\n", runMode, freqConvertType);
            }
            break;
        default:
            fprintf(stderr, "ERROR: unsupported CDL channel test mode!\n");
            exit(1);
    }
    
    printf("CDL channel test: using numerology %d.\n", mu);

    if(enableHalfPrecision)
    {
      printf("CDL channel test: enable half precision.\n");
    }
    else
    {
      printf("CDL channel test: full precision.\n");
    }

    if (!procSigFreq)
    {
      printf("CDL channel test: process tx samples in time domain.\n");
    }
    else
    {
      printf("CDL channel test: process tx samples in frequency domain.\n");
    }

    if (enableSwapTxRx)
    {
      printf("CDL channel test: enable swap tx rx.\n");
    }
    else
    {
      printf("CDL channel test: disable swap tx rx.\n");
    }
    
    cdlConfig_t * cdlCfg = new cdlConfig_t;
    // change default parameters if needed
    // cdlCfg -> delayProfile = 'C';
    // cdlCfg -> delaySpread = 300;
    // cdlCfg -> maxDopplerShift = 100;
    cdlCfg -> scSpacingHz = 15e3 * pow(2, mu);
    cdlCfg -> f_samp = 4096 * 15e3 * pow(2, mu);
    cdlCfg -> nCell  = nCell;
    cdlCfg -> nUe    = nUe;
    cdlCfg -> bsAntSize = bsAntSize;
    if (bsAntSize[2] == 1)
    {
      cdlCfg -> bsAntPolarAngles = {0, 90};
      cdlCfg -> bsAntPattern = 0;
      printf("CDL channel test: BS uses isotropic antenna. \n");
    }
    else if (bsAntSize[2] == 2)
    {
      cdlCfg -> bsAntPolarAngles = {45, -45};
      cdlCfg -> bsAntPattern = 1;
      printf("CDL channel test: BS uses polarized antenna. \n");
    }
    else
    {
      fprintf(stderr, "ERROR: unsupported bsAntSize!\n");
    }
    cdlCfg -> ueAntSize = ueAntSize;
    if (ueAntSize[2] == 1)
    {
      cdlCfg -> ueAntPolarAngles = {0, 90};
      cdlCfg -> ueAntPattern = 0;
      printf("CDL channel test: UE uses isotropic antenna. \n");
    }
    else if (ueAntSize[2] == 2)
    {
      cdlCfg -> ueAntPolarAngles = {45, -45};
      cdlCfg -> ueAntPattern = 1;
      printf("CDL channel test: UE uses polarized antenna. \n");
    }
    else
    {
      fprintf(stderr, "ERROR: unsupported ueAntSize!\n");
    }
    // cdlCfg -> fBatch = 15e3;
    // cdlCfg -> numPath = 48;
    // cdlCfg -> cfoHz = 200.0f;
    // cdlCfg ->  delay = 0.0f;
    cdlCfg -> sigLenPerAnt = sigLenPerAnt; // signal length per antenna
    cdlCfg -> runMode = runMode; // set to 1 or 2 will generate freq channel, see cdl_chan.cuh for details
    cdlCfg -> procSigFreq = procSigFreq; // process tx samples in time or freq domain
    cdlCfg -> freqConvertType = freqConvertType; // conversion type from SC to PRBG CFR, see cdl_chan.cuh for details
    cdlCfg -> scSampling = scSampling; // sc sampling rate, see cdl_chan.cuh for details

    // enable channel for 14 OFDM symbols, one channel per OFDM symbol
    // assign the batch size manually, see cdl_chan.cuh for details
    if (enableOfdmSymChan)
    {
      // 14 OFDM symbols for 14 batches, total 61440 samples
      std::vector<uint32_t> batchLen = {4448};  // First element
      batchLen.resize(14, 4384);  // Resize to 14 elements, filling remaining with 4384, total length = 4448 + 4384 * 13 = 61440
      cdlCfg -> batchLen = batchLen;
    }

    // run testing of CDL
    if(enableHalfPrecision)
    {
      test_CDL<__half, __half2>(cdlCfg, nTti, enableHalfPrecision, enableSwapTxRx, debugInd); 
    }
    else
    {
      test_CDL<float, cuComplex>(cdlCfg, nTti, enableHalfPrecision, enableSwapTxRx, debugInd); 
    }

    delete cdlCfg;

    return 0;
}