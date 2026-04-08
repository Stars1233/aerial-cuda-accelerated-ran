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

#include "tvLoadingTest.h"

// run cumac subcontexts
void runCumacIter(cumac::cumacSubcontext * subcontext, int nItr, int intervalUs, uint8_t* runSlotPattern, bool printTimeStamp, cudaStream_t strm)
{
    // warm-up
    subcontext->run(strm);
    CUDA_CHECK_ERR(cudaStreamSynchronize(strm));

    // for time measurement
    cudaEvent_t start, stop;
    CUDA_CHECK_ERR(cudaEventCreate(&start));
    CUDA_CHECK_ERR(cudaEventCreate(&stop));
    std::vector<float> elapsedTimeCudaEvtVec(nItr);
    std::vector<float> elapsedTimeCpuClockVec(nItr);
    
    { // print start time stamp
      // Get current time with millisecond and microsecond precision
      auto currentTime = std::chrono::system_clock::now();
      auto timeSinceEpoch = currentTime.time_since_epoch();
      auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(timeSinceEpoch);
      auto microseconds = std::chrono::duration_cast<std::chrono::microseconds>(timeSinceEpoch - milliseconds);

      // Convert milliseconds to time_t
      auto currentTime_t = std::chrono::system_clock::to_time_t(currentTime);
      auto currentTime_tm = *std::localtime(&currentTime_t);

      // Print current time with millisecond precision
      std::cout << "Iteration starts at: "
                << std::put_time(&currentTime_tm, "%Y-%m-%d %H:%M:%S") << " ms:"
                << std::setfill('0') << std::setw(3) << milliseconds.count() % 1000
                << std::setw(0) << " us:"
                << std::setfill('0') << std::setw(3) << microseconds.count() % 1000000
                << std::endl;
    }
    // Run the function nItr times with a delay of intervalUs microseconds between each call
    for (int itrIdx = 0; itrIdx < nItr; itrIdx++) 
    {
        // Get the current time
        auto startTime = std::chrono::high_resolution_clock::now();

        if(printTimeStamp)
        {
            // Get current time with millisecond and microsecond precision
            auto currentTime = std::chrono::system_clock::now();
            auto timeSinceEpoch = currentTime.time_since_epoch();
            auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(timeSinceEpoch);
            auto microseconds = std::chrono::duration_cast<std::chrono::microseconds>(timeSinceEpoch - milliseconds);

            // Convert milliseconds to time_t
            auto currentTime_t = std::chrono::system_clock::to_time_t(currentTime);
            auto currentTime_tm = *std::localtime(&currentTime_t);

            // Print current time with millisecond precision
            std::cout << "Iteration " << itrIdx << ' '
                    << "Current time is "
                    << std::put_time(&currentTime_tm, "%Y-%m-%d %H:%M:%S") << " ms:"
                    << std::setfill('0') << std::setw(3) << milliseconds.count() % 1000
                    << std::setw(0) << " us:"
                    << std::setfill('0') << std::setw(3) << microseconds.count() % 1000000
                    << std::endl;
        }

        // run cuMAC
        if(runSlotPattern[itrIdx % 10] == 1) // run scheduler based on runSlotPattern
        {
          CUDA_CHECK_ERR(cudaEventRecord(start, strm));
          subcontext->run(strm);
          CUDA_CHECK_ERR(cudaEventRecord(stop, strm));
          CUDA_CHECK_ERR(cudaEventSynchronize(stop));
          CUDA_CHECK_ERR(cudaEventElapsedTime(&elapsedTimeCudaEvtVec[itrIdx], start, stop));
        }

        // Calculate the elapsed time since the start of the loop
        auto endTime = std::chrono::high_resolution_clock::now();
        elapsedTimeCpuClockVec[itrIdx] = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count();

        // Calculate the remaining time to sleep
        int remainingTimeToSleep = intervalUs - static_cast<int>(elapsedTimeCpuClockVec[itrIdx]);

        // If the elapsed time is less than intervalUs, sleep for the remaining time
        if (remainingTimeToSleep > 0) {
            std::this_thread::sleep_for(std::chrono::microseconds(remainingTimeToSleep));
        }
    }

    { // print end time stamp
      // Get current time with millisecond and microsecond precision
      auto currentTime = std::chrono::system_clock::now();
      auto timeSinceEpoch = currentTime.time_since_epoch();
      auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(timeSinceEpoch);
      auto microseconds = std::chrono::duration_cast<std::chrono::microseconds>(timeSinceEpoch - milliseconds);

      // Convert milliseconds to time_t
      auto currentTime_t = std::chrono::system_clock::to_time_t(currentTime);
      auto currentTime_tm = *std::localtime(&currentTime_t);

      // Print current time with millisecond precision
      std::cout << "Iteration ends at: "
                << std::put_time(&currentTime_tm, "%Y-%m-%d %H:%M:%S") << " ms:"
                << std::setfill('0') << std::setw(3) << milliseconds.count() % 1000
                << std::setw(0) << " us:"
                << std::setfill('0') << std::setw(3) << microseconds.count() % 1000000
                << std::endl;
    }

    // Calculate min, max, and average elapsed times (ignoring zero elements)
    float minTimeCudaEvt = std::numeric_limits<float>::max();
    float maxTimeCudaEvt = std::numeric_limits<float>::min();
    float totalTime = 0.0f;
    int count = 0;
    for (int itrIdx = 0; itrIdx < nItr; itrIdx++) 
    {
        if (runSlotPattern[itrIdx % 10] == 1) 
        {
            minTimeCudaEvt = std::min(minTimeCudaEvt, elapsedTimeCudaEvtVec[itrIdx]);
            maxTimeCudaEvt = std::max(maxTimeCudaEvt, elapsedTimeCudaEvtVec[itrIdx]);
            totalTime += elapsedTimeCudaEvtVec[itrIdx];
            ++count;
        }
    }
    float avgTimeCudaEvt = (count > 0) ? (totalTime / count) : 0.0f;
    printf("Timing in ms using cudaEvent (avg over %d iterations): avg %4.4f (min %4.4f, max %4.4f) \n", count, avgTimeCudaEvt, minTimeCudaEvt, maxTimeCudaEvt);
    saveVectorToCSV<float>(elapsedTimeCudaEvtVec, "elapsedTimeCudaEvtVec.csv");

    float minTimeCpuClock = std::numeric_limits<float>::max();
    float maxTimeCpuClock = std::numeric_limits<float>::min();
    totalTime = 0.0f;
    count = 0;
    for (int itrIdx = 0; itrIdx < nItr; itrIdx++) 
    {
        if (runSlotPattern[itrIdx % 10] == 1) 
        {
            float tmpElapsedTimeCpuClock = elapsedTimeCpuClockVec[itrIdx] / 1000.0f;
            minTimeCpuClock = std::min(minTimeCpuClock, tmpElapsedTimeCpuClock);
            maxTimeCpuClock = std::max(maxTimeCpuClock, tmpElapsedTimeCpuClock);
            totalTime += tmpElapsedTimeCpuClock;
            ++count;
        }
    }
    float avgTimeCpuClock = (count > 0) ? (totalTime / count) : 0.0f;

    printf("Timing in ms using CPU clock (avg over %d iterations): avg %4.4f (min %4.4f, max %4.4f) \n", count, avgTimeCpuClock, minTimeCpuClock, maxTimeCpuClock);
}

/////////////////////////////////////////////////////////////////////////
void schdModuleInd() {
    printf("\nSize of the indication string after '-m' should be equal to 4;\n");
    printf("Each entry/digit of the string can be either 0 or 1\n");
    printf("Entry 0 is for UE selection:       0 - not being called, 1 - being called\n");
    printf("Entry 1 is for PRG allocation:     0 - not being called, 1 - being called\n");
    printf("Entry 2 is for layer selection:    0 - not being called, 1 - being called\n");
    printf("Entry 3 is for MCS selection:      0 - not being called, 1 - being called\n");
    printf("Examples: 0100 - only call PRG allocation; 1111 - call all scheduler modules\n");
    printf("0000 is invalid because no scheduler module is being called\n");
}

// usage()
void usage() {
    printf("cuMAC TV loading test with [Arguments]\n");
    printf("  Arguments:\n");
    printf("  -i  [cuMAC_HDF5_TV_file]\n");
    printf("  -m  [string to indicate scheduler modules being called]\n");
    printf("  -g  [GPU/CPU scheduler indication: 0 - CPU, 1 - GPU, 2 -both with comparison (default 1)]\n");
    printf("  -c  [heterogeneous UE selection config. across cells indication: 0 - disabled, 1 - enabled (default 0)]\n");
    printf("  -d  [DL/UL indication: 0 - UL, 1 - DL (default 1)]\n");
    printf("  -p  [Indication for using FP16 PRG allocation kernel: 0 - FP32, 1 - FP16 (default 0)]\n");
    printf("  -q  [Indication for using CQI-based MCS selection: 0 - not to use CQI, 1 - use CQI (default 0)]\n");
    printf("  -l  [Indication for using light-weight PRG allocation kernel: 0 - use heavy-weight kernel, 1 - use light-weight kernel (default 0)]\n");
    printf("  -u  [baseline UL MCS selection indication: 0 - multi-cell UL MCS selection, 1 - baseline single-cell UL MCS selection (default 0)]\n");
    printf("  -a  [Aerial Sim indication: 0 - not for Aerial Sim, 1 - for Aerial Sim (default 0)]\n");
    printf("  -R  [Indication for using RI-based layer selection: 0 - not tu use RI, 1 - use RI (default 0)]\n");
    printf("  -s  [Choose scheduling algorithm: 0 - round-robin, 1 - proportional fair (default 1)]\n");
    printf("  -n  [Number of iterations for running cuMAC scheduler] (default 1)\n");
    printf("  -t  [Enable TDD pattern, 0: full slot; 1: DDDSUUDDDD; 2: DDDSU; only run scheduler in U and D slots based on UL or DL (default 0)]\n");
    printf("  -v  [Verbose: print time stamp of each cuMAC run] (default: disable)\n");
    printf("  -r  [enable/disable HARQ re-transmission support: 0 - HARQ disabled, 1 - HARQ enabled (default 0)]\n");
    printf("Example (only call GPU DL PRG allocation module with a TV, not for Aerial Sim): './tvLoadingTest -i [cuMAC_HDF5_TV_file] -g 1 -d 1 -m 00100 -a 0 -u 0'\n");
    schdModuleInd();
}

int main(int argc, char* argv[]) 
{
    int iArg = 1;
    std::string inputFileName = std::string();
    
    const uint8_t numSchedulerModules = 4;
    std::string modulesCalledStr = std::string();
    uint8_t modulesCalled[numSchedulerModules] = {0, 0, 0, 0};

    // indicator for DL/UL
    uint8_t DL = 1;

    // indicator for GPU/CPU scheduler
    uint8_t GPU = 1;

    // indicator for heterogeneous UE selection config. across cells
    uint8_t heteroUeSelCells = 0;

    // indicator for baseline UL MCS selection
    uint8_t baselineUlMcs = 0;

    // indicator for Aerial Sim
    uint Asim = 0;

    // indicator for using FP16 PRG allocation kernel
    uint8_t halfPrecision = 0;

    // indicator for using light-weight PRG allocation kernel
    uint8_t lightWeight = 0;

    // choose scheduling algorithm from round-robin or proportional fair
    uint8_t schAlg = 1;

    // number of iterations
    int nItr = 1;

    // indicator for whether using TDD pattern fullSlot, DDDSUUDDDD
    uint8_t  tddPatternIdx  = 0;

    // whether print out timestamp
    bool printTimeStamp = false;

    // HARQ enabled/disabled
    uint8_t enableHarq = 0;

    // whether to use CQI-based MCS selection or not
    uint8_t mcsCqi = 0;

    // whether to use RI-based layer selection or not
    uint8_t layerRi = 0;

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
          case 'm': // indicator of scheduler modules being called
            if(++iArg >= argc) {
              fprintf(stderr, "ERROR: No scheduler module indicator given.\n");
              exit(1);
            } else {
              modulesCalledStr.assign(argv[iArg++]);
            }
            break;
          case 'p': // indicator for using FP16 PRG allocation kernel
            if((++iArg >= argc) || (1 != sscanf(argv[iArg], "%hhu", &halfPrecision)) || ((halfPrecision < 0) || (halfPrecision > 1)))
            {
              fprintf(stderr, "ERROR: Invalid using FP16 PRG allocation kernel.\n");
              exit(1);
            } 
            ++iArg;
            break;
          case 'q': // indicator for using CQI-based MCS selection
            if((++iArg >= argc) || (1 != sscanf(argv[iArg], "%hhu", &mcsCqi)) || ((mcsCqi < 0) || (mcsCqi > 1)))
            {
              fprintf(stderr, "ERROR: Invalid CQI MCS selection indicator.\n");
              exit(1);
            }    
            ++iArg;
            break;     
          case 'l': // indicator for using light-weight PRG allocation kernel
            if((++iArg >= argc) || (1 != sscanf(argv[iArg], "%hhu", &lightWeight)) || ((lightWeight < 0) || (lightWeight > 1)))
            {
              fprintf(stderr, "ERROR: Invalid light-weight PRG allocation kernel indicator.\n");
              exit(1);
            } 
            ++iArg;
            break;
          case 'a': // indicator for Aerial Sim
            if((++iArg >= argc) || (1 != sscanf(argv[iArg], "%u", &Asim)) || ((Asim < 0) || (Asim > 1)))
            {
              fprintf(stderr, "ERROR: Invalid Aerial Sim indicator.\n");
              exit(1);
            } 
            ++iArg;
            break;
          case 'g': // indicator of GPU/CPU
            if((++iArg >= argc) || (1 != sscanf(argv[iArg], "%hhu", &GPU)) || ((GPU < 0) || (GPU > 2)))
            {
              fprintf(stderr, "ERROR: Invalid GPU/CPU scheduler indicator.\n");
              exit(1);
            } 
            ++iArg;
            break;
          case 'c': // indicator for heterogeneous UE selection
            if((++iArg >= argc) || (1 != sscanf(argv[iArg], "%hhu", &heteroUeSelCells)) || ((heteroUeSelCells < 0) || (heteroUeSelCells > 1)))
            {
              fprintf(stderr, "ERROR: Invalid heterogeneous UE selection indicator.\n");
              exit(1);
            }
            ++iArg;
            break;
          case 'u': // indicator of baseline UL MCS selection
            if((++iArg >= argc) || (1 != sscanf(argv[iArg], "%hhu", &baselineUlMcs)) || ((baselineUlMcs < 0) || (baselineUlMcs > 1)))
            {
              fprintf(stderr, "ERROR: Invalid baseline UL MCS selection.\n");
              exit(1);
            } 
            ++iArg;
            break;
          case 'd': // indicator of DL/UL
            if((++iArg >= argc) || (1 != sscanf(argv[iArg], "%hhu", &DL)) || ((DL < 0) || (DL > 1)))
            {
              fprintf(stderr, "ERROR: Invalid DL/UL indicator.\n");
              exit(1);
            } 
            ++iArg;
            break;
          case 'R': // indicator for using RI-based layer selection
            if((++iArg >= argc) || (1 != sscanf(argv[iArg], "%hhu", &layerRi)) || ((layerRi < 0) || (layerRi > 1)))
            {
              fprintf(stderr, "ERROR: Invalid RI layer selection indicator.\n");
              exit(1);
            }
            ++iArg;
            break;
          case 's': // choose scheduling algorithm: round-robin or proportional fair
            if((++iArg >= argc) || (1 != sscanf(argv[iArg], "%hhu", &schAlg)) || ((schAlg < 0) || (schAlg > 1)))
            {
              fprintf(stderr, "ERROR: Invalid choice of scheduling algorithm.\n");
              exit(1);
            } 
            ++iArg;
            break;
          case 'n': // number of iterations
            if((++iArg >= argc) || (1 != sscanf(argv[iArg], "%i", &nItr)) || ((nItr <= 0)))
            {
              fprintf(stderr, "ERROR: Invalid number of iterations for cuMAC.\n");
              exit(1);
            } 
            ++iArg;
            break;
          case 't': // enable TDD pattern
            if((++iArg >= argc) || (1 != sscanf(argv[iArg], "%hhu", &tddPatternIdx)) || ((tddPatternIdx < 0) || (tddPatternIdx > 2)))
            {
              fprintf(stderr, "ERROR: Invalid TDD pattern for cuMAC.\n");
              exit(1);
            } 
            ++iArg;
            break;
          case 'r': // enable/disable HARQ re-transmission
            if((++iArg >= argc) || (1 != sscanf(argv[iArg], "%hhu", &enableHarq )) || (enableHarq  > 1))
            {
              fprintf(stderr, "ERROR: Invalid HARQ enable(0)/disable(1) only.\n");
              exit(1);
            }
            ++iArg;
            break;
          case 'v': // verbose, print time stamp of each run
            printTimeStamp = true;
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

    // unpack running modules
    if (modulesCalledStr.size() != numSchedulerModules) {
        printf("ERROR: format of scheduler module indication is incorrect:\n");
        schdModuleInd();
        return 0;
    }

    for (int i = 0; i < numSchedulerModules; i++) {
      modulesCalled[i] = static_cast<uint8_t>(modulesCalledStr[i] - '0');
    }

    // CPU and/or GPU schedulers
    if (GPU == 1) {
        printf("cuMAC TV loading test: GPU scheduler\n");
    } else if (GPU == 0) {
        printf("cuMAC TV loading test: CPU scheduler\n");
    } else if (GPU == 2) {
        printf("cuMAC TV loading test: both GPU and GPU schedulers, only run 1 iteration\n");
    } else {
        printf("Error: -g argument value not supported\n");
        return 0;
    }

    // heterogeneous or homogeneous UE selection config. across cells
    if (heteroUeSelCells == 1) {
        printf("cuMAC TV loading test: heterogeneous UE selection config. across cells\n");
    } else {
        printf("cuMAC TV loading test: homogeneous UE selection config. across cells\n");
    }

    // UL or DL
    if (DL == 1) {
        printf("cuMAC TV loading test: Downlink\n");
    } else {
        printf("cuMAC TV loading test: Uplink\n");
    }

    // precision
    if (halfPrecision == 1) {
        printf("cuMAC TV loading test: FP16 half-precision kernels\n");
    } else {
        printf("cuMAC TV loading test: FP32 kernels\n");
    }

    // light-weight kernel
    if (lightWeight == 1) {
        printf("cuMAC TV loading test: light-weight kernels\n");
    } else {
        printf("cuMAC TV loading test: non-light-weight kernels\n");
    }
    
    if (DL == 0) { // only for UL
      if (baselineUlMcs == 1) {
        printf("cuMAC TV loading test: using baseline single-cell UL MCS selection\n");
      } else {
        printf("cuMAC TV loading test: using multi-cell UL MCS selection\n");
      }
    }
    
    // RR or PF
    if(schAlg == 0)
    {
      printf("cuMAC TV loading test: round-robin scheduling\n");
    }
    else
    {
      printf("cuMAC TV loading test: Proportional fair scheduling\n");
    }

    if (mcsCqi == 0) {
      printf("cuMAC TV loading test: non-CQI-based MCS selection\n");
    } else {
      printf("cuMAC TV loading test: UE reported CQI-based MCS selection\n");
    }

    if (layerRi == 0) {
      printf("cuMAC TV loading test: non-RI-based layer selection\n");
    } else {
      printf("cuMAC TV loading test: UE reported RI-based layer selection\n");
    }

    if (enableHarq == 1) {
        printf("cuMAC TV loading test: HARQ enabled\n");
    } else {
        printf("cuMAC TV loading test: HARQ disabled\n");
    }
    
    if (Asim == 1) {
        printf("cuMAC TV loading test: TV generated from Aerial Sim\n");
    } else {
        printf("cuMAC TV loading test: TV generated from cuMAC\n");
    }

    if (Asim == 1 && GPU == 0) {
        printf("ERROR: For Aerial Sim, CPU scheduler is not supported yet\n");
        return 0;
    }

    if (Asim == 1 && (lightWeight == 1 || halfPrecision == 1)) {
        printf("ERROR: For Aerial Sim, light-weight or half-precision kernels are not supported yet\n");
        return 0;
    }

    // get runSlotPattern from TDD pattern
    uint8_t* runSlotPattern = nullptr;
    getTddRunSlotPattern(DL, tddPatternIdx, runSlotPattern);

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
    printf("cuMAC TV loading test: (if calling GPU scheduler) Running on GPU device %d (total devices: %d)\n", 
           my_dev, deviceCount);

    // setup randomness seed
    srand(seedConst);

    // create stream
    cudaStream_t cuStrmMain;
    CUDA_CHECK_ERR(cudaStreamCreate(&cuStrmMain));

    if (GPU != 2) {
        cumacSubcontext* subcontext = new cumacSubcontext(inputFileName, GPU, halfPrecision, layerRi, Asim, heteroUeSelCells, schAlg, modulesCalled, cuStrmMain);
        CUDA_CHECK_ERR(cudaStreamSynchronize(cuStrmMain));

        subcontext->setup(inputFileName, lightWeight, 2.0, cuStrmMain);
        CUDA_CHECK_ERR(cudaStreamSynchronize(cuStrmMain));

        // sleep for 1 second
        std::this_thread::sleep_for(std::chrono::seconds(1));

        // run cuMAC: hardcode for 500 us per slot
        runCumacIter(subcontext, nItr, 500, runSlotPattern, printTimeStamp, cuStrmMain);

        //subcontext->debugLog();
        if (Asim == 1) { // for Aerial Sim
            saveToH5_Asim("asimResultsTv.h5",
                          subcontext->cellGrpUeStatusGpu.get(),
                          subcontext->cellGrpPrmsGpu.get(),
                          subcontext->schdSolGpu.get());
        } else {
          if (GPU == 1) { // GPU
            saveToH5("gpuResultsTv.h5",
                    subcontext->cellGrpUeStatusGpu.get(),
                    subcontext->cellGrpPrmsGpu.get(),
                    subcontext->schdSolGpu.get());
          } else { // CPU
            saveToH5_CPU("cpuResultsTv.h5",
                        subcontext->cellGrpUeStatusCpu.get(),
                        subcontext->cellGrpPrmsCpu.get(),
                        subcontext->schdSolCpu.get());
          }
        }

        delete subcontext;

        return 0;
    } else {
        assert(layerRi == 0); // CQI-based MCS selection and RI-based layer selection are not supported for CPU scheduler yet

        cumacSubcontext* subcontextGpu = new cumacSubcontext(inputFileName, 1, halfPrecision, layerRi, Asim, heteroUeSelCells, schAlg, modulesCalled, cuStrmMain);
        CUDA_CHECK_ERR(cudaStreamSynchronize(cuStrmMain));

        subcontextGpu->setup(inputFileName, lightWeight, 2.0, cuStrmMain);
        CUDA_CHECK_ERR(cudaStreamSynchronize(cuStrmMain));

        subcontextGpu->run(cuStrmMain);

        cumacSubcontext* subcontextCpu = new cumacSubcontext(inputFileName, 0, halfPrecision, layerRi, Asim, heteroUeSelCells, schAlg, modulesCalled, cuStrmMain);

        subcontextCpu->setup(inputFileName, lightWeight, 2.0, cuStrmMain);

        subcontextCpu->run(cuStrmMain);
        
        bool pass = compareCpuGpuAllocSol(subcontextGpu, subcontextCpu, modulesCalled);

        if (pass) {
            printf("CPU and GPU scheduler solutions check result: PASS\n");
        } else {
            printf("CPU and GPU scheduler solutions check result: FAIL\n");
        }

        delete subcontextGpu;
        delete subcontextCpu;

        return !pass;
    }
}

inline void getTddRunSlotPattern(uint8_t & DL, uint8_t & tddPatternIdx, uint8_t* & runSlotPattern)
{
  runSlotPattern = DL ? runSlotPatternPool_DL[tddPatternIdx] :  runSlotPatternPool_UL[tddPatternIdx];
  uint8_t nSlotsSche = 0; // number of scheduling slots over 10 slots
  for (int i = 0; i < 10; i++) 
  {
    nSlotsSche += runSlotPattern[i];
  }

  switch(tddPatternIdx)
    {
      case 0: // full slot
        printf("cuMAC TV loading test: full slots scheduling, run %s scheduler in %u slots\n", DL ? "DL" : "UL", nSlotsSche);
        break;
      case 1: // DDDSUUDDDD
        printf("cuMAC TV loading test: using TDD pattern DDDSUUDDDD, run %s scheduler in %u slots\n", DL ? "DL" : "UL", nSlotsSche);
        break;
      case 2: // DDDSU
        printf("cuMAC TV loading test: using TDD pattern DDDSU, run %s scheduler in %u slots\n", DL ? "DL" : "UL", nSlotsSche);
        break;
      default: // full slot
        printf("cuMAC TV loading test: full slots scheduling, run %s scheduler in %u slots\n", DL ? "DL" : "UL", nSlotsSche);
        break;
    }
}
