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

#include "testBench.hpp"
#include "api.h"
#include "cumac.h"
#include "h5TvCreate.h"
#include "h5TvLoad.h" 

using namespace cumac;

/////////////////////////////////////////////////////////////////////////
// usage()
void usage(char* arg)
{
    printf("cuMAC DL/UL scheduler pipeline test with [Arguments]\n");
    printf("Arguments:\n");
    printf("  -d  [Indication for DL/UL: 0 - UL, 1 - DL (default 1)]\n");
    printf("  -b  [Indication for baseline CPU RR scheduler/CPU reference check: 0 - CPU reference check, 1 - baseline CPU RR scheduler (default 0)]\n");
    printf("  -p  [Indication for using FP16 PRG allocation kernel: 0 - FP32, 1 - FP16 (default 0)]\n");
    printf("  -t  [Indication for saving TV before return: 0 - not saving TV, 1 - save TV for GPU scheduler, 2 - save TV for CPU scheduler (default 0)]\n");
    printf("  -n  [number of TTI to run]\n");
    printf("Example 1 (call cuMAC DL scheduler pipeline with CPU reference check): '%s'\n",arg);
    printf("Example 2 (call cuMAC UL scheduler pipeline with CPU reference check): '%s -d 0'\n",arg);
    printf("Example 3 (call cuMAC DL scheduler pipeline with baseline CPU RR scheduler): '%s -b 1'\n",arg);
    printf("Example 4 (create cuMAC test vector for DL: '%s -t 1'\n",arg);
    // <channel_file> = ~/mnt/cuMAC/100randTTI570Ues2ta2raUMa_xpol_2.5GHz.mat
}


int main(int argc, char* argv[]) 
{
  int c;
  uint16_t numTti = numSimChnRlz;
  schedulerParams schedParams;

  // Default configs
  uint8_t DL = 0;  // schedule DL
  schedParams.saveTv = 0;   // Don't save TV
  schedParams.halfPrecision = 0; // Full precision
  schedParams.baseline = 0;
  schedParams.schedulerType = 1; // multi-cell scheduler
  schedParams.columnMajor = 1;   // Column major
  schedParams.enableHarq = 0;
  schedParams.mcsBaseline = 0;
  schedParams.direction = DL;  // Data Direction
  schedParams.precodingScheme = prdSchemeConst;
  schedParams.enable_pdsch = true;
  schedParams.periodicLightWt = false;
  schedParams.cpuMcsBaseline = false;
  
  while ((c = getopt(argc, argv, "b:d:hn:p:t:")) != -1) 
  {
    switch(c) 
    {
      case 'b':
        // indicator for baseline CPU RR scheduler/CPU reference check
        schedParams.baseline = atoi(optarg);
        break;
      case 'd':
        // indicator for DL/UL
        DL = atoi(optarg);
        break;
      case 'n':
        numTti = atoi(optarg);
        if((numSimChnRlz < numTti) || (0 > numTti))
        {
          printf("Number of TTI {%d} must be between 0 and %d.  Setting to %d\n",numTti,numSimChnRlz,numSimChnRlz);
          numTti = numSimChnRlz;
        }
        break;
      case 'p':
        // indicator for using FP16 PRG allocation kernel
        schedParams.halfPrecision = atoi(optarg);
        break;
      case 't':
        // indicator for saving TV before return
        schedParams.saveTv = atoi(optarg);
        break;
      case 'h':
      default:
        usage(argv[0]);
        return -1;
    }
  }

  // Setup NVLOG
  NvlogFmtHelper log;

  NVLOGC_FMT(NVLOG_TESTBENCH, "cuMAC scheduler pipeline test: {}",DL==1 ? "Downlink" : "Uplink");
  NVLOGC_FMT(NVLOG_TESTBENCH, "cuMAC scheduler pipeline test: {}",schedParams.halfPrecision == 1 ? "FP16 half-precision kernels" : "FP32 kernels");
  NVLOGC_FMT(NVLOG_TESTBENCH, "cuMAC scheduler pipeline test: {}",schedParams.baseline == 1 ? 
                                                                  "baseline CPU RR scheduler for performance benchmarking" : "CPU reference check");

  // validate GPU device with fallback mechanism
  int deviceCount{};
  CUDA_CHECK_ERR(cudaGetDeviceCount(&deviceCount));
  
  unsigned validGpuDeviceIdx = gpuDeviceIdx;
  if (static_cast<int>(gpuDeviceIdx) >= deviceCount) {
      printf("WARNING: Requested GPU device %u exceeds available device count (%d). Falling back to GPU device 0.\n", 
             gpuDeviceIdx, deviceCount);
      validGpuDeviceIdx = 0;
  }
  
  printf("cuMAC scheduler pipeline test: Running on GPU device %d (total devices: %d)\n", 
         validGpuDeviceIdx, deviceCount);

  testBench test(schedParams, validGpuDeviceIdx,seedConst);

  // begin to perform scheduling for numTti TTIs
  for (int t=0; t<numTti; t++) {
    NVLOGC_FMT(NVLOG_TESTBENCH, "~~~~~~~~~~~~~~~~~TTI {}~~~~~~~~~~~~~~~~~~~~",t);

    test.setup(t);

    test.run();

    test.check();

    test.updateDataRate(t);
  }

  bool cpuGpuPerfCheckPass = test.validate();

  test.save();


  return !cpuGpuPerfCheckPass;
}
