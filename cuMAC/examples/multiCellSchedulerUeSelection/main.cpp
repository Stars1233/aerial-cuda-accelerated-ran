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

// #define periodicLightWt_
 #define PDSCH_
// #define exitCheckFail_ //exit when solution check fails and create TV for debugging 

#include "network.h"
#include "api.h"
#include "cumac.h"
#include "h5TvCreate.h"
#include "h5TvLoad.h"
#include "trafficModel/trafficService.hpp"

using namespace cumac;

/////////////////////////////////////////////////////////////////////////
// usage()
void usage()
{
    printf("cuMAC DL/UL scheduler pipeline test with [Arguments]\n");
    printf("Arguments:\n");
    printf("  -d  [Indication for DL/UL: 0 - UL, 1 - DL (default 1)]\n");
    printf("  -b  [Indication for baseline CPU RR scheduler/CPU reference check: 0 - CPU reference check, 1 - baseline CPU RR scheduler (default 0)]\n");
    printf("  -p  [Indication for using FP16 PRG allocation kernel: 0 - FP32, 1 - FP16 (default 0)]\n");
    printf("  -t  [Indication for saving TV before return: 0 - not saving TV, 1 - save TV for GPU scheduler, 2 - save TV for CPU scheduler, 3 - save per-cell TVs for testMAC/cuMAC-CP (default 0)]\n");
    printf("  -f  [Indication for choosing fast fading: 0 - Rayleigh fading, 1 - GPU TDL CFR on Prg, 2 - GPU TDL CFR on Sc and Prg, 3 - GPU CDL CFR on Prg, 4 - GPU CDL CFR on Sc and Prg (default 0)]\n"); // currently only CFR on Prg is used in network class, so 2 / 4 is not recommended
    printf("  -g <percent> Enable traffic generation and specify percent of configured UEs to generate traffic on");
    printf("  -r <data_rate> Specify average data rate per UE for traffic generation");
    printf("Example 1 (call cuMAC DL scheduler pipeline with CPU reference check): './multiCellSchedulerUeSelection'\n");
    printf("Example 2 (call cuMAC UL scheduler pipeline with CPU reference check): './multiCellSchedulerUeSelection -d 0'\n");
    printf("Example 3 (call cuMAC DL scheduler pipeline with baseline CPU RR scheduler): './multiCellSchedulerUeSelection -b 1'\n");
    printf("Example 4 (call cuMAC DL scheduler pipeline using GPU TDL channel): './multiCellSchedulerUeSelection -f <1 or 2>'\n");
    printf("Example 4 (call cuMAC DL scheduler pipeline using GPU CDL channel): './multiCellSchedulerUeSelection -f <3 or 4>'\n");
    printf("Example 5 (create cuMAC test vector for DL: './multiCellSchedulerUeSelection -t 1'\n");
    // <channel_file> = ~/mnt/cuMAC/100randTTI570Ues2ta2raUMa_xpol_2.5GHz.mat
}


int main(int argc, char* argv[]) 
{
  int iArg = 1;

  // indicator for DL/UL
  std::string dlUlIndStr = std::string();
  uint8_t DL = 0;

  // indicator for baseline CPU RR scheduler/CPU reference check
  std::string cpuRrRefIndStr = std::string();
  uint8_t baseline = 0;

  // indicator for using FP16 PRG allocation kernel
  std::string hpIndStr = std::string();
  uint8_t halfPrecision = 0;

  // indicator for saving TV before return
  std::string stvIndStr = std::string();
  uint8_t saveTv = 0;

  // fast fading mode: 0 - Rayleigh fading, 1 - GPU TDL CFR on Prg, 2 - GPU TDL CFR on Sc and Prg, 3 - GPU CDL CFR on Prg, 4 - GPU CDL CFR on Sc and Prg (default 0)
  uint8_t fastFadingMode = 0;

  float percent_ue_traffic = 0.0;
  float data_rate = 5000.0;

  while(iArg < argc) {
    if('-' == argv[iArg][0]) {
      switch(argv[iArg][1]) {
        case 'd': // indicator of scheduler modules being called
            if(++iArg >= argc) {
              fprintf(stderr, "ERROR: No DL/UL indicator given.\n");
              exit(1);
            } else {
              dlUlIndStr.assign(argv[iArg++]);
            }
            break;
        case 'g':
            if((++iArg >= argc || (1 != sscanf(argv[iArg], "%f", &percent_ue_traffic)) || (percent_ue_traffic < 0) || (percent_ue_traffic > 100) )) {
              fprintf(stderr, "ERROR: Percent of UEs with traffic generation must be specified.\n");
              exit(1);
            }
            ++iArg;
            break;
        case 'r':
            if((++iArg >= argc || (1 != sscanf(argv[iArg], "%f", &data_rate)) || (data_rate < 0))) {
              fprintf(stderr, "ERROR: Invalid data rate for traffic generation.\n");
              exit(1);
            }
            ++iArg;
            break;
        case 'p': // indicator for using FP16 PRG allocation kernel
            if(++iArg >= argc) {
              fprintf(stderr, "ERROR: No indicator for using FP16 PRG allocation kernel given.\n");
              exit(1);
            } else {
              hpIndStr.assign(argv[iArg++]);
            }
            break;
        case 't': // indicator for saving TV before return
            if(++iArg >= argc) {
              fprintf(stderr, "ERROR: No indicator for saving TV before return given.\n");
              exit(1);
            } else {
              stvIndStr.assign(argv[iArg++]);
            }
            break;
        case 'b': // indicator of scheduler modules being called
            if(++iArg >= argc) {
              fprintf(stderr, "ERROR: No CPU RR/CPU reference check indicator given.\n");
              exit(1);
            } else {
              cpuRrRefIndStr.assign(argv[iArg++]);
            }
            break;
        case 'f': // set fast fading mode
            if((++iArg >= argc) || (1 != sscanf(argv[iArg], "%hhi", &fastFadingMode)) || (fastFadingMode < 0) || (fastFadingMode > 4) )
            {
                fprintf(stderr, "ERROR: Unsupported fast fading mode.\n");
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
    } else {
      fprintf(stderr, "ERROR: Invalid command line argument: %s\n", argv[iArg]);
      exit(1);
    }
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
  printf("cuMAC multi-cell scheduler: Running on GPU device %d (total devices: %d)\n", 
         my_dev, deviceCount);

  // setup randomness seed
  srand(seedConst);

  // create stream
  cudaStream_t cuStrmMain;
  CUDA_CHECK_ERR(cudaStreamCreate(&cuStrmMain));

  if (dlUlIndStr.size() == 0) {
      DL = 1;
  } else {
      DL = static_cast<uint8_t>(dlUlIndStr[0] - '0');
  }
  if (DL == 1) {
      printf("cuMAC scheduler pipeline test: Downlink\n");
  } else {
      printf("cuMAC scheduler pipeline test: Uplink\n");
  }

  if (hpIndStr.size() == 0) {
      halfPrecision = 0;
  } else {
      halfPrecision = static_cast<uint8_t>(hpIndStr[0] - '0');
  }
  if (halfPrecision == 1) {
      printf("cuMAC scheduler pipeline test: FP16 half-precision kernels\n");
  } else {
      printf("cuMAC scheduler pipeline test: FP32 kernels\n");
  }

  if (stvIndStr.size() == 0) {
      saveTv = 0;
  } else {
      saveTv = static_cast<uint8_t>(stvIndStr[0] - '0');
  }

  if (cpuRrRefIndStr.size() == 0) {
      baseline = 0;
  } else {
      baseline = static_cast<uint8_t>(cpuRrRefIndStr[0] - '0');
  }
  if (baseline == 1) {
    printf("cuMAC scheduler pipeline test: baseline CPU RR scheduler for performance benchmarking\n");
  } else {
    printf("cuMAC scheduler pipeline test: CPU reference check\n");
  }

  switch (fastFadingMode)
  {
    case 0:
      printf("cuMAC scheduler pipeline test: use Rayleigh fading\n");
      break;
    case 1:
      printf("cuMAC scheduler pipeline test: use GPU TDL channel, CFR on Prg\n");
      break;
    case 2:
      printf("cuMAC scheduler pipeline test: use GPU TDL channel, CFR on Sc and Prg\n");
      break;
    case 3:
      printf("cuMAC scheduler pipeline test: use GPU CDL channel, CFR on Prg\n");
      break;
    case 4:
      printf("cuMAC scheduler pipeline test: use GPU CDL channel, CFR on Sc and Prg\n");
      break;
    default:
      fprintf(stderr, "ERROR: Unsupported fast fading mode.\n");
      exit(1);
  }

  // specify scheduler type
  uint8_t schedulerType = 1; // multi-cell scheduler

  // specify channel matrix access type: 0 - row major, 1 - column major
  uint8_t columnMajor = 1;

  // use lightWeight kernels
  uint8_t lightWeight = 0;

  // percentage of SMs to determine the number of thread blocks used in light-weight kernels
  float percSmNumThrdBlk = 2.0;

  // CSI update indicator
  uint8_t csiUpdate = 1;

  // create network 
  network* net = new cumac::network(DL, schedulerType, 1 /*fixCellAssoc*/, fastFadingMode, percent_ue_traffic > 0.0, cuStrmMain);

  net->genNetTopology();
  net->genLSFading();

  // create API
  net->createAPI();


  // Initialize traffic service
  TrafficType basic_traffic(data_rate, 0, 1);
  TrafficConfig traf_cfg(basic_traffic,totNumUesConst*percent_ue_traffic/100.0);
  //TrafficType low_traffic(100, 0, 1);
  //traf_cfg.AddFlows(low_traffic,totNumUesConst/2);
  std::unique_ptr<TrafficService> trafSvc = std::make_unique<TrafficService>(traf_cfg,net->cellGrpUeStatusCpu.get(),net->cellGrpUeStatusGpu.get());

  // determine the number of interfering cells
  uint16_t nInterfCell = net->simParam.get()->totNumCell - net->cellGrpPrmsGpu.get()->nCell;

  assert(nInterfCell == 0);

  // post-eq SINR calculation
  cumac::mcSinrCalHndl_t mcSinrCalGpu = new cumac::multiCellSinrCal(net->cellGrpPrmsGpu.get());

  // PF UE selection
  cumac::mcUeSelHndl_t mcUeSelGpu = new cumac::multiCellUeSelection(net->cellGrpPrmsGpu.get());

  // create GPU multi-cell scheduler 
  cumac::mcSchdHndl_t mcSchGpu = new cumac::multiCellScheduler(net->cellGrpPrmsGpu.get());

#ifdef PDSCH_
  // GPU layer selection
  cumac::mcLayerSelHndl_t mcLayerSelGpu = new cumac::multiCellLayerSel(net->cellGrpPrmsGpu.get());

  // GPU MCS selection
  std::unique_ptr<cumac::mcsSelectionLUT> mcsSelGpu = std::make_unique<cumac::mcsSelectionLUT>(net->cellGrpPrmsGpu.get(), cuStrmMain);
  CUDA_CHECK_ERR(cudaStreamSynchronize(cuStrmMain));

  // CPU layer selection
  cumac::mcLayerSelCpuHndl_t mcLayerSelCpu = new cumac::multiCellLayerSelCpu(net->cellGrpPrmsCpu.get());

  // CPU MCS selection
  std::unique_ptr<cumac::mcsSelectionLUTCpu> mcsSelCpu = std::make_unique<cumac::mcsSelectionLUTCpu>(net->cellGrpPrmsCpu.get());
#endif

  rrUeSelCpuHndl_t  rrUeSelCpu  = nullptr;
  rrSchdCpuHndl_t   rrSchCpu    = nullptr;
  mcUeSelCpuHndl_t  mcUeSelCpu  = nullptr;
  mcSchdCpuHndl_t   mcSchCpu    = nullptr;
  if (baseline == 1) { // baseline CPU RR scheduler
    printf("Using CPU RR UE selection\n");
    // create CPU Round Robin UE selection
    rrUeSelCpu = new roundRobinUeSelCpu(net->cellGrpPrmsCpu.get());

    // create CPU Round Robin scheduler
    printf("Using CPU RR UE scheduler\n");
    rrSchCpu = new roundRobinSchedulerCpu(net->cellGrpPrmsCpu.get());
  } else { // CPU reference check
    printf("Using CPU multi-cell PF UE selection\n");
    // create CPU multi-cell PF UE selection
    mcUeSelCpu = new multiCellUeSelectionCpu(net->cellGrpPrmsCpu.get());

    printf("Using CPU multi-cell PF scheduler\n");
    // create CPU multi-cell scheduler
    mcSchCpu = new multiCellSchedulerCpu(net->cellGrpPrmsCpu.get());
  }
  
  // create SVD precoder
  svdPrdHndl_t svdPrd = nullptr;

  if (prdSchemeConst) {
    svdPrd = new svdPrecoding(net->cellGrpPrmsGpu.get());

    // Setup SVD precoder
    svdPrd->setup(net->cellGrpPrmsGpu.get(), cuStrmMain);
  }

  // begin to perform scheduling for numSimChnRlz TTIs
  for (int t=0; t<numSimChnRlz; t++) {
    std::cout<<"~~~~~~~~~~~~~~~~~TTI "<<t<<"~~~~~~~~~~~~~~~~~~~~"<<std::endl;
#ifdef periodicLightWt_
    if (t%16 == 0) {
      lightWeight = 0;
      csiUpdate = 1;
    } else {
      lightWeight = 1;
      csiUpdate = 0;
    }
#endif

    // generate user data traffic
    trafSvc->Update();

    // generate channel
    if(net->execStatus.get()->channelRenew)
    {
      net->genFastFadingGpu(t);
      CUDA_CHECK_ERR(cudaStreamSynchronize(cuStrmMain));
      std::cout<<"GPU channel generated"<<std::endl;

      if (prdSchemeConst) {
        // run SVD precoder
        svdPrd->run(net->cellGrpPrmsGpu.get());
        CUDA_CHECK_ERR(cudaStreamSynchronize(cuStrmMain));
        std::cout<<"SVD precoder and singular values computed"<<std::endl;
      }
    }

    // setup API 
    net->setupAPI(cuStrmMain);
    CUDA_CHECK_ERR(cudaStreamSynchronize(cuStrmMain));
    std::cout<<"API setup completed"<<std::endl;

#ifdef periodicLightWt_
    if (csiUpdate == 1) {
#endif
      // GPU post-eq SINR calculation
      mcSinrCalGpu->setup(net->cellGrpPrmsGpu.get(), columnMajor, cuStrmMain);
      CUDA_CHECK_ERR(cudaStreamSynchronize(cuStrmMain));
      std::cout<<"CSI update: subband SINR calculation setup completed"<<std::endl;

      mcSinrCalGpu->run(cuStrmMain);
      CUDA_CHECK_ERR(cudaStreamSynchronize(cuStrmMain));
      std::cout<<"CSI update: subband SINR calculation run completed"<<std::endl;
      
      //if (t == 0)
      //  mcSinrCalGpu->debugLog();

      mcSinrCalGpu->setup_wbSinr(net->cellGrpPrmsGpu.get(), cuStrmMain);
      CUDA_CHECK_ERR(cudaStreamSynchronize(cuStrmMain));
      std::cout<<"CSI update: wideband SINR calculation setup completed"<<std::endl;

      mcSinrCalGpu->run(cuStrmMain);
      CUDA_CHECK_ERR(cudaStreamSynchronize(cuStrmMain));
      std::cout<<"CSI update: wideband SINR calculation run completed"<<std::endl;

      net->cpySinrGpu2Cpu();
      std::cout<<"CSI update: subband and wideband SINRS copied to CPU structures"<<std::endl;
#ifdef periodicLightWt_
    }
#endif    

    // GPU UE selection
    mcUeSelGpu->setup(net->cellGrpUeStatusGpu.get(), net->schdSolGpu.get(), net->cellGrpPrmsGpu.get(), cuStrmMain);
    CUDA_CHECK_ERR(cudaStreamSynchronize(cuStrmMain));
    std::cout<<"GPU PF UE selection setup completed"<<std::endl;

    mcUeSelGpu->run(cuStrmMain);
    CUDA_CHECK_ERR(cudaStreamSynchronize(cuStrmMain));
    std::cout<<"GPU PF UE selection run completed"<<std::endl;

    // mcUeSelGpu->debugLog();

    net->ueDownSelectGpu();
    std::cout<<"GPU UE downselection completed"<<std::endl;

    //CPU UE selection
    if (baseline == 1) { 
      rrUeSelCpu->setup(net->cellGrpUeStatusCpu.get(), net->schdSolCpu.get(), net->cellGrpPrmsCpu.get());
      rrUeSelCpu->run();
      std::cout<<"CPU RR UE selection completed"<<std::endl;
    } else {
      mcUeSelCpu->setup(net->cellGrpUeStatusCpu.get(), net->schdSolCpu.get(), net->cellGrpPrmsCpu.get());
      mcUeSelCpu->run();
      std::cout<<"CPU PF UE selection completed"<<std::endl;
    }

    net->ueDownSelectCpu();
    std::cout<<"CPU UE downselection completed"<<std::endl;

    // only set coordinate cell IDs and perform cell assocation at the first time slot //
    if (t == 0) {
      net->execStatus.get()->cellIdRenew    = false;
      net->execStatus.get()->cellAssocRenew = false;
    }

    // setup GPU multi-cell scheduler
    if (lightWeight == 1) {
      printf("Multi-cell scheduler: calling light-weight kernel\n");
    }
    mcSchGpu->setup(net->cellGrpUeStatusGpu.get(), net->schdSolGpu.get(), net->cellGrpPrmsGpu.get(), net->simParam.get(), columnMajor, halfPrecision, lightWeight, percSmNumThrdBlk, cuStrmMain);
    CUDA_CHECK_ERR(cudaStreamSynchronize(cuStrmMain));
    
    if (baseline == 1) {
      // setup CPU Round Robin scheduler
      rrSchCpu->setup(net->cellGrpUeStatusCpu.get(), net->schdSolCpu.get(), net->cellGrpPrmsCpu.get());
    } else {
      // setup CPU multi-cell scheduler
      mcSchCpu->setup(net->cellGrpUeStatusCpu.get(), net->schdSolCpu.get(), net->cellGrpPrmsCpu.get(), net->simParam.get(), columnMajor);
    }

#ifdef PDSCH_
    // setup GPU layer selection
    mcLayerSelGpu->setup(net->cellGrpUeStatusGpu.get(), net->schdSolGpu.get(), net->cellGrpPrmsGpu.get(), 0, cuStrmMain);
    CUDA_CHECK_ERR(cudaStreamSynchronize(cuStrmMain));

    // setup GPU MCS selection
    mcsSelGpu->setup(net->cellGrpUeStatusGpu.get(), net->schdSolGpu.get(), net->cellGrpPrmsGpu.get(), cuStrmMain);
    CUDA_CHECK_ERR(cudaStreamSynchronize(cuStrmMain));

    // setup CPU layer selection
    mcLayerSelCpu->setup(net->cellGrpUeStatusCpu.get(), net->schdSolCpu.get(), net->cellGrpPrmsCpu.get());

    // setup CPU MCS selection
    mcsSelCpu->setup(net->cellGrpUeStatusCpu.get(), net->schdSolCpu.get(), net->cellGrpPrmsCpu.get());
#endif

    // run GPU multi-cell scheduler
    mcSchGpu->run(cuStrmMain);
    CUDA_CHECK_ERR(cudaStreamSynchronize(cuStrmMain));

    // debug parameter logging
    // mcSchGpu->debugLog();
    
    if (baseline == 1) { 
      // run CPU Round Robin scheduler
      rrSchCpu->run();
    } else {
      // run CPU multi-cell scheduler
      mcSchCpu->run();

      // debug parameter logging
      // mcSchCpu->debugLog();
    }
    std::cout<<"PRB scheduling solution computed"<<std::endl;

#ifdef PDSCH_
    // run GPU layer selection
    mcLayerSelGpu->run(cuStrmMain);
    CUDA_CHECK_ERR(cudaStreamSynchronize(cuStrmMain));
    std::cout<<"GPU Layer selection solution computed"<<std::endl;

    // run CPU layer selection
    mcLayerSelCpu->run();
    std::cout<<"CPU Layer selection solution computed"<<std::endl;

    // run GPU MCS selection
    mcsSelGpu->run(cuStrmMain);
    CUDA_CHECK_ERR(cudaStreamSynchronize(cuStrmMain));
    std::cout<<"GPU MCS selection solution computed"<<std::endl;
    // mcsSelGpu->debugLog();

    // run CPU MCS selection
    mcsSelCpu->run();
    std::cout<<"CPU MCS selection solution computed"<<std::endl;
    // mcsSelCpu->debugLog();
#endif

    // use scheduling solution
    net->run(cuStrmMain);
    CUDA_CHECK_ERR(cudaStreamSynchronize(cuStrmMain));
    std::cout<<"Scheduling solution transferred to host"<<std::endl;

    if (baseline == 0) { 
      bool solCheckPass = net->compareCpuGpuAllocSol();
    }

#ifdef exitCheckFail_
    if (!solCheckPass) {
      saveToH5("tvDebug.h5",
               net->cellGrpUeStatusGpu.get(),
               net->cellGrpPrmsGpu.get(),
               net->schdSolGpu.get());
      return 0;
    }
#endif 

#ifdef PDSCH_
    net->phyAbstract(2, t);
#else
    net->updateDataRateUeSelCpu(t);
    net->updateDataRateGpu(t);
#endif

    net->updateDataRateAllActiveUeCpu(t);
    net->updateDataRateAllActiveUeGpu(t);
  }

  bool cpuGpuPerfCheckPass;
  if (baseline == 1) { 
    cpuGpuPerfCheckPass = 1;
  } else {
    cpuGpuPerfCheckPass = net->compareCpuGpuSchdPerf();
    if (cpuGpuPerfCheckPass) {
      printf("CPU and GPU scheduler performance check result: PASS\n");
    } else {
      printf("CPU and GPU scheduler performance check result: FAIL\n");
    }
  }

  if (saveTv == 1) {
    std::string saveTvName = "TV_cumac_F08-MC-CC-" + std::to_string(net->cellGrpPrmsGpu.get()->nCell) +"PC_" + (DL ? "DL" : "UL") + ".h5";

    saveToH5(saveTvName,
             net->cellGrpUeStatusGpu.get(),
             net->cellGrpPrmsGpu.get(),
             net->schdSolGpu.get());
  } else if (saveTv == 2) {
    std::string saveTvName = "CPU_TV_cumac_F08-MC-CC-" + std::to_string(net->cellGrpPrmsCpu.get()->nCell) +"PC_" + (DL ? "DL" : "UL") + ".h5";

    saveToH5_CPU(saveTvName,
             net->cellGrpUeStatusCpu.get(),
             net->cellGrpPrmsCpu.get(),
             net->schdSolCpu.get());
  } else if (saveTv == 3) {
    std::string saveTvName = "TV_cumac_F08-MC-CC-" + std::to_string(net->cellGrpPrmsGpu.get()->nCell) +"PC_" + (DL ? "DL" : "UL") + ".h5";

    saveToH5(saveTvName,
             net->cellGrpUeStatusGpu.get(),
             net->cellGrpPrmsGpu.get(),
             net->schdSolGpu.get());

    saveTvName = "TV_F08_";
    for (int cellIdx = 0; cellIdx < net->cellGrpPrmsGpu.get()->nCell; cellIdx++) {
      saveToH5_testMAC_perCell(saveTvName,
                               cellIdx,
                               net->cellGrpUeStatusGpu.get(),
                               net->cellGrpPrmsGpu.get(),
                               net->schdSolGpu.get());
    }
  }

  net->writeToFileLargeNumActUe();
  net->writetoFileLargeNumActUe_short();

  // clean up memory
  if (prdSchemeConst) {
    svdPrd->destroy();
  }
  net->destroyAPI();

  if (mcSinrCalGpu) delete mcSinrCalGpu;
  if (mcUeSelGpu) delete mcUeSelGpu;
  if (mcSchGpu) delete mcSchGpu;
  if (rrSchCpu) delete rrSchCpu;
  if (mcSchCpu) delete mcSchCpu;
  if (rrUeSelCpu) delete rrUeSelCpu;
  if (mcUeSelCpu) delete mcUeSelCpu;
  if (svdPrd) delete svdPrd;
  if (net) delete net;
  if (mcLayerSelGpu) delete mcLayerSelGpu;
  if (mcLayerSelCpu) delete mcLayerSelCpu;

  return !cpuGpuPerfCheckPass;
}
