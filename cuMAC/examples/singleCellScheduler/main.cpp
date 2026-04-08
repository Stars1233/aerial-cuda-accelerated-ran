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

// #define PDSCH_

#include "network.h"
#include "api.h"
#include "cumac.h"
#include "channInput/channInput_src/channInput.h"

using namespace cumac;

#define numRandTTI numSimChnRlz
/////////////////////////////////////////////////////////////////////////
// usage()
void usage()
{
    printf("Single-cell PF scheduler [options]\n");
    printf("  Options:\n");
    printf("  -i  cuMAC Input channel file\n");
    printf("Examples: './singleCellSchedule' or './singleCellSchedule -i <channel_file>'\n");
    // <channel_file> = ~/mnt/cuMAC/100randTTI570Ues2ta2raUMa_xpol_2.5GHz.mat
}

int main(int argc, char* argv[]) 
{
  int iArg = 1;
  std::string inputFileName = std::string();
  std::string outputFileName = std::string();

  while(iArg < argc)
  {
    if('-' == argv[iArg][0])
    {
      switch(argv[iArg][1])
      {
      case 'i': // input channel file name
          if(++iArg >= argc)
          {
              fprintf(stderr, "ERROR: No input file name given. Using random channel\n");
          }
          else
          {
            inputFileName.assign(argv[iArg++]);
          }
          break;

      case 'o': // out put results name
          if(++iArg >= argc)
          {
              fprintf(stderr, "ERROR: No output file name given.\n");
          }
          outputFileName.assign(argv[iArg++]);
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
  printf("cuMAC single-cell scheduler: Running on GPU device %d (total devices: %d)\n", 
         my_dev, deviceCount);

  int supportsCoopLaunch = 0;
  CUDA_CHECK_ERR(cudaDeviceGetAttribute(&supportsCoopLaunch, cudaDevAttrCooperativeLaunch, my_dev));
  printf("supportsCoopLaunch = %d\n", supportsCoopLaunch);

  // setup randomness seed
  srand(seedConst);

  // create stream
  cudaStream_t cuStrmMain;
  CUDA_CHECK_ERR(cudaStreamCreate(&cuStrmMain));

  // specify DL or UL
  uint8_t DL = 1;

  // specify scheduler type
  uint8_t schedulerType = 0; // single-cell scheduler

  // create network 
  cumac::network* net = new cumac::network(1, schedulerType);
  // create API
  net->createAPI();

  // load the number of cells in the cell group
  uint16_t nCell = net->simParam.get()->totNumCell; // number of cells in the network including interfering cells

  // create GPU single-cell scheduler
  scSchdHndl_t* scSchArrGpu = new scSchdHndl_t[nCell];

  // create CPU single-cell scheduler
  scSchdCpuHndl_t* scSchArrCpu = new scSchdCpuHndl_t[nCell];
  for (int cIdx = 0; cIdx < nCell; cIdx++) {
      scSchArrGpu[cIdx] = new singleCellScheduler();
      scSchArrCpu[cIdx] = new singleCellSchedulerCpu();
  }

  // create GPU MCS selection
  mcsSelLUTHndl_t mcsSelGpu = new mcsSelectionLUT(net->cellGrpPrmsGpu.get(), cuStrmMain);
  CUDA_CHECK_ERR(cudaStreamSynchronize(cuStrmMain));

  // create CPU MCS selection
  mcsSelLUTCpuHndl_t mcsSelCpu = new mcsSelectionLUTCpu(net->cellGrpPrmsCpu.get());

  // create SVD precoder
  svdPrecoding* svdPrd = new svdPrecoding(net->cellGrpPrmsGpu.get());

  // Setup SVD precoder
  svdPrd->setup(net->cellGrpPrmsGpu.get(), cuStrmMain);

 // ------------   cell association --------------------------
  // creat GPU cellAssociation
  cellAssociation <cuComplex> * cellAssGpu = new cellAssociation<cuComplex>();
  // creat CPU cellAssociation
  cellAssociationCpu * cellAssCpu = new cellAssociationCpu();

  // creat channel input 
  channInput<cuComplex, cuComplex>* channIn = new channInput<cuComplex, cuComplex>(net->getEstH_fr(), net->cellGrpPrmsGpu.get(), net->simParam.get(), inputFileName, cuStrmMain, 1.0);

  // begin to perform scheduling for numSimChnRlz TTIs
  int order = -1;
  int channIdx = 0;
  for (int t=0; t<numSimChnRlz; t++) {
    if (channIdx == (numRandTTI-1) || channIdx == 0) {
      order *= -1;
    }
    std::cout<<"~~~~~~~~~~~~~~~~~TTI "<<t<<"~~~~~~~~~~~~~~~~~~~~"<<std::endl;
    // get channel
    channIn -> run(channIdx, net->execStatus.get()->channelRenew);
    if(net->execStatus.get()->channelRenew)
    {
      std::cout<<"Channel generated"<<std::endl;
      #ifdef CHANN_PRINT_SAMPLE_GPU_
        channIn -> printGpuOutChann();
      #endif
    }

    // cell association
    if (net->execStatus.get()->cellAssocRenew) {
        // setup GPU cell association
        cellAssGpu->setup(net->cellGrpPrmsGpu.get(), net->simParam.get(), cuStrmMain);
        // setup CPU cellAssociation
        cellAssCpu->setup(net->cellGrpPrmsCpu.get(), net->simParam.get());
        // run cell association on GPU
        cellAssGpu->run(cuStrmMain); 
        // run cell association on CPU
        cellAssCpu->run();
        CUDA_CHECK_ERR(cudaStreamSynchronize(cuStrmMain));
        std::cout<<"Cell Association done"<<std::endl;
    }
    
    // setup API to single-cell scheduler
    net->setupAPI(cuStrmMain);
    CUDA_CHECK_ERR(cudaStreamSynchronize(cuStrmMain));
    std::cout<<"API setup completed"<<std::endl;

    // only set coordinate cell IDs and perform cell assocation at the first time slot //
    if (t == 0) {
      net->execStatus.get()->cellIdRenew    = false;
      net->execStatus.get()->cellAssocRenew = false;
    }

    // setup GPU and CPU single-cell schedulers
    for (int cIdx = 0; cIdx < nCell; cIdx++) {
      scSchArrGpu[cIdx]->setup(cIdx, net->cellGrpUeStatusGpu.get(), net->schdSolGpu.get(), net->cellGrpPrmsGpu.get(), net->simParam.get(), cuStrmMain);
      scSchArrCpu[cIdx]->setup(cIdx, net->cellGrpUeStatusCpu.get(), net->schdSolCpu.get(), net->cellGrpPrmsCpu.get(), net->simParam.get());
    }

    // setup GPU MCS selection
    mcsSelGpu->setup(net->cellGrpUeStatusGpu.get(), net->schdSolGpu.get(), net->cellGrpPrmsGpu.get(), cuStrmMain);
    CUDA_CHECK_ERR(cudaStreamSynchronize(cuStrmMain));
    std::cout<<"Single-cell PF scheduler setup completed"<<std::endl;

    // setup CPU MCS selection
    mcsSelCpu->setup(net->cellGrpUeStatusCpu.get(), net->schdSolCpu.get(), net->cellGrpPrmsCpu.get());
    /////////////////////////////////////////////////////////////////////////////////////

    svdPrd->run(net->cellGrpPrmsGpu.get());
    CUDA_CHECK_ERR(cudaStreamSynchronize(cuStrmMain));

    net->copyPrdMatGpu2Cpu(cuStrmMain);
    CUDA_CHECK_ERR(cudaStreamSynchronize(cuStrmMain));

    for (int cIdx = 0; cIdx < nCell; cIdx++) {
        scSchArrGpu[cIdx]->run(cuStrmMain);
        CUDA_CHECK_ERR(cudaStreamSynchronize(cuStrmMain));
        scSchArrCpu[cIdx]->run();
    }
    std::cout<<"Scheduling solution computed"<<std::endl;

    // run GPU MCS selection
    mcsSelGpu->run(cuStrmMain);
    CUDA_CHECK_ERR(cudaStreamSynchronize(cuStrmMain));

    // run CPU MCS selection
    mcsSelCpu->run();

    // debug MCS selection
    // mcsSelCpu->printMcsSelSol(net->schdSolCpu.get(), t);

    std::cout<<"MCS selection solution computed"<<std::endl;

    // use scheduling solution
    net->run(cuStrmMain);
    CUDA_CHECK_ERR(cudaStreamSynchronize(cuStrmMain));
    std::cout<<"Scheduling solution transferred to host"<<std::endl;

    net->compareCpuGpuAllocSol();
    net->compareCpuGpuAllocSol();

#ifdef PDSCH_
    net->updateDataRatePdschCpu(t);
    net->updateDataRatePdschGpu(t);
#else
    net->updateDataRateCpu(t);
    net->updateDataRateGpu(t);
#endif

    channIdx += order;
  }

  net->writeToFile();

  // clean up memory
  svdPrd->destroy();
  net->destroyAPI();
  
  for (int cIdx = 0; cIdx < nCell; cIdx++) {
      delete scSchArrGpu[cIdx];
      delete scSchArrCpu[cIdx];
  }
  delete scSchArrGpu;
  delete scSchArrCpu;
  delete mcsSelGpu;
  delete mcsSelCpu;
  delete svdPrd;
  delete net;
  delete cellAssGpu;
  delete cellAssCpu;
  delete channIn;

  return 0;
}
