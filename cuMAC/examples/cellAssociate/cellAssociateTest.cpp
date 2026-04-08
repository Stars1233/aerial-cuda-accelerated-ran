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

 #include "network.h"
 #include "api.h"
 #include "cumac.h"
 #include "channInput/channInput_src/channInput.h"
 
 using namespace cumac;
/////////////////////////////////////////////////////////////////////////
// usage()
void usage()
{
    printf("Multi-cell PF scheduler [options]\n");
    printf("  Options:\n");
    printf("  -i  cuMAC Input channel file\n");
    printf("Examples: './cellAssociateTest' or './cellAssociateTest -i <channel_file>'\n");
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

  // setup randomness seed
  srand(time(NULL));

  // create stream
  cudaStream_t cuStrmMain;
  CUDA_CHECK_ERR(cudaStreamCreate(&cuStrmMain));

  // specify scheduler type
  uint8_t schedulerType = 0; // arbitrary for this test example

  // create network 
  network* net = new network(1, schedulerType);
  // create API
  net->createAPI();
  // creat GPU cellAssociation
  cellAssociation <cuComplex> * cellAssGpu = new cellAssociation<cuComplex>();
  // creat CPU cellAssociation
  cellAssociationCpu * cellAssCpu = new cellAssociationCpu();
  // creat channel input 
  channInput<cuComplex, cuComplex>* channIn = new channInput<cuComplex, cuComplex>(net->getEstH_fr(),net->cellGrpPrmsGpu.get(), net->simParam.get(), inputFileName, cuStrmMain, 1.0);

  // begin to perform scheduling for numSimChnRlz TTIs
  for (int t=0; t<numSimChnRlz; t++) {
    std::cout<<"~~~~~~~~~~~~~~~~~TTI "<<t<<"~~~~~~~~~~~~~~~~~~~~"<<std::endl;
    // Test using channel from channIn class
    channIn -> run(t, net->execStatus.get()->channelRenew);
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
        // copy GPU cell association to CPU buffer for comparison
        net->copyCellAssocResGpu2Cpu(cuStrmMain);
        // run cell association on CPU
        cellAssCpu -> run();
        CUDA_CHECK_ERR(cudaStreamSynchronize(cuStrmMain));
        std::cout<<"Cell Association done"<<std::endl;
    }
    net->compareCpuGpuCellAssocSol();
  }

  // clean up memory
  net->destroyAPI();
  
  delete net;
  delete cellAssGpu;
  delete cellAssCpu;
  delete channIn;
  return 0;
}
 