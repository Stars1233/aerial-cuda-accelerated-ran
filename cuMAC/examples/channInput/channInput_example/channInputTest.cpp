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

// Unit test for ChannInput, interface of channel input
#include "../channInput_src/channInput.h"
#include "network.h"

// usage ./channInputTest ~/mnt/cuMAC/chl_asim.h5
// ./channInputTest ~/mnt/cuMAC/100randTTI570Ues2ta2raUMa_xpol_2.5GHz.mat
// ~/mnt/cuMAC/testMatForH5.mat
// For ASIM file: ./channInputTest matx_channels 1
int main(int argc, char* argv[]) 
{
  srand(seedConst);

  std::string iFileName;
  uint8_t asimChanFlag = 0;

  if(argc == 1)
  {
    printf("Warning: No input channel file detected, using random channel!\n");
  }
  else
  {
    iFileName.assign(argv[1]);
    if(argc == 3)
    {
      int tempAsimChanFlag = 0;
      sscanf(argv[2], "%d", &tempAsimChanFlag);
      asimChanFlag = tempAsimChanFlag;
    }
  }

  cudaStream_t cuStrmMain;
  CUDA_CHECK_ERR(cudaStreamCreate(&cuStrmMain));
  cumacCellGrpPrms * cellGrpPrms     = new cumacCellGrpPrms;
  cumacSimParam* simParam            = new cumacSimParam;

  cuComplex*  estH_fr                = new cuComplex[nPrbGrpsConst*totNumUesConst*numCellConst*nBsAntConst*nUeAntConst];
           
  cellGrpPrms->nUe                   = totNumUesConst; // total number of UEs
  cellGrpPrms->nCell                 = numCoorCellConst; // number of coordinated cells
  cellGrpPrms->nPrbGrp               = nPrbGrpsConst;
  cellGrpPrms->nBsAnt                = nBsAntConst;
  cellGrpPrms->nUeAnt                = nUeAntConst;

  simParam->totNumCell               = numCellConst;
  
  CUDA_CHECK_ERR(cudaMalloc((void **)&cellGrpPrms->estH_fr, sizeof(cuComplex)*totNumUesConst*numCellConst*nPrbGrpsConst*nBsAntConst*nBsAntConst));

  uint16_t * cellID           = new uint16_t[numCoorCellConst];
  for (int cIdx = 0; cIdx < numCoorCellConst; cIdx++) 
  {
        cellID[cIdx] = cIdx;
  }  

  // cudaMalloc((void **)&cellGrpPrms->cellID, sizeof(uint16_t)*numCoorCellConst);
  // cudaMemcpyAsync(cellGrpPrms->cellID, cellID, sizeof(uint16_t)*numCoorCellConst, cudaMemcpyHostToDevice, cuStrmMain);
  // cudaStreamSynchronize(cuStrmMain);

  channInput<cuComplex, cuComplex> * channInputTest = new channInput<cuComplex, cuComplex>(estH_fr, cellGrpPrms, simParam, iFileName, cuStrmMain, 1.0, asimChanFlag);

  // set channel bias
  float * preset_cell_associate = new float[totNumUesConst * numCellConst];
  for(int ueIdx = 0; ueIdx < totNumUesConst * numCellConst; ueIdx++)
  {
    preset_cell_associate[ueIdx] = 100.0;
  }
  channInputTest -> setCellBias(preset_cell_associate); 

  // for (int t=0; t<numSimChnRlz; t++) 
  for (int t=0; t<1; t++) 
  {
    std::cout<<"~~~~~~~~~~~~~~~~~TTI "<< t <<"~~~~~~~~~~~~~~~~~~~~"<<std::endl;
    channInputTest -> run(t, true);
    uint16_t* chanDimPtr = channInputTest -> getOutChannDim();
    printf("Channel generated. Output channel: nPRBG = %d, nUE = %d, nCells = %d, nBSAnt = %d, nUEAnt = %d \n", chanDimPtr[0], chanDimPtr[1], chanDimPtr[2], chanDimPtr[3], chanDimPtr[4]);
    #ifdef CHANN_INPUT_DEBUG_
      channInputTest -> printRawChann(); // print first of outChannel
      channInputTest -> printCpuOutChann(); // print first of outChannel from GPU
    #endif
    channInputTest -> printGpuOutChann(); // print first of outChannel from GPU
  }

  // Test each function
  channInputTest -> getRawChannDim();
  channInputTest -> getOutChannDim();
  channInputTest -> printCellAssoc();
  // cellGrpPrms->estH_fr is the same with channInputTest->m_GPUestH_fr
  delete[] cellID;
  CUDA_CHECK_ERR(cudaFree(cellGrpPrms->estH_fr));
  CUDA_CHECK_ERR(cudaFree(cellGrpPrms->cellId));
  delete cellGrpPrms;
  delete estH_fr;
  delete[] preset_cell_associate;
  return 0;
}