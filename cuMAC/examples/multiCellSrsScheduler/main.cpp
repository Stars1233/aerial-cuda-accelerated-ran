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
#include <cstdlib> // for rand() and srand()
#include <ctime> // for time()
#include <yaml-cpp/yaml.h>

using namespace cumac;

int main(int argc, char* argv[]) 
{
    if (argc < 2) 
    {
        std::cerr << "Usage: " << argv[0] << " <config.yaml>\n";
        return EXIT_FAILURE;
    }
    std::string config_file = argv[1];

    YAML::Node config = YAML::LoadFile(config_file);
    YAML::Node srs_last_tx_counter;
    YAML::Node mu_mimo_indication;
    YAML::Node srs_wideband_signal_energy;
    YAML::Node srs_tx_pwr;
    YAML::Node srs_wb_snr;
    YAML::Node srs_tpc_accumulation_flag;
    YAML::Node srs_tpc_adjustment_state;
    YAML::Node srs_pwr_zero;
    YAML::Node srs_pwr_alpha;
    
    int testing_mode = config["testing_mode"].as<int>();
    
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
    printf("cuMAC multi-cell SRS scheduler: Running on GPU device %d (total devices: %d)\n", 
           my_dev, deviceCount);
    //std::srand(std::time(0));

    // create stream  
    cudaStream_t cuStrmMain;
    CUDA_CHECK_ERR(cudaStreamCreate(&cuStrmMain));
    
    uint16_t nTest = 2048;
    if(testing_mode)
    {
        nTest = 1;
        srs_last_tx_counter = config["srs_last_tx_counter"];
        mu_mimo_indication = config["mu_mimo_indication"];
        srs_wideband_signal_energy = config["srs_wideband_signal_energy"];
        srs_tx_pwr = config["srs_tx_pwr"];
        srs_wb_snr = config["srs_wb_snr"];
        srs_tpc_accumulation_flag = config["srs_tpc_accumulation_flag"];
        srs_tpc_adjustment_state = config["srs_tpc_adjustment_state"];
        srs_pwr_zero = config["srs_pwr_zero"];
        srs_pwr_alpha = config["srs_pwr_alpha"];
    }
    for(uint16_t idxTest = 0 ; idxTest < nTest; idxTest++)
    {
    
    std::srand(idxTest*10);
    // create SRS scheduler obj
    cumac::mcSrsSchedulerHndl_t mcSrsSchedulerGpu = new cumac::multiCellSrsScheduler();

    //--- create API for testbench ----
    cumacSrsCellGrpPrms*     srsCellGrpPramsCpu    = new cumacSrsCellGrpPrms;
    cumacSrsCellGrpUeStatus* srsCellGrpUeStatusCpu = new cumacSrsCellGrpUeStatus;
    cumacSchdSol*            schdSolCpu            = new cumacSchdSol;
    cumacSrsSchdSol*         srsSchdSolCpu         = new cumacSrsSchdSol;
    cumacSrsSchdSol*         srsSchdSolCpuRef      = new cumacSrsSchdSol;
    
    uint16_t    nCell = config["n_cell"].as<uint16_t>();
    uint16_t    nMaxActUePerCell = config["n_max_active_ue_per_cell"].as<uint16_t>(); 
    uint16_t    nActiveUe = nMaxActUePerCell * nCell; 
    uint16_t    nSymbsPerSlot = 14; 
    uint8_t     nBsAnt = config["n_bs_ant"].as<uint8_t>();
    uint8_t     srsSchedulingSel = (std::rand() &0x1);
    if(testing_mode)
    {      
        srsSchedulingSel = config["srs_scheduling_sel"].as<uint8_t>();
    }
    
    srsCellGrpPramsCpu->nActiveUe = nActiveUe;
    srsCellGrpPramsCpu->nMaxActUePerCell = nMaxActUePerCell;
    srsCellGrpPramsCpu->nCell = nCell;
    srsCellGrpPramsCpu->nSymbsPerSlot = nSymbsPerSlot;
    srsCellGrpPramsCpu->nBsAnt = nBsAnt;
    srsCellGrpPramsCpu->srsSchedulingSel = srsSchedulingSel;
    
    srsCellGrpPramsCpu->cellId = new uint16_t[nCell];
    for(int cIdx = 0; cIdx < nCell; cIdx++)
    {
        srsCellGrpPramsCpu->cellId[cIdx] = cIdx;
    }
    
    srsCellGrpPramsCpu->cellAssocActUe = new uint8_t[nCell*nActiveUe];
    for (int cIdx = 0; cIdx < nCell; cIdx++) 
    {
        for (int uIdx = 0; uIdx < nActiveUe; uIdx++) 
        {
            int cellIdx = floor(static_cast<float>(uIdx)/static_cast<float>(nMaxActUePerCell));
            if (cellIdx == cIdx)
                srsCellGrpPramsCpu->cellAssocActUe[cIdx*nActiveUe + uIdx] = 1;
            else
                srsCellGrpPramsCpu->cellAssocActUe[cIdx*nActiveUe + uIdx] = 0;
        }
    }
   srsCellGrpUeStatusCpu->newDataActUe = new int8_t[nActiveUe];
   for(int uIdx = 0; uIdx < nActiveUe; uIdx++)
   {
       srsCellGrpUeStatusCpu->newDataActUe[uIdx] = 1;
   }
   
   uint32_t* srsLastTxCounterInput = new uint32_t[nActiveUe];
   for(int uIdx = 0; uIdx < nActiveUe; uIdx++)
   {
       if(testing_mode)
       {
           srsLastTxCounterInput[uIdx] = srs_last_tx_counter[uIdx].as<uint32_t>();
       }
       else
       {
           srsLastTxCounterInput[uIdx] = (uint32_t)std::rand() % 20;    
       }
   }
   
   srsCellGrpUeStatusCpu->srsLastTxCounter = new uint32_t[nActiveUe];
   for(int uIdx = 0; uIdx < nActiveUe; uIdx++)
   {
       srsCellGrpUeStatusCpu->srsLastTxCounter[uIdx] = srsLastTxCounterInput[uIdx];
   }
   
   srsCellGrpUeStatusCpu->srsNumAntPorts = new uint8_t[nActiveUe];
   for(int uIdx = 0; uIdx < nActiveUe; uIdx++)
   {
       srsCellGrpUeStatusCpu->srsNumAntPorts[uIdx] = 4;
   }
   
   srsCellGrpUeStatusCpu->srsResourceType = new uint8_t[nActiveUe];
   for(int uIdx = 0; uIdx < nActiveUe; uIdx++)
   {
       srsCellGrpUeStatusCpu->srsResourceType[uIdx] = 0;
   }
   
   srsCellGrpUeStatusCpu->srsWbSnr = new float[nActiveUe];
   std::default_random_engine generator;
   std::uniform_real_distribution<float> distributionSrsWbSnr(-5.0f,15.0f); 
   for(int uIdx = 0; uIdx < nActiveUe; uIdx++)
   {
       if(testing_mode)
       {
           srsCellGrpUeStatusCpu->srsWbSnr[uIdx] = srs_wb_snr[uIdx].as<float>();
       }
       else
       {
           srsCellGrpUeStatusCpu->srsWbSnr[uIdx] = distributionSrsWbSnr(generator);
       }
   }
   
   srsCellGrpUeStatusCpu->srsWbSnrThreshold = new float[nActiveUe];
   for(int uIdx = 0; uIdx < nActiveUe; uIdx++)
   {
       if(testing_mode)
       {
           srsCellGrpUeStatusCpu->srsWbSnrThreshold[uIdx] = config["srs_wb_snr_th"].as<float>();
       }
       else
       {
           srsCellGrpUeStatusCpu->srsWbSnrThreshold[uIdx] = 5.0f;
       }
   } 
   
   srsCellGrpUeStatusCpu->srsWidebandSignalEnergy = new float[nActiveUe]; 
   std::uniform_real_distribution<float> distributionSrsWbSignelEnergy(0.0f,10.0f); 
   for(int uIdx = 0; uIdx < nActiveUe; uIdx++)
   {
       if(testing_mode)
       {
           srsCellGrpUeStatusCpu->srsWidebandSignalEnergy[uIdx] = srs_wideband_signal_energy[uIdx].as<float>();
       }
       else
       {
           srsCellGrpUeStatusCpu->srsWidebandSignalEnergy[uIdx] = distributionSrsWbSignelEnergy(generator);
       }
   } 
   
   srsCellGrpUeStatusCpu->srsTxPwrMax = new float[nActiveUe]; 
   for(int uIdx = 0; uIdx < nActiveUe; uIdx++)
   {
       srsCellGrpUeStatusCpu->srsTxPwrMax[uIdx] = 23.0f;
   }
   
   srsCellGrpUeStatusCpu->srsPwr0 = new float[nActiveUe];
   for(int uIdx = 0; uIdx < nActiveUe; uIdx++)
   {
       if(testing_mode)
       {
           srsCellGrpUeStatusCpu->srsPwr0[uIdx] = srs_pwr_zero[uIdx].as<float>();
       }
       else
       {
           srsCellGrpUeStatusCpu->srsPwr0[uIdx] = -10.0f;
       }
   } 
   
   srsCellGrpUeStatusCpu->srsPwrAlpha = new float[nActiveUe];
   for(int uIdx = 0; uIdx < nActiveUe; uIdx++)
   {
       if(testing_mode)
       {
           srsCellGrpUeStatusCpu->srsPwrAlpha[uIdx] = srs_pwr_alpha[uIdx].as<float>();
       }
       else
       {
           srsCellGrpUeStatusCpu->srsPwrAlpha[uIdx] = ((uint32_t)std::rand() % 5)*0.2f;
       }
   }
   
   srsCellGrpUeStatusCpu->srsTpcAccumulationFlag = new uint8_t[nActiveUe];
   for(int uIdx = 0; uIdx < nActiveUe; uIdx++)
   {
       if(testing_mode)
       {
           srsCellGrpUeStatusCpu->srsTpcAccumulationFlag[uIdx] = srs_tpc_accumulation_flag[uIdx].as<uint8_t>();
       }
       else
       {
           srsCellGrpUeStatusCpu->srsTpcAccumulationFlag[uIdx] = (std::rand() &0x1);
       }
   }
   
   float* srsPowerControlAdjustmentStateInput = new float[nActiveUe];
   for(int uIdx = 0; uIdx < nActiveUe; uIdx++)
   {   
       if(testing_mode)
       {
           srsPowerControlAdjustmentStateInput[uIdx] = srs_tpc_adjustment_state[uIdx].as<float>();
       }
       else
       {
           srsPowerControlAdjustmentStateInput[uIdx] = ((uint32_t)std::rand() % 20)*1.0f - 10.0f;
       }
   }
   srsCellGrpUeStatusCpu->srsPowerControlAdjustmentState = new float[nActiveUe];
   for(int uIdx = 0; uIdx < nActiveUe; uIdx++)
   {
       srsCellGrpUeStatusCpu->srsPowerControlAdjustmentState[uIdx] = srsPowerControlAdjustmentStateInput[uIdx];
   }

   srsCellGrpUeStatusCpu->srsPowerHeadroomReport = new uint8_t[nActiveUe];
   std::fill_n(srsCellGrpUeStatusCpu->srsPowerHeadroomReport, nActiveUe, 0);
   
   schdSolCpu->muMimoInd = new uint8_t[nActiveUe];
   for(int uIdx = 0; uIdx < nActiveUe; uIdx++)
   {
       if(testing_mode)
       {
           schdSolCpu->muMimoInd[uIdx] = mu_mimo_indication[uIdx].as<uint8_t>();
       }
       else
       {
           schdSolCpu->muMimoInd[uIdx] = (std::rand() &0x1);
       }
   }

   CUDA_CHECK_ERR(cudaMallocHost((void **)&schdSolCpu->sortedUeList, sizeof(uint16_t*)*nCell));
   for (int cIdx = 0; cIdx < nCell; cIdx++) {
       CUDA_CHECK_ERR(cudaMallocHost((void **)&schdSolCpu->sortedUeList[cIdx], sizeof(uint16_t)*nMaxActUePerCell));
   }
   
   for (int cIdx = 0; cIdx < nCell; cIdx++) 
   {
       std::vector<int> sortedUeVec;
       for (uint16_t uIdx=0; uIdx<nMaxActUePerCell; uIdx++) sortedUeVec.push_back(uIdx); 
       std::random_shuffle(sortedUeVec.begin(), sortedUeVec.end());
       
       for(int uIdx = 0; uIdx < nMaxActUePerCell; uIdx++)
       {
           schdSolCpu->sortedUeList[cIdx][uIdx] = sortedUeVec[uIdx] + cIdx * nMaxActUePerCell;
       }
   }
   
   srsSchdSolCpu->nSrsScheduledUePerCell = new uint16_t[nCell];
   srsSchdSolCpu->srsTxUe = new uint16_t[nActiveUe];
   srsSchdSolCpu->srsRxCell = new uint16_t[nActiveUe];
   srsSchdSolCpu->srsNumSymb = new uint8_t[nActiveUe];
   srsSchdSolCpu->srsTimeStart = new uint8_t[nActiveUe];
   srsSchdSolCpu->srsNumRep = new uint8_t[nActiveUe];
   srsSchdSolCpu->srsConfigIndex = new uint8_t[nActiveUe];
   srsSchdSolCpu->srsBwIndex = new uint8_t[nActiveUe];
   srsSchdSolCpu->srsCombSize = new uint8_t[nActiveUe];   
   srsSchdSolCpu->srsCombOffset = new uint8_t[nActiveUe];
   srsSchdSolCpu->srsFreqStart = new uint8_t[nActiveUe];
   srsSchdSolCpu->srsFreqShift = new uint16_t[nActiveUe];
   srsSchdSolCpu->srsFreqHopping = new uint8_t[nActiveUe];
   srsSchdSolCpu->srsSequenceId = new uint16_t[nActiveUe];
   srsSchdSolCpu->srsGroupOrSequenceHopping = new uint8_t[nActiveUe];
   srsSchdSolCpu->srsCyclicShift = new uint8_t[nActiveUe];
   srsSchdSolCpu->srsTxPwr = new float[nActiveUe];
   
   uint8_t* srsConfigIndexInput = new uint8_t[nActiveUe];
   uint8_t* srsBwIndexInput = new uint8_t[nActiveUe];
   float* srsTxPwrInput = new float[nActiveUe];
   std::uniform_real_distribution<float> distributionSrsTxPwrInput(10.0f,23.0f); 
   for(int uIdx = 0; uIdx < nActiveUe; uIdx++)
   {
       srsConfigIndexInput[uIdx] = 63;
       srsBwIndexInput[uIdx] = 0;
       if(testing_mode)
       {
            srsTxPwrInput[uIdx] = srs_tx_pwr[uIdx].as<float>();
       }
       else
       {
           srsTxPwrInput[uIdx] = distributionSrsTxPwrInput(generator);
       }
   }
   
   for(int uIdx = 0; uIdx < nActiveUe; uIdx++)
   {
       srsSchdSolCpu->srsConfigIndex[uIdx] = srsConfigIndexInput[uIdx];
       srsSchdSolCpu->srsBwIndex[uIdx] = srsBwIndexInput[uIdx];
       srsSchdSolCpu->srsTxPwr[uIdx] = srsTxPwrInput[uIdx];
   }
   
   srsSchdSolCpuRef->nSrsScheduledUePerCell = new uint16_t[nCell];
   srsSchdSolCpuRef->srsTxUe = new uint16_t[nActiveUe];
   srsSchdSolCpuRef->srsRxCell = new uint16_t[nActiveUe];
   srsSchdSolCpuRef->srsNumSymb = new uint8_t[nActiveUe];
   srsSchdSolCpuRef->srsTimeStart = new uint8_t[nActiveUe];
   srsSchdSolCpuRef->srsNumRep = new uint8_t[nActiveUe];
   srsSchdSolCpuRef->srsConfigIndex = new uint8_t[nActiveUe];
   srsSchdSolCpuRef->srsBwIndex = new uint8_t[nActiveUe];
   srsSchdSolCpuRef->srsCombSize = new uint8_t[nActiveUe];   
   srsSchdSolCpuRef->srsCombOffset = new uint8_t[nActiveUe];
   srsSchdSolCpuRef->srsFreqStart = new uint8_t[nActiveUe];
   srsSchdSolCpuRef->srsFreqShift = new uint16_t[nActiveUe];
   srsSchdSolCpuRef->srsFreqHopping = new uint8_t[nActiveUe];
   srsSchdSolCpuRef->srsSequenceId = new uint16_t[nActiveUe];
   srsSchdSolCpuRef->srsGroupOrSequenceHopping = new uint8_t[nActiveUe];
   srsSchdSolCpuRef->srsCyclicShift = new uint8_t[nActiveUe];
   srsSchdSolCpuRef->srsTxPwr = new float[nActiveUe];
   ////////////////////////////////////////////////////////////////////
   cumacSrsCellGrpPrms*     srsCellGrpPramsGpu    = new cumacSrsCellGrpPrms;
   cumacSrsCellGrpUeStatus* srsCellGrpUeStatusGpu = new cumacSrsCellGrpUeStatus;
   cumacSchdSol*            schdSolGpu            = new cumacSchdSol;
   cumacSrsSchdSol*         srsSchdSolGpu         = new cumacSrsSchdSol;
   
   srsCellGrpPramsGpu->nActiveUe = nActiveUe;
   srsCellGrpPramsGpu->nMaxActUePerCell = nMaxActUePerCell;
   srsCellGrpPramsGpu->nCell = nCell;
   srsCellGrpPramsGpu->nSymbsPerSlot = nSymbsPerSlot;
   srsCellGrpPramsGpu->nBsAnt = nBsAnt;
   srsCellGrpPramsGpu->srsSchedulingSel= srsSchedulingSel;
   
   CUDA_CHECK_ERR(cudaMalloc((void **)&srsCellGrpPramsGpu->cellId, sizeof(uint16_t)*nCell));
   CUDA_CHECK_ERR(cudaMalloc((void **)&srsCellGrpPramsGpu->cellAssocActUe, nCell*nActiveUe));
   CUDA_CHECK_ERR(cudaMemcpyAsync(srsCellGrpPramsGpu->cellId, srsCellGrpPramsCpu->cellId, sizeof(uint16_t)*nCell, cudaMemcpyHostToDevice, cuStrmMain));
   CUDA_CHECK_ERR(cudaMemcpyAsync(srsCellGrpPramsGpu->cellAssocActUe, srsCellGrpPramsCpu->cellAssocActUe, nCell*nActiveUe, cudaMemcpyHostToDevice, cuStrmMain));
   CUDA_CHECK_ERR(cudaStreamSynchronize(cuStrmMain));
   
   CUDA_CHECK_ERR(cudaMalloc((void **)&srsCellGrpUeStatusGpu->newDataActUe, nActiveUe));
   CUDA_CHECK_ERR(cudaMalloc((void **)&srsCellGrpUeStatusGpu->srsLastTxCounter, sizeof(uint32_t)*nActiveUe));
   CUDA_CHECK_ERR(cudaMalloc((void **)&srsCellGrpUeStatusGpu->srsNumAntPorts, nActiveUe));
   CUDA_CHECK_ERR(cudaMalloc((void **)&srsCellGrpUeStatusGpu->srsResourceType, nActiveUe));
   CUDA_CHECK_ERR(cudaMalloc((void **)&srsCellGrpUeStatusGpu->srsWbSnr, sizeof(float)*nActiveUe));
   CUDA_CHECK_ERR(cudaMalloc((void **)&srsCellGrpUeStatusGpu->srsWbSnrThreshold, sizeof(float)*nActiveUe));
   CUDA_CHECK_ERR(cudaMalloc((void **)&srsCellGrpUeStatusGpu->srsWidebandSignalEnergy, sizeof(float)*nActiveUe));
   CUDA_CHECK_ERR(cudaMalloc((void **)&srsCellGrpUeStatusGpu->srsTxPwrMax, sizeof(float)*nActiveUe));
   CUDA_CHECK_ERR(cudaMalloc((void **)&srsCellGrpUeStatusGpu->srsPwr0, sizeof(float)*nActiveUe));
   CUDA_CHECK_ERR(cudaMalloc((void **)&srsCellGrpUeStatusGpu->srsPwrAlpha, sizeof(float)*nActiveUe));
   CUDA_CHECK_ERR(cudaMalloc((void **)&srsCellGrpUeStatusGpu->srsTpcAccumulationFlag, nActiveUe));
   CUDA_CHECK_ERR(cudaMalloc((void **)&srsCellGrpUeStatusGpu->srsPowerControlAdjustmentState, sizeof(float)*nActiveUe));
   CUDA_CHECK_ERR(cudaMalloc((void **)&srsCellGrpUeStatusGpu->srsPowerHeadroomReport, nActiveUe));
   CUDA_CHECK_ERR(cudaMemcpyAsync(srsCellGrpUeStatusGpu->newDataActUe, srsCellGrpUeStatusCpu->newDataActUe, nActiveUe, cudaMemcpyHostToDevice, cuStrmMain));
   CUDA_CHECK_ERR(cudaMemcpyAsync(srsCellGrpUeStatusGpu->srsLastTxCounter, srsCellGrpUeStatusCpu->srsLastTxCounter, sizeof(uint32_t)*nActiveUe, cudaMemcpyHostToDevice, cuStrmMain));
   CUDA_CHECK_ERR(cudaMemcpyAsync(srsCellGrpUeStatusGpu->srsNumAntPorts, srsCellGrpUeStatusCpu->srsNumAntPorts, nActiveUe, cudaMemcpyHostToDevice, cuStrmMain));
   CUDA_CHECK_ERR(cudaMemcpyAsync(srsCellGrpUeStatusGpu->srsResourceType, srsCellGrpUeStatusCpu->srsResourceType, nActiveUe, cudaMemcpyHostToDevice, cuStrmMain));
   CUDA_CHECK_ERR(cudaMemcpyAsync(srsCellGrpUeStatusGpu->srsWbSnr, srsCellGrpUeStatusCpu->srsWbSnr, sizeof(float)*nActiveUe, cudaMemcpyHostToDevice, cuStrmMain));
   CUDA_CHECK_ERR(cudaMemcpyAsync(srsCellGrpUeStatusGpu->srsWbSnrThreshold, srsCellGrpUeStatusCpu->srsWbSnrThreshold, sizeof(float)*nActiveUe, cudaMemcpyHostToDevice, cuStrmMain));
   CUDA_CHECK_ERR(cudaMemcpyAsync(srsCellGrpUeStatusGpu->srsWidebandSignalEnergy, srsCellGrpUeStatusCpu->srsWidebandSignalEnergy, sizeof(float)*nActiveUe, cudaMemcpyHostToDevice, cuStrmMain));
   CUDA_CHECK_ERR(cudaMemcpyAsync(srsCellGrpUeStatusGpu->srsTxPwrMax, srsCellGrpUeStatusCpu->srsTxPwrMax, sizeof(float)*nActiveUe, cudaMemcpyHostToDevice, cuStrmMain));
   CUDA_CHECK_ERR(cudaMemcpyAsync(srsCellGrpUeStatusGpu->srsPwr0, srsCellGrpUeStatusCpu->srsPwr0, sizeof(float)*nActiveUe, cudaMemcpyHostToDevice, cuStrmMain));
   CUDA_CHECK_ERR(cudaMemcpyAsync(srsCellGrpUeStatusGpu->srsPwrAlpha, srsCellGrpUeStatusCpu->srsPwrAlpha, sizeof(float)*nActiveUe, cudaMemcpyHostToDevice, cuStrmMain));
   CUDA_CHECK_ERR(cudaMemcpyAsync(srsCellGrpUeStatusGpu->srsTpcAccumulationFlag, srsCellGrpUeStatusCpu->srsTpcAccumulationFlag, nActiveUe, cudaMemcpyHostToDevice, cuStrmMain));
   CUDA_CHECK_ERR(cudaMemcpyAsync(srsCellGrpUeStatusGpu->srsPowerControlAdjustmentState, srsCellGrpUeStatusCpu->srsPowerControlAdjustmentState, sizeof(float)*nActiveUe, cudaMemcpyHostToDevice, cuStrmMain));
   CUDA_CHECK_ERR(cudaMemcpyAsync(srsCellGrpUeStatusGpu->srsPowerHeadroomReport, srsCellGrpUeStatusCpu->srsPowerHeadroomReport, nActiveUe, cudaMemcpyHostToDevice, cuStrmMain));
   CUDA_CHECK_ERR(cudaStreamSynchronize(cuStrmMain));
   
   CUDA_CHECK_ERR(cudaMalloc((void **)&schdSolGpu->muMimoInd, nActiveUe));
   CUDA_CHECK_ERR(cudaMemcpyAsync(schdSolGpu->muMimoInd, schdSolCpu->muMimoInd, nActiveUe, cudaMemcpyHostToDevice, cuStrmMain));
   CUDA_CHECK_ERR(cudaMalloc((void **)&schdSolGpu->sortedUeList, sizeof(uint16_t*)*nCell));
   CUDA_CHECK_ERR(cudaMemcpyAsync(schdSolGpu->sortedUeList, schdSolCpu->sortedUeList, sizeof(uint16_t*)*nCell, cudaMemcpyHostToDevice, cuStrmMain));
   
   CUDA_CHECK_ERR(cudaMalloc((void **)&srsSchdSolGpu->nSrsScheduledUePerCell, sizeof(uint16_t)*nActiveUe));
   CUDA_CHECK_ERR(cudaMalloc((void **)&srsSchdSolGpu->srsTxUe, sizeof(uint16_t)*nMaxActUePerCell*nCell));
   CUDA_CHECK_ERR(cudaMalloc((void **)&srsSchdSolGpu->srsRxCell, sizeof(uint16_t)*nActiveUe));
   CUDA_CHECK_ERR(cudaMalloc((void **)&srsSchdSolGpu->srsNumSymb, nActiveUe));
   CUDA_CHECK_ERR(cudaMalloc((void **)&srsSchdSolGpu->srsTimeStart, nActiveUe));
   CUDA_CHECK_ERR(cudaMalloc((void **)&srsSchdSolGpu->srsNumRep, nActiveUe));
   CUDA_CHECK_ERR(cudaMalloc((void **)&srsSchdSolGpu->srsConfigIndex, nActiveUe));
   CUDA_CHECK_ERR(cudaMalloc((void **)&srsSchdSolGpu->srsBwIndex, nActiveUe));
   CUDA_CHECK_ERR(cudaMalloc((void **)&srsSchdSolGpu->srsCombSize, nActiveUe));
   CUDA_CHECK_ERR(cudaMalloc((void **)&srsSchdSolGpu->srsCombOffset, nActiveUe));
   CUDA_CHECK_ERR(cudaMalloc((void **)&srsSchdSolGpu->srsFreqStart, nActiveUe));
   CUDA_CHECK_ERR(cudaMalloc((void **)&srsSchdSolGpu->srsFreqShift, sizeof(uint16_t)*nActiveUe));
   CUDA_CHECK_ERR(cudaMalloc((void **)&srsSchdSolGpu->srsFreqHopping, nActiveUe));
   CUDA_CHECK_ERR(cudaMalloc((void **)&srsSchdSolGpu->srsSequenceId, sizeof(uint16_t)*nActiveUe));
   CUDA_CHECK_ERR(cudaMalloc((void **)&srsSchdSolGpu->srsGroupOrSequenceHopping, nActiveUe));
   CUDA_CHECK_ERR(cudaMalloc((void **)&srsSchdSolGpu->srsCyclicShift, nActiveUe));
   CUDA_CHECK_ERR(cudaMalloc((void **)&srsSchdSolGpu->srsTxPwr, sizeof(float)*nActiveUe));
   
   CUDA_CHECK_ERR(cudaMemcpyAsync(srsSchdSolGpu->srsConfigIndex, srsSchdSolCpu->srsConfigIndex, nActiveUe, cudaMemcpyHostToDevice, cuStrmMain));
   CUDA_CHECK_ERR(cudaMemcpyAsync(srsSchdSolGpu->srsBwIndex, srsSchdSolCpu->srsBwIndex, nActiveUe, cudaMemcpyHostToDevice, cuStrmMain));
   CUDA_CHECK_ERR(cudaMemcpyAsync(srsSchdSolGpu->srsTxPwr, srsSchdSolCpu->srsTxPwr, sizeof(float)*nActiveUe, cudaMemcpyHostToDevice, cuStrmMain));
   //////////////////////////////////////////////////////////////

   mcSrsSchedulerGpu->setup(srsCellGrpUeStatusGpu, srsCellGrpPramsGpu, schdSolGpu, srsSchdSolGpu, cuStrmMain);
   CUDA_CHECK_ERR(cudaStreamSynchronize(cuStrmMain));
   printf("GPU SRS scheduler setup completed\n");

   mcSrsSchedulerGpu->run(cuStrmMain);
   CUDA_CHECK_ERR(cudaStreamSynchronize(cuStrmMain));
   printf("GPU SRS scheduler run completed\n");
   
   CUDA_CHECK_ERR(cudaMemcpyAsync(srsSchdSolCpu->srsTxUe, srsSchdSolGpu->srsTxUe, sizeof(uint16_t)*nMaxActUePerCell*nCell, cudaMemcpyDeviceToHost, cuStrmMain));
   CUDA_CHECK_ERR(cudaMemcpyAsync(srsSchdSolCpu->srsRxCell, srsSchdSolGpu->srsRxCell, sizeof(uint16_t)*nActiveUe, cudaMemcpyDeviceToHost, cuStrmMain));
   CUDA_CHECK_ERR(cudaMemcpyAsync(srsCellGrpUeStatusCpu->srsLastTxCounter, srsCellGrpUeStatusGpu->srsLastTxCounter, sizeof(uint32_t)*nActiveUe, cudaMemcpyDeviceToHost, cuStrmMain));
   CUDA_CHECK_ERR(cudaMemcpyAsync(srsSchdSolCpu->srsNumSymb, srsSchdSolGpu->srsNumSymb, nActiveUe, cudaMemcpyDeviceToHost, cuStrmMain));
   CUDA_CHECK_ERR(cudaMemcpyAsync(srsSchdSolCpu->srsTimeStart, srsSchdSolGpu->srsTimeStart, nActiveUe, cudaMemcpyDeviceToHost, cuStrmMain));
   CUDA_CHECK_ERR(cudaMemcpyAsync(srsSchdSolCpu->srsNumRep, srsSchdSolGpu->srsNumRep, nActiveUe, cudaMemcpyDeviceToHost, cuStrmMain));
   CUDA_CHECK_ERR(cudaMemcpyAsync(srsSchdSolCpu->srsSequenceId, srsSchdSolGpu->srsSequenceId, sizeof(uint16_t)*nActiveUe, cudaMemcpyDeviceToHost, cuStrmMain));
   CUDA_CHECK_ERR(cudaMemcpyAsync(srsSchdSolCpu->srsGroupOrSequenceHopping, srsSchdSolGpu->srsGroupOrSequenceHopping, nActiveUe, cudaMemcpyDeviceToHost, cuStrmMain));
   CUDA_CHECK_ERR(cudaMemcpyAsync(srsSchdSolCpu->srsFreqHopping, srsSchdSolGpu->srsFreqHopping, nActiveUe, cudaMemcpyDeviceToHost, cuStrmMain));
   CUDA_CHECK_ERR(cudaMemcpyAsync(srsSchdSolCpu->srsCombSize, srsSchdSolGpu->srsCombSize, nActiveUe, cudaMemcpyDeviceToHost, cuStrmMain));
   CUDA_CHECK_ERR(cudaMemcpyAsync(srsSchdSolCpu->srsCombOffset, srsSchdSolGpu->srsCombOffset, nActiveUe, cudaMemcpyDeviceToHost, cuStrmMain));
   CUDA_CHECK_ERR(cudaMemcpyAsync(srsSchdSolCpu->srsConfigIndex, srsSchdSolGpu->srsConfigIndex, nActiveUe, cudaMemcpyDeviceToHost, cuStrmMain));
   CUDA_CHECK_ERR(cudaMemcpyAsync(srsSchdSolCpu->srsBwIndex, srsSchdSolGpu->srsBwIndex, nActiveUe, cudaMemcpyDeviceToHost, cuStrmMain));
   CUDA_CHECK_ERR(cudaMemcpyAsync(srsSchdSolCpu->srsFreqStart, srsSchdSolGpu->srsFreqStart, nActiveUe, cudaMemcpyDeviceToHost, cuStrmMain));
   CUDA_CHECK_ERR(cudaMemcpyAsync(srsSchdSolCpu->srsFreqShift, srsSchdSolGpu->srsFreqShift, sizeof(uint16_t)*nActiveUe, cudaMemcpyDeviceToHost, cuStrmMain));
   CUDA_CHECK_ERR(cudaMemcpyAsync(srsSchdSolCpu->nSrsScheduledUePerCell, srsSchdSolGpu->nSrsScheduledUePerCell, sizeof(uint16_t)*nCell, cudaMemcpyDeviceToHost, cuStrmMain));
   CUDA_CHECK_ERR(cudaMemcpyAsync(srsSchdSolCpu->srsCyclicShift, srsSchdSolGpu->srsCyclicShift, nActiveUe, cudaMemcpyDeviceToHost, cuStrmMain));
   CUDA_CHECK_ERR(cudaMemcpyAsync(srsSchdSolCpu->srsTxPwr, srsSchdSolGpu->srsTxPwr, sizeof(float)*nActiveUe, cudaMemcpyDeviceToHost, cuStrmMain));
   CUDA_CHECK_ERR(cudaStreamSynchronize(cuStrmMain));
   
//   for(int idx=0; idx < nCell; idx++)
//   {
//       printf("[%d]nSrsScheduledUePerCell[%d]\n", idx, srsSchdSolCpu->nSrsScheduledUePerCell[idx]);
//   }
//   for(int idx=0; idx<nActiveUe; idx++)
//   {
//       printf("[%d]srsTxUe[%d]\n", idx, srsSchdSolCpu->srsTxUe[idx]);
//   }
//   
//   for(int idx=0; idx<nActiveUe; idx++)
//   {
//       printf("uIdx[%d]RxCell[%d]LastTxCounter[%d]nSymb[%d]TimeStart[%d]NumRep[%d]SId[%d]GOSH[%d]FH[%d]CombSize[%d]CombOffset[%d]ConfIdx[%d]BwIdx[%d]FStart[%d]FShift[%d]CyclicShift[%d]\n", 
//                                                                                        idx, 
//                                                                                        srsSchdSolCpu->srsRxCell[idx], 
//                                                                                        srsCellGrpUeStatusCpu->srsLastTxCounter[idx], 
//                                                                                        srsSchdSolCpu->srsNumSymb[idx], 
//                                                                                        srsSchdSolCpu->srsTimeStart[idx], 
//                                                                                        srsSchdSolCpu->srsNumRep[idx],
//                                                                                        srsSchdSolCpu->srsSequenceId[idx],
//                                                                                        srsSchdSolCpu->srsGroupOrSequenceHopping[idx],
//                                                                                        srsSchdSolCpu->srsFreqHopping[idx],
//                                                                                        srsSchdSolCpu->srsCombSize[idx],
//                                                                                        srsSchdSolCpu->srsCombOffset[idx],
//                                                                                        srsSchdSolCpu->srsConfigIndex[idx],
//                                                                                        srsSchdSolCpu->srsBwIndex[idx],
//                                                                                        srsSchdSolCpu->srsFreqStart[idx],
//                                                                                        srsSchdSolCpu->srsFreqShift[idx],
//                                                                                        srsSchdSolCpu->srsCyclicShift[idx]);
//   }

   //// store srsLastTxCounter from GPU
   uint32_t* srsLastTxCounterGpu = new uint32_t[nActiveUe];
   for(int uIdx = 0; uIdx < nActiveUe; uIdx++)
   {
       srsLastTxCounterGpu[uIdx] = srsCellGrpUeStatusCpu->srsLastTxCounter[uIdx];
   }
   
   //// reset for CPU reference model
   for(int uIdx = 0; uIdx < nActiveUe; uIdx++)
   {
       srsCellGrpUeStatusCpu->srsLastTxCounter[uIdx] = srsLastTxCounterInput[uIdx];
       srsCellGrpUeStatusCpu->srsPowerControlAdjustmentState[uIdx] = srsPowerControlAdjustmentStateInput[uIdx];
       
       srsSchdSolCpuRef->srsConfigIndex[uIdx] = srsConfigIndexInput[uIdx];
       srsSchdSolCpuRef->srsBwIndex[uIdx] = srsBwIndexInput[uIdx];
       srsSchdSolCpuRef->srsTxPwr[uIdx] = srsTxPwrInput[uIdx];
   }
   
   if((nBsAnt == 64)&&(srsSchedulingSel==1))
       mcSrsSchedulerGpu->cpuScheduler_v1(srsCellGrpUeStatusCpu, srsCellGrpPramsCpu, schdSolCpu, srsSchdSolCpuRef); 
   else
       mcSrsSchedulerGpu->cpuScheduler_v0(srsCellGrpUeStatusCpu, srsCellGrpPramsCpu, srsSchdSolCpuRef); 
   
   bool testRes = true;
   
   for(uint16_t cIdx = 0; cIdx < nCell; cIdx++)
   {
       if(srsSchdSolCpu->nSrsScheduledUePerCell[cIdx]!=srsSchdSolCpuRef->nSrsScheduledUePerCell[cIdx])
       {
           printf("There is a mismatch for the number of the scheduled SRS UEs for cell %d: GPU result %d, CPU result %d.\n", cIdx, srsSchdSolCpu->nSrsScheduledUePerCell[cIdx], srsSchdSolCpuRef->nSrsScheduledUePerCell[cIdx]);
           testRes = false;
           break;
       }
       
       if(!testRes)
           break;
       
       for(uint16_t uIdx = 0; uIdx < srsSchdSolCpuRef->nSrsScheduledUePerCell[cIdx]; uIdx++)
       {
           if(srsSchdSolCpu->srsTxUe[uIdx+cIdx*nMaxActUePerCell]!=srsSchdSolCpuRef->srsTxUe[uIdx+cIdx*nMaxActUePerCell])
           {
               printf("There is a mismatch for the scheduled UE %d in cell %d: GPU result %d, CPU result %d.\n", uIdx, cIdx, srsSchdSolCpu->srsTxUe[uIdx+cIdx*nMaxActUePerCell], srsSchdSolCpuRef->srsTxUe[uIdx+cIdx*nMaxActUePerCell]);
               testRes = false;
               break;
           }
       }
       
       if(!testRes)
           break;
       
       for(uint16_t uIdx = 0; uIdx < srsSchdSolCpuRef->nSrsScheduledUePerCell[cIdx]; uIdx++)
       {
           uint16_t ueIdx = srsSchdSolCpu->srsTxUe[uIdx+cIdx*nMaxActUePerCell];
           
           if(srsSchdSolCpu->srsTimeStart[ueIdx]!=srsSchdSolCpuRef->srsTimeStart[ueIdx])
           {
               printf("There is a mismatch of srsTimeStart for the scheduled UE %d in cell %d: GPU result %d, CPU result %d.\n", ueIdx, cIdx, srsSchdSolCpu->srsTimeStart[ueIdx], srsSchdSolCpuRef->srsTimeStart[ueIdx]);
               testRes = false;
               break;
           }
           
       }
       
       if(!testRes)
           break;
           
       for(uint16_t uIdx = 0; uIdx < srsSchdSolCpuRef->nSrsScheduledUePerCell[cIdx]; uIdx++)
       {
           uint16_t ueIdx = srsSchdSolCpu->srsTxUe[uIdx+cIdx*nMaxActUePerCell];
           
           if(srsSchdSolCpu->srsCombOffset[ueIdx]!=srsSchdSolCpuRef->srsCombOffset[ueIdx])
           {
               printf("There is a mismatch of srsCombOffset for the scheduled UE %d in cell %d: GPU result %d, CPU result %d.\n", ueIdx, cIdx, srsSchdSolCpu->srsCombOffset[ueIdx], srsSchdSolCpuRef->srsCombOffset[ueIdx]);
               testRes = false;
               break;
           }
           
       }
       
       if(!testRes)
           break;
       
       for(uint16_t uIdx = 0; uIdx < srsSchdSolCpuRef->nSrsScheduledUePerCell[cIdx]; uIdx++)
       {
           uint16_t ueIdx = srsSchdSolCpu->srsTxUe[uIdx+cIdx*nMaxActUePerCell];
           
           if(std::fabs(srsSchdSolCpu->srsTxPwr[ueIdx]-srsSchdSolCpuRef->srsTxPwr[ueIdx])>0.001f)
           {
               printf("There is a mismatch of srsTxPwr for the scheduled UE %d in cell %d: GPU result %f, CPU result %f.\n", ueIdx, cIdx, srsSchdSolCpu->srsTxPwr[ueIdx], srsSchdSolCpuRef->srsTxPwr[ueIdx]);
               testRes = false;
               break;
           }
       }
       
       if(!testRes)
           break;
   }
   
   if(testRes)
   {
       for(int uIdx = 0; uIdx < nActiveUe; uIdx++)
       {
           if(srsLastTxCounterGpu[uIdx] != srsCellGrpUeStatusCpu->srsLastTxCounter[uIdx])
           {
               printf("There is a mismatch of srsLastTxCounterGpu for UE %d: GPU result %d, CPU result %d.\n", uIdx, srsLastTxCounterGpu[uIdx], srsCellGrpUeStatusCpu->srsLastTxCounter[uIdx]);
               testRes = false;
               break;
           }
       }
   }
   
   if(testRes)
   {
       printf("The test for SRS scheduler passes for nBsAnt = %d srsSchedulingSel = %d!!!\n", nBsAnt, srsSchedulingSel);
   }
   else
   {
       printf("The test for SRS scheduler fails for nBsAnt = %d srsSchedulingSel = %d!!!\n", nBsAnt, srsSchedulingSel);
       assert(testRes==true);
   }
   
   cudaFree(srsCellGrpPramsGpu->cellId);
   cudaFree(srsCellGrpPramsGpu->cellAssocActUe);
   
   cudaFree(srsCellGrpUeStatusGpu->newDataActUe);
   cudaFree(srsCellGrpUeStatusGpu->srsLastTxCounter);
   cudaFree(srsCellGrpUeStatusGpu->srsNumAntPorts);
   cudaFree(srsCellGrpUeStatusGpu->srsResourceType);
   cudaFree(srsCellGrpUeStatusGpu->srsWbSnr);
   cudaFree(srsCellGrpUeStatusGpu->srsWbSnrThreshold);
   cudaFree(srsCellGrpUeStatusGpu->srsWidebandSignalEnergy);
   cudaFree(srsCellGrpUeStatusGpu->srsTxPwrMax);
   cudaFree(srsCellGrpUeStatusGpu->srsPwr0);
   cudaFree(srsCellGrpUeStatusGpu->srsPwrAlpha);
   cudaFree(srsCellGrpUeStatusGpu->srsTpcAccumulationFlag);
   cudaFree(srsCellGrpUeStatusGpu->srsPowerControlAdjustmentState);
   cudaFree(srsCellGrpUeStatusGpu->srsPowerHeadroomReport);
   
   cudaFree(schdSolGpu->muMimoInd);
   cudaFree(schdSolGpu->sortedUeList);
   
   cudaFree(srsSchdSolGpu->nSrsScheduledUePerCell);
   cudaFree(srsSchdSolGpu->srsTxUe);
   cudaFree(srsSchdSolGpu->srsRxCell);
   cudaFree(srsSchdSolGpu->srsNumSymb);
   cudaFree(srsSchdSolGpu->srsTimeStart);
   cudaFree(srsSchdSolGpu->srsNumRep);
   cudaFree(srsSchdSolGpu->srsConfigIndex);
   cudaFree(srsSchdSolGpu->srsBwIndex);
   cudaFree(srsSchdSolGpu->srsCombSize);
   cudaFree(srsSchdSolGpu->srsCombOffset);
   cudaFree(srsSchdSolGpu->srsFreqStart);
   cudaFree(srsSchdSolGpu->srsFreqShift);
   cudaFree(srsSchdSolGpu->srsFreqHopping);
   cudaFree(srsSchdSolGpu->srsSequenceId);
   cudaFree(srsSchdSolGpu->srsGroupOrSequenceHopping);
   cudaFree(srsSchdSolGpu->srsCyclicShift);
   cudaFree(srsSchdSolGpu->srsTxPwr);
   
   delete[] srsLastTxCounterInput;
   delete[] srsPowerControlAdjustmentStateInput;
   delete[] srsConfigIndexInput;
   delete[] srsBwIndexInput;
   delete[] srsTxPwrInput;
   
   delete[] srsCellGrpPramsCpu->cellId;
   delete[] srsCellGrpPramsCpu->cellAssocActUe;

   delete[] srsCellGrpUeStatusCpu->newDataActUe;
   delete[] srsCellGrpUeStatusCpu->srsNumAntPorts;
   delete[] srsCellGrpUeStatusCpu->srsResourceType;
   delete[] srsCellGrpUeStatusCpu->srsLastTxCounter;
   delete[] srsCellGrpUeStatusCpu->srsWbSnr;
   delete[] srsCellGrpUeStatusCpu->srsWbSnrThreshold;
   delete[] srsCellGrpUeStatusCpu->srsWidebandSignalEnergy;
   delete[] srsCellGrpUeStatusCpu->srsTxPwrMax;
   delete[] srsCellGrpUeStatusCpu->srsPwr0;
   delete[] srsCellGrpUeStatusCpu->srsPwrAlpha;
   delete[] srsCellGrpUeStatusCpu->srsTpcAccumulationFlag;
   delete[] srsCellGrpUeStatusCpu->srsPowerControlAdjustmentState;
   delete[] srsCellGrpUeStatusCpu->srsPowerHeadroomReport;
   
   delete[] schdSolCpu->muMimoInd;
   for (int cIdx = 0; cIdx < nCell; cIdx++) 
   {
       cudaFreeHost(schdSolCpu->sortedUeList[cIdx]);
   }
   cudaFreeHost(schdSolCpu->sortedUeList);
   
   delete[] srsSchdSolCpu->nSrsScheduledUePerCell;
   delete[] srsSchdSolCpu->srsTxUe;
   delete[] srsSchdSolCpu->srsRxCell;
   delete[] srsSchdSolCpu->srsNumSymb;
   delete[] srsSchdSolCpu->srsTimeStart;
   delete[] srsSchdSolCpu->srsNumRep;
   delete[] srsSchdSolCpu->srsConfigIndex;
   delete[] srsSchdSolCpu->srsBwIndex;
   delete[] srsSchdSolCpu->srsCombSize;   
   delete[] srsSchdSolCpu->srsCombOffset;
   delete[] srsSchdSolCpu->srsFreqStart;
   delete[] srsSchdSolCpu->srsFreqShift;
   delete[] srsSchdSolCpu->srsFreqHopping;
   delete[] srsSchdSolCpu->srsSequenceId;
   delete[] srsSchdSolCpu->srsGroupOrSequenceHopping;
   delete[] srsSchdSolCpu->srsCyclicShift;
   delete[] srsSchdSolCpu->srsTxPwr;
   
   delete[] srsSchdSolCpuRef->nSrsScheduledUePerCell;
   delete[] srsSchdSolCpuRef->srsTxUe;
   delete[] srsSchdSolCpuRef->srsRxCell;
   delete[] srsSchdSolCpuRef->srsNumSymb;
   delete[] srsSchdSolCpuRef->srsTimeStart;
   delete[] srsSchdSolCpuRef->srsNumRep;
   delete[] srsSchdSolCpuRef->srsConfigIndex;
   delete[] srsSchdSolCpuRef->srsBwIndex;
   delete[] srsSchdSolCpuRef->srsCombSize;   
   delete[] srsSchdSolCpuRef->srsCombOffset;
   delete[] srsSchdSolCpuRef->srsFreqStart;
   delete[] srsSchdSolCpuRef->srsFreqShift;
   delete[] srsSchdSolCpuRef->srsFreqHopping;
   delete[] srsSchdSolCpuRef->srsSequenceId;
   delete[] srsSchdSolCpuRef->srsGroupOrSequenceHopping;
   delete[] srsSchdSolCpuRef->srsCyclicShift;
   delete[] srsSchdSolCpuRef->srsTxPwr;
   
   delete srsCellGrpPramsCpu;
   delete srsCellGrpUeStatusCpu;
   delete schdSolCpu;
   delete srsSchdSolCpu;
   delete srsSchdSolCpuRef;

   delete mcSrsSchedulerGpu; 
   } // for loop nTest  
}
