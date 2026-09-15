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

#include "cuphy.h"
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include "hdf5hpp.hpp"
#include "cuphy_hdf5.hpp"
#include "cuphy.hpp"
#include "datasets.hpp"
#include "polar_decoder/polar_cw_tree_layout.hpp"

#include <cstring>
#include <iostream>
#include <unistd.h> // for getcwd()
#include <dirent.h> // opendir, readdir
#include <errno.h>
#include <sys/stat.h> // for mkdir



////////////////////////////////////////////////////////////////////////
// usage()
void usage()
{
    printf("comp_cwTreeTypes [options]\n");
    printf("  Options:\n");
    printf("    -i  Input HDF5 filename\n");
}

////////////////////////////////////////////////////////////////////////
// main()
int main(int argc, char* argv[])
{
    int returnValue = 0;
    
    cuphyNvlogFmtHelper nvlog_fmt("comp_cwTreeTypes.log");
    
    try
    {
        //------------------------------------------------------------------
        // Parse command line arguments
        int         iArg = 1;
        std::string inputFilename  = std::string();

        while(iArg < argc)
        {
            if('-' == argv[iArg][0])
            {
                switch(argv[iArg][1])
                {
                case 'i':
                    if(++iArg >= argc)
                    {
                        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT,  "ERROR: No filename provided");
                    }
                    inputFilename.assign(argv[iArg++]);
                    break;
                default:
                    NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT,  "ERROR: Unknown option: {}", argv[iArg]);
                    usage();
                    exit(1);
                    break;
                }
            }
            else
            {
                NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT,  "ERROR: Invalid command line argument: {}", argv[iArg]);
                exit(1);
            }
        }
        if(inputFilename.empty())
        {
            usage();
            exit(1);
        }

        cuphy::stream cuStrmMain;

        //-----------------------------------------------------------------
        // Allocate GPU memory

        size_t max_PolUciSegPrms_mem = sizeof(cuphyPolarUciSegPrm_t) * CUPHY_MAX_N_POL_UCI_SEGS;
        size_t max_N                 = 1024;
        size_t max_cwTreeTypes_mem   = cuphy::polar::PolarCwTreeLayout::sizeBytes(max_N) * CUPHY_MAX_N_POL_UCI_SEGS;
        size_t max_mem               = max_PolUciSegPrms_mem + max_cwTreeTypes_mem;

        cuphy::linear_alloc<128, cuphy::device_alloc> linearAlloc(max_mem);

        //------------------------------------------------------------------
        // Load uci polar dataset

        UciPolarDataset uciPolarDataset(inputFilename, cuStrmMain.handle());
        cudaStreamSynchronize(cuStrmMain.handle()); // synch to ensure data copied

        //---------------------------------------------------------------------
        // Extract parameters needed to compute cwTreeTypes. 

        uint16_t nPolUciSegs                     = uciPolarDataset.nPolUciSegs;
        cuphyPolarUciSegPrm_t* pPolUciSegPrmsCpu = uciPolarDataset.polUciSegPrmsVec.data();

        //---------------------------------------------------------------------
        // Copy polUciSegPrms to GPU

        cuphyPolarUciSegPrm_t* pPolUciSegPrmsGpu = static_cast<cuphyPolarUciSegPrm_t*>(linearAlloc.alloc(sizeof(cuphyPolarUciSegPrm_t) * nPolUciSegs));
        cudaMemcpyAsync(static_cast<void*>(pPolUciSegPrmsGpu), static_cast<void*>(pPolUciSegPrmsCpu), sizeof(cuphyPolarUciSegPrm_t) * nPolUciSegs, cudaMemcpyHostToDevice, cuStrmMain.handle());
        cudaStreamSynchronize(cuStrmMain.handle());

        //------------------------------------------------------------------
        // Allocate GPU memory for cwTreeTypes

        std::vector<uint8_t*> cwTreeTypesAddrVec(nPolUciSegs);
        
        for(uint16_t segIdx = 0; segIdx < nPolUciSegs; ++segIdx)
        {
            uint16_t N_cw              = pPolUciSegPrmsCpu[segIdx].N_cw;
            cwTreeTypesAddrVec[segIdx] = static_cast<uint8_t*>(linearAlloc.alloc(cuphy::polar::PolarCwTreeLayout::sizeBytes(N_cw)));
        }

        //------------------------------------------------------------------
        // compCwTreeTypes descriptors

        size_t   dynDescrSizeBytes, dynDescrAlignBytes;

        cuphyStatus_t statusGetWorkspaceSize = cuphyCompCwTreeTypesGetDescrInfo(&dynDescrSizeBytes,
                                                                                &dynDescrAlignBytes);
        if(CUPHY_STATUS_SUCCESS != statusGetWorkspaceSize) throw cuphy::cuphy_exception(statusGetWorkspaceSize);

        cuphy::buffer<uint8_t, cuphy::pinned_alloc> dynDescrBufCpu(dynDescrSizeBytes);
        cuphy::buffer<uint8_t, cuphy::device_alloc> dynDescrBufGpu(dynDescrSizeBytes);

        // dynamic allocation for pCwTreeTypesAddrs in CPU descriptor
        size_t dynDescrTreeAddrsSizeBytes = sizeof(uint8_t**) * nPolUciSegs; // uint8_t** ==> compCwTreeTypesDynDescr_t::pCwTreeTypesAddrs

        cuphy::buffer<uint8_t, cuphy::pinned_alloc> dynDescrBufCpuTreeAddrs(dynDescrTreeAddrsSizeBytes);

        //------------------------------------------------------------------
        // Create compCwTreeTypes object

        cuphyCompCwTreeTypesHndl_t compCwTreeTypesHndl;
        cuphyStatus_t statusCreate = cuphyCreateCompCwTreeTypes(&compCwTreeTypesHndl);

        if(CUPHY_STATUS_SUCCESS != statusCreate) throw cuphy::cuphy_exception(statusCreate);


        //------------------------------------------------------------------
        // setup Pucch F0 reciver

        // Launch config holds everything needed to launch kernel using CUDA driver API or add it to a graph
        cuphyCompCwTreeTypesLaunchCfg_t  compCwTreeTypesLaunchCfg;

        // setup function populates dynamic descriptor and launch config
        bool enableCpuToGpuDescrAsyncCpy = false;

        cuphyStatus_t compCwTreeTypesSetupStatus = cuphySetupCompCwTreeTypes(compCwTreeTypesHndl,
                                                                             nPolUciSegs,
                                                                             pPolUciSegPrmsCpu,
                                                                             pPolUciSegPrmsGpu,
                                                                             cwTreeTypesAddrVec.data(),
                                                                             dynDescrBufCpu.addr(),
                                                                             dynDescrBufGpu.addr(),
                                                                             dynDescrBufCpuTreeAddrs.addr(),
                                                                             enableCpuToGpuDescrAsyncCpy,
                                                                             &compCwTreeTypesLaunchCfg,
                                                                             cuStrmMain.handle());

        if(CUPHY_STATUS_SUCCESS != compCwTreeTypesSetupStatus) throw cuphy::cuphy_exception(compCwTreeTypesSetupStatus);

        if(!enableCpuToGpuDescrAsyncCpy) {
            cudaMemcpyAsync(dynDescrBufGpu.addr(), dynDescrBufCpu.addr(), dynDescrSizeBytes, cudaMemcpyHostToDevice, cuStrmMain.handle());
            cudaStreamSynchronize(cuStrmMain.handle());
        }

        //------------------------------------------------------------------
        // run pucch F0 reciever

        // launch kernel using the CUDA driver API
        const CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriver = compCwTreeTypesLaunchCfg.kernelNodeParamsDriver;
        CUresult compCwTreeTypesRunStatus = cuLaunchKernel( kernelNodeParamsDriver.func,
                                                            kernelNodeParamsDriver.gridDimX,
                                                            kernelNodeParamsDriver.gridDimY, 
                                                            kernelNodeParamsDriver.gridDimZ,
                                                            kernelNodeParamsDriver.blockDimX, 
                                                            kernelNodeParamsDriver.blockDimY, 
                                                            kernelNodeParamsDriver.blockDimZ,
                                                            kernelNodeParamsDriver.sharedMemBytes,
                                                            static_cast<CUstream>(cuStrmMain.handle()),
                                                            kernelNodeParamsDriver.kernelParams,
                                                            kernelNodeParamsDriver.extra);                
        if(CUDA_SUCCESS != compCwTreeTypesRunStatus) throw cuphy::cuphy_exception(CUPHY_STATUS_INTERNAL_ERROR);
        cudaStreamSynchronize(cuStrmMain.handle()); // synch to make sure kernel finishes


        // -------------------------------------------------------------------
        // Evaluate Pucch F0 receiver

        uciPolarDataset.evalCwTreeTypes(cwTreeTypesAddrVec.data(), cuStrmMain.handle());

        // -------------------------------------------------------------------
        // cleanup

        cuphyStatus_t statusDestroy = cuphyDestroyCompCwTreeTypes(compCwTreeTypesHndl);
        if(CUPHY_STATUS_SUCCESS != statusDestroy) throw cuphy::cuphy_exception(statusDestroy);


        // // -------------------------------------------------------------------
        // // cleanup

        // cuphyStatus_t statusDestroy = cuphyDestroyPucchF0Rx(pucchF0RxHndl);
        // if(CUPHY_STATUS_SUCCESS != statusDestroy) throw cuphy::cuphy_exception(statusDestroy);

    }
    catch(std::exception& e)
    {
        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT,  "EXCEPTION: {}", e.what());
        returnValue = 1;
    }
    catch(...)
    {
        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT,  "UNKNOWN EXCEPTION");
        returnValue = 2;
    }
    return returnValue;
}
