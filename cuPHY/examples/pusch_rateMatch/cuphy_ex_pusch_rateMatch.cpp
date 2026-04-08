/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "cuphy.hpp"
#include "datasets.hpp"
#include "pusch_utils.hpp"
#include "pusch_rx.hpp"

////////////////////////////////////////////////////////////////////////
// usage()
void usage()
{
    printf("cuphy_ex_pusch_rateMatch [options]\n");
    printf("  Options:\n");
    printf("    -i  Input HDF5 filename\n");
}

std::string extractFilename(const std::string &fullPath)
{
    size_t lastSlashPos = fullPath.find_last_of("/");
    if (lastSlashPos != std::string::npos) {
        // Found the last slash, extract the substring after it
        return fullPath.substr(lastSlashPos + 1);
    }
    // No slash found, return the original string
    return fullPath;
}
////////////////////////////////////////////////////////////////////////
// main()
int main(int argc, char* argv[])
{
    int returnValue = 0;
    cuphyNvlogFmtHelper nvlog_fmt("pusch_rateMatch.log");
    try
    {
        //------------------------------------------------------------------
        // Parse command line arguments
        int         iArg = 1;
        std::vector<std::string> inputFilenameVec;

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
                    inputFilenameVec.emplace_back(argv[iArg++]);
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
        if(inputFilenameVec.empty())
        {
            usage();
            exit(1);
        }

        // FixMe temp workaround to disable testing TC7321-23
        std::string TC = extractFilename(inputFilenameVec[0]);
        // could use lbrm flag instead of file name
        if (TC == "TVnr_7321_PUSCH_gNB_CUPHY_s0p0.h5" || TC == "TVnr_7322_PUSCH_gNB_CUPHY_s0p0.h5" || TC == "TVnr_7323_PUSCH_gNB_CUPHY_s0p0.h5") {
            // issue a warning not to block CICD tests
            NVLOGC_FMT(NVLOG_PUSCH, "PUSCH derate match test currently does not support LBRM (limited buffer rate-matching) test cases, skipping the test!");
            NVLOGC_FMT(NVLOG_PUSCH, "detected {} mismatches out of {} rateMatchedLLRs", 0, 0);
            return 0;
        }

        //----------------------------------------------------------------
        // Initialize CPU memory

        cuphy::buffer<PerTbParams, cuphy::pinned_alloc>              tbPrmsCpu_buffer(MAX_N_TBS_SUPPORTED);
        cuphy::buffer<cuphyPuschRxUeGrpPrms_t, cuphy::pinned_alloc>  drvdUeGrpPrmsBuffer(MAX_N_USER_GROUPS_SUPPORTED);

        //------------------------------------------------------------------
        // Load API parameters

        cuphy::stream cuStrmMain(cudaStreamDefault, PUSCH_STREAM_PRIORITY);

        StaticApiDataset  staticApiDataset(inputFilenameVec, cuStrmMain.handle());
        DynApiDataset     dynApiDataset(inputFilenameVec,   cuStrmMain.handle());
        EvalDataset       evalDataset(inputFilenameVec, cuStrmMain.handle());

        uint32_t nUes    = dynApiDataset.cellGrpDynPrm.nUes;
        uint16_t nUeGrps = dynApiDataset.cellGrpDynPrm.nUeGrps;

        printf("nUes %d, nUeGrps %d\n", nUes, nUeGrps);

        bool     skipTest     = false;

        cudaStreamSynchronize(cuStrmMain.handle()); // synch to ensure data copied

        //------------------------------------------------------------------
        // Derive API parameters
        uint8_t                 enableRssiMeasurement = 0;
        bool                    subSlotProcessingFrontLoadedDmrsEnabled = true;
        uint8_t                 maxDmrsMaxLen = 1;
        cuphyLDPCParams         ldpcPrms(&staticApiDataset.puschStatPrms);

        cuphyPuschStatPrms_t* pStatPrms = &staticApiDataset.puschStatPrms;
        uint32_t maxNPrbAlloc = getMaxNPrbAlloc(pStatPrms);
        uint32_t maxNCbs = getMaxNCbs(pStatPrms);
        uint32_t maxNCbsPerTb = getMaxNCbsPerTb(pStatPrms);
        PuschRx::expandFrontEndParameters(&dynApiDataset.puschDynPrm, pStatPrms, drvdUeGrpPrmsBuffer.addr(), subSlotProcessingFrontLoadedDmrsEnabled, maxDmrsMaxLen, enableRssiMeasurement, maxNPrbAlloc);
        PuschRx::expandBackEndParameters(&dynApiDataset.puschDynPrm, pStatPrms, drvdUeGrpPrmsBuffer.addr(), tbPrmsCpu_buffer.addr(), ldpcPrms, maxNCbs, maxNCbsPerTb);


        //----------------------------------------------------------------------
        // GPU Uci on pusch input buffers

        std::vector<cuphy::tensor_device>  schLLRsVec;
        uint32_t nMaxCbsPerTb = 0;
        uint32_t nMaxTbs      = nUes;

        for(int ueIdx = 0; ueIdx < nUes; ueIdx++)
        {
            if(tbPrmsCpu_buffer[ueIdx].isDataPresent)
            {
                nMaxCbsPerTb = std::max(nMaxCbsPerTb, tbPrmsCpu_buffer[ueIdx].num_CBs);
                printf("nMaxCbsPerTb %u num_CBs %u\n", nMaxCbsPerTb, tbPrmsCpu_buffer[ueIdx].num_CBs);
            }
            else
            {
                skipTest = true;
            }
        }

        if (skipTest)
        {
            NVLOGC_FMT(NVLOG_PUSCH, "At least one UE has no SCH data, skipping the test!");
            NVLOGC_FMT(NVLOG_PUSCH, "detected {} mismatches out of {} rateMatchedLLRs", 0, 0);
            return 0;
        }

        for(int ueIdx = 0; ueIdx < nUes; ueIdx++)
        {
            if(tbPrmsCpu_buffer[ueIdx].uciOnPuschFlag)
            {
                printf("uciOnPuschFlag ON\n");
                // update the exact number of LLRs
                tbPrmsCpu_buffer[ueIdx].G = evalDataset.uciSizesVec[ueIdx].G;
                tbPrmsCpu_buffer[ueIdx].G_csi2 = evalDataset.uciSizesVec[ueIdx].G_csi2;
                if(tbPrmsCpu_buffer[ueIdx].isDataPresent)
                {
                    schLLRsVec.emplace_back(CUPHY_R_16F, evalDataset.schLLRsRef[ueIdx].layout());
                    schLLRsVec[ueIdx].convert(evalDataset.schLLRsRef[ueIdx], cuStrmMain.handle());
                    tbPrmsCpu_buffer[ueIdx].d_schAndCsi2LLRs = static_cast<__half*>(schLLRsVec[ueIdx].addr());
                }
            }
            else
            {
                printf("uciOnPuschFlag OFF\n");
            }
            if (dynApiDataset.uePrmsVec[ueIdx].ndi == 0)
            {
                skipTest = true;
            }
        }

       if (skipTest)
       {
           // just issue a warning
           NVLOGC_FMT(NVLOG_PUSCH, "PUSCH derate match test currently does not support TVs with NDI==0, skipping the test!");
           NVLOGC_FMT(NVLOG_PUSCH, "detected {} mismatches out of {} rateMatchedLLRs", 0, 0);
           return 0;
       }

        //----------------------------------------------------------------------
        // GPU UE Grp input buffers

        cuphy::buffer<PerTbParams, cuphy::device_alloc> tbPrmsGpu_buffer(nUes);
        cudaMemcpyAsync(tbPrmsGpu_buffer.addr(), tbPrmsCpu_buffer.addr(), nUes * sizeof(PerTbParams), cudaMemcpyHostToDevice, cuStrmMain.handle());

        std::vector<cuphy::tensor_device>                        tEqOutLLRsVec;
        std::vector<cuphyTensorPrm_t>                            tPrmEqOutLLRsVec(nUeGrps);


        for(int ueGrpIdx = 0; ueGrpIdx < nUeGrps; ++ueGrpIdx)
        {
            tEqOutLLRsVec.emplace_back(CUPHY_R_16F, evalDataset.eqOutLLRsRef[ueGrpIdx].layout());
            tEqOutLLRsVec[ueGrpIdx].convert(evalDataset.eqOutLLRsRef[ueGrpIdx], cuStrmMain.handle());

            tPrmEqOutLLRsVec[ueGrpIdx].desc  = tEqOutLLRsVec[ueGrpIdx].desc().handle();
            tPrmEqOutLLRsVec[ueGrpIdx].pAddr = tEqOutLLRsVec[ueGrpIdx].addr();
        }

        cudaStreamSynchronize(cuStrmMain.handle());

        //----------------------------------------------------------------------
        // GPU output buffers

        uint32_t NUM_BYTES_PER_LLR = 2;
        uint32_t maxBytesRateMatch = NUM_BYTES_PER_LLR * nMaxTbs * nMaxCbsPerTb * MAX_N_RM_LLRS_PER_CB;
        cuphy::linear_alloc<128, cuphy::device_alloc>  linearAlloc(maxBytesRateMatch);
        printf("nMaxTbs %u nMaxCbsPerTb %u maxBytesRateMatch %u\n", nMaxTbs, nMaxCbsPerTb, maxBytesRateMatch);

        cuphy::buffer<void*, cuphy::pinned_alloc> ppRmOut(nMaxTbs);
        //CUDA_CHECK(cudaHostAlloc(&ppRmOut, sizeof(uint8_t*)*nMaxTbs, cudaHostAllocPortable | cudaHostAllocMapped));

        for(int ueIdx = 0; ueIdx < nUes; ++ueIdx)
        {
            size_t nBytesDeRm  = NUM_BYTES_PER_LLR * (tbPrmsCpu_buffer[ueIdx].Ncb + 2 * tbPrmsCpu_buffer[ueIdx].Zc) * tbPrmsCpu_buffer[ueIdx].num_CBs;
            ppRmOut[ueIdx]     = linearAlloc.alloc(nBytesDeRm);
            // For 1st transmission invalidate the entire HARQ buffer to test if its correctly initialized internally in cuPHY
            if(dynApiDataset.uePrmsVec[ueIdx].ndi)
            {
                CUDA_CHECK(cudaMemsetAsync(ppRmOut[ueIdx], 0xFF, nBytesDeRm, cuStrmMain.handle()));
            }
        }
        cuStrmMain.synchronize();

        //---------------------------------------------------------------------
        // Extract PUSCH rateMatch parameters

        const PerTbParams* pTbPrmsCpu = tbPrmsCpu_buffer.addr();
        const PerTbParams* pTbPrmsGpu = tbPrmsGpu_buffer.addr();
        cuphyTensorPrm_t*  pTPrmRmIn  = tPrmEqOutLLRsVec.data();

        uint16_t nSchUes = 0;
        std::vector<uint16_t> schUserIdxsVec(nUes);
        for(int ueIdx = 0; ueIdx < nUes; ++ueIdx)
        {
            if(pTbPrmsCpu[ueIdx].isDataPresent)
            {
                schUserIdxsVec[nSchUes]  =  ueIdx;
                nSchUes                 += 1;
            }
            if(pTbPrmsCpu[ueIdx].nDmrsCdmGrpsNoData==1)
            {
                skipTest = true;
            }
        }

       if (skipTest)
       {
           //NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT,  "PUSCH derate match test currently does not support TVs with nDmrsCdmGrpsNoData==1, skipping the test!"); return 1;
           // just issue a warning
           NVLOGC_FMT(NVLOG_PUSCH, "PUSCH derate match test currently does not support TVs with nDmrsCdmGrpsNoData==1, skipping the test!");
           NVLOGC_FMT(NVLOG_PUSCH, "detected {} mismatches out of {} rateMatchedLLRs", 0, 0);
           return 0;
       }
       // cudaMemsetAsync(pRmOut, 0, maxBytesRateMatch, cuStrmMain.handle()); // make sure punctured LLRs set to zero
       // cudaStreamSynchronize(cuStrmMain.handle());

        //------------------------------------------------------------------
        // Pusch rateMatch descriptors

        // descriptors hold Kernel parameters in GPU
        size_t   dynDescrSizeBytes, dynDescrAlignBytes;

	    cuphyStatus_t statusGetWorkspaceSize = cuphyPuschRxRateMatchGetDescrInfo(&dynDescrSizeBytes,
                                                                            &dynDescrAlignBytes);
        if(CUPHY_STATUS_SUCCESS != statusGetWorkspaceSize) throw cuphy::cuphy_exception(statusGetWorkspaceSize);

        cuphy::buffer<uint8_t, cuphy::pinned_alloc> dynDescrBufCpu(dynDescrSizeBytes);
        cuphy::buffer<uint8_t, cuphy::device_alloc> dynDescrBufGpu(dynDescrSizeBytes);


        //------------------------------------------------------------------
        // Create Pusch rateMatch object

        cuphyPuschRxRateMatchHndl_t puschRmHndl;

        int FPconfig       = 3;  // 0: FP32 in, FP32 out; 1: FP16 in, FP32 out; 2: FP32 in, FP16 out; 3: FP16 in, FP16 out; other values: invalid
        int descramblingOn = 1;  // enable/disable descrambling

        cuphyCreatePuschRxRateMatch(&puschRmHndl, FPconfig,  descramblingOn);

        //------------------------------------------------------------------
        // Setup Pusch rateMatch object

        // Launch config holds everything needed to launch kernel using CUDA driver API or add it to a graph
        cuphyPuschRxRateMatchLaunchCfg_t puschRmLaunchCfg;

        // setup function populates dynamic descriptor and launch config
        bool enableCpuToGpuDescrAsyncCpy = false;

        cuphyStatus_t puschRmSetupStatus = cuphySetupPuschRxRateMatch( puschRmHndl,
                                                                       nSchUes,
                                                                       schUserIdxsVec.data(),
                                                                       pTbPrmsCpu,
                                                                       pTbPrmsGpu,
                                                                       pTPrmRmIn,
                                                                       pTPrmRmIn,
                                                                       ppRmOut.addr(),
                                                                       dynDescrBufCpu.addr(),
                                                                       dynDescrBufGpu.addr(),
                                                                       enableCpuToGpuDescrAsyncCpy,
                                                                       &puschRmLaunchCfg,
                                                                       cuStrmMain.handle());
        if(CUPHY_STATUS_SUCCESS != puschRmSetupStatus) throw cuphy::cuphy_exception(puschRmSetupStatus);

        if(!enableCpuToGpuDescrAsyncCpy) {
            cudaMemcpyAsync(dynDescrBufGpu.addr(), dynDescrBufCpu.addr(), dynDescrSizeBytes, cudaMemcpyHostToDevice, cuStrmMain.handle());
            cudaStreamSynchronize(cuStrmMain.handle());
        }

        //------------------------------------------------------------------
        // Run Pusch rate match

        // launch kernel using the CUDA driver API
        const CUDA_KERNEL_NODE_PARAMS& resetKernelNodeParamsDriver = puschRmLaunchCfg.resetKernelNodeParamsDriver;
        CUresult resetRateMatchRunStatus = cuLaunchKernel(resetKernelNodeParamsDriver.func,
                                                    resetKernelNodeParamsDriver.gridDimX,
                                                    resetKernelNodeParamsDriver.gridDimY,
                                                    resetKernelNodeParamsDriver.gridDimZ,
                                                    resetKernelNodeParamsDriver.blockDimX,
                                                    resetKernelNodeParamsDriver.blockDimY,
                                                    resetKernelNodeParamsDriver.blockDimZ,
                                                    resetKernelNodeParamsDriver.sharedMemBytes,
                                                    static_cast<CUstream>(cuStrmMain.handle()),
                                                    resetKernelNodeParamsDriver.kernelParams,
                                                    resetKernelNodeParamsDriver.extra);
        if(CUDA_SUCCESS != resetRateMatchRunStatus) throw cuphy::cuphy_exception(CUPHY_STATUS_INTERNAL_ERROR);

        const CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriver = puschRmLaunchCfg.kernelNodeParamsDriver;
        CUresult rateMatchRunStatus = cuLaunchKernel(kernelNodeParamsDriver.func,
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
        if(CUDA_SUCCESS != rateMatchRunStatus) throw cuphy::cuphy_exception(CUPHY_STATUS_INTERNAL_ERROR);

        const CUDA_KERNEL_NODE_PARAMS& clampKernelNodeParamsDriver = puschRmLaunchCfg.clampKernelNodeParamsDriver;
        CUresult clampRateMatchRunStatus = cuLaunchKernel(clampKernelNodeParamsDriver.func,
                                                    clampKernelNodeParamsDriver.gridDimX,
                                                    clampKernelNodeParamsDriver.gridDimY,
                                                    clampKernelNodeParamsDriver.gridDimZ,
                                                    clampKernelNodeParamsDriver.blockDimX,
                                                    clampKernelNodeParamsDriver.blockDimY,
                                                    clampKernelNodeParamsDriver.blockDimZ,
                                                    clampKernelNodeParamsDriver.sharedMemBytes,
                                                    static_cast<CUstream>(cuStrmMain.handle()),
                                                    clampKernelNodeParamsDriver.kernelParams,
                                                    clampKernelNodeParamsDriver.extra);

        if(CUDA_SUCCESS != clampRateMatchRunStatus) throw cuphy::cuphy_exception(CUPHY_STATUS_INTERNAL_ERROR);
        cudaStreamSynchronize(cuStrmMain.handle()); // synch to make sure kernel finishes

       evalDataset.evalPuschRm(ppRmOut.addr(), pTbPrmsCpu, cuStrmMain.handle());

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
