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


#include <cstring>
#include <iostream>
#include <unistd.h> // for getcwd()
#include <dirent.h> // opendir, readdir
#include <errno.h>
#include <sys/stat.h> // for mkdir

using Clock     = std::chrono::high_resolution_clock;
using TimePoint = std::chrono::time_point<Clock>;
template <typename T, typename unit>
    using duration = std::chrono::duration<T, unit>;

/////////////////////////////////////////////////////////////////////////
// usage()
void usage()
{
    printf("srs_chEst [options]\n");
    printf("  Options:\n");
    printf("    -i  Input HDF5 filename\n");
    printf("    -m  processing mode    SRS proc mode: streams(0x0), graphs (0x1) (default = 0x0)\n");
    printf("    -r  # of iterations    Number of run iterations to run (default = 10)\n");
}

////////////////////////////////////////////////////////////////////////
// main()
int main(int argc, char* argv[])
{
    int returnValue = 0;
    char nvlog_yaml_file[1024];
    // Relative path from binary to default nvlog_config.yaml
    std::string relative_path = std::string("../../../../").append(NVLOG_DEFAULT_CONFIG_FILE);
    std::string log_name = "srs_rx.log";
    nv_get_absolute_path(nvlog_yaml_file, relative_path.c_str());
    pthread_t log_thread_id = -1;
    try
    {
        //------------------------------------------------------------------
        // Parse command line arguments
        int         iArg = 1;
        std::vector<std::string> inputFilenameVec;
        int32_t     totalIters = 1000;

        uint64_t procModeBmsk = SRS_PROC_MODE_FULL_SLOT; // default stream mode

        while(iArg < argc)
        {
            if('-' == argv[iArg][0])
            {
                switch(argv[iArg][1])
                {
                case 'i':
                    if(++iArg >= argc)
                    {
                        NVLOGE_FMT(NVLOG_SRS, AERIAL_CUPHY_EVENT,  "ERROR: No filename provided");
                    }
                    inputFilenameVec.push_back(argv[iArg++]);
                    break;
                case 'r':
                    if((++iArg >= argc) || (1 != sscanf(argv[iArg], "%i", &totalIters)) || ((totalIters <= 0)))
                    {
                        throw std::invalid_argument(fmt::format("Invalid number of run iterations: {}", totalIters));
                    }
                    ++iArg;
                    break;
                case 'm':
                    if((++iArg >= argc) || (1 != sscanf(argv[iArg], "%lu", &procModeBmsk)) || ((SRS_PROC_MODE_FULL_SLOT != procModeBmsk) && (SRS_PROC_MODE_FULL_SLOT_GRAPHS != procModeBmsk)))
                    {
                        throw std::invalid_argument(fmt::format("Invalid processing mode (0x{:x})", procModeBmsk));
                    }
                    ++iArg;
                    break;
                default:
                    NVLOGE_FMT(NVLOG_SRS, AERIAL_CUPHY_EVENT,  "ERROR: Unknown option: {}", argv[iArg]);
                    usage();
                    exit(1);
                    break;
                }
            }
            else
            {
                NVLOGE_FMT(NVLOG_SRS, AERIAL_CUPHY_EVENT,  "ERROR: Invalid command line argument: {}", argv[iArg]);
                exit(1);
            }
        }
        if(inputFilenameVec.empty())
        {
            usage();
            exit(1);
        }

        // ---------------------------------------------------------------
        // Initialize main stream

        cuphy::stream cuStrmMain(cudaStreamNonBlocking);
        cudaStream_t  cuStrm           = cuStrmMain.handle();

        log_thread_id = nvlog_fmtlog_init(nvlog_yaml_file, log_name.c_str(),NULL);
        nvlog_fmtlog_thread_init();
        if(procModeBmsk == SRS_PROC_MODE_FULL_SLOT_GRAPHS)
        {
            NVLOGI_FMT(NVLOG_SRS, "CUDA graph enabled!");
        } else {
            NVLOGI_FMT(NVLOG_SRS, "CUDA stream mode");
        }
        // CUDA Timers
        cuphy::event_timer evtTmrSetup;
        cuphy::event_timer evtTmrRun;
        // CPU Start/Stop times
        TimePoint timePtStartSetup, timePtStopSetup;
        TimePoint timePtStartRun, timePtStopRun;

        duration<float, std::micro> elpasedTimeDurationUs;
        typedef enum _elapsedTypes
        {
            ELAPSED_CPU_SETUP = 0,
            ELAPSED_EVT_SETUP = 1,
            ELAPSED_CPU_RUN   = 2,
            ELAPSED_EVT_RUN   = 3,
            ELAPSED_TYPES_MAX
        } elapsedTypes_t;
        std::array<std::vector<float>,ELAPSED_TYPES_MAX> m_elapsedTimes;
        for(auto& timeVec : m_elapsedTimes)
        {
            timeVec.resize(totalIters);
        }

        //-----------------------------------------------------------------
        // Initialize GPU memory
        size_t max_cells          =  CUPHY_SRS_MAX_N_USERS / 8; // NOTE: assuming 8 activate UEs per cell to calculate max buffer size
        size_t max_rbSnr_mem      =  CUPHY_SRS_MAX_N_USERS * 273 * sizeof(float);
        size_t max_srsReport_mem  =  CUPHY_SRS_MAX_N_USERS * sizeof(cuphySrsReport_t);
        size_t max_chEstToL2_mem  =  max_cells * 273 * 128 * 16 * CUPHY_SRS_MAX_FULL_BAND_SRS_ANT_PORTS_SLOT_PER_CELL * 2; // MAX_N_SRS_CELL * max_prbs * max_ants * CUPHY_SRS_MAX_FULL_BAND_SRS_ANT_PORTS_SLOT_PER_CELL
        size_t max_mem            =  max_rbSnr_mem + max_srsReport_mem + max_chEstToL2_mem;

        cuphy::linear_alloc<128, cuphy::device_alloc> linearAlloc(max_mem);
        // memset() is to suppress compute-sanitizer initcheck errors.
        // It can be removed in the future, once compute-sanitizer allows to suppress errors/warnings.
        linearAlloc.memset(0, cuStrm);
        CUDA_CHECK(cudaStreamSynchronize(cuStrm));

        //------------------------------------------------------------------
        // Load API parameters

        srsStaticApiDataset  srsStaticApiDataset(inputFilenameVec, cuStrm);
        srsDynApiDataset     srsDynApiDataset(inputFilenameVec,    cuStrm, procModeBmsk);
        srsEvalDataset       srsEvalDataset(inputFilenameVec,      cuStrm);  

        uint16_t                           nSrsUes        = srsDynApiDataset.cellGrpDynPrm.nSrsUes;
        uint16_t                           nCells         = srsDynApiDataset.cellGrpDynPrm.nCells;
        std::vector<cuphySrsCellDynPrm_t>& cellDynPrmVec  = srsDynApiDataset.cellDynPrmVec;
        std::vector<cuphyUeSrsPrm_t>&      ueSrsPrmVec    = srsDynApiDataset.ueSrsPrmVec;
        std::vector<cuphyCellStatPrm_t>&   cellStatPrmVec = srsStaticApiDataset.cellStatPrmVec;
        cuphySrsChEstBuffInfo_t*           pChEstBuffInfo = srsDynApiDataset.dataOut.pChEstBuffInfo;
        cuphyTensorPrm_t*                  pTDataRx       = srsDynApiDataset.dataIn.pTDataRx;
        cuphySrsFilterPrms_t&              srsFilterPrms  = srsStaticApiDataset.srsStatPrms.srsFilterPrms;
        cuphySrsChEstToL2_t*               pSrsChEstToL2  = srsDynApiDataset.chEstToL2Vec.data();
        cuphySrsRkhsPrms_t*                pSrsRkhsPrms   = srsStaticApiDataset.srsStatPrms.pSrsRkhsPrms;
        cuphySrsChEstAlgoType_t            chEstAlgo      = srsStaticApiDataset.srsStatPrms.chEstAlgo;
        uint8_t              chEstToL2NormalizationAlgo   = srsStaticApiDataset.srsStatPrms.chEstToL2NormalizationAlgo;
        float                   chEstToL2ConstantScaler   = srsStaticApiDataset.srsStatPrms.chEstToL2ConstantScaler;


        //---------------------------------------------------------------------
        // Combine static and Dynamic cell parameters

        std::vector<cuphySrsCellPrms_t> srsCellPrmsVec(nCells);

        for(int cellIdx = 0; cellIdx < nCells; ++cellIdx)
        {
            srsCellPrmsVec[cellIdx].slotNum     = cellDynPrmVec[cellIdx].slotNum;
            srsCellPrmsVec[cellIdx].frameNum    = cellDynPrmVec[cellIdx].frameNum;
            srsCellPrmsVec[cellIdx].srsStartSym = cellDynPrmVec[cellIdx].srsStartSym;
            srsCellPrmsVec[cellIdx].nSrsSym     = cellDynPrmVec[cellIdx].nSrsSym;
            srsCellPrmsVec[cellIdx].nRxAntSrs   = cellStatPrmVec[cellIdx].nRxAntSrs;
            srsCellPrmsVec[cellIdx].mu          = cellStatPrmVec[cellIdx].mu;
        }
        
        //----------------------------------------------------------------
        // Allocate workspace

        void* d_workspace = linearAlloc.alloc(CUPHY_SRS_RKHS_WORKSPACE_SIZE_PER_CELL);

        //-----------------------------------------------------------------
        // Allocate output memory

        uint32_t            rbSnrBufferSize = nSrsUes * 273;
        float*              d_rbSnrBuffer   = static_cast<float*>(linearAlloc.alloc(sizeof(float) * rbSnrBufferSize));
        cuphySrsReport_t*   d_srsReports    = static_cast<cuphySrsReport_t*>(linearAlloc.alloc(sizeof(cuphySrsReport_t) * nSrsUes));
        std::vector<void*>  addrGpuChEstToL2InnerVec(nSrsUes);
        std::vector<void*>  addrGpuChEstToL2Vec(nSrsUes);
        // since cuphySrsReport_t has few parameters that need to be initialized to 0, perform the following copy
        // ToDo?? any better approach?
        std::vector<cuphySrsReport_t> h_srsReports(nSrsUes);
        for (auto& srs : h_srsReports)
        {
            srs.widebandSignalEnergy = 0.f;
            srs.widebandNoiseEnergy = 0.f;
            srs.widebandScCorr = __floats2half2_rn(0.f, 0.f);
            srs.widebandCsCorrRatioDb = 0.f;
            srs.widebandCsCorrUse     = 0.f;
            srs.widebandCsCorrNotUse  = 0.f; 
        }
        CUDA_CHECK(cudaMemcpyAsync (d_srsReports, h_srsReports.data(), sizeof(cuphySrsReport_t) * nSrsUes, cudaMemcpyHostToDevice, cuStrm));
        CUDA_CHECK(cudaStreamSynchronize(cuStrm));

        
        for(int ueIdx = 0; ueIdx < nSrsUes; ++ueIdx){
            size_t maxChEstSize = 273*128*4*sizeof(float2);
            addrGpuChEstToL2InnerVec[ueIdx] = linearAlloc.alloc(maxChEstSize);
            addrGpuChEstToL2Vec[ueIdx] = linearAlloc.alloc(maxChEstSize);
        }

        //----------------------------------------------------------------------------------------------------------

        std::vector<uint32_t> rbSnrBuffOffsetsVec(nSrsUes);
        for(int ueIdx = 0; ueIdx < nSrsUes; ++ueIdx)
        {
            rbSnrBuffOffsetsVec[ueIdx] = ueIdx * 273;
        }

       // ------------------------------------------------------------------
       // srsChEst descriptors

        // descriptors hold Kernel parameters in GPU
        size_t statDescrSizeBytes, statDescrAlignBytes, dynDescrSizeBytes, dynDescrAlignBytes;
        cuphyStatus_t statusGetWorkspaceSize = cuphySrsChEstGetDescrInfo(&statDescrSizeBytes,
                                                                          &statDescrAlignBytes,
                                                                          &dynDescrSizeBytes,
                                                                          &dynDescrAlignBytes);
        if(CUPHY_STATUS_SUCCESS != statusGetWorkspaceSize) throw cuphy::cuphy_exception(statusGetWorkspaceSize);

        cuphy::buffer<uint8_t, cuphy::pinned_alloc> statDescrBufCpu(statDescrSizeBytes);
        cuphy::buffer<uint8_t, cuphy::device_alloc> statDescrBufGpu(statDescrSizeBytes);
        cuphy::buffer<uint8_t, cuphy::pinned_alloc> dynDescrBufCpu(dynDescrSizeBytes);
        cuphy::buffer<uint8_t, cuphy::device_alloc> dynDescrBufGpu(dynDescrSizeBytes);

        //------------------------------------------------------------------
        // Create srsChEst object

        cuphySrsChEstHndl_t srsChEstHndl;
        bool enableCpuToGpuDescrAsyncCpy = false;
        uint8_t enableDelayOffsetCorrection = 1;

        cuphyStatus_t statusCreate = cuphyCreateSrsChEst(&srsChEstHndl, 
                                                          &srsFilterPrms,
                                                          pSrsRkhsPrms,
                                                          chEstAlgo,
                                                          chEstToL2NormalizationAlgo,
                                                          chEstToL2ConstantScaler,
                                                          enableDelayOffsetCorrection,
                                                          static_cast<uint8_t>(enableCpuToGpuDescrAsyncCpy),
                                                          statDescrBufCpu.addr(),                     
                                                          statDescrBufGpu.addr(), 
                                                          cuStrm);

        if(CUPHY_STATUS_SUCCESS != statusCreate) throw cuphy::cuphy_exception(statusCreate);

        if(!enableCpuToGpuDescrAsyncCpy){
            CUDA_CHECK(cudaMemcpyAsync(statDescrBufGpu.addr(), statDescrBufCpu.addr(), statDescrSizeBytes, cudaMemcpyHostToDevice, cuStrm));
            cudaStreamSynchronize(cuStrm);
        }

        //------------------------------------------------------------------
        // setup srsChEst

        // Launch config holds everything needed to launch kernel using CUDA driver API or add it to a graph
        cuphySrsChEstLaunchCfg_t  srsChEstLaunchCfg;
        cuphySrsChEstNormalizationLaunchCfg_t  srsChEstNormalizationLaunchCfg;

        for(int iterIdx=0; iterIdx<totalIters; iterIdx++)
        {
            
            auto& elapsedTimeUsSetup      = m_elapsedTimes[ELAPSED_CPU_SETUP][iterIdx];
            auto& elapsedTimeUsRun        = m_elapsedTimes[ELAPSED_CPU_RUN][iterIdx];
            auto& elapsedEvtTimeUsSetup   = m_elapsedTimes[ELAPSED_EVT_SETUP][iterIdx];
            auto& elapsedEvtTimeUsRun     = m_elapsedTimes[ELAPSED_EVT_RUN][iterIdx];

            // Record GPU & CPU time before setup
            evtTmrSetup.record_begin(cuStrm);
            timePtStartSetup = Clock::now();
            // Setup function populates dynamic descriptor and launch config. Option to copy descriptors to GPU during setup call.
            cuphyStatus_t setupStatus = cuphySetupSrsChEst(srsChEstHndl,
                                                            nSrsUes,
                                                            ueSrsPrmVec.data(),
                                                            nCells,
                                                            pTDataRx, 
                                                            srsCellPrmsVec.data(),
                                                            d_rbSnrBuffer,
                                                            rbSnrBuffOffsetsVec.data(),
                                                            d_srsReports,
                                                            pChEstBuffInfo,
                                                            addrGpuChEstToL2InnerVec.data(),
                                                            addrGpuChEstToL2Vec.data(),
                                                            pSrsChEstToL2,
                                                            d_workspace,
                                                            static_cast<uint8_t>(enableCpuToGpuDescrAsyncCpy),
                                                            dynDescrBufCpu.addr(),                     
                                                            dynDescrBufGpu.addr(), 
                                                            &srsChEstLaunchCfg,
                                                            &srsChEstNormalizationLaunchCfg,
                                                            cuStrm);
            // Record GPU & CPU time after setup
            timePtStopSetup = Clock::now();
            evtTmrSetup.record_end(cuStrm);

            if(CUPHY_STATUS_SUCCESS != setupStatus) throw cuphy::cuphy_exception(setupStatus);
            if(!enableCpuToGpuDescrAsyncCpy) {
                cudaMemcpyAsync(dynDescrBufGpu.addr(), dynDescrBufCpu.addr(), dynDescrSizeBytes, cudaMemcpyHostToDevice, cuStrm);
                cudaStreamSynchronize(cuStrm);
            }

            //------------------------------------------------------------------
            // run srs ChEst

            // launch kernel using the CUDA driver API
            const CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriver = srsChEstLaunchCfg.kernelNodeParamsDriver;

            // Record GPU & CPU time before run
            evtTmrRun.record_begin(cuStrm);
            timePtStartRun = Clock::now();
            // run SRS iterations
            CUresult srsChEstRunStatus = cuLaunchKernel(kernelNodeParamsDriver.func,
                                                    kernelNodeParamsDriver.gridDimX,
                                                    kernelNodeParamsDriver.gridDimY, 
                                                    kernelNodeParamsDriver.gridDimZ,
                                                    kernelNodeParamsDriver.blockDimX, 
                                                    kernelNodeParamsDriver.blockDimY, 
                                                    kernelNodeParamsDriver.blockDimZ,
                                                    kernelNodeParamsDriver.sharedMemBytes,
                                                    static_cast<CUstream>(cuStrm),
                                                    kernelNodeParamsDriver.kernelParams,
                                                    kernelNodeParamsDriver.extra);                
            // Record GPU & CPU time after run
            timePtStopRun = Clock::now();
            evtTmrRun.record_end(cuStrm);
            if(CUDA_SUCCESS != srsChEstRunStatus) throw cuphy::cuphy_exception(CUPHY_STATUS_INTERNAL_ERROR);

            //------------------------------------------------------------------

            // Process timing data
            evtTmrSetup.synchronize();
            evtTmrRun.synchronize();
            cuStrmMain.synchronize();

            elpasedTimeDurationUs = timePtStopSetup - timePtStartSetup;
            elapsedTimeUsSetup    = elpasedTimeDurationUs.count();
            elpasedTimeDurationUs = timePtStopRun - timePtStartRun;
            elapsedTimeUsRun      = elpasedTimeDurationUs.count();
            elapsedEvtTimeUsSetup = evtTmrSetup.elapsed_time_ms()*1000;
            elapsedEvtTimeUsRun   = evtTmrRun.elapsed_time_ms()*1000;
        }


        //------------------------------------------------------------------
        // Evaluate results    

        // copy to buffers:
        CUDA_CHECK(cudaMemcpyAsync(srsDynApiDataset.dataOut.pSrsReports,  d_srsReports , sizeof(cuphySrsReport_t) * nSrsUes, cudaMemcpyDeviceToHost, cuStrm));
        CUDA_CHECK(cudaMemcpyAsync(srsDynApiDataset.dataOut.pRbSnrBuffer, d_rbSnrBuffer, sizeof(float) * rbSnrBufferSize   , cudaMemcpyDeviceToHost, cuStrm));
        // copy ChEstToL2 to CPU buffer
        for(int ueIdx = 0; ueIdx < nSrsUes; ++ueIdx)
        {
            uint16_t nRxAntSrs = srsCellPrmsVec[ueSrsPrmVec[ueIdx].cellIdx].nRxAntSrs;
            uint16_t nPrbGrps  = pSrsChEstToL2[ueIdx].nPrbGrps;
            uint8_t  nAntPorts = ueSrsPrmVec[ueIdx].nAntPorts;
            size_t   chEstToL2MemSize = nRxAntSrs * nPrbGrps * nAntPorts * sizeof(float2);

            CUDA_CHECK(cudaMemcpyAsync(pSrsChEstToL2[ueIdx].pChEstCpuBuff, addrGpuChEstToL2Vec[ueIdx], chEstToL2MemSize, cudaMemcpyDeviceToHost, cuStrm));
        }    
        cudaStreamSynchronize(cuStrm);

        srsEvalDataset.evalSrsRx(srsDynApiDataset.srsDynPrm, srsDynApiDataset.tSrsChEstVec, srsDynApiDataset.dataOut.pRbSnrBuffer, srsDynApiDataset.dataOut.pSrsReports, cuStrm);

        // Timing Debug
        float avgElapsedTimesUs[ELAPSED_TYPES_MAX];
        float minElapsedTimesUs[ELAPSED_TYPES_MAX];
        float maxElapsedTimesUs[ELAPSED_TYPES_MAX];
        for(int i=0;i<ELAPSED_TYPES_MAX;i++)
        {
            const auto minmax_pair = std::minmax_element(std::begin(m_elapsedTimes[i]),std::end(m_elapsedTimes[i]));
            float mean = std::accumulate(std::begin(m_elapsedTimes[i]),std::end(m_elapsedTimes[i]),0.0)/m_elapsedTimes[i].size();
            avgElapsedTimesUs[i] = mean;
            minElapsedTimesUs[i] = *minmax_pair.first;
            maxElapsedTimesUs[i] = *minmax_pair.second;

        }

        NVLOGC_FMT(NVLOG_SRS,"Timing results {}, format: avg (min, max) ",
            procModeBmsk == SRS_PROC_MODE_FULL_SLOT ? "in stream mode" : "in graph mode");

        NVLOGC_FMT(NVLOG_SRS,"{} Pipeline[{:02d}]: Metric - GPU Time usec (using CUDA events, over {:04d} runs): Run {: 9.4f} ({: 9.4f}, {: 9.4f}) Setup {: 9.4f} ({: 9.4f}, {: 9.4f}) Total {: 9.4f}",
               "SrsRx",
               0,
               m_elapsedTimes[0].size(),
               avgElapsedTimesUs[ELAPSED_EVT_RUN],
               minElapsedTimesUs[ELAPSED_EVT_RUN],
               maxElapsedTimesUs[ELAPSED_EVT_RUN],
               avgElapsedTimesUs[ELAPSED_EVT_SETUP],
               minElapsedTimesUs[ELAPSED_EVT_SETUP],
               maxElapsedTimesUs[ELAPSED_EVT_SETUP],
               avgElapsedTimesUs[ELAPSED_EVT_RUN] + avgElapsedTimesUs[ELAPSED_EVT_SETUP]);

        NVLOGC_FMT(NVLOG_SRS,"{} Pipeline[{:02d}]: Metric - CPU Time usec (using wall clock,  over {:04d} runs): Run {: 9.4f} ({: 9.4f}, {: 9.4f}) Setup {: 9.4f} ({: 9.4f}, {: 9.4f}) Total {: 9.4f}",
               "SrsRx",
               0,
               m_elapsedTimes[0].size(),
               avgElapsedTimesUs[ELAPSED_CPU_RUN],
               minElapsedTimesUs[ELAPSED_CPU_RUN],
               maxElapsedTimesUs[ELAPSED_CPU_RUN],
               avgElapsedTimesUs[ELAPSED_CPU_SETUP],
               minElapsedTimesUs[ELAPSED_CPU_SETUP],
               maxElapsedTimesUs[ELAPSED_CPU_SETUP],
               avgElapsedTimesUs[ELAPSED_CPU_RUN] + avgElapsedTimesUs[ELAPSED_CPU_SETUP]);

        //------------------------------------------------------------------
        // cleanup

        cuphyStatus_t statusDestroy = cuphyDestroySrsChEst(srsChEstHndl);
        if(CUPHY_STATUS_SUCCESS != statusDestroy) throw cuphy::cuphy_exception(statusDestroy);

    }
    catch(std::exception& e)
    {
        NVLOGE_FMT(NVLOG_SRS, AERIAL_CUPHY_EVENT,  "EXCEPTION: {}", e.what());
        returnValue = 1;
    }
    catch(...)
    {
        NVLOGE_FMT(NVLOG_SRS, AERIAL_CUPHY_EVENT,  "UNKNOWN EXCEPTION");
        returnValue = 2;
    }
    nvlog_fmtlog_close(log_thread_id);
    return returnValue;
}
