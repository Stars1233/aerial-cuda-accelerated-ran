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

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <string>
#include <filesystem>
#include <vector>
#include <algorithm>
#include <functional>
#include <unistd.h>  /* For SYS_xxx definitions */
#include <syscall.h> /* For SYS_xxx definitions */
#include <sched.h>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include "nvlog.hpp"


#include "cuda_profiler_api.h"
#include "cuphy.h"
#include "cuphy.hpp"
#include "cuphy_hdf5.hpp"
#include "util.hpp"

#include "datasets.hpp"
#include "pusch_rx_test.hpp"
#include "hdf5hpp.hpp"


#define OUTPUT_TB_FNAME ("outputBits")


template <typename T>
using ms = std::chrono::milliseconds;
template <typename T>
using us = std::chrono::microseconds;

PuschRxTest::PuschRxTest(std::string const& name, uint32_t nPuschRxInst, bool useCuCtxs, cudaStream_t& delayCuStrm, std::vector<int>& cuStrmPrios, bool& startSyncPt, std::mutex& cvStartSyncPtMutex, std::condition_variable& cvStartSyncPt, std::atomic<std::uint32_t>& atmSyncPtWaitCnt, std::vector<uint32_t>& mpsActiveThrdPcts, uint32_t harq_attempts, uint32_t ldpcLaunchMode, uint32_t nMaxLdpcHetConfigs, bool drmDebug, bool debug, bool debugEqualizer, bool useGreenCtxs, uint8_t nMaxTbPerNode) :
    m_name(name),
    m_nPuschRxInst(nPuschRxInst),
    m_useCuCtxs(useCuCtxs),
    m_useGreenCtxs(useGreenCtxs),
    m_cuCtxs(nPuschRxInst),
    m_wrkrThrds(nPuschRxInst),
    m_wrkrThrdMutexes(nPuschRxInst),
    m_startSyncPt(startSyncPt),
    m_cvStartSyncPtMutex(cvStartSyncPtMutex),
    m_cvStartSyncPt(cvStartSyncPt),
    m_atmSyncPtWaitCnt(atmSyncPtWaitCnt),
    m_eStartProcRecorded(false),
    m_mpsActiveThrdPcts(mpsActiveThrdPcts),
    m_ldpcLaunchMode(static_cast<cuphyPuschLdpcKernelLaunch_t>(ldpcLaunchMode)),
    m_nMaxLdpcHetConfigs(nMaxLdpcHetConfigs),
    m_nMaxTbPerNode(nMaxTbPerNode),
    m_debug(debug),
    m_enableNvProf(false),
    m_nIterations(0),
    m_descramblingOn(true),
    m_delayMs(0),
    m_sleepDurationUs(0),
    m_harqAttempts(harq_attempts),
    m_drmDebug(drmDebug),
    m_debugEqualizer(debugEqualizer),
    m_delayCuStrm(delayCuStrm),
    m_atmEndProcCnt(0),
    m_evtTmrsSetup(nPuschRxInst),
    m_evtTmrsRun(nPuschRxInst),
    m_elapsedEvtTimeUsSetup(nPuschRxInst),
    m_elapsedEvtTimeUsRun(nPuschRxInst),
    m_elapsedTimesUsSetup(nPuschRxInst),
    m_elapsedTimesUsRun(nPuschRxInst),
    m_dbgTimePts0(nPuschRxInst),
    m_dbgTimePts1(nPuschRxInst),
    m_dbgTimePts2(nPuschRxInst),
    m_outputFileNameVec(nPuschRxInst)
{
    // Create streams
    for(const int& cuStrmPrio : cuStrmPrios) 
    {
        m_cuphyStrms.emplace_back(cudaStreamNonBlocking, cuStrmPrio);
    }
    if(m_cuphyStrms.size() != nPuschRxInst)
    {
        std::string err = "Number of cuPHY stream priorities " + std::to_string(m_cuphyStrms.size()) + " do not match number of PUSCH instances " + std::to_string(nPuschRxInst);
        throw std::runtime_error(err);
    }

    // CUDA_CHECK(cudaEventCreateWithFlags(&m_eStartProc, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&m_eStartProc, cudaEventBlockingSync));
    CUDA_CHECK(cudaEventCreateWithFlags(&m_eEndProc, cudaEventBlockingSync));
}

PuschRxTest::~PuschRxTest()
{
    CUDA_CHECK_NO_THROW(cudaEventDestroy(m_eStartProc));
    CUDA_CHECK_NO_THROW(cudaEventDestroy(m_eEndProc));

    for(uint32_t instIdx = 0; instIdx < m_nPuschRxInst; ++instIdx)
    {
        if(m_useCuCtxs)
        {
            CU_CHECK(cuCtxDestroy(m_cuCtxs[instIdx]));
        }
    }
}

void PuschRxTest::Setup(uint32_t                               nIterations,
                        bool                                   enableNvProf,
                        int                                    descramblingOn,
                        uint32_t                               delayMs,
                        uint32_t                               sleepDurationUs,
                        std::vector<int> const&                cpuIds,
                        std::vector<int> const&                thrdSchdPolicies,
                        std::vector<int> const&                thrdPrios,
                        std::vector<int> const&                gpuIds,
                        std::vector<std::vector<std::string>>& inputFileNameVec,
                        std::vector<std::string>&              outputFileNameVec,
                        uint64_t                               procModeBmsk,
                        uint32_t                               fp16Mode,
                        std::string                            trtYamlInput,
                        const cuphy::cudaGreenContext&         greenCtx)

{
    // timing parameters
    m_nIterations     = nIterations;
    m_enableNvProf    = enableNvProf;
    m_delayMs         = delayMs;
    m_sleepDurationUs = sleepDurationUs;

    m_wrkrThrdSchdPolicies = thrdSchdPolicies;
    m_wrkrThrdPrios        = thrdPrios;

    m_cpuIds = cpuIds;
    m_gpuIds = gpuIds;

    m_eStartProcRecorded = false;
    m_atmEndProcCnt      = 0;

    //pipeline parameters
    m_descramblingOn    = descramblingOn;
    m_procModeBmsk      = procModeBmsk;
    m_fp16Mode          = fp16Mode;
    m_inputFileNameVec  = inputFileNameVec;
    m_outputFileNameVec = outputFileNameVec;

    // Allocate
    m_dbgStartTimePt.resize(m_nIterations);    
    for(uint32_t instIdx = 0; instIdx < m_nPuschRxInst; ++instIdx)
    {
        m_elapsedEvtTimeUsSetup[instIdx].resize(m_nIterations);
        m_elapsedEvtTimeUsRun[instIdx].resize(m_nIterations);
        m_elapsedTimesUsSetup[instIdx].resize(m_nIterations);
        m_elapsedTimesUsRun[instIdx].resize(m_nIterations);

        m_dbgTimePts0[instIdx].resize(m_nIterations);
        m_dbgTimePts1[instIdx].resize(m_nIterations);
        m_dbgTimePts2[instIdx].resize(m_nIterations);
    }
    
    // Open output file
    if(!m_outputFileNameVec[0].empty())
    {
        m_debugFile.reset(new hdf5hpp::hdf5_file(hdf5hpp::hdf5_file::create(m_outputFileNameVec[0].c_str())));
    }

    SpawnPipelineWrkrThrds(trtYamlInput, greenCtx);
}

void PuschRxTest::SpawnPipelineWrkrThrds(const std::string& trtYamlInput, const cuphy::cudaGreenContext& greenCtx)
{
    // Launch CPU worker threads which will wait for run condition
    uint32_t instIdx = 0;
    for(auto& wrkrThrd : m_wrkrThrds)
    {
        wrkrThrd = std::thread(&PuschRxTest::PipelineWrkrEntry,
                               this,
                               instIdx,
                               trtYamlInput,
                               std::cref(greenCtx));
        instIdx++;
    }
}

void PuschRxTest::runTest(uint32_t instIdx, uint32_t transmission, uint32_t iterIdx, cuphy::pusch_rx &puschRxPipe, StaticApiDataset &staticApiDataset, DynApiDataset &dynApiDataset)
{
    cuphy::stream& cuphyStrm      = m_cuphyStrms[instIdx];
    cudaStream_t cuStrm           = cuphyStrm.handle();
    auto& evtTmrSetup             = m_evtTmrsSetup[instIdx];
    auto& evtTmrRun               = m_evtTmrsRun[instIdx];
    auto& elapsedEvtTimeUsSetup   = m_elapsedEvtTimeUsSetup[instIdx][iterIdx];
    auto& elapsedEvtTimeUsRun     = m_elapsedEvtTimeUsRun[instIdx][iterIdx]; 

    std::array<TimePoint, PUSCH_SETUP_MAX_PHASES> timePtStartSetup, timePtStopSetup;
    std::array<TimePoint, PUSCH_RUN_MAX_PHASES> timePtStartRun, timePtStopRun;
    duration<float, std::micro> elpasedTimeDurationUs;    

    auto& elapsedTimeUsSetup = m_elapsedTimesUsSetup[instIdx][iterIdx];
    auto& elapsedTimeUsRun   = m_elapsedTimesUsRun[instIdx][iterIdx];
    
    // dummy settings (updated when batching and multi-slot processing added)
    uint64_t procModeBmsk0 = 0; // unused         
    cuphyPuschBatchPrmHndl_t batchPrmHndl0 = nullptr;

    //-------------------------------------------------------------------
    // setup pipeline - phase 1
    cuphyPuschSetupPhase_t puschSetupPhase = PUSCH_SETUP_PHASE_1;
    dynApiDataset.puschDynPrm.setupPhase = puschSetupPhase;

    evtTmrSetup[puschSetupPhase].record_begin(cuStrm);

    timePtStartSetup[puschSetupPhase] = Clock::now();
    puschRxPipe.setup(dynApiDataset.puschDynPrm, batchPrmHndl0);
    timePtStopSetup[puschSetupPhase] = Clock::now();

    evtTmrSetup[puschSetupPhase].record_end(cuStrm);

#if 0    
    //-------------------------------------------------------------------    
    // Allocate HARQ buffers based on the calculated requirements from setupPhase 1
    // @todo: for first transmission, this results in a memory allocation on host and device side
    // consider moving this section out of critical path by pre-allocating
    if (transmission == 0)
    {
        m_harqBuffers.clear();
        m_harqBuffers.resize(dynApiDataset.DataOut.totNumTbs); 
    }
    
    // Reuse previously allocated HARQ buffers for retransmissions
    for (int k=0; k<dynApiDataset.DataOut.totNumTbs; k++)
    {
        if (transmission == 0)
        {
            m_harqBuffers[k] = std::move(cuphy::buffer<uint8_t, cuphy::device_alloc>(dynApiDataset.DataOut.h_harqBufferSizeInBytes[k]));
            // Make sure punctured LLRs are set to zero
            CUDA_CHECK(cudaMemsetAsync(m_harqBuffers[k].addr(), 0, dynApiDataset.DataOut.h_harqBufferSizeInBytes[k], cuStrm));
            // printf("tb[%d] allocated HARQ buffer of size %lu at %p\n", k, dynApiDataset.DataOut.h_harqBufferSizeInBytes[k], m_harqBuffers[k].addr());
        }                
        dynApiDataset.DataInOut.pHarqBuffersInOut[k] = m_harqBuffers[k].addr();
    }
#endif    
    
    //-------------------------------------------------------------------
    // setup pipeline - phase 2
    puschSetupPhase = PUSCH_SETUP_PHASE_2;
    dynApiDataset.puschDynPrm.setupPhase = puschSetupPhase;

    evtTmrSetup[puschSetupPhase].record_begin(cuStrm);

    timePtStartSetup[puschSetupPhase] = Clock::now(); 
    puschRxPipe.setup(dynApiDataset.puschDynPrm, batchPrmHndl0);
    timePtStopSetup[puschSetupPhase] = Clock::now();

    evtTmrSetup[puschSetupPhase].record_end(cuStrm);
    
    //-------------------------------------------------------------------
    // Run Pipeline - sub-slot processing phase
    cuphyPuschRunPhase_t puschRunPhase = PUSCH_RUN_SUB_SLOT_PROC;
    evtTmrRun[puschRunPhase].record_begin(cuStrm);

    timePtStartRun[puschRunPhase] = Clock::now();
    puschRxPipe.run(puschRunPhase);
    timePtStopRun[puschRunPhase] = Clock::now();

    // printf("%s Pipeline[%d]: End processing\n", m_name.c_str(), instIdx);
    evtTmrRun[puschRunPhase].record_end(cuStrm);
    
    // copy early-HARQ results
    if(dynApiDataset.DataOut.isEarlyHarqPresent == 1)
    {
        CUDA_CHECK_EXCEPTION(cudaMemcpyAsync(dynApiDataset.evalHarqDetectionStatus, dynApiDataset.DataOut.HarqDetectionStatus, dynApiDataset.totNumUes, cudaMemcpyHostToHost, cuStrm));
        CUDA_CHECK_EXCEPTION(cudaMemcpyAsync(dynApiDataset.evalUciPayloads, dynApiDataset.DataOut.pUciPayloads, dynApiDataset.nUciPayloadBytes, cudaMemcpyHostToHost, cuStrm));
        CUDA_CHECK_EXCEPTION(cudaMemcpyAsync(dynApiDataset.evalUciCrcFlags, dynApiDataset.DataOut.pUciCrcFlags, dynApiDataset.nUciSegs, cudaMemcpyHostToHost, cuStrm));
        CUDA_CHECK_EXCEPTION(cudaMemcpyAsync(dynApiDataset.evalPreEqSinrEhq, dynApiDataset.DataOut.pSinrPreEq, sizeof(float)*dynApiDataset.cellGrpDynPrm.nUes, cudaMemcpyHostToHost, cuStrm));
        CUDA_CHECK_EXCEPTION(cudaMemcpyAsync(dynApiDataset.evalRsrpEhq, dynApiDataset.DataOut.pRsrp, sizeof(float)*dynApiDataset.cellGrpDynPrm.nUes, cudaMemcpyHostToHost, cuStrm));
        CUDA_CHECK_EXCEPTION(cudaMemcpyAsync(dynApiDataset.evalRssiEhq, dynApiDataset.DataOut.pRssi, sizeof(float)*dynApiDataset.cellGrpDynPrm.nUeGrps, cudaMemcpyHostToHost, cuStrm));
    }
    //-------------------------------------------------------------------
    // Run Pipeline - full-slot processing phase
    puschRunPhase = PUSCH_RUN_FULL_SLOT_PROC;
    evtTmrRun[puschRunPhase].record_begin(cuStrm);

    timePtStartRun[puschRunPhase] = Clock::now();
    puschRxPipe.run(puschRunPhase);
#if 0 // For local testing to force a change in the workCancel condition (host pinned memory location) after the graph has been launched; sleep delays were set experimentally for the yaml/config tested
    if (iterIdx == 1) {
        //std::this_thread::sleep_for(std::chrono::microseconds(1200)); // 1st DGL runs, 2nd and 3rd don't
        std::this_thread::sleep_for(std::chrono::microseconds(1500)); // 1st and 2nd DGL run but 3rd one doesn't for my in flux case
        staticApiDataset.setWorkCancelValue((uint8_t)1);
    } else if (iterIdx > 1) {
        staticApiDataset.setWorkCancelValue((uint8_t)0);
    }
#endif
    timePtStopRun[puschRunPhase] = Clock::now();

    // printf("%s Pipeline[%d]: End processing\n", m_name.c_str(), instIdx);
    evtTmrRun[puschRunPhase].record_end(cuStrm);
    //-------------------------------------------------------------------

    // Run Pipeline - D2H mem copy of full-slot processing phase
    puschRunPhase = PUSCH_RUN_FULL_SLOT_COPY;
    evtTmrRun[puschRunPhase].record_begin(cuStrm);

    timePtStartRun[puschRunPhase] = Clock::now();
    puschRxPipe.run(puschRunPhase);
    timePtStopRun[puschRunPhase] = Clock::now();

    // printf("%s Pipeline[%d]: End processing\n", m_name.c_str(), instIdx);
    evtTmrRun[puschRunPhase].record_end(cuStrm);
    //-------------------------------------------------------------------

    // Some updates including potentially expensive atomic increment before GPU completes
    m_atmEndProcCnt++;
    uint32_t tstCnt = m_nPuschRxInst;
    uint32_t newCnt = 0;
    m_dbgTimePts2[instIdx][iterIdx] = timePtStartRun[PUSCH_RUN_SUB_SLOT_PROC];

    evtTmrSetup[PUSCH_SETUP_PHASE_1].synchronize();
    evtTmrSetup[PUSCH_SETUP_PHASE_2].synchronize();
    evtTmrRun[PUSCH_RUN_SUB_SLOT_PROC].synchronize();
    evtTmrRun[PUSCH_RUN_FULL_SLOT_PROC].synchronize();
    evtTmrRun[PUSCH_RUN_FULL_SLOT_COPY].synchronize();
    cuphyStrm.synchronize();
    // timePtStopRun = Clock::now();          // ?? try wall clock here too

    //------------------------------------------------------------------
    // Record an event at the end of last pipeline completion
    // m_atmEndProcCnt.compare_exchange_strong(tstCnt, newCnt)) returns true if (tstCnt == m_atmEndProcCnt)
    // if (tstCnt != m_atmEndProcCnt) => tstCnt = m_atmEndProcCnt
    // if (tstCnt == m_atmEndProcCnt) => m_atmEndProcCnt = newCnt
    if(m_atmEndProcCnt.compare_exchange_strong(tstCnt, newCnt))
    {
#if 0
        CUDA_CHECK(cudaEventRecord(m_eEndProc, cuStrm));

        printf("%s Pipeline[%d]: End processing recorded, profile end time (wall clock) %lu\n",
            m_name.c_str(),
            instIdx,
            std::chrono::duration_cast<std::chrono::microseconds>(m_stopTimePts[instIdx].time_since_epoch()).count());
#endif            
    }
    uint32_t atmEndProcCnt = m_atmEndProcCnt.load();

    elapsedEvtTimeUsSetup[PUSCH_SETUP_PHASE_1] = evtTmrSetup[PUSCH_SETUP_PHASE_1].elapsed_time_ms()*1000;
    elapsedEvtTimeUsSetup[PUSCH_SETUP_PHASE_2] = evtTmrSetup[PUSCH_SETUP_PHASE_2].elapsed_time_ms()*1000;
    
    elapsedEvtTimeUsRun[PUSCH_RUN_SUB_SLOT_PROC] = evtTmrRun[PUSCH_RUN_SUB_SLOT_PROC].elapsed_time_ms()*1000;
    elapsedEvtTimeUsRun[PUSCH_RUN_FULL_SLOT_PROC]  = evtTmrRun[PUSCH_RUN_FULL_SLOT_PROC].elapsed_time_ms()*1000;
    elapsedEvtTimeUsRun[PUSCH_RUN_FULL_SLOT_COPY]  = evtTmrRun[PUSCH_RUN_FULL_SLOT_COPY].elapsed_time_ms()*1000;

    elpasedTimeDurationUs = timePtStopSetup[PUSCH_SETUP_PHASE_1] - timePtStartSetup[PUSCH_SETUP_PHASE_1];
    elapsedTimeUsSetup[PUSCH_SETUP_PHASE_1] = elpasedTimeDurationUs.count();
    elpasedTimeDurationUs = timePtStopSetup[PUSCH_SETUP_PHASE_2] - timePtStartSetup[PUSCH_SETUP_PHASE_2];
    elapsedTimeUsSetup[PUSCH_SETUP_PHASE_2] = elpasedTimeDurationUs.count();

    elpasedTimeDurationUs = timePtStopRun[PUSCH_RUN_SUB_SLOT_PROC] - timePtStartRun[PUSCH_RUN_SUB_SLOT_PROC];
    elapsedTimeUsRun[PUSCH_RUN_SUB_SLOT_PROC] = elpasedTimeDurationUs.count();
    elpasedTimeDurationUs = timePtStopRun[PUSCH_RUN_FULL_SLOT_PROC] - timePtStartRun[PUSCH_RUN_FULL_SLOT_PROC];
    elapsedTimeUsRun[PUSCH_RUN_FULL_SLOT_PROC] = elpasedTimeDurationUs.count();
    elpasedTimeDurationUs = timePtStopRun[PUSCH_RUN_FULL_SLOT_COPY] - timePtStartRun[PUSCH_RUN_FULL_SLOT_COPY];
    elapsedTimeUsRun[PUSCH_RUN_FULL_SLOT_COPY] = elpasedTimeDurationUs.count();
}

void PuschRxTest::PipelineWrkrEntry(uint32_t instIdx, const std::string& trtYamlInput, const cuphy::cudaGreenContext& puschGreenCtx)
{
    try
    {
        int gpuId = m_gpuIds[instIdx];

        //------------------------------------------------------------------
        // Create and bind CUDA sub-context to current pipeline worker thread
        if(m_useCuCtxs)
        {
            {
                std::lock_guard<std::mutex> mpsActiveThrdPctEnvVarMutexGuard(m_mpsActiveThrdPctEnvVarMutex);

    #if 0
                setenv("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", std::to_string(m_mpsActiveThrdPcts[instIdx]).c_str(), true);
                char* pMpsActiveThrdPcts = getenv("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE");
                if(nullptr != pMpsActiveThrdPcts) printf("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE: %s\n", pMpsActiveThrdPcts);
    #elif 1
                char* pMpsServerPid = getenv("MPS_SERVER_PID");
                if(nullptr != pMpsServerPid) printf("MPS_SERVER_PID: %s\n", pMpsServerPid);

                std::string cmd = "echo set_active_thread_percentage " + std::string(pMpsServerPid) + " " + std::to_string(m_mpsActiveThrdPcts[instIdx]) + " | nvidia-cuda-mps-control";
                printf("PuschRx Pipeline[%d]: command %s\n", instIdx, cmd.c_str());

                if(0 != system(cmd.c_str())) printf("Error in setting active thread percentage\n");
    #endif

                int deviceCount = 0;
                CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

                CUdevice device;
                CU_CHECK(cuDeviceGet(&device, gpuId));

                int smCount = 0;
                CU_CHECK(cuDeviceGetAttribute(&smCount, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device));

                printf("PuschRx Pipeline[%d]: GPU ordinal %d (gpuId %d, gpuCount %d), SM usage %d\n", instIdx, static_cast<int>(device), gpuId, deviceCount, smCount);

#if CUDA_VERSION >= 13000
                CUctxCreateParams params{};
                CU_CHECK(cuCtxCreate(&m_cuCtxs[instIdx], &params, CU_CTX_SCHED_AUTO | CU_CTX_MAP_HOST, device));
#else
                CU_CHECK(cuCtxCreate(&m_cuCtxs[instIdx], CU_CTX_SCHED_AUTO | CU_CTX_MAP_HOST, device));
#endif
                CU_CHECK(cuCtxSetCurrent(m_cuCtxs[instIdx]));
            }
        }

        //------------------------------------------------------------------
        int cpuId  = m_cpuIds[instIdx];
        int policy = m_wrkrThrdSchdPolicies[instIdx]; // SCHED_FIFO // SCHED_RR
        int prio   = m_wrkrThrdPrios[instIdx];

        // Bump up prio
        pid_t       pid = (pid_t)syscall(SYS_gettid);
        sched_param schdPrm;
        schdPrm.sched_priority = prio;

        // pid_t pid = (pid_t) syscall(SYS_gettid);
        int schdSetRet = sched_setscheduler(pid, policy, &schdPrm);
        if(0 == schdSetRet)
        {
            printf("%s Pipeline[%d]: pid %d policy %d prio %d\n", m_name.c_str(), instIdx, pid, policy, prio);
        }
        else
        {
            printf("%s Pipeline[%d]: Failed to set scheduling algo pid %d, prio %d, return code %d: err %s\n",
                m_name.c_str(),
                instIdx,
                pid,
                prio,
                schdSetRet,
                strerror(errno));
        }

        //------------------------------------------------------------------
        // Set thread affinity to specified CPU

        cpu_set_t cpuSet;
        CPU_ZERO(&cpuSet);
        CPU_SET(cpuId, &cpuSet);
        // printf("%s Pipeline[%d]: setting affinity of pipeline %d (pid %d) to CPU Id %d\n", m_name.c_str(), instIdx, pid, cpuId);

        int affinitySetRet = sched_setaffinity(pid, sizeof(cpuSet), &cpuSet);
        if(0 == affinitySetRet)
        {
            printf("%s Pipeline[%d]: pid %d set affinity to CPU %d\n", m_name.c_str(), instIdx, pid, cpuId);
        }
        else
        {
            printf("%s Pipeline[%d]: failed to set affinity pid %d to CPU %d, return code %d err %s\n",
                m_name.c_str(),
                instIdx,
                pid,
                cpuId,
                affinitySetRet,
                strerror(errno));
        }

        // Make sure no cudaSetDevice call happens *after* the puschGreenCtx.bind() is called, because the cudaSetDevice binds the primary context to the calling thread.
        CUDA_CHECK(cudaSetDevice(gpuId));

        if(m_useGreenCtxs)
        {
#if CUDA_VERSION >= 12040
            puschGreenCtx.bind();
            printf("%s Pipeline[%d]: green context will have access to %d SMs.\n", m_name.c_str(), instIdx, puschGreenCtx.getSmCount());
#endif
        }


        //------------------------------------------------------------------
        // Load datasets from h5 file
        cuphy::stream& cuphyStrm = m_cuphyStrms[instIdx];
        cudaStream_t cuStrm = cuphyStrm.handle();
        StaticApiDataset staticApiDataset(m_inputFileNameVec[0], cuStrm, m_outputFileNameVec[instIdx], m_descramblingOn, 0, true, nullptr, m_ldpcLaunchMode);
        staticApiDataset.puschStatPrms.nMaxLdpcHetConfigs = m_nMaxLdpcHetConfigs;
        staticApiDataset.puschStatPrms.nMaxTbPerNode = m_nMaxTbPerNode;
        CUDA_CHECK(cudaStreamSynchronize(cuStrm));
        
        if (not trtYamlInput.empty()) {
            if (not std::filesystem::exists(trtYamlInput)) {
                NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "TrtEngine YAML config file:{} - no such file",
                           trtYamlInput);
            }
            else
            {
                NVLOGC_FMT(NVLOG_PUSCH, "TrtEngine YAML config file:{}, size {}", trtYamlInput, trtYamlInput.size());
                staticApiDataset.puschStatPrms.puschrxChestFactorySettingsFilename = trtYamlInput.c_str();
            }
        }

        //------------------------------------------------------------------
        // Create PuschRx pipeline

        staticApiDataset.puschStatPrms.enableDebugEqOutput = (m_debugEqualizer) ? 1 : 0;
        cuphy::pusch_rx puschRxPipe(staticApiDataset.puschStatPrms, cuStrm);

        //------------------------------------------------------------------
        // Wait for syncpoint
        {
            printf("%s Pipeline[%d]: Wait for start-sync point\n", m_name.c_str(), instIdx);
            m_atmSyncPtWaitCnt++;
            std::unique_lock<std::mutex> cvStartSyncPtMutexLock(m_cvStartSyncPtMutex);
            m_cvStartSyncPt.wait(cvStartSyncPtMutexLock, [this] { return m_startSyncPt; });
        }
        printf("%s Pipeline[%d]: start-syncpoint hit\n", m_name.c_str(), instIdx);

        if(m_enableNvProf)
        {
            cudaProfilerStart();
        }

        //--------------------------------------------------------------------------
        // Setup and run pipeline tests
        size_t numSlots = m_inputFileNameVec.size();

        for (uint32_t transmissions = 0; transmissions < m_harqAttempts; transmissions++)
        {
            for (uint32_t slotIdx = 0; slotIdx < numSlots; slotIdx++)
            {
                int filenameLen = m_inputFileNameVec[slotIdx][instIdx].length();
                if (transmissions != 0) m_inputFileNameVec[slotIdx][instIdx][filenameLen-6]='0'+transmissions;
                std::cout << "PuschRx Pipeline[" << instIdx << "]: For transmission " << transmissions+1 << " using filename " << m_inputFileNameVec[slotIdx][instIdx] << std::endl;

                //-------------------------------------------------------------------
                // Setup datasets
                DynApiDataset dynApiDataset(m_inputFileNameVec[slotIdx], cuStrm, m_procModeBmsk, true, m_fp16Mode, 0, m_drmDebug);
                EvalDataset   evalDataset(m_inputFileNameVec[slotIdx], cuStrm, 0, m_drmDebug);

                //-------------------------------------------------------------------
                // Pre-allocate HARQ buffers based on the calculated requirements from setupPhase 1
                // Note: to avoid HARQ buffer allocation in critical path (requires memory allocation on host and device side) invoke a dummy
                // PUSCH_SETUP_PHASE_1 and pre-allocate
                
                if (transmissions == 0)
                {
                    m_foCompensationBuffer.clear();
                    m_foCompensationBuffer.resize(dynApiDataset.cellGrpDynPrm.nUes);
                }
                
                for (int k=0; k < dynApiDataset.cellGrpDynPrm.nUes; k++)
                {
                    if (transmissions == 0)
                    { 
                        m_foCompensationBuffer[k] = std::move(cuphy::buffer<float, cuphy::device_alloc>(2));
                        CUDA_CHECK(cudaMemsetAsync(m_foCompensationBuffer[k].addr(), 0, 2*sizeof(float), cuStrm));
                    }
                    dynApiDataset.DataInOut.pFoCompensationBuffersInOut[k] = m_foCompensationBuffer[k].addr();
                }
                    
                // Dummy call to setup pipeline - phase 1 to pre-allocate HARQ buffer
                dynApiDataset.puschDynPrm.setupPhase = PUSCH_SETUP_PHASE_1;
                cuphyStatus_t s = puschRxPipe.setup(dynApiDataset.puschDynPrm, nullptr);
                if(CUPHY_STATUS_SUCCESS != s)
                {
                    if((dynApiDataset.puschDynPrm.pStatusOut->status == cuphyPuschStatusType_t::CUPHY_PUSCH_STATUS_UNSUPPORTED_MAX_ER_PER_CB)||(dynApiDataset.puschDynPrm.pStatusOut->status == cuphyPuschStatusType_t::CUPHY_PUSCH_STATUS_TBSIZE_MISMATCH))
                    {
                        // only for cuPHY unit test purpose
                        printf("Cell # 0 : TbIdx: 0 Metric - Block Error Rate      : 1.0000 (Error CBs 1, Mismatched CBs 0, MismatchedCRC CBs 0, Total CBs 1)\n");
                        printf("Cell # 0 :          Metric - TB CRC Error      :(MismatchedCRC TBs 0, Total TBs 1)\n");
                        return;
                    }
                }

                // Allocate HARQ buffers based on the calculated requirements from setupPhase 1
                if (transmissions == 0)
                {
                    m_harqBuffers.clear();
                    m_harqBuffers.resize(dynApiDataset.cellGrpDynPrm.nUes);
                }

                // Reuse previously allocated HARQ buffers for retransmissions
                for (int k=0; k < dynApiDataset.cellGrpDynPrm.nUes; k++)
                {
                    if (transmissions == 0)
                    {
                        m_harqBuffers[k] = std::move(cuphy::buffer<uint8_t, cuphy::device_alloc>(dynApiDataset.DataOut.h_harqBufferSizeInBytes[k]));
                        // printf("TB[%d] HARQ buffer address: %p size %d bytes\n", k, m_harqBuffers[k].addr(), dynApiDataset.DataOut.h_harqBufferSizeInBytes[k]);

                        // For 1st transmission invalidate the entire HARQ buffer to test if its correctly initialized internally in cuPHY
                        if(dynApiDataset.uePrmsVec[k].ndi)
                        {
                            CUDA_CHECK(cudaMemsetAsync(m_harqBuffers[k].addr(), 0xFF, dynApiDataset.DataOut.h_harqBufferSizeInBytes[k], cuStrm));
                        }
                    }
                    dynApiDataset.DataInOut.pHarqBuffersInOut[k] = m_harqBuffers[k].addr();
                }

                cuphyStrm.synchronize();

                //------------------------------------------------------------------
                for(uint32_t i = 0; i < m_nIterations; ++i)
                {
                    //------------------------------------------------------------------
                    // First pipeline instance special handling to synchronize multiple pipelines
                    if(0 == instIdx)
                    {
                        // Sync whole GPU and start clean
                        CUDA_CHECK(cudaDeviceSynchronize());

                        // Launch a delay kernel to keep GPU busy until the CPU bursts out kernel launches (for all pipelines)
                        if(m_delayMs)
                        {
                            gpu_ms_delay(m_delayMs, gpuId, cuStrm);
                            m_delayCuStrm = cuStrm;
                        }

                        // Place a start event on the first stream which all other streams will wait on. This is used to
                        // model concurrent workload submission from all streams (i.e. pipelines) as close to each other
                        // in time as possible
                        CUDA_CHECK(cudaEventRecord(m_eStartProc, m_delayCuStrm));

                        // Notify all other worker threads that m_eStartProc is recorded
                        {
                            std::lock_guard<std::mutex> cvStartProcRecMutexGuard(m_cvStartProcRecMutex);
                            m_eStartProcRecorded = true;
                        }

                        m_dbgStartTimePt[i] = Clock::now();
                        m_cvStartProcRec.notify_all();
                    }

                    //------------------------------------------------------------------
                    // Ensure pipeline 0 has recorded the start event before ungating other threads
                    if(0 != instIdx)
                    {
                        std::unique_lock<std::mutex> cvStartProcRecMutexLock(m_cvStartProcRecMutex);
                        m_cvStartProcRec.wait(cvStartProcRecMutexLock, [this] { return m_eStartProcRecorded; });
                    }

                    m_dbgTimePts0[instIdx][i] = Clock::now();

                    // To launch the pipelines as close to each other as possible, all other pipeline streams wait for
                    // a cuda event from the first pipeline stream. Since a cuda event can be waited on after it is
                    // recorded, the first pipeline waits for run signal from main thread after recording the cuda
                    // event and all other pipelines wait for cuda event wait after receiving the run signal from
                    // main thread.
                    CUDA_CHECK(cudaStreamWaitEvent(cuStrm, m_eStartProc, 0));

                    m_dbgTimePts1[instIdx][i] = Clock::now();

                    if(m_sleepDurationUs)
                    {
                        std::this_thread::sleep_for(std::chrono::microseconds(m_sleepDurationUs));
                    }

                    //------------------------------------------------------------------
                    runTest(instIdx, transmissions, i, puschRxPipe, staticApiDataset, dynApiDataset);
                }

                //---------------------------------------------------------------------------
                // Display timing and TB/CB/UCI Bler
                DisplayTiming(instIdx, slotIdx, evalDataset.nBytesVec[0], m_debug);
                DisplayBler(evalDataset, dynApiDataset, instIdx, m_debug, m_drmDebug);
                evalDataset.computeNumUciCbErrors(dynApiDataset, true);
                //  evalDataset.reportPuschCrcErrors(dynApiDataset.puschDynPrm);

                //----------------------------------------------------------------------------
                //  Copy receiver and debug output from GPU to CPU. Print parameters.
                puschRxPipe.writeDbgSynch(cuStrm);
                cuphyStrm.synchronize();

                //---------------------------------------------------------------------------
                // Verify internal pipeline probes
                evalDataset.evalPuschRx(m_outputFileNameVec[instIdx], staticApiDataset, dynApiDataset, cuStrm);

                if(m_useCuCtxs) CU_CHECK(cuCtxSynchronize());
            }
        }
        if(m_enableNvProf)
        {
            cudaProfilerStop();
        }
    }
    catch(std::exception& e)
    {
        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT,  "EXCEPTION: {}", e.what());
        exit(1);
    }
    catch(...)
    {
        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT,  "UNKNOWN EXCEPTION");
        exit(1);
    }

}

void PuschRxTest::WaitForCompletion()
{
    // Join all threads
    uint32_t instIdx = 0;
    for(auto& wrkrThrd : m_wrkrThrds)
    {
        wrkrThrd.join();
        printf("%s Pipeline[%d]: Joining worker thread\n", m_name.c_str(), instIdx);
        instIdx++;
    }
}


void PuschRxTest::DisplayTiming(uint32_t instIdx, uint32_t slotIdx, uint32_t nBytes, bool debug, bool throughput, bool execTime)
{
    // Helper to update accumulated, max and min values
    auto updateAccumMinMaxVals = [](float instVal, float& accumVal, float& maxVal, float& minVal)
    {
        accumVal += instVal;
        maxVal    = std::max(instVal, maxVal);
        minVal    = std::min(instVal, minVal);
    };

    //-------------------------------------------------------------------------------------------        
    auto& instElapsedEvtTimeUsSetup = m_elapsedEvtTimeUsSetup[instIdx];
    auto& instElapsedTimeUsSetup    = m_elapsedTimesUsSetup[instIdx];

    auto& instElapsedEvtTimeUsRun   = m_elapsedEvtTimeUsRun[instIdx]; 
    auto& instElapsedTimeUsRun      = m_elapsedTimesUsRun[instIdx];

    std::array<float, PUSCH_SETUP_MAX_PHASES> accumElapsedEvtTimeUsSetup, accumElapsedTimeUsSetup;
    accumElapsedEvtTimeUsSetup.fill(0.0f);
    accumElapsedTimeUsSetup.fill(0.0f);
    
    std::array<float, PUSCH_RUN_MAX_PHASES> accumElapsedEvtTimeUsRun, accumElapsedTimeUsRun;
    accumElapsedEvtTimeUsRun.fill(0.0f);
    accumElapsedTimeUsRun.fill(0.0f);

    std::array<float, PUSCH_SETUP_MAX_PHASES> maxElapsedEvtTimeUsSetup, maxElapsedTimeUsSetup;
    maxElapsedEvtTimeUsSetup.fill(std::numeric_limits<float>::min());
    maxElapsedTimeUsSetup.fill(std::numeric_limits<float>::min());
    
    std::array<float, PUSCH_RUN_MAX_PHASES> maxElapsedEvtTimeUsRun, maxElapsedTimeUsRun;
    maxElapsedEvtTimeUsRun.fill(std::numeric_limits<float>::min());
    maxElapsedTimeUsRun.fill(std::numeric_limits<float>::min());

    std::array<float, PUSCH_SETUP_MAX_PHASES> minElapsedEvtTimeUsSetup, minElapsedTimeUsSetup;
    minElapsedEvtTimeUsSetup.fill(std::numeric_limits<float>::max());
    minElapsedTimeUsSetup.fill(std::numeric_limits<float>::max());
    
    std::array<float, PUSCH_RUN_MAX_PHASES> minElapsedEvtTimeUsRun, minElapsedTimeUsRun;
    minElapsedEvtTimeUsRun.fill(std::numeric_limits<float>::max());
    minElapsedTimeUsRun.fill(std::numeric_limits<float>::max());
    
    for(uint32_t iterIdx = 0; iterIdx < m_nIterations; ++iterIdx)
    {    
        // Setup phase-1
        updateAccumMinMaxVals(instElapsedEvtTimeUsSetup[iterIdx][PUSCH_SETUP_PHASE_1], 
                              accumElapsedEvtTimeUsSetup[PUSCH_SETUP_PHASE_1],
                              maxElapsedEvtTimeUsSetup[PUSCH_SETUP_PHASE_1],
                              minElapsedEvtTimeUsSetup[PUSCH_SETUP_PHASE_1]);

        updateAccumMinMaxVals(instElapsedTimeUsSetup[iterIdx][PUSCH_SETUP_PHASE_1],
                              accumElapsedTimeUsSetup[PUSCH_SETUP_PHASE_1],
                              maxElapsedTimeUsSetup[PUSCH_SETUP_PHASE_1],
                              minElapsedTimeUsSetup[PUSCH_SETUP_PHASE_1]);

        // Setup phase-2
        updateAccumMinMaxVals(instElapsedEvtTimeUsSetup[iterIdx][PUSCH_SETUP_PHASE_2], 
                              accumElapsedEvtTimeUsSetup[PUSCH_SETUP_PHASE_2],
                              maxElapsedEvtTimeUsSetup[PUSCH_SETUP_PHASE_2],
                              minElapsedEvtTimeUsSetup[PUSCH_SETUP_PHASE_2]);

        updateAccumMinMaxVals(instElapsedTimeUsSetup[iterIdx][PUSCH_SETUP_PHASE_2],
                              accumElapsedTimeUsSetup[PUSCH_SETUP_PHASE_2],
                              maxElapsedTimeUsSetup[PUSCH_SETUP_PHASE_2],
                              minElapsedTimeUsSetup[PUSCH_SETUP_PHASE_2]);

        // Run early-HARQ phase
        updateAccumMinMaxVals(instElapsedEvtTimeUsRun[iterIdx][PUSCH_RUN_SUB_SLOT_PROC],
                              accumElapsedEvtTimeUsRun[PUSCH_RUN_SUB_SLOT_PROC],
                              maxElapsedEvtTimeUsRun[PUSCH_RUN_SUB_SLOT_PROC],
                              minElapsedEvtTimeUsRun[PUSCH_RUN_SUB_SLOT_PROC]);

        updateAccumMinMaxVals(instElapsedTimeUsRun[iterIdx][PUSCH_RUN_SUB_SLOT_PROC],
                              accumElapsedTimeUsRun[PUSCH_RUN_SUB_SLOT_PROC],
                              maxElapsedTimeUsRun[PUSCH_RUN_SUB_SLOT_PROC],
                              minElapsedTimeUsRun[PUSCH_RUN_SUB_SLOT_PROC]);

        // Run full-slot processing phase
        updateAccumMinMaxVals(instElapsedEvtTimeUsRun[iterIdx][PUSCH_RUN_FULL_SLOT_PROC],
                              accumElapsedEvtTimeUsRun[PUSCH_RUN_FULL_SLOT_PROC],
                              maxElapsedEvtTimeUsRun[PUSCH_RUN_FULL_SLOT_PROC],
                              minElapsedEvtTimeUsRun[PUSCH_RUN_FULL_SLOT_PROC]);

        updateAccumMinMaxVals(instElapsedTimeUsRun[iterIdx][PUSCH_RUN_FULL_SLOT_PROC],
                              accumElapsedTimeUsRun[PUSCH_RUN_FULL_SLOT_PROC],
                              maxElapsedTimeUsRun[PUSCH_RUN_FULL_SLOT_PROC],
                              minElapsedTimeUsRun[PUSCH_RUN_FULL_SLOT_PROC]);
                              
        // Run full-slot D2H copy phase
        updateAccumMinMaxVals(instElapsedEvtTimeUsRun[iterIdx][PUSCH_RUN_FULL_SLOT_COPY],
                              accumElapsedEvtTimeUsRun[PUSCH_RUN_FULL_SLOT_COPY],
                              maxElapsedEvtTimeUsRun[PUSCH_RUN_FULL_SLOT_COPY],
                              minElapsedEvtTimeUsRun[PUSCH_RUN_FULL_SLOT_COPY]);

        updateAccumMinMaxVals(instElapsedTimeUsRun[iterIdx][PUSCH_RUN_FULL_SLOT_COPY],
                              accumElapsedTimeUsRun[PUSCH_RUN_FULL_SLOT_COPY],
                              maxElapsedTimeUsRun[PUSCH_RUN_FULL_SLOT_COPY],
                              minElapsedTimeUsRun[PUSCH_RUN_FULL_SLOT_COPY]);
    }

    std::array<float, PUSCH_SETUP_MAX_PHASES> avgElapsedEvtTimeUsSetup;
    avgElapsedEvtTimeUsSetup[PUSCH_SETUP_PHASE_1] = accumElapsedEvtTimeUsSetup[PUSCH_SETUP_PHASE_1] / m_nIterations;
    avgElapsedEvtTimeUsSetup[PUSCH_SETUP_PHASE_2] = accumElapsedEvtTimeUsSetup[PUSCH_SETUP_PHASE_2] / m_nIterations;
    
    std::array<float, PUSCH_RUN_MAX_PHASES> avgElapsedEvtTimeUsRun;
    avgElapsedEvtTimeUsRun[PUSCH_RUN_SUB_SLOT_PROC] = accumElapsedEvtTimeUsRun[PUSCH_RUN_SUB_SLOT_PROC] / m_nIterations;
    avgElapsedEvtTimeUsRun[PUSCH_RUN_FULL_SLOT_PROC]  = accumElapsedEvtTimeUsRun[PUSCH_RUN_FULL_SLOT_PROC] / m_nIterations;
    avgElapsedEvtTimeUsRun[PUSCH_RUN_FULL_SLOT_COPY]  = accumElapsedEvtTimeUsRun[PUSCH_RUN_FULL_SLOT_COPY] / m_nIterations;

    std::array<float, PUSCH_SETUP_MAX_PHASES> avgElapsedTimeUsSetup;
    avgElapsedTimeUsSetup[PUSCH_SETUP_PHASE_1] = accumElapsedTimeUsSetup[PUSCH_SETUP_PHASE_1] / m_nIterations;
    avgElapsedTimeUsSetup[PUSCH_SETUP_PHASE_2] = accumElapsedTimeUsSetup[PUSCH_SETUP_PHASE_2] / m_nIterations;
    
    std::array<float, PUSCH_RUN_MAX_PHASES> avgElapsedTimeUsRun;
    avgElapsedTimeUsRun[PUSCH_RUN_SUB_SLOT_PROC] = accumElapsedTimeUsRun[PUSCH_RUN_SUB_SLOT_PROC] / m_nIterations;
    avgElapsedTimeUsRun[PUSCH_RUN_FULL_SLOT_PROC]  = accumElapsedTimeUsRun[PUSCH_RUN_FULL_SLOT_PROC] / m_nIterations;
    avgElapsedTimeUsRun[PUSCH_RUN_FULL_SLOT_COPY]  = accumElapsedTimeUsRun[PUSCH_RUN_FULL_SLOT_COPY] / m_nIterations;

    // Total GPU
    float avgTotalElapsedEvtTimeUs = (accumElapsedEvtTimeUsSetup[PUSCH_SETUP_PHASE_1] + accumElapsedEvtTimeUsSetup[PUSCH_SETUP_PHASE_2] +
                                      accumElapsedEvtTimeUsRun[PUSCH_RUN_SUB_SLOT_PROC] + accumElapsedEvtTimeUsRun[PUSCH_RUN_FULL_SLOT_PROC] +
                                      accumElapsedEvtTimeUsRun[PUSCH_RUN_FULL_SLOT_COPY]) / m_nIterations;

    // Total CPU
    float avgTotalElapsedTimeUs = (accumElapsedTimeUsSetup[PUSCH_SETUP_PHASE_1] + accumElapsedTimeUsSetup[PUSCH_SETUP_PHASE_2] +
                                   accumElapsedTimeUsRun[PUSCH_RUN_SUB_SLOT_PROC] + accumElapsedTimeUsRun[PUSCH_RUN_FULL_SLOT_PROC] +
                                   accumElapsedTimeUsRun[PUSCH_RUN_FULL_SLOT_COPY]) / m_nIterations;

    printf("slot %u ---------------------------------------------------------------\n", slotIdx);
    //------------------------------------------------------------------------------------------------------
    // Display throughput:
    if(throughput)
    {
        // Note: the throughput calculation assumes GPU Run time (i.e. slot processing time) only under the assumption that the 
        // CPU time (Setup/Run API host processing) and GPU setup time (CPU to GPU transfers and any other setup work on GPU) can be 
        // pipelined with GPU processing time
        size_t nEncodedBits = nBytes * 8;
        printf("PuschRx Pipeline[%02d]: Metric - Throughput (w/ GPU runtime only): %07.4f Gbps (encoded input bits %lu) \n",
               instIdx,
               (static_cast<float>(nEncodedBits) / ((avgElapsedEvtTimeUsRun[PUSCH_RUN_SUB_SLOT_PROC] + avgElapsedEvtTimeUsRun[PUSCH_RUN_FULL_SLOT_PROC] + avgElapsedEvtTimeUsRun[PUSCH_RUN_FULL_SLOT_COPY]) * 1e-6)) / 1e9,
               nEncodedBits);
    }

    //-------------------------------------------------------------------------------------------
    // Display timing:
    if(execTime)
    {
        printf("%s Pipeline[%02d]: Metric - GPU Time usec (using CUDA events, over %04d runs): Run-P1 %07.4f (%07.4f, %07.4f) Run-P2 %07.4f (%07.4f, %07.4f) Run-P3 %07.4f (%07.4f, %07.4f) Setup-P1 %07.4f (%07.4f, %07.4f) Setup-P2 %07.4f (%07.4f, %07.4f) Total %07.4f\n",
               m_name.c_str(),
               instIdx,
               m_nIterations,
               avgElapsedEvtTimeUsRun[PUSCH_RUN_SUB_SLOT_PROC],
               minElapsedEvtTimeUsRun[PUSCH_RUN_SUB_SLOT_PROC],
               maxElapsedEvtTimeUsRun[PUSCH_RUN_SUB_SLOT_PROC],
               avgElapsedEvtTimeUsRun[PUSCH_RUN_FULL_SLOT_PROC],
               minElapsedEvtTimeUsRun[PUSCH_RUN_FULL_SLOT_PROC],
               maxElapsedEvtTimeUsRun[PUSCH_RUN_FULL_SLOT_PROC],
               avgElapsedEvtTimeUsRun[PUSCH_RUN_FULL_SLOT_COPY],
               minElapsedEvtTimeUsRun[PUSCH_RUN_FULL_SLOT_COPY],
               maxElapsedEvtTimeUsRun[PUSCH_RUN_FULL_SLOT_COPY],
               avgElapsedEvtTimeUsSetup[PUSCH_SETUP_PHASE_1],
               minElapsedEvtTimeUsSetup[PUSCH_SETUP_PHASE_1],
               maxElapsedEvtTimeUsSetup[PUSCH_SETUP_PHASE_1],
               avgElapsedEvtTimeUsSetup[PUSCH_SETUP_PHASE_2],
               minElapsedEvtTimeUsSetup[PUSCH_SETUP_PHASE_2],
               maxElapsedEvtTimeUsSetup[PUSCH_SETUP_PHASE_2],
               avgTotalElapsedEvtTimeUs);
    
        printf("%s Pipeline[%02d]: Metric - CPU Time usec (using wall clock w/ %u ms delay kernel, over %04d runs): Run-P1 %07.4f (%07.4f, %07.4f) Run-P2 %07.4f (%07.4f, %07.4f) Run-P3 %07.4f (%07.4f, %07.4f) Setup-P1 %07.4f (%07.4f, %07.4f) Setup-P2 %07.4f (%07.4f, %07.4f) Total %07.4f\n",
               m_name.c_str(),
               instIdx,
               m_delayMs,
               m_nIterations,
               avgElapsedTimeUsRun[PUSCH_RUN_SUB_SLOT_PROC],
               minElapsedTimeUsRun[PUSCH_RUN_SUB_SLOT_PROC],
               maxElapsedTimeUsRun[PUSCH_RUN_SUB_SLOT_PROC],
               avgElapsedTimeUsRun[PUSCH_RUN_FULL_SLOT_PROC],
               minElapsedTimeUsRun[PUSCH_RUN_FULL_SLOT_PROC],
               maxElapsedTimeUsRun[PUSCH_RUN_FULL_SLOT_PROC],
               avgElapsedTimeUsRun[PUSCH_RUN_FULL_SLOT_COPY],
               minElapsedTimeUsRun[PUSCH_RUN_FULL_SLOT_COPY],
               maxElapsedTimeUsRun[PUSCH_RUN_FULL_SLOT_COPY],
               avgElapsedTimeUsSetup[PUSCH_SETUP_PHASE_1],
               minElapsedTimeUsSetup[PUSCH_SETUP_PHASE_1],
               maxElapsedTimeUsSetup[PUSCH_SETUP_PHASE_1],
               avgElapsedTimeUsSetup[PUSCH_SETUP_PHASE_2],
               minElapsedTimeUsSetup[PUSCH_SETUP_PHASE_2],
               maxElapsedTimeUsSetup[PUSCH_SETUP_PHASE_2],
               avgTotalElapsedTimeUs);
    
        printf("%s Pipeline[%02d]: Total time usec GPU (CUDA event) %07.4f CPU (wall clock) %07.4f\n",
               m_name.c_str(),
               instIdx,
               avgTotalElapsedEvtTimeUs,
               avgTotalElapsedTimeUs);

        // if(debug)
        {
            float accumCpuThrdNotifyDelayUs = 0.0f, accumIdealDelayKernelTimeUs = 0.0f;
            float maxCpuThrdNotifyDelayUs = std::numeric_limits<float>::min(), maxIdealDelayKernelTimeUs = std::numeric_limits<float>::min();
            float minCpuThrdNotifyDelayUs = std::numeric_limits<float>::max(), minIdealDelayKernelTimeUs = std::numeric_limits<float>::max();
            for(uint32_t iterIdx = 0; iterIdx < m_nIterations; ++iterIdx)
            {                
                // Start-event record to notify delay: delay in CPU thread signaling 
                duration<float, std::micro> diff = m_dbgTimePts0[instIdx][iterIdx] - m_dbgStartTimePt[iterIdx];
                float cpuThrdNotifyDelayUs = diff.count();

                updateAccumMinMaxVals(cpuThrdNotifyDelayUs,
                                      accumCpuThrdNotifyDelayUs,
                                      maxCpuThrdNotifyDelayUs,
                                      minCpuThrdNotifyDelayUs);

                // Start-event notify to pipelne Run start delay: this should be the delay kernel time
                diff = m_dbgTimePts2[instIdx][iterIdx] - m_dbgStartTimePt[iterIdx];
                float idealDelayKernelTimeUs = diff.count();

                updateAccumMinMaxVals(idealDelayKernelTimeUs,
                                      accumIdealDelayKernelTimeUs,
                                      maxIdealDelayKernelTimeUs,
                                      minIdealDelayKernelTimeUs);
            }

            printf("%s Pipeline[%02d]: Debug - start-event record to notify delay in usec (wall clock) %07.4f (%07.4f, %07.4f)\n",
                   m_name.c_str(),
                   instIdx,
                   accumCpuThrdNotifyDelayUs/m_nIterations,
                   minCpuThrdNotifyDelayUs,
                   maxCpuThrdNotifyDelayUs);

            printf("%s Pipeline[%02d]: Debug - start-event notify to pipelne launch start delay in usec (wall clock) %07.4f (%07.4f, %07.4f)\n",
                   m_name.c_str(),
                   instIdx,
                   accumIdealDelayKernelTimeUs/m_nIterations,
                   minIdealDelayKernelTimeUs,
                   maxIdealDelayKernelTimeUs);
        }
    }
}


//-------------------------------------------------------------------------------------
// function computes and displays codeblock and CRC errors

void DisplayBler(EvalDataset& evalDataset, DynApiDataset const& dynApiDataset, uint32_t instIdx, bool debug, bool drmDebug)
{
    // output of Rx pipeline:
    uint32_t* pCbCrcs                 = dynApiDataset.DataOut.pCbCrcs;
    uint32_t* pTbCrcs                 = dynApiDataset.DataOut.pTbCrcs;
    uint8_t*  pEstTbBytes             = dynApiDataset.DataOut.pTbPayloads;


    // compute SCH Bler
    uint32_t nCbErrors =  evalDataset.computeNumCbErrors(dynApiDataset);

    // compute UCI BLER
    evalDataset.computeNumUciCbErrors(dynApiDataset, false);
    // uint32_t nUciCbs, nUciCbErrors, nSchCbs, nSchCbErrors;
    // evalDataset.computeNumUciCbErrors(dynApiDataset, nUciCbErrors, nUciCbs);

    // display UCI BLER
    // if(nUciCbs>0)
    // {
    //     printf("Cell # %d : Metric - UCI Block Error Rate           : %4.4f (Error CBs %d, Mismatched CBs %d, Total CBs %d)\n",
    //             instIdx,
    //             static_cast<float>(nUciCbErrors) / static_cast<float>(nUciCbs),
    //             nUciCbErrors,
    //             nUciCbErrors,
    //             nUciCbs);
    // }
    // else if(nUciCbs==0)
    // {
    //     printf("Cell # %d : Metric - UCI Block Error Rate           : %4.4f (Error CBs %d, Mismatched CBs %d, Total CBs %d)\n",
    //             instIdx,
    //             0.0,
    //             nUciCbErrors,
    //             nUciCbErrors,
    //             nUciCbs);
    // }
    

    // Display CRC/Byte errors if any
    if (debug)
    {
        for(int i = 0; i < evalDataset.nCbs; i++)
        {
            if(pCbCrcs[i] != 0)
            {
                printf("ERROR: PuschRx Pipeline[%d]: CRC of code block [%d] failed!\n", instIdx, i);
            }
            else
            {
                printf("PuschRx Pipeline[%d]: CRC of code block [%d] pass!\n", instIdx, i);
            }
        }
        for(int i = 0; i < evalDataset.nTbs; i++)
        {
            if(pTbCrcs[i] != 0)
            {
                printf("ERROR: PuschRx Pipeline[%d]: CRC of transport block [%d] failed!\n", instIdx, i);
            }
            else
            {
                printf("PuschRx Pipeline[%d]: CRC of transport block [%d] pass!\n", instIdx, i);
            }
        }
#if 0
        for(int j = 0; j < evalDataset.nBytes; j++)
        {
            if(pTrueTbBytes[j] != pEstTbBytes[j])
            {
                printf("ERROR in PuschRx pipeline %d: output byte at position %d: %x != %x\n", instIdx, j, pTrueTbBytes[j], pEstTbBytes[j]);
            }
        }
#else
        uint32_t estTbBytesCount = 0;
        for(int j = 0; j < evalDataset.nBytesVec.size(); j++)
        {
            // true TB bytes:
            uint8_t*  pTrueTbBytes = static_cast<uint8_t*> (evalDataset.tTrueTbBytesVec[j].addr());
            for(int k = 0; k < evalDataset.nBytesVec[j]; k++)
            {
                if(pTrueTbBytes[k] != pEstTbBytes[estTbBytesCount++])
                {
                    printf("PuschRx pipeline %d: Error byteIdx[%d] Ref 0x%02x Gpu 0x%02x\n", instIdx, j, pTrueTbBytes[j], pEstTbBytes[j]);
                }
                else
                {
                    printf("PuschRx pipeline %d: byteIdx[%d] Ref 0x%02x Gpu 0x%02x\n", instIdx, j, pTrueTbBytes[j], pEstTbBytes[j]);
                }
            }
        }
#endif
    }

    // Look at de-rate-match index buffer
    if (drmDebug)
    {
        uint32_t *pRef = static_cast<uint32_t*> (evalDataset.tReference_derateCbsIndices.addr());
        uint32_t *pRefSizes = static_cast<uint32_t*> (evalDataset.tReference_derateCbsIndicesSizes.addr());
        uint32_t cb_tot = 0;
        bool errorFlag = false;

        for (int n=0; n<evalDataset.nTbs; n++) // loop over TBs
        {
            uint32_t *buf = dynApiDataset.uePrmsVec[n].debug_d_derateCbsIndices;

            for (int cb=0; cb<evalDataset.nCbsPerTbVec[n]; cb++) // loop over CBs for this TB
            {
                for (int k=0; k<pRefSizes[cb_tot]; k++)
                {
                    uint32_t refIdx = *pRef++;
                    uint32_t dutIdx = *buf++;
                    if (refIdx != dutIdx)
                    {
                        printf("De-rate-match indexing error in TB %d CB %d Position %d: actual %u != %u expected\n",n,cb,k,dutIdx,refIdx);
                        errorFlag = true;
                    }
                }
                cb_tot++;
            }
        }
        if (errorFlag)
        {
            exit(1);
        }
    }
}



