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

#if !defined(PUSCH_RX_TEST_HPP_INCLUDED_)
#define PUSCH_RX_TEST_HPP_INCLUDED_

#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>

#include <chrono>
#include "pusch_rx.hpp"
#include "cuphy_channels.hpp"
#include "datasets.hpp"


using Clock     = std::chrono::high_resolution_clock;
using TimePoint = std::chrono::time_point<Clock>;

class PuschRxTest {
public:
    void Setup(uint32_t                               nIterations,
               bool                                   enableNvprof,
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
               const cuphy::cudaGreenContext&         greenCtx);

    void SetupPipelines(std::vector<std::string>& inputFileNameVec,
                        std::vector<std::string>& outputFileNameVec,
                        int                       descramblingOn,
                        uint32_t                  fp16Mode);

    void PipelineWrkrEntry(uint32_t instIdx, const std::string& trtYamlInput, const cuphy::cudaGreenContext& puschGreenCtx);
    void WaitForCompletion();
    void DisplayMetricsApi(bool debug = false, bool throughput = false, bool bler = true, bool execTime = true);

    PuschRxTest(std::string const& name, uint32_t nPuschRxInst, bool useCuCtxs, cudaStream_t& delayCuStrm, std::vector<int>& cuStrmPrios,
                bool& startSyncPt, std::mutex& cvStartSyncPtMutex, std::condition_variable& cvStartSyncPt, std::atomic<std::uint32_t>& atmSyncPtWaitCnt,
                std::vector<uint32_t>& mpsActiveThrdPcts, uint32_t harq_attempts, uint32_t ldpcLaunchMode, uint32_t nMaxLdpcHetConfigs, bool drmDebug,
                bool debug, bool debugEqualizer, bool useGreenCtx, uint8_t nMaxTbPerNode);
    PuschRxTest(PuschRxTest const&) = delete;
    PuschRxTest& operator=(PuschRxTest const&) = delete;
    ~PuschRxTest();

    static constexpr uint32_t N_SLOT_DATA_BUF = 1;

private:
    void SpawnPipelineWrkrThrds(const std::string& trtYamlInput, const cuphy::cudaGreenContext& greenCtx);
    void runTest(uint32_t instIdx, uint32_t transmission, uint32_t iterIdx, cuphy::pusch_rx &puschRxPipe, StaticApiDataset &staticApiDataset, DynApiDataset &dynApiDataset);

    void DisplayTiming(uint32_t instIdx, uint32_t slotIdx, uint32_t nBytes, bool debug = false, bool throughput = true, bool execTime = true);

    std::string                               m_name;
    std::vector<std::vector<std::string>>     m_inputFileNameVec;
    uint32_t                                  m_nPuschRxInst;
    bool                                      m_useCuCtxs;
    bool                                      m_useGreenCtxs;
    std::vector<CUcontext>                    m_cuCtxs;
    std::vector<std::thread>                  m_wrkrThrds;
    std::vector<std::mutex>                   m_wrkrThrdMutexes;

    // Condition variable and supporting mutex to issue sync point from signal from main thread to all threads
    // and pipelines
    bool&                       m_startSyncPt;
    std::mutex&                 m_cvStartSyncPtMutex;
    std::condition_variable&    m_cvStartSyncPt;
    std::atomic<std::uint32_t>& m_atmSyncPtWaitCnt;

    // Condition variable and supporting mutex to signal from pipeline 0 back to main thread
    bool                    m_eStartProcRecorded = false;
    std::mutex              m_cvStartProcRecMutex, m_mpsActiveThrdPctEnvVarMutex;
    std::condition_variable m_cvStartProcRec;

    std::vector<int> m_wrkrThrdSchdPolicies;
    std::vector<int> m_wrkrThrdPrios;
    std::vector<uint32_t> m_mpsActiveThrdPcts;

    std::vector<int> m_cpuIds;
    std::vector<int> m_gpuIds;

    cuphyPuschLdpcKernelLaunch_t m_ldpcLaunchMode;
    uint32_t m_nMaxLdpcHetConfigs;
    uint8_t  m_nMaxTbPerNode   = 1;

    bool     m_debug           = false;
    bool     m_enableNvProf    = false;
    uint32_t m_nIterations     = 0;
    int      m_descramblingOn  = true;
    uint32_t m_delayMs         = 0;
    uint32_t m_sleepDurationUs = 0;
    uint32_t m_fp16Mode        = 1;
    uint32_t m_procModeBmsk    = 0;
    uint32_t m_harqAttempts    = 1;
    bool     m_drmDebug        = false;
    bool     m_debugEqualizer  = false;

    cudaStream_t  m_delayCuStrm = 0; // CUDA stream on which the delay kernel is launched
    cudaEvent_t   m_eStartProc;
    cudaEvent_t   m_eEndProc;

    std::atomic<std::uint32_t> m_atmEndProcCnt;

    std::vector<cuphy::buffer<uint8_t, cuphy::device_alloc>> m_harqBuffers;
    
    std::vector<cuphy::buffer<float, cuphy::device_alloc>> m_foCompensationBuffer;

    template <typename T, typename unit>
    using duration = std::chrono::duration<T, unit>;
    
    // Stream for workload submission
    std::vector<cuphy::stream> m_cuphyStrms;
    
    // Event timers to measure timing
    std::vector<std::array<cuphy::event_timer, PUSCH_SETUP_MAX_PHASES>> m_evtTmrsSetup;
    std::vector<std::array<cuphy::event_timer, PUSCH_RUN_MAX_PHASES>> m_evtTmrsRun;

    // Elapsed times measured with events and CPU clocks
    std::vector<std::vector<std::array<float, PUSCH_SETUP_MAX_PHASES>>> m_elapsedEvtTimeUsSetup;
    std::vector<std::vector<std::array<float, PUSCH_RUN_MAX_PHASES>>> m_elapsedEvtTimeUsRun;
    std::vector<std::vector<std::array<float, PUSCH_SETUP_MAX_PHASES>>> m_elapsedTimesUsSetup;
    std::vector<std::vector<std::array<float, PUSCH_RUN_MAX_PHASES>>> m_elapsedTimesUsRun;

    // For debug
    std::vector<TimePoint> m_dbgStartTimePt;
    std::vector<std::vector<TimePoint>> m_dbgTimePts0, m_dbgTimePts1, m_dbgTimePts2;
    std::vector<std::string> m_outputFileNameVec;
    std::unique_ptr<hdf5hpp::hdf5_file> m_debugFile;
};

void DisplayBler(EvalDataset& evalDataset, DynApiDataset const& dynApiDataset, uint32_t instIdx = 0, bool debug = false, bool drmDebug = false);




#endif // !defined(PUSCH_RX_TEST_HPP_INCLUDED_)
