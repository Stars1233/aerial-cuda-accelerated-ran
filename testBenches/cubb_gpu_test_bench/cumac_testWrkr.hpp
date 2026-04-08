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

#if !defined(CUMAC_TESTWRKR_HPP_INCLUDED_)
#define CUMAC_TESTWRKR_HPP_INCLUDED_

#include <type_traits>

// test bench common header
#include "testbench_common.hpp"

// Convert CMake flag to compile-time constant
#ifdef AERIAL_CUMAC_ENABLE
constexpr bool cumac_enabled = true;
#else
constexpr bool cumac_enabled = false;
#endif

// Forward declarations
class CuMACTestWorkerImpl;
class NullCuMACTestWorker;

// Max supported MAC slots per pattern (15 for mode 6 DDDSUUDDDD+DDDSU)
constexpr uint8_t kMaxMacSlotsPerPattern = 15;

/// cuMAC options configurable from YAML (see cumac_options in config YAML)
struct CumacOptions
{
    uint8_t modules_called[4] = {1, 1, 1, 1}; //!< UE selection, PRG allocation, layer selection, MCS selection: 0 disable, 1 enable
    int     cumac_light_weight_flag[kMaxMacSlotsPerPattern] = {1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2}; //!< Per-slot kernel weight: 0 heavy, 1 light (SRS comp), 2 light (SRS load)
    float   perc_sm_num_thrd_blk[kMaxMacSlotsPerPattern] = {6.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f}; //!< Per-slot SM/thread-block percentage for light-weight kernels
    uint8_t half_precision = 0;    //!< Precision mode: 0 float32, 1 half
    uint8_t sch_alg = 1;           //!< Scheduling algorithm: 0 round-robin, 1 proportional fair
    uint8_t hetero_ue_sel_cells = 0; //!< Heterogeneous UE selection across cells: 0 disabled, 1 enabled
    uint8_t config_slot_count = 0; //!< Length of per-slot arrays from YAML (0 = not from YAML). Must match nMacSlots when set.
};

// cuMAC-specific includes and message payloads (only when cuMAC is enabled)
#ifdef AERIAL_CUMAC_ENABLE
// cuMAC header
#include "cumac.h"

//----------------------------------------------------------------------------------------------------------
// cuMAC test worker message payloads for testWrkrMsg

// CUMAC_TEST_WRKR_CMD_MSG_MAC_RUN payload
struct cuMACTestMacRunMsgPayload
{
    bool                       rsp;
    cudaEvent_t                startEvent;
    std::vector<cuphy::event>* externalInterSlotEventVec; // external events for sync up with pattern
};

// CUMAC_TEST_WRKR_RSP_MSG_MAC_RUN payload
struct cuMACTestMacRunRspMsgPayload
{
    uint32_t                      workerId;
    std::shared_ptr<cuphy::event> shPtrStopEvent;
};

// CUMAC_TEST_WRKR_CMD_MSG_MAC_SETUP payload
struct cuMACTestMacSetupMsgPayload
{
    bool                       rsp;
    std::vector<std::string>   inFileNamesMac;
    std::vector<uint8_t>       macSlotRunFlag;  // Slot configuration for MAC
};
#endif // AERIAL_CUMAC_ENABLE

//----------------------------------------------------------------------------------------------------------
// CuMACTestWorkerImpl - Real implementation when cuMAC is enabled
//----------------------------------------------------------------------------------------------------------
#ifdef AERIAL_CUMAC_ENABLE
class CuMACTestWorkerImpl : public testWorker{
public:
    CuMACTestWorkerImpl(std::string const& name, uint32_t workerId, int cpuId, int gpuId, int cpuThrdSchdPolicy, int cpuThrdPrio, uint32_t mpsSubctxSmCount, std::shared_ptr<testWrkrCmdQ>& cmdQ, std::shared_ptr<testWrkrRspQ>& rspQ, int uldlMode, uint32_t debugMessageLevel, bool useGreenContexts, const cuphy::cudaGreenContext& greenCtx);

    CuMACTestWorkerImpl(CuMACTestWorkerImpl const&) = delete;
    CuMACTestWorkerImpl& operator=(CuMACTestWorkerImpl const&) = delete;
    CuMACTestWorkerImpl(CuMACTestWorkerImpl&&) = default;
    CuMACTestWorkerImpl& operator=(CuMACTestWorkerImpl&&) = delete;
    virtual ~CuMACTestWorkerImpl();

    void init(uint32_t nStrms, uint32_t nItrsPerStrm, uint32_t nTimingItrs, std::map<std::string, int>& cuStrmPrioMap, std::shared_ptr<cuphy::buffer<uint32_t, cuphy::pinned_alloc>>& shPtrCpuGpuSyncFlag, bool waitRsp = true, uint32_t longPattern = 0, bool internalTimer = false, uint8_t lightWeightFlagOffset = 0);
    void macInit(std::vector<std::string> inFileNamesMac, uint32_t nMacSlots, bool ref_check_mac, bool waitRsp = true);
    void macSetup(std::vector<std::string> inFileNamesMac, std::vector<uint8_t> macSlotRunFlag = {}, bool waitRsp = true);
    void macRun(cudaEvent_t startEvent, std::shared_ptr<cuphy::event>& shPtrStopEvent, bool waitRsp, std::vector<cuphy::event>* pdschInterSlotEventVec);
    /**
     * @brief Apply cuMAC configuration options to this worker.
     * @param opts cuMAC options (scheduler modules, kernel weights, precision, algorithm, etc.) typically parsed from YAML.
     */
    void setCumacOptions(const CumacOptions& opts);

    [[nodiscard]] std::optional<std::vector<float>> getTotMACSlotStartTime();
    [[nodiscard]] std::optional<std::vector<float>> getTotMacSlotEndTime();
    [[nodiscard]] std::optional<std::vector<cuphy::event>*> getMacInterSlotEventVecPtr();

private:
    void msgProcess(std::shared_ptr<testWrkrCmdMsg>& shPtrMsg); // reference to shared_ptr optional (used for performance)
    void initHandler(std::shared_ptr<void>& shPtrPayload);
    void macInitHandler(std::shared_ptr<void>& shPtrPayload);
    void exitHandler(std::shared_ptr<void>& shPtrPayload);
    void macSetupHandler(std::shared_ptr<void>& shPtrPayload);
    void macRunHandler(std::shared_ptr<void>& shPtrPayload);
    void setWaitValHandler(std::shared_ptr<void>& shPtrPayload);
    void evalHandler(std::shared_ptr<void>& shPtrPayload);
    void printHandler(std::shared_ptr<void>& shPtrPayload);
    void resetEvalHandler(std::shared_ptr<void>& shPtrPayload);
    void readSmIdsHandler(std::shared_ptr<void>& shPtrPayload);

    void run(const cuphy::cudaGreenContext& my_green_context);
    void createCuOrGreenCtx(const cuphy::cudaGreenContext& my_green_context); // also calls testWorker::createCuCtx() or testWorker::createCuGreenCtx() under the hood

    uint32_t m_nStrmsMac;       // number of parallel streams to execute mac workload, currently fix to 1
    uint32_t m_nItrsPerStrmMac;       // number of iteration per stream
    uint32_t m_nMacSlots; // number of cuMAC slots per pattern, currently 0 or 8
    // m_nMacSlots = m_nStrmsMac * m_nItrsPerStrmMac

    // Stream PRIOs
    uint32_t m_cuStrmPrioMac;

    // streams. Dim: m_nStrms
    std::vector<cuphy::stream> m_cuStrmsMac;

    // Pipeline handles. Dim: m_nStrms x 1
    std::vector<std::unique_ptr<cumac::cumacSubcontext>> m_macPipes; // Dim: m_nMacSlots = m_nStrmsMac * m_nItrsPerStrmMac
    std::vector<std::unique_ptr<cumac::cumacSubcontext>> m_macPipes_cpuRef; // Dim: m_nMacSlots = m_nStrmsMac * m_nItrsPerStrmMac

    std::unique_ptr<cuphy::event>  m_uqPtrMacIterStopEvent;

    std::vector<cuphy::event> m_stopEvents;       // Dim: m_nStrms * 8
    std::vector<cuphy::event> m_macInterSlotStartEventVec;  // Dim: m_nStrms * 8, events for each MAC slot, only useful when m_internalTimer = true

    // start event
    std::unique_ptr<cuphy::event> m_uqPtrTimeStartEvent;
    // multiple MAC slots
    std::vector<cuphy::event>              m_timeMacSlotStartEvents;
    std::vector<cuphy::event>              m_timeMacSlotEndEvents;

    // timing objects.
    std::vector<float>                           m_totMacSlotStartTime; // Dim: nItrsPerStrm,  from common start event  -> each cuMAC slot start
    std::vector<float>                           m_totMacSlotEndTime; // Dim: nItrsPerStrm, from common start event  -> each cuMAC slot end

    // variable delay memory pointer
    bool m_internalTimer; // true: using internal timer, false: using external timer
    cuphy::buffer<uint64_t, cuphy::device_alloc> m_GPUtime_d;

    // evaluation objects
    std::vector<std::vector<uint32_t>> m_maxNumCbErrors; // Dim: m_nStrms x m_nItrsPerStrm
    uint32_t                           m_nTimingItrs;

    // mac configurations
    bool                                         m_ref_check_mac;
    uint32_t                                     m_fp16Mode;
    bool                                         m_runMac;
    bool                                         m_macCtx;
    uint8_t                                      m_lightWeightFlagOffset;
    std::vector<uint8_t>                         m_macSlotRunFlag;  // Slot configuration for MAC

    // cuMAC kernel configurations, more details in cuMAC/src/cumacSubcontext.h (defaults; overridable via setCumacOptions from YAML)
    uint8_t m_modulesCalled[4] = {1, 1, 1, 1}; // UE selection, PRG allocation, layer selection, MCS selection: 0 disable, 1 enable
    int     m_cumacLightWeightFlag[kMaxMacSlotsPerPattern] = {1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2}; // 0 heavy; 1 light (SRS comp); 2 light (SRS load) per slot
    float   m_percSmNumThrdBlk[kMaxMacSlotsPerPattern] = {6.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f}; // percentage of SMs for thread blocks in light-weight kernels
    uint8_t m_halfPrecision = 0;      // 0 float32, 1 half
    uint8_t m_schAlg = 1;             // 0 RR, 1 PF
    uint8_t m_heteroUeSelCells = 0;   // 0 disabled, 1 heterogeneous UE selection across cells
};
#else
// Stub class when cuMAC is disabled - only declaration needed
class CuMACTestWorkerImpl : public testWorker {
public:
    CuMACTestWorkerImpl(std::string const& name, uint32_t workerId, int cpuId, int gpuId, int cpuThrdSchdPolicy, int cpuThrdPrio, uint32_t mpsSubctxSmCount, std::shared_ptr<testWrkrCmdQ>& cmdQ, std::shared_ptr<testWrkrRspQ>& rspQ, int uldlMode, uint32_t debugMessageLevel, bool useGreenContexts, const cuphy::cudaGreenContext& greenCtx) 
        : testWorker(name, workerId, cpuId, gpuId, cpuThrdSchdPolicy, cpuThrdPrio, mpsSubctxSmCount, cmdQ, rspQ, uldlMode, debugMessageLevel, useGreenContexts) {}
    
    void init(uint32_t, uint32_t, uint32_t, std::map<std::string, int>&, std::shared_ptr<cuphy::buffer<uint32_t, cuphy::pinned_alloc>>&, bool = true, uint32_t = 0, bool = false, uint8_t = 0) {}
    void macInit(std::vector<std::string>, uint32_t, bool, bool = true) {}
    void macSetup(std::vector<std::string>, std::vector<uint8_t> = {}, bool = true) {}
    void macRun(cudaEvent_t, std::shared_ptr<cuphy::event>&, bool, std::vector<cuphy::event>*) {}
    void setCumacOptions(const CumacOptions&) {}
    [[nodiscard]] std::optional<std::vector<float>> getTotMACSlotStartTime() { return std::nullopt; }
    [[nodiscard]] std::optional<std::vector<float>> getTotMacSlotEndTime() { return std::nullopt; }
    [[nodiscard]] std::optional<std::vector<cuphy::event>*> getMacInterSlotEventVecPtr() { return std::nullopt; }
};
#endif

//----------------------------------------------------------------------------------------------------------
// NullCuMACTestWorker - Null Object implementation when cuMAC is disabled
//----------------------------------------------------------------------------------------------------------
class NullCuMACTestWorker : public testWorker
{
public:
    NullCuMACTestWorker(std::string const& name, uint32_t workerId, int cpuId, int gpuId, int cpuThrdSchdPolicy, int cpuThrdPrio, uint32_t mpsSubctxSmCount, std::shared_ptr<testWrkrCmdQ>& cmdQ, std::shared_ptr<testWrkrRspQ>& rspQ, int uldlMode, uint32_t debugMessageLevel, bool useGreenContexts, const cuphy::cudaGreenContext& greenCtx)
        : testWorker(name, workerId, cpuId, gpuId, cpuThrdSchdPolicy, cpuThrdPrio, mpsSubctxSmCount, cmdQ, rspQ, uldlMode, debugMessageLevel, useGreenContexts) 
    {
        NVLOGW_FMT(NVLOG_TESTBENCH_MAC, "cuMAC support disabled - using null cuMACTestWorker implementation");
    }

    NullCuMACTestWorker(NullCuMACTestWorker const&) = delete;
    NullCuMACTestWorker& operator=(NullCuMACTestWorker const&) = delete;
    NullCuMACTestWorker(NullCuMACTestWorker&&) = default;
    NullCuMACTestWorker& operator=(NullCuMACTestWorker&&) = delete;
    virtual ~NullCuMACTestWorker() = default;
    
    // No-op implementations with identical signatures to CuMACTestWorkerImpl
    void init(uint32_t nStrms, uint32_t nItrsPerStrm, uint32_t nTimingItrs, std::map<std::string, int>& cuStrmPrioMap, std::shared_ptr<cuphy::buffer<uint32_t, cuphy::pinned_alloc>>& shPtrCpuGpuSyncFlag, bool waitRsp = true, uint32_t longPattern = 0, bool internalTimer = false, uint8_t lightWeightFlagOffset = 0) 
    {
        // No-op implementation
    }

    void macInit(std::vector<std::string> inFileNamesMac, uint32_t nMacSlots, bool ref_check_mac, bool waitRsp = true) 
    {
        // No-op implementation
    }

    void macSetup(std::vector<std::string> inFileNamesMac, std::vector<uint8_t> macSlotRunFlag, bool waitRsp = true) 
    {
        // No-op implementation
    }

    void macRun(cudaEvent_t startEvent, std::shared_ptr<cuphy::event>& shPtrStopEvent, bool waitRsp, std::vector<cuphy::event>* pdschInterSlotEventVec) 
    {
        // No-op implementation
    }

    void setCumacOptions(const CumacOptions&) {}

    [[nodiscard]] std::optional<std::vector<float>> getTotMACSlotStartTime() 
    { 
        return std::nullopt; 
    }

    [[nodiscard]] std::optional<std::vector<float>> getTotMacSlotEndTime() 
    { 
        return std::nullopt; 
    }

    [[nodiscard]] std::optional<std::vector<cuphy::event>*> getMacInterSlotEventVecPtr() 
    { 
        return std::nullopt; 
    }

private:
    void run(const cuphy::cudaGreenContext& my_green_context) 
    {
        // No-op implementation
    }
};

//----------------------------------------------------------------------------------------------------------
// Type selection using std::conditional_t
//----------------------------------------------------------------------------------------------------------
using CuMACTestWorkerType = std::conditional_t<cumac_enabled, CuMACTestWorkerImpl, NullCuMACTestWorker>;

// Alias for backward compatibility
using cuMACTestWorker = CuMACTestWorkerType;

#endif // !defined(CUMAC_TESTWRKR_HPP_INCLUDED_)
