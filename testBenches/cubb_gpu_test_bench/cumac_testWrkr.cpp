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

#include "cumac_testWrkr.hpp"

#ifdef AERIAL_CUMAC_ENABLE
// Real cuMAC implementation when enabled
#include "tvLoadingTest.h" // for comparing GPU and CPU results if ref_check_mac enabled

CuMACTestWorkerImpl::CuMACTestWorkerImpl(std::string const& name, uint32_t workerId, int cpuId, int gpuId, int cpuThrdSchdPolicy, int cpuThrdPrio, uint32_t mpsSubctxSmCount, std::shared_ptr<testWrkrCmdQ>& cmdQ, std::shared_ptr<testWrkrRspQ>& rspQ, int uldlMode, uint32_t debugMessageLevel, bool useGreenContexts, const cuphy::cudaGreenContext& greenCtx) :
    testWorker(name, workerId, cpuId, gpuId, cpuThrdSchdPolicy, cpuThrdPrio, mpsSubctxSmCount, cmdQ, rspQ, uldlMode, debugMessageLevel, useGreenContexts),
    m_nStrmsMac(0),
    m_runMac(false),
    m_fp16Mode(1),
    m_ref_check_mac(false)
{
    // Default cuMAC config (overridable via setCumacOptions from YAML)
    CumacOptions def;
    setCumacOptions(def);
    // Start worker thread
    m_thrd = std::thread(&CuMACTestWorkerImpl::run, this, std::cref(greenCtx));
}

void CuMACTestWorkerImpl::setCumacOptions(const CumacOptions& opts)
{
    for(int i = 0; i < 4; i++)
        m_modulesCalled[i] = opts.modules_called[i];
    for(int i = 0; i < kMaxMacSlotsPerPattern; i++)
    {
        m_cumacLightWeightFlag[i] = opts.cumac_light_weight_flag[i];
        m_percSmNumThrdBlk[i]     = opts.perc_sm_num_thrd_blk[i];
    }
    m_halfPrecision     = opts.half_precision;
    m_schAlg            = opts.sch_alg;
    m_heteroUeSelCells = opts.hetero_ue_sel_cells;
}

CuMACTestWorkerImpl::~CuMACTestWorkerImpl()
{
    // implicit call destructor of testWorker::~testWorker()
}

void CuMACTestWorkerImpl::createCuOrGreenCtx(const cuphy::cudaGreenContext& my_green_context)
{
    if (m_useGreenContexts)
    {
#if CUDA_VERSION >= 12040
        //createCuGreenCtx is a misnomer as the context has already been created.
        //Here we just set it for a given thread; each thread only has a single green context set as current
        testWorker::createCuGreenCtx(my_green_context);
#endif
    }
    else
    {
        testWorker::createCuCtx();
    }

    m_uqPtrMacIterStopEvent     = std::make_unique<cuphy::event>(cudaEventDisableTiming);
    
    printf("\n--> Runs %d stream(s) in parallel", m_nStrmsMac);
    if(m_name.compare("MacTestWorker") == 0)
    {
        printf("\n--> Each stream processes %d cuMAC test-vector(s) in series\n\n", m_nItrsPerStrmMac);
    }
}

void CuMACTestWorkerImpl::run(const cuphy::cudaGreenContext& my_green_context)
{
    createCuOrGreenCtx(std::cref(my_green_context));

    DEBUG_TRACE("%s [tid %s][wrkrCtxId 0x%0lx currCtxId 0x%0lx]: Enter run loop for worker %d\n", m_name.c_str(), getThreadIdStr().c_str(), getCuCtxId(), getCurrCuCtxId(), m_wrkrId);
    for(;;)
    {
        std::shared_ptr<testWrkrCmdMsg> shPtrCmd;
        DEBUG_TRACE("%s id %d [tid %s][wrkrCtxId 0x%0lx currCtxId 0x%0lx]: run loop for worker\n", m_name.c_str(), m_wrkrId, getThreadIdStr().c_str(), getCuCtxId(), getCurrCuCtxId());
        // DEBUG_TRACE("Thread %s: Begin message receive\n", m_name.c_str());
        m_shPtrCmdQ->receive(shPtrCmd, m_wrkrId);
        DEBUG_TRACE("%s id %d [tid %s][wrkrCtxId 0x%0lx currCtxId 0x%0lx]: Received message: %s\n", m_name.c_str(), m_wrkrId, getThreadIdStr().c_str(), getCuCtxId(), getCurrCuCtxId(), TEST_WRKR_CMD_MSG_TO_STR[shPtrCmd->type]);

        // DEBUG_TRACE("Thread %s: Begin message processing: %s\n", m_name.c_str(), TEST_WRKR_CMD_MSG_TO_STR[shPtrMsg->type];
        msgProcess(shPtrCmd);
        if(COMMON_TEST_WRKR_CMD_MSG_EXIT == shPtrCmd->type) break;

        // DEBUG_TRACE("Thread %s: Processed message: %s\n", m_name.c_str(), TEST_WRKR_CMD_MSG_TO_STR[shPtrMsg->type]);
    }
    DEBUG_TRACE("%s [tid %s][wrkrCtxId 0x%0lx currCtxId 0x%0lx]: Exit run loop for worker %d\n", m_name.c_str(), getThreadIdStr().c_str(), getCuCtxId(), getCurrCuCtxId(), m_wrkrId);
}

// Start of sending receiving message functions
void CuMACTestWorkerImpl::init(uint32_t nStrms, uint32_t nItrsPerStrm, uint32_t nTimingItrs, std::map<std::string, int>& cuStrmPrioMap, std::shared_ptr<cuphy::buffer<uint32_t, cuphy::pinned_alloc>>& shPtrCpuGpuSyncFlag, bool waitRsp, uint32_t longPattern, bool internalTimer, uint8_t lightWeightFlagOffset)
{
    m_nStrmsMac             = nStrms;
    m_nItrsPerStrmMac       = nItrsPerStrm;
    m_nTimingItrs           = nTimingItrs; 
    m_longPattern           = longPattern;
    m_internalTimer         = internalTimer;
    m_lightWeightFlagOffset = lightWeightFlagOffset;

    if(m_longPattern < 0 || m_longPattern > 8)
    {
        NVLOGF_FMT(NVLOG_TESTBENCH_MAC, AERIAL_TESTBENCH_EVENT, "Invalid longPattern value in cuMACTestWorker : {}\n", m_longPattern);
    }
    
    // send init message
    auto shPtrPayload                 = std::make_shared<commnTestInitMsgPayload>();
    shPtrPayload->rsp                 = true;
    shPtrPayload->shPtrCpuGpuSyncFlag = shPtrCpuGpuSyncFlag;
    shPtrPayload->cuStrmPrioMap       = cuStrmPrioMap;
    
    auto shPtrMsg = std::make_shared<testWrkrCmdMsg>(COMMON_TEST_WRKR_CMD_MSG_INIT, m_wrkrId, shPtrPayload);
    DEBUG_TRACE("MainThread [tid %s][currCtxId 0x%0lx]: Sending message: %s\n", getThreadIdStr().c_str(), getCurrCuCtxId(), TEST_WRKR_CMD_MSG_TO_STR[shPtrMsg->type]);
    m_shPtrCmdQ->send(shPtrMsg);

    // response
    if(waitRsp)
    {
        std::shared_ptr<testWrkrRspMsg> shPtrRsp;
        m_shPtrRspQ->receive(shPtrRsp, COMMON_TEST_WRKR_RSP_MSG_INIT, m_wrkrId);
    }
}

void CuMACTestWorkerImpl::macInit(std::vector<std::string> inFileNamesMac, uint32_t nMacSlots, bool ref_check_mac, bool waitRsp)
{
    if (nMacSlots > kMaxMacSlotsPerPattern) {
        printf("ERROR: nMacSlots (%u) exceeds kMaxMacSlotsPerPattern (%u)\n", nMacSlots, kMaxMacSlotsPerPattern);
        exit(1);
    }
    m_nMacSlots         = nMacSlots;
    m_ref_check_mac     = ref_check_mac;
    
    // Send initialization message
    auto shPtrPayload         = std::make_shared<commnTestInitMsgPayload>();
    shPtrPayload->rsp         = true;
    shPtrPayload->inFileNames = inFileNamesMac;

    if(inFileNamesMac.size() == 0)
    {
        m_runMac = false;
        return;
    }
    m_runMac    = true;
    auto shPtrMsg = std::make_shared<testWrkrCmdMsg>(CUMAC_TEST_WRKR_CMD_MSG_MAC_INIT, m_wrkrId, shPtrPayload);
    DEBUG_TRACE("MainThread [tid %s][currCtxId 0x%0lx]: Sending message: %s\n", getThreadIdStr().c_str(), getCurrCuCtxId(), TEST_WRKR_CMD_MSG_TO_STR[shPtrMsg->type]);
    m_shPtrCmdQ->send(shPtrMsg);
    // wait for response
    if(waitRsp)
    {
        std::shared_ptr<testWrkrRspMsg> shPtrRsp;
        m_shPtrRspQ->receive(shPtrRsp, CUMAC_TEST_WRKR_RSP_MSG_MAC_INIT, m_wrkrId);
    }
}

void CuMACTestWorkerImpl::macSetup(std::vector<std::string> inFileNamesMac, std::vector<uint8_t> macSlotRunFlag, bool waitRsp)
{
    if(inFileNamesMac.size() == 0)
    {
        m_runMac = false;
        return;
    }

    if(inFileNamesMac.size() != 0)
    {
        // send message
        auto shPtrPayload                = std::make_shared<cuMACTestMacSetupMsgPayload>();
        shPtrPayload->rsp                = true;
        shPtrPayload->inFileNamesMac     = inFileNamesMac;
        shPtrPayload->macSlotRunFlag     = macSlotRunFlag;

        auto shPtrMsg = std::make_shared<testWrkrCmdMsg>(CUMAC_TEST_WRKR_CMD_MSG_MAC_SETUP, m_wrkrId, shPtrPayload);
        DEBUG_TRACE("MainThread [tid %s][currCtxId 0x%0lx]: Sending message: %s\n", getThreadIdStr().c_str(), getCurrCuCtxId(), TEST_WRKR_CMD_MSG_TO_STR[shPtrMsg->type]);

        m_shPtrCmdQ->send(shPtrMsg);
    }
    // wait for response
    if(waitRsp)
    {
        std::shared_ptr<testWrkrRspMsg> shPtrRsp;
        m_shPtrRspQ->receive(shPtrRsp, CUMAC_TEST_WRKR_RSP_MSG_MAC_SETUP, m_wrkrId);
    }
}

void CuMACTestWorkerImpl::macRun(cudaEvent_t startEvent, std::shared_ptr<cuphy::event>& shPtrStopEvent, bool waitRsp, std::vector<cuphy::event>* externalInterSlotEventVec)
{
    auto shPtrPayload                       = std::make_shared<cuMACTestMacRunMsgPayload>();
    shPtrPayload->rsp                       = true;
    shPtrPayload->startEvent                = startEvent;
    shPtrPayload->externalInterSlotEventVec = externalInterSlotEventVec; // external events for sync up on 500 us, not used if m_internalTimer = true

    auto shPtrMsg = std::make_shared<testWrkrCmdMsg>(CUMAC_TEST_WRKR_CMD_MSG_MAC_RUN, m_wrkrId, shPtrPayload);
    DEBUG_TRACE("MainThread [tid %s][currCtxId 0x%0lx]: Sending message: %s\n", getThreadIdStr().c_str(), getCurrCuCtxId(), TEST_WRKR_CMD_MSG_TO_STR[shPtrMsg->type]);

    m_shPtrCmdQ->send(shPtrMsg);

    if(waitRsp)
    {
        std::shared_ptr<testWrkrRspMsg> shPtrRsp;
        m_shPtrRspQ->receive(shPtrRsp, CUMAC_TEST_WRKR_RSP_MSG_MAC_RUN, m_wrkrId);
        cuMACTestMacRunRspMsgPayload& MacRunRspMsgPayload = *std::static_pointer_cast<cuMACTestMacRunRspMsgPayload>(shPtrRsp->payload);
        shPtrStopEvent                                            = MacRunRspMsgPayload.shPtrStopEvent;
    }
}

// End of sending receiving message functions
void CuMACTestWorkerImpl::msgProcess(std::shared_ptr<testWrkrCmdMsg>& shPtrMsg) // reference to shared_ptr optional (used for performance)
{
 using msgHandler_t = void (CuMACTestWorkerImpl::*)(std::shared_ptr<void> & shPtrPayload);

    static constexpr std::array<std::pair<testWrkrCmdMsgType, msgHandler_t>, N_TEST_WRKR_CMD_MSGS> CUMAC_MSG_HANDLER_LUT{
        {{COMMON_TEST_WRKR_CMD_MSG_INIT, &CuMACTestWorkerImpl::initHandler},
         {COMMON_TEST_WRKR_CMD_MSG_EVAL, &CuMACTestWorkerImpl::evalHandler},
         {COMMON_TEST_WRKR_CMD_MSG_PRINT, &CuMACTestWorkerImpl::printHandler},
         {COMMON_TEST_WRKR_CMD_MSG_RESET_EVAL, &CuMACTestWorkerImpl::resetEvalHandler},
         {COMMON_TEST_WRKR_CMD_MSG_SET_WAIT_VAL, &CuMACTestWorkerImpl::setWaitValHandler},
         {COMMON_TEST_WRKR_CMD_MSG_READ_SM_IDS, &CuMACTestWorkerImpl::readSmIdsHandler},
         {COMMON_TEST_WRKR_CMD_MSG_EXIT, &CuMACTestWorkerImpl::exitHandler},
         {CUMAC_TEST_WRKR_CMD_MSG_MAC_INIT, &CuMACTestWorkerImpl::macInitHandler},
         {CUMAC_TEST_WRKR_CMD_MSG_MAC_SETUP, &CuMACTestWorkerImpl::macSetupHandler},
         {CUMAC_TEST_WRKR_CMD_MSG_MAC_RUN, &CuMACTestWorkerImpl::macRunHandler}}};
    if(shPtrMsg->type < N_TEST_WRKR_CMD_MSGS)
    {
        DEBUG_TRACE("%s id %d [tid %s][wrkrCtxId 0x%0lx currCtxId 0x%0lx]: msgProcess msgType %d %s\n", m_name.c_str(), m_wrkrId, getThreadIdStr().c_str(), getCuCtxId(), getCurrCuCtxId(), shPtrMsg->type, TEST_WRKR_CMD_MSG_TO_STR[shPtrMsg->type]);

        // If assert fails, ensure the CUMAC_MSG_HANDLER_LUT table matches the enumerated message types in testWrkrCmdMsgType
        int MSG_IN_LUT = shPtrMsg->type - ((int)shPtrMsg->type < N_COMMN_MSG ? 0 : CUMAC_MSG_OFFSET);
        assert(shPtrMsg->type == CUMAC_MSG_HANDLER_LUT[MSG_IN_LUT].first);
        (this->*(CUMAC_MSG_HANDLER_LUT[MSG_IN_LUT].second))(shPtrMsg->payload);
    }
    else
    {
        DEBUG_TRACE("%s id %d [tid %s][wrkrCtxId 0x%0lx currCtxId 0x%0lx]: msgProcess - Message type not supported\n", m_name.c_str(), m_wrkrId, getThreadIdStr().c_str(), getCuCtxId(), getCurrCuCtxId());
    }
}

// Start of handler functions // TODO clean up init
void CuMACTestWorkerImpl::initHandler(std::shared_ptr<void>& shPtrPayload)
{
DEBUG_TRACE("%s id %d [tid %s][wrkrCtxId 0x%0lx currCtxId 0x%0lx]: initHandler\n", m_name.c_str(), m_wrkrId, getThreadIdStr().c_str(), getCuCtxId(), getCurrCuCtxId());

    commnTestInitMsgPayload const& initMsgPayload = *std::static_pointer_cast<commnTestInitMsgPayload>(shPtrPayload);

    m_thrdId = std::this_thread::get_id();
    setThrdProps();

    m_uqPtrTimeStartEvent = std::make_unique<cuphy::event>();

    // cuMAC timing vectors
    m_timeMacSlotStartEvents.resize(m_nItrsPerStrmMac);
    m_timeMacSlotEndEvents.resize(m_nItrsPerStrmMac);

    m_totMacSlotStartTime.resize(m_nItrsPerStrmMac);
    m_totMacSlotEndTime.resize(m_nItrsPerStrmMac);

    auto& cuStrmPrioMap = initMsgPayload.cuStrmPrioMap;
    m_cuStrmPrioMac   = (cuStrmPrioMap.find("MAC") != cuStrmPrioMap.end()) ? cuStrmPrioMap.at("MAC") : 0;
    printf("CUDA stream prios: MAC %d\n", m_cuStrmPrioMac);
#if 1
    m_shPtrGpuStartSyncFlag = initMsgPayload.shPtrCpuGpuSyncFlag;
    CU_CHECK(cuMemHostGetDevicePointer(&m_ptrGpuStartSyncFlag, m_shPtrGpuStartSyncFlag->addr(), 0));
#else
    static std::mutex syncFlagWriteMutex;
    {
        std::lock_guard<std::mutex> syncFlagWriteMutexLock(syncFlagWriteMutex);

        m_shPtrGpuStartSyncFlag = initMsgPayload.shPtrCpuGpuSyncFlag;
        CU_CHECK(cuMemHostGetDevicePointer(&m_ptrGpuStartSyncFlag, m_shPtrGpuStartSyncFlag->addr(), 0));
        CU_CHECK(cuStreamWriteValue32(reinterpret_cast<CUstream>(m_cuStrms[0].handle()), (CUdeviceptr)m_ptrGpuStartSyncFlag, m_wrkrId, 0));
        m_cuStrms[0].synchronize();
        printf("shPtrGpuStartSyncFlag %d, wrkrId %d\n", (*m_shPtrGpuStartSyncFlag)[0], m_wrkrId);
    }
#endif

    // send response
    if(initMsgPayload.rsp)
    {
        // Send init done response
        auto shPtrRspPayload      = std::make_shared<commnTestRspMsgPayload>();
        shPtrRspPayload->workerId = m_wrkrId;

        auto shPtrRsp = std::make_shared<testWrkrRspMsg>(COMMON_TEST_WRKR_RSP_MSG_INIT, m_wrkrId, shPtrRspPayload);
        m_shPtrRspQ->send(shPtrRsp);
    }
}

void CuMACTestWorkerImpl::macInitHandler(std::shared_ptr<void>& shPtrPayload)
{
    DEBUG_TRACE("%s id %d [tid %s][wrkrCtxId 0x%0lx currCtxId 0x%0lx]: macInitHandler\n", m_name.c_str(), m_wrkrId, getThreadIdStr().c_str(), getCuCtxId(), getCurrCuCtxId());

    // unpack message
    commnTestInitMsgPayload const& initMsgPayload  = *std::static_pointer_cast<commnTestInitMsgPayload>(shPtrPayload);
    std::vector<std::string>       inFileNamesMac  = initMsgPayload.inFileNames;

    if(inFileNamesMac.size() == 0)
    {
        m_runMac = false;
    }
    else
    {
        m_runMac = true;
        if(m_internalTimer)
            m_GPUtime_d = std::move(cuphy::buffer<uint64_t, cuphy::device_alloc>(1));

        // cuMAC runs in a single slot in DDDSUUDDDD patterns
        for(uint32_t strmIdx = 0; strmIdx < m_nStrmsMac; ++strmIdx)
        {
            m_stopEvents.emplace_back(cudaEventDisableTiming);
            m_cuStrmsMac.emplace_back(cudaStreamNonBlocking, m_cuStrmPrioMac);
        }

        // creat cuMAC pipes
        // cuMAC subcontext run on GPU
        //m_macPipes.clear();
        //m_macPipes.resize(m_nMacSlots);
        // cuMAC subcontext reference on CPU
        //m_macPipes_cpuRef.clear();
        if(m_ref_check_mac)
        {
            //m_macPipes_cpuRef.resize(m_nMacSlots);
        }
        // Resize timing events to m_nMacSlots (one per MAC run); initHandler may have used m_nItrsPerStrmMac
        m_timeMacSlotStartEvents.resize(m_nMacSlots);
        m_timeMacSlotEndEvents.resize(m_nMacSlots);
        m_totMacSlotStartTime.resize(m_nMacSlots);
        m_totMacSlotEndTime.resize(m_nMacSlots);
        for(uint32_t strmIdx = 0; strmIdx < m_nStrmsMac; ++strmIdx)
        {
            for(uint32_t itrPerStrmIdx = 0; itrPerStrmIdx < m_nItrsPerStrmMac; itrPerStrmIdx ++)
            {
                uint32_t macSlotIdx = itrPerStrmIdx * m_nStrmsMac + strmIdx;
                if(m_internalTimer && strmIdx == 0)
                    m_macInterSlotStartEventVec.emplace_back(cudaEventDisableTiming);
                
                m_macPipes.emplace_back(std::make_unique<cumac::cumacSubcontext>(inFileNamesMac[macSlotIdx], 1/*using GPU*/, m_halfPrecision/*using half precision*/, 0/*no RI-based layer selection*/, 0/*not Asim*/, m_heteroUeSelCells/*UE selection config*/, m_schAlg/*scheduling algorithm*/, m_modulesCalled /*UE selection, PRG allocation, layer selection, and MCS selection*/, m_cuStrmsMac[strmIdx].handle())); // using cumac::cumacSubcontext class,

                if(m_ref_check_mac)
                {
                    m_macPipes_cpuRef.emplace_back(std::make_unique<cumac::cumacSubcontext>(inFileNamesMac[macSlotIdx], 0/*using CPU*/, m_halfPrecision/*using half precision*/, 0/*no RI-based layer selection*/, 0/*not Asim*/, m_heteroUeSelCells/*UE selection config*/, m_schAlg/*scheduling algorithm*/, m_modulesCalled /*UE selection, PRG allocation, layer selection, and MCS selection*/, m_cuStrmsMac[strmIdx].handle())); // using cumac::cumacSubcontext class,

                    // TODO: GPU and CPU per slot solution might not match if using half precision on GPU
                    if(m_halfPrecision)
                    {
                        printf("Warning: Using half precision, per-slot reference check may fail!\n"); // long-term performance matach
                    }
                }
            }
            if(m_internalTimer && strmIdx == 0)
                // extra event for last slot in loncPattern == 3 and longPattern == 6
                m_macInterSlotStartEventVec.emplace_back(cudaEventDisableTiming);
        }

        for(uint32_t strmIdx = 0; strmIdx < m_nStrmsMac; ++strmIdx)
        {
            CUDA_CHECK(cudaStreamSynchronize(m_cuStrmsMac[strmIdx].handle())); // sync steams for data copy
        }
    }
    if(initMsgPayload.rsp)
    {
        auto shPtrRspPayload      = std::make_shared<commnTestRspMsgPayload>();
        shPtrRspPayload->workerId = m_wrkrId;

        auto shPtrRsp = std::make_shared<testWrkrRspMsg>(CUMAC_TEST_WRKR_RSP_MSG_MAC_INIT, m_wrkrId, shPtrRspPayload);
        m_shPtrRspQ->send(shPtrRsp);
    }
}

void CuMACTestWorkerImpl::exitHandler(std::shared_ptr<void>& shPtrPayload)
{
    DEBUG_TRACE("%s id %d [tid %s][wrkrCtxId 0x%0lx currCtxId 0x%0lx]: exitHandler\n", m_name.c_str(), m_wrkrId, getThreadIdStr().c_str(), getCuCtxId(), getCurrCuCtxId());

    commnTestExitMsgPayload const& exitMsgPayload = *std::static_pointer_cast<commnTestExitMsgPayload>(shPtrPayload);
    if(exitMsgPayload.rsp)
    {
        // Send exit done response
        auto shPtrRspPayload      = std::make_shared<commnTestRspMsgPayload>();
        shPtrRspPayload->workerId = m_wrkrId;

        auto shPtrRsp = std::make_shared<testWrkrRspMsg>(COMMON_TEST_WRKR_RSP_MSG_EXIT, m_wrkrId, shPtrRspPayload);
        m_shPtrRspQ->send(shPtrRsp);
    }
}

void CuMACTestWorkerImpl::macSetupHandler(std::shared_ptr<void>& shPtrPayload)
{
    DEBUG_TRACE("%s id %d [tid %s][wrkrCtxId 0x%0lx currCtxId 0x%0lx]: macSetupHandler\n", m_name.c_str(), m_wrkrId, getThreadIdStr().c_str(), getCuCtxId(), getCurrCuCtxId());

    // unpack message
    cuMACTestMacSetupMsgPayload const& initMsgPayload  = *std::static_pointer_cast<cuMACTestMacSetupMsgPayload>(shPtrPayload);
    std::vector<std::string>       inFileNamesMac      = initMsgPayload.inFileNamesMac;
    m_macSlotRunFlag                                   = initMsgPayload.macSlotRunFlag;

    if(m_runMac)
    {
        uint32_t runIdx = 0;
        const uint32_t nSlotConfig = static_cast<uint32_t>(m_macSlotRunFlag.size());
        for(uint32_t itrPerStrmIdx = 0; itrPerStrmIdx < m_nItrsPerStrmMac; itrPerStrmIdx++)
        {
            for(uint32_t strmIdx = 0; strmIdx < m_nStrmsMac && runIdx < m_nMacSlots; ++strmIdx)
            {
                const uint32_t slotIdx = itrPerStrmIdx * m_nStrmsMac + strmIdx;
                if(slotIdx >= nSlotConfig || m_macSlotRunFlag[slotIdx] == 0)
                    continue;

                if(runIdx >= m_macPipes.size() || runIdx >= inFileNamesMac.size())
                {
                    printf("ERROR: macSetupHandler runIdx (%u) out of bounds (m_macPipes=%zu, inFileNamesMac=%zu)\n", runIdx, m_macPipes.size(), inFileNamesMac.size());
                    break;
                }

                const uint32_t macSlotIdx = runIdx;
                const uint32_t lwSlot = (macSlotIdx - m_lightWeightFlagOffset + m_nMacSlots) % m_nMacSlots;
                m_macPipes[runIdx]->setup(inFileNamesMac[runIdx], m_cumacLightWeightFlag[lwSlot], m_percSmNumThrdBlk[lwSlot], m_cuStrmsMac[strmIdx].handle());

                if(m_cumacLightWeightFlag[lwSlot] <= 2)
                {
                    m_macPipes[runIdx]->run(m_cuStrmsMac[strmIdx].handle());
                }

                if(m_ref_check_mac)
                {
                    m_macPipes_cpuRef[runIdx]->setup(inFileNamesMac[runIdx], m_cumacLightWeightFlag[lwSlot], m_percSmNumThrdBlk[lwSlot], m_cuStrmsMac[strmIdx].handle());
                }
                runIdx++;
            }
        }

        for(uint32_t strmIdx = 0; strmIdx < m_nStrmsMac; ++strmIdx)
        {
            CUDA_CHECK(cudaStreamSynchronize(m_cuStrmsMac[strmIdx].handle())); // sync steams for data copy
        }
    }

    if(initMsgPayload.rsp)
    {
        // Send setup done response
        auto shPtrRspPayload      = std::make_shared<commnTestRspMsgPayload>();
        shPtrRspPayload->workerId = m_wrkrId;

        auto shPtrRsp = std::make_shared<testWrkrRspMsg>(CUMAC_TEST_WRKR_RSP_MSG_MAC_SETUP, m_wrkrId, shPtrRspPayload);
        m_shPtrRspQ->send(shPtrRsp);
    }
}

// Run func and record time
void CuMACTestWorkerImpl::macRunHandler(std::shared_ptr<void>& shPtrPayload)
{
    DEBUG_TRACE("%s id %d [tid %s][wrkrCtxId 0x%0lx currCtxId 0x%0lx]: macRunHandler\n", m_name.c_str(), m_wrkrId, getThreadIdStr().c_str(), getCuCtxId(), getCurrCuCtxId());

    cuMACTestMacRunMsgPayload& macRunMsgPayload = *std::static_pointer_cast<cuMACTestMacRunMsgPayload>(shPtrPayload);

    if(m_runMac)
    {
        if (m_uldlMode == 5 || m_uldlMode == 6)
        {
        #if USE_NVTX
            nvtxRangePush("MAC");
        #endif

            // place holder to run cuMAC functions
            // common starting point, same with the first PDSCH
            CUDA_CHECK(cudaStreamWaitEvent(m_cuStrmsMac[0].handle(), macRunMsgPayload.startEvent, 0));
            CUDA_CHECK(cudaEventRecord(m_uqPtrTimeStartEvent->handle(), m_cuStrmsMac[0].handle()));
            
            if(m_internalTimer)
            {
                get_gpu_time(m_GPUtime_d.addr(), m_cuStrmsMac[0].handle()); // record start of time slot
                CUDA_CHECK(cudaEventRecord(m_macInterSlotStartEventVec[0].handle(), m_cuStrmsMac[0].handle()));
                // note: no initial delay is introduced in cuMAC internal timer
            } 

            const uint32_t nSlotConfig = static_cast<uint32_t>(m_macSlotRunFlag.size());
            uint32_t runIdx = 0;
            for(uint32_t itrPerStrmIdx = 0; itrPerStrmIdx < m_nItrsPerStrmMac; itrPerStrmIdx++)
            {
                for(uint32_t strmIdx = 0; strmIdx < m_nStrmsMac; ++strmIdx)
                {
                    const uint32_t slotIdx = itrPerStrmIdx * m_nStrmsMac + strmIdx;
                    if(slotIdx >= nSlotConfig || runIdx >= m_nMacSlots || m_macSlotRunFlag[slotIdx] == 0)
                        continue;

                    const uint32_t macSlotIdx = runIdx; // contiguous enabled-run index

                    if(strmIdx != 0)
                    {
                        CUDA_CHECK(cudaStreamWaitEvent(m_cuStrmsMac[strmIdx].handle(), macRunMsgPayload.startEvent, 0));
                    }

                    if(m_internalTimer)
                    {
                        CUDA_CHECK(cudaStreamWaitEvent(m_cuStrmsMac[strmIdx].handle(), m_macInterSlotStartEventVec[itrPerStrmIdx].handle(), 0)); // wait for slot boundaries
                    }
                    else if(macRunMsgPayload.externalInterSlotEventVec != nullptr)
                    {
                        if(slotIdx < macRunMsgPayload.externalInterSlotEventVec->size())
                        {
                            CUDA_CHECK(cudaStreamWaitEvent(m_cuStrmsMac[strmIdx].handle(), (*macRunMsgPayload.externalInterSlotEventVec)[slotIdx].handle(), 0));
                        }
                        else
                        {
                            printf("WARNING: externalInterSlotEventVec out of bounds (slotIdx=%u, size=%zu), skipping wait\n", slotIdx, macRunMsgPayload.externalInterSlotEventVec->size());
                        }
                    }

                    CUDA_CHECK(cudaEventRecord(m_timeMacSlotStartEvents[macSlotIdx].handle(), m_cuStrmsMac[strmIdx].handle()));

                    // run multi cell schedulers
                    m_macPipes[macSlotIdx]->run(m_cuStrmsMac[strmIdx].handle());

                    CUDA_CHECK(cudaEventRecord(m_timeMacSlotEndEvents[macSlotIdx].handle(), m_cuStrmsMac[strmIdx].handle()));
                    
                    if(m_ref_check_mac)
                    {
                        m_macPipes_cpuRef[macSlotIdx]->run(m_cuStrmsMac[strmIdx].handle());
                                
                        bool pass = compareCpuGpuAllocSol(m_macPipes[macSlotIdx].get(), m_macPipes_cpuRef[macSlotIdx].get(), m_modulesCalled);

                        if(macSlotIdx == 0) // change this to any slot index from 0~7
                        {
                            saveToH5("gpuResultsTv.h5",
                            m_macPipes[macSlotIdx]->cellGrpUeStatusGpu.get(),
                            m_macPipes[macSlotIdx]->cellGrpPrmsGpu.get(),
                            m_macPipes[macSlotIdx]->schdSolGpu.get());
                            // CPU
                            saveToH5_CPU("cpuResultsTv.h5",
                            m_macPipes_cpuRef[macSlotIdx]->cellGrpUeStatusCpu.get(),
                            m_macPipes_cpuRef[macSlotIdx]->cellGrpPrmsCpu.get(),
                            m_macPipes_cpuRef[macSlotIdx]->schdSolCpu.get());
                        }
                        if (pass) 
                        {
                            printf("cuMAC REFERENCE CHECK at slot %d: PASSED, CPU and GPU scheduler solutions match! \n", macSlotIdx);
                        } 
                        else 
                        {
                            printf("cuMAC REFERENCE CHECK at slot %d: FAILED, CPU and GPU scheduler solutions do not match! \n", macSlotIdx);
                        }
                    }
                    // // record end, currently not in effect
                    // if(strmIdx != 0)
                    // {
                    //     m_stopEvents[i].record(m_cuStrmsMac[i].handle());
                    //     CUDA_CHECK(cudaStreamWaitEvent(m_cuStrmsMac[macSlotIdx].handle(), m_stopEvents[i].handle(), 0));
                    // }

                    runIdx++;
                }

                if(m_internalTimer && (itrPerStrmIdx + 1) < m_macInterSlotStartEventVec.size())
                {
                    // record next slot boundary (end of this iteration)
                    uint64_t gpu_slot_start_time_offset_ns = static_cast<uint64_t>(itrPerStrmIdx + 1) * (time_slot_duration * NS_PER_US);
                    gpu_ns_delay_until(m_GPUtime_d.addr(), gpu_slot_start_time_offset_ns, m_cuStrmsMac[0].handle());
                    CUDA_CHECK(cudaEventRecord(m_macInterSlotStartEventVec[itrPerStrmIdx + 1].handle(), m_cuStrmsMac[0].handle()));
                }
            }

            // send GPU response message
            CUDA_CHECK(cudaEventRecord(m_shPtrStopEvent->handle(), m_cuStrmsMac[0].handle()));
        #if USE_NVTX
            nvtxRangePop();
        #endif
        }
        else
        {
            printf("cuMAC currently can only be run with long pattern ");
        }
    }   

    // send CPU response message
    if(macRunMsgPayload.rsp)
    {
        auto shPtrRspPayload            = std::make_shared<cuMACTestMacRunRspMsgPayload>();
        shPtrRspPayload->workerId       = m_wrkrId;
        shPtrRspPayload->shPtrStopEvent = m_shPtrStopEvent;

        auto shPtrRsp = std::make_shared<testWrkrRspMsg>(CUMAC_TEST_WRKR_RSP_MSG_MAC_RUN, m_wrkrId, shPtrRspPayload);
        m_shPtrRspQ->send(shPtrRsp);
    }
}

void CuMACTestWorkerImpl::setWaitValHandler(std::shared_ptr<void>& shPtrPayload)
{
    DEBUG_TRACE("%s id %d [tid %s][wrkrCtxId 0x%0lx currCtxId 0x%0lx]: setWaitValHandler\n", m_name.c_str(), m_wrkrId, getThreadIdStr().c_str(), getCuCtxId(), getCurrCuCtxId());

    commnTestSetWaitValCmdMsgPayload& setWaitValCmdMsgPayload = *std::static_pointer_cast<commnTestSetWaitValCmdMsgPayload>(shPtrPayload);

    // Wait on device value for all streams
    for(auto& cuStrm : m_cuStrmsMac)
    {

        CU_CHECK(cuStreamWaitValue32(cuStrm.handle(), m_ptrGpuStartSyncFlag, setWaitValCmdMsgPayload.syncFlagVal, CU_STREAM_WAIT_VALUE_GEQ));
    }

    if(setWaitValCmdMsgPayload.rsp)
    {
        // Send run completion response
        auto shPtrRspPayload      = std::make_shared<commnTestRspMsgPayload>();
        shPtrRspPayload->workerId = m_wrkrId;

        auto shPtrRsp = std::make_shared<testWrkrRspMsg>(COMMON_TEST_WRKR_RSP_MSG_SET_WAIT_VAL, m_wrkrId, shPtrRspPayload);
        m_shPtrRspQ->send(shPtrRsp);
    }
}

void CuMACTestWorkerImpl::evalHandler(std::shared_ptr<void>& shPtrPayload)
{
    DEBUG_TRACE("%s id %d [tid %s][wrkrCtxId 0x%0lx currCtxId 0x%0lx]: evalHandler\n", m_name.c_str(), m_wrkrId, getThreadIdStr().c_str(), getCuCtxId(), getCurrCuCtxId());
    commnTestEvalMsgPayload& evalMsgPayload = *std::static_pointer_cast<commnTestEvalMsgPayload>(shPtrPayload);

    if(m_runMac)
    {
        float       elapsedTimeMs = 0.0f;
        for(int macSlotIdx = 0; macSlotIdx < m_nMacSlots; macSlotIdx ++) // calculate MAC time among slots
        {
            // from common start event  -> each cuMAC slot end
            cudaError_t e = cudaEventElapsedTime(&elapsedTimeMs,
                                             m_uqPtrTimeStartEvent->handle(),
                                             m_timeMacSlotEndEvents[macSlotIdx].handle());

            if(cudaSuccess != e) throw cuphy::cuda_exception(e);

            m_totMacSlotEndTime[macSlotIdx] += elapsedTimeMs; // Fixme: m_totMacSlotEndTime[macSlotIdx] = 0 when executing this

            // from common start event  -> each cuMAC slot start
            e             = cudaEventElapsedTime(&elapsedTimeMs,
                                    m_uqPtrTimeStartEvent->handle(),
                                    m_timeMacSlotStartEvents[macSlotIdx].handle());

            if(cudaSuccess != e) throw cuphy::cuda_exception(e);

            m_totMacSlotStartTime[macSlotIdx] += elapsedTimeMs; // Fixme: m_totMacSlotStartTime[macSlotIdx] = 0 when executing this
        }
    }

    // send response
    if(evalMsgPayload.rsp)
    {
        // Send run completion response
        auto shPtrRspPayload      = std::make_shared<commnTestRspMsgPayload>();
        shPtrRspPayload->workerId = m_wrkrId;

        auto shPtrRsp = std::make_shared<testWrkrRspMsg>(COMMON_TEST_WRKR_RSP_MSG_EVAL, m_wrkrId, shPtrRspPayload);
        m_shPtrRspQ->send(shPtrRsp);
    }
}

void CuMACTestWorkerImpl::printHandler(std::shared_ptr<void>& shPtrPayload)
{
    DEBUG_TRACE("%s id %d [tid %s][wrkrCtxId 0x%0lx currCtxId 0x%0lx]: printHandler\n", m_name.c_str(), m_wrkrId, getThreadIdStr().c_str(), getCuCtxId(), getCurrCuCtxId());

    printf("\n%s worker # %d\n", m_name.c_str(), m_wrkrId);

    commnTestPrintMsgPayload& printMsgPayload = *std::static_pointer_cast<commnTestPrintMsgPayload>(shPtrPayload);

    // TODO: what do we need to print for cuMAC results

    // send response
    if(printMsgPayload.rsp)
    {
        // Send run completion response
        auto shPtrRspPayload      = std::make_shared<commnTestRspMsgPayload>();
        shPtrRspPayload->workerId = m_wrkrId;

        auto shPtrRsp = std::make_shared<testWrkrRspMsg>(COMMON_TEST_WRKR_RSP_MSG_PRINT, m_wrkrId, shPtrRspPayload);
        m_shPtrRspQ->send(shPtrRsp);
    }
}

void CuMACTestWorkerImpl::resetEvalHandler(std::shared_ptr<void>& shPtrPayload)
{
    DEBUG_TRACE("%s id %d [tid %s][wrkrCtxId 0x%0lx currCtxId 0x%0lx]: resetEvalHandler\n", m_name.c_str(), m_wrkrId, getThreadIdStr().c_str(), getCuCtxId(), getCurrCuCtxId());

    commnTestResetEvalMsgPayload& resetEvalPayload = *std::static_pointer_cast<commnTestResetEvalMsgPayload>(shPtrPayload);

    // reset timers (one per MAC run)
    for(uint32_t itrIdx = 0; itrIdx < m_nMacSlots; ++itrIdx)
    {
        m_totMacSlotStartTime[itrIdx] = 0;
        m_totMacSlotEndTime[itrIdx]   = 0;
    }

    // send response
    if(resetEvalPayload.rsp)
    {
        // Send run completion response
        auto shPtrRspPayload      = std::make_shared<commnTestRspMsgPayload>();
        shPtrRspPayload->workerId = m_wrkrId;

        auto shPtrRsp = std::make_shared<testWrkrRspMsg>(COMMON_TEST_WRKR_RSP_MSG_RESET_EVAL, m_wrkrId, shPtrRspPayload);
        m_shPtrRspQ->send(shPtrRsp);
    }
}

void CuMACTestWorkerImpl::readSmIdsHandler(std::shared_ptr<void>& shPtrPayload)
{
    DEBUG_TRACE("%s id %d [tid %s][wrkrCtxId 0x%0lx currCtxId 0x%0lx]: readSmIdsHandler\n", m_name.c_str(), m_wrkrId, getThreadIdStr().c_str(), getCuCtxId(), getCurrCuCtxId());

    readSmIds();

    commnTestReadSmIdsCmdMsgPayload& readSmIdsCmdMsgPayload = *std::static_pointer_cast<commnTestReadSmIdsCmdMsgPayload>(shPtrPayload);
    if(readSmIdsCmdMsgPayload.rsp)
    {
        // Send run completion response
        auto shPtrRspPayload            = std::make_shared<commnTestReadSmIdsRspMsgPayload>();
        shPtrRspPayload->workerId       = m_wrkrId;
        shPtrRspPayload->shPtrWaitEvent = m_shPtrRdSmIdWaitEvent;

        auto shPtrRsp = std::make_shared<testWrkrRspMsg>(COMMON_TEST_WRKR_RSP_MSG_READ_SM_IDS, m_wrkrId, shPtrRspPayload);
        m_shPtrRspQ->send(shPtrRsp);
    }
}

std::optional<std::vector<float>> CuMACTestWorkerImpl::getTotMACSlotStartTime()
{
    return m_totMacSlotStartTime;
}
std::optional<std::vector<float>> CuMACTestWorkerImpl::getTotMacSlotEndTime()
{
    return m_totMacSlotEndTime;
}
std::optional<std::vector<cuphy::event>*> CuMACTestWorkerImpl::getMacInterSlotEventVecPtr()
{
    return &m_macInterSlotStartEventVec;
}

#endif // AERIAL_CUMAC_ENABLE