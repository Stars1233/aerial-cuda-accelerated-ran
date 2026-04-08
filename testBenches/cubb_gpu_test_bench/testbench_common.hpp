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

#if !defined(TESTBENCH_COMMON_HPP_INCLUDED_)
#define TESTBENCH_COMMON_HPP_INCLUDED_

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <mutex>
#include <memory>
#include <queue>
#include <condition_variable>
#include <bitset>
#include <map>
#include <unistd.h>     /* For SYS_xxx definitions */
#include <syscall.h>    /* For SYS_xxx definitions */
#include <cuda.h>
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h> // NVTX CUDA profiler annotations
#include <type_traits>
#include <optional>
#include <cuda_profiler_api.h> // cudaProfilerStart / cudaProfilerStop
#include <unordered_map>

#include "cuphy.hpp" // NOTE: dependences of the following required in test workers:
// cuda related:   cuphy::stream, cuphy::event, cuphy::cudaContext, cuphy::cuda_exception
// buffer related: cuphy::buffer, cuphy::device_alloc, cuphy::pinned_alloc

// The base + offset values should match the tag numbers in nvlog_config.yaml and nvlog_fmt.hpp
#define NVLOG_TESTBENCH      (NVLOG_TAG_BASE_TESTBENCH + 0)  // "TESTBENCH
#define NVLOG_TESTBENCH_PHY  (NVLOG_TAG_BASE_TESTBENCH + 1)  // "TESTBENCH.PHY"
#define NVLOG_TESTBENCH_MAC  (NVLOG_TAG_BASE_TESTBENCH + 2)  // "TESTBENCH.MAC"

#define DEBUG_TRACE(...)                           \
    do                                             \
    {                                              \
        if(m_dbgMsgLevel > 0) printf(__VA_ARGS__); \
    } while(0)

constexpr int mu                 = 1; // hardcoded for numerology, can also read from PDSCH TV: m_pdschTxStaticApiDataSets[0].data()->pdschStatPrms.pCellStatPrms->mu;
constexpr int time_slot_duration = 1000 / (1 << mu);
constexpr uint64_t NS_PER_US     = 1000UL;
//----------------------------------------------------------------------------------------------------------
// testWrkrCmdMsgType - Available message types from main test orchestration thread to the worker thread
//----------------------------------------------------------------------------------------------------------
enum testWrkrCmdMsgType
{
    // common command messages
    COMMON_TEST_WRKR_CMD_MSG_INIT         = 0,
    COMMON_TEST_WRKR_CMD_MSG_EVAL         = 1,
    COMMON_TEST_WRKR_CMD_MSG_PRINT        = 2,
    COMMON_TEST_WRKR_CMD_MSG_RESET_EVAL   = 3,
    COMMON_TEST_WRKR_CMD_MSG_SET_WAIT_VAL = 4,
    COMMON_TEST_WRKR_CMD_MSG_READ_SM_IDS  = 5,
    COMMON_TEST_WRKR_CMD_MSG_EXIT         = 6,
    // cuPHY command messages
    CUPHY_TEST_WRKR_CMD_MSG_PUSCH_INIT    = 7,
    CUPHY_TEST_WRKR_CMD_MSG_PDSCH_INIT    = 8,
    CUPHY_TEST_WRKR_CMD_MSG_PUSCH_SETUP   = 9,
    CUPHY_TEST_WRKR_CMD_MSG_PDSCH_SETUP   = 10,
    CUPHY_TEST_WRKR_CMD_MSG_PUSCH_RUN     = 11,
    CUPHY_TEST_WRKR_CMD_MSG_PDSCH_RUN     = 12,
    CUPHY_TEST_WRKR_CMD_MSG_PSCH_RUN      = 13,
    CUPHY_TEST_WRKR_CMD_MSG_PDSCH_CLEAN   = 14,
    CUPHY_TEST_WRKR_CMD_MSG_DEINIT        = 15,
    CUPHY_TEST_WRKR_CMD_MSG_DLBFW_INIT    = 16,
    CUPHY_TEST_WRKR_CMD_MSG_DLBFW_SETUP   = 17,
    CUPHY_TEST_WRKR_CMD_MSG_ULBFW_INIT    = 18,
    CUPHY_TEST_WRKR_CMD_MSG_ULBFW_SETUP   = 19,
    CUPHY_TEST_WRKR_CMD_MSG_SRS_INIT      = 20,
    CUPHY_TEST_WRKR_CMD_MSG_SRS_SETUP     = 21,
    CUPHY_TEST_WRKR_CMD_MSG_PRACH_INIT    = 22,
    CUPHY_TEST_WRKR_CMD_MSG_PRACH_SETUP   = 23,
    CUPHY_TEST_WRKR_CMD_MSG_PUCCH_INIT    = 24,
    CUPHY_TEST_WRKR_CMD_MSG_PUCCH_SETUP   = 25,
    CUPHY_TEST_WRKR_CMD_MSG_PDCCH_INIT    = 26,
    CUPHY_TEST_WRKR_CMD_MSG_PDCCH_SETUP   = 27,
    CUPHY_TEST_WRKR_CMD_MSG_SSB_INIT      = 28,
    CUPHY_TEST_WRKR_CMD_MSG_SSB_SETUP     = 29,
    CUPHY_TEST_WRKR_CMD_MSG_CSIRS_INIT    = 30,
    CUPHY_TEST_WRKR_CMD_MSG_CSIRS_SETUP   = 31,
    // cuMAC command messages
    CUMAC_TEST_WRKR_CMD_MSG_MAC_INIT      = 32,
    CUMAC_TEST_WRKR_CMD_MSG_MAC_SETUP     = 33,
    CUMAC_TEST_WRKR_CMD_MSG_MAC_RUN       = 34,
    // total number of messages
    N_TEST_WRKR_CMD_MSGS                  = 35,
    TEST_WRKR_CMD_MSG_INVALID             = N_TEST_WRKR_CMD_MSGS
};

static constexpr int N_COMMN_MSG = COMMON_TEST_WRKR_CMD_MSG_EXIT + 1; // total number of common messages 
static constexpr int CUMAC_MSG_OFFSET = CUMAC_TEST_WRKR_CMD_MSG_MAC_INIT - COMMON_TEST_WRKR_CMD_MSG_EXIT - 1; // cumac message offset

static constexpr std::array<const char*, N_TEST_WRKR_CMD_MSGS> TEST_WRKR_CMD_MSG_TO_STR{
    "COMMON_TEST_WRKR_CMD_MSG_INIT",
    "COMMON_TEST_WRKR_CMD_MSG_EVAL",
    "COMMON_TEST_WRKR_CMD_MSG_PRINT",
    "COMMON_TEST_WRKR_CMD_MSG_RESET_EVAL",
    "COMMON_TEST_WRKR_CMD_MSG_SET_WAIT_VAL",
    "COMMON_TEST_WRKR_CMD_MSG_READ_SM_IDS",
    "COMMON_TEST_WRKR_CMD_MSG_EXIT",
    "CUPHY_TEST_WRKR_CMD_MSG_PUSCH_INIT",
    "CUPHY_TEST_WRKR_CMD_MSG_PDSCH_INIT",
    "CUPHY_TEST_WRKR_CMD_MSG_PUSCH_SETUP",
    "CUPHY_TEST_WRKR_CMD_MSG_PDSCH_SETUP",
    "CUPHY_TEST_WRKR_CMD_MSG_PUSCH_RUN",
    "CUPHY_TEST_WRKR_CMD_MSG_PDSCH_RUN",
    "CUPHY_TEST_WRKR_CMD_MSG_PSCH_RUN",
    "CUPHY_TEST_WRKR_CMD_MSG_PDSCH_CLEAN",
    "CUPHY_TEST_WRKR_CMD_MSG_DEINIT",
    "CUPHY_TEST_WRKR_CMD_MSG_DLBFW_INIT",
    "CUPHY_TEST_WRKR_CMD_MSG_DLBFW_SETUP",
    "CUPHY_TEST_WRKR_CMD_MSG_ULBFW_INIT",
    "CUPHY_TEST_WRKR_CMD_MSG_ULBFW_SETUP",
    "CUPHY_TEST_WRKR_CMD_MSG_SRS_INIT",
    "CUPHY_TEST_WRKR_CMD_MSG_SRS_SETUP",
    "CUPHY_TEST_WRKR_CMD_MSG_PRACH_INIT",
    "CUPHY_TEST_WRKR_CMD_MSG_PRACH_SETUP",
    "CUPHY_TEST_WRKR_CMD_MSG_PUCCH_INIT",
    "CUPHY_TEST_WRKR_CMD_MSG_PUCCH_SETUP",
    "CUPHY_TEST_WRKR_CMD_MSG_PDCCH_INIT",
    "CUPHY_TEST_WRKR_CMD_MSG_PDCCH_SETUP",
    "CUPHY_TEST_WRKR_CMD_MSG_SSB_INIT",
    "CUPHY_TEST_WRKR_CMD_MSG_SSB_SETUP",
    "CUPHY_TEST_WRKR_CMD_MSG_CSIRS_INIT",
    "CUPHY_TEST_WRKR_CMD_MSG_CSIRS_SETUP",
    "CUMAC_TEST_WRKR_CMD_MSG_MAC_INIT",   
    "CUMAC_TEST_WRKR_CMD_MSG_MAC_SETUP",    
    "CUMAC_TEST_WRKR_CMD_MSG_MAC_RUN"
    };

enum testWrkrRspMsgType
{
    // common response messages
    COMMON_TEST_WRKR_RSP_MSG_INIT         = 0,
    COMMON_TEST_WRKR_RSP_MSG_EVAL         = 1,
    COMMON_TEST_WRKR_RSP_MSG_PRINT        = 2,
    COMMON_TEST_WRKR_RSP_MSG_RESET_EVAL   = 3,
    COMMON_TEST_WRKR_RSP_MSG_SET_WAIT_VAL = 4,
    COMMON_TEST_WRKR_RSP_MSG_READ_SM_IDS  = 5,
    COMMON_TEST_WRKR_RSP_MSG_EXIT         = 6,
    // cuPHY response messages
    CUPHY_TEST_WRKR_RSP_MSG_PUSCH_INIT    = 7,
    CUPHY_TEST_WRKR_RSP_MSG_PDSCH_INIT    = 8,
    CUPHY_TEST_WRKR_RSP_MSG_PUSCH_SETUP   = 9,
    CUPHY_TEST_WRKR_RSP_MSG_PDSCH_SETUP   = 10,
    CUPHY_TEST_WRKR_RSP_MSG_PUSCH_RUN     = 11,
    CUPHY_TEST_WRKR_RSP_MSG_PDSCH_RUN     = 12,
    CUPHY_TEST_WRKR_RSP_MSG_PSCH_RUN      = 13,
    CUPHY_TEST_WRKR_RSP_MSG_PDSCH_CLEAN   = 14,
    CUPHY_TEST_WRKR_RSP_MSG_DEINIT        = 15,
    CUPHY_TEST_WRKR_RSP_MSG_DLBFW_INIT    = 16,
    CUPHY_TEST_WRKR_RSP_MSG_DLBFW_SETUP   = 17,
    CUPHY_TEST_WRKR_RSP_MSG_ULBFW_INIT    = 18,
    CUPHY_TEST_WRKR_RSP_MSG_ULBFW_SETUP   = 19,
    CUPHY_TEST_WRKR_RSP_MSG_SRS_INIT      = 20,
    CUPHY_TEST_WRKR_RSP_MSG_SRS_SETUP     = 21,
    CUPHY_TEST_WRKR_RSP_MSG_PRACH_INIT    = 22,
    CUPHY_TEST_WRKR_RSP_MSG_PRACH_SETUP   = 23,
    CUPHY_TEST_WRKR_RSP_MSG_PUCCH_INIT    = 24,
    CUPHY_TEST_WRKR_RSP_MSG_PUCCH_SETUP   = 25,
    CUPHY_TEST_WRKR_RSP_MSG_PDCCH_INIT    = 26,
    CUPHY_TEST_WRKR_RSP_MSG_PDCCH_SETUP   = 27,
    CUPHY_TEST_WRKR_RSP_MSG_SSB_INIT      = 28,
    CUPHY_TEST_WRKR_RSP_MSG_SSB_SETUP     = 29,
    CUPHY_TEST_WRKR_RSP_MSG_CSIRS_INIT    = 30,
    CUPHY_TEST_WRKR_RSP_MSG_CSIRS_SETUP   = 31,
    // cuMAC response messages
    CUMAC_TEST_WRKR_RSP_MSG_MAC_INIT      = 32,
    CUMAC_TEST_WRKR_RSP_MSG_MAC_SETUP     = 33,
    CUMAC_TEST_WRKR_RSP_MSG_MAC_RUN       = 34,
    // total number of response messages
    N_TEST_WRKR_RSP_MSGS                  = 35,
    TEST_WRKR_RSP_MSG_INVALID             = N_TEST_WRKR_RSP_MSGS
};

static constexpr std::array<const char*, N_TEST_WRKR_RSP_MSGS> TEST_WRKR_RSP_MSG_TO_STR =
    {
        "COMMON_TEST_WRKR_RSP_MSG_INIT",
        "COMMON_TEST_WRKR_RSP_MSG_EVAL",
        "COMMON_TEST_WRKR_RSP_MSG_PRINT",
        "COMMON_TEST_WRKR_RSP_MSG_RESET_EVAL",
        "COMMON_TEST_WRKR_RSP_MSG_SET_WAIT_VAL",
        "COMMON_TEST_WRKR_RSP_MSG_READ_SM_IDS",
        "COMMON_TEST_WRKR_RSP_MSG_EXIT",
        "CUPHY_TEST_WRKR_RSP_MSG_PUSCH_INIT",
        "CUPHY_TEST_WRKR_RSP_MSG_PDSCH_INIT",
        "CUPHY_TEST_WRKR_RSP_MSG_PUSCH_SETUP",
        "CUPHY_TEST_WRKR_RSP_MSG_PDSCH_SETUP",
        "CUPHY_TEST_WRKR_RSP_MSG_PUSCH_RUN",
        "CUPHY_TEST_WRKR_RSP_MSG_PDSCH_RUN",
        "CUPHY_TEST_WRKR_RSP_MSG_PSCH_RUN",
        "CUPHY_TEST_WRKR_RSP_MSG_PDSCH_CLEAN",
        "CUPHY_TEST_WRKR_RSP_MSG_DEINIT",
        "CUPHY_TEST_WRKR_RSP_MSG_DLBFW_INIT",
        "CUPHY_TEST_WRKR_RSP_MSG_DLBFW_SETUP",
        "CUPHY_TEST_WRKR_RSP_MSG_ULBFW_INIT",
        "CUPHY_TEST_WRKR_RSP_MSG_ULBFW_SETUP",
        "CUPHY_TEST_WRKR_RSP_MSG_SRS_INIT",
        "CUPHY_TEST_WRKR_RSP_MSG_SRS_SETUP",
        "CUPHY_TEST_WRKR_RSP_MSG_PRACH_INIT",
        "CUPHY_TEST_WRKR_RSP_MSG_PRACH_SETUP",
        "CUPHY_TEST_WRKR_RSP_MSG_PUCCH_INIT",
        "CUPHY_TEST_WRKR_RSP_MSG_PUCCH_SETUP",
        "CUPHY_TEST_WRKR_RSP_MSG_PDCCH_INIT",
        "CUPHY_TEST_WRKR_RSP_MSG_PDCCH_SETUP",
        "CUPHY_TEST_WRKR_RSP_MSG_SSB_INIT",
        "CUPHY_TEST_WRKR_RSP_MSG_SSB_SETUP",
        "CUPHY_TEST_WRKR_RSP_MSG_CSIRS_INIT",
        "CUPHY_TEST_WRKR_RSP_MSG_CSIRS_SETUP",
        "CUMAC_TEST_WRKR_RSP_MSG_MAC_INIT",   
        "CUMAC_TEST_WRKR_RSP_MSG_MAC_SETUP",    
        "CUMAC_TEST_WRKR_RSP_MSG_MAC_RUN"
    };

//----------------------------------------------------------------------------------------------------------
// test workers common command and response queue
//----------------------------------------------------------------------------------------------------------
// N_MAX_RX: maximum number of threads the message is broadcasted to
static constexpr uint32_t TEST_WRKR_N_MAX_MSG_RX = 64;
template <typename MSG_T, uint32_t N_MAX_MSG_RX = TEST_WRKR_N_MAX_MSG_RX>
struct testWrkrMsg
{
    testWrkrMsg(MSG_T inType, int32_t inRxId, std::shared_ptr<void> inPayload = std::shared_ptr<void>()) :
        type(inType),
        payload(inPayload),
        rxIdBset(std::bitset<N_MAX_MSG_RX>().set(inRxId)){};
    testWrkrMsg(MSG_T inType, std::bitset<N_MAX_MSG_RX> inRxIdBset = 0, std::shared_ptr<void> inPayload = std::shared_ptr<void>()) :
        type(inType),
        payload(inPayload),
        rxIdBset(inRxIdBset){};
    testWrkrMsg()                    = delete;
    testWrkrMsg(testWrkrMsg const&) = delete;
    testWrkrMsg& operator=(testWrkrMsg const&) = delete;
    ~testWrkrMsg()                              = default;

    MSG_T                     type;
    std::bitset<N_MAX_MSG_RX> rxIdBset = 0; // if same message is broadcasted to multiple threads
    std::shared_ptr<void>     payload;
};

inline std::string getThreadIdStr(std::thread::id thrdId = std::this_thread::get_id())
{
    static std::mutex m_ostreamMutex;

    std::lock_guard<std::mutex> ostreamMutexLock(m_ostreamMutex);
    std::ostringstream          oss;
    oss << thrdId;
    return oss.str();
}

// Address of the context
inline uint64_t getCurrCuCtxId()
{
    CUcontext cuCtx;
    cuCtxGetCurrent(&cuCtx);
    CUDA_CHECK(cudaGetLastError());
    return reinterpret_cast<uint64_t>(cuCtx);
}

//----------------------------------------------------------------------------------------------------------
// testMsgQ - A class of test messages to be used for control of the workers
//----------------------------------------------------------------------------------------------------------
template <typename MSG_TYPE, typename MSG>
class testMsgQ {
public:
    testMsgQ(std::string name = "") :
        m_name(name){};
    ~testMsgQ() = default;

    void send(std::shared_ptr<MSG>& shPtrMsg) // reference to shared_ptr optional (used for performance)
    {
        {
            std::lock_guard<std::mutex> mutexLockGuard(m_mutex);
            m_queue.push(shPtrMsg);
        }
        // printf("%s tid %s: send message %d with rxIdBset 0x%llx\n", m_name.c_str(), getThreadIdStr().c_str(), shPtrMsg->type, shPtrMsg->rxIdBset.to_ullong());
        m_cv.notify_all();
    }

    void receive(std::shared_ptr<MSG>& shPtrMsg, int32_t rxId = -1) // reference to shared_ptr needed for functionality
    {
        std::unique_lock<std::mutex> mutexLock(m_mutex);

        // while(m_testMsgQueue.empty()) {m_msgCv.wait(msgMutexLock);}
        m_cv.wait(mutexLock, [this, &shPtrMsg, &rxId] {
            bool msgAvail = false;
            if(!m_queue.empty())
            {
                shPtrMsg = m_queue.front();
                // printf("%s tid %s: received-p1 message %d with rxIdBset 0x%llx rxId %d msgAvail %u bitVal %u\n", m_name.c_str(), getThreadIdStr().c_str(), shPtrMsg->type, shPtrMsg->rxIdBset.to_ullong(), rxId, (rxId >= 0) ? shPtrMsg->rxIdBset[rxId] : true, static_cast<bool>(shPtrMsg->rxIdBset[rxId]));

                if(rxId >= 0)
                {
                    msgAvail = shPtrMsg->rxIdBset[rxId];
                    shPtrMsg->rxIdBset.reset(rxId); // ensure bit clear occurs under mutex protection
                }
                else
                {
                    msgAvail = true;
                }
            }
            return msgAvail;
        });

        // printf("%s tid %s: received-p2 message %d with rxIdBset 0x%llx\n", m_name.c_str(), getThreadIdStr().c_str(), shPtrMsg->type, shPtrMsg->rxIdBset.to_ullong());

        if(rxId >= 0)
        {
            if(shPtrMsg->rxIdBset.none()) m_queue.pop();
        }
        else
        {
            m_queue.pop();
        }
    }

    void receive(std::shared_ptr<MSG>& shPtrMsg, MSG_TYPE msgType, int32_t rxId = -1) // reference to shared_ptr needed for functionality
    {
        std::unique_lock<std::mutex> mutexLock(m_mutex);

        // while(m_testMsgQueue.empty()) {m_msgCv.wait(msgMutexLock);}
        m_cv.wait(mutexLock, [this, &shPtrMsg, &rxId, &msgType] {
            bool msgAvail = false;
            if(!m_queue.empty())
            {
                shPtrMsg = m_queue.front();
                // printf("%s tid %s: received-p1 message %d with rxIdBset 0x%llx rxId %d msgAvail %u bitVal %u\n", m_name.c_str(), getThreadIdStr().c_str(), shPtrMsg->type, shPtrMsg->rxIdBset.to_ullong(), rxId, (rxId >= 0) ? shPtrMsg->rxIdBset[rxId] : true, static_cast<bool>(shPtrMsg->rxIdBset[rxId]));

                if(msgType == shPtrMsg->type)
                {
                    if(rxId >= 0)
                    {
                        msgAvail = shPtrMsg->rxIdBset[rxId];
                        shPtrMsg->rxIdBset.reset(rxId); // ensure bit clear occurs under mutex protection
                    }
                    else
                    {
                        msgAvail = true;
                    }
                }
            }
            return msgAvail;
        });

        // printf("%s tid %s: received-p2 message %d with rxIdBset 0x%llx\n", m_name.c_str(), getThreadIdStr().c_str(), shPtrMsg->type, shPtrMsg->rxIdBset.to_ullong());

        if(rxId >= 0)
        {
            if(shPtrMsg->rxIdBset.none()) m_queue.pop();
        }
        else
        {
            m_queue.pop();
        }
    }

    bool isEmpty() const { return m_queue.empty(); };

private:
    std::string                      m_name;
    std::queue<std::shared_ptr<MSG>> m_queue;
    std::mutex                       m_mutex;
    std::condition_variable          m_cv;
};

//----------------------------------------------------------------------------------------------------------
// test workers common command and response message payload
//----------------------------------------------------------------------------------------------------------

//---------------- command message payload --------------------      

// COMMON_TEST_WRKR_CMD_MSG_INIT payload
struct commnTestInitMsgPayload
{
    bool                                                          rsp;
    std::vector<std::string>                                      inFileNames;
    std::shared_ptr<cuphy::buffer<uint32_t, cuphy::pinned_alloc>> shPtrCpuGpuSyncFlag;
    std::map<std::string, int>                                    cuStrmPrioMap;
};

// COMMON_TEST_WRKR_CMD_MSG_EVAL payload
struct commnTestEvalMsgPayload
{
    bool rsp;
    bool cbErrors;
    bool isPschTxRx;
};

// COMMON_TEST_WRKR_CMD_MSG_PRINT payload
struct commnTestPrintMsgPayload
{
    bool rsp;
    bool cbErrors;
    bool isPschTxRx;
};

// COMMON_TEST_WRKR_CMD_MSG_RESET_EVAL payload
struct commnTestResetEvalMsgPayload
{
    bool rsp;
    bool cbErrors;
};

// COMMON_TEST_WRKR_CMD_MSG_SET_WAIT_VAL payload
struct commnTestSetWaitValCmdMsgPayload
{
    bool     rsp;
    uint32_t workerId;
    uint32_t syncFlagVal;
};

// COMMON_TEST_WRKR_CMD_MSG_READ_SM_IDS payload
struct commnTestReadSmIdsCmdMsgPayload
{
    bool rsp;
};

// COMMON_TEST_WRKR_CMD_MSG_EXIT payload
struct commnTestExitMsgPayload
{
    bool rsp;
};

//---------------- response message payload -------------------- 
// COMMON_TEST_WRKR_RSP_MSG_READ_SM_IDS payload
struct commnTestReadSmIdsRspMsgPayload
{
    uint32_t                      workerId;
    std::shared_ptr<cuphy::event> shPtrWaitEvent;
};

// general response payload
struct commnTestRspMsgPayload
{
    uint32_t workerId;
};

using testWrkrCmdMsg = testWrkrMsg<testWrkrCmdMsgType>;
using testWrkrRspMsg = testWrkrMsg<testWrkrRspMsgType>;
using testWrkrCmdQ   = testMsgQ<testWrkrCmdMsgType, testWrkrCmdMsg>;
using testWrkrRspQ   = testMsgQ<testWrkrRspMsgType, testWrkrRspMsg>;
using sharedPtrTestWrkrCmdQ = std::shared_ptr<testMsgQ<testWrkrCmdMsgType, testWrkrMsg<testWrkrCmdMsgType>>>;
using sharedPtrTestWrkrRspQ = std::shared_ptr<testMsgQ<testWrkrRspMsgType, testWrkrMsg<testWrkrRspMsgType>>>;

//----------------------------------------------------------------------------------------------------------
// testBenchNvlogFmtHelper - A class which enables testbench to use NVLOG
//----------------------------------------------------------------------------------------------------------
class testBenchNvlogFmtHelper
{
public:
    testBenchNvlogFmtHelper(std::string nvlog_name="nvlog.log")
    {
        // Relative path from binary to default nvlog_config.yaml
        relative_path = std::string("../../../").append(NVLOG_DEFAULT_CONFIG_FILE);
        printf("relative_path = %s \n", relative_path.c_str());
        fflush(stdout);
        nv_get_absolute_path(nvlog_yaml_file, relative_path.c_str());
        log_thread_id = nvlog_fmtlog_init(nvlog_yaml_file, nvlog_name.c_str(),NULL);
        nvlog_fmtlog_thread_init();
    }

    ~testBenchNvlogFmtHelper()
    {
        nvlog_fmtlog_close(log_thread_id);
    }

private:
    char nvlog_yaml_file[1024];
    std::string relative_path;
    pthread_t log_thread_id;
};

//----------------------------------------------------------------------------------------------------------
// testWorker - A base worker class that performs common operations of different types of workers
// Public inherited by cuPHYTestWorker and cuMACTestWorker
//----------------------------------------------------------------------------------------------------------
class testWorker{
public: 
    testWorker(){};
    testWorker(std::string const& name, uint32_t workerId, int cpuId, int gpuId, int cpuThrdSchdPolicy, int cpuThrdPrio, uint32_t mpsSubctxSmCount, std::shared_ptr<testWrkrCmdQ>& cmdQ, std::shared_ptr<testWrkrRspQ>& rspQ, int uldlMode, uint32_t debugMessageLevel, bool useGreenContexts);
    testWorker(testWorker const&) 
    { 
        NVLOGE_FMT(NVLOG_TESTBENCH, AERIAL_TESTBENCH_EVENT,  "Error: Copy construction of testWorker not supported");
    };
    testWorker& operator=(testWorker const&) = delete;

    ~testWorker();

    uint32_t* getSmIdsGpu(uint32_t& nSmIds)
    {
        nSmIds = m_nSmIds;
        return m_smIdsGpu.addr();
    };
    uint32_t* getSmIdsCpu(uint32_t& nSmIds)
    {
        nSmIds = m_nSmIds;
        return m_smIdsCpu.addr();
    };

    inline std::string getName()
    {
        return m_name;
    };

    void print(bool cbErrors = false, bool isPschTxRx = false, bool waitRsp = true);
    void eval(bool cbErrors = false, bool isPschTxRx = false, bool waitRsp = true);
    void resetEvalBuffers(bool cbErrors = false, bool waitRsp = true);
    void setWaitVal(uint32_t syncFlagVal, bool waitRsp = true);
    void readSmIds();
    void readSmIds(std::shared_ptr<cuphy::event>& shPtrRdSmIdWaitEvent, bool waitRsp = true);

protected:
    void createCuCtx();
    void createCuGreenCtx(const cuphy::cudaGreenContext& my_green_context);
    inline uint64_t getCuCtxId() { return (!m_useGreenContexts) ? reinterpret_cast<uint64_t>(m_cuCtx.handle()) : m_greenCtxId; } // Only relevant in DEBUG_TRACE
    void setThrdProps();

    // worker identity
    std::string         m_name;
    std::thread::id     m_thrdId;
    int32_t             m_wrkrId;
    int                 m_uldlMode;
    uint32_t            m_dbgMsgLevel;
    // Run parameters (specify how workload divided for each slot pattern)
    uint32_t            m_longPattern;  // 0: DDDSU, >= 1: DDDSUUDDDD patterns
    // GPU side info
    int                 m_gpuId;
    int32_t             m_mpsSubctxSmCount; // Fraction of GPU SMs allocated to the CUDA sub-context
    cuphy::cudaContext  m_cuCtx;            // CUDA sub-context
    //cuphy::cudaGreenContext m_cuGreenCtx;  // CUDA green context (if needed DEBUG_TRACE)
    uint64_t            m_greenCtxId;
    bool                m_useGreenContexts;

    // CPU side info
    int                 m_cpuId;
    std::thread         m_thrd;
    int                 m_schdPolicy;
    int                 m_prio;

    // Communication
    std::shared_ptr<testWrkrCmdQ> m_shPtrCmdQ; // to be renamed with mac
    std::shared_ptr<testWrkrRspQ> m_shPtrRspQ;

    // optional SM Ids
    cuphy::buffer<uint32_t, cuphy::pinned_alloc> m_smIdsCpu;
    cuphy::buffer<uint32_t, cuphy::device_alloc> m_smIdsGpu;
    int32_t                                      m_smCount;
    uint32_t                                     m_nSmIds;

    // common cudaEvent
    std::shared_ptr<cuphy::event>  m_shPtrStopEvent;
    std::shared_ptr<cuphy::event>  m_shPtrRdSmIdWaitEvent;
    std::unique_ptr<cuphy::stream> m_uqPtrWrkrCuStrm;

    // GPU synchronization flag
    std::shared_ptr<cuphy::buffer<uint32_t, cuphy::pinned_alloc>> m_shPtrGpuStartSyncFlag;
    CUdeviceptr                                                   m_ptrGpuStartSyncFlag;
};

//----------------------------------------------------------------------------------------------------------
// delays for DDDSUUDDDD (u5) all delays are with respected to start of slot 0
//----------------------------------------------------------------------------------------------------------
constexpr uint32_t srsStartDelayUsU5_         = 501;
constexpr uint32_t srs2StartDelayUsU5_        = 501;
constexpr uint32_t puschStartDelayUsU5_       = 516; // delay of PUSCH1 when BFW and SRS are disable; delay of PUSCH2 will be (puschStartDelayUsU5_ + time_slot_duration) delay if BFW and SRS are enabled
constexpr uint32_t dlbfwStartSlot0DelayUsU5_  = 375; // delay of DLBFW in slot 0 if BFW and SRS are enabled; in other slots, DLBFW starts right after PDSCH
//----------------------------------------------------------------------------------------------------------
// delays for 64TR new pattern (u6) all delays are with respected to start of slot 10
//----------------------------------------------------------------------------------------------------------
// delays for UL channels
// we use 1 (implied) or 2 to specify the UL workload per channel
constexpr uint32_t srsStartDelayUsU6_    = 150 + time_slot_duration;
constexpr uint32_t srs2StartDelayUsU6_   = 150 + time_slot_duration; // currently SRS2 is not used
constexpr uint32_t ulbfwStartDelayUsU6_  = 200 + time_slot_duration;
constexpr uint32_t ulbfw2StartDelayUsU6_ = 200 + time_slot_duration*2;
constexpr uint32_t puschStartDelayUsU6_  = 1045 + time_slot_duration*4;
constexpr uint32_t pusch2StartDelayUsU6_ = 2545 + time_slot_duration*5;
constexpr uint32_t pucch2StartDelayUsU6_ = 1400 + time_slot_duration*5;
// PUSCH2 starts after fixed delay and/or PUSCH1 completed
// PUCCH starts the same as PUSCH1
// PRACH starts the same as PUCCH2

// delays for DL channels
// we use vector to specify the DL workload per channel since the actual workload depending on configs
constexpr uint32_t dlbfwStartDelayUsU6_ = 17;
constexpr uint32_t pdschStartDelayAfterBfwUsU6_ = 58;
constexpr uint32_t pdschStartDelayNoBfwUsU6_ = dlbfwStartDelayUsU6_ + 150 + pdschStartDelayAfterBfwUsU6_; // 150 is DLBFW budget
// |        |  DLBFW  |        |  PDSCH  |        |
// |<- 17 ->|<- 150 ->|<- 58 ->|<- 200 ->|<- 75 ->| 
// SSB + PDCCH + CSI-RS starts to run at the same time of PDSCH
// If PDSCH runs in a slot, there is always a DLBFW running before it

/**
 * @brief Selects the UL timeline anchor channel used by start-delay scheduling.
 *
 * This mode is parsed from start_delay.UL_ANCHOR and consumed by the test bench
 * UL scheduler to interpret PUSCH/PRACH/PUCCH start delays relative to a chosen
 * anchor event.
 *
 * @note Underlying type is uint8_t for compact config storage and stable values.
 */
enum class ul_anchor_mode_t : uint8_t
{
    PUSCH = 0, ///< Use PUSCH as UL anchor (default/legacy-aligned timeline behavior).
    PRACH = 1, ///< Use PRACH as UL anchor for relative UL channel start delays.
    PUCCH = 2  ///< Use PUCCH as UL anchor for relative UL channel start delays.
};

// Runtime-configurable start delays (YAML overrides optional).
// Defaults match the constexpr values above.
struct start_delay_cfg_us
{
    uint32_t srs_u5        = srsStartDelayUsU5_;
    uint32_t srs2_u5       = srs2StartDelayUsU5_;
    int32_t  pusch_u5      = -1;
    int32_t  pusch2_u5     = -1;
    uint32_t pucch_u5      = puschStartDelayUsU5_;
    uint32_t prach_u5      = 0;
    ul_anchor_mode_t ul_anchor_mode = ul_anchor_mode_t::PUSCH; //!< UL anchor selection mode (PUSCH/PRACH/PUCCH) for start-delay scheduling.
    bool     ul_anchor_from_yaml = false; //!< True when UL anchor mode is explicitly provided by YAML start_delay.UL_ANCHOR.
    bool     prach_delay_from_yaml = false;  // if true, PRACH delay is relative to PUSCH1 start; else relative to PUSCH1 end for backward compatibility
    uint32_t dlbfw_slot0_u5 = dlbfwStartSlot0DelayUsU5_;
    uint32_t ssb_u5        = 0;

    uint32_t srs_u6        = srsStartDelayUsU6_;
    uint32_t srs2_u6       = srs2StartDelayUsU6_;
    uint32_t ulbfw_u6      = ulbfwStartDelayUsU6_;
    uint32_t ulbfw2_u6     = ulbfw2StartDelayUsU6_;
    int32_t  pusch_u6      = -1;
    int32_t  pusch2_u6     = -1;
    uint32_t pucch_u6      = puschStartDelayUsU6_;
    int32_t  pucch2_u6     = -1;
    uint32_t prach_u6      = pucch2StartDelayUsU6_;

    uint32_t dlbfw_u6      = dlbfwStartDelayUsU6_;
    int32_t  pdcch_u6      = -1;
    int32_t  pdcch_csirs_u6 = -1;
    uint32_t ssb_u6        = pdschStartDelayNoBfwUsU6_;
    uint32_t pdsch_after_bfw_u6 = pdschStartDelayAfterBfwUsU6_;
    uint32_t pdsch_no_bfw_u6    = pdschStartDelayNoBfwUsU6_;
};

extern start_delay_cfg_us g_start_delay_cfg_us;

// Runtime-configurable per-channel test-vector parameter overrides (optional).
// These override parameters loaded from the H5 TV files, and should be applied
// before pipeline creation. If not provided, the H5 defaults are used.
struct tv_override_cfg
{
    bool enable = false;

    struct pusch_cfg
    {
        bool has_polar_list_length = false;
        uint8_t polar_list_length = 0;

        bool has_enable_cfo_correction = false;
        uint8_t enable_cfo_correction = 0;
        bool has_enable_weighted_average_cfo = false;
        uint8_t enable_weighted_average_cfo = 0;
        bool has_enable_to_estimation = false;
        uint8_t enable_to_estimation = 0;
        bool has_tdi_mode = false;
        uint8_t tdi_mode = 0;
        bool has_enable_dft_sofdm = false;
        uint8_t enable_dft_sofdm = 0;
        bool has_enable_rssi_measurement = false;
        uint8_t enable_rssi_measurement = 0;
        bool has_enable_sinr_measurement = false;
        uint8_t enable_sinr_measurement = 0;
        bool has_enable_static_dynamic_beamforming = false;
        uint8_t enable_static_dynamic_beamforming = 0;
        bool has_enable_early_harq = false;
        uint8_t enable_early_harq = 0;

        bool has_ldpc_early_termination = false;
        uint8_t ldpc_early_termination = 0;
        bool has_ldpc_algorithm_index = false;
        uint16_t ldpc_algorithm_index = 0;
        bool has_ldpc_flags = false;
        uint32_t ldpc_flags = 0;
        bool has_ldpc_use_half = false;
        uint8_t ldpc_use_half = 0;
        bool has_ldpc_max_num_iterations = false;
        uint8_t ldpc_max_num_iterations = 0;
        bool has_ldpc_max_num_iterations_algorithm_index = false;
        uint8_t ldpc_max_num_iterations_algorithm_index = 0;

        bool has_dmrs_channel_estimation_algorithm_index = false;
        uint8_t dmrs_channel_estimation_algorithm_index = 0;
        bool has_enable_per_prg_channel_estimation = false;
        uint8_t enable_per_prg_channel_estimation = 0;
        bool has_eq_coefficient_algorithm_index = false;
        uint8_t eq_coefficient_algorithm_index = 0;
    };

    struct pucch_cfg
    {
        bool has_polar_list_length = false;
        uint8_t polar_list_length = 0;
    };

    struct srs_cfg
    {
        bool has_chest_alg_index = false;
        uint8_t chest_alg_index = 0;
    };

    pusch_cfg pusch;
    pucch_cfg pucch;
    srs_cfg srs;
};

extern tv_override_cfg g_tv_override_cfg;

#endif // !defined(TESTBENCH_COMMON_HPP_INCLUDED_)
