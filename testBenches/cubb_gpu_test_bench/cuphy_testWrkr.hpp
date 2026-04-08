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

#if !defined(CUPHY_TESTWRKR_HPP_INCLUDED_)
#define CUPHY_TESTWRKR_HPP_INCLUDED_

// test bench common header
#include "testbench_common.hpp"

// cuPHY header
#include "util.hpp"
#include "cuphy_channels.hpp"
#include "datasets.hpp"
#include "pdsch_tx.hpp"
#include "pdcch_tx.hpp"
#include "pusch_rx_test.hpp"
#include "pucch_rx.hpp"

// #define ENABLE_F01_STREAM_PRIO

// 0 means not scheduled for a cuPHYTestWorker
// currently same for all patterns
struct numPhyCells_t
{
    uint32_t nPUSCHCells      = 0;
    uint32_t nUlbfwCells      = 0;
    uint32_t nPUCCHCells      = 0;
    uint32_t nPRACHCells      = 0;
    uint32_t nSRSCells        = 0;
    uint32_t nPDSCHCells      = 0;
    uint32_t nDlbfwCells      = 0;
    uint32_t nPDCCHCells      = 0;
    uint32_t nSSBCells        = 0;
    uint32_t nCSIRSCells      = 0;
};

//----------------------------------------------------------------------------------------------------------
// cuPHY test worker message payloads for testWrkrMsg

struct cuPHYTestPuschRxInitMsgPayload : commnTestInitMsgPayload
{
    bool         enableLdpcThroughputMode;
    maxPUSCHPrms puschPrms;
};
struct cuPHYTestPdschTxInitMsgPayload : commnTestInitMsgPayload
{
    maxPDSCHPrms pdschPrms;
};

// CUPHY_TEST_WRKR_CMD_MSG_DEINIT payload
struct cuPHYTestDeinitMsgPayload
{
    bool rsp;
};

// CUPHY_TEST_WRKR_CMD_MSG_PUSCH_SETUP payload
struct cuPHYTestPuschRxSetupMsgPayload
{
    std::vector<std::string> inFileNamesPuschRx;
    bool                     rsp;
};

// CUPHY_TEST_WRKR_CMD_MSG_PDSCH_SETUP payload
struct cuPHYTestPdschTxSetupMsgPayload
{
    std::vector<std::string> inFileNamesPdschTx;
    bool                     rsp;
};

// CUPHY_TEST_WRKR_CMD_MSG_PUSCH_RUN payload
struct cuPHYTestPuschRxRunMsgPayload
{
    bool        rsp;
    cudaEvent_t startEvent;
    cudaEvent_t prachStartEvent;
    cudaEvent_t pucchStartEvent;
};

// CUPHY_TEST_WRKR_CMD_MSG_PDSCH_RUN payload
struct cuPHYTestPdschTxRunMsgPayload
{
    bool                       rsp;
    cudaEvent_t                startEvent;
    std::vector<cuphy::event>* pdcchStopEventVec;
    std::vector<cuphy::event>* pdschInterSlotEventVec;
};

// CUPHY_TEST_WRKR_CMD_MSG_BFC_RUN payload
struct cuPHYTestBFCRunMsgPayload
{
    bool        rsp;
    cudaEvent_t startEvent;
};

// CUPHY_TEST_WRKR_CMD_MSG_PDSCH_RUN payload
struct cuPHYTestPschTxRxRunMsgPayload
{
    bool        rsp;
    cudaEvent_t startEvent;
};

// CUPHY_TEST_WRKR_CMD_MSG_PDSCH_CLEAN payload
struct cuPHYTestPdschTxCleanMsgPayload
{
    bool rsp;
};

// CUPHY_TEST_WRKR_RSP_MSG_PUSCH_RUN payload
struct cuPHYTestPuschRxRunRspMsgPayload
{
    uint32_t                      workerId;
    std::shared_ptr<cuphy::event> shPtrStopEvent;
};

// CUPHY_TEST_WRKR_RSP_MSG_PDSCH_RUN payload
struct cuPHYTestPdschTxRunRspMsgPayload
{
    uint32_t                      workerId;
    std::shared_ptr<cuphy::event> shPtrStopEvent;
};

// CUPHY_TEST_WRKR_RSP_MSG_PSCH_RUN payload
struct cuPHYTestPschRunRspMsgPayload
{
    uint32_t                      workerId;
    std::shared_ptr<cuphy::event> shPtrStopEvent;
};

// CUPHY_TEST_WRKR_RSP_MSG_SRS_RUN payload
struct cuPHYTestSRSRunRspMsgPayload
{
    uint32_t                      workerId;
    std::shared_ptr<cuphy::event> shPtrStopEvent;
};

//----------------------------------------------------------------------------------------------------------
// cuPHYTestWorker - A worker class which uses resources (CPU thread, CUDA sub-context, HW CPU/GPU Ids) to
// setup/run one or more UL/DL pipelines. Worker accepts commands via message based interface.
//----------------------------------------------------------------------------------------------------------
class cuPHYTestWorker : public testWorker{
public:
    cuPHYTestWorker(std::string const& name, uint32_t workerId, int cpuId, int gpuId, int cpuThrdSchdPolicy, int cpuThrdPrio, uint32_t mpsSubctxSmCount, std::shared_ptr<testWrkrCmdQ>& cmdQ, std::shared_ptr<testWrkrRspQ>& rspQ, int uldlMode, uint32_t debugMessageLevel, bool useGreenContexts, const cuphy::cudaGreenContext& greenCtx);

    cuPHYTestWorker(cuPHYTestWorker const&) 
    { 
        NVLOGE_FMT(NVLOG_TESTBENCH_PHY, AERIAL_TESTBENCH_EVENT,  "Error: Copy construction of cuPHYTestWorker not supported");
    };
    cuPHYTestWorker& operator=(cuPHYTestWorker const&) = delete;
    ~cuPHYTestWorker();

    void init(uint32_t nStrms, uint32_t nSlotsPerPattern, uint32_t nItrsPerStrm, uint32_t nTimingItrs, std::map<std::string, int>& cuStrmPrioMap, std::shared_ptr<cuphy::buffer<uint32_t, cuphy::pinned_alloc>>& shPtrCpuGpuSyncFlag, numPhyCells_t numPhyCells, bool srsSplit = false, bool waitRsp = true, uint32_t longPattern = 0, bool srsCtx = false); // nSlotsPattern is the number of slots (may or may not have workload) per pattern, nItrsPerStrm is the number of iterations with workloads per stream
    // NOTE on BFW workloads
    // currently we do not differentiate DLBFW and ULBFW workload, so we use the same proc_mode for both DLBFW and ULBFW
    // For DLBFW, it runs in the same subcontext of PDSCH, pdsch_proc_mode is set during pdschTxInit()
    // For ULBFW, it runs in the same subcontext of PUSCH, need to explicitly pass pdsch_proc_mode to the worker
    void dlbfwInit(std::vector<std::string> inFileNamesDLBFW, uint32_t dlbfw_nItrsPerStrm, bool ref_check_bfw, bool waitRsp = true);
    void dlbfwSetup(std::vector<std::string> inFileNamesDLBFW, bool waitRsp = true);
    void ulbfwInit(std::vector<std::string> inFileNamesULBFW, bool ref_check_bfw, cuphyPdschProcMode_t pdsch_proc_mode, bool waitRsp = true);
    void ulbfwSetup(std::vector<std::string> inFileNamesULBFW, bool waitRsp = true);
    void prachInit(std::vector<std::string> inFileNamesPRACH, uint64_t proc_mode, bool ref_check_prach, bool group_cells, uint32_t cells_per_stream, bool waitRsp = true);
    void prachSetup(std::vector<std::string> inFileNamesPRACH, bool waitRsp = true);
    // for DL workloads, channelSlotRunFlag is an array of 0 or 1, with length equal to the number of slots per pattern.
    // pdcchSlotRunFlag[slotIdx] = 0 means not running this channel in slot slotIdx
    // pdcchSlotRunFlag[slotIdx] = 1 means running this channel in slot slotIdx
    void pdcchTxSetup(std::vector<std::string> inFileNamesPDCCH, uint8_t* pdcchSlotRunFlag, bool waitRsp = true);
    void srsSetup(std::vector<std::string> inFileNamesBFC, bool waitRsp = true);
    void csirsInit(std::vector<std::string> inFileNamesCSIRS, uint32_t csirs_nItrsPerStrm, bool ref_check_csirs, bool group_cells, uint64_t csirs_proc_mode, bool waitRsp = true);
    void csirsSetup(std::vector<std::string> inFileNamesCSIRS, uint8_t* csirsSlotRunFlag, bool waitRsp = true);
    void ssbInit(std::vector<std::string> inFileNamesSSB, uint32_t pbch_nItrsPerStrm, bool ref_check_ssb, bool group_cells, uint64_t ssb_proc_mode, bool waitRsp = true);
    void ssbSetup(std::vector<std::string> inFileNamesSSB, uint8_t* pbchSlotRunFlag, bool waitRsp = true);
    void srsInit(std::vector<std::string> inFileNamesSRS, bool ref_check_srs, uint64_t srs_proc_mode, bool splitSRS = false, bool waitRsp = true);
    void pdcchTxInit(std::vector<std::string> inFileNamesPDCCH, uint32_t m_pdcch_nItrsPerStrm, bool group_cells, uint32_t cells_per_stream, bool ref_check_pdcch, uint64_t pdcch_proc_mode, bool waitRsp = true);
    void pucchRxInit(std::vector<std::string> inFileNamesPUCCH, bool ref_check_pucch, bool groupCells, uint64_t pucch_proc_mode, bool waitRsp = true);
    void pucchRxSetup(std::vector<std::string> inFileNamesPUCCH, bool waitRsp = true);
    void puschRxInit(std::vector<std::string> inFileNamesPuschRx, uint32_t fp16Mode, int puschRxDescramblingOn, bool printCbErrors, uint64_t pusch_proc_mode, bool enableLdpcThroughputMode, bool groupCells, maxPUSCHPrms puschPrms, uint32_t ldpcLaunchMode, uint8_t* puschSubslotProcFlag, bool waitRsp = true);
    void pdschTxInit(std::vector<std::string> inFileNamesPdschTx, uint32_t pdsch_nItrsPerStrm, bool ref_check_pdsch, bool identical_ldpc_configs, cuphyPdschProcMode_t pdsch_proc_mode, bool group_cells, uint32_t cells_per_stream, maxPDSCHPrms pdschPrms, bool waitRsp = true);

    void deinit(bool waitRsp = true);
    void puschRxSetup(std::vector<std::string> inFileNamesPuschRx, bool waitRsp = true);
    void pdschTxSetup(std::vector<std::string> inFileNamesPdschTx, uint8_t* pdschSlotRunFlag, bool waitRsp = true);
    void puschRxRun(cudaEvent_t startEvent, std::shared_ptr<cuphy::event>& shPtrStopEvent, cudaEvent_t prachStartEvent = nullptr, cudaEvent_t pucchStartEvent = nullptr, bool waitRsp = true); // pucchStartEvent is the start time of PUCCH1, prachStartEvent is the start time of PRACH and PUCCH2
    void pdschTxRun(cudaEvent_t startEvent, std::shared_ptr<cuphy::event>& shPtrStopEvent, bool waitRsp = true, std::vector<cuphy::event>* pdcchStopEventVec = nullptr, std::vector<cuphy::event>* pdschInterSlotEventVec = nullptr);
    void pschTxRxRun(cudaEvent_t startEvent, std::shared_ptr<cuphy::event>& shPtrStopEvent, bool waitRsp = true);
    void pdschTxClean(bool waitRsp = true);

    void runSSB(std::shared_ptr<void>& shPtrPayload);
    void runPRACH(const cudaEvent_t& startEvent);
    void runPUCCH(const cudaEvent_t& startEvent); // run PUCCH if not u5 or u6
    void runPUCCH_U5_U6(const cudaEvent_t& startEvent, const cudaEvent_t& startEvent2); // run PUCCH if u5 or u6; startEvent is for PUCCH and startEvent2 is for PUCCH2
    void runPDSCH_U5_3_6(std::shared_ptr<void>& shPtrPayload);
    void runPDSCH_U3_U5_1_2_4_5(std::shared_ptr<void>& shPtrPayload);
    void runPDSCH_U6(std::shared_ptr<void>& shPtrPayload); // for new mMIMO pattern
    void runSRS1(const cudaEvent_t& startEvent);
    void runPUSCH(const cudaEvent_t& startEvent);
    void runPUSCH_U5_U6(const cudaEvent_t& startEvent);
    void runSRS2();
    void runPDCCH(std::shared_ptr<void>& shPtrPayload);
    void runPDCCHItr(const cudaEvent_t& pdcchSlotStartEvent, uint32_t itrIdx);

    void                       getPuschRxRunRsp(std::shared_ptr<testWrkrRspMsg>& shPtrRsp);
    void                       getPdschTxRunRsp(std::shared_ptr<testWrkrRspMsg>& shPtrRsp);
    void                       getPschTxRxRunRsp(std::shared_ptr<testWrkrRspMsg>& shPtrRsp);
    std::vector<float>         getDlbfwIterStartTimes();
    std::vector<float>         getDlbfwIterTimes();
    std::vector<float>         getPdcchStartTimes();
    std::vector<float>         getPdcchIterTimes();
    std::vector<float>         getCSIRSStartTimes();
    std::vector<float>         getCSIRSIterTimes();
    std::vector<float>         getPdschIterTimes();
    std::vector<float>         getPdschSlotStartTimes();
    std::vector<float>         getTotSSBStartTime();
    std::vector<float>         getTotSSBRunTime();
    float                      getTotSRSStartTime();
    float                      getTotSRSRunTime();
    float                      getTotSRS2StartTime();
    float                      getTotSRS2RunTime();
    float                      getTotPrachStartTime();
    float                      getTotPrachRunTime();
    float                      getTotPuschStartTime();
    float                      getTotPuschSubslotProcRunTime();
    float                      getTotPuschRunTime();
    float                      getTotPusch2StartTime();
    float                      getTotPusch2SubslotProcRunTime();
    float                      getTotPusch2RunTime();
    float                      getTotPucchStartTime();
    float                      getTotPucchRunTime();
    float                      getTotPucch2StartTime();
    float                      getTotPucch2RunTime();
    float                      getTotUlbfwStartTime();
    float                      getTotUlbfwRunTime();
    float                      getTotUlbfw2StartTime();
    float                      getTotUlbfw2RunTime();
    std::vector<cuphy::event>* getPdschInterSlotEventVecPtr();
    std::vector<cuphy::event>* getSlotBoundaryEventVecPtr();
    std::vector<cuphy::event>* getpdcchCsirsInterSlotEndEventVec();
    cudaEvent_t                getPuschStartEvent();
    cudaEvent_t                getPusch2StartEvent();
    cudaEvent_t                getPucch2DelayStopEvent();
    /**
     * @brief Get the PRACH start CUDA event.
     * @return cudaEvent_t handle representing the PRACH start event.
     */
    cudaEvent_t                getPrachStartEvent();
    /**
     * @brief Get the PUCCH start CUDA event.
     * @return cudaEvent_t handle representing the PUCCH start event.
     */
    cudaEvent_t                getPucchStartEvent();
    cudaEvent_t                getPusch1EndEvent();
private:
    void msgProcess(std::shared_ptr<testWrkrCmdMsg>& shPtrMsg); // reference to shared_ptr optional (used for performance)
    void initHandler(std::shared_ptr<void>& shPtrPayload);
    void puschRxInitHandler(std::shared_ptr<void>& shPtrPayload);
    void pdschTxInitHandler(std::shared_ptr<void>& shPtrPayload);
    void pdcchTxInitHandler(std::shared_ptr<void>& shPtrPayload);
    void csirsInitHandler(std::shared_ptr<void>& shPtrPayload);
    void ssbInitHandler(std::shared_ptr<void>& shPtrPayload);
    void dlbfwInitHandler(std::shared_ptr<void>& shPtrPayload);
    void dlbfwSetupHandler(std::shared_ptr<void>& shPtrPayload);
    void ulbfwInitHandler(std::shared_ptr<void>& shPtrPayload);
    void ulbfwSetupHandler(std::shared_ptr<void>& shPtrPayload);
    void pdcchTxSetupHandler(std::shared_ptr<void>& shPtrPayload);
    void prachInitHandler(std::shared_ptr<void>& shPtrPayload);
    void srsInitHandler(std::shared_ptr<void>& shPtrPayload);
    void pucchRxInitHandler(std::shared_ptr<void>& shPtrPayload);
    void deinitHandler(std::shared_ptr<void>& shPtrPayload);
    void exitHandler(std::shared_ptr<void>& shPtrPayload);
    void puschRxSetupHandler(std::shared_ptr<void>& shPtrPayload);
    void pdschTxSetupHandler(std::shared_ptr<void>& shPtrPayload);
    void srsSetupHandler(std::shared_ptr<void>& shPtrPayload);
    void pucchRxSetupHandler(std::shared_ptr<void>& shPtrPayload);
    void ssbSetupHandler(std::shared_ptr<void>& shPtrPayload);
    void csirsSetupHandler(std::shared_ptr<void>& shPtrPayload);
    void prachSetupHandler(std::shared_ptr<void>& shPtrPayload);
    void puschRxRunHandler(std::shared_ptr<void>& shPtrPayload);
    void pdschTxRunHandler(std::shared_ptr<void>& shPtrPayload);
    void pdcchTxRunHandler(std::shared_ptr<void>& shPtrPayload);
    void pschTxRxRunHandler(std::shared_ptr<void>& shPtrPayload);
    void pschTxRxRunHandlerNoStrmPrio(std::shared_ptr<void>& shPtrPayload);
    void pschTxRxRunHandlerStrmPrio(std::shared_ptr<void>& shPtrPayload);
    void evalHandler(std::shared_ptr<void>& shPtrPayload);
    void printHandler(std::shared_ptr<void>& shPtrPayload);
    void resetEvalHandler(std::shared_ptr<void>& shPtrPayload);
    void pdschTxCleanHandler(std::shared_ptr<void>& shPtrPayload);
    void emptyHandler(std::shared_ptr<void>& shPtrPayload);
    void pschRdSmIdHandler(std::shared_ptr<void>& shPtrPayload);
    void setWaitValHandler(std::shared_ptr<void>& shPtrPayload);

    void run(const cuphy::cudaGreenContext& my_green_context);
    void createCuOrGreenCtx(const cuphy::cudaGreenContext& my_green_context);

    // Run parameters (specify how workload divided for each slot pattern)
    uint32_t m_nPUSCHCells;  // number of PUSCH cells
    uint32_t m_nUlbfwCells;  // number of Ulbfw cells
    uint32_t m_nPUCCHCells;  // number of PUCCH cells
    uint32_t m_nPRACHCells;  // number of PRACH cells
    uint32_t m_nSRSCells;    // number of SRS cells
    uint32_t m_nPDSCHCells;  // number of PDSCH cells
    uint32_t m_nDlbfwCells;  // number of Dlbfw cells
    uint32_t m_nPDCCHCells;  // number of PDCCH cells
    uint32_t m_nSSBCells;    // number of SSB cells
    uint32_t m_nCSIRSCells;  // number of CSIRS cells

    uint32_t m_nStrms;       // number of parallel workers
    uint32_t m_nSlotsPerPattern; // number of slots per pattern
    uint32_t m_nItrsPerStrm; // number of iterations per parallel worker. Note: m_nCells = m_nStrms * m_nItrsPerStrm
    uint32_t m_nCellsPerStrm_pdsch; //
    uint32_t m_nCellsPerStrm_pdcch; //
    uint32_t m_nCellsPerStrm_pucch; // number of iterations per parallel worker. Note: m_nPUCCHCells = m_nStrms_pucch * m_nCellsPerStrm_pucch
    uint32_t m_nStrms_pdsch;        // number of parallel workers for pdsch for uldl==4
    uint32_t m_nStrms_pdcch;        // number of parallel workers for pdcch for uldl==4
    uint32_t m_nStrms_pucch;        // number of parallel workers for pucch for uldl==4 //ToDO?? is it better to change the name to m_nCellGroups_pucch?
    uint32_t m_nCellsPerStrm_prach; //
    uint32_t m_nStrms_prach;        // number of parallel workers for prach

    // Stream PRIOs
    uint32_t m_cuStrmPrioPusch;
    uint32_t m_cuStrmPrioPusch2;
    uint32_t m_cuStrmPrioPucch;
    uint32_t m_cuStrmPrioPucch2;
    uint32_t m_cuStrmPrioUlbfw;
    uint32_t m_cuStrmPrioPrach;
    uint32_t m_cuStrmPrioSrs;
    uint32_t m_cuStrmPrioPdsch;
    uint32_t m_cuStrmPrioDlbfw;
    uint32_t m_cuStrmPrioPdcch;
    uint32_t m_cuStrmPrioCsirs;
    uint32_t m_cuStrmPrioSsb;
    
    // streams. Dim: m_nStrms
    std::vector<cuphy::stream> m_cuStrms;
    std::vector<cuphy::stream> m_cuStrmsPusch;
    std::vector<cuphy::stream> m_cuStrmsUlbfw;
    std::vector<cuphy::stream> m_cuStrmsPucch;
    std::vector<cuphy::stream> m_cuStrmsPrach;
    std::vector<cuphy::stream> m_cuStrmsSrs; // used for both SRS1 and SRS2
    std::vector<cuphy::stream> m_cuStrmsPdsch;
    std::vector<cuphy::stream> m_cuStrmsDlbfw;
    std::vector<cuphy::stream> m_cuStrmsPdcch;
    std::vector<cuphy::stream> m_cuStrmsSsb;
    std::vector<cuphy::stream> m_cuStrmsCsirs;

    // Pipeline handles. Dim: m_nStrms x m_nItrsPerStrm
    std::vector<std::vector<cuphy::pusch_rx>> m_puschRxPipes;
    std::vector<std::vector<cuphy::pdsch_tx>> m_pdschTxPipes;
    std::vector<std::vector<cuphy::pdcch_tx>> m_pdcchTxPipes;
    std::vector<std::vector<cuphy::pucch_rx>> m_pucchRxPipes;

    std::unique_ptr<cuphy::event>  m_uqPtrPdschIterStopEvent;
    std::unique_ptr<cuphy::event>  m_uqPtrUlbfwDelayStopEvent;
    std::unique_ptr<cuphy::event>  m_uqPtrUlbfw2DelayStopEvent;
    std::unique_ptr<cuphy::event>  m_uqPtrPuschDelayStopEvent;
    std::unique_ptr<cuphy::event>  m_uqPtrPusch2DelayStopEvent;
    std::unique_ptr<cuphy::event>  m_uqPtrPucch2DelayStopEvent;
    std::unique_ptr<cuphy::event>  m_uqPtrSRSDelayStopEvent;
    std::unique_ptr<cuphy::event>  m_uqPtrSRSStopEvent;
    std::unique_ptr<cuphy::event>  m_uqPtrSRS2StopEvent;
    std::unique_ptr<cuphy::event>  m_uqPtrUlbfwStopEvent;
    std::unique_ptr<cuphy::event>  m_uqPtrUlbfw2StopEvent;
    std::unique_ptr<cuphy::event>  m_uqPtrSSBStopEvent;
    std::unique_ptr<cuphy::event>  m_uqPtrPRACHStopEvent;
    std::unique_ptr<cuphy::event>  m_uqPtrDlbfwStopEvent;
    std::unique_ptr<cuphy::event>  m_uqPtrPUCCHStopEvent;

    std::vector<cuphy::event> m_stopEvents;       // Dim: m_nStrms
    std::vector<cuphy::event> m_stop2Events;      // Dim: m_nStrms
    std::vector<cuphy::event> m_SRSStopEvents;    // Dim: m_nStrms
    std::vector<cuphy::event> m_PDCCHStopEvents;  // Dim: m_nStrms
    std::vector<cuphy::event> m_CSIRSStopEvents;  // Dim: m_nStrms
    std::vector<cuphy::event> m_PRACHStopEvents;  // Dim: m_nStrms
    std::vector<cuphy::event> m_PUCCHStopEvents;  // Dim: m_nStrms
    std::vector<cuphy::event> m_PUCCHStop2Events; // Dim: m_nStrms
    std::vector<cuphy::event> m_pdschInterSlotStartEventVec;
    std::vector<cuphy::event> m_SlotBoundaryEventVec; // only used in u6 where PDSCH start in the middle of a slot, otherwise, it's empty and m_pdschInterSlotStartEventVec should be used for slot boundary per 500 us
    std::vector<cuphy::event> m_pdcchCsirsInterSlotEndEventVec;

    std::unique_ptr<cuphy::event> m_uqPtrTimeStartEvent;
    std::unique_ptr<cuphy::event> m_uqPtrTimeSRSStartEvent;
    std::unique_ptr<cuphy::event> m_uqPtrTimeSRSEndEvent;
    std::unique_ptr<cuphy::event> m_uqPtrTimeSRS2StartEvent;
    std::unique_ptr<cuphy::event> m_uqPtrTimeSRS2EndEvent;
    std::unique_ptr<cuphy::event> m_uqPtrTimePRACHStartEvent;
    std::unique_ptr<cuphy::event> m_uqPtrTimePRACHEndEvent;
    std::unique_ptr<cuphy::event> m_uqPtrTimePUSCHStartEvent;
    std::unique_ptr<cuphy::event> m_uqPtrTimePUSCHEndEvent;
    std::unique_ptr<cuphy::event> m_uqPtrTimePUSCH2StartEvent;
    std::unique_ptr<cuphy::event> m_uqPtrTimePUSCH2EndEvent;
    std::unique_ptr<cuphy::event> m_uqPtrTimePUCCHStartEvent;
    std::unique_ptr<cuphy::event> m_uqPtrTimePUCCHEndEvent;
    std::unique_ptr<cuphy::event> m_uqPtrTimePUCCH2StartEvent;
    std::unique_ptr<cuphy::event> m_uqPtrTimePUCCH2EndEvent;
    std::unique_ptr<cuphy::event> m_uqPtrTimeULBfwStartEvent;
    std::unique_ptr<cuphy::event> m_uqPtrTimeULBfwEndEvent;
    std::unique_ptr<cuphy::event> m_uqPtrTimeULBfw2StartEvent;
    std::unique_ptr<cuphy::event> m_uqPtrTimeULBfw2EndEvent;

    // change to multiple SSB slots
    std::vector<cuphy::event>              m_timeSSBSlotStartEvents;
    std::vector<cuphy::event>              m_timeSSBSlotEndEvents;

    std::vector<std::vector<cuphy::event>> m_pschDlUlSyncEvents; // Dim: m_nStrms x nItrsPerStrm
    std::vector<cuphy::event>              m_timePdschSlotEndEvents;
    std::vector<cuphy::event>              m_timePdcchSlotStartEvents;
    std::vector<cuphy::event>              m_timeCSIRSSlotStartEvents;
    std::vector<cuphy::event>              m_timeDlbfwSlotStartEvents;
    std::vector<cuphy::event>              m_timeDlbfwSlotEndEvents;
    std::vector<cuphy::event>              m_timePdcchSlotEndEvents;
    std::vector<cuphy::event>              m_timeCSIRSSlotEndEvents;

    // Datasets. Dim: m_nStrms x m_nItrsPerStrm
    std::vector<std::vector<StaticApiDataset>>      m_puschRxStaticApiDataSets;
    std::vector<std::vector<DynApiDataset>>         m_puschRxDynamicApiDataSets;
    std::vector<std::vector<EvalDataset>>           m_puschRxEvalDataSets;
    std::vector<std::vector<pdschStaticApiDataset>> m_pdschTxStaticApiDataSets;
    std::vector<std::vector<pdschDynApiDataset>>    m_pdschTxDynamicApiDataSets;
    std::vector<std::vector<pdcchStaticApiDataset>> m_pdcchTxStaticApiDataSets;
    std::vector<std::vector<pdcchDynApiDataset>>    m_pdcchTxDynamicApiDataSets;

    // PUCCH
    std::vector<std::vector<pucchStaticApiDataset>>    m_pucchStaticDatasetVec;
    std::vector<std::vector<pucchDynApiDataset>>       m_pucchDynDatasetVec;
    std::vector<std::vector<EvalPucchDataset>>         m_pucchEvalDatasetVec;
    std::vector<std::vector<cuphyPucchBatchPrmHndl_t>> m_pucchBatchPrmHndlVec; // not used for the moment

    //SRS
    std::vector<std::vector<srsStaticApiDataset>> m_srsStaticApiDatasetVec;
    std::vector<std::vector<srsDynApiDataset>>    m_srsDynamicApiDatasetVec;
    std::vector<std::vector<srsEvalDataset>>      m_srsEvalDatasetVec;

    std::vector<std::vector<srsStaticApiDataset>> m_srsStaticApiDatasetVec2;
    std::vector<std::vector<srsDynApiDataset>>    m_srsDynamicApiDatasetVec2;
    std::vector<std::vector<srsEvalDataset>>      m_srsEvalDatasetVec2;

    cuphySrsRxHndl_t m_srsRxHndl;
    cuphySrsRxHndl_t m_srsRxHndl2;

    std::vector<std::vector<uint32_t>>                                    m_nSRSCellsVec;
    std::vector<std::vector<cuphy::tensor_device>>                        m_tDataRxVec;
    std::vector<std::vector<cuphy::tensor_device>>                        m_tFreqInterpCoefsVec;
    std::vector<std::vector<cuphy::tensor_device>>                        m_tSRSHEstVec;
    std::vector<std::vector<cuphy::tensor_device>>                        m_tSRSDbgVec;
    std::vector<std::vector<cuphy::buffer<uint8_t, cuphy::pinned_alloc>>> m_statDescrBufCpuVec;
    std::vector<std::vector<cuphy::buffer<uint8_t, cuphy::pinned_alloc>>> m_dynDescrBufCpuVec;
    std::vector<std::vector<cuphy::buffer<uint8_t, cuphy::device_alloc>>> m_statDescrBufGpuVec;
    std::vector<std::vector<cuphy::buffer<uint8_t, cuphy::device_alloc>>> m_dynDescrBufGpuVec;
    std::vector<std::vector<cuphySrsChEstHndl_t>>                         m_srsChEstHndlVec;

    // DLBFW parameters
    std::vector<std::vector<bfwStaticApiDataset>> m_dlbfwStaticApiDatasetVec;
    std::vector<std::vector<bfwDynApiDataset>>    m_dlbfwDynamicApiDatasetVec;
    std::vector<std::vector<bfwEvalDataset>>      m_dlbfwEvalDatasetVec;
    std::vector<std::vector<cuphy::bfw_tx>>       m_dlbfwPipelineVec;

    // ULBFW parameters
    std::vector<std::vector<bfwStaticApiDataset>> m_ulbfwStaticApiDatasetVec;
    std::vector<std::vector<bfwDynApiDataset>>    m_ulbfwDynamicApiDatasetVec;
    std::vector<std::vector<bfwEvalDataset>>      m_ulbfwEvalDatasetVec;
    // TODO ULBFW is the same with DLBFW kernels
    std::vector<std::vector<cuphy::bfw_tx>>       m_ulbfwPipelineVec;

    // PRACH
    std::vector<std::vector<PrachApiDataset>> m_prachDatasetVec;
    std::vector<std::vector<cuphy::prach_rx>> m_prachRxPipes;

    //SSB
    std::vector<std::vector<cuphy::ssb_tx>>       m_ssbTxPipes;
    std::vector<std::vector<ssbStaticApiDataset>> m_ssbTxStaticApiDataSets;
    std::vector<std::vector<ssbDynApiDataset>>    m_ssbTxDynamicApiDataSets;

    //CSIRS
    std::vector<std::vector<cuphy::csirs_tx>>       m_csirsTxPipes;
    std::vector<std::vector<csirsStaticApiDataset>> m_csirsTxStaticApiDataSets;
    std::vector<std::vector<csirsDynApiDataset>>    m_csirsTxDynamicApiDataSets;

    // timing objects.
    std::vector<float>                           m_totRunTimePdschItr;    // Dim: pdsch_nItrsPerStrm
    std::vector<float>                           m_totPdschSlotStartTime; // Dim: pdsch_nItrsPerStrm
    std::vector<float>                           m_totDlbfwIterStartTime;      // Dim: dlbfw_nItrsPerStrm
    std::vector<float>                           m_totRunTimeDlbfwItr;      // Dim: dlbfw_nItrsPerStrm
    std::vector<float>                           m_totPdcchStartTimes;    // Dim: pdcch_nItrsPerStrm
    std::vector<float>                           m_totRunTimePdcchItr;    // Dim: pdcch_nItrsPerStrm
    std::vector<float>                           m_totCSIRSStartTimes;    // Dim: csirs_nItrsPerStrm
    std::vector<float>                           m_totRunTimeCSIRSItr;    // Dim: csirs_nItrsPerStrm
    std::vector<float>                           m_totSSBStartTime; // Dim: pbch_nItrsPerStrm
    std::vector<float>                           m_totRunTimeSSBItr;   // Dim: pbch_nItrsPerStrm
    float                                        m_totSRSStartTime;
    float                                        m_totSRSRunTime;
    float                                        m_totSRS2StartTime;
    float                                        m_totSRS2RunTime;
    float                                        m_totPRACHStartTime;
    float                                        m_totPRACHRunTime;
    float                                        m_totPUSCHStartTime;
    float                                        m_totPUSCHSubslotProcRunTime;
    float                                        m_totPUSCHRunTime;
    float                                        m_totPUSCH2StartTime;
    float                                        m_totPUSCH2SubslotProcRunTime;
    float                                        m_totPUSCH2RunTime;
    float                                        m_totULBFWStartTime;
    float                                        m_totULBFWRunTime;
    float                                        m_totULBFW2StartTime;
    float                                        m_totULBFW2RunTime;
    float                                        m_totPUCCHRunTime;
    float                                        m_totPUCCHStartTime;
    float                                        m_totPUCCH2RunTime;
    float                                        m_totPUCCH2StartTime;

    // variable delay memory pointer
    // only PDSCH and PUSCH worker will have timer objects
    cuphy::buffer<uint64_t, cuphy::device_alloc> m_GPUtimeUl_d;
    cuphy::buffer<uint64_t, cuphy::device_alloc> m_GPUtimeDl_d;

    // evaluation objects
    std::vector<std::vector<uint32_t>> m_maxNumCbErrors; // Dim: m_nStrms x m_nItrsPerStrm
    uint32_t                           m_nTimingItrs;
    bool                               m_printCbErrors;

    // pusch/pdsch configurations
    uint32_t                                     m_descramblingOn;
    bool                                         m_ref_check_pdsch;
    bool                                         m_ref_check_pdcch;
    bool                                         m_ref_check_csirs;
    bool                                         m_ref_check_pucch;
    bool                                         m_ref_check_prach;
    bool                                         m_ref_check_ssb;
    bool                                         m_ref_check_srs;
    bool                                         m_ref_check_dlbfw;
    bool                                         m_ref_check_ulbfw;
    bool                                         m_identical_ldpc_configs;
    cuphyPdschProcMode_t                         m_pdsch_proc_mode;
    uint64_t                                     m_pdcch_proc_mode;
    uint64_t                                     m_csirs_proc_mode;
    uint64_t                                     m_ssb_proc_mode;
    uint64_t                                     m_pusch_proc_mode;
    uint64_t                                     m_pucch_proc_mode;
    uint64_t                                     m_prach_proc_mode;
    uint64_t                                     m_srs_proc_mode;
    uint32_t                                     m_ldpc_kernel_launch_mode;
    uint32_t                                     m_fp16Mode;
    bool                                         m_runDlbfw;
    bool                                         m_runUlbfw;
    bool                                         m_runPDSCH;
    bool                                         m_runSRS;
    bool                                         m_runSRS2;
    bool                                         m_runPRACH;
    bool                                         m_runPUSCH;
    bool                                         m_runPDCCH;
    bool                                         m_runCSIRS;
    bool                                         m_runPUCCH;
    bool                                         m_runSSB;
    bool                                         m_pdsch_group_cells;
    bool                                         m_pdcch_group_cells;
    bool                                         m_pusch_group_cells;
    bool                                         m_pucch_group_cells;
    bool                                         m_prach_group_cells;
    bool                                         m_srsCtx;  // if true, srs runs on a separate context with a different fixed delay
    uint32_t                                     m_nUlbfwWorkloads; // 2 workloads in u5 or u6; otherwise 1

    // PUSCH subslot processing configuration
    // 1 - enable, 0 - disable
    uint8_t m_puschProcModeSubslotProcFlag;
    uint8_t m_pusch2ProcModeSubslotProcFlag;

    // DL channel slot config
    uint32_t                    m_pdsch_nItrsPerStrm;
    uint32_t                    m_dlbfw_nItrsPerStrm;
    uint32_t                    m_pdcch_nItrsPerStrm;
    uint32_t                    m_csirs_nItrsPerStrm;
    uint32_t                    m_pbch_nItrsPerStrm;
    uint8_t*                    m_pdschSlotRunFlag;
    uint8_t*                    m_pdcchSlotRunFlag;
    uint8_t*                    m_csirsSlotRunFlag;
    uint8_t*                    m_pbchSlotRunFlag;
    uint8_t                     m_pdschRunSlotIdx;
    uint8_t                     m_pdcchRunSlotIdx;
    uint8_t                     m_csirsRunSlotIdx;
    uint8_t                     m_pbchRunSlotIdx;
};

#endif // !defined(CUPHY_TESTWRKR_HPP_INCLUDED_)
