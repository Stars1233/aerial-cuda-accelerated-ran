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

#include "cuphy_testWrkr.hpp"

#include <atomic>

namespace
{
std::atomic<bool> g_logged_pusch_tv_override_apply{false};
std::atomic<bool> g_logged_pucch_tv_override_apply{false};
std::atomic<bool> g_logged_srs_tv_override_apply{false};

void apply_pusch_tv_overrides(cuphyPuschStatPrms_t& p)
{
    if(!g_tv_override_cfg.enable)
        return;

    uint32_t applied_count = 0;
    auto log_applied = [&](const char* key, uint32_t value) {
        if(!g_logged_pusch_tv_override_apply.load(std::memory_order_relaxed))
        {
            NVLOGI_FMT(NVLOG_PUSCH, "[TV-OVERRIDE][PUSCH][APPLIED] {}={}", key, value);
        }
        ++applied_count;
    };

    const auto& o = g_tv_override_cfg.pusch;

    if(o.has_polar_list_length)
    {
        p.polarDcdrListSz = o.polar_list_length;
        log_applied("list_length", o.polar_list_length);
    }
    if(o.has_enable_cfo_correction)
    {
        p.enableCfoCorrection = o.enable_cfo_correction;
        log_applied("enable_cfo_correction", o.enable_cfo_correction);
    }
    if(o.has_enable_weighted_average_cfo)
    {
        p.enableWeightedAverageCfo = o.enable_weighted_average_cfo;
        log_applied("enable_weighted_average_cfo", o.enable_weighted_average_cfo);
    }
    if(o.has_enable_to_estimation)
    {
        p.enableToEstimation = o.enable_to_estimation;
        log_applied("enable_to_estimation", o.enable_to_estimation);
    }
    if(o.has_tdi_mode)
    {
        p.enablePuschTdi = (o.tdi_mode == 1);
        log_applied("tdi_mode", o.tdi_mode);
    }
    if(o.has_enable_dft_sofdm)
    {
        p.enableDftSOfdm = o.enable_dft_sofdm;
        log_applied("enable_dft_sofdm", o.enable_dft_sofdm);
    }
    if(o.has_enable_rssi_measurement)
    {
        p.enableRssiMeasurement = o.enable_rssi_measurement;
        log_applied("enable_rssi_measurement", o.enable_rssi_measurement);
    }
    if(o.has_enable_sinr_measurement)
    {
        p.enableSinrMeasurement = o.enable_sinr_measurement;
        log_applied("enable_sinr_measurement", o.enable_sinr_measurement);
    }
    if(o.has_enable_static_dynamic_beamforming)
    {
        p.enableMassiveMIMO = o.enable_static_dynamic_beamforming;
        log_applied("enable_static_dynamic_beamforming", o.enable_static_dynamic_beamforming);
    }
    if(o.has_enable_early_harq)
    {
        p.enableEarlyHarq = o.enable_early_harq;
        log_applied("enable_early_harq", o.enable_early_harq);
    }

    if(o.has_ldpc_early_termination)
    {
        p.ldpcEarlyTermination = o.ldpc_early_termination;
        log_applied("ldpc_early_termination", o.ldpc_early_termination);
    }
    if(o.has_ldpc_algorithm_index)
    {
        p.ldpcAlgoIndex = o.ldpc_algorithm_index;
        log_applied("ldpc_algorithm_index", o.ldpc_algorithm_index);
    }
    if(o.has_ldpc_flags)
    {
        p.ldpcFlags = o.ldpc_flags;
        log_applied("ldpc_flags", o.ldpc_flags);
    }
    if(o.has_ldpc_use_half)
    {
        p.ldpcUseHalf = o.ldpc_use_half;
        log_applied("ldpc_use_half", o.ldpc_use_half);
    }

    if(o.has_ldpc_max_num_iterations)
    {
        p.fixedMaxNumLdpcItrs = o.ldpc_max_num_iterations;
        log_applied("ldpc_max_num_iterations", o.ldpc_max_num_iterations);
    }
    if(o.has_ldpc_max_num_iterations_algorithm_index)
    {
        bool valid = false;
        switch(o.ldpc_max_num_iterations_algorithm_index)
        {
            case 0:
                p.ldpcMaxNumItrAlgo = LDPC_MAX_NUM_ITR_ALGO_TYPE_FIXED;
                valid = true;
                break;
            case 1:
                p.ldpcMaxNumItrAlgo = LDPC_MAX_NUM_ITR_ALGO_TYPE_LUT;
                valid = true;
                break;
            case 2:
                p.ldpcMaxNumItrAlgo = LDPC_MAX_NUM_ITR_ALGO_TYPE_PER_UE;
                valid = true;
                break;
            default:
                break;
        }
        if(valid)
            log_applied("ldpc_max_num_iterations_algorithm_index", o.ldpc_max_num_iterations_algorithm_index);
        else
            NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "[TV-OVERRIDE][PUSCH] invalid ldpc_max_num_iterations_algorithm_index={}", static_cast<unsigned int>(o.ldpc_max_num_iterations_algorithm_index));
    }

    if(o.has_eq_coefficient_algorithm_index)
    {
        bool valid = false;
        switch(o.eq_coefficient_algorithm_index)
        {
            case 0:
                p.eqCoeffAlgo = PUSCH_EQ_ALGO_TYPE_RZF;
                valid = true;
                break;
            case 1:
                p.eqCoeffAlgo = PUSCH_EQ_ALGO_TYPE_NOISE_DIAG_MMSE;
                valid = true;
                break;
            case 2:
                p.eqCoeffAlgo = PUSCH_EQ_ALGO_TYPE_MMSE_IRC;
                valid = true;
                break;
            case 3:
                p.eqCoeffAlgo = PUSCH_EQ_ALGO_TYPE_MMSE_IRC_SHRINK_RBLW;
                valid = true;
                break;
            case 4:
                p.eqCoeffAlgo = PUSCH_EQ_ALGO_TYPE_MMSE_IRC_SHRINK_OAS;
                valid = true;
                break;
            default:
                break;
        }
        if(valid)
            log_applied("eq_coefficient_algorithm_index", o.eq_coefficient_algorithm_index);
        else
            NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "[TV-OVERRIDE][PUSCH] invalid eq_coefficient_algorithm_index={}", static_cast<unsigned int>(o.eq_coefficient_algorithm_index));
    }

    if(o.has_dmrs_channel_estimation_algorithm_index)
    {
        bool valid = false;
        switch(o.dmrs_channel_estimation_algorithm_index)
        {
            case 0:
                p.chEstAlgo = PUSCH_CH_EST_ALGO_TYPE_LEGACY_MMSE;
                valid = true;
                break;
            case 1:
                p.chEstAlgo = PUSCH_CH_EST_ALGO_TYPE_MULTISTAGE_MMSE_WITH_DELAY_EST;
                valid = true;
                break;
            case 2:
                p.chEstAlgo = PUSCH_CH_EST_ALGO_TYPE_RKHS;
                valid = true;
                break;
            default:
                break;
        }
        if(valid)
            log_applied("dmrs_channel_estimation_algorithm_index", o.dmrs_channel_estimation_algorithm_index);
        else
            NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "[TV-OVERRIDE][PUSCH] invalid dmrs_channel_estimation_algorithm_index={}", static_cast<unsigned int>(o.dmrs_channel_estimation_algorithm_index));
    }

    if(o.has_enable_per_prg_channel_estimation)
    {
        p.enablePerPrgChEst = o.enable_per_prg_channel_estimation;
        log_applied("enable_per_prg_channel_estimation", o.enable_per_prg_channel_estimation);
    }

    if(applied_count > 0 && !g_logged_pusch_tv_override_apply.load(std::memory_order_relaxed))
    {
        NVLOGI_FMT(NVLOG_PUSCH, "[TV-OVERRIDE][PUSCH] applied {} override field(s)", applied_count);
        NVLOGI_FMT(
            NVLOG_PUSCH,
            "[TV-OVERRIDE][PUSCH][SUMMARY] applied={}, ldpcEarlyTermination={}, ldpcAlgoIndex={}, "
            "ldpcFlags={}, ldpcUseHalf={}, fixedMaxNumLdpcItrs={}, eqCoeffAlgo={}",
            applied_count,
            static_cast<unsigned int>(p.ldpcEarlyTermination),
            static_cast<unsigned int>(p.ldpcAlgoIndex),
            static_cast<unsigned int>(p.ldpcFlags),
            static_cast<unsigned int>(p.ldpcUseHalf),
            static_cast<unsigned int>(p.fixedMaxNumLdpcItrs),
            static_cast<unsigned int>(p.eqCoeffAlgo));
        NVLOGI_FMT(
            NVLOG_PUSCH,
            "[TV-OVERRIDE][PUSCH][EFFECTIVE] polarDcdrListSz={} enableCfoCorrection={} "
            "enableWeightedAverageCfo={} enableToEstimation={} enablePuschTdi={} enableDftSOfdm={} "
            "enableRssiMeasurement={} enableSinrMeasurement={} enableMassiveMIMO={} enableEarlyHarq={} "
            "ldpcEarlyTermination={} ldpcAlgoIndex={} ldpcFlags={} ldpcUseHalf={} fixedMaxNumLdpcItrs={} "
            "ldpcMaxNumItrAlgo={} eqCoeffAlgo={} chEstAlgo={} enablePerPrgChEst={}",
            static_cast<unsigned int>(p.polarDcdrListSz),
            static_cast<unsigned int>(p.enableCfoCorrection),
            static_cast<unsigned int>(p.enableWeightedAverageCfo),
            static_cast<unsigned int>(p.enableToEstimation),
            static_cast<unsigned int>(p.enablePuschTdi),
            static_cast<unsigned int>(p.enableDftSOfdm),
            static_cast<unsigned int>(p.enableRssiMeasurement),
            static_cast<unsigned int>(p.enableSinrMeasurement),
            static_cast<unsigned int>(p.enableMassiveMIMO),
            static_cast<unsigned int>(p.enableEarlyHarq),
            static_cast<unsigned int>(p.ldpcEarlyTermination),
            static_cast<unsigned int>(p.ldpcAlgoIndex),
            static_cast<unsigned int>(p.ldpcFlags),
            static_cast<unsigned int>(p.ldpcUseHalf),
            static_cast<unsigned int>(p.fixedMaxNumLdpcItrs),
            static_cast<unsigned int>(p.ldpcMaxNumItrAlgo),
            static_cast<unsigned int>(p.eqCoeffAlgo),
            static_cast<unsigned int>(p.chEstAlgo),
            static_cast<unsigned int>(p.enablePerPrgChEst));
        g_logged_pusch_tv_override_apply.store(true, std::memory_order_relaxed);
    }
}

void apply_pucch_tv_overrides(cuphyPucchStatPrms_t& p)
{
    if(!g_tv_override_cfg.enable)
        return;

    const auto& o = g_tv_override_cfg.pucch;
    if(o.has_polar_list_length)
    {
        p.polarDcdrListSz = o.polar_list_length;
        if(!g_logged_pucch_tv_override_apply.load(std::memory_order_relaxed))
        {
            NVLOGI_FMT(NVLOG_PUCCH, "[TV-OVERRIDE][PUCCH][APPLIED] list_length={}", static_cast<unsigned int>(o.polar_list_length));
            NVLOGI_FMT(NVLOG_PUCCH, "[TV-OVERRIDE][PUCCH] applied 1 override field(s)");
            printf("[TV-OVERRIDE][PUCCH][EFFECTIVE] polarDcdrListSz=%u\n", static_cast<unsigned int>(p.polarDcdrListSz));
            g_logged_pucch_tv_override_apply.store(true, std::memory_order_relaxed);
        }
    }
}

void apply_srs_tv_overrides(cuphySrsStatPrms_t& p)
{
    if(!g_tv_override_cfg.enable)
        return;
    const auto& o = g_tv_override_cfg.srs;
    if(!o.has_chest_alg_index)
        return;
    bool valid = false;
    switch(o.chest_alg_index)
    {
        case 0:
            p.chEstAlgo = SRS_CH_EST_ALGO_TYPE_MMSE;
            valid = true;
            break;
        case 1:
            p.chEstAlgo = SRS_CH_EST_ALGO_TYPE_RKHS;
            valid = true;
            break;
        default:
            break;
    }
    if(valid)
    {
        if(!g_logged_srs_tv_override_apply.load(std::memory_order_relaxed))
        {
            NVLOGI_FMT(NVLOG_SRS, "[TV-OVERRIDE][SRS][APPLIED] chEst_alg_selector={}", static_cast<unsigned int>(o.chest_alg_index));
            NVLOGI_FMT(NVLOG_SRS, "[TV-OVERRIDE][SRS] applied 1 override field(s)");
            printf("[TV-OVERRIDE][SRS][EFFECTIVE] chEstAlgo=%u\n", static_cast<unsigned int>(p.chEstAlgo));
            g_logged_srs_tv_override_apply.store(true, std::memory_order_relaxed);
        }
    }
    else
        NVLOGE_FMT(NVLOG_SRS, AERIAL_CUPHY_EVENT, "[TV-OVERRIDE][SRS] invalid chest_alg_index={}", static_cast<unsigned int>(o.chest_alg_index));
}
} // namespace

cuPHYTestWorker::cuPHYTestWorker(std::string const& name, uint32_t workerId, int cpuId, int gpuId, int cpuThrdSchdPolicy, int cpuThrdPrio, uint32_t mpsSubctxSmCount, std::shared_ptr<testWrkrCmdQ>& cmdQ, std::shared_ptr<testWrkrRspQ>& rspQ, int uldlMode, uint32_t debugMessageLevel, bool useGreenContexts, const cuphy::cudaGreenContext& greenCtx) :
    testWorker(name, workerId, cpuId, gpuId, cpuThrdSchdPolicy, cpuThrdPrio, mpsSubctxSmCount, cmdQ, rspQ, uldlMode, debugMessageLevel, useGreenContexts),
    m_fp16Mode(1),
    m_descramblingOn(1),
    m_ref_check_pdsch(false),
    m_ref_check_pdcch(false),
    m_ref_check_csirs(false),
    m_ref_check_pucch(false),
    m_ref_check_prach(false),
    m_ref_check_ssb(false),
    m_ref_check_srs(false),
    m_ref_check_ulbfw(false),
    m_ref_check_dlbfw(false),
    m_identical_ldpc_configs(true),
    m_pdsch_group_cells(false), /* Will be updated in pdschTxInit as needed */
    m_pusch_group_cells(false), /* Will be updated in puschRxInit as needed */
    m_pucch_group_cells(false), /* Will be updated in pucchRxInit as needed */
    m_pdsch_proc_mode(PDSCH_PROC_MODE_NO_GRAPHS),
    m_pdcch_proc_mode(0),
    m_csirs_proc_mode(0),
    m_pusch_proc_mode(0),
    m_pucch_proc_mode(0),
    m_prach_proc_mode(0),
    m_srs_proc_mode(0),
    m_nStrms(0),
    m_nStrms_pdsch(0),
    m_nItrsPerStrm(0),
    m_nSlotsPerPattern(0),
    m_nTimingItrs(0),
    m_nSRSCells(0),
    m_nUlbfwCells(0),
    m_nDlbfwCells(0),
    m_nPDCCHCells(0),
    m_nPUCCHCells(0),
    m_nPRACHCells(0),
    m_nSSBCells(0),
    m_runUlbfw(false),
    m_runDlbfw(false),
    m_runSRS(false),
    m_runSRS2(false),
    m_runPDSCH(false),
    m_runPRACH(false),
    m_runPUSCH(false),
    m_runPUCCH(false),
    m_runPDCCH(false),
    m_runCSIRS(false),
    m_runSSB(false),
    m_srsCtx(false),
    m_totSRSStartTime(0),
    m_totSRSRunTime(0),
    m_totSRS2StartTime(0),
    m_totSRS2RunTime(0),
    m_totPUSCHStartTime(0),
    m_totPUSCHSubslotProcRunTime(0),
    m_totPUSCHRunTime(0),
    m_totPUSCH2StartTime(0),
    m_totPUSCH2SubslotProcRunTime(0),
    m_totPUSCH2RunTime(0),
    m_totPUCCHStartTime(0),
    m_totPUCCHRunTime(0),
    m_totPUCCH2StartTime(0),
    m_totPUCCH2RunTime(0),
    m_totPRACHStartTime(0),
    m_totPRACHRunTime(0),
    m_totULBFWStartTime(0),
    m_totULBFWRunTime(0),
    m_totULBFW2StartTime(0),
    m_totULBFW2RunTime(0),
    m_pdschSlotRunFlag(nullptr),
    m_pdcchSlotRunFlag(nullptr),
    m_csirsSlotRunFlag(nullptr),
    m_pbchSlotRunFlag(nullptr),
    m_pdschRunSlotIdx(0),
    m_pdcchRunSlotIdx(0),
    m_csirsRunSlotIdx(0),
    m_pbchRunSlotIdx(0),
    m_pdsch_nItrsPerStrm(0),
    m_dlbfw_nItrsPerStrm(0),
    m_pdcch_nItrsPerStrm(0),
    m_csirs_nItrsPerStrm(0),
    m_pbch_nItrsPerStrm(0),
    m_puschProcModeSubslotProcFlag(0),
    m_pusch2ProcModeSubslotProcFlag(0)
{
    // Start worker thread
    m_thrd = std::thread(&cuPHYTestWorker::run, this, std::cref(greenCtx));
}

cuPHYTestWorker::~cuPHYTestWorker()
{
    if(m_runSRS)
    {
        cuphyStatus_t statusDestroy = cuphyDestroySrsRx(m_srsRxHndl);
        if(CUPHY_STATUS_SUCCESS != statusDestroy)
            NVLOGE_FMT(NVLOG_SRS, AERIAL_CUPHY_EVENT,  "cuPHYTestWorker Destructor Error: cuphyDestroySrsRx (SRS1)");
    }
    if(m_runSRS2)
    {
        cuphyStatus_t statusDestroy = cuphyDestroySrsRx(m_srsRxHndl2);
        if(CUPHY_STATUS_SUCCESS != statusDestroy)
            NVLOGE_FMT(NVLOG_SRS, AERIAL_CUPHY_EVENT,  "cuPHYTestWorker Destructor Error: cuphyDestroySrsRx (SRS2)");
    }

    // implicit call destructor of testWorker::~testWorker()
}

void cuPHYTestWorker::createCuOrGreenCtx(const cuphy::cudaGreenContext& my_green_context)
{
    if (m_useGreenContexts)
    {
#if CUDA_VERSION >= 12040
        // createCuGreenCtx is a misnomer as the context has already been created.
        // Here we just set it for a given thread; each thread only has a single green context set as current
        testWorker::createCuGreenCtx(my_green_context);
#endif
    }
    else
    {
        testWorker::createCuCtx();
    }

    m_uqPtrSRSStopEvent         = std::make_unique<cuphy::event>(cudaEventDisableTiming);
    m_uqPtrSRS2StopEvent        = std::make_unique<cuphy::event>(cudaEventDisableTiming);
    m_uqPtrSSBStopEvent         = std::make_unique<cuphy::event>(cudaEventDisableTiming);
    m_uqPtrUlbfwDelayStopEvent  = std::make_unique<cuphy::event>(cudaEventDisableTiming);
    m_uqPtrUlbfw2DelayStopEvent = std::make_unique<cuphy::event>(cudaEventDisableTiming);
    m_uqPtrPuschDelayStopEvent  = std::make_unique<cuphy::event>(cudaEventDisableTiming);
    m_uqPtrPusch2DelayStopEvent = std::make_unique<cuphy::event>(cudaEventDisableTiming);
    m_uqPtrPucch2DelayStopEvent = std::make_unique<cuphy::event>(cudaEventDisableTiming);
    m_uqPtrSRSDelayStopEvent    = std::make_unique<cuphy::event>(cudaEventDisableTiming);
    m_uqPtrPdschIterStopEvent   = std::make_unique<cuphy::event>(cudaEventDisableTiming);
    m_uqPtrDlbfwStopEvent       = std::make_unique<cuphy::event>(cudaEventDisableTiming);
    m_uqPtrUlbfwStopEvent       = std::make_unique<cuphy::event>(cudaEventDisableTiming);
    m_uqPtrUlbfw2StopEvent      = std::make_unique<cuphy::event>(cudaEventDisableTiming);

    m_uqPtrTimeSRSStartEvent    = std::make_unique<cuphy::event>();
    m_uqPtrTimeSRSEndEvent      = std::make_unique<cuphy::event>();
    m_uqPtrTimeSRS2StartEvent   = std::make_unique<cuphy::event>();
    m_uqPtrTimeSRS2EndEvent     = std::make_unique<cuphy::event>();
    m_uqPtrTimePRACHStartEvent  = std::make_unique<cuphy::event>();
    m_uqPtrTimePRACHEndEvent    = std::make_unique<cuphy::event>();
    m_uqPtrTimePUCCHStartEvent  = std::make_unique<cuphy::event>();
    m_uqPtrTimePUCCHEndEvent    = std::make_unique<cuphy::event>();
    m_uqPtrTimePUCCH2StartEvent = std::make_unique<cuphy::event>();
    m_uqPtrTimePUCCH2EndEvent   = std::make_unique<cuphy::event>();

    m_uqPtrTimePUSCHStartEvent  = std::make_unique<cuphy::event>();
    m_uqPtrTimePUSCHEndEvent    = std::make_unique<cuphy::event>();
    m_uqPtrTimePUSCH2StartEvent = std::make_unique<cuphy::event>();
    m_uqPtrTimePUSCH2EndEvent   = std::make_unique<cuphy::event>();

    m_uqPtrTimeULBfwStartEvent  = std::make_unique<cuphy::event>();
    m_uqPtrTimeULBfwEndEvent    = std::make_unique<cuphy::event>();
    m_uqPtrTimeULBfw2StartEvent = std::make_unique<cuphy::event>();
    m_uqPtrTimeULBfw2EndEvent   = std::make_unique<cuphy::event>();

    printf("\n--> Runs %d stream(s) in parallel", m_nStrms);
    if(m_name.compare("PschTxRxTestWorker") == 0)
    {
        printf("\n--> Each stream processes %d PUSCH + PDSCH test-vector(s) in series\n\n", m_nItrsPerStrm);
    }
    if(m_name.compare("PdschTxTestWorker") == 0)
    {
        printf("\n--> Each stream processes %d PDSCH test-vector(s) in series\n\n", m_nItrsPerStrm);
    }
    if(m_name.compare("PuschRxTestWorker") == 0)
    {
        printf("\n--> Each stream processes %d PUSCH test-vector(s) in series\n\n", m_nItrsPerStrm);
    }
}

//----------------------------------------------------------------------------------------------------------
// Message handlers
void cuPHYTestWorker::emptyHandler(std::shared_ptr<void>& shPtrPayload)
{
    DEBUG_TRACE("%s id %d [tid %s][wrkrCtxId 0x%0lx currCtxId 0x%0lx]: emptyHandler\n", m_name.c_str(), m_wrkrId, getThreadIdStr().c_str(), getCuCtxId(), getCurrCuCtxId());
}

void cuPHYTestWorker::initHandler(std::shared_ptr<void>& shPtrPayload)
{
    DEBUG_TRACE("%s id %d [tid %s][wrkrCtxId 0x%0lx currCtxId 0x%0lx]: initHandler\n", m_name.c_str(), m_wrkrId, getThreadIdStr().c_str(), getCuCtxId(), getCurrCuCtxId());

    commnTestInitMsgPayload const& initMsgPayload = *std::static_pointer_cast<commnTestInitMsgPayload>(shPtrPayload);

    m_thrdId = std::this_thread::get_id();
    setThrdProps();

    m_uqPtrTimeStartEvent = std::make_unique<cuphy::event>();

    m_pschDlUlSyncEvents.resize(m_nStrms);
    m_maxNumCbErrors.resize(m_nStrms);

    auto& cuStrmPrioMap = initMsgPayload.cuStrmPrioMap;
    m_cuStrmPrioPusch   = (cuStrmPrioMap.find("PUSCH") != cuStrmPrioMap.end()) ? cuStrmPrioMap.at("PUSCH") : 0;
    m_cuStrmPrioPusch2  = (cuStrmPrioMap.find("PUSCH2") != cuStrmPrioMap.end()) ? cuStrmPrioMap.at("PUSCH2") : 0;
    m_cuStrmPrioPdsch   = (cuStrmPrioMap.find("PDSCH") != cuStrmPrioMap.end()) ? cuStrmPrioMap.at("PDSCH") : 0;
    m_cuStrmPrioSrs     = (cuStrmPrioMap.find("SRS") != cuStrmPrioMap.end()) ? cuStrmPrioMap.at("SRS") : 0;
    m_cuStrmPrioPrach   = (cuStrmPrioMap.find("PRACH") != cuStrmPrioMap.end()) ? cuStrmPrioMap.at("PRACH") : 0;
    m_cuStrmPrioPucch   = (cuStrmPrioMap.find("PUCCH") != cuStrmPrioMap.end()) ? cuStrmPrioMap.at("PUCCH") : 0;
    m_cuStrmPrioPucch2  = (cuStrmPrioMap.find("PUCCH2") != cuStrmPrioMap.end()) ? cuStrmPrioMap.at("PUCCH2") : 0;
    m_cuStrmPrioPdcch   = (cuStrmPrioMap.find("PDCCH") != cuStrmPrioMap.end()) ? cuStrmPrioMap.at("PDCCH") : 0;
    m_cuStrmPrioSsb     = (cuStrmPrioMap.find("SSB") != cuStrmPrioMap.end()) ? cuStrmPrioMap.at("SSB") : 0;
    m_cuStrmPrioCsirs   = (cuStrmPrioMap.find("CSIRS") != cuStrmPrioMap.end()) ? cuStrmPrioMap.at("CSIRS") : 0;
    m_cuStrmPrioUlbfw   = (cuStrmPrioMap.find("ULBFW") != cuStrmPrioMap.end()) ? cuStrmPrioMap.at("ULBFW") : 0;
    m_cuStrmPrioDlbfw   = (cuStrmPrioMap.find("DLBFW") != cuStrmPrioMap.end()) ? cuStrmPrioMap.at("DLBFW") : 0;

    printf("CUDA stream prios: PUSCH %d, PUSCH2 %d, PDSCH %d, SRS %d, PRACH %d, PDCCH %d, PUCCH %d, PUCCH2 %d, SSB %d, CSIRS %d, ULBFW %d, DLBFW %d \n", m_cuStrmPrioPusch, m_cuStrmPrioPusch2, m_cuStrmPrioPdsch, m_cuStrmPrioSrs, m_cuStrmPrioPrach, m_cuStrmPrioPdcch, m_cuStrmPrioPucch, m_cuStrmPrioPucch2, m_cuStrmPrioSsb, m_cuStrmPrioCsirs, m_cuStrmPrioUlbfw, m_cuStrmPrioDlbfw);
    for(uint32_t strmIdx = 0; strmIdx < m_nStrms; ++strmIdx)
    {
        m_cuStrms.emplace_back(cudaStreamNonBlocking);
        m_stopEvents.emplace_back(cudaEventDisableTiming);
        m_stop2Events.emplace_back(cudaEventDisableTiming);

        for(uint32_t itrIdx = 0; itrIdx < m_nItrsPerStrm; ++itrIdx)
        {
            m_maxNumCbErrors[strmIdx].emplace_back(0);

            m_pschDlUlSyncEvents[strmIdx].emplace_back();
        }
    }
    // extra stop event for PUSCH1 in longPattern == 3 or longPattern == 6
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

void cuPHYTestWorker::csirsInitHandler(std::shared_ptr<void>& shPtrPayload)
{
    DEBUG_TRACE("%s id %d [tid %s][wrkrCtxId 0x%0lx currCtxId 0x%0lx]: csirsInitHandler\n", m_name.c_str(), m_wrkrId, getThreadIdStr().c_str(), getCuCtxId(), getCurrCuCtxId());

    // unpack message
    commnTestInitMsgPayload const& initMsgPayload   = *std::static_pointer_cast<commnTestInitMsgPayload>(shPtrPayload);
    std::vector<std::string>       inFileNamesCSIRS = initMsgPayload.inFileNames;

    if(inFileNamesCSIRS.size() == 0)
    {
        m_runCSIRS = false;
    }
    else
    {
        m_runCSIRS = true;

        int nCSIRSObjects = m_pdsch_group_cells ? 1 : m_nCSIRSCells;

        m_timeCSIRSSlotStartEvents.resize(m_csirs_nItrsPerStrm);
        m_timeCSIRSSlotEndEvents.resize(m_csirs_nItrsPerStrm);
        m_totRunTimeCSIRSItr.resize(m_csirs_nItrsPerStrm);
        m_totCSIRSStartTimes.resize(m_csirs_nItrsPerStrm);
        m_csirsTxPipes.resize(nCSIRSObjects);
        m_csirsTxStaticApiDataSets.resize(nCSIRSObjects);
        m_csirsTxDynamicApiDataSets.resize(nCSIRSObjects);

        // CSIRS runs in a single slot after PDCCH in both DDDSU and DDDSUUDDDD patterns; same with PDCCH in DDDSUUDDDD mMIMO pattern

        for(uint32_t strmIdx = 0; strmIdx < nCSIRSObjects; ++strmIdx)
        {
            m_CSIRSStopEvents.emplace_back(cudaEventDisableTiming);
            m_csirsTxStaticApiDataSets[strmIdx].reserve(1);
            m_csirsTxDynamicApiDataSets[strmIdx].reserve(1);
            m_cuStrmsCsirs.emplace_back(cudaStreamNonBlocking, m_cuStrmPrioCsirs);
        }

        for(uint32_t i = 0; i < m_nCSIRSCells; ++i)
        {
            if(m_pdsch_group_cells)
            {
                if(i == 0)
                {
                    m_csirsTxStaticApiDataSets[0].emplace_back(inFileNamesCSIRS[i], m_nCSIRSCells);
                    m_csirsTxPipes[0].emplace_back(m_csirsTxStaticApiDataSets[i][0].csirsStatPrms);
                }
                else
                {
                    m_csirsTxStaticApiDataSets[0][0].cumulativeUpdate(inFileNamesCSIRS[i]);
                }
            }
            else
            {
                m_csirsTxStaticApiDataSets[i].emplace_back(inFileNamesCSIRS[i]);
                m_csirsTxPipes[i].emplace_back(m_csirsTxStaticApiDataSets[i][0].csirsStatPrms);
            }
        }
    }
    if(initMsgPayload.rsp)
    {
        auto shPtrRspPayload      = std::make_shared<commnTestRspMsgPayload>();
        shPtrRspPayload->workerId = m_wrkrId;

        auto shPtrRsp = std::make_shared<testWrkrRspMsg>(CUPHY_TEST_WRKR_RSP_MSG_CSIRS_INIT, m_wrkrId, shPtrRspPayload);
        m_shPtrRspQ->send(shPtrRsp);
    }
}

void cuPHYTestWorker::csirsSetupHandler(std::shared_ptr<void>& shPtrPayload)
{
    DEBUG_TRACE("%s id %d [tid %s][wrkrCtxId 0x%0lx currCtxId 0x%0lx]: csirsSetupHandler\n", m_name.c_str(), m_wrkrId, getThreadIdStr().c_str(), getCuCtxId(), getCurrCuCtxId());

    // unpack message
    commnTestInitMsgPayload const& initMsgPayload   = *std::static_pointer_cast<commnTestInitMsgPayload>(shPtrPayload);
    std::vector<std::string>       inFileNamesCSIRS = initMsgPayload.inFileNames;

    if(inFileNamesCSIRS.size() == 0)
    {
        m_runCSIRS = false;
        return;
    }
    m_runCSIRS = true;

    int nCSIRSObjects = m_pdsch_group_cells ? 1 : m_nCSIRSCells;

    m_csirsTxDynamicApiDataSets.clear();
    m_csirsTxDynamicApiDataSets.resize(nCSIRSObjects);
    for(uint32_t i = 0; i < m_nCSIRSCells; ++i)
    {
        if(m_pdsch_group_cells)
        {
            if(i == 0)
            {
                m_csirsTxDynamicApiDataSets[i].emplace_back(inFileNamesCSIRS[i], m_csirsTxStaticApiDataSets[i][0].csirsStatPrms.nMaxCellsPerSlot, m_cuStrmsCsirs[i].handle(), m_csirs_proc_mode);
            }
            else
            {
                // Nothing to cumulative update for the static parameters.
                // Update the dyanmic parameters
                m_csirsTxDynamicApiDataSets[0][0].cumulativeUpdate(inFileNamesCSIRS[i], m_cuStrmsCsirs[0].handle());
            }
        }
        else
        {
            m_csirsTxDynamicApiDataSets[i].emplace_back(inFileNamesCSIRS[i], m_csirsTxStaticApiDataSets[i][0].csirsStatPrms.nMaxCellsPerSlot, m_cuStrmsCsirs[i].handle(), m_csirs_proc_mode);
        }
    }
    for(int i = 0; i < nCSIRSObjects; i++)
    {
        m_csirsTxPipes[i][0].setup(m_csirsTxDynamicApiDataSets[i][0].csirs_dyn_params);
    }

    if(initMsgPayload.rsp)
    {
        auto shPtrRspPayload      = std::make_shared<commnTestRspMsgPayload>();
        shPtrRspPayload->workerId = m_wrkrId;

        auto shPtrRsp = std::make_shared<testWrkrRspMsg>(CUPHY_TEST_WRKR_RSP_MSG_CSIRS_SETUP, m_wrkrId, shPtrRspPayload);
        m_shPtrRspQ->send(shPtrRsp);
    }
}

void cuPHYTestWorker::ssbInitHandler(std::shared_ptr<void>& shPtrPayload)
{
    DEBUG_TRACE("%s id %d [tid %s][wrkrCtxId 0x%0lx currCtxId 0x%0lx]: ssbInitHandler\n", m_name.c_str(), m_wrkrId, getThreadIdStr().c_str(), getCuCtxId(), getCurrCuCtxId());

    // unpack message
    commnTestInitMsgPayload const& initMsgPayload = *std::static_pointer_cast<commnTestInitMsgPayload>(shPtrPayload);
    std::vector<std::string>       inFileNamesSSB = initMsgPayload.inFileNames;

    if(inFileNamesSSB.size() == 0)
    {
        m_runSSB = false;
    }
    else
    {
        m_runSSB = true;

        int nSSBObjects = m_pdsch_group_cells ? 1 : m_nSSBCells;

        m_timeSSBSlotStartEvents.resize(m_pbch_nItrsPerStrm);
        m_timeSSBSlotEndEvents.resize(m_pbch_nItrsPerStrm);
        m_totSSBStartTime.resize(m_pbch_nItrsPerStrm);
        m_totRunTimeSSBItr.resize(m_pbch_nItrsPerStrm);
        m_ssbTxPipes.resize(nSSBObjects);
        m_ssbTxStaticApiDataSets.resize(nSSBObjects);
        m_ssbTxDynamicApiDataSets.resize(nSSBObjects);

        for(uint32_t strmIdx = 0; strmIdx < nSSBObjects; ++strmIdx)
        {
            m_ssbTxStaticApiDataSets[strmIdx].reserve(1);
            m_ssbTxDynamicApiDataSets[strmIdx].reserve(1);
            m_cuStrmsSsb.emplace_back(cudaStreamNonBlocking, m_cuStrmPrioSsb);
        }

        for(uint32_t i = 0; i < m_nSSBCells; ++i)
        {
            if(m_pdsch_group_cells)
            {
                if(i == 0)
                {
                    m_ssbTxStaticApiDataSets[i].emplace_back(m_nSSBCells);
                    m_ssbTxPipes[i].emplace_back(m_ssbTxStaticApiDataSets[i][0].ssbStatPrms);
                }
                else
                {
                    // Nothing to cumulative update for the static parameters.
                    // Update the dyanmic parameters
                }
            }
            else
            {
                m_ssbTxStaticApiDataSets[i].emplace_back();
                m_ssbTxPipes[i].emplace_back(m_ssbTxStaticApiDataSets[i][0].ssbStatPrms);
            }
        }
    }
    if(initMsgPayload.rsp)
    {
        auto shPtrRspPayload      = std::make_shared<commnTestRspMsgPayload>();
        shPtrRspPayload->workerId = m_wrkrId;

        auto shPtrRsp = std::make_shared<testWrkrRspMsg>(CUPHY_TEST_WRKR_RSP_MSG_SSB_INIT, m_wrkrId, shPtrRspPayload);
        m_shPtrRspQ->send(shPtrRsp);
    }
}

void cuPHYTestWorker::ssbSetupHandler(std::shared_ptr<void>& shPtrPayload)
{
    DEBUG_TRACE("%s id %d [tid %s][wrkrCtxId 0x%0lx currCtxId 0x%0lx]: ssbSetupHandler\n", m_name.c_str(), m_wrkrId, getThreadIdStr().c_str(), getCuCtxId(), getCurrCuCtxId());

    // unpack message
    commnTestInitMsgPayload const& initMsgPayload = *std::static_pointer_cast<commnTestInitMsgPayload>(shPtrPayload);
    std::vector<std::string>       inFileNamesSSB = initMsgPayload.inFileNames;

    if(inFileNamesSSB.size() == 0)
    {
        m_runSSB = false;
    }
    else
    {
        m_runSSB = true;

        int nSSBObjects = m_pdsch_group_cells ? 1 : m_nSSBCells;

        m_ssbTxDynamicApiDataSets.clear();
        m_ssbTxDynamicApiDataSets.resize(nSSBObjects);
        for(uint32_t i = 0; i < m_nSSBCells; ++i)
        {
            if(m_pdsch_group_cells)
            {
                if(i == 0)
                {
                    m_ssbTxDynamicApiDataSets[i].emplace_back(inFileNamesSSB[i], m_ssbTxStaticApiDataSets[i][0].ssbStatPrms.nMaxCellsPerSlot, m_cuStrmsSsb[i].handle(), m_ssb_proc_mode);
                }
                else
                {
                    // Nothing to cumulative update for the static parameters.
                    // Update the dyanmic parameters
                    m_ssbTxDynamicApiDataSets[0][0].cumulativeUpdate(inFileNamesSSB[i], m_cuStrmsSsb[0].handle());
                }
            }
            else
            {
                m_ssbTxDynamicApiDataSets[i].emplace_back(inFileNamesSSB[i], m_ssbTxStaticApiDataSets[i][0].ssbStatPrms.nMaxCellsPerSlot, m_cuStrmsSsb[i].handle(), m_ssb_proc_mode);
            }
        }
        for(int i = 0; i < nSSBObjects; i++)
        {
            m_ssbTxPipes[i][0].setup(m_ssbTxDynamicApiDataSets[i][0].ssb_dyn_params);
        }
    }
    if(initMsgPayload.rsp)
    {
        auto shPtrRspPayload      = std::make_shared<commnTestRspMsgPayload>();
        shPtrRspPayload->workerId = m_wrkrId;

        auto shPtrRsp = std::make_shared<testWrkrRspMsg>(CUPHY_TEST_WRKR_RSP_MSG_SSB_SETUP, m_wrkrId, shPtrRspPayload);
        m_shPtrRspQ->send(shPtrRsp);
    }
}

void cuPHYTestWorker::pucchRxInitHandler(std::shared_ptr<void>& shPtrPayload)
{
    DEBUG_TRACE("%s id %d [tid %s][wrkrCtxId 0x%0lx currCtxId 0x%0lx]: pucchRxInitHandler\n", m_name.c_str(), m_wrkrId, getThreadIdStr().c_str(), getCuCtxId(), getCurrCuCtxId());

    // unpack message
    commnTestInitMsgPayload const& initMsgPayload   = *std::static_pointer_cast<commnTestInitMsgPayload>(shPtrPayload);
    std::vector<std::string>       inFileNamesPUCCH = initMsgPayload.inFileNames;
    //std::vector<std::vector<std::string>> pucchStreamFiles;
    std::vector<std::string> tempInput;

    if(m_pucch_group_cells)
    {
        if(m_uldlMode == 5 || m_uldlMode == 6)
        {
            m_nStrms_pucch = 2;
        }
        else
            m_nStrms_pucch = 1;
        tempInput.resize(inFileNamesPUCCH.size() / m_nStrms_pucch);
    }
    else
    {
        m_nStrms_pucch = inFileNamesPUCCH.size();
        tempInput.resize(1);
    }

    m_pucchRxPipes.resize(m_nStrms_pucch);
    m_pucchStaticDatasetVec.resize(m_nStrms_pucch);
    size_t MAX_N_F234_UCI = CUPHY_PUCCH_F2_MAX_UCI + CUPHY_PUCCH_F3_MAX_UCI;

    std::string outputFilename = std::string();

    // loop over streams
    for(uint32_t strmIdx = 0; strmIdx < m_nStrms_pucch; ++strmIdx)
    {
        if(m_longPattern && strmIdx >= m_nStrms_pucch / 2)
        {
            m_cuStrmsPucch.emplace_back(cudaStreamNonBlocking, m_cuStrmPrioPucch2);
        }
        else
        {
            m_cuStrmsPucch.emplace_back(cudaStreamNonBlocking, m_cuStrmPrioPucch);
        }

        CUDA_CHECK(cudaStreamSynchronize(m_cuStrmsPucch[strmIdx].handle()));

        m_PUCCHStopEvents.emplace_back(cudaEventDisableTiming);
        m_PUCCHStop2Events.emplace_back(cudaEventDisableTiming);

        for(uint32_t i = 0; i < m_nItrsPerStrm; ++i)
        {
            //uint32_t cellIdx = strmIdx + i * m_nPUCCHCells;

            if(m_pucch_group_cells)
            {
                tempInput.assign(inFileNamesPUCCH.begin() + i * (inFileNamesPUCCH.size() / m_nStrms_pucch), inFileNamesPUCCH.begin() + (i + 1) * (inFileNamesPUCCH.size() / m_nStrms_pucch));
            }
            else
            {
                tempInput.assign(1, inFileNamesPUCCH[strmIdx]);
            }

            CUDA_CHECK(cudaStreamSynchronize(m_cuStrmsPucch[strmIdx].handle()));

            m_pucchStaticDatasetVec[strmIdx].emplace_back(tempInput, m_cuStrmsPucch[strmIdx].handle(), outputFilename); // empty output filename for now
            apply_pucch_tv_overrides(m_pucchStaticDatasetVec[strmIdx][i].pucchStatPrms);

            m_pucchRxPipes[strmIdx].emplace_back(m_pucchStaticDatasetVec[strmIdx][i].pucchStatPrms, m_cuStrmsPucch[strmIdx].handle());
        }

        CUDA_CHECK(cudaStreamSynchronize(m_cuStrmsPucch[strmIdx].handle()));
    }

    if(initMsgPayload.rsp)
    {
        auto shPtrRspPayload      = std::make_shared<commnTestRspMsgPayload>();
        shPtrRspPayload->workerId = m_wrkrId;

        auto shPtrRsp = std::make_shared<testWrkrRspMsg>(CUPHY_TEST_WRKR_RSP_MSG_PUCCH_INIT, m_wrkrId, shPtrRspPayload);
        m_shPtrRspQ->send(shPtrRsp);
    }
}
void cuPHYTestWorker::prachInitHandler(std::shared_ptr<void>& shPtrPayload)
{
    DEBUG_TRACE("%s id %d [tid %s][wrkrCtxId 0x%0lx currCtxId 0x%0lx]: prachInitHandler\n", m_name.c_str(), m_wrkrId, getThreadIdStr().c_str(), getCuCtxId(), getCurrCuCtxId());

    // unpack message
    commnTestInitMsgPayload const& initMsgPayload   = *std::static_pointer_cast<commnTestInitMsgPayload>(shPtrPayload);
    std::vector<std::string>       inFileNamesPRACH = initMsgPayload.inFileNames;

    m_prachRxPipes.resize(m_nStrms_prach);
    m_prachDatasetVec.resize(m_nStrms_prach);

    // loop over streams (i.e., pipeline objects)
    for(uint32_t strmIdx = 0; strmIdx < m_nStrms_prach; ++strmIdx)
    {
        m_cuStrmsPrach.emplace_back(cudaStreamNonBlocking, m_cuStrmPrioPrach);
        m_PRACHStopEvents.emplace_back(cudaEventDisableTiming);

        m_prachDatasetVec[strmIdx].reserve(m_nItrsPerStrm);

        // loop over stream iterations; each iteration here is work
        // submitted on a given pipeline object in separate setup/run calls
        for(uint32_t itrIdx = 0; itrIdx < m_nItrsPerStrm; ++itrIdx)
        {
            // loop over all cells that need to be processed per pipeline in a single iteration.
            // This number should be 1, unless the m_prach_group_cells is true.
            for(uint32_t cellPerStrmIdx = 0; cellPerStrmIdx < m_nCellsPerStrm_prach; ++cellPerStrmIdx)
            {
                uint32_t cellIdx = strmIdx + itrIdx * m_nStrms_prach * m_nCellsPerStrm_prach + cellPerStrmIdx;

                if(cellPerStrmIdx == 0)
                {
                    m_prachDatasetVec[strmIdx].emplace_back(inFileNamesPRACH[cellIdx], m_cuStrmsPrach[strmIdx].handle(), m_prach_proc_mode, m_ref_check_prach);
                }
                else
                {
                    m_prachDatasetVec[strmIdx][itrIdx].cumulativeUpdate(inFileNamesPRACH[cellIdx], std::string(), m_cuStrmsPrach[strmIdx].handle());
                }
            }

            m_prachDatasetVec[strmIdx][itrIdx].finalize(m_cuStrmsPrach[strmIdx].handle());

            // initialize pipeline
            cuphyPrachStatPrms_t& prachStatPrms = m_prachDatasetVec[strmIdx][itrIdx].prachStatPrms;
            m_prachRxPipes[strmIdx].emplace_back(prachStatPrms);
        }
    }

    if(initMsgPayload.rsp)
    {
        auto shPtrRspPayload      = std::make_shared<commnTestRspMsgPayload>();
        shPtrRspPayload->workerId = m_wrkrId;

        auto shPtrRsp = std::make_shared<testWrkrRspMsg>(CUPHY_TEST_WRKR_RSP_MSG_PRACH_INIT, m_wrkrId, shPtrRspPayload);
        m_shPtrRspQ->send(shPtrRsp);
    }
}

void cuPHYTestWorker::pucchRxSetupHandler(std::shared_ptr<void>& shPtrPayload)
{
    DEBUG_TRACE("%s id %d [tid %s][wrkrCtxId 0x%0lx currCtxId 0x%0lx]: pucchRxSetupHandler\n", m_name.c_str(), m_wrkrId, getThreadIdStr().c_str(), getCuCtxId(), getCurrCuCtxId());

    // unpack message
    commnTestInitMsgPayload const& initMsgPayload   = *std::static_pointer_cast<commnTestInitMsgPayload>(shPtrPayload);
    std::vector<std::string>       inFileNamesPUCCH = initMsgPayload.inFileNames;
    std::vector<std::string>       tempInput(1);

    if(m_pucch_group_cells)
    {
        tempInput.resize(inFileNamesPUCCH.size() / m_nStrms_pucch);
    }
    else
        tempInput.resize(1);

    m_pucchDynDatasetVec.clear();
    m_pucchEvalDatasetVec.clear();
    m_pucchDynDatasetVec.resize(m_nStrms_pucch);
    m_pucchEvalDatasetVec.resize(m_nStrms_pucch);

    // loop over streams
    for(uint32_t strmIdx = 0; strmIdx < m_nStrms_pucch; ++strmIdx)
    {
        for(uint32_t i = 0; i < m_nItrsPerStrm; ++i)
        {
            uint32_t cellIdx = strmIdx + i * m_nStrms_pucch;

            // Load datasets
            if(m_pucch_group_cells)
            {
                tempInput.assign(inFileNamesPUCCH.begin() + i * (inFileNamesPUCCH.size() / m_nStrms_pucch), inFileNamesPUCCH.begin() + (i + 1) * (inFileNamesPUCCH.size() / m_nStrms_pucch));
            }
            else
            {
                tempInput.assign(1, inFileNamesPUCCH[strmIdx]);
            }
            //CUDA_CHECK(cudaStreamSynchronize(m_cuStrmsPucch[strmIdx].handle()));

            m_pucchDynDatasetVec[strmIdx].emplace_back(tempInput, m_cuStrmsPucch[strmIdx].handle(), m_pucch_proc_mode);
            m_pucchEvalDatasetVec[strmIdx].emplace_back(tempInput, m_cuStrmsPucch[strmIdx].handle());
            cuphyPucchDynPrms_t& pucchDynPrm                       = m_pucchDynDatasetVec[strmIdx][i].pucchDynPrm;
            m_pucchDynDatasetVec[strmIdx][i].pucchDynPrm.cuStream  = m_cuStrmsPucch[strmIdx].handle(); // save stream in dynamic parameters
            m_pucchDynDatasetVec[strmIdx][i].pucchDynPrm.cpuCopyOn = 1; // always enable D2H copy after processing
            
            // Setup pucch receiver object

            cuphyPucchBatchPrmHndl_t const batchPrmHndl = nullptr; // batchPrms currently un-used

            m_pucchRxPipes[strmIdx][i].setup(pucchDynPrm, batchPrmHndl);
        }

        CUDA_CHECK(cudaStreamSynchronize(m_cuStrmsPucch[strmIdx].handle()));
    }
    // send response
    if(initMsgPayload.rsp)
    {
        auto shPtrRspPayload      = std::make_shared<commnTestRspMsgPayload>();
        shPtrRspPayload->workerId = m_wrkrId;

        auto shPtrRsp = std::make_shared<testWrkrRspMsg>(CUPHY_TEST_WRKR_RSP_MSG_PUCCH_SETUP, m_wrkrId, shPtrRspPayload);
        m_shPtrRspQ->send(shPtrRsp);
    }
}

void cuPHYTestWorker::puschRxInitHandler(std::shared_ptr<void>& shPtrPayload)
{
    DEBUG_TRACE("%s id %d [tid %s][wrkrCtxId 0x%0lx currCtxId 0x%0lx]: puschRxInitHandler\n", m_name.c_str(), m_wrkrId, getThreadIdStr().c_str(), getCuCtxId(), getCurrCuCtxId());

    // unpack message
    //commnTestInitMsgPayload const& initMsgPayload     = *std::static_pointer_cast<commnTestInitMsgPayload>(shPtrPayload);

    cuPHYTestPuschRxInitMsgPayload const& initMsgPayload     = *std::static_pointer_cast<cuPHYTestPuschRxInitMsgPayload>(shPtrPayload);
    std::vector<std::string>              inFileNamesPuschRx = initMsgPayload.inFileNames;
    std::vector<std::string>              tempInput;

    if(m_pusch_group_cells)
    {
        if(m_uldlMode == 5 || m_uldlMode == 6)
        {
            m_nStrms = 2;
        }
        else
            m_nStrms = 1;
        tempInput.resize(inFileNamesPuschRx.size() / m_nStrms);
    }
    else
    {
        m_nStrms = inFileNamesPuschRx.size();
        tempInput.resize(1);
    }
    // resize
    m_puschRxStaticApiDataSets.resize(m_nStrms);
    m_puschRxPipes.resize(m_nStrms);
    m_maxNumCbErrors.resize(m_nStrms);

    // loop over streams
    for(uint32_t strmIdx = 0; strmIdx < m_nStrms; ++strmIdx)
    {
        m_puschRxStaticApiDataSets[strmIdx].reserve(m_nItrsPerStrm);
        // DDDSUUDDDD

        if(m_longPattern && strmIdx >= m_nStrms / 2)
        {
            m_cuStrmsPusch.emplace_back(cudaStreamNonBlocking, m_cuStrmPrioPusch2);
        }
        else
        {
            m_cuStrmsPusch.emplace_back(cudaStreamNonBlocking, m_cuStrmPrioPusch);
        }

        // loop over stream iterations
        for(uint32_t i = 0; i < m_nItrsPerStrm; ++i)
        {
            // uint32_t cellIdx = strmIdx + i * m_nStrms;
            if(m_pusch_group_cells)
            {
                tempInput.assign(inFileNamesPuschRx.begin() + i * (inFileNamesPuschRx.size() / m_nStrms), inFileNamesPuschRx.begin() + (i + 1) * (inFileNamesPuschRx.size() / m_nStrms));
                m_puschRxStaticApiDataSets[strmIdx].emplace_back(tempInput, m_cuStrmsPusch[strmIdx].handle(), std::string(), 1, 0, initMsgPayload.enableLdpcThroughputMode, &initMsgPayload.puschPrms,
                                                                 static_cast<cuphyPuschLdpcKernelLaunch_t>(m_ldpc_kernel_launch_mode));
            }
            else
            {
                tempInput.assign(1, inFileNamesPuschRx[strmIdx]);
                m_puschRxStaticApiDataSets[strmIdx].emplace_back(tempInput, m_cuStrmsPusch[strmIdx].handle(), std::string(), 1, 0, initMsgPayload.enableLdpcThroughputMode, &initMsgPayload.puschPrms,
                                                                 static_cast<cuphyPuschLdpcKernelLaunch_t>(m_ldpc_kernel_launch_mode));
            }
            // load static
            apply_pusch_tv_overrides(m_puschRxStaticApiDataSets[strmIdx][i].puschStatPrms);

            CUDA_CHECK(cudaStreamSynchronize(m_cuStrmsPusch[strmIdx].handle()));

            // initialize pipeline

            m_puschRxPipes[strmIdx].emplace_back(m_puschRxStaticApiDataSets[strmIdx][i].puschStatPrms, m_cuStrmsPusch[strmIdx].handle());

            CUDA_CHECK(cudaStreamSynchronize(m_cuStrmsPusch[strmIdx].handle()));

            // initialize error counter
            m_maxNumCbErrors[strmIdx].emplace_back(0);
        }
    }

    // send response
    if(initMsgPayload.rsp)
    {
        auto shPtrRspPayload      = std::make_shared<commnTestRspMsgPayload>();
        shPtrRspPayload->workerId = m_wrkrId;

        auto shPtrRsp = std::make_shared<testWrkrRspMsg>(CUPHY_TEST_WRKR_RSP_MSG_PUSCH_INIT, m_wrkrId, shPtrRspPayload);
        m_shPtrRspQ->send(shPtrRsp);
    }
}

void cuPHYTestWorker::srsInitHandler(std::shared_ptr<void>& shPtrPayload)
{
    DEBUG_TRACE("%s id %d [tid %s][wrkrCtxId 0x%0lx currCtxId 0x%0lx]: srsInitHandler\n", m_name.c_str(), m_wrkrId, getThreadIdStr().c_str(), getCuCtxId(), getCurrCuCtxId());

    // unpack message
    commnTestInitMsgPayload const& initMsgPayload = *std::static_pointer_cast<commnTestInitMsgPayload>(shPtrPayload);
    std::vector<std::string>       inFileNamesSRS = initMsgPayload.inFileNames;

    uint32_t nSRSStrms = 1;
    uint32_t nCells1   = m_runSRS2 ? (m_nSRSCells + 1) / 2 : m_nSRSCells;

    std::vector<std::string> inFileNamesSRS1(inFileNamesSRS.begin(), inFileNamesSRS.begin() + nCells1);
    std::vector<std::string> inFileNamesSRS2(inFileNamesSRS.begin() + nCells1, inFileNamesSRS.end());

    m_cuStrmsSrs.emplace_back(cudaStreamNonBlocking, m_cuStrmPrioSrs);
    m_SRSStopEvents.emplace_back(cudaEventDisableTiming);

    m_srsStaticApiDatasetVec.resize(1);
    m_srsStaticApiDatasetVec[0].emplace_back(inFileNamesSRS1, m_cuStrmsSrs[0].handle());
    apply_srs_tv_overrides(m_srsStaticApiDatasetVec[0][0].srsStatPrms);

    if(m_runSRS2)
    {
        m_srsStaticApiDatasetVec2.resize(1);
        m_srsStaticApiDatasetVec2[0].emplace_back(inFileNamesSRS2, m_cuStrmsSrs[0].handle());
        apply_srs_tv_overrides(m_srsStaticApiDatasetVec2[0][0].srsStatPrms);
    }

    CUDA_CHECK(cudaStreamSynchronize(m_cuStrmsSrs[0].handle()));

    cuphyStatus_t statusCreate = cuphyCreateSrsRx(&m_srsRxHndl, &m_srsStaticApiDatasetVec[0][0].srsStatPrms, m_cuStrmsSrs[0].handle());
    if(CUPHY_STATUS_SUCCESS != statusCreate) throw cuphy::cuphy_exception(statusCreate);

    if(m_runSRS2)
    {
        statusCreate = cuphyCreateSrsRx(&m_srsRxHndl2, &m_srsStaticApiDatasetVec2[0][0].srsStatPrms, m_cuStrmsSrs[0].handle());
        if(CUPHY_STATUS_SUCCESS != statusCreate) throw cuphy::cuphy_exception(statusCreate);
    }

    // send response
    if(initMsgPayload.rsp)
    {
        auto shPtrRspPayload      = std::make_shared<commnTestRspMsgPayload>();
        shPtrRspPayload->workerId = m_wrkrId;

        auto shPtrRsp = std::make_shared<testWrkrRspMsg>(CUPHY_TEST_WRKR_RSP_MSG_SRS_INIT, m_wrkrId, shPtrRspPayload);
        m_shPtrRspQ->send(shPtrRsp);
    }
}

void cuPHYTestWorker::pdcchTxInitHandler(std::shared_ptr<void>& shPtrPayload)
{
    DEBUG_TRACE("%s id %d [tid %s][wrkrCtxId 0x%0lx currCtxId 0x%0lx]: pdcchTxInitHandler\n", m_name.c_str(), m_wrkrId, getThreadIdStr().c_str(), getCuCtxId(), getCurrCuCtxId());

    // unpack message
    commnTestInitMsgPayload const& initMsgPayload     = *std::static_pointer_cast<commnTestInitMsgPayload>(shPtrPayload);
    std::vector<std::string>       inFileNamesPdcchTx = initMsgPayload.inFileNames;

    if(inFileNamesPdcchTx.size() == 0)
    {
        m_runPDCCH = false;
    }
    else
    {
        m_runPDCCH = true;

        m_timePdcchSlotStartEvents.resize(m_pdcch_nItrsPerStrm);
        m_timePdcchSlotEndEvents.resize(m_pdcch_nItrsPerStrm);
        m_totPdcchStartTimes.resize(m_pdcch_nItrsPerStrm);
        m_totRunTimePdcchItr.resize(m_pdcch_nItrsPerStrm);
        // resize
        m_pdcchTxStaticApiDataSets.resize(m_nStrms_pdcch);
        m_pdcchTxPipes.resize(m_nStrms_pdcch);
        //printf("# streams %d for PDCCH (general is %d) with cells per stream %d\n", m_nStrms_pdcch, m_nStrms, m_nCellsPerStrm_pdcch);

        // loop over streams (i.e., pipeline objects)
        for(uint32_t strmIdx = 0; strmIdx < m_nStrms_pdcch; ++strmIdx)
        {
            m_pdcchTxStaticApiDataSets[strmIdx].reserve(m_nItrsPerStrm); //NOTE: pdcch datasets need to be reserved.
                                                                         // loop over stream iterations; each iteration here is work submitted on a given pipeline object in separate setup/run calls
                                                                         // FIXME Do we only read parameters for 1st "pattern"?
            m_cuStrmsPdcch.emplace_back(cudaStreamNonBlocking, m_cuStrmPrioPdcch);

            m_PDCCHStopEvents.emplace_back(cudaEventDisableTiming);
            // loop over stream iterations
            for(uint32_t itrIdx = 0; itrIdx < m_nSlotsPerPattern; ++itrIdx)
            {
                if(strmIdx == 0)
                    m_pdcchCsirsInterSlotEndEventVec.emplace_back(cudaEventDisableTiming);
            }

            for(uint32_t itrIdx = 0; itrIdx < m_nItrsPerStrm; ++itrIdx)
            {
                // loop over all cells that need to be processed per pipeline in a single iteration.
                // This number should be 1, unless the m_pdcch_group_cells is true.
                // FIXME Assuming that when m_nCellsPerStrm_pdcch != 1, m_nStrms_pdcch is 1. Confirm this holds for all uldl use cases.
                for(uint32_t cellPerStrmIdx = 0; cellPerStrmIdx < m_nCellsPerStrm_pdcch; ++cellPerStrmIdx)
                {
                    uint32_t cellIdx = strmIdx + itrIdx * m_nStrms_pdcch * m_nCellsPerStrm_pdcch + cellPerStrmIdx;
                    /*printf("PDCCH init handler, strmIdx %d, itrIdx %d and group_cells %d: cellIdx %d with name %s\n",
                       strmIdx,
                       itrIdx,
                       m_pdcch_group_cells,
                       cellIdx,
                       inFileNamesPdcchTx[cellIdx].c_str());*/

                    if(cellPerStrmIdx == 0)
                    {
                        m_pdcchTxStaticApiDataSets[strmIdx].emplace_back(m_nCellsPerStrm_pdcch); // Providing no argument would default to 1 max. cell
                    }
                    /*else
                    {
                        ; //cumulatively update m_pdcchTxStatApiDataSets[strmIdx][itrIdx] Nothing to update for now.
                    }*/
                }

                // intialize pipeline
                m_pdcchTxPipes[strmIdx].emplace_back(m_pdcchTxStaticApiDataSets[strmIdx][itrIdx].pdcchStatPrms);
                //FIXME shall I catch a exception and exit(1)?
            }
        }
    }

    // send response
    if(initMsgPayload.rsp)
    {
        // Send init done response
        auto shPtrRspPayload      = std::make_shared<commnTestRspMsgPayload>();
        shPtrRspPayload->workerId = m_wrkrId;

        auto shPtrRsp = std::make_shared<testWrkrRspMsg>(CUPHY_TEST_WRKR_RSP_MSG_PDCCH_INIT, m_wrkrId, shPtrRspPayload);
        m_shPtrRspQ->send(shPtrRsp);
    }
}
void cuPHYTestWorker::pdschTxInitHandler(std::shared_ptr<void>& shPtrPayload)
{
    DEBUG_TRACE("%s id %d [tid %s][wrkrCtxId 0x%0lx currCtxId 0x%0lx]: pdschTxInitHandler\n", m_name.c_str(), m_wrkrId, getThreadIdStr().c_str(), getCuCtxId(), getCurrCuCtxId());

    // unpack message
    cuPHYTestPdschTxInitMsgPayload const& initMsgPayload     = *std::static_pointer_cast<cuPHYTestPdschTxInitMsgPayload>(shPtrPayload);
    std::vector<std::string>              inFileNamesPdschTx = initMsgPayload.inFileNames;
    if(inFileNamesPdschTx.size() == 0)
    {
        m_runPDSCH = false;
    }
    else
    {
        m_runPDSCH = true;
        
        m_timePdschSlotEndEvents.resize(m_pdsch_nItrsPerStrm);
        m_pdschInterSlotStartEventVec.resize(m_nSlotsPerPattern); // PDSCH events are always provisioned by m_nSlotsPerPattern since it may serve as a timer for the testbench
        m_SlotBoundaryEventVec.resize(m_nSlotsPerPattern); // 500 us slot boundaries (no PDSCH delay); used by PDCCH/SSB
        m_totRunTimePdschItr.resize(m_pdsch_nItrsPerStrm);
        m_totPdschSlotStartTime.resize(m_pdsch_nItrsPerStrm);
        // resize
        m_pdschTxStaticApiDataSets.resize(m_nStrms_pdsch);
        m_pdschTxPipes.resize(m_nStrms_pdsch);
        //printf("# streams %d for PDSCH (general is %d) with cells per stream %d\n", m_nStrms_pdsch, m_nStrms, m_nCellsPerStrm_pdsch);
        // loop over streams (i.e., pipeline objects)

        m_GPUtimeDl_d = std::move(cuphy::buffer<uint64_t, cuphy::device_alloc>(1));

        for(uint32_t strmIdx = 0; strmIdx < m_nStrms_pdsch; ++strmIdx)
        {
            m_pdschTxStaticApiDataSets[strmIdx].reserve(m_nItrsPerStrm); //NOTE: pdsch datasets need to be reserved.
                                                                         // loop over stream iterations; each iteration here is work submitted on a given pipeline object in separate setup/run calls
                                                                         // FIXME Do we only read parameters for 1st "pattern"?
            m_cuStrmsPdsch.emplace_back(cudaStreamNonBlocking, m_cuStrmPrioPdsch);

            for(uint32_t itrIdx = 0; itrIdx < m_nItrsPerStrm; ++itrIdx)
            {
                // loop over all cells that need to be processed per pipeline in a single iteration.
                // This number should be 1, unless the m_pdsch_group_cells is true.
                // FIXME Assuming that when m_nCellsPerStrm_pdsch != 1, m_nStrms_pdsch is 1. Confirm this holds for all uldl use cases.
                for(uint32_t cellPerStrmIdx = 0; cellPerStrmIdx < m_nCellsPerStrm_pdsch; ++cellPerStrmIdx)
                {
                    uint32_t cellIdx = strmIdx + itrIdx * m_nStrms_pdsch * m_nCellsPerStrm_pdsch + cellPerStrmIdx;
                    /*printf("PDSCH init handler, strmIdx %d, itrIdx %d and group_cells %d: cellIdx %d with name %s\n",
                       strmIdx,
                       itrIdx,
                       m_pdsch_group_cells,
                       cellIdx,
                       inFileNamesPdschTx[cellIdx].c_str());*/
                    // load static
                    std::string outFileName = std::string();
                    if(cellPerStrmIdx == 0)
                    {
                        m_pdschTxStaticApiDataSets[strmIdx].emplace_back(inFileNamesPdschTx[cellIdx], outFileName, m_ref_check_pdsch, m_identical_ldpc_configs, m_cuStrmPrioPdsch, initMsgPayload.pdschPrms.maxNCbsPerTb, initMsgPayload.pdschPrms.maxNTbs, initMsgPayload.pdschPrms.maxNPrbs);
                    }
                    else
                    {
                        //Cumulatively update the static parameters for a given {pipeline, iteration}
                        //This else clause is exercised when multiple cells are grouped in a cellg group for a single pipeline object (PdschTx)
                        m_pdschTxStaticApiDataSets[strmIdx][itrIdx].cumulativeUpdate(inFileNamesPdschTx[cellIdx], outFileName, m_ref_check_pdsch, m_identical_ldpc_configs);
                    }
                }
                CUDA_CHECK(cudaStreamSynchronize(m_cuStrmsPdsch[strmIdx].handle()));

                // DBG print static parameters
                //m_pdschTxStaticApiDataSets[strmIdx][itrIdx].print();

                // intialize pipeline
                m_pdschTxPipes[strmIdx].emplace_back(m_pdschTxStaticApiDataSets[strmIdx][itrIdx].pdschStatPrms);
            }
        }
    }

    // send response
    if(initMsgPayload.rsp)
    {
        // Send init done response
        auto shPtrRspPayload      = std::make_shared<commnTestRspMsgPayload>();
        shPtrRspPayload->workerId = m_wrkrId;

        auto shPtrRsp = std::make_shared<testWrkrRspMsg>(CUPHY_TEST_WRKR_RSP_MSG_PDSCH_INIT, m_wrkrId, shPtrRspPayload);
        m_shPtrRspQ->send(shPtrRsp);
    }
}

void cuPHYTestWorker::pdcchTxSetupHandler(std::shared_ptr<void>& shPtrPayload)
{
    DEBUG_TRACE("%s id %d [tid %s][wrkrCtxId 0x%0lx currCtxId 0x%0lx]: PDCCHSetupHandler\n", m_name.c_str(), m_wrkrId, getThreadIdStr().c_str(), getCuCtxId(), getCurrCuCtxId());

    // unpack message
    commnTestInitMsgPayload const& initMsgPayload   = *std::static_pointer_cast<commnTestInitMsgPayload>(shPtrPayload);
    std::vector<std::string>       inFileNamesPDCCH = initMsgPayload.inFileNames;

    if(inFileNamesPDCCH.size() == 0)
        return;
    // resize

    m_runPDCCH = true;

    // reset
    m_pdcchTxDynamicApiDataSets.clear();
    m_pdcchTxDynamicApiDataSets.resize(m_nPDCCHCells);

    // loop over streams
    for(uint32_t strmIdx = 0; strmIdx < m_nStrms_pdcch; ++strmIdx)
    {
        m_pdcchTxDynamicApiDataSets[strmIdx].reserve(m_nItrsPerStrm);

        // loop over stream iterations
        for(uint32_t itrIdx = 0; itrIdx < m_nItrsPerStrm; ++itrIdx)
        {
            for(uint32_t cellPerStrmIdx = 0; cellPerStrmIdx < m_nCellsPerStrm_pdcch; ++cellPerStrmIdx)
            {
                uint32_t cellIdx = strmIdx + itrIdx * m_nStrms_pdcch * m_nCellsPerStrm_pdcch + cellPerStrmIdx;
                /*printf("PDCCH setup handler, strmIdx %d, itrIdx %d: cellIdx %d with name %s\n",
                       strmIdx,
                       itrIdx,
                       cellIdx,
                       inFileNamesPdcchTx[cellIdx].c_str());*/

                //uint32_t cellIdx = strmIdx + itrIdx * m_nPDCCHCells;
                //printf("strmIdx %d, itrIdx %d, cellIdx %d, m PDCCH cells %d\n", strmIdx, itrIdx, cellIdx, m_nPDCCHCells);

                if(cellPerStrmIdx == 0)
                {
                    m_pdcchTxDynamicApiDataSets[strmIdx].emplace_back(inFileNamesPDCCH[cellIdx], m_nCellsPerStrm_pdcch, m_cuStrmsPdcch[strmIdx].handle(), m_pdcch_proc_mode);
                }
                else
                {
                    //Cumulatively update the dynamic parameters for a given pipeline, iteration
                    m_pdcchTxDynamicApiDataSets[strmIdx][itrIdx].cumulativeUpdate(inFileNamesPDCCH[cellIdx], m_cuStrmsPdcch[strmIdx].handle());
                }
            }
            CUDA_CHECK(cudaStreamSynchronize(m_cuStrmsPdcch[strmIdx].handle())); //FIXME needed?

            // setup pipeline
            //FIXME exit on exception?
            m_pdcchTxPipes[strmIdx][itrIdx].setup(m_pdcchTxDynamicApiDataSets[strmIdx][itrIdx].pdcch_dyn_params);
            //printf("strmIdx %d itrIdx %d\n", strmIdx, itrIdx);
            //m_pdcchTxPipes[strmIdx][itrIdx].handle()->pipeline->printPdcchConfig(m_pdcchTxDynamicApiDataSets[strmIdx][itrIdx].pdcch_dyn_params);
            //FIXME reset output buffer? TODO
            CUDA_CHECK(cudaStreamSynchronize(m_cuStrmsPdcch[strmIdx].handle()));
        }
    }
    // send response
    if(initMsgPayload.rsp)
    {
        // Send init done response
        auto shPtrRspPayload      = std::make_shared<commnTestRspMsgPayload>();
        shPtrRspPayload->workerId = m_wrkrId;

        auto shPtrRsp = std::make_shared<testWrkrRspMsg>(CUPHY_TEST_WRKR_RSP_MSG_PDCCH_SETUP, m_wrkrId, shPtrRspPayload);
        m_shPtrRspQ->send(shPtrRsp);
    }
}

void cuPHYTestWorker::dlbfwInitHandler(std::shared_ptr<void>& shPtrPayload)
{
    DEBUG_TRACE("%s id %d [tid %s][wrkrCtxId 0x%0lx currCtxId 0x%0lx]: dlbfwInitHandler\n", m_name.c_str(), m_wrkrId, getThreadIdStr().c_str(), getCuCtxId(), getCurrCuCtxId());

    // unpack message
    commnTestInitMsgPayload const& initMsgPayload   = *std::static_pointer_cast<commnTestInitMsgPayload>(shPtrPayload);
    std::vector<std::string>       inFileNamesDLBFW = initMsgPayload.inFileNames;

    m_timeDlbfwSlotStartEvents.resize(m_dlbfw_nItrsPerStrm);
    m_timeDlbfwSlotEndEvents.resize(m_dlbfw_nItrsPerStrm);
    m_totDlbfwIterStartTime.resize(m_dlbfw_nItrsPerStrm);
    m_totRunTimeDlbfwItr.resize(m_dlbfw_nItrsPerStrm);
    m_dlbfwStaticApiDatasetVec.clear();
    m_dlbfwStaticApiDatasetVec.resize(m_nItrsPerStrm);
    m_dlbfwPipelineVec.clear();
    m_dlbfwPipelineVec.resize(m_nItrsPerStrm);
    m_cuStrmsDlbfw.emplace_back(cudaStreamNonBlocking, m_cuStrmPrioDlbfw);
    for(int32_t iSlot = 0; iSlot < m_nItrsPerStrm; ++iSlot)
    {
        std::vector<std::string> inFileNamesDLBFWSlot(inFileNamesDLBFW.begin() + iSlot * m_nDlbfwCells, inFileNamesDLBFW.begin() + (iSlot + 1) * m_nDlbfwCells);
        m_dlbfwStaticApiDatasetVec[iSlot].emplace_back(inFileNamesDLBFWSlot, m_cuStrmsDlbfw[0].handle());
        m_dlbfwPipelineVec[iSlot].emplace_back(m_dlbfwStaticApiDatasetVec[iSlot][0].bfwStatPrms, m_cuStrmsDlbfw[0].handle());
    }
    CUDA_CHECK(cudaStreamSynchronize(m_cuStrmsDlbfw[0].handle()));

    // send response
    if(initMsgPayload.rsp)
    {
        // Send init done response
        auto shPtrRspPayload      = std::make_shared<commnTestRspMsgPayload>();
        shPtrRspPayload->workerId = m_wrkrId;

        auto shPtrRsp = std::make_shared<testWrkrRspMsg>(CUPHY_TEST_WRKR_RSP_MSG_DLBFW_INIT, m_wrkrId, shPtrRspPayload);
        m_shPtrRspQ->send(shPtrRsp);
    }
}

void cuPHYTestWorker::dlbfwSetupHandler(std::shared_ptr<void>& shPtrPayload)
{
    DEBUG_TRACE("%s id %d [tid %s][wrkrCtxId 0x%0lx currCtxId 0x%0lx]: dlbfwSetupHandler\n", m_name.c_str(), m_wrkrId, getThreadIdStr().c_str(), getCuCtxId(), getCurrCuCtxId());

    // unpack message
    commnTestInitMsgPayload const& initMsgPayload   = *std::static_pointer_cast<commnTestInitMsgPayload>(shPtrPayload);
    std::vector<std::string>       inFileNamesDLBFW = initMsgPayload.inFileNames;
    // resize

    uint64_t procModeBmsk = m_pdsch_proc_mode;

    m_dlbfwDynamicApiDatasetVec.clear();
    m_dlbfwEvalDatasetVec.clear();
    m_dlbfwDynamicApiDatasetVec.resize(m_nItrsPerStrm);
    m_dlbfwEvalDatasetVec.resize(m_nItrsPerStrm);

    for(int32_t iSlot = 0; iSlot < m_nItrsPerStrm; ++iSlot)
    {
        // set datasets and pipeline
        std::vector<std::string> inFileNamesDLBFWSlot(inFileNamesDLBFW.begin() + iSlot * m_nDlbfwCells, inFileNamesDLBFW.begin() + (iSlot + 1) * m_nDlbfwCells);

        m_dlbfwDynamicApiDatasetVec[iSlot].emplace_back(inFileNamesDLBFWSlot, m_cuStrmsDlbfw[0].handle(), procModeBmsk, OutputBufferMode::Host, 1024);
        m_dlbfwEvalDatasetVec[iSlot].emplace_back(inFileNamesDLBFWSlot, m_cuStrmsDlbfw[0].handle());
        m_dlbfwPipelineVec[iSlot][0].setup(m_dlbfwDynamicApiDatasetVec[iSlot][0].bfwDynPrms);
    }

    CUDA_CHECK(cudaStreamSynchronize(m_cuStrmsDlbfw[0].handle()));

    // send response
    if(initMsgPayload.rsp)
    {
        // Send init done response
        auto shPtrRspPayload      = std::make_shared<commnTestRspMsgPayload>();
        shPtrRspPayload->workerId = m_wrkrId;

        auto shPtrRsp = std::make_shared<testWrkrRspMsg>(CUPHY_TEST_WRKR_RSP_MSG_DLBFW_SETUP, m_wrkrId, shPtrRspPayload);
        m_shPtrRspQ->send(shPtrRsp);
    }
}
void cuPHYTestWorker::ulbfwInitHandler(std::shared_ptr<void>& shPtrPayload)
{
    DEBUG_TRACE("%s id %d [tid %s][wrkrCtxId 0x%0lx currCtxId 0x%0lx]: ulbfwInitHandler\n", m_name.c_str(), m_wrkrId, getThreadIdStr().c_str(), getCuCtxId(), getCurrCuCtxId());

    // unpack message
    commnTestInitMsgPayload const& initMsgPayload   = *std::static_pointer_cast<commnTestInitMsgPayload>(shPtrPayload);
    std::vector<std::string>       inFileNamesULBFW = initMsgPayload.inFileNames;

    m_nUlbfwWorkloads = (m_uldlMode == 5 || m_uldlMode == 6) ? 2 : 1;

    m_ulbfwStaticApiDatasetVec.clear();
    m_ulbfwStaticApiDatasetVec.resize(m_nItrsPerStrm*m_nUlbfwWorkloads);
    m_ulbfwPipelineVec.clear();
    m_ulbfwPipelineVec.resize(m_nItrsPerStrm*m_nUlbfwWorkloads);

    m_cuStrmsUlbfw.emplace_back(cudaStreamNonBlocking, m_cuStrmPrioUlbfw);

    for(int32_t iSlot = 0; iSlot < m_nItrsPerStrm*m_nUlbfwWorkloads; ++iSlot)
    {
        std::vector<std::string> inFileNamesULBFWSlot(inFileNamesULBFW.begin() + iSlot * m_nUlbfwCells, inFileNamesULBFW.begin() + (iSlot + 1) * m_nUlbfwCells);
        m_ulbfwStaticApiDatasetVec[iSlot].emplace_back(inFileNamesULBFWSlot, m_cuStrmsUlbfw[0].handle());
        m_ulbfwPipelineVec[iSlot].emplace_back(m_ulbfwStaticApiDatasetVec[iSlot][0].bfwStatPrms, m_cuStrmsUlbfw[0].handle());
    }
    CUDA_CHECK(cudaStreamSynchronize(m_cuStrmsUlbfw[0].handle()));

    // send response
    if(initMsgPayload.rsp)
    {
        // Send init done response
        auto shPtrRspPayload      = std::make_shared<commnTestRspMsgPayload>();
        shPtrRspPayload->workerId = m_wrkrId;

        auto shPtrRsp = std::make_shared<testWrkrRspMsg>(CUPHY_TEST_WRKR_RSP_MSG_ULBFW_INIT, m_wrkrId, shPtrRspPayload);
        m_shPtrRspQ->send(shPtrRsp);
    }
}

void cuPHYTestWorker::ulbfwSetupHandler(std::shared_ptr<void>& shPtrPayload)
{
    DEBUG_TRACE("%s id %d [tid %s][wrkrCtxId 0x%0lx currCtxId 0x%0lx]: ulbfwSetupHandler\n", m_name.c_str(), m_wrkrId, getThreadIdStr().c_str(), getCuCtxId(), getCurrCuCtxId());

    // unpack message
    commnTestInitMsgPayload const& initMsgPayload   = *std::static_pointer_cast<commnTestInitMsgPayload>(shPtrPayload);
    std::vector<std::string>       inFileNamesULBFW = initMsgPayload.inFileNames;
    // resize

    uint64_t procModeBmsk = m_pdsch_proc_mode;

    m_ulbfwDynamicApiDatasetVec.clear();
    m_ulbfwEvalDatasetVec.clear();
    m_ulbfwDynamicApiDatasetVec.resize(m_nItrsPerStrm*m_nUlbfwWorkloads);
    m_ulbfwEvalDatasetVec.resize(m_nItrsPerStrm*m_nUlbfwWorkloads);

    for(int32_t iSlot = 0; iSlot < m_nItrsPerStrm*m_nUlbfwWorkloads; ++iSlot)
    {
        // set datasets and pipeline
        std::vector<std::string> inFileNamesULBFWSlot(inFileNamesULBFW.begin() + iSlot * m_nUlbfwCells, inFileNamesULBFW.begin() + (iSlot + 1) * m_nUlbfwCells);

        m_ulbfwDynamicApiDatasetVec[iSlot].emplace_back(inFileNamesULBFWSlot, m_cuStrmsUlbfw[0].handle(), procModeBmsk);
        m_ulbfwEvalDatasetVec[iSlot].emplace_back(inFileNamesULBFWSlot, m_cuStrmsUlbfw[0].handle());
        m_ulbfwPipelineVec[iSlot][0].setup(m_ulbfwDynamicApiDatasetVec[iSlot][0].bfwDynPrms);
    }

    CUDA_CHECK(cudaStreamSynchronize(m_cuStrmsUlbfw[0].handle()));

    // send response
    if(initMsgPayload.rsp)
    {
        // Send init done response
        auto shPtrRspPayload      = std::make_shared<commnTestRspMsgPayload>();
        shPtrRspPayload->workerId = m_wrkrId;

        auto shPtrRsp = std::make_shared<testWrkrRspMsg>(CUPHY_TEST_WRKR_RSP_MSG_ULBFW_SETUP, m_wrkrId, shPtrRspPayload);
        m_shPtrRspQ->send(shPtrRsp);
    }
}

void cuPHYTestWorker::srsSetupHandler(std::shared_ptr<void>& shPtrPayload)
{
    DEBUG_TRACE("%s id %d [tid %s][wrkrCtxId 0x%0lx currCtxId 0x%0lx]: srsSetupHandler\n", m_name.c_str(), m_wrkrId, getThreadIdStr().c_str(), getCuCtxId(), getCurrCuCtxId());

    // unpack message
    commnTestInitMsgPayload const& initMsgPayload = *std::static_pointer_cast<commnTestInitMsgPayload>(shPtrPayload);
    std::vector<std::string>       inFileNamesSRS = initMsgPayload.inFileNames;

    uint32_t nSRSStrms = 1;
    uint32_t nCells1   = m_runSRS2 ? (m_nSRSCells + 1) / 2 : m_nSRSCells;

    std::vector<std::string> inFileNamesSRS1(inFileNamesSRS.begin(), inFileNamesSRS.begin() + nCells1);
    std::vector<std::string> inFileNamesSRS2(inFileNamesSRS.begin() + nCells1, inFileNamesSRS.end());

    m_srsDynamicApiDatasetVec.clear();
    m_srsDynamicApiDatasetVec.resize(1);
    m_srsEvalDatasetVec.clear();
    m_srsEvalDatasetVec.resize(1);

    m_srsDynamicApiDatasetVec[0].emplace_back(inFileNamesSRS1, m_cuStrmsSrs[0].handle(), m_srs_proc_mode);
    m_srsEvalDatasetVec[0].emplace_back(inFileNamesSRS1, m_cuStrmsSrs[0].handle());

    if(m_runSRS2)
    {
        m_srsEvalDatasetVec2.clear();
        m_srsEvalDatasetVec2.resize(1);
        m_srsDynamicApiDatasetVec2.clear();
        m_srsDynamicApiDatasetVec2.resize(1);
        m_srsDynamicApiDatasetVec2[0].emplace_back(inFileNamesSRS2, m_cuStrmsSrs[0].handle());
        m_srsEvalDatasetVec2[0].emplace_back(inFileNamesSRS2, m_cuStrmsSrs[0].handle());
    }

    CUDA_CHECK(cudaStreamSynchronize(m_cuStrmsSrs[0].handle()));
    cuphySrsBatchPrmHndl_t const batchPrmHndl = nullptr; // batchPrms currently un-used

    m_srsDynamicApiDatasetVec[0][0].srsDynPrm.cpuCopyOn = 1; // always enable D2H copy after processing
    cuphyStatus_t statusSetup = cuphySetupSrsRx(m_srsRxHndl, &m_srsDynamicApiDatasetVec[0][0].srsDynPrm, batchPrmHndl);
    if(CUPHY_STATUS_SUCCESS != statusSetup) throw cuphy::cuphy_exception(statusSetup);
    if(m_runSRS2)
    {
        m_srsDynamicApiDatasetVec2[0][0].srsDynPrm.cpuCopyOn = 1; // always enable D2H copy after processing
        statusSetup = cuphySetupSrsRx(m_srsRxHndl2, &m_srsDynamicApiDatasetVec2[0][0].srsDynPrm, batchPrmHndl);
        if(CUPHY_STATUS_SUCCESS != statusSetup) throw cuphy::cuphy_exception(statusSetup);
    }

    // send response
    if(initMsgPayload.rsp)
    {
        // Send init done response
        auto shPtrRspPayload      = std::make_shared<commnTestRspMsgPayload>();
        shPtrRspPayload->workerId = m_wrkrId;

        auto shPtrRsp = std::make_shared<testWrkrRspMsg>(CUPHY_TEST_WRKR_RSP_MSG_SRS_SETUP, m_wrkrId, shPtrRspPayload);
        m_shPtrRspQ->send(shPtrRsp);
    }
}

void cuPHYTestWorker::prachSetupHandler(std::shared_ptr<void>& shPtrPayload)
{
    DEBUG_TRACE("%s id %d [tid %s][wrkrCtxId 0x%0lx currCtxId 0x%0lx]: PRACHSetupHandler\n", m_name.c_str(), m_wrkrId, getThreadIdStr().c_str(), getCuCtxId(), getCurrCuCtxId());

    // unpack message
    commnTestInitMsgPayload const& initMsgPayload   = *std::static_pointer_cast<commnTestInitMsgPayload>(shPtrPayload);
    std::vector<std::string>       inFileNamesPRACH = initMsgPayload.inFileNames;

    // resize

    if(inFileNamesPRACH.size() == 0)
        return;

    // loop over streams
    for(uint32_t strmIdx = 0; strmIdx < m_nStrms_prach; ++strmIdx)
    {
        // loop over stream iterations
        for(uint32_t itrIdx = 0; itrIdx < m_nItrsPerStrm; ++itrIdx)
        {
            cuphyPrachDynPrms_t& prachDynPrms = m_prachDatasetVec[strmIdx][itrIdx].prachDynPrms;
            m_prachRxPipes[strmIdx][itrIdx].setup(prachDynPrms);
        }

        CUDA_CHECK(cudaStreamSynchronize(m_cuStrmsPrach[strmIdx].handle()));
    }

    // send response
    if(initMsgPayload.rsp)
    {
        // Send init done response
        auto shPtrRspPayload      = std::make_shared<commnTestRspMsgPayload>();
        shPtrRspPayload->workerId = m_wrkrId;

        auto shPtrRsp = std::make_shared<testWrkrRspMsg>(CUPHY_TEST_WRKR_RSP_MSG_PRACH_SETUP, m_wrkrId, shPtrRspPayload);
        m_shPtrRspQ->send(shPtrRsp);
    }
}

void cuPHYTestWorker::deinitHandler(std::shared_ptr<void>& shPtrPayload)
{
    DEBUG_TRACE("%s id %d [tid %s][wrkrCtxId 0x%0lx currCtxId 0x%0lx]: deinitHandler\n", m_name.c_str(), m_wrkrId, getThreadIdStr().c_str(), getCuCtxId(), getCurrCuCtxId());

    cuPHYTestDeinitMsgPayload const& deinitMsgPayload = *std::static_pointer_cast<cuPHYTestDeinitMsgPayload>(shPtrPayload);

    // clean-up PRACH
    // We need an explcit clean-up here and not rely on implicit clean-up in destructor.
    // Destructor is called in main thread with CUDA primary context whereas PRACH pipeline and associated CUDA resources 
    // are created with CUDA sub-context in worker thread. We need to ensure CUDA resources are freed-up in same context  
    // in which they were created. Bug: http://nvbugs/3612084
    if(m_runPRACH)
    {
        m_prachRxPipes.clear();
        m_prachDatasetVec.clear();
    }

    if(deinitMsgPayload.rsp)
    {
        // Send deinit done response
        auto shPtrRspPayload      = std::make_shared<commnTestRspMsgPayload>();
        shPtrRspPayload->workerId = m_wrkrId;

        auto shPtrRsp = std::make_shared<testWrkrRspMsg>(CUPHY_TEST_WRKR_RSP_MSG_DEINIT, m_wrkrId, shPtrRspPayload);
        m_shPtrRspQ->send(shPtrRsp);
    }
}

void cuPHYTestWorker::exitHandler(std::shared_ptr<void>& shPtrPayload)
{
    DEBUG_TRACE("%s id %d [tid %s][wrkrCtxId 0x%0lx currCtxId 0x%0lx]: exitHandler\n", m_name.c_str(), m_wrkrId, getThreadIdStr().c_str(), getCuCtxId(), getCurrCuCtxId());

    commnTestExitMsgPayload const& exitMsgPayload = *std::static_pointer_cast<commnTestExitMsgPayload>(shPtrPayload);
    if(exitMsgPayload.rsp)
    {
        // Send deinit done response
        auto shPtrRspPayload      = std::make_shared<commnTestRspMsgPayload>();
        shPtrRspPayload->workerId = m_wrkrId;

        auto shPtrRsp = std::make_shared<testWrkrRspMsg>(COMMON_TEST_WRKR_RSP_MSG_EXIT, m_wrkrId, shPtrRspPayload);
        m_shPtrRspQ->send(shPtrRsp);
    }
}

void cuPHYTestWorker::puschRxSetupHandler(std::shared_ptr<void>& shPtrPayload)
{
    DEBUG_TRACE("%s id %d [tid %s][wrkrCtxId 0x%0lx currCtxId 0x%0lx]: puschRxSetupHandler\n", m_name.c_str(), m_wrkrId, getThreadIdStr().c_str(), getCuCtxId(), getCurrCuCtxId());

    // unpack message
    cuPHYTestPuschRxSetupMsgPayload&      puschRxSetupMsgPayload = *std::static_pointer_cast<cuPHYTestPuschRxSetupMsgPayload>(shPtrPayload);
    std::vector<std::string>              inFileNamesPuschRx     = puschRxSetupMsgPayload.inFileNamesPuschRx;
    bool                                  rsp                    = puschRxSetupMsgPayload.rsp;
    std::vector<std::string>              tempInput;

    if(m_pusch_group_cells)
    {
        tempInput.resize(inFileNamesPuschRx.size() / m_nStrms);
    }
    else
        tempInput.resize(1);

    if(m_runPUSCH)
    {
        // reset
        m_puschRxDynamicApiDataSets.clear();
        m_puschRxDynamicApiDataSets.resize(m_nStrms);
        m_puschRxEvalDataSets.clear();
        m_puschRxEvalDataSets.resize(m_nStrms);

        cuphyPuschBatchPrmHndl_t batchPrmHndl = nullptr;
        // loop over streams
        for(uint32_t strmIdx = 0; strmIdx < m_nStrms; ++strmIdx)
        {
            m_puschRxDynamicApiDataSets[strmIdx].reserve(m_nItrsPerStrm);
            m_puschRxEvalDataSets[strmIdx].reserve(m_nItrsPerStrm);

            // Loop over iterations
            for(uint32_t itrIdx = 0; itrIdx < m_nItrsPerStrm; ++itrIdx)
            {
                uint32_t cellIdx = strmIdx + itrIdx * m_nStrms;
                uint32_t procModeSubslotProcFlag = ((strmIdx < m_nStrms/2) ? m_puschProcModeSubslotProcFlag : m_pusch2ProcModeSubslotProcFlag) * 2; // PUSCH_RUN_SUB_SLOT_PROC is the second bit, hence we multiple 2 to apply to m_pusch_proc_mode 

                // Load datasets
                if(m_pusch_group_cells)
                {
                    tempInput.assign(inFileNamesPuschRx.begin() + itrIdx * (inFileNamesPuschRx.size() / m_nStrms), inFileNamesPuschRx.begin() + (itrIdx + 1) * (inFileNamesPuschRx.size() / m_nStrms));
                    m_puschRxDynamicApiDataSets[strmIdx].emplace_back(tempInput, m_cuStrmsPusch[strmIdx].handle(), m_pusch_proc_mode | procModeSubslotProcFlag, false, m_fp16Mode);
                    m_puschRxEvalDataSets[strmIdx].emplace_back(tempInput, m_cuStrmsPusch[strmIdx].handle());
                }
                else
                {
                    std::vector<std::string> v;
                    v.push_back(inFileNamesPuschRx[cellIdx]);
                    m_puschRxDynamicApiDataSets[strmIdx].emplace_back(v, m_cuStrmsPusch[strmIdx].handle(), m_pusch_proc_mode | procModeSubslotProcFlag, false, m_fp16Mode);
                    m_puschRxEvalDataSets[strmIdx].emplace_back(v, m_cuStrmsPusch[strmIdx].handle());
                }
                CUDA_CHECK(cudaStreamSynchronize(m_cuStrmsPusch[strmIdx].handle()));

                // initialize pipeline - phase 1
                m_puschRxDynamicApiDataSets[strmIdx][itrIdx].puschDynPrm.setupPhase = PUSCH_SETUP_PHASE_1;
                m_puschRxDynamicApiDataSets[strmIdx][itrIdx].puschDynPrm.cpuCopyOn  = 1; // always enable D2H copy after processing
                m_puschRxPipes[strmIdx][itrIdx].setup(m_puschRxDynamicApiDataSets[strmIdx][itrIdx].puschDynPrm, batchPrmHndl);

                // Allocate HARQ buffers based on the calculated requirements from setupPhase 1
                m_puschRxDynamicApiDataSets[strmIdx][itrIdx].EasyAllocHarqBuffers(m_cuStrmsPusch[strmIdx].handle());

                // initialize pipeline - phase 2
                m_puschRxDynamicApiDataSets[strmIdx][itrIdx].puschDynPrm.setupPhase = PUSCH_SETUP_PHASE_2;
                m_puschRxPipes[strmIdx][itrIdx].setup(m_puschRxDynamicApiDataSets[strmIdx][itrIdx].puschDynPrm, batchPrmHndl);
                CUDA_CHECK(cudaStreamSynchronize(m_cuStrmsPusch[strmIdx].handle()));

                // check if subslot processing is enabled in PUSCH
                if(procModeSubslotProcFlag > 0 && (m_puschRxDynamicApiDataSets[strmIdx][itrIdx].puschDynPrm.pDataOut->isEarlyHarqPresent == 0) && (m_puschRxDynamicApiDataSets[strmIdx][itrIdx].puschDynPrm.pDataOut->isFrontLoadedDmrsPresent == 0))
                {
                    NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT,  "Error: PUSCH{} pipeline disabled subslot processing (possibly due to no early harq or front-loaded-DMRS UEs) but testbench requires subslot processing", (strmIdx < m_nStrms/2) ? 1 : 2);
                }
            }
        }
    }
    if(puschRxSetupMsgPayload.rsp)
    {
        // Send setup done response
        auto shPtrRspPayload      = std::make_shared<commnTestRspMsgPayload>();
        shPtrRspPayload->workerId = m_wrkrId;

        auto shPtrRsp = std::make_shared<testWrkrRspMsg>(CUPHY_TEST_WRKR_RSP_MSG_PUSCH_SETUP, m_wrkrId, shPtrRspPayload);
        m_shPtrRspQ->send(shPtrRsp);
    }
}

void cuPHYTestWorker::pdschTxSetupHandler(std::shared_ptr<void>& shPtrPayload)
{
    DEBUG_TRACE("%s id %d [tid %s][wrkrCtxId 0x%0lx currCtxId 0x%0lx]: pdschTxSetupHandler\n", m_name.c_str(), m_wrkrId, getThreadIdStr().c_str(), getCuCtxId(), getCurrCuCtxId());

    // unpack message
    cuPHYTestPdschTxSetupMsgPayload& pdschTxSetupMsgPayload = *std::static_pointer_cast<cuPHYTestPdschTxSetupMsgPayload>(shPtrPayload);
    std::vector<std::string>         inFileNamesPdschTx     = pdschTxSetupMsgPayload.inFileNamesPdschTx;
    bool                             rsp                    = pdschTxSetupMsgPayload.rsp;

    if(m_runPDSCH)
    {
        // reset
        m_pdschTxDynamicApiDataSets.clear();
        m_pdschTxDynamicApiDataSets.resize(m_nStrms_pdsch);

#ifdef ENABLE_F01_STREAM_PRIO
        std::vector<cuphy::stream>& cuStrmsPdsch = m_cuStrmsPdsch;
#else
        std::vector<cuphy::stream>& cuStrmsPdsch = (4 != m_uldlMode) ? m_cuStrmsPdsch : m_cuStrms;
#endif // ENABLE_F01_STREAM_PRIO

        cuphyPdschBatchPrmHndl_t batchPrmHndl = nullptr;
        // loop over streams
        for(uint32_t strmIdx = 0; strmIdx < m_nStrms_pdsch; ++strmIdx)
        {
            m_pdschTxDynamicApiDataSets[strmIdx].reserve(m_nItrsPerStrm); // Note: pdsch datasets need to be resereved to prevent segFault.
            // Loop over iterations
            for(uint32_t itrIdx = 0; itrIdx < m_nItrsPerStrm; ++itrIdx)
            {
                cuphy::pdsch_tx* pdschTxPipe = &m_pdschTxPipes[strmIdx][itrIdx];

                // loop over all cells that need to be processed per pipeline in a single iteration.
                // This number should be 1, unless the m_pdsch_group_cells is true.
                // FIXME Assuming that when m_nCellsPerStrm_pdsch != 1, m_nStrms_pdsch is 1. Confirm this holds for all uldl use cases.
                for(uint32_t cellPerStrmIdx = 0; cellPerStrmIdx < m_nCellsPerStrm_pdsch; ++cellPerStrmIdx)
                {
                    uint32_t cellIdx = strmIdx + itrIdx * m_nStrms_pdsch * m_nCellsPerStrm_pdsch + cellPerStrmIdx;
                    /*printf("PDSCH setup handler, strmIdx %d, itrIdx %d: cellIdx %d with name %s\n",
                       strmIdx,
                       itrIdx,
                       cellIdx,
                       inFileNamesPdschTx[cellIdx].c_str());*/

                    // Load datasets
                    //if(cellPerStrmIdx == 0)
                    if(cellPerStrmIdx == 0)
                    {
                        m_pdschTxDynamicApiDataSets[strmIdx].emplace_back(inFileNamesPdschTx[cellIdx], m_nCellsPerStrm_pdsch, cuStrmsPdsch[strmIdx].handle(), m_pdsch_proc_mode, m_pdschTxStaticApiDataSets[strmIdx][itrIdx].pdschStatPrms);
                    }
                    else
                    {
                        //Cumulatively update the dynamic parameters for a given pipeline, iteration
                        m_pdschTxDynamicApiDataSets[strmIdx][itrIdx].cumulativeUpdate(inFileNamesPdschTx[cellIdx], cuStrmsPdsch[strmIdx].handle(), m_pdsch_proc_mode);
                    }

                    // Load datasets
                    // Update filename for ref_check
                    // Workaround as the filename used for ref. checks is stored in the static parameters. TODO change this in the future
                    updateFileNameMultipleCells(pdschTxPipe->handle(), cellPerStrmIdx, inFileNamesPdschTx[cellIdx].c_str());
                }
                CUDA_CHECK(cudaStreamSynchronize(cuStrmsPdsch[strmIdx].handle()));

                // DBG print dynamic parameters
                //m_pdschTxDynamicApiDataSets[strmIdx][itrIdx].print();

                // setup pipeline
                pdschTxPipe->setup(m_pdschTxDynamicApiDataSets[strmIdx][itrIdx].pdsch_dyn_params, batchPrmHndl);

                // reset output buffer
                m_pdschTxDynamicApiDataSets[strmIdx][itrIdx].resetOutputTensors(cuStrmsPdsch[strmIdx].handle());
                CUDA_CHECK(cudaStreamSynchronize(cuStrmsPdsch[strmIdx].handle()));
            }
        }
    }
    if(pdschTxSetupMsgPayload.rsp)
    {
        // Send setup done response
        auto shPtrRspPayload      = std::make_shared<commnTestRspMsgPayload>();
        shPtrRspPayload->workerId = m_wrkrId;

        auto shPtrRsp = std::make_shared<testWrkrRspMsg>(CUPHY_TEST_WRKR_RSP_MSG_PDSCH_SETUP, m_wrkrId, shPtrRspPayload);
        m_shPtrRspQ->send(shPtrRsp);
    }
}

void cuPHYTestWorker::runPUCCH(const cudaEvent_t& startEvent)
{
    // wait for start event

    CUDA_CHECK(cudaStreamWaitEvent(m_cuStrmsPucch[0].handle(), startEvent, 0));

    for(uint32_t strmIdx = 1; strmIdx < m_nStrms_pucch; ++strmIdx)
    {
        CUDA_CHECK(cudaStreamWaitEvent(m_cuStrmsPucch[strmIdx].handle(), startEvent, 0));
    }

    CUDA_CHECK(cudaEventRecord(m_uqPtrTimePUCCHStartEvent->handle(), m_cuStrmsPucch[0].handle()));

    // Loop over iterations
    for(uint32_t itrIdx = 0; itrIdx < m_nItrsPerStrm; ++itrIdx)
    {
#if USE_NVTX
        nvtxRangePush("PUCCH");
#endif
        for(uint32_t strmIdx = 0; strmIdx < m_nStrms_pucch; ++strmIdx)
        {
            // run
            uint64_t procModeBmsk = m_pucch_proc_mode;
            m_pucchRxPipes[strmIdx][itrIdx].run(procModeBmsk);

            if(m_ref_check_pucch)
            {
                cuphyPucchDynPrms_t& pucchDynPrm = m_pucchDynDatasetVec[strmIdx][itrIdx].pucchDynPrm;
                CUDA_CHECK(cudaStreamSynchronize(m_cuStrmsPucch[strmIdx].handle()));
                int pucchErrors = m_pucchEvalDatasetVec[strmIdx][itrIdx].evalPucchRxPipeline(pucchDynPrm);

                if(pucchErrors != 0)
                {
                    NVLOGE_FMT(NVLOG_PUCCH, AERIAL_CUPHY_EVENT,  "PUCCH reference checks: {} errors", pucchErrors);
                    throw cuphy::cuphy_exception(CUPHY_STATUS_REF_MISMATCH);
                }
                else
                {
                    printf("PUCCH REFERENCE CHECK: PASSED!\n");
                }
            }

            // synch
            if(strmIdx != 0)
            {
                m_PUCCHStopEvents[strmIdx].record(m_cuStrmsPucch[strmIdx].handle());
                CUDA_CHECK(cudaStreamWaitEvent(m_cuStrmsPucch[0].handle(), m_PUCCHStopEvents[strmIdx].handle(), 0));
            }
        }
#if USE_NVTX
        nvtxRangePop();
#endif
    }
    // send GPU response message
    CUDA_CHECK(cudaEventRecord(m_uqPtrTimePUCCHEndEvent->handle(), m_cuStrmsPucch[0].handle()));
    m_PUCCHStopEvents[0].record(m_cuStrmsPucch[0].handle());
}

void cuPHYTestWorker::runPUCCH_U5_U6(const cudaEvent_t& startEvent, const cudaEvent_t& startEvent2)
{
    uint32_t pucch2SyncStrmId = m_nStrms_pucch / 2;
    uint64_t procModeBmsk     = m_pucch_proc_mode;
    uint32_t pucch1DelayUs    = (m_uldlMode == 6) ? g_start_delay_cfg_us.pucch_u6 : g_start_delay_cfg_us.pucch_u5;
    auto resolve_delay_us = [](int32_t yaml_delay_us, uint32_t default_delay_us) -> uint32_t {
        return yaml_delay_us >= 0 ? static_cast<uint32_t>(yaml_delay_us) : default_delay_us;
    };
    const uint32_t pucch2AfterPucch1DelayUs = resolve_delay_us(g_start_delay_cfg_us.pucch2_u6, 0);
    const bool nonPuschAnchor = g_start_delay_cfg_us.ul_anchor_from_yaml
        && g_start_delay_cfg_us.ul_anchor_mode != ul_anchor_mode_t::PUSCH;
    const bool puschCascaded = (m_longPattern > 3) && (m_longPattern != 7);

    // PUCCH1
    for(uint32_t strmIdx = 0; strmIdx < pucch2SyncStrmId; ++strmIdx)
    {
        CUDA_CHECK(cudaStreamWaitEvent(m_cuStrmsPucch[strmIdx].handle(), startEvent, 0));
        if(pucch1DelayUs)
            gpu_us_delay(pucch1DelayUs, 0, m_cuStrmsPucch[strmIdx].handle(), 1);
    }

    CUDA_CHECK(cudaEventRecord(m_uqPtrTimePUCCHStartEvent->handle(), m_cuStrmsPucch[0].handle()));
    // Note: m_nItrsPerStrm  represents the # of cells processed per slot sequentially
    // Note: m_nStrms represents the # of cells processed per slot concurrently

    // Loop over iterations

    for(uint32_t itrIdx = 0; itrIdx < m_nItrsPerStrm; ++itrIdx)
    {
#if USE_NVTX
        nvtxRangePush("PUCCH1");
#endif
        for(uint32_t strmIdx = 0; strmIdx < pucch2SyncStrmId; ++strmIdx)
        {
            // run
            m_pucchRxPipes[strmIdx][itrIdx].run(procModeBmsk);

            // cuphyPucchDynPrms_t& pucchDynPrm = m_pucchDynDatasetVec[strmIdx][itrIdx].pucchDynPrm;
            //  m_pucchEvalDatasetVec[strmIdx][itrIdx].evalPucchRxPipeline(pucchDynPrm);

            if(m_ref_check_pucch)
            {
                cuphyPucchDynPrms_t& pucchDynPrm = m_pucchDynDatasetVec[strmIdx][itrIdx].pucchDynPrm;
                CUDA_CHECK(cudaStreamSynchronize(m_cuStrmsPucch[strmIdx].handle()));
                int pucchErrors = m_pucchEvalDatasetVec[strmIdx][itrIdx].evalPucchRxPipeline(pucchDynPrm);

                if(pucchErrors != 0)
                {
                    NVLOGE_FMT(NVLOG_PUCCH, AERIAL_CUPHY_EVENT,  "PUCCH reference checks: {} errors", pucchErrors);
                    throw cuphy::cuphy_exception(CUPHY_STATUS_REF_MISMATCH);
                }
                else
                {
                    printf("PUCCH REFERENCE CHECK: PASSED!\n");
                }
            }

            // synch
            if(strmIdx != 0)
            {
                m_PUCCHStopEvents[strmIdx].record(m_cuStrmsPucch[strmIdx].handle());
                CUDA_CHECK(cudaStreamWaitEvent(m_cuStrmsPucch[0].handle(), m_PUCCHStopEvents[strmIdx].handle(), 0));
            }
        }
#if USE_NVTX
        nvtxRangePop();
#endif
    }

    CUDA_CHECK(cudaEventRecord(m_uqPtrTimePUCCHEndEvent->handle(), m_cuStrmsPucch[0].handle()));
    CUDA_CHECK(cudaEventRecord(m_PUCCHStopEvents[0].handle(), m_cuStrmsPucch[0].handle()));

    // PUCCH2
    if(nonPuschAnchor && puschCascaded)
    {
        // Non-PUSCH anchor + cascaded PUSCH: PUCCH2 delay is relative to PUCCH1 completion.
        CUDA_CHECK(cudaStreamWaitEvent(m_cuStrmsPucch[pucch2SyncStrmId].handle(), m_uqPtrTimePUCCHEndEvent->handle(), 0));
        if(pucch2AfterPucch1DelayUs > 0)
            gpu_us_delay(pucch2AfterPucch1DelayUs, 0, m_cuStrmsPucch[pucch2SyncStrmId].handle(), 1);
        CUDA_CHECK(cudaEventRecord(m_uqPtrTimePUCCH2StartEvent->handle(), m_cuStrmsPucch[pucch2SyncStrmId].handle()));
        for(uint32_t strmIdx = pucch2SyncStrmId; strmIdx < m_nStrms_pucch; ++strmIdx)
        {
            CUDA_CHECK(cudaStreamWaitEvent(m_cuStrmsPucch[strmIdx].handle(), m_uqPtrTimePUCCH2StartEvent->handle(), 0));
        }
    }
    else
    {
        // Non-cascaded PUSCH: PUCCH2 remains relative to anchor start.
        // Legacy/default PUSCH anchor: keep startEvent2 behavior from PUSCH timeline.
        for(uint32_t strmIdx = pucch2SyncStrmId; strmIdx < m_nStrms_pucch; ++strmIdx)
        {
            CUDA_CHECK(cudaStreamWaitEvent(m_cuStrmsPucch[strmIdx].handle(), startEvent2, 0));
        }
        if(nonPuschAnchor && pucch2AfterPucch1DelayUs > 0)
            gpu_us_delay(pucch2AfterPucch1DelayUs, 0, m_cuStrmsPucch[pucch2SyncStrmId].handle(), 1);
        CUDA_CHECK(cudaEventRecord(m_uqPtrTimePUCCH2StartEvent->handle(), m_cuStrmsPucch[pucch2SyncStrmId].handle()));
    }

    // Note: m_nItrsPerStrm  represents the # of cells processed per slot sequentially
    // Note: m_nStrms represents the # of cells processed per slot concurrently

    // Loop over iterations
    for(uint32_t itrIdx = 0; itrIdx < m_nItrsPerStrm; ++itrIdx)
    {
#if USE_NVTX
        nvtxRangePush("PUCCH2");
#endif
        for(uint32_t strmIdx = pucch2SyncStrmId; strmIdx < m_nStrms_pucch; ++strmIdx)
        {
            // run
            m_pucchRxPipes[strmIdx][itrIdx].run(procModeBmsk);

            // cuphyPucchDynPrms_t& pucchDynPrm = m_pucchDynDatasetVec[strmIdx][itrIdx].pucchDynPrm;
            // m_pucchEvalDatasetVec[strmIdx][itrIdx].evalPucchRxPipeline(pucchDynPrm);

            // synch
            if(strmIdx != pucch2SyncStrmId)
            {
                m_PUCCHStop2Events[strmIdx - pucch2SyncStrmId].record(m_cuStrmsPucch[strmIdx].handle());
                CUDA_CHECK(cudaStreamWaitEvent(m_cuStrmsPucch[pucch2SyncStrmId].handle(), m_PUCCHStop2Events[strmIdx - pucch2SyncStrmId].handle(), 0));
            }
        }
#if USE_NVTX
        nvtxRangePop();
#endif
    }

    CUDA_CHECK(cudaEventRecord(m_uqPtrTimePUCCH2EndEvent->handle(), m_cuStrmsPucch[pucch2SyncStrmId].handle()));
    CUDA_CHECK(cudaEventRecord(m_PUCCHStop2Events[0].handle(), m_cuStrmsPucch[pucch2SyncStrmId].handle()));
    CUDA_CHECK(cudaStreamWaitEvent(m_cuStrmsPucch[0].handle(), m_PUCCHStop2Events[0].handle(), 0));
}
void cuPHYTestWorker::runPRACH(const cudaEvent_t& startEvent)
{
    uint32_t prachDelayUs = (m_uldlMode == 6) ? g_start_delay_cfg_us.prach_u6 : g_start_delay_cfg_us.prach_u5;
    // wait for start event
    for(uint32_t strmIdx = 0; strmIdx < m_nStrms_prach; ++strmIdx)
    {
        CUDA_CHECK(cudaStreamWaitEvent(m_cuStrmsPrach[strmIdx].handle(), startEvent, 0));
        if(prachDelayUs)
            gpu_us_delay(prachDelayUs, 0, m_cuStrmsPrach[strmIdx].handle(), 1);
    }

    CUDA_CHECK(cudaEventRecord(m_uqPtrTimePRACHStartEvent->handle(), m_cuStrmsPrach[0].handle()));
    // Loop over iterations
    for(uint32_t itrIdx = 0; itrIdx < m_nItrsPerStrm; ++itrIdx)
    {
#if USE_NVTX
        nvtxRangePush("PRACH");
#endif
        for(uint32_t strmIdx = 0; strmIdx < m_nStrms_prach; ++strmIdx)
        {
            // run
            m_prachRxPipes[strmIdx][itrIdx].run();

            // synch
            if(strmIdx != 0)
            {
                m_PRACHStopEvents[strmIdx].record(m_cuStrmsPrach[strmIdx].handle());
                CUDA_CHECK(cudaStreamWaitEvent(m_cuStrmsPrach[0].handle(), m_PRACHStopEvents[strmIdx].handle(), 0));
            }
        }

        if(m_ref_check_prach)
        {
            int errors = 0;
            for(int i = 0; i < m_nStrms_prach; i++)
            {
                CUDA_CHECK(cudaStreamSynchronize(m_cuStrmsPrach[i].handle()));
                errors += m_prachDatasetVec[i][itrIdx].evaluateOutput();
            }

            if(errors != 0)
            {
                NVLOGE_FMT(NVLOG_PRACH, AERIAL_CUPHY_EVENT,  "PRACH reference checks: {} errors", errors);
                throw cuphy::cuphy_exception(CUPHY_STATUS_REF_MISMATCH);
            }
            else
            {
                NVLOGI_FMT(NVLOG_PRACH, "PRACH REFERENCE CHECK: PASSED");
            }
        }
#if USE_NVTX
        nvtxRangePop();
#endif
    }
    // send GPU response message
    CUDA_CHECK(cudaEventRecord(m_uqPtrTimePRACHEndEvent->handle(), m_cuStrmsPrach[0].handle()));
    m_PRACHStopEvents[0].record(m_cuStrmsPrach[0].handle());
}

void cuPHYTestWorker::runSRS1(const cudaEvent_t& startEvent)
{
    uint32_t nSRSStrms = 1;

    CUDA_CHECK(cudaStreamWaitEvent(m_cuStrmsSrs[0].handle(), startEvent, 0));
    
    // SRS always start after srsStartDelayUs (YAML-overridable).
    gpu_us_delay(m_uldlMode == 6 ? g_start_delay_cfg_us.srs_u6 : g_start_delay_cfg_us.srs_u5, m_gpuId, m_cuStrmsSrs[0].handle(), 1);
    
    CUDA_CHECK(cudaEventRecord(m_uqPtrSRSDelayStopEvent->handle(), m_cuStrmsSrs[0].handle()));

    // wait for start event
    for(uint32_t strmIdx = 0; strmIdx < nSRSStrms; ++strmIdx)
    {
        CUDA_CHECK(cudaStreamWaitEvent(m_cuStrmsSrs[strmIdx].handle(), m_uqPtrSRSDelayStopEvent->handle(), 0));
    }
    CUDA_CHECK(cudaEventRecord(m_uqPtrTimeSRSStartEvent->handle(), m_cuStrmsSrs[0].handle()));
    // Note: m_nItrsPerStrm  represents the # of cells processed per slot sequentially
    // Note: m_nStrms represents the # of cells processed per slot concurrently

    // Loop over iterations
    for(uint32_t itrIdx = 0; itrIdx < m_nItrsPerStrm; ++itrIdx)
    {
#if USE_NVTX
        nvtxRangePush("SRS1");
#endif
        for(uint32_t strmIdx = 0; strmIdx < nSRSStrms; ++strmIdx)
        {
            // run
            uint64_t procModeBmsk = m_srs_proc_mode;

            cuphyStatus_t statusRun = cuphyRunSrsRx(m_srsRxHndl, procModeBmsk);
            if(CUPHY_STATUS_SUCCESS != statusRun) throw cuphy::cuphy_exception(statusRun);

            // synch
            if(strmIdx != 0)
            {
                m_SRSStopEvents[strmIdx].record(m_cuStrmsSrs[strmIdx].handle());
                CUDA_CHECK(cudaStreamWaitEvent(m_cuStrmsSrs[0].handle(), m_SRSStopEvents[strmIdx].handle(), 0));
            }
            if(m_ref_check_srs)
            {
                CUDA_CHECK(cudaStreamSynchronize(m_cuStrmsSrs[strmIdx].handle()));
                m_srsEvalDatasetVec[strmIdx][itrIdx].evalSrsRx(m_srsDynamicApiDatasetVec[strmIdx][itrIdx].srsDynPrm, m_srsDynamicApiDatasetVec[strmIdx][itrIdx].tSrsChEstVec, m_srsDynamicApiDatasetVec[strmIdx][itrIdx].dataOut.pRbSnrBuffer, m_srsDynamicApiDatasetVec[strmIdx][itrIdx].dataOut.pSrsReports, m_cuStrmsSrs[strmIdx].handle());
            }
        }
#if USE_NVTX
        nvtxRangePop();
#endif
    }

    // send GPU response message
    CUDA_CHECK(cudaEventRecord(m_uqPtrSRSStopEvent->handle(), m_cuStrmsSrs[0].handle()));
    CUDA_CHECK(cudaEventRecord(m_uqPtrTimeSRSEndEvent->handle(), m_cuStrmsSrs[0].handle()));
}

void cuPHYTestWorker::runSRS2()
{
    //Run second SRS
    uint32_t nSRS2Strms = 1;
    // wait for start event

    // SRS always start after srsStartDelayUs (YAML-overridable).
    gpu_us_delay(m_uldlMode == 6 ? g_start_delay_cfg_us.srs2_u6 : g_start_delay_cfg_us.srs2_u5, m_gpuId, m_cuStrmsSrs[0].handle(), 1);

    for(uint32_t strmIdx = 0; strmIdx < nSRS2Strms; ++strmIdx)
    {
        CUDA_CHECK(cudaStreamWaitEvent(m_cuStrmsSrs[strmIdx].handle(), m_shPtrStopEvent->handle(), 0));
    }

    CUDA_CHECK(cudaEventRecord(m_uqPtrTimeSRS2StartEvent->handle(), m_cuStrmsSrs[0].handle()));
    // Note: m_nItrsPerStrm  represents the # of cells processed per slot sequentially
    // Note: m_nStrms represents the # of cells processed per slot concurrently
    // Loop over iterations
    for(uint32_t itrIdx = 0; itrIdx < m_nItrsPerStrm; ++itrIdx)
    {
#if USE_NVTX
        nvtxRangePush("SRS2");
#endif
        for(uint32_t strmIdx = 0; strmIdx < nSRS2Strms; ++strmIdx)
        {
            // run
            uint64_t procModeBmsk = m_srs_proc_mode;

            cuphyStatus_t statusRun = cuphyRunSrsRx(m_srsRxHndl2, procModeBmsk);
            if(CUPHY_STATUS_SUCCESS != statusRun) throw cuphy::cuphy_exception(statusRun);

            // synch
            if(strmIdx != 0)
            {
                m_SRSStopEvents[strmIdx].record(m_cuStrmsSrs[strmIdx].handle());
                CUDA_CHECK(cudaStreamWaitEvent(m_cuStrmsSrs[0].handle(), m_SRSStopEvents[strmIdx].handle(), 0));
            }
            if(m_ref_check_srs)
            {
                CUDA_CHECK(cudaStreamSynchronize(m_cuStrmsSrs[strmIdx].handle()));
                m_srsEvalDatasetVec2[strmIdx][itrIdx].evalSrsRx(m_srsDynamicApiDatasetVec2[strmIdx][itrIdx].srsDynPrm, m_srsDynamicApiDatasetVec2[strmIdx][itrIdx].tSrsChEstVec, m_srsDynamicApiDatasetVec2[strmIdx][itrIdx].dataOut.pRbSnrBuffer, m_srsDynamicApiDatasetVec2[strmIdx][itrIdx].dataOut.pSrsReports, m_cuStrmsSrs[strmIdx].handle());
            }
        }
#if USE_NVTX
        nvtxRangePop();
#endif
    }

    // send GPU response message
    CUDA_CHECK(cudaEventRecord(m_uqPtrSRS2StopEvent->handle(), m_cuStrmsSrs[0].handle()));
    CUDA_CHECK(cudaEventRecord(m_uqPtrTimeSRS2EndEvent->handle(), m_cuStrmsSrs[0].handle()));
}

void cuPHYTestWorker::runPUSCH_U5_U6(const cudaEvent_t& startEvent)
{
    uint32_t pusch2SyncStrmId   = m_nStrms / 2;
    uint32_t timelineSyncStrmId = m_longPattern > 6 ? 0 : pusch2SyncStrmId;
    uint64_t procModeBmsk       = m_pusch_proc_mode;
    auto resolve_delay_us = [](int32_t yaml_delay_us, uint32_t default_delay_us) -> uint32_t {
        return yaml_delay_us >= 0 ? static_cast<uint32_t>(yaml_delay_us) : default_delay_us;
    };
    const uint32_t configuredPusch1DelayUs = resolve_delay_us(g_start_delay_cfg_us.pusch_u5, puschStartDelayUsU5_);
    const uint32_t configuredPusch2DelayUs = resolve_delay_us(g_start_delay_cfg_us.pusch2_u5, 0);
    const uint32_t configuredPusch1DelayU6Us = resolve_delay_us(g_start_delay_cfg_us.pusch_u6, puschStartDelayUsU6_);
    const uint32_t configuredPusch2DelayU6Us = resolve_delay_us(g_start_delay_cfg_us.pusch2_u6, pusch2StartDelayUsU6_);
    const uint32_t configuredPusch2AfterPusch1DelayU6Us = resolve_delay_us(g_start_delay_cfg_us.pusch2_u6, 0);
    const uint32_t configuredPucch2DelayU6Us = resolve_delay_us(g_start_delay_cfg_us.pucch2_u6, pucch2StartDelayUsU6_);
    const uint32_t configuredPucch2DelayShortUs = resolve_delay_us(g_start_delay_cfg_us.pucch2_u6, 0);
    uint32_t delay1Us;
    uint32_t delay2Us;

    switch(m_longPattern)
    {
    case 1:
        delay1Us = 0;
        delay2Us = time_slot_duration;
        break;
    case 2:
        delay1Us = 0;
        delay2Us = time_slot_duration;
        break;
    case 3:
        delay1Us = time_slot_duration;
        delay2Us = time_slot_duration * 2;
        break;
    case 4:
        delay1Us = 0;
        delay2Us = 0;
        break;
    case 5:
        delay1Us = puschStartDelayUsU5_;
        delay2Us = 0;
        break;
    case 6:
        delay1Us = puschStartDelayUsU5_ + time_slot_duration;
        delay2Us = 0;
        break;
    case 7:
        delay1Us = g_start_delay_cfg_us.ulbfw_u6;
        delay2Us = 0;
        break;
    case 8:
        delay1Us = g_start_delay_cfg_us.ulbfw_u6;
        delay2Us = 0;
        break;
    default:
        delay1Us = 0;
        delay2Us = 0;
    }
    if(g_start_delay_cfg_us.pusch_u5 >= 0)
        delay1Us = configuredPusch1DelayUs;
    if(g_start_delay_cfg_us.pusch2_u5 >= 0)
        delay2Us = configuredPusch2DelayUs;
    const bool puschCascaded = (m_longPattern > 3) && (m_longPattern != 7);

    // hold all PUSCH streams until start event
    for(uint32_t strmIdx = 0; strmIdx < m_nStrms; ++strmIdx)
    {
        CUDA_CHECK(cudaStreamWaitEvent(m_cuStrmsPusch[strmIdx].handle(), startEvent, 0));
    }
    get_gpu_time(m_GPUtimeUl_d.addr(), m_cuStrmsPusch[timelineSyncStrmId].handle()); // record start time of ULBFW or PUSCH

    if(delay1Us)
        gpu_us_delay(delay1Us, 0, m_cuStrmsPusch[timelineSyncStrmId].handle(), 1);

    // ULBFW1
    if(m_runUlbfw)
    {
        CUDA_CHECK(cudaEventRecord(m_uqPtrUlbfwDelayStopEvent->handle(), m_cuStrmsPusch[timelineSyncStrmId].handle())); 
        CUDA_CHECK(cudaStreamWaitEvent(m_cuStrmsUlbfw[0].handle(), m_uqPtrUlbfwDelayStopEvent->handle(), 0));
#if USE_NVTX
        nvtxRangePush("ULBFW1");
#endif
        CUDA_CHECK(cudaEventRecord(m_uqPtrTimeULBfwStartEvent->handle(), m_cuStrmsUlbfw[0].handle()));
        for(uint32_t itrIdx = 0; itrIdx < m_nItrsPerStrm; ++itrIdx)
        {
            m_ulbfwPipelineVec[itrIdx][0].run(m_pdsch_proc_mode);
            if(m_ref_check_ulbfw)
            {
                float refCheckSnrThd = 30.0f;
                CUDA_CHECK(cudaStreamSynchronize(m_cuStrmsUlbfw[0].handle()));
                m_ulbfwEvalDatasetVec[itrIdx][0].bfwEvalCoefs(m_ulbfwStaticApiDatasetVec[itrIdx][0], m_ulbfwDynamicApiDatasetVec[itrIdx][0], m_cuStrmsUlbfw[0].handle(), refCheckSnrThd, true);
            }
        }
        CUDA_CHECK(cudaEventRecord(m_uqPtrTimeULBfwEndEvent->handle(), m_cuStrmsUlbfw[0].handle()));
#if USE_NVTX
        nvtxRangePop();
#endif
        CUDA_CHECK(cudaStreamWaitEvent(m_cuStrmsPusch[timelineSyncStrmId].handle(), m_uqPtrTimeULBfwEndEvent->handle(), 0));
    }

    // ULBFW2
    if(m_runUlbfw)
    {
        // delay between ULBFW1 start and ULBFW2 start
        if(m_longPattern > 6)
        {
            gpu_ns_delay_until(m_GPUtimeUl_d.addr(), static_cast<uint64_t>(g_start_delay_cfg_us.ulbfw2_u6) * NS_PER_US, m_cuStrmsPusch[timelineSyncStrmId].handle());
            CUDA_CHECK(cudaEventRecord(m_uqPtrUlbfw2DelayStopEvent->handle(), m_cuStrmsPusch[timelineSyncStrmId].handle()));
            CUDA_CHECK(cudaStreamWaitEvent(m_cuStrmsUlbfw[0].handle(), m_uqPtrUlbfw2DelayStopEvent->handle(), 0));
        }
        // else no need to delay
        
#if USE_NVTX
        nvtxRangePush("ULBFW2");
#endif
        CUDA_CHECK(cudaEventRecord(m_uqPtrTimeULBfw2StartEvent->handle(), m_cuStrmsUlbfw[0].handle()));
        for(uint32_t itrIdx = m_nItrsPerStrm; itrIdx < m_nItrsPerStrm*m_nUlbfwWorkloads; ++itrIdx)
        {
           m_ulbfwPipelineVec[itrIdx][0].run(m_pdsch_proc_mode);
           if(m_ref_check_ulbfw)
            {
                float refCheckSnrThd = 30.0f;
                CUDA_CHECK(cudaStreamSynchronize(m_cuStrmsUlbfw[0].handle()));
                m_ulbfwEvalDatasetVec[itrIdx][0].bfwEvalCoefs(m_ulbfwStaticApiDatasetVec[itrIdx][0], m_ulbfwDynamicApiDatasetVec[itrIdx][0], m_cuStrmsUlbfw[0].handle(), refCheckSnrThd, true);
            }
        }
        CUDA_CHECK(cudaEventRecord(m_uqPtrTimeULBfw2EndEvent->handle(), m_cuStrmsUlbfw[0].handle()));
#if USE_NVTX
        nvtxRangePop();
#endif
        CUDA_CHECK(cudaStreamWaitEvent(m_cuStrmsPusch[timelineSyncStrmId].handle(), m_uqPtrTimeULBfw2EndEvent->handle(), 0));
    }

    // PUSCH1
    // wait for delay
    if(m_longPattern > 6)
    {
        gpu_ns_delay_until(m_GPUtimeUl_d.addr(), static_cast<uint64_t>(configuredPusch1DelayU6Us) * NS_PER_US, m_cuStrmsPusch[timelineSyncStrmId].handle());
    }    
    
    CUDA_CHECK(cudaEventRecord(m_uqPtrPuschDelayStopEvent->handle(), m_cuStrmsPusch[timelineSyncStrmId].handle()));  
        
    for(uint32_t strmIdx = 0; strmIdx < pusch2SyncStrmId; ++strmIdx)
    {
        CUDA_CHECK(cudaStreamWaitEvent(m_cuStrmsPusch[strmIdx].handle(), m_uqPtrPuschDelayStopEvent->handle(), 0));
        // Uncomment line below to force PUSCH to wait for SRS to be finished
        //    CUDA_CHECK(cudaStreamWaitEvent(m_cuStrmsPusch[strmIdx].handle(), m_uqPtrSRSStopEvent->handle(), 0));
    }

    CUDA_CHECK(cudaEventRecord(m_uqPtrTimePUSCHStartEvent->handle(), m_cuStrmsPusch[0].handle()));
    // Note: m_nItrsPerStrm  represents the # of cells processed per slot sequentially
    // Note: m_nStrms represents the # of cells processed per slot concurrently

    // Loop over iterations
    for(uint32_t itrIdx = 0; itrIdx < m_nItrsPerStrm; ++itrIdx)
    {
#if USE_NVTX
        nvtxRangePush("PUSCH1");
#endif
        for(uint32_t strmIdx = 0; strmIdx < pusch2SyncStrmId; ++strmIdx)
        {
            // run
            m_puschRxPipes[strmIdx][itrIdx].run(PUSCH_RUN_ALL_PHASES);

            // synch
            if(strmIdx != 0)
            {
                m_stopEvents[strmIdx].record(m_cuStrmsPusch[strmIdx].handle());
                CUDA_CHECK(cudaStreamWaitEvent(m_cuStrmsPusch[0].handle(), m_stopEvents[strmIdx].handle(), 0));
            }
        }
#if USE_NVTX
        nvtxRangePop();
#endif
    }

    CUDA_CHECK(cudaEventRecord(m_uqPtrTimePUSCHEndEvent->handle(), m_cuStrmsPusch[0].handle()));
    CUDA_CHECK(cudaEventRecord(m_stopEvents[0].handle(), m_cuStrmsPusch[0].handle()));

    // PUSCH2
    if(m_longPattern > 6)
    {
        // Keep legacy PUCCH2 anchor event behavior when anchor is PUSCH/default.
        const bool legacyPuschAnchor = !g_start_delay_cfg_us.ul_anchor_from_yaml
            || g_start_delay_cfg_us.ul_anchor_mode == ul_anchor_mode_t::PUSCH;
        if(legacyPuschAnchor)
        {
            gpu_ns_delay_until(m_GPUtimeUl_d.addr(), static_cast<uint64_t>(configuredPucch2DelayU6Us) * NS_PER_US, m_cuStrmsPusch[timelineSyncStrmId].handle());
            CUDA_CHECK(cudaEventRecord(m_uqPtrPucch2DelayStopEvent->handle(), m_cuStrmsPusch[timelineSyncStrmId].handle()));
        }
        else
        {
            CUDA_CHECK(cudaStreamWaitEvent(m_cuStrmsPusch[timelineSyncStrmId].handle(), m_uqPtrTimePUSCHEndEvent->handle(), 0));
            CUDA_CHECK(cudaEventRecord(m_uqPtrPucch2DelayStopEvent->handle(), m_cuStrmsPusch[timelineSyncStrmId].handle()));
        }

        if(puschCascaded)
        {
            // Cascaded: PUSCH2 delay is after PUSCH1 completion.
            CUDA_CHECK(cudaStreamWaitEvent(m_cuStrmsPusch[timelineSyncStrmId].handle(), m_uqPtrTimePUSCHEndEvent->handle(), 0));
            if(configuredPusch2AfterPusch1DelayU6Us > 0)
                gpu_us_delay(configuredPusch2AfterPusch1DelayU6Us, 0, m_cuStrmsPusch[timelineSyncStrmId].handle(), 1);
        }
        else
        {
            // Non-cascaded: PUSCH2 delay is relative to anchor start.
            gpu_ns_delay_until(m_GPUtimeUl_d.addr(), static_cast<uint64_t>(configuredPusch2DelayU6Us) * NS_PER_US, m_cuStrmsPusch[timelineSyncStrmId].handle());
        }
        CUDA_CHECK(cudaEventRecord(m_uqPtrPusch2DelayStopEvent->handle(), m_cuStrmsPusch[timelineSyncStrmId].handle()));
    }
    else
    {
        if(puschCascaded)
        {
            // Cascaded: PUSCH2 delay is after PUSCH1 completion.
            CUDA_CHECK(cudaStreamWaitEvent(m_cuStrmsPusch[timelineSyncStrmId].handle(), m_uqPtrTimePUSCHEndEvent->handle(), 0));
            if(delay2Us)
                gpu_us_delay(delay2Us, 0, m_cuStrmsPusch[timelineSyncStrmId].handle(), 1);
        }
        else
        {
            // Non-cascaded: PUSCH2 delay remains relative to anchor start.
            if(delay2Us)
                gpu_us_delay(delay2Us, 0, m_cuStrmsPusch[pusch2SyncStrmId].handle(), 1);
        }

        // Keep legacy PUCCH2 anchor event behavior when anchor is PUSCH/default.
        const bool legacyPuschAnchor = !g_start_delay_cfg_us.ul_anchor_from_yaml
            || g_start_delay_cfg_us.ul_anchor_mode == ul_anchor_mode_t::PUSCH;
        if(legacyPuschAnchor && g_start_delay_cfg_us.pucch2_u6 >= 0 && configuredPucch2DelayShortUs > 0)
            gpu_us_delay(configuredPucch2DelayShortUs, 0, m_cuStrmsPusch[timelineSyncStrmId].handle(), 1);

        CUDA_CHECK(cudaEventRecord(m_uqPtrPucch2DelayStopEvent->handle(), m_cuStrmsPusch[timelineSyncStrmId].handle()));
        CUDA_CHECK(cudaEventRecord(m_uqPtrPusch2DelayStopEvent->handle(), m_cuStrmsPusch[timelineSyncStrmId].handle()));
    }

    for(uint32_t strmIdx = pusch2SyncStrmId; strmIdx < m_nStrms; ++strmIdx)
    {
        CUDA_CHECK(cudaStreamWaitEvent(m_cuStrmsPusch[strmIdx].handle(), m_uqPtrPusch2DelayStopEvent->handle(), 0));
    }

    CUDA_CHECK(cudaEventRecord(m_uqPtrTimePUSCH2StartEvent->handle(), m_cuStrmsPusch[pusch2SyncStrmId].handle()));
    // Note: m_nItrsPerStrm  represents the # of cells processed per slot sequentially
    // Note: m_nStrms represents the # of cells processed per slot concurrently

    // Loop over iterations
    for(uint32_t itrIdx = 0; itrIdx < m_nItrsPerStrm; ++itrIdx)
    {
#if USE_NVTX
        nvtxRangePush("PUSCH2");
#endif
        for(uint32_t strmIdx = pusch2SyncStrmId; strmIdx < m_nStrms; ++strmIdx)
        {
            // run
            m_puschRxPipes[strmIdx][itrIdx].run(PUSCH_RUN_ALL_PHASES);

            // synch
            if(strmIdx != pusch2SyncStrmId)
            {
                m_stop2Events[strmIdx - pusch2SyncStrmId].record(m_cuStrmsPusch[strmIdx].handle());
                CUDA_CHECK(cudaStreamWaitEvent(m_cuStrmsPusch[pusch2SyncStrmId].handle(), m_stop2Events[strmIdx - pusch2SyncStrmId].handle(), 0));
            }
        }
#if USE_NVTX
        nvtxRangePop();
#endif
    }

    CUDA_CHECK(cudaEventRecord(m_uqPtrTimePUSCH2EndEvent->handle(), m_cuStrmsPusch[pusch2SyncStrmId].handle()));
    CUDA_CHECK(cudaEventRecord(m_stop2Events[0].handle(), m_cuStrmsPusch[pusch2SyncStrmId].handle()));

    CUDA_CHECK(cudaStreamWaitEvent(m_cuStrmsPusch[0].handle(), m_stop2Events[0].handle(), 0));
    CUDA_CHECK(cudaEventRecord(m_shPtrStopEvent->handle(), m_cuStrmsPusch[0].handle()));

}
void cuPHYTestWorker::runPUSCH(const cudaEvent_t& startEvent)
{
    uint64_t procModeBmsk = m_pusch_proc_mode;
    gpu_us_delay(time_slot_duration, 0, m_cuStrmsPusch[0].handle(), 1);
    CUDA_CHECK(cudaEventRecord(m_uqPtrPuschDelayStopEvent->handle(), m_cuStrmsPusch[0].handle()));

    // ULBFW
    if(m_runUlbfw)
    {
#if USE_NVTX
        nvtxRangePush("ULBFW");
#endif
        CUDA_CHECK(cudaStreamWaitEvent(m_cuStrmsUlbfw[0].handle(), m_uqPtrPuschDelayStopEvent->handle(), 0));
        CUDA_CHECK(cudaEventRecord(m_uqPtrTimeULBfwStartEvent->handle(), m_cuStrmsUlbfw[0].handle()));
        for(uint32_t itrIdx = 0; itrIdx < m_nItrsPerStrm; ++itrIdx)
        {
           m_ulbfwPipelineVec[itrIdx][0].run(m_pdsch_proc_mode);
        }
        CUDA_CHECK(cudaEventRecord(m_uqPtrTimeULBfwEndEvent->handle(), m_cuStrmsUlbfw[0].handle()));
#if USE_NVTX
        nvtxRangePop();
#endif
    }

    // PUSCH1
    // wait for delay
    for(uint32_t strmIdx = 0; strmIdx < m_nStrms; ++strmIdx)
    {
        CUDA_CHECK(cudaStreamWaitEvent(m_cuStrmsPusch[strmIdx].handle(), m_runUlbfw ? m_uqPtrTimeULBfwEndEvent->handle() : m_uqPtrPuschDelayStopEvent->handle(), 0));
        // Uncomment line below to force PUSCH to wait for SRS to be finished
        //    CUDA_CHECK(cudaStreamWaitEvent(m_cuStrmsPusch[strmIdx].handle(), m_uqPtrSRSStopEvent->handle(), 0));
    }

    CUDA_CHECK(cudaEventRecord(m_uqPtrTimePUSCHStartEvent->handle(), m_cuStrmsPusch[0].handle()));
    // Note: m_nItrsPerStrm  represents the # of cells processed per slot sequentially
    // Note: m_nStrms represents the # of cells processed per slot concurrently

    // Loop over iterations
    for(uint32_t itrIdx = 0; itrIdx < m_nItrsPerStrm; ++itrIdx)
    {
#if USE_NVTX
        nvtxRangePush("PUSCH");
#endif
        for(uint32_t strmIdx = 0; strmIdx < m_nStrms; ++strmIdx)
        {
            // run
            m_puschRxPipes[strmIdx][itrIdx].run(PUSCH_RUN_ALL_PHASES);

            // synch
            if(strmIdx != 0)
            {
                m_stopEvents[strmIdx].record(m_cuStrmsPusch[strmIdx].handle());
                CUDA_CHECK(cudaStreamWaitEvent(m_cuStrmsPusch[strmIdx].handle(), m_stopEvents[strmIdx].handle(), 0));
            }
        }
#if USE_NVTX
        nvtxRangePop();
#endif
    }

    // send GPU response message
    CUDA_CHECK(cudaEventRecord(m_uqPtrTimePUSCHEndEvent->handle(), m_cuStrmsPusch[0].handle()));
    CUDA_CHECK(cudaEventRecord(m_shPtrStopEvent->handle(), m_cuStrmsPusch[0].handle()));
}


void cuPHYTestWorker::puschRxRunHandler(std::shared_ptr<void>& shPtrPayload)
{
    DEBUG_TRACE("%s id %d [tid %s][wrkrCtxId 0x%0lx currCtxId 0x%0lx]: puschRxRunHandler\n", m_name.c_str(), m_wrkrId, getThreadIdStr().c_str(), getCuCtxId(), getCurrCuCtxId());

    // unpack message
    cuPHYTestPuschRxRunMsgPayload& puschRxRunMsgPayload = *std::static_pointer_cast<cuPHYTestPuschRxRunMsgPayload>(shPtrPayload);

    cuphy::stream* syncStrm = nullptr;

    if(m_runPUSCH)
        syncStrm = &m_cuStrmsPusch[0];
    else if(m_runPRACH)
        syncStrm = &m_cuStrmsPrach[0];
    else if(m_runSRS)
        syncStrm = &m_cuStrmsSrs[0];
    else if(m_runPUCCH)
        syncStrm = &m_cuStrmsPucch[0];

    if(m_runPUSCH || m_runPRACH || m_runSRS || m_runPUCCH || m_runUlbfw)
    {
        CUDA_CHECK(cudaStreamWaitEvent(syncStrm->handle(), puschRxRunMsgPayload.startEvent, 0));

        CUDA_CHECK(cudaEventRecord(m_uqPtrTimeStartEvent->handle(), syncStrm->handle()));

        if(m_runSRS)
            runSRS1(puschRxRunMsgPayload.startEvent);

        if(m_runPUSCH)
        {
            if(m_longPattern)
            {
                runPUSCH_U5_U6(puschRxRunMsgPayload.startEvent);
            }
            else
                runPUSCH(puschRxRunMsgPayload.startEvent);
        }

        if(m_runPUCCH)
        {
            if(m_runPUSCH)
            {
                if(m_longPattern)
                    runPUCCH_U5_U6(getPuschStartEvent() ? getPuschStartEvent() : puschRxRunMsgPayload.startEvent, getPucch2DelayStopEvent());
                else
                    runPUCCH(getPuschStartEvent());
            }
            else
            {
                if(m_longPattern)
                    runPUCCH_U5_U6(puschRxRunMsgPayload.pucchStartEvent ? puschRxRunMsgPayload.pucchStartEvent : puschRxRunMsgPayload.startEvent, puschRxRunMsgPayload.prachStartEvent ? puschRxRunMsgPayload.prachStartEvent : puschRxRunMsgPayload.startEvent);
                else
                    runPUCCH(puschRxRunMsgPayload.pucchStartEvent ? puschRxRunMsgPayload.pucchStartEvent : puschRxRunMsgPayload.startEvent);
            }
        }

        if(m_runPRACH)
        {
            if(m_runPUSCH)
            {
                // External schedulers may explicitly provide PRACH anchor event.
                if(puschRxRunMsgPayload.prachStartEvent)
                {
                    runPRACH(puschRxRunMsgPayload.prachStartEvent);
                }
                // If PRACH delay from YAML: relative to PUSCH1 start; else relative to PUSCH1 end
                else if(g_start_delay_cfg_us.prach_delay_from_yaml)
                {
                    auto puschStartEvt = getPuschStartEvent();
                    runPRACH(puschStartEvt ? puschStartEvt : puschRxRunMsgPayload.startEvent);
                }
                else
                    runPRACH(getPusch1EndEvent());
            }
            else
            {
                runPRACH(puschRxRunMsgPayload.prachStartEvent ? puschRxRunMsgPayload.prachStartEvent : puschRxRunMsgPayload.startEvent);
            }
        }
    }
    if(m_runSRS2)
        runSRS2();

    if(m_runPRACH)
        CUDA_CHECK(cudaStreamWaitEvent(syncStrm->handle(), m_PRACHStopEvents[0].handle(), 0));
    if(m_runPUCCH)
        CUDA_CHECK(cudaStreamWaitEvent(syncStrm->handle(), m_PUCCHStopEvents[0].handle(), 0));
    if(m_runSRS)
        CUDA_CHECK(cudaStreamWaitEvent(syncStrm->handle(), m_uqPtrSRSStopEvent->handle(), 0));
    if(m_runSRS2)
        CUDA_CHECK(cudaStreamWaitEvent(syncStrm->handle(), m_uqPtrSRS2StopEvent->handle(), 0));

    if(syncStrm) // if any stream is available, otherwise no need to record event for measuring timing
    {
        CUDA_CHECK(cudaEventRecord(m_shPtrStopEvent->handle(), syncStrm->handle()));
    }

    // send CPU response message
    if(puschRxRunMsgPayload.rsp)
    {
        auto shPtrRspPayload            = std::make_shared<cuPHYTestPuschRxRunRspMsgPayload>();
        shPtrRspPayload->workerId       = m_wrkrId;
        shPtrRspPayload->shPtrStopEvent = m_shPtrStopEvent;

        auto shPtrRsp = std::make_shared<testWrkrRspMsg>(CUPHY_TEST_WRKR_RSP_MSG_PUSCH_RUN, m_wrkrId, shPtrRspPayload);
        m_shPtrRspQ->send(shPtrRsp);
    }
}

void cuPHYTestWorker::pschTxRxRunHandlerNoStrmPrio(std::shared_ptr<void>& shPtrPayload)
{
    DEBUG_TRACE("%s id %d [tid %s][wrkrCtxId 0x%0lx currCtxId 0x%0lx]: pschTxRxRunHandler\n", m_name.c_str(), m_wrkrId, getThreadIdStr().c_str(), getCuCtxId(), getCurrCuCtxId());

    // unpack message
    cuPHYTestPschTxRxRunMsgPayload& pschTxRxRunMsgPayload = *std::static_pointer_cast<cuPHYTestPschTxRxRunMsgPayload>(shPtrPayload);

    // wait for start event
    for(uint32_t strmIdx = 0; strmIdx < m_nStrms; ++strmIdx)
    {
        CUDA_CHECK(cudaStreamWaitEvent(m_cuStrms[strmIdx].handle(), pschTxRxRunMsgPayload.startEvent, 0));
    }
    CUDA_CHECK(cudaEventRecord(m_uqPtrTimeStartEvent->handle(), m_cuStrms[0].handle()));

    // Note: m_nItrsPerStrm  represents the # of cells processed per slot sequentially
    // Note: m_nStrms represents the # of cells processed per slot concurrently
    //printf("m_nStrms %d, m_nItrsPerStrm %d, PDSCH strms %d\n", m_nStrms, m_nItrsPerStrm, m_nStrms_pdsch);

    // Loop over stremas
    // FIXME: use default stream to record end of PDSCH
    for(uint32_t strmIdx = 0; strmIdx < m_nStrms; ++strmIdx)
    {
        for(uint32_t itrIdx = 0; itrIdx < m_nItrsPerStrm; ++itrIdx)
        {
            // run
            if(strmIdx < m_nStrms_pdsch)
            {
                m_pdschTxPipes[strmIdx][itrIdx].run(m_pdsch_proc_mode);
            }
            m_stop2Events[strmIdx].record(m_cuStrms[strmIdx].handle());
            m_puschRxPipes[strmIdx][itrIdx].run(PUSCH_RUN_ALL_PHASES);
        }

        // synch
        if(strmIdx != 0)
        {
            m_stopEvents[strmIdx].record(m_cuStrms[strmIdx].handle());
            CUDA_CHECK(cudaStreamWaitEvent(m_cuStrms[0].handle(), m_stopEvents[strmIdx].handle(), 0));
        }

        CUDA_CHECK(cudaStreamWaitEvent(0, m_stop2Events[strmIdx].handle(), 0));
    }

    CUDA_CHECK(cudaEventRecord(m_timePdschSlotEndEvents[0].handle(), 0));
    CUDA_CHECK(cudaEventRecord(m_uqPtrTimePUSCHEndEvent->handle(), m_cuStrms[0].handle()));
    // send GPU response message
    CUDA_CHECK(cudaEventRecord(m_shPtrStopEvent->handle(), m_cuStrms[0].handle()));
    m_pdschRunSlotIdx++;

    // send CPU response message
    if(pschTxRxRunMsgPayload.rsp)
    {
        auto shPtrRspPayload            = std::make_shared<cuPHYTestPschRunRspMsgPayload>();
        shPtrRspPayload->workerId       = m_wrkrId;
        shPtrRspPayload->shPtrStopEvent = m_shPtrStopEvent;

        auto shPtrRsp = std::make_shared<testWrkrRspMsg>(CUPHY_TEST_WRKR_RSP_MSG_PSCH_RUN, m_wrkrId, shPtrRspPayload);
        m_shPtrRspQ->send(shPtrRsp);
    }
}

void cuPHYTestWorker::runPDCCHItr(const cudaEvent_t& pdcchSlotStartEvent, uint32_t itrIdx)
{
    uint32_t pdcchDelayUs = 0;
    if(g_start_delay_cfg_us.pdcch_u6 >= 0)
    {
        pdcchDelayUs = static_cast<uint32_t>(g_start_delay_cfg_us.pdcch_u6);
    }
    else if(m_longPattern > 6)
    {
        pdcchDelayUs = pdschStartDelayNoBfwUsU6_;
    }
    for(uint32_t strmIdx = 0; strmIdx < m_nStrms_pdcch; ++strmIdx)
    {
        // hold all the streams (i.e cells) until startEvent for the slot
        CUDA_CHECK(cudaStreamWaitEvent(m_cuStrmsPdcch[strmIdx].handle(), pdcchSlotStartEvent, 0));
        if(pdcchDelayUs > 0)
            gpu_us_delay(pdcchDelayUs, 0, m_cuStrmsPdcch[strmIdx].handle(), 1);
    }

    if(m_pdcchSlotRunFlag[itrIdx])
    {
#if USE_NVTX
        nvtxRangePush("PDCCH");
#endif
        CUDA_CHECK(cudaEventRecord(m_timePdcchSlotStartEvents[m_pdcchRunSlotIdx].handle(), m_cuStrmsPdcch[0].handle())); // if PDCCH is run standalone record start event here
        for(uint32_t strmIdx = 0; strmIdx < m_nStrms_pdcch; ++strmIdx)
        {
            m_pdcchTxPipes[strmIdx][m_pdcchRunSlotIdx].run(0 /* unused proc. bitmask */);

            if(m_ref_check_pdcch)
            {
                // Providing no argument to refCheck (default is verbose=false) results in no prints during ref. checks
                int err = m_pdcchTxDynamicApiDataSets[strmIdx][m_pdcchRunSlotIdx].refCheck(true); //FIXME potentially move somewhere where it won't affect perf. Can call w/o arg.
                if(err != 0)
                {
                    // NVLOG will log number of mismatch in pdcchDynApiDataSET::refCheck
                    throw cuphy::cuphy_exception(CUPHY_STATUS_REF_MISMATCH);
                }
                else
                {
                    printf("PDCCH REFERENCE CHECK: PASSED!\n");
                }
            }

            // record end
            if(strmIdx != 0)
            {
                m_PDCCHStopEvents[strmIdx].record(m_cuStrmsPdcch[strmIdx].handle());
                CUDA_CHECK(cudaStreamWaitEvent(m_cuStrmsPdcch[0].handle(), m_PDCCHStopEvents[strmIdx].handle(), 0));
            }
        }
        CUDA_CHECK(cudaEventRecord(m_timePdcchSlotEndEvents[m_pdcchRunSlotIdx].handle(), m_cuStrmsPdcch[0].handle()));
        m_pdcchRunSlotIdx++;
#if USE_NVTX
    nvtxRangePop();
#endif
    }
    CUDA_CHECK(cudaEventRecord(m_PDCCHStopEvents[0].handle(), m_cuStrmsPdcch[0].handle()));

    // CSIRS
    if(m_runCSIRS && m_csirsSlotRunFlag[itrIdx])
    {
#if USE_NVTX
        nvtxRangePush("CSI-RS");
#endif
        int nCSIRSObjects = m_pdsch_group_cells ? 1 : m_nCSIRSCells;
        const bool hasPdcchCsirsOverride = (g_start_delay_cfg_us.pdcch_csirs_u6 >= 0);
        const uint32_t pdcchCsirsGapUs = hasPdcchCsirsOverride ? static_cast<uint32_t>(g_start_delay_cfg_us.pdcch_csirs_u6) : 0;
        for(uint32_t strmIdx = 0; strmIdx < nCSIRSObjects; ++strmIdx)
        {
            // If explicitly configured, apply requested behavior regardless of pattern:
            // 0 => start with PDCCH slot start, >0 => start after PDCCH plus extra gap.
            if(hasPdcchCsirsOverride)
            {
                if(pdcchCsirsGapUs == 0)
                {
                    CUDA_CHECK(cudaStreamWaitEvent(m_cuStrmsCsirs[strmIdx].handle(), pdcchSlotStartEvent, 0));
                }
                else
                {
                    CUDA_CHECK(cudaStreamWaitEvent(m_cuStrmsCsirs[strmIdx].handle(), m_PDCCHStopEvents[0].handle(), 0));
                    gpu_us_delay(pdcchCsirsGapUs, 0, m_cuStrmsCsirs[strmIdx].handle(), 1);
                }
            }
            else
            {
                // Legacy default behavior when not configured from YAML.
                if(m_longPattern > 6)
                {
                    CUDA_CHECK(cudaStreamWaitEvent(m_cuStrmsCsirs[strmIdx].handle(), pdcchSlotStartEvent, 0));
                }
                else
                {
                    CUDA_CHECK(cudaStreamWaitEvent(m_cuStrmsCsirs[strmIdx].handle(), m_PDCCHStopEvents[0].handle(), 0));
                }
            }

            if(strmIdx == 0) // if PDCCH is run standalone record start event here
            {
                CUDA_CHECK(cudaEventRecord(m_timeCSIRSSlotStartEvents[m_csirsRunSlotIdx].handle(), m_cuStrmsCsirs[0].handle()));
            }
            // run iteration and time
            // RUN CSIRS
            m_csirsTxPipes[strmIdx][0].run();
            // record end

            if(m_ref_check_csirs)
            {
                // Providing no argument to refCheck (default is verbose=false) results in no prints during ref. checks
                int err = m_csirsTxDynamicApiDataSets[strmIdx][0].refCheck(/*true*/); //FIXME potentially move somewhere where it won't affect perf.
                if(err != 0)
                {
                    // NVLOG will log number of mismatch in csirsDynApiDataset::refCheck
                    throw cuphy::cuphy_exception(CUPHY_STATUS_REF_MISMATCH);
                }
                else
                {
                    printf("CSIRS REFERENCE CHECK: PASSED!\n");
                }
            }

            if(strmIdx != 0)
            {
                m_CSIRSStopEvents[strmIdx].record(m_cuStrmsCsirs[strmIdx].handle());
                CUDA_CHECK(cudaStreamWaitEvent(m_cuStrmsCsirs[0].handle(), m_CSIRSStopEvents[strmIdx].handle(), 0));
            }
        }

        CUDA_CHECK(cudaEventRecord(m_timeCSIRSSlotEndEvents[m_csirsRunSlotIdx].handle(), m_cuStrmsCsirs[0].handle()));
        CUDA_CHECK(cudaEventRecord(m_CSIRSStopEvents[0].handle(), m_cuStrmsCsirs[0].handle()));
        CUDA_CHECK(cudaStreamWaitEvent(m_cuStrmsPdcch[0].handle(), m_CSIRSStopEvents[0].handle(), 0));
#if USE_NVTX
        nvtxRangePop();
#endif
        m_csirsRunSlotIdx++;
    }

    CUDA_CHECK(cudaEventRecord(m_pdcchCsirsInterSlotEndEventVec[itrIdx].handle(), m_cuStrmsPdcch[0].handle()));
}

void cuPHYTestWorker::runPDSCH_U5_3_6(std::shared_ptr<void>& shPtrPayload)
{
    // unpack message
    cuPHYTestPdschTxRunMsgPayload& pdschTxRunMsgPayload = *std::static_pointer_cast<cuPHYTestPdschTxRunMsgPayload>(shPtrPayload);
    uint64_t                       procModeBmsk         = m_pdsch_proc_mode;

    // Note: m_nItrsPerStrm  represents the # of cells processed per slot sequentially
    // Note: m_nStrms represents the # of cells processed per slot concurrently

    // Loop over iterations
    //printf("m_runPDSCH %d, m_runDlbfw %d, m_nDlbfwCells %d, m_nStrms_pdsch %d\n", m_runPDSCH, m_runDlbfw, m_nDlbfwCells, m_nStrms_pdsch);

    if(m_runPDSCH)
    {
        // find time slot duration in us
        // DDDSUUDDDD, run one Dlbfw first
        CUDA_CHECK(cudaStreamWaitEvent(m_cuStrmsPdsch[0].handle(), pdschTxRunMsgPayload.startEvent, 0));
        CUDA_CHECK(cudaStreamWaitEvent(m_cuStrmsDlbfw[0].handle(), pdschTxRunMsgPayload.startEvent, 0));
        get_gpu_time(m_GPUtimeDl_d.addr(), m_cuStrmsPdsch[0].handle()); // record start of time slot
        CUDA_CHECK(cudaEventRecord(m_uqPtrTimeStartEvent->handle(), m_cuStrmsPdsch[0].handle()));
        gpu_us_delay(g_start_delay_cfg_us.dlbfw_slot0_u5, 0, m_cuStrmsDlbfw[0].handle(), 1); // delay for first DLBFW (YAML-overridable)
#if USE_NVTX
        nvtxRangePush("DLBFW");
#endif
        CUDA_CHECK(cudaEventRecord(m_timeDlbfwSlotStartEvents[0].handle(), m_cuStrmsDlbfw[0].handle()));
        if(m_runDlbfw && m_pdschSlotRunFlag[0])
        {
            m_dlbfwPipelineVec[0][0].run(m_pdsch_proc_mode);
            if(m_ref_check_dlbfw)
            {
                float refCheckSnrThd = 30.0f;
                CUDA_CHECK(cudaStreamSynchronize(m_cuStrmsDlbfw[0].handle()));
                m_dlbfwEvalDatasetVec[0][0].bfwEvalCoefs(m_dlbfwStaticApiDatasetVec[0][0], m_dlbfwDynamicApiDatasetVec[0][0], m_cuStrmsDlbfw[0].handle(), refCheckSnrThd, true);
            }
        }
        CUDA_CHECK(cudaEventRecord(m_timeDlbfwSlotEndEvents[0].handle(), m_cuStrmsDlbfw[0].handle()));
#if USE_NVTX
        nvtxRangePop();
#endif
        // wait for DLBFW then var. delay
        CUDA_CHECK(cudaStreamWaitEvent(m_cuStrmsPdsch[0].handle(), m_timeDlbfwSlotEndEvents[0].handle(), 0));
        get_gpu_time(m_GPUtimeDl_d.addr(), m_cuStrmsPdsch[0].handle());
        CUDA_CHECK(cudaEventRecord(m_SlotBoundaryEventVec[0].handle(), m_cuStrmsPdsch[0].handle())); // slot 0 boundary at t=0
        // First 500 us slot boundary, then PDSCH delay, then record PDSCH slot start
        uint64_t           gpu_slot_start_time_offset_ns = static_cast<uint64_t>(0 + 1) * time_slot_duration * NS_PER_US;
        gpu_ns_delay_until(m_GPUtimeDl_d.addr(), gpu_slot_start_time_offset_ns, m_cuStrmsPdsch[0].handle());
        CUDA_CHECK(cudaEventRecord(m_SlotBoundaryEventVec[1].handle(), m_cuStrmsPdsch[0].handle())); // slot 1 boundary at 500 us
        get_gpu_time(m_GPUtimeDl_d.addr(), m_cuStrmsPdsch[0].handle());
        if(g_start_delay_cfg_us.pdsch_no_bfw_u6 > 0)
            gpu_us_delay(static_cast<uint32_t>(g_start_delay_cfg_us.pdsch_no_bfw_u6), 0, m_cuStrmsPdsch[0].handle(), 1);
        // record start of time slot
        CUDA_CHECK(cudaEventRecord(m_pdschInterSlotStartEventVec[0].handle(), m_cuStrmsPdsch[0].handle()));

        for(uint32_t itrIdx = 0; itrIdx < m_nSlotsPerPattern; ++itrIdx)
        {
            if(m_pdschSlotRunFlag[itrIdx])
            {
                for(uint32_t strmIdx = 0; strmIdx < m_nStrms_pdsch; ++strmIdx)
                {
#if USE_NVTX
                    nvtxRangePush("PDSCH");
#endif
                    // hold all the streams (i.e cells) until startEvent for the slot
                    CUDA_CHECK(cudaStreamWaitEvent(m_cuStrmsPdsch[strmIdx].handle(), m_pdschInterSlotStartEventVec[itrIdx].handle(), 0));
                    
                    m_pdschTxPipes[strmIdx][m_pdschRunSlotIdx].run(procModeBmsk);

                    // record end
                    if(strmIdx != 0)
                    {
                        m_stopEvents[strmIdx].record(m_cuStrmsPdsch[strmIdx].handle());
                        CUDA_CHECK(cudaStreamWaitEvent(m_cuStrmsPdsch[0].handle(), m_stopEvents[strmIdx].handle(), 0));
                    }
                    CUDA_CHECK(cudaEventRecord(m_timePdschSlotEndEvents[m_pdschRunSlotIdx].handle(), m_cuStrmsPdsch[0].handle()));
#if USE_NVTX
                    nvtxRangePop();
#endif
                }
            }

            CUDA_CHECK(cudaEventRecord(m_uqPtrPdschIterStopEvent->handle(), m_cuStrmsPdsch[0].handle()));

            if((m_pdschRunSlotIdx != m_nItrsPerStrm - 1) && (itrIdx != m_nSlotsPerPattern - 1))
            {
                if(m_runDlbfw && m_pdschSlotRunFlag[itrIdx])
                {
                    // wait for PDSCH cells to be processed
                    CUDA_CHECK(cudaStreamWaitEvent(m_cuStrmsDlbfw[0].handle(), m_uqPtrPdschIterStopEvent->handle(), 0));
#if USE_NVTX
                    nvtxRangePush("DLBFW");
#endif
                    CUDA_CHECK(cudaEventRecord(m_timeDlbfwSlotStartEvents[m_pdschRunSlotIdx + 1].handle(), m_cuStrmsDlbfw[0].handle())); // DLBFW calculated in slot itrIdx is for PDSCH in slot itrIdx+1
                    
                    m_dlbfwPipelineVec[m_pdschRunSlotIdx + 1][0].run(m_pdsch_proc_mode);
                    if(m_ref_check_dlbfw)
                    {
                        float refCheckSnrThd = 30.0f;
                        CUDA_CHECK(cudaStreamSynchronize(m_cuStrmsDlbfw[0].handle()));
                        m_dlbfwEvalDatasetVec[m_pdschRunSlotIdx + 1][0].bfwEvalCoefs(m_dlbfwStaticApiDatasetVec[m_pdschRunSlotIdx + 1][0], m_dlbfwDynamicApiDatasetVec[m_pdschRunSlotIdx + 1][0], m_cuStrmsDlbfw[0].handle(), refCheckSnrThd, true);
                    }

                    CUDA_CHECK(cudaEventRecord(m_timeDlbfwSlotEndEvents[m_pdschRunSlotIdx + 1].handle(), m_cuStrmsDlbfw[0].handle()));
                    CUDA_CHECK(cudaEventRecord(m_uqPtrDlbfwStopEvent->handle(), m_cuStrmsDlbfw[0].handle()));
                    
#if USE_NVTX
                    nvtxRangePop();
#endif
                    // wait for DLBFW then var. delay
                    CUDA_CHECK(cudaStreamWaitEvent(m_cuStrmsPdsch[0].handle(), m_uqPtrDlbfwStopEvent->handle(), 0));
                }
            }
            
            // wait for PDSCH cells to be processed 
            if(m_runPDCCH)
            {
                runPDCCHItr(m_SlotBoundaryEventVec[itrIdx].handle(), itrIdx);
                CUDA_CHECK(cudaStreamWaitEvent(m_cuStrmsPdsch[0].handle(), m_PDCCHStopEvents[0].handle(), 0));
            }
            else if(pdschTxRunMsgPayload.pdcchStopEventVec)
            {
                CUDA_CHECK(cudaStreamWaitEvent(m_cuStrmsPdsch[0].handle(), (*pdschTxRunMsgPayload.pdcchStopEventVec)[itrIdx].handle(), 0));
            }

            // timer for next slot boundary (500 us), then record slot boundary, then PDSCH delay, then record next PDSCH slot start
            // added for the last slot to preserve timeline in power measurement mode
            gpu_slot_start_time_offset_ns = static_cast<uint64_t>(itrIdx + 1) * time_slot_duration * NS_PER_US;
            gpu_ns_delay_until(m_GPUtimeDl_d.addr(), gpu_slot_start_time_offset_ns, m_cuStrmsPdsch[0].handle());
            if(itrIdx != m_nSlotsPerPattern - 1)
            {
                CUDA_CHECK(cudaEventRecord(m_SlotBoundaryEventVec[itrIdx + 1].handle(), m_cuStrmsPdsch[0].handle()));
                if(g_start_delay_cfg_us.pdsch_no_bfw_u6 > 0)
                    gpu_us_delay(static_cast<uint32_t>(g_start_delay_cfg_us.pdsch_no_bfw_u6), 0, m_cuStrmsPdsch[0].handle(), 1);
                CUDA_CHECK(cudaEventRecord(m_pdschInterSlotStartEventVec[itrIdx + 1].handle(), m_cuStrmsPdsch[0].handle()));
            }

            if(m_pdschSlotRunFlag[itrIdx])
            {
                m_pdschRunSlotIdx++;
            }
        }

        // send GPU response message
        CUDA_CHECK(cudaEventRecord(m_shPtrStopEvent->handle(), m_cuStrmsPdsch[0].handle()));
    }
}

void cuPHYTestWorker::runPDSCH_U6(std::shared_ptr<void>& shPtrPayload)
{
    // unpack message
    cuPHYTestPdschTxRunMsgPayload& pdschTxRunMsgPayload = *std::static_pointer_cast<cuPHYTestPdschTxRunMsgPayload>(shPtrPayload);
    uint64_t                       procModeBmsk         = m_pdsch_proc_mode;

    // Note: m_nItrsPerStrm  represents the # of cells processed per slot sequentially
    // Note: m_nStrms represents the # of cells processed per slot concurrently

    // Loop over iterations
    //printf("m_runPDSCH %d, m_runDlbfw %d, m_nDlbfwCells %d, m_nStrms_pdsch %d\n", m_runPDSCH, m_runDlbfw, m_nDlbfwCells, m_nStrms_pdsch);

    if(m_runPDSCH)
    {
        // find time slot duration in us
        // DDDSUUDDDD, run one Dlbfw first
        CUDA_CHECK(cudaStreamWaitEvent(m_cuStrmsPdsch[0].handle(), pdschTxRunMsgPayload.startEvent, 0));
        get_gpu_time(m_GPUtimeDl_d.addr(), m_cuStrmsPdsch[0].handle()); // record start of time slot
        CUDA_CHECK(cudaEventRecord(m_uqPtrTimeStartEvent->handle(), m_cuStrmsPdsch[0].handle()));
        CUDA_CHECK(cudaEventRecord(m_SlotBoundaryEventVec[0].handle(), m_cuStrmsPdsch[0].handle()));
        
        for(uint32_t itrIdx = 0; itrIdx < m_nSlotsPerPattern; ++itrIdx)
        {
            if(m_runDlbfw && m_pdschSlotRunFlag[itrIdx])
            {
                // wait for 500 us slot boundary
                CUDA_CHECK(cudaStreamWaitEvent(m_cuStrmsDlbfw[0].handle(), m_SlotBoundaryEventVec[itrIdx].handle(), 0));
                gpu_us_delay(g_start_delay_cfg_us.dlbfw_u6, 0, m_cuStrmsDlbfw[0].handle(), 1); // delay for DLBFW (YAML-overridable)
#if USE_NVTX
                nvtxRangePush("DLBFW");
#endif
                CUDA_CHECK(cudaEventRecord(m_timeDlbfwSlotStartEvents[m_pdschRunSlotIdx].handle(), m_cuStrmsDlbfw[0].handle())); // DLBFW start for PDSCH in slot itrIdx
                
                m_dlbfwPipelineVec[m_pdschRunSlotIdx][0].run(m_pdsch_proc_mode);
                if(m_ref_check_dlbfw)
                {
                    float refCheckSnrThd = 30.0f;
                    CUDA_CHECK(cudaStreamSynchronize(m_cuStrmsDlbfw[0].handle()));
                    m_dlbfwEvalDatasetVec[m_pdschRunSlotIdx][0].bfwEvalCoefs(m_dlbfwStaticApiDatasetVec[m_pdschRunSlotIdx][0], m_dlbfwDynamicApiDatasetVec[m_pdschRunSlotIdx][0], m_cuStrmsDlbfw[0].handle(), refCheckSnrThd, true);
                }

                CUDA_CHECK(cudaEventRecord(m_timeDlbfwSlotEndEvents[m_pdschRunSlotIdx].handle(), m_cuStrmsDlbfw[0].handle()));
                CUDA_CHECK(cudaEventRecord(m_uqPtrDlbfwStopEvent->handle(), m_cuStrmsDlbfw[0].handle()));
#if USE_NVTX
                nvtxRangePop();
#endif
                // wait for DLBFW then var. delay
                CUDA_CHECK(cudaStreamWaitEvent(m_cuStrmsPdsch[0].handle(), m_uqPtrDlbfwStopEvent->handle(), 0));
                
            }

            // record PDSCH stat time regardless of whether PDSCH is running or not
            // this event will be used to synchronize the start of PDCCH/CSI-RS/SSB
            // First wait for 500 us slot boundary, then add PDSCH delay, then record PDSCH slot start
            uint64_t           gpu_slot_start_time_offset_ns = static_cast<uint64_t>(itrIdx) * time_slot_duration * NS_PER_US;
            gpu_ns_delay_until(m_GPUtimeDl_d.addr(), gpu_slot_start_time_offset_ns, m_cuStrmsPdsch[0].handle());
            if(g_start_delay_cfg_us.pdsch_no_bfw_u6 > 0)
                gpu_us_delay(static_cast<uint32_t>(g_start_delay_cfg_us.pdsch_no_bfw_u6), 0, m_cuStrmsPdsch[0].handle(), 1);
            CUDA_CHECK(cudaEventRecord(m_pdschInterSlotStartEventVec[itrIdx].handle(), m_cuStrmsPdsch[0].handle()));

            // run PDSCH
            if(m_pdschSlotRunFlag[itrIdx])
            {
                
                for(uint32_t strmIdx = 0; strmIdx < m_nStrms_pdsch; ++strmIdx)
                {
#if USE_NVTX
                    nvtxRangePush("PDSCH");
#endif
                    // hold all the streams (i.e cells) until startEvent for the slot
                    CUDA_CHECK(cudaStreamWaitEvent(m_cuStrmsPdsch[strmIdx].handle(), m_pdschInterSlotStartEventVec[itrIdx].handle(), 0));
                    
                    m_pdschTxPipes[strmIdx][m_pdschRunSlotIdx].run(procModeBmsk);

                    // record end
                    if(strmIdx != 0)
                    {
                        m_stopEvents[strmIdx].record(m_cuStrmsPdsch[strmIdx].handle());
                        CUDA_CHECK(cudaStreamWaitEvent(m_cuStrmsPdsch[0].handle(), m_stopEvents[strmIdx].handle(), 0));
                    }
                    CUDA_CHECK(cudaEventRecord(m_timePdschSlotEndEvents[m_pdschRunSlotIdx].handle(), m_cuStrmsPdsch[0].handle()));
#if USE_NVTX
                    nvtxRangePop();
#endif
                }
                m_pdschRunSlotIdx++;
            }        

            // wait for PDSCH cells to be processed 
            if(m_runPDCCH)
            {
                runPDCCHItr(m_SlotBoundaryEventVec[itrIdx].handle(), itrIdx);
                CUDA_CHECK(cudaStreamWaitEvent(m_cuStrmsPdsch[0].handle(), m_PDCCHStopEvents[0].handle(), 0));
            }
            else if(pdschTxRunMsgPayload.pdcchStopEventVec)
            {
                CUDA_CHECK(cudaStreamWaitEvent(m_cuStrmsPdsch[0].handle(), (*pdschTxRunMsgPayload.pdcchStopEventVec)[itrIdx].handle(), 0));
            }

            // timer for next slot boundary
            // added for the last slot to preserve timeline in power measurement mode
            gpu_slot_start_time_offset_ns = static_cast<uint64_t>(itrIdx + 1) * (time_slot_duration * NS_PER_US);
            gpu_ns_delay_until(m_GPUtimeDl_d.addr(), gpu_slot_start_time_offset_ns, m_cuStrmsPdsch[0].handle());
            if(itrIdx != m_nSlotsPerPattern - 1) 
            {
                CUDA_CHECK(cudaEventRecord(m_SlotBoundaryEventVec[itrIdx + 1].handle(), m_cuStrmsPdsch[0].handle()));
            }
        }

        // send GPU response message
        CUDA_CHECK(cudaEventRecord(m_shPtrStopEvent->handle(), m_cuStrmsPdsch[0].handle()));
    }
}

void cuPHYTestWorker::runPDCCH(std::shared_ptr<void>& shPtrPayload)
{
    // unpack message
    cuPHYTestPdschTxRunMsgPayload& pdschTxRunMsgPayload = *std::static_pointer_cast<cuPHYTestPdschTxRunMsgPayload>(shPtrPayload);

    CUDA_CHECK(cudaStreamWaitEvent(m_cuStrmsPdcch[0].handle(), pdschTxRunMsgPayload.startEvent, 0));
    CUDA_CHECK(cudaEventRecord(m_uqPtrTimeStartEvent->handle(), m_cuStrmsPdcch[0].handle()));
    for(uint32_t itrIdx = 0; itrIdx < m_nSlotsPerPattern; ++itrIdx)
    {
        // hold all the streams (i.e cells) until startEvent for the slot
        if(itrIdx != 0)
        {
            if(pdschTxRunMsgPayload.pdschInterSlotEventVec) //PDCCH running in dedicated context alongside PDSCH
            {
                runPDCCHItr((*pdschTxRunMsgPayload.pdschInterSlotEventVec)[itrIdx].handle(), itrIdx);
            }
            else // PDCCH running standalone
            {
                runPDCCHItr(m_pdcchCsirsInterSlotEndEventVec[itrIdx - 1].handle(), itrIdx);
            }
        }
        else
        {
            if(pdschTxRunMsgPayload.pdschInterSlotEventVec) //PDCCH running in dedicated context alongside PDSCH
                runPDCCHItr((*pdschTxRunMsgPayload.pdschInterSlotEventVec)[0].handle(), itrIdx);
            else
                runPDCCHItr(pdschTxRunMsgPayload.startEvent, itrIdx);
        }
    }

    CUDA_CHECK(cudaEventRecord(m_shPtrStopEvent->handle(), m_cuStrmsPdcch[0].handle()));
}

void cuPHYTestWorker::pdschTxRunHandler(std::shared_ptr<void>& shPtrPayload)
{
    DEBUG_TRACE("%s id %d [tid %s][wrkrCtxId 0x%0lx currCtxId 0x%0lx]: pdschTxRunHandler\n", m_name.c_str(), m_wrkrId, getThreadIdStr().c_str(), getCuCtxId(), getCurrCuCtxId());
    /*printf("pdschTxRunHandle with m_runPDSCH %d, m_run_PDCCH %d, m_long_pattern %d\n",
        m_runPDSCH, m_runPDCCH, m_longPattern);*/
    m_pdschRunSlotIdx = 0;
    m_pdcchRunSlotIdx = 0;
    m_csirsRunSlotIdx = 0;
    m_pbchRunSlotIdx  = 0;
    if(m_runPDSCH)
    {
        // if m_runPDCCH is set, the PDCCH will be run too in the runPDSCH_* calls below
        if(m_longPattern == 3 or m_longPattern == 6) // has ULBFW or DLBFW
        {
            runPDSCH_U5_3_6(shPtrPayload);
        }
        else if(m_longPattern > 6 && m_longPattern <= 8) // has
        {
            runPDSCH_U6(shPtrPayload);
        }
        else
            runPDSCH_U3_U5_1_2_4_5(shPtrPayload);
    }
    else if(m_runPDCCH || m_runCSIRS) // Only exercised when there's no PDSCH
    {
        runPDCCH(shPtrPayload);
    }
    
    if(m_runSSB)
    {
        runSSB(shPtrPayload);
    }

    cuPHYTestPdschTxRunMsgPayload& pdschTxRunMsgPayload = *std::static_pointer_cast<cuPHYTestPdschTxRunMsgPayload>(shPtrPayload);
    // send CPU response message
    if(pdschTxRunMsgPayload.rsp)
    {
        auto shPtrRspPayload            = std::make_shared<cuPHYTestPdschTxRunRspMsgPayload>();
        shPtrRspPayload->workerId       = m_wrkrId;
        shPtrRspPayload->shPtrStopEvent = m_shPtrStopEvent;

        auto shPtrRsp = std::make_shared<testWrkrRspMsg>(CUPHY_TEST_WRKR_RSP_MSG_PDSCH_RUN, m_wrkrId, shPtrRspPayload);
        m_shPtrRspQ->send(shPtrRsp);
    }
}
void cuPHYTestWorker::runSSB(std::shared_ptr<void>& shPtrPayload)
{
    cuPHYTestPdschTxRunMsgPayload& pdschTxRunMsgPayload = *std::static_pointer_cast<cuPHYTestPdschTxRunMsgPayload>(shPtrPayload);

    if(m_runSSB)
    {
        int nSSBObjects = m_pdsch_group_cells ? 1 : m_nSSBCells;
        // common starting point
        CUDA_CHECK(cudaStreamWaitEvent(m_cuStrmsSsb[0].handle(), pdschTxRunMsgPayload.startEvent, 0));
        if(!(m_runPDSCH || m_runPDCCH || m_runCSIRS)) // only record start if no PDSCH or PDCCH run in the same worker
            CUDA_CHECK(cudaEventRecord(m_uqPtrTimeStartEvent->handle(), m_cuStrmsSsb[0].handle()));

        for(uint32_t itrIdx = 0; itrIdx < m_nSlotsPerPattern; ++itrIdx)
        {
            if(m_pbchSlotRunFlag[itrIdx] == 0) // not an ssb slot
                continue;
#if USE_NVTX
            nvtxRangePush("SSB");
#endif
            for(int i = 0; i < nSSBObjects; i++)
            {
                CUDA_CHECK(cudaStreamWaitEvent(m_cuStrmsSsb[i].handle(), pdschTxRunMsgPayload.startEvent, 0));

                if(pdschTxRunMsgPayload.pdschInterSlotEventVec != nullptr)
                {
                    CUDA_CHECK(cudaStreamWaitEvent(m_cuStrmsSsb[i].handle(), (*pdschTxRunMsgPayload.pdschInterSlotEventVec)[itrIdx].handle(), 0));
                }
                uint32_t ssbDelayUs = (m_uldlMode == 6) ? g_start_delay_cfg_us.ssb_u6 : g_start_delay_cfg_us.ssb_u5;
                if(ssbDelayUs)
                    gpu_us_delay(ssbDelayUs, 0, m_cuStrmsSsb[i].handle(), 1);
            }

            CUDA_CHECK(cudaEventRecord(m_timeSSBSlotStartEvents[m_pbchRunSlotIdx].handle(), m_cuStrmsSsb[0].handle()));

            for(int i = 0; i < nSSBObjects; i++)
            {
                m_ssbTxPipes[i][0].run(0 /*ssb_proc_mode*/);

                if(m_ref_check_ssb)
                {
                    // Providing no argument to refCheck (default is verbose=false) results in no prints during ref. checks
                    int err = m_ssbTxDynamicApiDataSets[i][0].refCheck(/*true*/); //FIXME potentially move somewhere where it won't affect perf.
                    if(err != 0)
                    {
                        // NVLOG will log number of mismatch in ssbDynApiDataset::refCheck
                        throw cuphy::cuphy_exception(CUPHY_STATUS_REF_MISMATCH);
                    }
                    else
                    {
                        NVLOGI_FMT(NVLOG_SSB, "SSB REFERENCE CHECK: PASSED");
                    }
                }
                // record end
                if(i != 0)
                {
                    m_stopEvents[i].record(m_cuStrmsSsb[i].handle());
                    CUDA_CHECK(cudaStreamWaitEvent(m_cuStrmsSsb[0].handle(), m_stopEvents[i].handle(), 0));
                }
            }

            CUDA_CHECK(cudaEventRecord(m_timeSSBSlotEndEvents[m_pbchRunSlotIdx].handle(), m_cuStrmsSsb[0].handle()));
#if USE_NVTX
            nvtxRangePop();
#endif
            m_pbchRunSlotIdx++;
        }
        // send GPU response message
        CUDA_CHECK(cudaEventRecord(m_shPtrStopEvent->handle(), m_cuStrmsSsb[0].handle()));
    }
}
void cuPHYTestWorker::runPDSCH_U3_U5_1_2_4_5(std::shared_ptr<void>& shPtrPayload)
{
    // unpack message
    cuPHYTestPdschTxRunMsgPayload& pdschTxRunMsgPayload = *std::static_pointer_cast<cuPHYTestPdschTxRunMsgPayload>(shPtrPayload);
    uint64_t                       procModeBmsk         = m_pdsch_proc_mode;

    // Note: m_nItrsPerStrm  represents the # of cells processed per slot sequentially
    // Note: m_nStrms represents the # of cells processed per slot concurrently

    // Loop over iterations
    //printf("m_runPDSCH %d, m_runDlbfw %d, m_nDlbfwCells %d, m_nStrms_pdsch %d\n", m_runPDSCH, m_runDlbfw, m_nDlbfwCells, m_nStrms_pdsch);

    if(m_runPDSCH)
    {
        // find time slot duration in us
        CUDA_CHECK(cudaStreamWaitEvent(m_cuStrmsPdsch[0].handle(), pdschTxRunMsgPayload.startEvent, 0));
        get_gpu_time(m_GPUtimeDl_d.addr(), m_cuStrmsPdsch[0].handle()); 
        CUDA_CHECK(cudaEventRecord(m_uqPtrTimeStartEvent->handle(), m_cuStrmsPdsch[0].handle()));
        CUDA_CHECK(cudaEventRecord(m_SlotBoundaryEventVec[0].handle(), m_cuStrmsPdsch[0].handle())); // slot 0 boundary at t=0
        // Slot 0: then add PDSCH delay, then record PDSCH slot start
        if(g_start_delay_cfg_us.pdsch_no_bfw_u6 > 0)
            gpu_us_delay(static_cast<uint32_t>(g_start_delay_cfg_us.pdsch_no_bfw_u6), 0, m_cuStrmsPdsch[0].handle(), 1);
        // record start of time slot
        CUDA_CHECK(cudaEventRecord(m_pdschInterSlotStartEventVec[0].handle(), m_cuStrmsPdsch[0].handle()));

        for(uint32_t itrIdx = 0; itrIdx < m_nSlotsPerPattern; ++itrIdx)
        {
            // DDDSUUDDDD
            if(m_pdschSlotRunFlag[itrIdx])
            {
#if USE_NVTX
                nvtxRangePush("PDSCH");
#endif
                for(uint32_t strmIdx = 0; strmIdx < m_nStrms_pdsch; ++strmIdx)
                {
                    // hold all the streams (i.e cells) until startEvent for the slot
                    CUDA_CHECK(cudaStreamWaitEvent(m_cuStrmsPdsch[strmIdx].handle(), m_pdschInterSlotStartEventVec[itrIdx].handle(), 0));

                    m_pdschTxPipes[strmIdx][m_pdschRunSlotIdx].run(procModeBmsk);

                    // record end
                    if(strmIdx != 0)
                    {
                        m_stopEvents[strmIdx].record(m_cuStrmsPdsch[strmIdx].handle());
                        CUDA_CHECK(cudaStreamWaitEvent(m_cuStrmsPdsch[0].handle(), m_stopEvents[strmIdx].handle(), 0));
                    }
                    CUDA_CHECK(cudaEventRecord(m_timePdschSlotEndEvents[m_pdschRunSlotIdx].handle(), m_cuStrmsPdsch[0].handle()));
                }
#if USE_NVTX
                nvtxRangePop();
#endif              
            }

            CUDA_CHECK(cudaEventRecord(m_uqPtrPdschIterStopEvent->handle(), m_cuStrmsPdsch[0].handle()));

            // only differnece beteeen -u3 and -u5 longPattern 1 2 4 5
            if((!m_longPattern))
            {
                if(m_runDlbfw && m_pdschSlotRunFlag[itrIdx])
                {
#if USE_NVTX
                    nvtxRangePush("DLBFW");
#endif
                    // wait for PDSCH cells to be processed
                    CUDA_CHECK(cudaStreamWaitEvent(m_cuStrmsDlbfw[0].handle(), m_uqPtrPdschIterStopEvent->handle(), 0));

                    CUDA_CHECK(cudaEventRecord(m_timeDlbfwSlotStartEvents[m_pdschRunSlotIdx].handle(), m_cuStrmsDlbfw[0].handle()));
                    if(m_pdschSlotRunFlag[itrIdx])
                    {
                        m_dlbfwPipelineVec[m_pdschRunSlotIdx][0].run(m_pdsch_proc_mode); 

                        if(m_ref_check_dlbfw)
                        {
                            float refCheckSnrThd = 30.0f;
                            CUDA_CHECK(cudaStreamSynchronize(m_cuStrmsDlbfw[0].handle()));
                            m_dlbfwEvalDatasetVec[m_pdschRunSlotIdx][0].bfwEvalCoefs(m_dlbfwStaticApiDatasetVec[m_pdschRunSlotIdx][0], m_dlbfwDynamicApiDatasetVec[m_pdschRunSlotIdx][0], m_cuStrmsDlbfw[0].handle(), refCheckSnrThd, true);
                        }
                    }

                    CUDA_CHECK(cudaEventRecord(m_timeDlbfwSlotEndEvents[m_pdschRunSlotIdx].handle(), m_cuStrmsDlbfw[0].handle()));
                    CUDA_CHECK(cudaEventRecord(m_uqPtrDlbfwStopEvent->handle(), m_cuStrmsDlbfw[0].handle()));
                    CUDA_CHECK(cudaStreamWaitEvent(m_cuStrmsPdsch[0].handle(), m_uqPtrDlbfwStopEvent->handle(), 0));
#if USE_NVTX
                    nvtxRangePop();
#endif
                }
            }

            if(m_runPDCCH || m_runCSIRS)
            {
                runPDCCHItr(m_SlotBoundaryEventVec[itrIdx].handle(), itrIdx);
                CUDA_CHECK(cudaStreamWaitEvent(m_cuStrmsPdsch[0].handle(), m_PDCCHStopEvents[0].handle(), 0));
            }
            else if(pdschTxRunMsgPayload.pdcchStopEventVec)
            {
                CUDA_CHECK(cudaStreamWaitEvent(m_cuStrmsPdsch[0].handle(), (*pdschTxRunMsgPayload.pdcchStopEventVec)[itrIdx].handle(), 0));
            }

            // timer for next slot boundary (500 us), then record slot boundary, then PDSCH delay, then record next PDSCH slot start
            // added for the last slot to preserve timeline in power measurement mode
            uint64_t           gpu_slot_start_time_offset_ns = static_cast<uint64_t>(itrIdx + 1) * time_slot_duration * NS_PER_US;
            gpu_ns_delay_until(m_GPUtimeDl_d.addr(), gpu_slot_start_time_offset_ns, m_cuStrmsPdsch[0].handle());
            if(itrIdx != m_nSlotsPerPattern - 1)
            {
                CUDA_CHECK(cudaEventRecord(m_SlotBoundaryEventVec[itrIdx + 1].handle(), m_cuStrmsPdsch[0].handle()));
                if(g_start_delay_cfg_us.pdsch_no_bfw_u6 > 0)
                    gpu_us_delay(static_cast<uint32_t>(g_start_delay_cfg_us.pdsch_no_bfw_u6), 0, m_cuStrmsPdsch[0].handle(), 1);
                CUDA_CHECK(cudaEventRecord(m_pdschInterSlotStartEventVec[itrIdx + 1].handle(), m_cuStrmsPdsch[0].handle()));
            }

            if(m_pdschSlotRunFlag[itrIdx])
            {
                m_pdschRunSlotIdx++;
            }
        }

        // send GPU response message
        CUDA_CHECK(cudaEventRecord(m_shPtrStopEvent->handle(), m_cuStrmsPdsch[0].handle()));
    }
}

void cuPHYTestWorker::pschTxRxRunHandlerStrmPrio(std::shared_ptr<void>& shPtrPayload)
{
    DEBUG_TRACE("%s id %d [tid %s][wrkrCtxId 0x%0lx currCtxId 0x%0lx]: pschTxRxRunHandler\n", m_name.c_str(), m_wrkrId, getThreadIdStr().c_str(), getCuCtxId(), getCurrCuCtxId());

    // unpack message
    cuPHYTestPschTxRxRunMsgPayload& pschTxRxRunMsgPayload = *std::static_pointer_cast<cuPHYTestPschTxRxRunMsgPayload>(shPtrPayload);

    // wait for start event
    for(uint32_t strmIdx = 0; strmIdx < m_nStrms; ++strmIdx)
    {
        CUDA_CHECK(cudaStreamWaitEvent(m_cuStrmsPdsch[strmIdx].handle(), pschTxRxRunMsgPayload.startEvent, 0));
    }
    CUDA_CHECK(cudaEventRecord(m_uqPtrTimeStartEvent->handle(), m_cuStrmsPdsch[0].handle()));

    // Note: m_nItrsPerStrm  represents the # of cells processed per slot sequentially
    // Note: m_nStrms represents the # of cells processed per slot concurrently
    //printf("m_nStrms %d, m_nItrsPerStrm %d, PDSCH strms %d\n", m_nStrms, m_nItrsPerStrm, m_nStrms_pdsch);

    // Loop over stremas
    for(uint32_t strmIdx = 0; strmIdx < m_nStrms; ++strmIdx)
    {
        for(uint32_t itrIdx = 0; itrIdx < m_nItrsPerStrm; ++itrIdx)
        {
            // run
            if(strmIdx < m_nStrms_pdsch)
            {
                m_pdschTxPipes[strmIdx][itrIdx].run(m_pdsch_proc_mode);
            }
            CUDA_CHECK(cudaEventRecord(m_pschDlUlSyncEvents[strmIdx][itrIdx].handle(), m_cuStrmsPdsch[strmIdx].handle()));

            CUDA_CHECK(cudaStreamWaitEvent(m_cuStrmsPusch[strmIdx].handle(), m_pschDlUlSyncEvents[strmIdx][itrIdx].handle()));
            m_puschRxPipes[strmIdx][itrIdx].run(PUSCH_RUN_ALL_PHASES);
        }

        // synch
        if(strmIdx != 0)
        {
            m_stopEvents[strmIdx].record(m_cuStrmsPusch[strmIdx].handle());
            CUDA_CHECK(cudaStreamWaitEvent(m_cuStrmsPusch[0].handle(), m_stopEvents[strmIdx].handle(), 0));
        }
    }

    // send GPU response message
    CUDA_CHECK(cudaEventRecord(m_shPtrStopEvent->handle(), m_cuStrmsPusch[0].handle()));
    m_pdschRunSlotIdx++;

    // send CPU response message
    if(pschTxRxRunMsgPayload.rsp)
    {
        auto shPtrRspPayload            = std::make_shared<cuPHYTestPschRunRspMsgPayload>();
        shPtrRspPayload->workerId       = m_wrkrId;
        shPtrRspPayload->shPtrStopEvent = m_shPtrStopEvent;

        auto shPtrRsp = std::make_shared<testWrkrRspMsg>(CUPHY_TEST_WRKR_RSP_MSG_PSCH_RUN, m_wrkrId, shPtrRspPayload);
        m_shPtrRspQ->send(shPtrRsp);
    }
}

void cuPHYTestWorker::pschTxRxRunHandler(std::shared_ptr<void>& shPtrPayload)
{
    m_pdschRunSlotIdx = 0;
    m_pdcchRunSlotIdx = 0;
    m_csirsRunSlotIdx = 0;
    m_pbchRunSlotIdx  = 0;
#ifdef ENABLE_F01_STREAM_PRIO
    pschTxRxRunHandlerStrmPrio(shPtrPayload);
#else
    pschTxRxRunHandlerNoStrmPrio(shPtrPayload);
#endif // ENABLE_F01_STREAM_PRIO
}

void cuPHYTestWorker::evalHandler(std::shared_ptr<void>& shPtrPayload)
{
    DEBUG_TRACE("%s id %d [tid %s][wrkrCtxId 0x%0lx currCtxId 0x%0lx]: evalHandler\n", m_name.c_str(), m_wrkrId, getThreadIdStr().c_str(), getCuCtxId(), getCurrCuCtxId());
    commnTestEvalMsgPayload& evalMsgPayload = *std::static_pointer_cast<commnTestEvalMsgPayload>(shPtrPayload);

    if(m_runPUSCH)
    {
        // update codeblock errors
        if(evalMsgPayload.cbErrors)
        {
            for(uint32_t strmIdx = 0; strmIdx < m_nStrms; ++strmIdx)
            {
                for(uint32_t itrIdx = 0; itrIdx < m_nItrsPerStrm; ++itrIdx)
                {
#ifndef ENABLE_F01_STREAM_PRIO
                    if(m_uldlMode == 4)
                        m_cuStrms[strmIdx].synchronize();
#endif
                    m_cuStrmsPusch[strmIdx].synchronize();
                    m_puschRxPipes[strmIdx][itrIdx].writeDbgSynch(m_cuStrmsPusch[strmIdx].handle());
                    m_cuStrmsPusch[strmIdx].synchronize();
                    uint32_t numCbErrors = m_puschRxEvalDataSets[strmIdx][itrIdx].computeNumCbErrors(m_puschRxDynamicApiDataSets[strmIdx][itrIdx]);

                    m_maxNumCbErrors[strmIdx][itrIdx] = std::max(m_maxNumCbErrors[strmIdx][itrIdx], numCbErrors);
                }
            }
        }
        float       elapsedTimeMs = 0.0f;
        if(m_puschRxDynamicApiDataSets[0][0].puschDynPrm.pDataOut->isEarlyHarqPresent || m_puschRxDynamicApiDataSets[0][0].puschDynPrm.pDataOut->isFrontLoadedDmrsPresent) // [0][0] means PUSCH1
        {
            cudaError_t e             = cudaEventElapsedTime(&elapsedTimeMs,
                                                m_uqPtrTimeStartEvent->handle(),
                                                m_puschRxStaticApiDataSets[0][0].puschStatPrms.subSlotCompletedEvent);
            if(cudaSuccess != e) throw cuphy::cuda_exception(e);

            m_totPUSCHSubslotProcRunTime += elapsedTimeMs;
        }

        cudaError_t e             = cudaEventElapsedTime(&elapsedTimeMs,
                                             m_uqPtrTimeStartEvent->handle(),
                                             m_uqPtrTimePUSCHEndEvent->handle());
        if(cudaSuccess != e) throw cuphy::cuda_exception(e);

        m_totPUSCHRunTime += elapsedTimeMs;

        if(m_uldlMode != 4)
        {
            elapsedTimeMs = 0.0f;
            e             = cudaEventElapsedTime(&elapsedTimeMs,
                                     m_uqPtrTimeStartEvent->handle(),
                                     m_uqPtrTimePUSCHStartEvent->handle());

            if(cudaSuccess != e) throw cuphy::cuda_exception(e);

            m_totPUSCHStartTime += elapsedTimeMs;
        }

        if(m_longPattern)
        {
            if(m_puschRxDynamicApiDataSets[1][0].puschDynPrm.pDataOut->isEarlyHarqPresent || m_puschRxDynamicApiDataSets[1][0].puschDynPrm.pDataOut->isFrontLoadedDmrsPresent) // [1][0] means PUSCH2
            {
                cudaError_t e             = cudaEventElapsedTime(&elapsedTimeMs,
                                                    m_uqPtrTimeStartEvent->handle(),
                                                    m_puschRxStaticApiDataSets[1][0].puschStatPrms.subSlotCompletedEvent);
                if(cudaSuccess != e) throw cuphy::cuda_exception(e);

                m_totPUSCH2SubslotProcRunTime += elapsedTimeMs;
            }

            float       elapsedTimeMs = 0.0f;
            cudaError_t e             = cudaEventElapsedTime(&elapsedTimeMs,
                                                 m_uqPtrTimeStartEvent->handle(),
                                                 m_uqPtrTimePUSCH2EndEvent->handle());

            if(cudaSuccess != e) throw cuphy::cuda_exception(e);

            m_totPUSCH2RunTime += elapsedTimeMs;

            elapsedTimeMs = 0.0f;
            e             = cudaEventElapsedTime(&elapsedTimeMs,
                                     m_uqPtrTimeStartEvent->handle(),
                                     m_uqPtrTimePUSCH2StartEvent->handle());

            if(cudaSuccess != e) throw cuphy::cuda_exception(e);

            m_totPUSCH2StartTime += elapsedTimeMs;
        }
    }

    if(m_runPRACH)
    {
        float       elapsedTimeMs = 0.0f;
        cudaError_t e             = cudaEventElapsedTime(&elapsedTimeMs,
                                             m_uqPtrTimeStartEvent->handle(),
                                             m_uqPtrTimePRACHEndEvent->handle());

        if(cudaSuccess != e) throw cuphy::cuda_exception(e);

        m_totPRACHRunTime += elapsedTimeMs;

        elapsedTimeMs = 0.0f;
        e             = cudaEventElapsedTime(&elapsedTimeMs,
                                 m_uqPtrTimeStartEvent->handle(),
                                 m_uqPtrTimePRACHStartEvent->handle());

        if(cudaSuccess != e) throw cuphy::cuda_exception(e);

        m_totPRACHStartTime += elapsedTimeMs;
    }

    if(m_runUlbfw)
    {
        float       elapsedTimeMs = 0.0f;
        cudaError_t e             = cudaEventElapsedTime(&elapsedTimeMs,
                                             m_uqPtrTimeStartEvent->handle(),
                                             m_uqPtrTimeULBfwEndEvent->handle());

        if(cudaSuccess != e) throw cuphy::cuda_exception(e);

        m_totULBFWRunTime += elapsedTimeMs;

        elapsedTimeMs = 0.0f;
        e             = cudaEventElapsedTime(&elapsedTimeMs,
                                 m_uqPtrTimeStartEvent->handle(),
                                 m_uqPtrTimeULBfwStartEvent->handle());

        if(cudaSuccess != e) throw cuphy::cuda_exception(e);

        m_totULBFWStartTime += elapsedTimeMs;

        if(m_longPattern)
        {
            float       elapsedTimeMs = 0.0f;
            cudaError_t e             = cudaEventElapsedTime(&elapsedTimeMs,
                                                 m_uqPtrTimeStartEvent->handle(),
                                                 m_uqPtrTimeULBfw2EndEvent->handle());

            if(cudaSuccess != e) throw cuphy::cuda_exception(e);

            m_totULBFW2RunTime += elapsedTimeMs;

            elapsedTimeMs = 0.0f;
            e             = cudaEventElapsedTime(&elapsedTimeMs,
                                     m_uqPtrTimeStartEvent->handle(),
                                     m_uqPtrTimeULBfw2StartEvent->handle());

            if(cudaSuccess != e) throw cuphy::cuda_exception(e);

            m_totULBFW2StartTime += elapsedTimeMs;
        }
    }

    if(m_runPUCCH)
    {
        float       elapsedTimeMs = 0.0f;
        cudaError_t e             = cudaEventElapsedTime(&elapsedTimeMs,
                                             m_uqPtrTimeStartEvent->handle(),
                                             m_uqPtrTimePUCCHEndEvent->handle());

        if(cudaSuccess != e) throw cuphy::cuda_exception(e);

        m_totPUCCHRunTime += elapsedTimeMs;

        elapsedTimeMs = 0.0f;
        e             = cudaEventElapsedTime(&elapsedTimeMs,
                                 m_uqPtrTimeStartEvent->handle(),
                                 m_uqPtrTimePUCCHStartEvent->handle());

        if(cudaSuccess != e) throw cuphy::cuda_exception(e);

        m_totPUCCHStartTime += elapsedTimeMs;

        if(m_longPattern)
        {
            float       elapsedTimeMs = 0.0f;
            cudaError_t e             = cudaEventElapsedTime(&elapsedTimeMs,
                                                 m_uqPtrTimeStartEvent->handle(),
                                                 m_uqPtrTimePUCCH2EndEvent->handle());

            if(cudaSuccess != e) throw cuphy::cuda_exception(e);

            m_totPUCCH2RunTime += elapsedTimeMs;

            elapsedTimeMs = 0.0f;
            e             = cudaEventElapsedTime(&elapsedTimeMs,
                                     m_uqPtrTimeStartEvent->handle(),
                                     m_uqPtrTimePUCCH2StartEvent->handle());

            if(cudaSuccess != e) throw cuphy::cuda_exception(e);

            m_totPUCCH2StartTime += elapsedTimeMs;
        }
    }

    if(m_runSRS)
    {
        float       elapsedTimeMs = 0.0f;
        cudaError_t e             = cudaEventElapsedTime(&elapsedTimeMs,
                                             m_uqPtrTimeStartEvent->handle(),
                                             m_uqPtrTimeSRSEndEvent->handle());

        if(cudaSuccess != e) throw cuphy::cuda_exception(e);

        m_totSRSRunTime += elapsedTimeMs;

        elapsedTimeMs = 0.0f;
        e             = cudaEventElapsedTime(&elapsedTimeMs,
                                 m_uqPtrTimeStartEvent->handle(),
                                 m_uqPtrTimeSRSStartEvent->handle());

        if(cudaSuccess != e) throw cuphy::cuda_exception(e);

        m_totSRSStartTime += elapsedTimeMs;
    }

    if(m_runSRS2)
    {
        float       elapsedTimeMs = 0.0f;
        cudaError_t e             = cudaEventElapsedTime(&elapsedTimeMs,
                                             m_uqPtrTimeStartEvent->handle(),
                                             m_uqPtrTimeSRS2EndEvent->handle());

        if(cudaSuccess != e) throw cuphy::cuda_exception(e);

        m_totSRS2RunTime += elapsedTimeMs;

        elapsedTimeMs = 0.0f;
        e             = cudaEventElapsedTime(&elapsedTimeMs,
                                 m_uqPtrTimeStartEvent->handle(),
                                 m_uqPtrTimeSRS2StartEvent->handle());

        if(cudaSuccess != e) throw cuphy::cuda_exception(e);

        m_totSRS2StartTime += elapsedTimeMs;
    }

    // check whether number of DL slots run according to config
    for(uint32_t itrIdx = 0; itrIdx < m_nSlotsPerPattern; ++itrIdx)
    {
        if(m_runPDSCH && m_pdschSlotRunFlag[itrIdx])
            m_pdschRunSlotIdx--;

        if(m_runPDCCH && m_pdcchSlotRunFlag[itrIdx])
            m_pdcchRunSlotIdx--;

        if(m_runCSIRS && m_csirsSlotRunFlag[itrIdx])
            m_csirsRunSlotIdx--;
        
        if(m_runSSB && m_pbchSlotRunFlag[itrIdx])
            m_pbchRunSlotIdx--;
    }
    // all slotIdx should be zero
    assert((m_pdschRunSlotIdx || m_pdcchRunSlotIdx || m_csirsRunSlotIdx || m_pbchRunSlotIdx) == 0);

    for(uint32_t itrIdx = 0; itrIdx < m_nSlotsPerPattern; ++itrIdx)
    {
        float       elapsedTimeMs = 0.0f;
        cudaError_t e;

        if(m_runPDSCH && m_pdschSlotRunFlag[itrIdx])
        {
            e = cudaEventElapsedTime(&elapsedTimeMs,
                                     m_uqPtrTimeStartEvent->handle(),
                                     m_timePdschSlotEndEvents[m_pdschRunSlotIdx].handle());

            if(cudaSuccess != e) throw cuphy::cuda_exception(e);

            m_totRunTimePdschItr[m_pdschRunSlotIdx] += elapsedTimeMs;
            
            if(m_uldlMode != 4)
            {
                e = cudaEventElapsedTime(&elapsedTimeMs,
                                         m_uqPtrTimeStartEvent->handle(),
                                         m_pdschInterSlotStartEventVec[itrIdx].handle());

                if(cudaSuccess != e) throw cuphy::cuda_exception(e);

                m_totPdschSlotStartTime[m_pdschRunSlotIdx] += elapsedTimeMs;
            }

            if(m_runDlbfw)
            {
                e             = cudaEventElapsedTime(&elapsedTimeMs,
                                        m_uqPtrTimeStartEvent->handle(),
                                        m_timeDlbfwSlotStartEvents[m_pdschRunSlotIdx].handle());
                if(cudaSuccess != e) throw cuphy::cuda_exception(e);

                m_totDlbfwIterStartTime[m_pdschRunSlotIdx] += elapsedTimeMs;

                e             = cudaEventElapsedTime(&elapsedTimeMs,
                                        m_uqPtrTimeStartEvent->handle(),
                                        m_timeDlbfwSlotEndEvents[m_pdschRunSlotIdx].handle());
                if(cudaSuccess != e) throw cuphy::cuda_exception(e);

                m_totRunTimeDlbfwItr[m_pdschRunSlotIdx] += elapsedTimeMs;
            }
            m_pdschRunSlotIdx++;
        }

        if(m_runCSIRS && m_csirsSlotRunFlag[itrIdx])
        {
            e = cudaEventElapsedTime(&elapsedTimeMs,
                                     m_uqPtrTimeStartEvent->handle(),
                                     m_timeCSIRSSlotEndEvents[m_csirsRunSlotIdx].handle());

            if(cudaSuccess != e) throw cuphy::cuda_exception(e);
            m_totRunTimeCSIRSItr[m_csirsRunSlotIdx] += elapsedTimeMs;

            e = cudaEventElapsedTime(&elapsedTimeMs,
                                     m_uqPtrTimeStartEvent->handle(),
                                     m_timeCSIRSSlotStartEvents[m_csirsRunSlotIdx].handle());

            if(cudaSuccess != e) throw cuphy::cuda_exception(e);

            m_totCSIRSStartTimes[m_csirsRunSlotIdx] += elapsedTimeMs;
            m_csirsRunSlotIdx++;
        }

        if(m_runPDCCH && m_pdcchSlotRunFlag[itrIdx])
        {
            e = cudaEventElapsedTime(&elapsedTimeMs,
                                     m_uqPtrTimeStartEvent->handle(),
                                     m_timePdcchSlotEndEvents[m_pdcchRunSlotIdx].handle());

            if(cudaSuccess != e) throw cuphy::cuda_exception(e);
            m_totRunTimePdcchItr[m_pdcchRunSlotIdx] += elapsedTimeMs;

            e = cudaEventElapsedTime(&elapsedTimeMs,
                                     m_uqPtrTimeStartEvent->handle(),
                                     m_timePdcchSlotStartEvents[m_pdcchRunSlotIdx].handle());

            if(cudaSuccess != e) throw cuphy::cuda_exception(e);

            m_totPdcchStartTimes[m_pdcchRunSlotIdx] += elapsedTimeMs;
            m_pdcchRunSlotIdx++;
        }

        if(m_runSSB && m_pbchSlotRunFlag[itrIdx])
        {
            cudaError_t e             = cudaEventElapsedTime(&elapsedTimeMs,
                                            m_uqPtrTimeStartEvent->handle(),
                                            m_timeSSBSlotEndEvents[m_pbchRunSlotIdx].handle());

            if(cudaSuccess != e) throw cuphy::cuda_exception(e);

            m_totRunTimeSSBItr[m_pbchRunSlotIdx] += elapsedTimeMs;

            elapsedTimeMs = 0.0f;
            e             = cudaEventElapsedTime(&elapsedTimeMs,
                                    m_uqPtrTimeStartEvent->handle(),
                                    m_timeSSBSlotStartEvents[m_pbchRunSlotIdx].handle());

            if(cudaSuccess != e) throw cuphy::cuda_exception(e);

            m_totSSBStartTime[m_pbchRunSlotIdx] += elapsedTimeMs;

            m_pbchRunSlotIdx++;
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

void cuPHYTestWorker::printHandler(std::shared_ptr<void>& shPtrPayload)
{
    DEBUG_TRACE("%s id %d [tid %s][wrkrCtxId 0x%0lx currCtxId 0x%0lx]: printHandler\n", m_name.c_str(), m_wrkrId, getThreadIdStr().c_str(), getCuCtxId(), getCurrCuCtxId());

    // std::string pipeline_str = (m_runPipelineInParallel) ? "parallel":"serial";
    printf("\n%s worker # %d\n", m_name.c_str(), m_wrkrId);
    // printf("\n--> Runs %d streams in parallel", m_nStrms);
    // printf("\n--> Each stream processes %d slot-cell(s) in series\n\n", m_nItrsPerStrm);

    commnTestPrintMsgPayload& printMsgPayload = *std::static_pointer_cast<commnTestPrintMsgPayload>(shPtrPayload);

    uint32_t nSRSStrms  = m_runSRS2 ? (m_nSRSCells + 1) / 2 : m_nSRSCells;
    uint32_t nSRS2Strms = (m_nSRSCells) / 2;
    uint32_t maxNStrms  = std::max(std::max(m_nStrms, m_nPRACHCells), nSRSStrms);
    float    avgLatency;

    // print max Cb errors
    if(printMsgPayload.cbErrors)
    {
        for(uint32_t strmIdx = 0; strmIdx < m_nStrms; ++strmIdx)
        {
            for(uint32_t itrIdx = 0; itrIdx < m_nItrsPerStrm; ++itrIdx)
            {
                printf("--> strm # %d, itr # %d : max number of Cb Errors  :  %d out of %d \n", strmIdx, itrIdx, m_maxNumCbErrors[strmIdx][itrIdx], m_puschRxEvalDataSets[strmIdx][itrIdx].nCbs);
            }
        }
    }

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

void cuPHYTestWorker::resetEvalHandler(std::shared_ptr<void>& shPtrPayload)
{
    DEBUG_TRACE("%s id %d [tid %s][wrkrCtxId 0x%0lx currCtxId 0x%0lx]: resetEvalHandler\n", m_name.c_str(), m_wrkrId, getThreadIdStr().c_str(), getCuCtxId(), getCurrCuCtxId());

    commnTestResetEvalMsgPayload& resetEvalPayload = *std::static_pointer_cast<commnTestResetEvalMsgPayload>(shPtrPayload);

    // reset max Cb error buffers
    if(resetEvalPayload.cbErrors)
    {
        for(uint32_t strmIdx = 0; strmIdx < m_nStrms; ++strmIdx)
        {
            for(uint32_t itrIdx = 0; itrIdx < m_nItrsPerStrm; ++itrIdx)
            {
                m_maxNumCbErrors[strmIdx][itrIdx] = 0;
            }
        }
    }
    m_totSRSStartTime    = 0;
    m_totSRSRunTime      = 0;
    m_totSRS2StartTime   = 0;
    m_totSRS2RunTime     = 0;
    m_totPRACHStartTime  = 0;
    m_totPRACHRunTime    = 0;
    m_totPUSCHStartTime  = 0;
    m_totPUSCHSubslotProcRunTime = 0;
    m_totPUSCHRunTime    = 0;
    m_totPUSCH2StartTime = 0;
    m_totPUSCH2SubslotProcRunTime = 0;
    m_totPUSCH2RunTime   = 0;
    m_totPUCCHStartTime  = 0;
    m_totPUCCHRunTime    = 0;
    m_totPUCCH2StartTime = 0;
    m_totPUCCH2RunTime   = 0;
    m_totULBFWStartTime  = 0;
    m_totULBFWRunTime    = 0;
    m_totULBFW2StartTime = 0;
    m_totULBFW2RunTime   = 0;

    // reset timers    
    m_totPdschSlotStartTime.assign(m_totPdschSlotStartTime.size(), 0.0f);
    m_totRunTimePdschItr.assign(m_totRunTimePdschItr.size(), 0.0f);
    m_totDlbfwIterStartTime.assign(m_totDlbfwIterStartTime.size(), 0.0f);
    m_totRunTimeDlbfwItr.assign(m_totRunTimeDlbfwItr.size(), 0.0f);
    m_totPdcchStartTimes.assign(m_totPdcchStartTimes.size(), 0.0f);
    m_totRunTimePdcchItr.assign(m_totRunTimePdcchItr.size(), 0.0f);
    m_totCSIRSStartTimes.assign(m_totCSIRSStartTimes.size(), 0.0f);
    m_totRunTimeCSIRSItr.assign(m_totRunTimeCSIRSItr.size(), 0.0f);
    m_totSSBStartTime.assign(m_totSSBStartTime.size(), 0.0f);
    m_totRunTimeSSBItr.assign(m_totRunTimeSSBItr.size(), 0.0f);

    uint32_t nSRSStrms = m_runSRS2 ? (m_nSRSCells + 1) / 2 : m_nSRSCells;

    m_pdschRunSlotIdx = 0;
    m_pdcchRunSlotIdx = 0;
    m_csirsRunSlotIdx = 0;
    m_pbchRunSlotIdx = 0;

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

void cuPHYTestWorker::pdschTxCleanHandler(std::shared_ptr<void>& shPtrPayload)
{
    DEBUG_TRACE("%s id %d [tid %s][wrkrCtxId 0x%0lx currCtxId 0x%0lx]: pdschTxCleanHandler\n", m_name.c_str(), m_wrkrId, getThreadIdStr().c_str(), getCuCtxId(), getCurrCuCtxId());
    cuPHYTestPdschTxCleanMsgPayload& pdschTxCleanMsgPayload = *std::static_pointer_cast<cuPHYTestPdschTxCleanMsgPayload>(shPtrPayload);

    if(m_runPDSCH)
    {
        for(uint32_t strmIdx = 0; strmIdx < m_nStrms_pdsch; ++strmIdx)
        {
            for(uint32_t itrIdx = 0; itrIdx < m_nItrsPerStrm; ++itrIdx)
            {
                cuphy::pdsch_tx& pdschTxPipe = m_pdschTxPipes[strmIdx][itrIdx];

                PdschTx*                         pipeline_ptr = static_cast<PdschTx*>(pdschTxPipe.handle());
                const cuphyPdschCellGrpDynPrm_t* cell_group   = pipeline_ptr->dynamic_params->pCellGrpDynPrm;

                for(int ue_group_id = 0; ue_group_id < cell_group->nUeGrps; ue_group_id++)
                {
                    delete[] cell_group->pUeGrpPrms[ue_group_id].pUePrmIdxs;
                    delete[] cell_group->pUeGrpPrms[ue_group_id].pDmrsDynPrm;
                }

                for(int ue_id = 0; ue_id < cell_group->nUes; ue_id++)
                {
                    delete[] cell_group->pUePrms[ue_id].pCwIdxs;
                }
            }
        }
    }
    if(pdschTxCleanMsgPayload.rsp)
    {
        // Send run completion response
        auto shPtrRspPayload      = std::make_shared<commnTestRspMsgPayload>();
        shPtrRspPayload->workerId = m_wrkrId;

        auto shPtrRsp = std::make_shared<testWrkrRspMsg>(CUPHY_TEST_WRKR_RSP_MSG_PDSCH_CLEAN, m_wrkrId, shPtrRspPayload);
        m_shPtrRspQ->send(shPtrRsp);
    }
}

void cuPHYTestWorker::setWaitValHandler(std::shared_ptr<void>& shPtrPayload)
{
    DEBUG_TRACE("%s id %d [tid %s][wrkrCtxId 0x%0lx currCtxId 0x%0lx]: setWaitValHandler\n", m_name.c_str(), m_wrkrId, getThreadIdStr().c_str(), getCuCtxId(), getCurrCuCtxId());

    commnTestSetWaitValCmdMsgPayload& setWaitValCmdMsgPayload = *std::static_pointer_cast<commnTestSetWaitValCmdMsgPayload>(shPtrPayload);

    // Wait on device value for all streams
    for(auto& cuStrm : m_cuStrms)
    {
        // CU_CHECK(cuStreamWaitValue32(cuStrm.handle(), reinterpret_cast<CUdeviceptr>(m_shPtrGpuStartSyncFlag->addr()), setWaitValCmdMsgPayload.syncFlagVal, CU_STREAM_WAIT_VALUE_GEQ));
        CU_CHECK(cuStreamWaitValue32(cuStrm.handle(), m_ptrGpuStartSyncFlag, setWaitValCmdMsgPayload.syncFlagVal, CU_STREAM_WAIT_VALUE_GEQ));
    }
    // printf("m_wrkrId %d m_shPtrGpuStartSyncFlag %d syncFlagVal %d\n", m_wrkrId, (*m_shPtrGpuStartSyncFlag)[0], setWaitValCmdMsgPayload.syncFlagVal);
    for(auto& cuStrm : m_cuStrmsPusch)
    {
        // CU_CHECK(cuStreamWaitValue32(cuStrm.handle(), reinterpret_cast<CUdeviceptr>(m_shPtrGpuStartSyncFlag->addr()), setWaitValCmdMsgPayload.syncFlagVal, CU_STREAM_WAIT_VALUE_GEQ));
        CU_CHECK(cuStreamWaitValue32(cuStrm.handle(), m_ptrGpuStartSyncFlag, setWaitValCmdMsgPayload.syncFlagVal, CU_STREAM_WAIT_VALUE_GEQ));
    }
    for(auto& cuStrm : m_cuStrmsPdsch)
    {
        // CU_CHECK(cuStreamWaitValue32(cuStrm.handle(), reinterpret_cast<CUdeviceptr>(m_shPtrGpuStartSyncFlag->addr()), setWaitValCmdMsgPayload.syncFlagVal, CU_STREAM_WAIT_VALUE_GEQ));
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

void cuPHYTestWorker::pschRdSmIdHandler(std::shared_ptr<void>& shPtrPayload)
{
    DEBUG_TRACE("%s id %d [tid %s][wrkrCtxId 0x%0lx currCtxId 0x%0lx]: pschRdSmIdHandler\n", m_name.c_str(), m_wrkrId, getThreadIdStr().c_str(), getCuCtxId(), getCurrCuCtxId());

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

void cuPHYTestWorker::msgProcess(std::shared_ptr<testWrkrCmdMsg>& shPtrMsg)
{
    using msgHandler_t = void (cuPHYTestWorker::*)(std::shared_ptr<void> & shPtrPayload);

    static constexpr std::array<std::pair<testWrkrCmdMsgType, msgHandler_t>, N_TEST_WRKR_CMD_MSGS> CUPHY_MSG_HANDLER_LUT{
        {{COMMON_TEST_WRKR_CMD_MSG_INIT, &cuPHYTestWorker::initHandler},
         {COMMON_TEST_WRKR_CMD_MSG_EVAL, &cuPHYTestWorker::evalHandler},
         {COMMON_TEST_WRKR_CMD_MSG_PRINT, &cuPHYTestWorker::printHandler},
         {COMMON_TEST_WRKR_CMD_MSG_RESET_EVAL, &cuPHYTestWorker::resetEvalHandler},
         {COMMON_TEST_WRKR_CMD_MSG_SET_WAIT_VAL, &cuPHYTestWorker::setWaitValHandler},
         {COMMON_TEST_WRKR_CMD_MSG_READ_SM_IDS, &cuPHYTestWorker::pschRdSmIdHandler},
         {COMMON_TEST_WRKR_CMD_MSG_EXIT, &cuPHYTestWorker::exitHandler},
         {CUPHY_TEST_WRKR_CMD_MSG_PUSCH_INIT, &cuPHYTestWorker::puschRxInitHandler},
         {CUPHY_TEST_WRKR_CMD_MSG_PDSCH_INIT, &cuPHYTestWorker::pdschTxInitHandler},
         {CUPHY_TEST_WRKR_CMD_MSG_PUSCH_SETUP, &cuPHYTestWorker::puschRxSetupHandler},
         {CUPHY_TEST_WRKR_CMD_MSG_PDSCH_SETUP, &cuPHYTestWorker::pdschTxSetupHandler},
         {CUPHY_TEST_WRKR_CMD_MSG_PUSCH_RUN, &cuPHYTestWorker::puschRxRunHandler},
         {CUPHY_TEST_WRKR_CMD_MSG_PDSCH_RUN, &cuPHYTestWorker::pdschTxRunHandler},
         {CUPHY_TEST_WRKR_CMD_MSG_PSCH_RUN, &cuPHYTestWorker::pschTxRxRunHandler},
         {CUPHY_TEST_WRKR_CMD_MSG_PDSCH_CLEAN, &cuPHYTestWorker::pdschTxCleanHandler},
         {CUPHY_TEST_WRKR_CMD_MSG_DEINIT, &cuPHYTestWorker::deinitHandler},
         {CUPHY_TEST_WRKR_CMD_MSG_DLBFW_INIT, &cuPHYTestWorker::dlbfwInitHandler},
         {CUPHY_TEST_WRKR_CMD_MSG_DLBFW_SETUP, &cuPHYTestWorker::dlbfwSetupHandler},
         {CUPHY_TEST_WRKR_CMD_MSG_ULBFW_INIT, &cuPHYTestWorker::ulbfwInitHandler},
         {CUPHY_TEST_WRKR_CMD_MSG_ULBFW_SETUP, &cuPHYTestWorker::ulbfwSetupHandler},
         {CUPHY_TEST_WRKR_CMD_MSG_SRS_INIT, &cuPHYTestWorker::srsInitHandler},
         {CUPHY_TEST_WRKR_CMD_MSG_SRS_SETUP, &cuPHYTestWorker::srsSetupHandler},
         {CUPHY_TEST_WRKR_CMD_MSG_PRACH_INIT, &cuPHYTestWorker::prachInitHandler},
         {CUPHY_TEST_WRKR_CMD_MSG_PRACH_SETUP, &cuPHYTestWorker::prachSetupHandler},
         {CUPHY_TEST_WRKR_CMD_MSG_PUCCH_INIT, &cuPHYTestWorker::pucchRxInitHandler},
         {CUPHY_TEST_WRKR_CMD_MSG_PUCCH_SETUP, &cuPHYTestWorker::pucchRxSetupHandler},
         {CUPHY_TEST_WRKR_CMD_MSG_PDCCH_INIT, &cuPHYTestWorker::pdcchTxInitHandler},
         {CUPHY_TEST_WRKR_CMD_MSG_PDCCH_SETUP, &cuPHYTestWorker::pdcchTxSetupHandler},
         {CUPHY_TEST_WRKR_CMD_MSG_SSB_INIT, &cuPHYTestWorker::ssbInitHandler},
         {CUPHY_TEST_WRKR_CMD_MSG_SSB_SETUP, &cuPHYTestWorker::ssbSetupHandler},
         {CUPHY_TEST_WRKR_CMD_MSG_CSIRS_INIT, &cuPHYTestWorker::csirsInitHandler},
         {CUPHY_TEST_WRKR_CMD_MSG_CSIRS_SETUP, &cuPHYTestWorker::csirsSetupHandler}}};
    if(shPtrMsg->type < N_TEST_WRKR_CMD_MSGS)
    {
        DEBUG_TRACE("%s id %d [tid %s][wrkrCtxId 0x%0lx currCtxId 0x%0lx]: msgProcess msgType %d %s\n", m_name.c_str(), m_wrkrId, getThreadIdStr().c_str(), getCuCtxId(), getCurrCuCtxId(), shPtrMsg->type, TEST_WRKR_CMD_MSG_TO_STR[shPtrMsg->type]);

        // If assert fails, ensure the CUPHY_MSG_HANDLER_LUT table matches the enumerated message types in testWrkrCmdMsgType
        assert(shPtrMsg->type == CUPHY_MSG_HANDLER_LUT[shPtrMsg->type].first);
        (this->*(CUPHY_MSG_HANDLER_LUT[shPtrMsg->type].second))(shPtrMsg->payload);
    }
    else
    {
        DEBUG_TRACE("%s id %d [tid %s][wrkrCtxId 0x%0lx currCtxId 0x%0lx]: msgProcess - Message type not supported\n", m_name.c_str(), m_wrkrId, getThreadIdStr().c_str(), getCuCtxId(), getCurrCuCtxId());
    }
}

void cuPHYTestWorker::run(const cuphy::cudaGreenContext& my_green_context)
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

#if 0 // doesn't appear necessary
        // If I am using green contexts, pop the current context
        if(m_useGreenContexts && (CUPHY_TEST_WRKR_CMD_MSG_DEINIT == shPtrCmd->type))
        {
           //Pop the current CUDA context from the current CPU thread)
           CUcontext tmp_ctx;
           CU_CHECK(cuCtxPopCurrent(&tmp_ctx));
        }
#endif

        // DEBUG_TRACE("Thread %s: Processed message: %s\n", m_name.c_str(), TEST_WRKR_CMD_MSG_TO_STR[shPtrMsg->type]);
    }
    DEBUG_TRACE("%s [tid %s][wrkrCtxId 0x%0lx currCtxId 0x%0lx]: Exit run loop for worker %d\n", m_name.c_str(), getThreadIdStr().c_str(), getCuCtxId(), getCurrCuCtxId(), m_wrkrId);
}

// //----------------------------------------------------------------------------------------------------------
// // APIs invoked by orchestration thread

void cuPHYTestWorker::init(uint32_t nStrms, uint32_t nSlotsPerPattern, uint32_t nItrsPerStrm, uint32_t nTimingItrs, std::map<std::string, int>& cuStrmPrioMap, std::shared_ptr<cuphy::buffer<uint32_t, cuphy::pinned_alloc>>& shPtrCpuGpuSyncFlag, numPhyCells_t numPhyCells, bool srsSplit, bool waitRsp, uint32_t longPattern, bool srsCtx)
{
    // set run configuration
    m_nStrms             = nStrms;
    m_nStrms_pdsch       = nStrms; // Will be overwritten to 1 in pdschTxInit iff pdsch_group_cells is 1.
    m_nSlotsPerPattern   = nSlotsPerPattern;
    m_nItrsPerStrm       = nItrsPerStrm;
    m_nTimingItrs        = nTimingItrs;
    m_nPUSCHCells        = numPhyCells.nPUSCHCells;
    m_nPDSCHCells        = numPhyCells.nPDSCHCells;
    m_nPDCCHCells        = numPhyCells.nPDCCHCells;
    m_nCSIRSCells        = numPhyCells.nCSIRSCells;
    m_nUlbfwCells        = numPhyCells.nUlbfwCells;
    m_nDlbfwCells        = numPhyCells.nDlbfwCells;
    m_nSRSCells          = numPhyCells.nSRSCells;
    m_nPRACHCells        = numPhyCells.nPRACHCells;
    m_nPUCCHCells        = numPhyCells.nPUCCHCells;
    m_nSSBCells          = numPhyCells.nSSBCells;
    m_runSRS2            = srsSplit;
    m_longPattern        = longPattern;
    m_srsCtx             = srsCtx;

    // verify inputs
    if(m_longPattern < 0 || m_longPattern > 8)
    {
        NVLOGF_FMT(NVLOG_TESTBENCH_PHY, AERIAL_TESTBENCH_EVENT, "Invalid longPattern value in cuPHYTestWorker init: {}\n", m_longPattern);
    }

    if(m_nPDCCHCells == 0 && m_nCSIRSCells > 0)
    {
        NVLOGF_FMT(NVLOG_TESTBENCH_PHY, AERIAL_TESTBENCH_EVENT, "CSI-RS can only be run with PDCCH: m_nPDCCHCells = {} , m_nCSIRSCells = {}\n", m_nPDCCHCells, m_nCSIRSCells);
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

void cuPHYTestWorker::pdcchTxInit(std::vector<std::string> inFileNamesPdcchTx, uint32_t pdcch_nItrsPerStrm, bool group_cells, uint32_t cells_per_stream, bool ref_check, uint64_t pdcch_proc_mode, bool waitRsp)
{
    // set pdcch configuration
    m_pdcch_nItrsPerStrm  = pdcch_nItrsPerStrm;
    m_ref_check_pdcch     = ref_check;
    m_pdcch_group_cells   = group_cells;
    m_nCellsPerStrm_pdcch = cells_per_stream;
    m_pdcch_proc_mode     = pdcch_proc_mode;
    m_nStrms_pdcch        = (group_cells) ? 1 : m_nPDCCHCells; // NB: should not use m_nStrms if no grouping

    // Send initialization message
    auto shPtrPayload         = std::make_shared<commnTestInitMsgPayload>();
    shPtrPayload->rsp         = true;
    shPtrPayload->inFileNames = inFileNamesPdcchTx;

    if(inFileNamesPdcchTx.size() == 0)
    {
        m_runPDCCH = false;
        return;
    }
    m_runPDCCH    = true;
    auto shPtrMsg = std::make_shared<testWrkrCmdMsg>(CUPHY_TEST_WRKR_CMD_MSG_PDCCH_INIT, m_wrkrId, shPtrPayload);
    DEBUG_TRACE("MainThread [tid %s][currCtxId 0x%0lx]: Sending message: %s\n", getThreadIdStr().c_str(), getCurrCuCtxId(), TEST_WRKR_CMD_MSG_TO_STR[shPtrMsg->type]);
    m_shPtrCmdQ->send(shPtrMsg);
    // wait for response
    if(waitRsp)
    {
        std::shared_ptr<testWrkrRspMsg> shPtrRsp;
        m_shPtrRspQ->receive(shPtrRsp, CUPHY_TEST_WRKR_RSP_MSG_PDCCH_INIT, m_wrkrId);
    }
}

void cuPHYTestWorker::puschRxInit(std::vector<std::string> inFileNamesPuschRx, uint32_t fp16Mode, int puschRxDescramblingOn, bool printCbErrors, uint64_t pusch_proc_mode, bool enableLdpcThroughputMode, bool groupCells, maxPUSCHPrms puschPrms, uint32_t ldpcLaunchMode, uint8_t* puschSubslotProcFlag, bool waitRsp)
{
    // pusch configuration
    if(inFileNamesPuschRx.size() == 0)
    {
        m_runPUSCH = false;
        return;
    }

    m_fp16Mode                = fp16Mode;
    m_descramblingOn          = puschRxDescramblingOn;
    m_printCbErrors           = printCbErrors;
    m_pusch_proc_mode         = pusch_proc_mode;
    m_ldpc_kernel_launch_mode = ldpcLaunchMode;
    m_pusch_group_cells       = groupCells;
    m_puschProcModeSubslotProcFlag  = puschSubslotProcFlag[0];
    m_pusch2ProcModeSubslotProcFlag = puschSubslotProcFlag[1]; // if there is no PUSCH2, this indicator is not used

    // Send initialization message
    auto shPtrPayload                      = std::make_shared<cuPHYTestPuschRxInitMsgPayload>();
    shPtrPayload->rsp                      = true;
    shPtrPayload->enableLdpcThroughputMode = enableLdpcThroughputMode;
    shPtrPayload->inFileNames              = inFileNamesPuschRx;
    shPtrPayload->puschPrms                = puschPrms;

    m_runPUSCH = true;
    m_GPUtimeUl_d = std::move(cuphy::buffer<uint64_t, cuphy::device_alloc>(1)); // used for timing in PUSCH/PUCCH/PRACH/PDSCH

    auto shPtrMsg = std::make_shared<testWrkrCmdMsg>(CUPHY_TEST_WRKR_CMD_MSG_PUSCH_INIT, m_wrkrId, shPtrPayload);
    DEBUG_TRACE("MainThread [tid %s][currCtxId 0x%0lx]: Sending message: %s\n", getThreadIdStr().c_str(), getCurrCuCtxId(), TEST_WRKR_CMD_MSG_TO_STR[shPtrMsg->type]);
    m_shPtrCmdQ->send(shPtrMsg);

    // wait for response
    if(waitRsp)
    {
        std::shared_ptr<testWrkrRspMsg> shPtrRsp;
        m_shPtrRspQ->receive(shPtrRsp, CUPHY_TEST_WRKR_RSP_MSG_PUSCH_INIT, m_wrkrId);
    }
}

void cuPHYTestWorker::pdschTxInit(std::vector<std::string> inFileNamesPdschTx, uint32_t pdsch_nItrsPerStrm, bool ref_check_pdsch, bool identical_ldpc_configs, cuphyPdschProcMode_t pdsch_proc_mode, bool group_cells, uint32_t cells_per_stream, maxPDSCHPrms pdschPrms, bool waitRsp)
{
    if(inFileNamesPdschTx.size() == 0)
    {
        m_runPDSCH = false;
        return;
    }
    m_runPDSCH = true;
    // set pdsch configuration
    m_pdsch_nItrsPerStrm     = pdsch_nItrsPerStrm;
    m_ref_check_pdsch        = ref_check_pdsch;
    m_identical_ldpc_configs = identical_ldpc_configs;
    m_pdsch_proc_mode        = pdsch_proc_mode;
    m_pdsch_group_cells      = group_cells;
    m_nCellsPerStrm_pdsch    = cells_per_stream;
    m_nStrms_pdsch           = (group_cells) ? 1 : m_nStrms;

    // pack initialization message
    auto shPtrPayload         = std::make_shared<cuPHYTestPdschTxInitMsgPayload>();
    shPtrPayload->rsp         = true;
    shPtrPayload->inFileNames = inFileNamesPdschTx;
    shPtrPayload->pdschPrms   = pdschPrms;

    // send initalization message
    auto shPtrMsg = std::make_shared<testWrkrCmdMsg>(CUPHY_TEST_WRKR_CMD_MSG_PDSCH_INIT, m_wrkrId, shPtrPayload);
    DEBUG_TRACE("MainThread [tid %s][currCtxId 0x%0lx]: Sending message: %s\n", getThreadIdStr().c_str(), getCurrCuCtxId(), TEST_WRKR_CMD_MSG_TO_STR[shPtrMsg->type]);

    m_shPtrCmdQ->send(shPtrMsg);

    // wait for response
    if(waitRsp)
    {
        std::shared_ptr<testWrkrRspMsg> shPtrRsp;
        m_shPtrRspQ->receive(shPtrRsp, CUPHY_TEST_WRKR_RSP_MSG_PDSCH_INIT, m_wrkrId);
    }
}

void cuPHYTestWorker::csirsInit(std::vector<std::string> inFileNamesCSIRS, uint32_t csirs_nItrsPerStrm, bool ref_check_csirs, bool group_cells, uint64_t csirs_proc_mode, bool waitRsp)
{
    if(inFileNamesCSIRS.size() == 0)
    {
        m_runCSIRS = false;
        return;
    }
    m_runCSIRS = true;
    m_csirs_nItrsPerStrm = csirs_nItrsPerStrm;

    m_ref_check_csirs   = ref_check_csirs;
    m_pdsch_group_cells = group_cells;
    m_csirs_proc_mode   = csirs_proc_mode;

    // pack initialization message
    auto shPtrPayload         = std::make_shared<commnTestInitMsgPayload>();
    shPtrPayload->rsp         = true;
    shPtrPayload->inFileNames = inFileNamesCSIRS;

    // send initalization message
    auto shPtrMsg = std::make_shared<testWrkrCmdMsg>(CUPHY_TEST_WRKR_CMD_MSG_CSIRS_INIT, m_wrkrId, shPtrPayload);

    DEBUG_TRACE("MainThread [tid %s][currCtxId 0x%0lx]: Sending message: %s\n", getThreadIdStr().c_str(), getCurrCuCtxId(), TEST_WRKR_CMD_MSG_TO_STR[shPtrMsg->type]);

    m_shPtrCmdQ->send(shPtrMsg);
    // wait for response
    if(waitRsp)
    {
        std::shared_ptr<testWrkrRspMsg> shPtrRsp;
        m_shPtrRspQ->receive(shPtrRsp, CUPHY_TEST_WRKR_RSP_MSG_CSIRS_INIT, m_wrkrId);
    }
}

void cuPHYTestWorker::ssbInit(std::vector<std::string> inFileNamesSSB, uint32_t pbch_nItrsPerStrm, bool ref_check_ssb, bool group_cells, uint64_t ssb_proc_mode, bool waitRsp)
{
    if(inFileNamesSSB.size() == 0)
    {
        m_runSSB = false;
        return;
    }
    m_runSSB = true;

    m_pbch_nItrsPerStrm = pbch_nItrsPerStrm;
    m_pdsch_group_cells = group_cells;
    m_ref_check_ssb     = ref_check_ssb;
    m_ssb_proc_mode     = ssb_proc_mode;

    // pack initialization message
    auto shPtrPayload         = std::make_shared<commnTestInitMsgPayload>();
    shPtrPayload->rsp         = true;
    shPtrPayload->inFileNames = inFileNamesSSB;

    // send initalization message
    auto shPtrMsg = std::make_shared<testWrkrCmdMsg>(CUPHY_TEST_WRKR_CMD_MSG_SSB_INIT, m_wrkrId, shPtrPayload);

    DEBUG_TRACE("MainThread [tid %s][currCtxId 0x%0lx]: Sending message: %s\n", getThreadIdStr().c_str(), getCurrCuCtxId(), TEST_WRKR_CMD_MSG_TO_STR[shPtrMsg->type]);

    m_shPtrCmdQ->send(shPtrMsg);
    // wait for response
    if(waitRsp)
    {
        std::shared_ptr<testWrkrRspMsg> shPtrRsp;
        m_shPtrRspQ->receive(shPtrRsp, CUPHY_TEST_WRKR_RSP_MSG_SSB_INIT, m_wrkrId);
    }
}

void cuPHYTestWorker::prachInit(std::vector<std::string> inFileNamesPRACH, uint64_t proc_mode, bool ref_check_prach, bool group_cells, uint32_t cells_per_stream, bool waitRsp)
{
    if(inFileNamesPRACH.size() == 0)
    {
        m_runPRACH = false;
        return;
    }
    m_runPRACH = true;
    m_prach_proc_mode = proc_mode;

    m_prach_group_cells   = group_cells;
    m_ref_check_prach     = ref_check_prach;
    m_nCellsPerStrm_prach = cells_per_stream;
    m_nStrms_prach        = (group_cells) ? 1 : m_nPRACHCells; // NB: should not use m_nStrms if no grouping

    // pack initialization message
    auto shPtrPayload         = std::make_shared<commnTestInitMsgPayload>();
    shPtrPayload->rsp         = true;
    shPtrPayload->inFileNames = inFileNamesPRACH;

    // send initalization message
    auto shPtrMsg = std::make_shared<testWrkrCmdMsg>(CUPHY_TEST_WRKR_CMD_MSG_PRACH_INIT, m_wrkrId, shPtrPayload);
    DEBUG_TRACE("MainThread [tid %s][currCtxId 0x%0lx]: Sending message: %s\n", getThreadIdStr().c_str(), getCurrCuCtxId(), TEST_WRKR_CMD_MSG_TO_STR[shPtrMsg->type]);

    m_shPtrCmdQ->send(shPtrMsg);
    // wait for response
    if(waitRsp)
    {
        std::shared_ptr<testWrkrRspMsg> shPtrRsp;
        m_shPtrRspQ->receive(shPtrRsp, CUPHY_TEST_WRKR_RSP_MSG_PRACH_INIT, m_wrkrId);
    }
}

void cuPHYTestWorker::pucchRxInit(std::vector<std::string> inFileNamesPUCCH, bool ref_check_pucch, bool groupCells, uint64_t pucch_proc_mode, bool waitRsp)
{
    if(inFileNamesPUCCH.size() == 0)
    {
        m_runPUCCH = false;
        return;
    }
    m_runPUCCH        = true;
    m_pucch_proc_mode = pucch_proc_mode;

    m_pucch_group_cells   = groupCells;
    m_ref_check_pucch     = ref_check_pucch;
    m_nCellsPerStrm_pucch = groupCells ? inFileNamesPUCCH.size() : 1;

    // pack initialization message
    auto shPtrPayload         = std::make_shared<commnTestInitMsgPayload>();
    shPtrPayload->rsp         = true;
    shPtrPayload->inFileNames = inFileNamesPUCCH;

    // send initalization message
    auto shPtrMsg = std::make_shared<testWrkrCmdMsg>(CUPHY_TEST_WRKR_CMD_MSG_PUCCH_INIT, m_wrkrId, shPtrPayload);
    DEBUG_TRACE("MainThread [tid %s][currCtxId 0x%0lx]: Sending message: %s\n", getThreadIdStr().c_str(), getCurrCuCtxId(), TEST_WRKR_CMD_MSG_TO_STR[shPtrMsg->type]);

    m_shPtrCmdQ->send(shPtrMsg);
    // wait for response
    if(waitRsp)
    {
        std::shared_ptr<testWrkrRspMsg> shPtrRsp;
        m_shPtrRspQ->receive(shPtrRsp, CUPHY_TEST_WRKR_RSP_MSG_PUCCH_INIT, m_wrkrId);
    }
}

void cuPHYTestWorker::srsInit(std::vector<std::string> inFileNamesSRS, bool ref_check_srs, uint64_t srs_proc_mode, bool splitSRS, bool waitRsp)
{
    if(inFileNamesSRS.size() == 0)
    {
        m_runSRS  = false;
        m_runSRS2 = false;
        return;
    }
    m_srs_proc_mode = srs_proc_mode;

    // pack initialization message
    auto shPtrPayload         = std::make_shared<commnTestInitMsgPayload>();
    shPtrPayload->rsp         = true;
    shPtrPayload->inFileNames = inFileNamesSRS;
    m_runSRS                  = true;
    m_ref_check_srs           = ref_check_srs;

    if(splitSRS)
        m_runSRS2 = true;

    // send initalization message
    auto shPtrMsg = std::make_shared<testWrkrCmdMsg>(CUPHY_TEST_WRKR_CMD_MSG_SRS_INIT, m_wrkrId, shPtrPayload);
    DEBUG_TRACE("MainThread [tid %s][currCtxId 0x%0lx]: Sending message: %s\n", getThreadIdStr().c_str(), getCurrCuCtxId(), TEST_WRKR_CMD_MSG_TO_STR[shPtrMsg->type]);

    m_shPtrCmdQ->send(shPtrMsg);
    // wait for response
    if(waitRsp)
    {
        std::shared_ptr<testWrkrRspMsg> shPtrRsp;
        m_shPtrRspQ->receive(shPtrRsp, CUPHY_TEST_WRKR_RSP_MSG_SRS_INIT, m_wrkrId);
    }
}

void cuPHYTestWorker::csirsSetup(std::vector<std::string> inFileNamesCSIRS, uint8_t* csirsSlotRunFlag, bool waitRsp)
{
    if(inFileNamesCSIRS.size() == 0)
    {
        m_runCSIRS = false;
        return;
    }
    m_runCSIRS = true;
    m_csirsSlotRunFlag = csirsSlotRunFlag;

    // pack initialization message
    auto shPtrPayload         = std::make_shared<commnTestInitMsgPayload>();
    shPtrPayload->rsp         = true;
    shPtrPayload->inFileNames = inFileNamesCSIRS;

    // send initalization message
    auto shPtrMsg = std::make_shared<testWrkrCmdMsg>(CUPHY_TEST_WRKR_CMD_MSG_CSIRS_SETUP, m_wrkrId, shPtrPayload);
    DEBUG_TRACE("MainThread [tid %s][currCtxId 0x%0lx]: Sending message: %s\n", getThreadIdStr().c_str(), getCurrCuCtxId(), TEST_WRKR_CMD_MSG_TO_STR[shPtrMsg->type]);

    m_shPtrCmdQ->send(shPtrMsg);
    // wait for response
    if(waitRsp)
    {
        std::shared_ptr<testWrkrRspMsg> shPtrRsp;
        m_shPtrRspQ->receive(shPtrRsp, CUPHY_TEST_WRKR_RSP_MSG_CSIRS_SETUP, m_wrkrId);
    }
}

void cuPHYTestWorker::ssbSetup(std::vector<std::string> inFileNamesSSB, uint8_t* pbchSlotRunFlag, bool waitRsp)
{
    if(inFileNamesSSB.size() == 0)
    {
        m_runSSB = false;
        return;
    }
    m_runSSB = true;
    m_pbchSlotRunFlag = pbchSlotRunFlag;

    // pack initialization message
    auto shPtrPayload         = std::make_shared<commnTestInitMsgPayload>();
    shPtrPayload->rsp         = true;
    shPtrPayload->inFileNames = inFileNamesSSB;

    // send initalization message
    auto shPtrMsg = std::make_shared<testWrkrCmdMsg>(CUPHY_TEST_WRKR_CMD_MSG_SSB_SETUP, m_wrkrId, shPtrPayload);
    DEBUG_TRACE("MainThread [tid %s][currCtxId 0x%0lx]: Sending message: %s\n", getThreadIdStr().c_str(), getCurrCuCtxId(), TEST_WRKR_CMD_MSG_TO_STR[shPtrMsg->type]);

    m_shPtrCmdQ->send(shPtrMsg);
    // wait for response
    if(waitRsp)
    {
        std::shared_ptr<testWrkrRspMsg> shPtrRsp;
        m_shPtrRspQ->receive(shPtrRsp, CUPHY_TEST_WRKR_RSP_MSG_SSB_SETUP, m_wrkrId);
    }
}

void cuPHYTestWorker::pucchRxSetup(std::vector<std::string> inFileNamesPUCCH, bool waitRsp)
{
    if(inFileNamesPUCCH.size() == 0)
    {
        m_runPUCCH = false;
        return;
    }
    // pack initialization message
    auto shPtrPayload         = std::make_shared<commnTestInitMsgPayload>();
    shPtrPayload->rsp         = true;
    shPtrPayload->inFileNames = inFileNamesPUCCH;
    m_runPUCCH                = true;

    // send initalization message
    auto shPtrMsg = std::make_shared<testWrkrCmdMsg>(CUPHY_TEST_WRKR_CMD_MSG_PUCCH_SETUP, m_wrkrId, shPtrPayload);
    DEBUG_TRACE("MainThread [tid %s][currCtxId 0x%0lx]: Sending message: %s\n", getThreadIdStr().c_str(), getCurrCuCtxId(), TEST_WRKR_CMD_MSG_TO_STR[shPtrMsg->type]);

    m_shPtrCmdQ->send(shPtrMsg);
    // wait for response
    if(waitRsp)
    {
        std::shared_ptr<testWrkrRspMsg> shPtrRsp;
        m_shPtrRspQ->receive(shPtrRsp, CUPHY_TEST_WRKR_RSP_MSG_PUCCH_SETUP, m_wrkrId);
    }
}

void cuPHYTestWorker::pdcchTxSetup(std::vector<std::string> inFileNamesPDCCH, uint8_t* pdcchSlotRunFlag, bool waitRsp)
{
    if(inFileNamesPDCCH.size() == 0)
    {
        m_runPDCCH = false;
        return;
    }
    // pack initialization message
    auto shPtrPayload         = std::make_shared<commnTestInitMsgPayload>();
    shPtrPayload->rsp         = true;
    shPtrPayload->inFileNames = inFileNamesPDCCH;
    m_runPDCCH                = true;
    m_pdcchSlotRunFlag        = pdcchSlotRunFlag;

    // send initalization message
    auto shPtrMsg = std::make_shared<testWrkrCmdMsg>(CUPHY_TEST_WRKR_CMD_MSG_PDCCH_SETUP, m_wrkrId, shPtrPayload);
    DEBUG_TRACE("MainThread [tid %s][currCtxId 0x%0lx]: Sending message: %s\n", getThreadIdStr().c_str(), getCurrCuCtxId(), TEST_WRKR_CMD_MSG_TO_STR[shPtrMsg->type]);

    m_shPtrCmdQ->send(shPtrMsg);
    // wait for response
    if(waitRsp)
    {
        std::shared_ptr<testWrkrRspMsg> shPtrRsp;
        m_shPtrRspQ->receive(shPtrRsp, CUPHY_TEST_WRKR_RSP_MSG_PDCCH_SETUP, m_wrkrId);
    }
}

void cuPHYTestWorker::prachSetup(std::vector<std::string> inFileNamesPRACH, bool waitRsp)
{
    if(inFileNamesPRACH.size() == 0)
    {
        m_runPRACH = false;
        return;
    }
    // pack initialization message
    auto shPtrPayload         = std::make_shared<commnTestInitMsgPayload>();
    shPtrPayload->rsp         = true;
    shPtrPayload->inFileNames = inFileNamesPRACH;
    m_runPRACH                = true;
    // send initalization message
    auto shPtrMsg = std::make_shared<testWrkrCmdMsg>(CUPHY_TEST_WRKR_CMD_MSG_PRACH_SETUP, m_wrkrId, shPtrPayload);
    DEBUG_TRACE("MainThread [tid %s][currCtxId 0x%0lx]: Sending message: %s\n", getThreadIdStr().c_str(), getCurrCuCtxId(), TEST_WRKR_CMD_MSG_TO_STR[shPtrMsg->type]);

    m_shPtrCmdQ->send(shPtrMsg);
    // wait for response
    if(waitRsp)
    {
        std::shared_ptr<testWrkrRspMsg> shPtrRsp;
        m_shPtrRspQ->receive(shPtrRsp, CUPHY_TEST_WRKR_RSP_MSG_PRACH_SETUP, m_wrkrId);
    }
}

void cuPHYTestWorker::dlbfwInit(std::vector<std::string> inFileNamesDLBFW, uint32_t dlbfw_nItrsPerStrm, bool ref_check_bfw, bool waitRsp)
{
    // pack initialization message
    auto shPtrPayload         = std::make_shared<commnTestInitMsgPayload>();
    shPtrPayload->rsp         = true;
    shPtrPayload->inFileNames = inFileNamesDLBFW;
    m_dlbfw_nItrsPerStrm      = dlbfw_nItrsPerStrm;
    if(inFileNamesDLBFW.size() == 0)
    {
        m_runDlbfw = false;
        return;
    }
    m_runDlbfw        = true;
    m_ref_check_dlbfw = ref_check_bfw;
    // send initalization message
    auto shPtrMsg = std::make_shared<testWrkrCmdMsg>(CUPHY_TEST_WRKR_CMD_MSG_DLBFW_INIT, m_wrkrId, shPtrPayload);
    DEBUG_TRACE("MainThread [tid %s][currCtxId 0x%0lx]: Sending message: %s\n", getThreadIdStr().c_str(), getCurrCuCtxId(), TEST_WRKR_CMD_MSG_TO_STR[shPtrMsg->type]);

    m_shPtrCmdQ->send(shPtrMsg);
    // wait for response
    if(waitRsp)
    {
        std::shared_ptr<testWrkrRspMsg> shPtrRsp;
        m_shPtrRspQ->receive(shPtrRsp, CUPHY_TEST_WRKR_RSP_MSG_DLBFW_INIT, m_wrkrId);
    }
}

void cuPHYTestWorker::dlbfwSetup(std::vector<std::string> inFileNamesDLBFW, bool waitRsp)
{
    // pack initialization message
    auto shPtrPayload         = std::make_shared<commnTestInitMsgPayload>();
    shPtrPayload->rsp         = true;
    shPtrPayload->inFileNames = inFileNamesDLBFW;
    if(inFileNamesDLBFW.size() == 0)
    {
        m_runDlbfw = false;
        return;
    }
    m_runDlbfw = true;
    // send initalization message
    auto shPtrMsg = std::make_shared<testWrkrCmdMsg>(CUPHY_TEST_WRKR_CMD_MSG_DLBFW_SETUP, m_wrkrId, shPtrPayload);
    DEBUG_TRACE("MainThread [tid %s][currCtxId 0x%0lx]: Sending message: %s\n", getThreadIdStr().c_str(), getCurrCuCtxId(), TEST_WRKR_CMD_MSG_TO_STR[shPtrMsg->type]);

    m_shPtrCmdQ->send(shPtrMsg);
    // wait for response
    if(waitRsp)
    {
        std::shared_ptr<testWrkrRspMsg> shPtrRsp;
        m_shPtrRspQ->receive(shPtrRsp, CUPHY_TEST_WRKR_RSP_MSG_DLBFW_SETUP, m_wrkrId);
    }
}

void cuPHYTestWorker::ulbfwInit(std::vector<std::string> inFileNamesULBFW, bool ref_check_bfw, cuphyPdschProcMode_t pdsch_proc_mode, bool waitRsp)
{
    // pack initialization message
    auto shPtrPayload         = std::make_shared<commnTestInitMsgPayload>();
    shPtrPayload->rsp         = true;
    shPtrPayload->inFileNames = inFileNamesULBFW;
    if(inFileNamesULBFW.size() == 0)
    {
        m_runUlbfw = false;
        return;
    }
    m_runUlbfw        = true;
    m_ref_check_ulbfw = ref_check_bfw;
    m_pdsch_proc_mode = pdsch_proc_mode;
    // send initalization message
    auto shPtrMsg = std::make_shared<testWrkrCmdMsg>(CUPHY_TEST_WRKR_CMD_MSG_ULBFW_INIT, m_wrkrId, shPtrPayload);
    DEBUG_TRACE("MainThread [tid %s][currCtxId 0x%0lx]: Sending message: %s\n", getThreadIdStr().c_str(), getCurrCuCtxId(), TEST_WRKR_CMD_MSG_TO_STR[shPtrMsg->type]);

    m_shPtrCmdQ->send(shPtrMsg);
    // wait for response
    if(waitRsp)
    {
        std::shared_ptr<testWrkrRspMsg> shPtrRsp;
        m_shPtrRspQ->receive(shPtrRsp, CUPHY_TEST_WRKR_RSP_MSG_ULBFW_INIT, m_wrkrId);
    }
}

void cuPHYTestWorker::ulbfwSetup(std::vector<std::string> inFileNamesULBFW, bool waitRsp)
{
    // pack initialization message
    auto shPtrPayload         = std::make_shared<commnTestInitMsgPayload>();
    shPtrPayload->rsp         = true;
    shPtrPayload->inFileNames = inFileNamesULBFW;
    if(inFileNamesULBFW.size() == 0)
    {
        m_runUlbfw = false;
        return;
    }
    m_runUlbfw = true;
    // send initalization message
    auto shPtrMsg = std::make_shared<testWrkrCmdMsg>(CUPHY_TEST_WRKR_CMD_MSG_ULBFW_SETUP, m_wrkrId, shPtrPayload);
    DEBUG_TRACE("MainThread [tid %s][currCtxId 0x%0lx]: Sending message: %s\n", getThreadIdStr().c_str(), getCurrCuCtxId(), TEST_WRKR_CMD_MSG_TO_STR[shPtrMsg->type]);

    m_shPtrCmdQ->send(shPtrMsg);
    // wait for response
    if(waitRsp)
    {
        std::shared_ptr<testWrkrRspMsg> shPtrRsp;
        m_shPtrRspQ->receive(shPtrRsp, CUPHY_TEST_WRKR_RSP_MSG_ULBFW_SETUP, m_wrkrId);
    }
}

void cuPHYTestWorker::deinit(bool waitRsp)
{
    auto shPtrPayload = std::make_shared<cuPHYTestDeinitMsgPayload>();
    shPtrPayload->rsp = true;

    // Cleanup
    auto shPtrMsg = std::make_shared<testWrkrCmdMsg>(CUPHY_TEST_WRKR_CMD_MSG_DEINIT, m_wrkrId, shPtrPayload);
    DEBUG_TRACE("MainThread [tid %s][currCtxId 0x%0lx]: Sending message: %s\n", getThreadIdStr().c_str(), getCurrCuCtxId(), TEST_WRKR_CMD_MSG_TO_STR[shPtrMsg->type]);

    m_shPtrCmdQ->send(shPtrMsg);

    if(waitRsp)
    {
        std::shared_ptr<testWrkrRspMsg> shPtrRsp;
        m_shPtrRspQ->receive(shPtrRsp, CUPHY_TEST_WRKR_RSP_MSG_DEINIT, m_wrkrId);
    }
}

void cuPHYTestWorker::puschRxSetup(std::vector<std::string> inFileNamesPuschRx, bool waitRsp)
{
    if(inFileNamesPuschRx.size() == 0)
    {
        m_runPUSCH = false;
        return;
    }

    if(inFileNamesPuschRx.size() != 0)
    {
        // send message
        auto shPtrPayload                = std::make_shared<cuPHYTestPuschRxSetupMsgPayload>();
        shPtrPayload->rsp                = true;
        shPtrPayload->inFileNamesPuschRx = inFileNamesPuschRx;

        auto shPtrMsg = std::make_shared<testWrkrCmdMsg>(CUPHY_TEST_WRKR_CMD_MSG_PUSCH_SETUP, m_wrkrId, shPtrPayload);
        DEBUG_TRACE("MainThread [tid %s][currCtxId 0x%0lx]: Sending message: %s\n", getThreadIdStr().c_str(), getCurrCuCtxId(), TEST_WRKR_CMD_MSG_TO_STR[shPtrMsg->type]);

        m_shPtrCmdQ->send(shPtrMsg);
    }
    // wait for response
    if(waitRsp)
    {
        std::shared_ptr<testWrkrRspMsg> shPtrRsp;
        m_shPtrRspQ->receive(shPtrRsp, CUPHY_TEST_WRKR_RSP_MSG_PUSCH_SETUP, m_wrkrId);
    }
}

void cuPHYTestWorker::srsSetup(std::vector<std::string> inFileNamesSRS, bool waitRsp)
{
    if(inFileNamesSRS.size() == 0)
    {
        m_runSRS = false;
        return;
    }
    // pack initialization message
    auto shPtrPayload         = std::make_shared<commnTestInitMsgPayload>();
    shPtrPayload->rsp         = true;
    shPtrPayload->inFileNames = inFileNamesSRS;
    // send initalization message
    auto shPtrMsg = std::make_shared<testWrkrCmdMsg>(CUPHY_TEST_WRKR_CMD_MSG_SRS_SETUP, m_wrkrId, shPtrPayload);
    DEBUG_TRACE("MainThread [tid %s][currCtxId 0x%0lx]: Sending message: %s\n", getThreadIdStr().c_str(), getCurrCuCtxId(), TEST_WRKR_CMD_MSG_TO_STR[shPtrMsg->type]);

    m_shPtrCmdQ->send(shPtrMsg);
    // wait for response
    if(waitRsp)
    {
        std::shared_ptr<testWrkrRspMsg> shPtrRsp;
        m_shPtrRspQ->receive(shPtrRsp, CUPHY_TEST_WRKR_RSP_MSG_SRS_SETUP, m_wrkrId);
    }
}
void cuPHYTestWorker::pdschTxSetup(std::vector<std::string> inFileNamesPdschTx, uint8_t* pdschSlotRunFlag, bool waitRsp)
{
    if(inFileNamesPdschTx.size() == 0)
    {
        m_runPDSCH = false;
        return;
    }
    m_pdschSlotRunFlag = pdschSlotRunFlag;

    // send message
    auto shPtrPayload                = std::make_shared<cuPHYTestPdschTxSetupMsgPayload>();
    shPtrPayload->rsp                = true;
    shPtrPayload->inFileNamesPdschTx = inFileNamesPdschTx;

    auto shPtrMsg = std::make_shared<testWrkrCmdMsg>(CUPHY_TEST_WRKR_CMD_MSG_PDSCH_SETUP, m_wrkrId, shPtrPayload);
    DEBUG_TRACE("MainThread [tid %s][currCtxId 0x%0lx]: Sending message: %s\n", getThreadIdStr().c_str(), getCurrCuCtxId(), TEST_WRKR_CMD_MSG_TO_STR[shPtrMsg->type]);

    m_shPtrCmdQ->send(shPtrMsg);
    // wait for response
    if(waitRsp)
    {
        std::shared_ptr<testWrkrRspMsg> shPtrRsp;
        m_shPtrRspQ->receive(shPtrRsp, CUPHY_TEST_WRKR_RSP_MSG_PDSCH_SETUP, m_wrkrId);
    }
}

void cuPHYTestWorker::puschRxRun(cudaEvent_t startEvent, std::shared_ptr<cuphy::event>& shPtrStopEvent, cudaEvent_t prachStartEvent, cudaEvent_t pucchStartEvent, bool waitRsp)
{
    auto shPtrPayload             = std::make_shared<cuPHYTestPuschRxRunMsgPayload>();
    shPtrPayload->rsp             = true;
    shPtrPayload->startEvent      = startEvent;
    shPtrPayload->prachStartEvent = prachStartEvent;
    shPtrPayload->pucchStartEvent = pucchStartEvent;

    auto shPtrMsg = std::make_shared<testWrkrCmdMsg>(CUPHY_TEST_WRKR_CMD_MSG_PUSCH_RUN, m_wrkrId, shPtrPayload);
    DEBUG_TRACE("MainThread [tid %s][currCtxId 0x%0lx]: Sending message: %s\n", getThreadIdStr().c_str(), getCurrCuCtxId(), TEST_WRKR_CMD_MSG_TO_STR[shPtrMsg->type]);

    m_shPtrCmdQ->send(shPtrMsg);

    if(waitRsp)
    {
        std::shared_ptr<testWrkrRspMsg> shPtrRsp;
        getPuschRxRunRsp(shPtrRsp);
        cuPHYTestPuschRxRunRspMsgPayload& puschRxRunRspMsgPayload = *std::static_pointer_cast<cuPHYTestPuschRxRunRspMsgPayload>(shPtrRsp->payload);
        shPtrStopEvent                                            = puschRxRunRspMsgPayload.shPtrStopEvent;
    }
}

void cuPHYTestWorker::pdschTxRun(cudaEvent_t startEvent, std::shared_ptr<cuphy::event>& shPtrStopEvent, bool waitRsp, std::vector<cuphy::event>* pdcchStopEventVec, std::vector<cuphy::event>* pdschInterSlotEventVec)
{
    auto shPtrPayload                    = std::make_shared<cuPHYTestPdschTxRunMsgPayload>();
    shPtrPayload->rsp                    = true;
    shPtrPayload->startEvent             = startEvent;
    shPtrPayload->pdcchStopEventVec      = pdcchStopEventVec;
    shPtrPayload->pdschInterSlotEventVec = pdschInterSlotEventVec;

    auto shPtrMsg = std::make_shared<testWrkrCmdMsg>(CUPHY_TEST_WRKR_CMD_MSG_PDSCH_RUN, m_wrkrId, shPtrPayload);
    DEBUG_TRACE("MainThread [tid %s][currCtxId 0x%0lx]: Sending message: %s\n", getThreadIdStr().c_str(), getCurrCuCtxId(), TEST_WRKR_CMD_MSG_TO_STR[shPtrMsg->type]);

    m_shPtrCmdQ->send(shPtrMsg);

    if(waitRsp)
    {
        std::shared_ptr<testWrkrRspMsg> shPtrRsp;
        getPdschTxRunRsp(shPtrRsp);

        cuPHYTestPdschTxRunRspMsgPayload& pdschTxRunRspMsgPayload = *std::static_pointer_cast<cuPHYTestPdschTxRunRspMsgPayload>(shPtrRsp->payload);
        shPtrStopEvent                                            = pdschTxRunRspMsgPayload.shPtrStopEvent;
    }
}

void cuPHYTestWorker::pschTxRxRun(cudaEvent_t startEvent, std::shared_ptr<cuphy::event>& shPtrStopEvent, bool waitRsp)
{
    auto shPtrPayload        = std::make_shared<cuPHYTestPschTxRxRunMsgPayload>();
    shPtrPayload->rsp        = true;
    shPtrPayload->startEvent = startEvent;

    auto shPtrMsg = std::make_shared<testWrkrCmdMsg>(CUPHY_TEST_WRKR_CMD_MSG_PSCH_RUN, m_wrkrId, shPtrPayload);
    DEBUG_TRACE("MainThread [tid %s][currCtxId 0x%0lx]: Sending message: %s\n", getThreadIdStr().c_str(), getCurrCuCtxId(), TEST_WRKR_CMD_MSG_TO_STR[shPtrMsg->type]);

    m_shPtrCmdQ->send(shPtrMsg);

    if(waitRsp)
    {
        std::shared_ptr<testWrkrRspMsg> shPtrRsp;
        getPschTxRxRunRsp(shPtrRsp);

        cuPHYTestPschRunRspMsgPayload& pschRunRspMsgPayload = *std::static_pointer_cast<cuPHYTestPschRunRspMsgPayload>(shPtrRsp->payload);
        shPtrStopEvent                                      = pschRunRspMsgPayload.shPtrStopEvent;
    }
}

void cuPHYTestWorker::getPuschRxRunRsp(std::shared_ptr<testWrkrRspMsg>& shPtrRsp)
{
    m_shPtrRspQ->receive(shPtrRsp, CUPHY_TEST_WRKR_RSP_MSG_PUSCH_RUN, m_wrkrId);
}

void cuPHYTestWorker::getPdschTxRunRsp(std::shared_ptr<testWrkrRspMsg>& shPtrRsp)
{
    m_shPtrRspQ->receive(shPtrRsp, CUPHY_TEST_WRKR_RSP_MSG_PDSCH_RUN, m_wrkrId);
}

void cuPHYTestWorker::getPschTxRxRunRsp(std::shared_ptr<testWrkrRspMsg>& shPtrRsp)
{
    m_shPtrRspQ->receive(shPtrRsp, CUPHY_TEST_WRKR_RSP_MSG_PSCH_RUN, m_wrkrId);
}

void cuPHYTestWorker::pdschTxClean(bool waitRsp)
{
    auto shPtrPayload = std::make_shared<cuPHYTestPdschTxCleanMsgPayload>();
    shPtrPayload->rsp = waitRsp;

    auto shPtrMsg = std::make_shared<testWrkrCmdMsg>(CUPHY_TEST_WRKR_CMD_MSG_PDSCH_CLEAN, m_wrkrId, shPtrPayload);
    DEBUG_TRACE("MainThread [tid %s][currCtxId 0x%0lx]: Sending message: %s\n", getThreadIdStr().c_str(), getCurrCuCtxId(), TEST_WRKR_CMD_MSG_TO_STR[shPtrMsg->type]);

    m_shPtrCmdQ->send(shPtrMsg);

    if(waitRsp)
    {
        std::shared_ptr<testWrkrRspMsg> shPtrRsp;
        m_shPtrRspQ->receive(shPtrRsp, CUPHY_TEST_WRKR_RSP_MSG_PDSCH_CLEAN, m_wrkrId);
    }
}

std::vector<float> cuPHYTestWorker::getDlbfwIterStartTimes()
{
    return m_totDlbfwIterStartTime;
}
std::vector<float> cuPHYTestWorker::getDlbfwIterTimes()
{
    return m_totRunTimeDlbfwItr;
}
std::vector<float> cuPHYTestWorker::getPdschIterTimes()
{
    return m_totRunTimePdschItr;
}
std::vector<float> cuPHYTestWorker::getPdschSlotStartTimes()
{
    return m_totPdschSlotStartTime;
}
std::vector<float> cuPHYTestWorker::getCSIRSStartTimes()
{
    return m_totCSIRSStartTimes;
}
std::vector<float> cuPHYTestWorker::getCSIRSIterTimes()
{
    return m_totRunTimeCSIRSItr;
}
std::vector<float> cuPHYTestWorker::getPdcchStartTimes()
{
    return m_totPdcchStartTimes;
}
std::vector<float> cuPHYTestWorker::getPdcchIterTimes()
{
    return m_totRunTimePdcchItr;
}
std::vector<float> cuPHYTestWorker::getTotSSBStartTime()
{
    return m_totSSBStartTime;
}
std::vector<float> cuPHYTestWorker::getTotSSBRunTime()
{
    return m_totRunTimeSSBItr;
}
float cuPHYTestWorker::getTotSRSStartTime()
{
    return m_totSRSStartTime;
}
float cuPHYTestWorker::getTotSRSRunTime()
{
    return m_totSRSRunTime;
}
float cuPHYTestWorker::getTotSRS2StartTime()
{
    return m_totSRS2StartTime;
}
float cuPHYTestWorker::getTotSRS2RunTime()
{
    return m_totSRS2RunTime;
}
float cuPHYTestWorker::getTotPrachStartTime()
{
    return m_totPRACHStartTime;
}
float cuPHYTestWorker::getTotPrachRunTime()
{
    return m_totPRACHRunTime;
}
float cuPHYTestWorker::getTotPuschStartTime()
{
    return m_totPUSCHStartTime;
}
float cuPHYTestWorker::getTotPuschSubslotProcRunTime()
{
    return m_totPUSCHSubslotProcRunTime;
}
float cuPHYTestWorker::getTotPuschRunTime()
{
    return m_totPUSCHRunTime;
}
float cuPHYTestWorker::getTotPusch2StartTime()
{
    return m_totPUSCH2StartTime;
}
float cuPHYTestWorker::getTotPusch2SubslotProcRunTime()
{
    return m_totPUSCH2SubslotProcRunTime;
}
float cuPHYTestWorker::getTotPusch2RunTime()
{
    return m_totPUSCH2RunTime;
}
float cuPHYTestWorker::getTotPucchStartTime()
{
    return m_totPUCCHStartTime;
}
float cuPHYTestWorker::getTotPucchRunTime()
{
    return m_totPUCCHRunTime;
}
float cuPHYTestWorker::getTotPucch2StartTime()
{
    return m_totPUCCH2StartTime;
}
float cuPHYTestWorker::getTotPucch2RunTime()
{
    return m_totPUCCH2RunTime;
}
float cuPHYTestWorker::getTotUlbfwStartTime()
{
    return m_totULBFWStartTime;
}
float cuPHYTestWorker::getTotUlbfwRunTime()
{
    return m_totULBFWRunTime;
}
float cuPHYTestWorker::getTotUlbfw2StartTime()
{
    return m_totULBFW2StartTime;
}
float cuPHYTestWorker::getTotUlbfw2RunTime()
{
    return m_totULBFW2RunTime;
}
cudaEvent_t cuPHYTestWorker::getPuschStartEvent()
{
    return m_uqPtrPuschDelayStopEvent->handle();
}
cudaEvent_t cuPHYTestWorker::getPusch1EndEvent()
{
    return m_uqPtrTimePUSCHEndEvent->handle();
}
cudaEvent_t cuPHYTestWorker::getPusch2StartEvent()
{
    if(m_longPattern)
    {
        if(m_longPattern < 4)
        {
            return m_uqPtrPusch2DelayStopEvent->handle();
        }
        else
            return m_stopEvents[0].handle();
    }
    else
    {
        return m_uqPtrPuschDelayStopEvent->handle();
    }
}
cudaEvent_t cuPHYTestWorker::getPucch2DelayStopEvent()
{
    if(m_longPattern)
    {
        return m_uqPtrPucch2DelayStopEvent->handle();
    }
    else
    {
        return m_uqPtrPusch2DelayStopEvent->handle();
    }
}
cudaEvent_t cuPHYTestWorker::getPrachStartEvent()
{
    return m_uqPtrTimePRACHStartEvent->handle();
}
cudaEvent_t cuPHYTestWorker::getPucchStartEvent()
{
    return m_uqPtrTimePUCCHStartEvent->handle();
}
std::vector<cuphy::event>* cuPHYTestWorker::getpdcchCsirsInterSlotEndEventVec()
{
    return &m_pdcchCsirsInterSlotEndEventVec;
}
std::vector<cuphy::event>* cuPHYTestWorker::getPdschInterSlotEventVecPtr()
{
    return &m_pdschInterSlotStartEventVec;
}

std::vector<cuphy::event>* cuPHYTestWorker::getSlotBoundaryEventVecPtr()
{
    return (!m_SlotBoundaryEventVec.empty() ? &m_SlotBoundaryEventVec : &m_pdschInterSlotStartEventVec);
}
