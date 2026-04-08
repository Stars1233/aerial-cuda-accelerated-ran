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

#include <string.h>
#include <sys/time.h>
#include <semaphore.h>

#include "nvlog.hpp"

#include "nv_phy_utils.hpp"
#include "cumac_app.hpp"
#include "cumac.h"
#include "cumac_cp_handler.hpp"
#include "api.h"

#include "nv_phy_mac_transport.hpp"
#include "nv_phy_epoll_context.hpp"

#include "cumac_cp_tv.hpp"
#include "cumac_cuda.hpp"

using namespace cumac;

#define TAG (NVLOG_TAG_BASE_CUMAC_CP + 5) // "CUMCP.TASK"

#define CPU_COPY_TEST 0

#define CHECK_VALUE_EQUAL_ERR(v1, v2)                                                                                                   \
    do                                                                                                                                  \
    {                                                                                                                                   \
        if ((v1) != (v2))                                                                                                               \
        {                                                                                                                               \
            NVLOGE_FMT(TAG, AERIAL_CUMAC_CP_EVENT, "{} line {}: values not equal: {}={}, {}={}", __func__, __LINE__, #v1, v1, #v2, v2); \
        }                                                                                                                               \
    } while (0);

#define CHECK_VALUE_MAX_ERR(val, max)                                                                                                    \
    do                                                                                                                                  \
    {                                                                                                                                   \
        if ((val) > (max))                                                                                                               \
        {                                                                                                                               \
            NVLOGE_FMT(TAG, AERIAL_CUMAC_CP_EVENT, "{} line {}: value > max: {}={} > {}={}", __func__, __LINE__, #val, static_cast<uint32_t>(val), #max, static_cast<uint32_t>(max)); \
        }                                                                                                                               \
    } while (0);

// Generate global unique task_id for cumac_task instances
static uint32_t task_instance_id = 0;

cumac_task::cumac_task()
{
    task_id = task_instance_id++;
    ss = {.u32 = SFN_SLOT_INVALID};

    CHECK_CUDA_ERR(cudaEventCreate(&ev_start));
    CHECK_CUDA_ERR(cudaEventCreate(&ev_copy1));
    CHECK_CUDA_ERR(cudaEventCreate(&ev_copy2));
    CHECK_CUDA_ERR(cudaEventCreate(&ev_setup_end));
    CHECK_CUDA_ERR(cudaEventCreate(&ev_setup));
    CHECK_CUDA_ERR(cudaEventCreate(&ev_run_start));
    CHECK_CUDA_ERR(cudaEventCreate(&ev_run1));
    CHECK_CUDA_ERR(cudaEventCreate(&ev_run2));
    CHECK_CUDA_ERR(cudaEventCreate(&ev_run3));
    CHECK_CUDA_ERR(cudaEventCreate(&ev_run4));
    CHECK_CUDA_ERR(cudaEventCreate(&ev_run_end));
    CHECK_CUDA_ERR(cudaEventCreate(&ev_callback_start));
    CHECK_CUDA_ERR(cudaEventCreate(&ev_callback_end));
    CHECK_CUDA_ERR(cudaEventCreate(&ev_debug));
}

void cumac_task::reset_cumac_task(sfn_slot_t _ss)
{
    ss = _ss;

    taskBitMask = 0;

    ts_enqueue = 0;
    ts_dequeue = 0;
    ts_setup = 0;
    ts_run = 0;
    ts_callback = 0;

    output_copy_time = 0;
    run_copy_time = 0;

    tm_setup = 0;
    tm_copy1 = 0;
    tm_copy2 = 0;
    tm_run = 0;
    tm_run1 = 0;
    tm_run2 = 0;
    tm_run3 = 0;
    tm_run4 = 0;
    tm_callback = 0;
    tm_total = 0;
    tm_debug = 0;

    enableHarq = 0;
    lightWeight = 0;
    halfPrecision = 0;
    baselineUlMcsInd = 0;
    CQI = 0;
    RI = 0;
    dlIndicator = 1;
    columnMajor = 1;

    cell_num = grpPrms.nCell;
    simParam.totNumCell = cell_num;

    grpPrms.nActiveUe = 0;

    NVLOGI_FMT(TAG, "SFN {}.{} reset nUe={} nActiveUe={} numUeSchdPerCellTTI={} nCell={} nPrbGrp={} nBsAnt={} nUeAnt={} precodingScheme={} receiverScheme={} allocType={} prioWeightStep={}", ss.u16.sfn, ss.u16.slot,
               grpPrms.nUe, grpPrms.nActiveUe, grpPrms.numUeSchdPerCellTTI, grpPrms.nCell, grpPrms.nPrbGrp, grpPrms.nBsAnt, grpPrms.nUeAnt, grpPrms.precodingScheme, grpPrms.receiverScheme, grpPrms.allocType, grpPrms.prioWeightStep);

    memset(&data_num, 0, sizeof(data_num));

    if (run_in_cpu)
    {
        memset(grpPrms.cellAssocActUe, 0, static_cast<size_t>(grpPrms.nCell) * grpPrms.nActiveUe);
        memset(grpPrms.cellAssoc, 0, static_cast<size_t>(grpPrms.nCell) * grpPrms.nUe);
    }
    else if (group_buf_enabled == 0)
    {
        CHECK_CUDA_ERR(cudaMemset(grpPrms.cellAssocActUe, 0, static_cast<size_t>(grpPrms.nCell) * grpPrms.nActiveUe));
        CHECK_CUDA_ERR(cudaMemset(grpPrms.cellAssoc, 0, static_cast<size_t>(grpPrms.nCell) * grpPrms.nUe));
    }

    // PFM sorting
    pfmSortTask.num_cell = cell_num;
    pfmSortTask.strm = strm;
    pfmSortTask.gpu_buf = reinterpret_cast<uint8_t*>(pfmCellInfo);
}

void cumac_task::calculate_output_data_num()
{
    if (grpPrms.nUe != grpPrms.numUeSchdPerCellTTI * grpPrms.nCell)
    {
        NVLOGE_FMT(TAG, AERIAL_CUMAC_CP_EVENT, "{}: invalid parameters: nUe={} nCell={} numUeSchdPerCellTTI={}", __func__, grpPrms.nUe, grpPrms.nCell, grpPrms.numUeSchdPerCellTTI);
    }
    data_num.avgRates = grpPrms.nUe;
    data_num.setSchdUePerCellTTI = grpPrms.nUe;
    data_num.allocSol = grpPrms.allocType == 1 ? grpPrms.nUe * 2 : grpPrms.nCell * grpPrms.nPrbGrp;
    data_num.layerSelSol = grpPrms.nUe;
    data_num.mcsSelSol = grpPrms.nUe;
    data_num.tbErrLast = grpPrms.nUe;
}

void cumac_task::init_cumac_modules()
{
    // Init cellAssocActUe buffer to 0
    if (run_in_cpu)
    {
        mcUeSelCpu = new cumac::multiCellUeSelectionCpu(&grpPrms);
        mcSchCpu = new cumac::multiCellSchedulerCpu(&grpPrms);
        mcLayerSelCpu = new cumac::multiCellLayerSelCpu(&grpPrms);
        mcMcsSelCpu = new cumac::mcsSelectionLUTCpu(&grpPrms);
    }
    else
    {
        mcUeSelGpu = new cumac::multiCellUeSelection(&grpPrms);
        mcSchGpu = new cumac::multiCellScheduler(&grpPrms);
        mcLayerSelGpu = new cumac::multiCellLayerSel(&grpPrms);
        mcMcsSelGpu = new cumac::mcsSelectionLUT(&grpPrms, strm);

        // Initialize pfmSort module object
        pfmSortGpu = new cumac::pfmSort();
    }
}

template <typename T>
int cumac_task::compare_array(const char *info, T *tv_buf, T *cumac_buf, uint32_t num, uint32_t data_in_cpu)
{
    int check_result = 0;
    if (tv_buf == nullptr || cumac_buf == nullptr)
    {
        NVLOGE_FMT(TAG, AERIAL_CUMAC_CP_EVENT, "ARRAY SFN {}.{} {} DIFF pointer is null: tv_buf=0x{} cumac_buf=0x{}", ss.u16.sfn, ss.u16.slot, info, tv_buf, cumac_buf);
        return -1;
    }

    size_t size = sizeof(T) * num;
    if (run_in_cpu == 0 && data_in_cpu == 0)
    {
        if (size > CUMAC_TASK_DEBUG_BUF_MAX_SIZE)
        {
            NVLOGE_FMT(TAG, AERIAL_CUMAC_CP_EVENT, "ARRAY SFN {}.{} {} DIFF debug_buffer size {} < required {}", ss.u16.sfn, ss.u16.slot, info, CUMAC_TASK_DEBUG_BUF_MAX_SIZE, size);
            return -1;
        }
        CHECK_CUDA_ERR(cudaMemcpy(debug_buffer, cumac_buf, size, cudaMemcpyDeviceToHost));
        cumac_buf = reinterpret_cast<T *>(debug_buffer);
    }

    if (memcmp(tv_buf, cumac_buf, size))
    {
        char info_str[64];
        T *v1 = reinterpret_cast<T *>(tv_buf);
        T *v2 = reinterpret_cast<T *>(cumac_buf);

        uint32_t i;
        for (i = 0; i < num; i++)
        {
            if (*(v1 + i) != *(v2 + i))
            {
                break;
            }
        }
        v1 += i;
        v2 += i;

        snprintf(info_str, 64, "ARRAY SFN %u.%u %s DIFF from %u TV", ss.u16.sfn, ss.u16.slot, info, i);
        NVLOGI_FMT_ARRAY(TAG, info_str, v1, num - i);

        snprintf(info_str, 64, "ARRAY SFN %u.%u %s DIFF from %u CUMAC", ss.u16.sfn, ss.u16.slot, info, i);
        NVLOGI_FMT_ARRAY(TAG, info_str, v2, num - i);

        check_result = -1;
    }

    if (check_result == 0) // SAME
    {
        NVLOGI_FMT(TAG, "ARRAY SFN {}.{} {}[{}] SAME", ss.u16.sfn, ss.u16.slot, info, num);
    }

    return check_result;
}

template <typename T>
void cumac_task::print_array(const char *info, T *array, uint32_t num, uint32_t buf_in_cpu)
{
    if (array == nullptr)
    {
        NVLOGE_FMT(TAG, AERIAL_CUMAC_CP_EVENT, "ARRAY SFN {}.{} {} pointer is null: array=0x{}", ss.u16.sfn, ss.u16.slot, info, array);
        return;
    }

    size_t size = sizeof(T) * num;

    bool in_gpu = run_in_cpu == 0 && buf_in_cpu == 0;
    if (in_gpu)
    {
        if (size > CUMAC_TASK_DEBUG_BUF_MAX_SIZE)
        {
            NVLOGE_FMT(TAG, AERIAL_CUMAC_CP_EVENT, "GPU ARRAY SFN {}.{} {} debug_buffer size {} < required {}", ss.u16.sfn, ss.u16.slot, info, CUMAC_TASK_DEBUG_BUF_MAX_SIZE, size);
            return;
        }
        CHECK_CUDA_ERR(cudaMemcpy(debug_buffer, array, size, cudaMemcpyDeviceToHost));
        array = reinterpret_cast<T *>(debug_buffer);
    }

    char info_str[64];
    snprintf(info_str, 64, "%s ARRAY SFN %u.%u %s", in_gpu ? "GPU" : "CPU", ss.u16.sfn, ss.u16.slot, info);
    NVLOGI_FMT_ARRAY(TAG, info_str, array, num);
}

int cumac_task::setup()
{
    NVLOGI_FMT(TAG, "SFN {}.{} setup in {}: nUe={} nActiveUe={} numUeSchdPerCellTTI={} nCell={} nPrbGrp={} nBsAnt={} nUeAnt={} precodingScheme={} receiverScheme={} allocType={} prioWeightStep={}", ss.u16.sfn, ss.u16.slot, run_in_cpu ? "CPU" : "GPU",
               grpPrms.nUe, grpPrms.nActiveUe, grpPrms.numUeSchdPerCellTTI, grpPrms.nCell, grpPrms.nPrbGrp, grpPrms.nBsAnt, grpPrms.nUeAnt, grpPrms.precodingScheme, grpPrms.receiverScheme, grpPrms.allocType, grpPrms.prioWeightStep);

    ts_copy = std::chrono::system_clock::now().time_since_epoch().count();
    if (run_in_cpu == 0)
    {
        CHECK_CUDA_ERR(cudaEventRecord(ev_start, strm));
    }

    for (uint32_t cell_id = 0; cell_id < grpPrms.nCell; cell_id++)
    {
        cp_handler->on_sch_tti_request(tti_reqs[cell_id], this);
        if (run_in_cpu != 0 || group_buf_enabled == 0)
        {
            cp_handler->cell_copy_task(tti_reqs[cell_id], this);
        }
    }

    cumac_buf_num_t &buf_num = cp_handler->get_buf_num();

    if (taskBitMask & (0x1 << CUMAC_TASK_UE_SELECTION))
    {
        CHECK_VALUE_EQUAL_ERR(buf_num.cellId, data_num.cellId);
        CHECK_VALUE_EQUAL_ERR(buf_num.prgMsk, data_num.prgMsk);
        // Allow data_num <= buf_num when nActiveUe < nMaxActUePerCell
        CHECK_VALUE_MAX_ERR(data_num.wbSinr, buf_num.wbSinr);
        CHECK_VALUE_MAX_ERR(data_num.avgRatesActUe, buf_num.avgRatesActUe);
        CHECK_VALUE_MAX_ERR(data_num.cellAssocActUe, buf_num.cellAssocActUe);
        NVLOGI_FMT(TAG, "SFN {}.{} DATA_NUM UE_SEL: cellId={} prgMsk={} wbSinr={} avgRatesActUe={} cellAssocActUe={} setSchdUePerCellTTI={}",
            ss.u16.sfn, ss.u16.slot, data_num.cellId, data_num.prgMsk, data_num.wbSinr, data_num.avgRatesActUe, data_num.cellAssocActUe, data_num.setSchdUePerCellTTI);
    }

    if (taskBitMask & (0x1 << CUMAC_TASK_PRB_ALLOCATION))
    {
        // Allow data_num <= buf_num when nActiveUe < nMaxActUePerCell
        CHECK_VALUE_MAX_ERR(data_num.estH_fr, buf_num.estH_fr);
        CHECK_VALUE_MAX_ERR(data_num.postEqSinr, buf_num.postEqSinr);
        CHECK_VALUE_MAX_ERR(data_num.sinVal, buf_num.sinVal);
        CHECK_VALUE_MAX_ERR(data_num.detMat, buf_num.detMat);
        CHECK_VALUE_MAX_ERR(data_num.prdMat, buf_num.prdMat);
        NVLOGI_FMT(TAG, "SFN {}.{} DATA_NUM PRB_ALLOC: estH_fr={} postEqSinr={} sinVal={} detMat={} prdMat={} allocSol={}",
            ss.u16.sfn, ss.u16.slot, data_num.estH_fr, data_num.postEqSinr, data_num.sinVal, data_num.detMat, data_num.prdMat, data_num.allocSol);
    }

    if (taskBitMask & (0x1 << CUMAC_TASK_PFM_SORT))
    {
        CHECK_VALUE_EQUAL_ERR(buf_num.pfmCellInfo, data_num.pfmCellInfo);
        NVLOGI_FMT(TAG, "SFN {}.{} DATA_NUM PFM_SORT: pfmCellInfo={}",
            ss.u16.sfn, ss.u16.slot, data_num.pfmCellInfo);
    }

    if (run_in_cpu == 0)
    {
        CHECK_CUDA_ERR(cudaEventRecord(ev_copy1, strm));
        if (group_buf_enabled)
        {
            if (debug_option & DBG_OPT_PRINT_CUMAC_BUF)
            {
                // Compare difference between data_num and handler's buf_num:
                for (int i = 0; i < sizeof(cumac_buf_num_t) / sizeof(uint32_t); i++)
                {
                    uint32_t *pdata_num = reinterpret_cast<uint32_t*>(&data_num) + i;
                    uint32_t *pbuf_num = reinterpret_cast<uint32_t*>(&buf_num) + i;
                    if (*pdata_num != *pbuf_num)
                    {
                        NVLOGI_FMT(TAG, "SFN {}.{} DATA_NUM mismatch: i={} data_num={} buf_num={}", ss.u16.sfn, ss.u16.slot, i, *pdata_num, *pbuf_num);
                    }
                }
            }

            // gpu_copy_cell_bufs uses task_info->data_num for per-cell strides and copy lengths. It must match this TTI's IPC payload sizes (task->data_num), not buf_num from init.
            CHECK_CUDA_ERR(cudaMemcpyAsync(&gpu_task_info->data_num, &data_num, sizeof(cumac_buf_num_t), cudaMemcpyHostToDevice, strm));
            CHECK_CUDA_ERR(cudaMemcpyAsync(gpu_cell_descs, cpu_cell_descs, sizeof(cell_desc_t) * cell_num, cudaMemcpyHostToDevice, strm));
            cumac_copy_cell_to_group(this, cp_handler->configs.cuda_block_num);
        }
        CHECK_CUDA_ERR(cudaEventRecord(ev_copy2, strm));
    }

    if (debug_option & DBG_OPT_PRINT_CUMAC_BUF) // Dump buffers
    {
        CHECK_CUDA_ERR(cudaStreamSynchronize(strm));
        if (taskBitMask & (0x1 << CUMAC_TASK_UE_SELECTION))
        {
            for (int i = 0; i < cell_num; i++)
            {
                std::string text = "prgMsk-" + std::to_string(i);
                print_array(text.c_str(), grpPrms.prgMsk[i], data_num.prgMsk);
            }
            print_array("cellId", grpPrms.cellId, data_num.cellId);
            print_array("wbSinr", grpPrms.wbSinr, data_num.wbSinr);
            print_array("cellAssocActUe", grpPrms.cellAssocActUe, data_num.cellAssocActUe);
            print_array("cellAssoc", grpPrms.cellAssoc, data_num.cellAssoc);
            print_array("avgRatesActUe", ueStatus.avgRatesActUe, data_num.avgRatesActUe);
        }

        if (taskBitMask & (0x1 << CUMAC_TASK_PFM_SORT))
        {
            print_array("pfmCellInfo", reinterpret_cast<uint8_t*>(pfmCellInfo), sizeof(cumac_pfm_cell_info_t));
        }
    }

    // Copy the contiguous estH_fr buffer
    if (taskBitMask & (0x1 << CUMAC_TASK_PRB_ALLOCATION))
    {
        CHECK_VALUE_MAX_ERR(data_num.estH_fr, buf_num.estH_fr);
        CHECK_VALUE_MAX_ERR(data_num.postEqSinr, buf_num.postEqSinr);
        CHECK_VALUE_MAX_ERR(data_num.sinVal, buf_num.sinVal);
        CHECK_VALUE_MAX_ERR(data_num.detMat, buf_num.detMat);
        CHECK_VALUE_MAX_ERR(data_num.prdMat, buf_num.prdMat);

        cuComplex *src_estH_fr = input_estH_fr;
        // if (tv != nullptr && debug_option & DBG_OPT_WAR_COPY_GROUP_TV)
        // {
        //     src_estH_fr = tv->estH_fr;
        // }

        if (run_in_cpu)
        {
            memcpy(grpPrms.estH_fr, src_estH_fr, data_num.estH_fr * sizeof(cuComplex));
        }
        else if (group_buf_enabled == 0)
        {
            CHECK_CUDA_ERR(cudaMemcpyAsync(grpPrms.estH_fr, src_estH_fr, data_num.estH_fr * sizeof(cuComplex), cudaMemcpyHostToDevice, strm));
        }

        if (debug_option & DBG_OPT_PRINT_CUMAC_BUF) // Dump buffers
        {
            CHECK_CUDA_ERR(cudaStreamSynchronize(strm));
            print_array("postEqSinr", grpPrms.postEqSinr, data_num.postEqSinr);
            print_array("sinVal", grpPrms.sinVal, data_num.sinVal);
            print_array("estH_fr", reinterpret_cast<float *>(grpPrms.estH_fr), data_num.estH_fr * 2);
            print_array("detMat", reinterpret_cast<float *>(grpPrms.detMat), data_num.detMat * 2);
            print_array("prdMat", reinterpret_cast<float *>(grpPrms.prdMat), data_num.prdMat * 2);
        }
    }

    if (debug_option & DBG_OPT_COMPARE_GROUP_TV_BUF)
    {
        validate_buffer_setup();
    }

    ts_setup = std::chrono::system_clock::now().time_since_epoch().count();
    if (run_in_cpu == 0)
    {
        CHECK_CUDA_ERR(cudaEventRecord(ev_copy2, strm));
    }

    // Start call cuMAC setup()
    if (taskBitMask & (0x1 << CUMAC_TASK_UE_SELECTION))
    {
        if (run_in_cpu)
        {
            NVLOGI_FMT(TAG, "SFN {}.{} {} CPU multiCellUeSelection", ss.u16.sfn, ss.u16.slot, __func__);
            mcUeSelCpu->setup(&ueStatus, &schdSol, &grpPrms);
        }
        else
        {
            NVLOGI_FMT(TAG, "SFN {}.{} {} GPU multiCellUeSelection", ss.u16.sfn, ss.u16.slot, __func__);
            mcUeSelGpu->setup(&ueStatus, &schdSol, &grpPrms, strm);
        }
    }

    if (taskBitMask & (0x1 << CUMAC_TASK_PRB_ALLOCATION))
    {
        if (run_in_cpu)
        {
            NVLOGI_FMT(TAG, "SFN {}.{} {} CPU multiCellScheduler totNumCell={} columnMajor={}",
                       ss.u16.sfn, ss.u16.slot, __func__, simParam.totNumCell, columnMajor);
            mcSchCpu->setup(&ueStatus, &schdSol, &grpPrms, &simParam, columnMajor);
        }
        else
        {
            NVLOGI_FMT(TAG, "SFN {}.{} {} GPU multiCellScheduler totNumCell={} columnMajor={} halfPrecision={} lightWeight={}",
                       ss.u16.sfn, ss.u16.slot, __func__, simParam.totNumCell, columnMajor, halfPrecision, lightWeight);
            mcSchGpu->setup(&ueStatus, &schdSol, &grpPrms, &simParam, columnMajor, halfPrecision, lightWeight, 2.0, strm);
        }
    }

    if (taskBitMask & (0x1 << CUMAC_TASK_LAYER_SELECTION))
    {
        if (run_in_cpu)
        {
            NVLOGI_FMT(TAG, "SFN {}.{} {} CPU mcsSelectionLUT", ss.u16.sfn, ss.u16.slot, __func__);
            mcLayerSelCpu->setup(&ueStatus, &schdSol, &grpPrms);
        }
        else
        {
            NVLOGI_FMT(TAG, "SFN {}.{} {} GPU mcsSelectionLUT", ss.u16.sfn, ss.u16.slot, __func__);
            mcLayerSelGpu->setup(&ueStatus, &schdSol, &grpPrms, RI, strm);
        }
    }

    if (taskBitMask & (0x1 << CUMAC_TASK_MCS_SELECTION))
    {
        if (run_in_cpu)
        {
            NVLOGI_FMT(TAG, "SFN {}.{} {} CPU multiCellLayerSel", ss.u16.sfn, ss.u16.slot, __func__);
            mcMcsSelCpu->setup(&ueStatus, &schdSol, &grpPrms);
        }
        else
        {
            NVLOGI_FMT(TAG, "SFN {}.{} {} GPU multiCellLayerSel", ss.u16.sfn, ss.u16.slot, __func__);
            mcMcsSelGpu->setup(&ueStatus, &schdSol, &grpPrms, strm);
        }
    }

    if (taskBitMask & (0x1 << CUMAC_TASK_PFM_SORT))
    {
        if (run_in_cpu)
        {
            NVLOGE_FMT(TAG, AERIAL_CUMAC_CP_EVENT, "SFN {}.{} {} CPU pfmSort not supported", ss.u16.sfn, ss.u16.slot, __func__);
        }
        else
        {
            NVLOGI_FMT(TAG, "SFN {}.{} {} GPU pfmSort", ss.u16.sfn, ss.u16.slot, __func__);
            // Setup pfmSortTask structure will be done in cumac_cp_handler::on_sch_tti_request
            pfmSortGpu->setup(&pfmSortTask);
        }
    }

    if (run_in_cpu == 0)
    {
        CHECK_CUDA_ERR(cudaEventRecord(ev_setup_end, strm));
    }
    return 0;
}

int cumac_task::run()
{
    if (slot_concurrent_enable == 0)
    {
        // Wait for the previous slot to finish
        sem_wait(&cp_handler->gpu_sem);
    }

    NVLOGI_FMT(TAG, "SFN {}.{} run in {}: W={} sigmaSqrd={} Pt_Rbg={} Pt_rbgAnt={} betaCoeff={} sinValThr={} corrThr={}", ss.u16.sfn, ss.u16.slot, run_in_cpu ? "CPU" : "GPU",
               grpPrms.W, grpPrms.sigmaSqrd, grpPrms.Pt_Rbg, grpPrms.Pt_rbgAnt, grpPrms.betaCoeff, grpPrms.sinValThr, grpPrms.corrThr);

    uint64_t ts_copy_start{}, ts_copy_end{}, ts_run_start{};

    ts_run = std::chrono::system_clock::now().time_since_epoch().count();

    if (run_in_cpu == 0)
    {
        CHECK_CUDA_ERR(cudaEventRecord(ev_run_start, strm));
    }

    // UE_SELECTION run
    if (taskBitMask & (0x1 << CUMAC_TASK_UE_SELECTION))
    {
        if (run_in_cpu)
        {
            NVLOGI_FMT(TAG, "SFN {}.{} {} CPU multiCellUeSelection", ss.u16.sfn, ss.u16.slot, __func__);
            mcUeSelCpu->run();
        }
        else
        {
            NVLOGI_FMT(TAG, "SFN {}.{} {} GPU multiCellUeSelection", ss.u16.sfn, ss.u16.slot, __func__);
            mcUeSelGpu->run(strm);
        }

        ts_copy_start = std::chrono::system_clock::now().time_since_epoch().count();

        // Copy setSchdUePerCellTTI to CPU output buffer
        if (run_in_cpu)
        {
            memcpy(output_setSchdUePerCellTTI, schdSol.setSchdUePerCellTTI, sizeof(*schdSol.setSchdUePerCellTTI) * data_num.setSchdUePerCellTTI);
        }
        else if (group_buf_enabled == 0)
        {
            CHECK_CUDA_ERR(cudaMemcpyAsync(output_setSchdUePerCellTTI, schdSol.setSchdUePerCellTTI, sizeof(*schdSol.setSchdUePerCellTTI) * data_num.setSchdUePerCellTTI, cudaMemcpyDeviceToHost, strm));
            CHECK_CUDA_ERR(cudaStreamSynchronize(strm));
        }

        ts_copy_end = std::chrono::system_clock::now().time_since_epoch().count();
        output_copy_time += ts_copy_end - ts_copy_start;
    }

    if (run_in_cpu == 0)
    {
        CHECK_CUDA_ERR(cudaEventRecord(ev_run1, strm));
    }

    // PRB_ALLOCATION run
    if (taskBitMask & (0x1 << CUMAC_TASK_PRB_ALLOCATION))
    {

        // Copy avgRatesActUe to avgRates CPU buffer
        for (int buf_id = 0; buf_id < data_num.setSchdUePerCellTTI; buf_id++)
        {
            uint32_t ue_id = output_setSchdUePerCellTTI[buf_id];
            input_avgRates[buf_id] = input_avgRatesActUe[ue_id];
        }

        // Copy avgRates to cuMAC buffer
        if (run_in_cpu)
        {
            memcpy(ueStatus.avgRates, input_avgRates, sizeof(float) * data_num.avgRates);
        }
        else if (group_buf_enabled == 0)
        {
            CHECK_CUDA_ERR(cudaMemcpyAsync(ueStatus.avgRates, input_avgRates, sizeof(float) * data_num.avgRates, cudaMemcpyHostToDevice, strm));
            CHECK_CUDA_ERR(cudaStreamSynchronize(strm));
        }
        else
        {
            cumac_copy_avgRates(this, cp_handler->configs.cuda_block_num);
        }

        if (debug_option & DBG_OPT_PRINT_CUMAC_BUF)
        {
            print_array("setSchdUePerCellTTI", schdSol.setSchdUePerCellTTI, data_num.setSchdUePerCellTTI);

            print_array("postEqSinr", grpPrms.postEqSinr, data_num.postEqSinr);
            print_array("sinVal", grpPrms.sinVal, data_num.sinVal);

            print_array("detMat", reinterpret_cast<float *>(grpPrms.detMat), data_num.detMat * 2);
            print_array("prdMat", reinterpret_cast<float *>(grpPrms.prdMat), data_num.prdMat * 2);
            print_array("estH_fr", reinterpret_cast<float *>(grpPrms.estH_fr), data_num.estH_fr * 2);

            print_array("avgRates", ueStatus.avgRates, data_num.avgRates);
            print_array("cellAssoc", grpPrms.cellAssoc, data_num.cellAssoc);
        }

        ts_run_start = std::chrono::system_clock::now().time_since_epoch().count();
        run_copy_time += ts_run_start - ts_copy_end;

        if (run_in_cpu)
        {
            NVLOGI_FMT(TAG, "SFN {}.{} {} CPU multiCellScheduler", ss.u16.sfn, ss.u16.slot, __func__);
            mcSchCpu->run();
        }
        else
        {
            NVLOGI_FMT(TAG, "SFN {}.{} {} GPU multiCellScheduler", ss.u16.sfn, ss.u16.slot, __func__);
            mcSchGpu->run(strm);
        }

        ts_copy_start = std::chrono::system_clock::now().time_since_epoch().count();

        // Copy setSchdUePerCellTTI to CPU output buffer
        if (run_in_cpu)
        {
            memcpy(output_allocSol, schdSol.allocSol, sizeof(*schdSol.allocSol) * data_num.allocSol);
        }
        else if (group_buf_enabled == 0)
        {
            CHECK_CUDA_ERR(cudaMemcpyAsync(output_allocSol, schdSol.allocSol, sizeof(*schdSol.allocSol) * data_num.allocSol, cudaMemcpyDeviceToHost, strm));
            CHECK_CUDA_ERR(cudaStreamSynchronize(strm));
        }

        if (debug_option & DBG_OPT_PRINT_CUMAC_BUF) // Dump output buffers
        {
            print_array("output_allocSol", output_allocSol, data_num.allocSol, 1);
        }
        ts_copy_end = std::chrono::system_clock::now().time_since_epoch().count();
        output_copy_time += ts_copy_end - ts_copy_start;
    }

    if (run_in_cpu == 0)
    {
        CHECK_CUDA_ERR(cudaEventRecord(ev_run2, strm));
    }

    // LAYER_SELECTION run
    if (taskBitMask & (0x1 << CUMAC_TASK_LAYER_SELECTION))
    {
        if (run_in_cpu)
        {
            NVLOGI_FMT(TAG, "SFN {}.{} {} CPU multiCellLayerSel", ss.u16.sfn, ss.u16.slot, __func__);
            mcLayerSelCpu->run();
        }
        else
        {
            NVLOGI_FMT(TAG, "SFN {}.{} {} GPU multiCellLayerSel", ss.u16.sfn, ss.u16.slot, __func__);
            mcLayerSelGpu->run(strm);
        }

        ts_copy_start = std::chrono::system_clock::now().time_since_epoch().count();

        // Copy setSchdUePerCellTTI to CPU output buffer
        if (run_in_cpu)
        {
            memcpy(output_layerSelSol, schdSol.layerSelSol, sizeof(*schdSol.layerSelSol) * data_num.layerSelSol);
        }
        else if (group_buf_enabled == 0)
        {
            CHECK_CUDA_ERR(cudaMemcpyAsync(output_layerSelSol, schdSol.layerSelSol, sizeof(*schdSol.layerSelSol) * data_num.layerSelSol, cudaMemcpyDeviceToHost, strm));
            CHECK_CUDA_ERR(cudaStreamSynchronize(strm));
        }

        if (debug_option & DBG_OPT_PRINT_CUMAC_BUF) // Dump output buffers
        {
            print_array("output_layerSelSol", output_layerSelSol, data_num.layerSelSol, 1);
        }

        ts_copy_end = std::chrono::system_clock::now().time_since_epoch().count();
        output_copy_time += ts_copy_end - ts_copy_start;
    }

    if (run_in_cpu == 0)
    {
        CHECK_CUDA_ERR(cudaEventRecord(ev_run3, strm));
    }

    // MCS_SELECTION run
    if (taskBitMask & (0x1 << CUMAC_TASK_MCS_SELECTION))
    {
        // Copy avgRatesActUe to avgRates CPU buffer
        for (int buf_id = 0; buf_id < data_num.setSchdUePerCellTTI; buf_id++)
        {
            uint32_t ue_id = output_setSchdUePerCellTTI[buf_id];
            input_tbErrLast[buf_id] = input_tbErrLastActUe[ue_id];
        }

        // Copy avgRates to cuMAC buffer
        if (run_in_cpu)
        {
            cumac_cp_tv_t *ptr = get_cumac_tv_ptr();
            if (ptr != nullptr && ptr->parsed)
            { // TODO: DIFF
              // memcpy(grpPrms.postEqSinr, ptr->postEqSinr, data_num.postEqSinr * sizeof(*grpPrms.postEqSinr));
            }
            memcpy(ueStatus.tbErrLast, input_tbErrLast, sizeof(*ueStatus.tbErrLast) * data_num.tbErrLast);
        }
        else if (group_buf_enabled == 0)
        {
            CHECK_CUDA_ERR(cudaMemcpyAsync(ueStatus.tbErrLast, input_tbErrLast, sizeof(*ueStatus.tbErrLast) * data_num.tbErrLast, cudaMemcpyHostToDevice, strm));
            CHECK_CUDA_ERR(cudaStreamSynchronize(strm));
        }
        else
        {
            cumac_copy_tbErrLast(this, cp_handler->configs.cuda_block_num);
        }

        ts_run_start = std::chrono::system_clock::now().time_since_epoch().count();
        run_copy_time += ts_run_start - ts_copy_end;

        if (run_in_cpu)
        {
            NVLOGI_FMT(TAG, "SFN {}.{} {} CPU mcsSelectionLUT", ss.u16.sfn, ss.u16.slot, __func__);
            mcMcsSelCpu->run();
        }
        else
        {
            NVLOGI_FMT(TAG, "SFN {}.{} {} GPU mcsSelectionLUT", ss.u16.sfn, ss.u16.slot, __func__);
            mcMcsSelGpu->run(strm);
        }

        ts_copy_start = std::chrono::system_clock::now().time_since_epoch().count();

        // Copy mcsSelSol to CPU output buffer
        if (run_in_cpu)
        {
            memcpy(output_mcsSelSol, schdSol.mcsSelSol, sizeof(*schdSol.mcsSelSol) * data_num.mcsSelSol);
        }
        else if (group_buf_enabled == 0)
        {
            CHECK_CUDA_ERR(cudaMemcpyAsync(output_mcsSelSol, schdSol.mcsSelSol, sizeof(*schdSol.mcsSelSol) * data_num.mcsSelSol, cudaMemcpyDeviceToHost, strm));
            CHECK_CUDA_ERR(cudaStreamSynchronize(strm));
        }

        if (debug_option & DBG_OPT_PRINT_CUMAC_BUF) // Dump output buffers
        {
            print_array("output_mcsSelSol", output_mcsSelSol, data_num.mcsSelSol, 1);
        }
        ts_copy_end = std::chrono::system_clock::now().time_since_epoch().count();
        output_copy_time += ts_copy_end - ts_copy_start;
    }

    // PFM_SORT run
    if (taskBitMask & (0x1 << CUMAC_TASK_PFM_SORT))
    {
        if (run_in_cpu)
        {
            NVLOGE_FMT(TAG, AERIAL_CUMAC_CP_EVENT, "SFN {}.{} {} CPU pfmSort not supported", ss.u16.sfn, ss.u16.slot, __func__);
        }
        else
        {
            // NVLOGI_FMT(TAG, "SFN {}.{} {} GPU pfmSort num_cell={}", ss.u16.sfn, ss.u16.slot, __func__, pfmSortTask.num_cell);
            pfmSortGpu->run(reinterpret_cast<uint8_t*>(output_pfmSortSol));
        }
    }

    if (run_in_cpu == 0)
    {
        CHECK_CUDA_ERR(cudaEventRecord(ev_run4, strm));
        if (slot_concurrent_enable == 0)
        {
            CHECK_CUDA_ERR(cudaStreamSynchronize(strm));
            sem_post(&cp_handler->gpu_sem);
        }
    }

    return 0;
}

int cumac_task::callback()
{
    NVLOGI_FMT(TAG, "SFN {}.{} {} in {} nCell={} numUeSchdPerCellTTI={}", ss.u16.sfn, ss.u16.slot, __func__, run_in_cpu ? "CPU" : "GPU", grpPrms.nCell, grpPrms.numUeSchdPerCellTTI);

    ts_callback = std::chrono::system_clock::now().time_since_epoch().count();

    if (run_in_cpu == 0)
    {
        CHECK_CUDA_ERR(cudaEventRecord(ev_callback_start, strm));

        if (taskBitMask & (0x1 << CUMAC_TASK_UE_SELECTION))
        {
            CHECK_CUDA_ERR(cudaMemcpyAsync(output_setSchdUePerCellTTI, schdSol.setSchdUePerCellTTI, sizeof(*schdSol.setSchdUePerCellTTI) * data_num.setSchdUePerCellTTI, cudaMemcpyDeviceToHost, strm));
        }
        if (taskBitMask & (0x1 << CUMAC_TASK_PRB_ALLOCATION))
        {
            CHECK_CUDA_ERR(cudaMemcpyAsync(output_allocSol, schdSol.allocSol, sizeof(*schdSol.allocSol) * data_num.allocSol, cudaMemcpyDeviceToHost, strm));
        }
        if (taskBitMask & (0x1 << CUMAC_TASK_LAYER_SELECTION))
        {
            CHECK_CUDA_ERR(cudaMemcpyAsync(output_layerSelSol, schdSol.layerSelSol, sizeof(*schdSol.layerSelSol) * data_num.layerSelSol, cudaMemcpyDeviceToHost, strm));
        }
        if (taskBitMask & (0x1 << CUMAC_TASK_MCS_SELECTION))
        {
            CHECK_CUDA_ERR(cudaMemcpyAsync(output_mcsSelSol, schdSol.mcsSelSol, sizeof(*schdSol.mcsSelSol) * data_num.mcsSelSol, cudaMemcpyDeviceToHost, strm));
        }
        CHECK_CUDA_ERR(cudaEventRecord(ev_callback_end, strm));
        CHECK_CUDA_ERR(cudaStreamSynchronize(strm));
    }

    if (grpPrms.allocType == 0 && taskBitMask & (0x1 << CUMAC_TASK_PRB_ALLOCATION))
    {
        // Copy allocSol to output buffer for allocType = 0
        int16_t tmp_copy[grpPrms.nPrbGrp * grpPrms.nCell];
        for (int cellId = 0; cellId < grpPrms.nCell; cellId++) {
            for (int prgIdx = 0; prgIdx < grpPrms.nPrbGrp; prgIdx++) {
                int ueIdx = output_allocSol[prgIdx * grpPrms.nCell + cellId] - cellId * grpPrms.numUeSchdPerCellTTI;
                tmp_copy[prgIdx + cellId * grpPrms.nPrbGrp] = output_setSchdUePerCellTTI[ueIdx + cellId * grpPrms.numUeSchdPerCellTTI] - grpPrms.nActiveUe / grpPrms.nCell * cellId;
            }
        }
        memcpy(output_allocSol, tmp_copy, sizeof(int16_t) * grpPrms.nPrbGrp * grpPrms.nCell);
    }

    ts_resp = std::chrono::system_clock::now().time_since_epoch().count();

    callback_fun(this, callback_args);

    // Release nvipc buffers when task finished
    nv::phy_mac_transport_wrapper &wrapper = cp_handler->transport_wrapper();
    for (uint32_t cell_id = 0; cell_id < grpPrms.nCell; cell_id++)
    {
        wrapper.rx_release(tti_reqs[cell_id]);
        tti_reqs[cell_id].reset();
    }

    ts_end = std::chrono::system_clock::now().time_since_epoch().count();

    if (debug_option & DBG_OPT_COMPARE_GROUP_TV_BUF)
    {
        CHECK_CUDA_ERR(cudaStreamSynchronize(strm));
        validate_buffer_callback();
    }

    if (debug_option & DBG_OPT_PRINT_CUMAC_BUF) // Dump output buffers
    {
        CHECK_CUDA_ERR(cudaStreamSynchronize(strm));

        if (taskBitMask & (0x1 << CUMAC_TASK_UE_SELECTION))
        {
            print_array("OUT: setSchdUePerCellTTI", schdSol.setSchdUePerCellTTI, data_num.setSchdUePerCellTTI);
        }
        if (taskBitMask & (0x1 << CUMAC_TASK_PRB_ALLOCATION))
        {
            print_array("OUT: allocSol", schdSol.allocSol, data_num.allocSol);
        }
        if (taskBitMask & (0x1 << CUMAC_TASK_LAYER_SELECTION))
        {
            print_array("OUT: layerSelSol", schdSol.layerSelSol, data_num.layerSelSol);
        }
        if (taskBitMask & (0x1 << CUMAC_TASK_MCS_SELECTION))
        {
            print_array("OUT: mcsSelSol", schdSol.mcsSelSol, data_num.mcsSelSol);
        }
        if (taskBitMask & (0x1 << CUMAC_TASK_PFM_SORT))
        {
            print_array("OUT: pfmSortSol", reinterpret_cast<uint8_t*>(output_pfmSortSol), sizeof(cumac_pfm_output_cell_info_t) * cell_num, 1);
        }
    }

    if (run_in_cpu == 0)
    {
        // Below are for timing and performance debug
        CHECK_CUDA_ERR(cudaEventElapsedTime(&tm_copy1, ev_start, ev_copy1));
        CHECK_CUDA_ERR(cudaEventElapsedTime(&tm_copy2, ev_copy1, ev_copy2));
        CHECK_CUDA_ERR(cudaEventElapsedTime(&tm_setup, ev_copy2, ev_setup_end));

        CHECK_CUDA_ERR(cudaEventElapsedTime(&tm_run1, ev_run_start, ev_run1));
        CHECK_CUDA_ERR(cudaEventElapsedTime(&tm_run2, ev_run1, ev_run2));
        CHECK_CUDA_ERR(cudaEventElapsedTime(&tm_run3, ev_run2, ev_run3));
        CHECK_CUDA_ERR(cudaEventElapsedTime(&tm_run4, ev_run3, ev_run4));
        CHECK_CUDA_ERR(cudaEventElapsedTime(&tm_run, ev_run_start, ev_run4));

        CHECK_CUDA_ERR(cudaEventElapsedTime(&tm_callback, ev_callback_start, ev_callback_end));

        CHECK_CUDA_ERR(cudaEventElapsedTime(&tm_total, ev_start, ev_callback_end));

        tm_debug = 0;
        // CHECK_CUDA_ERR(cudaEventRecord(ev_debug, strm));
        // CHECK_CUDA_ERR(cudaStreamSynchronize(strm));
        // CHECK_CUDA_ERR(cudaEventElapsedTime(&tm_debug, ev_callback_end, ev_debug));

        float slot_total = static_cast<float>(ts_end - ts_start) / 1E6;
        NVLOGI_FMT(TAG, "SFN {}.{} in {} 0x{:X} TIMING: slot_total={:.3f} gpu_total={:.3f} copy1={:.3f} copy2={:.3f} setup={:.3f} run={:.3f} callback={:.3f} debug={:.3f} | run {:.3f} {:.3f} {:.3f} {:.3f}",
                   ss.u16.sfn, ss.u16.slot, run_in_cpu ? "CPU" : "GPU", taskBitMask,
                   slot_total, tm_total, tm_copy1, tm_copy2, tm_setup, tm_run, tm_callback, tm_debug, tm_run1, tm_run2, tm_run3, tm_run4);
    }

    ts_debug = std::chrono::system_clock::now().time_since_epoch().count();

    NVLOGI_FMT(TAG, "SFN {}.{} in {} 0x{:X} CPU_DURATION: msg_send={} msg_recv={} task_enq={} task_deq={} wait={} copy={} setup={} run={} callback={} resp={} total={} debug={}", ss.u16.sfn, ss.u16.slot, run_in_cpu ? "CPU" : "GPU",
               taskBitMask, ts_last_send - ts_start, ts_last_recv - ts_last_send, ts_enqueue - ts_last_recv, ts_dequeue - ts_enqueue, ts_copy - ts_dequeue, ts_setup - ts_copy, ts_run - ts_setup, ts_callback - ts_run, ts_resp - ts_callback, ts_end - ts_resp, ts_resp - ts_copy, ts_debug - ts_end);

    return taskBitMask;
}

int cumac_task::validate_buffer_setup()
{
    if (tv == nullptr)
    {
        return -1;
    }

    CHECK_CUDA_ERR(cudaStreamSynchronize(strm));

    if (taskBitMask & (0x1 << CUMAC_TASK_UE_SELECTION))
    {
        compare_array("SETUP_cellId", tv->cellId, grpPrms.cellId, data_num.cellId);
        compare_array("SETUP_wbSinr", tv->wbSinr, grpPrms.wbSinr, data_num.wbSinr);
        if (group_buf_enabled) {
            // compare_array("SETUP_prgMsk", tv->prgMsk, *grpPrms.prgMsk, data_num.prgMsk * cell_num);
        }
        compare_array("SETUP_avgRatesActUe", tv->avgRatesActUe, ueStatus.avgRatesActUe, data_num.avgRatesActUe);
        compare_array("SETUP_cellAssoc", tv->cellAssoc, grpPrms.cellAssoc, data_num.cellAssoc);
        compare_array("SETUP_cellAssocActUe", tv->cellAssocActUe, grpPrms.cellAssocActUe, data_num.cellAssocActUe);
    }

    if (taskBitMask & (0x1 << CUMAC_TASK_PRB_ALLOCATION))
    {
        compare_array("SETUP_estH_fr", reinterpret_cast<float *>(tv->estH_fr), reinterpret_cast<float *>(grpPrms.estH_fr), data_num.estH_fr * 2);

        // compare_array("SETUP_avgRates", tv->avgRates, ueStatus.avgRates, data_num.avgRates);
        compare_array("SETUP_sinVal", tv->sinVal, grpPrms.sinVal, data_num.sinVal);

        compare_array("SETUP_prdMat", reinterpret_cast<float *>(tv->prdMat), reinterpret_cast<float *>(grpPrms.prdMat), data_num.prdMat * 2);
        compare_array("SETUP_detMat", reinterpret_cast<float *>(tv->detMat), reinterpret_cast<float *>(grpPrms.detMat), data_num.detMat * 2);

        compare_array("SETUP_postEqSinr", tv->postEqSinr, grpPrms.postEqSinr, data_num.postEqSinr);
    }

    if (taskBitMask & (0x1 << CUMAC_TASK_PFM_SORT))
    {
        compare_array("SETUP_pfmCellInfo", reinterpret_cast<uint8_t*>(tv->pfmCellInfo.data()), reinterpret_cast<uint8_t*>(pfmCellInfo), cell_num * sizeof(cumac_pfm_cell_info_t));
    }
    return 0;
}

int cumac_task::validate_buffer_callback()
{
    if (tv == nullptr)
    {
        return -1;
    }

    CHECK_CUDA_ERR(cudaStreamSynchronize(strm));

    if (taskBitMask & (0x1 << CUMAC_TASK_UE_SELECTION))
    {
        compare_array("cellId", tv->cellId, grpPrms.cellId, data_num.cellId);
        compare_array("cellAssoc", tv->cellAssoc, grpPrms.cellAssoc, data_num.cellAssoc);
        compare_array("cellAssocActUe", tv->cellAssocActUe, grpPrms.cellAssocActUe, data_num.cellAssocActUe);
        compare_array("avgRatesActUe", tv->avgRatesActUe, ueStatus.avgRatesActUe, data_num.avgRatesActUe);
        compare_array("wbSinr", tv->wbSinr, grpPrms.wbSinr, data_num.wbSinr);
        compare_array("setSchdUePerCellTTI", tv->setSchdUePerCellTTI, schdSol.setSchdUePerCellTTI, data_num.setSchdUePerCellTTI);
    }

    if (taskBitMask & (0x1 << CUMAC_TASK_PRB_ALLOCATION))
    {
        compare_array("avgRates", tv->avgRates, ueStatus.avgRates, data_num.avgRates);
        compare_array("sinVal", tv->sinVal, grpPrms.sinVal, data_num.sinVal);

        compare_array("prdMat", reinterpret_cast<float *>(tv->prdMat), reinterpret_cast<float *>(grpPrms.prdMat), data_num.prdMat * 2);
        compare_array("detMat", reinterpret_cast<float *>(tv->detMat), reinterpret_cast<float *>(grpPrms.detMat), data_num.detMat * 2);

        compare_array("END_estH_fr", reinterpret_cast<float *>(tv->estH_fr), reinterpret_cast<float *>(grpPrms.estH_fr), data_num.estH_fr * 2);
        compare_array("END_postEqSinr", tv->postEqSinr, grpPrms.postEqSinr, data_num.postEqSinr);

        compare_array("allocSol", tv->allocSol, schdSol.allocSol, data_num.allocSol);
    }

    if (taskBitMask & (0x1 << CUMAC_TASK_LAYER_SELECTION))
    {
        compare_array("layerSelSol", tv->layerSelSol, schdSol.layerSelSol, data_num.layerSelSol);
    }

    if (taskBitMask & (0x1 << CUMAC_TASK_MCS_SELECTION))
    {
        compare_array("tbErrLast", tv->tbErrLast, ueStatus.tbErrLast, data_num.tbErrLast);
        compare_array("tbErrLastActUe", tv->tbErrLastActUe, ueStatus.tbErrLastActUe, data_num.tbErrLastActUe);
        compare_array("mcsSelSol", tv->mcsSelSol, schdSol.mcsSelSol, data_num.mcsSelSol);
    }

    if (taskBitMask & (0x1 << CUMAC_TASK_PFM_SORT))
    {
        compare_array("pfmSortSol", reinterpret_cast<uint8_t*>(tv->pfmSortSol.data()), reinterpret_cast<uint8_t*>(output_pfmSortSol), cell_num * sizeof(cumac_pfm_output_cell_info_t), 1);
    }
    return 0;
}
