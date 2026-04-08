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

#include <new>

#include "nvlog.hpp"
#include "nv_utils.h"

#include "nv_phy_utils.hpp"
#include "cumac_app.hpp"
#include "cumac.h"
#include "api.h"
#include "cumac_msg.h"
#include "cumac_cp_handler.hpp"
#include "cumac_cp_tv.hpp"

#include "nv_phy_mac_transport.hpp"
#include "nv_phy_epoll_context.hpp"


#include "nvlog.hpp"

using namespace std;
using namespace nv;
using namespace cumac;
using namespace std::chrono;

#define TAG (NVLOG_TAG_BASE_CUMAC_CP + 4) // "CUMCP.HANDLER"

#define CHECK_PTR_NULL_FATAL(ptr)                                                                                  \
    do                                                                                                             \
    {                                                                                                              \
        if ((ptr) == nullptr)                                                                                      \
        {                                                                                                          \
            NVLOGF_FMT(TAG, AERIAL_CUMAC_CP_EVENT, "{} line {}: pointer {} is nullptr", __func__, __LINE__, #ptr); \
        }                                                                                                          \
    } while (0);

#define CHECK_VALUE_EQUAL_ERR(v1, v2)                                                                                                   \
    do                                                                                                                                  \
    {                                                                                                                                   \
        if ((v1) != (v2))                                                                                                               \
        {                                                                                                                               \
            NVLOGE_FMT(TAG, AERIAL_CUMAC_CP_EVENT, "{} line {}: values not equal: {}={}, {}={}", __func__, __LINE__, #v1, v1, #v2, v2); \
        }                                                                                                                               \
    } while (0);

#define CHECK_VALUE_MAX_ERR(val, max)                                                                                                                                                 \
    do                                                                                                                                                                                \
    {                                                                                                                                                                                 \
        if ((val) > (max))                                                                                                                                                            \
        {                                                                                                                                                                             \
            NVLOGE_FMT(TAG, AERIAL_CUMAC_CP_EVENT, "{} line {}: value > max: {}={} > {}={}", __func__, __LINE__, #val, static_cast<uint32_t>(val), #max, static_cast<uint32_t>(max)); \
        }                                                                                                                                                                             \
    } while (0);

template <typename T>
static T* cumac_init_msg_header(nv_ipc_msg_t* msg, int msg_id, int cell_id)
{
    size_t msg_size = sizeof(T);
    msg->msg_id = msg_id;
    msg->cell_id = cell_id;
    msg->msg_len = msg_size;
    msg->data_len = 0;

    cumac_msg_header_t *header = (cumac_msg_header_t*) msg->msg_buf;
    header->message_count = 1;
    header->handle_id = cell_id;
    header->type_id = msg_id;
    header->body_len = msg_size - sizeof(cumac_msg_header_t);
    return reinterpret_cast<T*>(msg->msg_buf);
}

void sched_slot_data::init_slot_data(cumac_cp_handler* _handler, uint32_t _cell_num) {
    cell_num = _cell_num;
    handler = _handler;
    slot_msgs.resize(cell_num);
    task = nullptr;
}

void sched_slot_data::reset_slot_data(sfn_slot_t ss) {
    NVLOGI_FMT(TAG, "SFN {}.{} init slot_data", ss.u16.sfn, ss.u16.slot);

    curr_cell_id = 0;

    if (task != nullptr) { // || ss_sched.u32 != SFN_SLOT_INVALID) {
        NVLOGE_FMT(TAG, AERIAL_CUMAC_CP_EVENT, "SFN {}.{} dropping previous incomplete slot SFN {}.{}",
                   ss.u16.sfn, ss.u16.slot, ss_sched.u16.sfn, ss_sched.u16.slot);

        // Free unhandled nvipc buffers
        nv::phy_mac_transport_wrapper &wrapper = handler->transport_wrapper();
        for (struct nv::phy_mac_msg_desc &msg_desc : task->tti_reqs)
        {
            if (msg_desc.msg_buf != nullptr)
            {
                NVLOGW_FMT(TAG, "SFN {}.{} dropping message SFN {}.{} cell_id={} msg_id=0x{:02X}", ss.u16.sfn, ss.u16.slot, ss_sched.u16.sfn, ss_sched.u16.slot, msg_desc.cell_id, msg_desc.msg_id);
                wrapper.rx_release(msg_desc);
                msg_desc.reset();
            }
        }

        // Free the cumac_task_t buffer
        handler->task_ring->free(task);
    }

    // TODO: for reorder
    for (uint32_t cell_id = 0; cell_id < cell_num; cell_id ++) {
        slot_msgs[cell_id].reset_cell_data();
    }

    if (handler == nullptr || handler->task_ring == nullptr) {
        NVLOGF_FMT(TAG, AERIAL_CUMAC_CP_EVENT, "Invalid pointer: handler or handler->task_ring is null");
        return;
    }

    ss_sched = ss;
    if ((task = handler->task_ring->alloc()) == nullptr) {
        NVLOGW_FMT(TAG, "SFN {}.{} task process can't catch up with enqueue, drop slot", ss.u16.sfn, ss.u16.slot);
        return;
    }

    task->reset_cumac_task(ss);
}

cumac_cp_handler::cumac_cp_handler(cumac_cp_configs& _configs, nv::phy_mac_transport_wrapper& wrapper) :
    configs(_configs),
    trans_wrapper(wrapper)
{
    configured_cell_num = 0;
    global_tick = 0;
    group_buf_size = configs.get_max_data_size() * configs.cell_num;

    task_ring = nullptr;
    ss_curr = {.u32 = SFN_SLOT_INVALID};

    cell_configs.resize(configs.cell_num);
    thrputs.resize(configs.cell_num);

    memset(&buf_num, 0, sizeof(cumac_buf_num_t));

    for (uint32_t buf_id = 0; buf_id < SCHED_SLOT_BUF_NUM; buf_id++)
    {
        sched_slots[buf_id].init_slot_data(this, configs.cell_num);
    }

    if (parse_group_tv(group_tv, configs.cell_num) < 0)
    {
        NVLOGW_FMT(TAG, "Failed to parse group TV");
    }

    sem_init(&gpu_sem, 0, 1);
}

cumac_cp_handler::~cumac_cp_handler() {
    if (blerTargetActUe != nullptr) {
        free(blerTargetActUe);
    }
}

int cumac_cp_handler::check_config_params()
{
    memset(&group_params, 0, sizeof(group_params));
    group_params.sigmaSqrd = 1.0; // Default; per-TTI value is taken from SCH_TTI.request in on_sch_tti_request

    for (int cell_id = 0; cell_id < configs.cell_num; cell_id++)
    {
        cumac_cell_configs_t &cell = cell_configs[cell_id];

        group_params.nUe += cell.nMaxSchUePerCell;
        group_params.nCell += 1;
        group_params.totNumCell += 1;
        group_params.nMaxSchdUePerRnd += cell.nMaxSchUePerCell;
        group_params.nActiveUe += cell.nMaxActUePerCell; // group_params.nActiveUe is used for max buffer size calculation, not the real active UEs count.

        if (cell_id == 0)
        {
            group_params.nPrbGrp = cell.nMaxPrg;

            group_params.nBsAnt = cell.nMaxBsAnt;
            group_params.nUeAnt = cell.nMaxUeAnt;

            // 12 * subcarrier spacing * number of PRBs per PRG
            group_params.W = 12 * cell.scSpacing * cell.nPrbPerPrg;

            group_params.maxNumUePerCell = cell.nMaxActUePerCell;
            group_params.betaCoeff = cell.betaCoeff;

            group_params.numUeSchdPerCellTTI = cell.nMaxSchUePerCell;
            group_params.precodingScheme = cell.precoderType;
            group_params.receiverScheme = cell.receiverType;
            group_params.allocType = cell.allocType;

            group_params.columnMajor = cell.colMajChanAccess;
            group_params.sinValThr = cell.sinValThr;

            group_params.mcsSelSinrCapThr = cell.mcsSelSinrCapThr;
            group_params.mcsSelLutType = cell.mcsSelLutType;
            group_params.harqEnabledInd = cell.harqEnabledInd;
            group_params.mcsSelCqi = cell.mcsSelCqi;
        }
        else
        {
            CHECK_VALUE_EQUAL_ERR(group_params.nPrbGrp, cell.nMaxPrg);

            CHECK_VALUE_EQUAL_ERR(group_params.nBsAnt, cell.nMaxBsAnt);
            CHECK_VALUE_EQUAL_ERR(group_params.nUeAnt, cell.nMaxUeAnt);

            CHECK_VALUE_EQUAL_ERR(group_params.W, 12 * cell.scSpacing * cell.nPrbPerPrg);

            CHECK_VALUE_EQUAL_ERR(group_params.maxNumUePerCell, cell.nMaxActUePerCell);
            CHECK_VALUE_EQUAL_ERR(group_params.betaCoeff, cell.betaCoeff);

            CHECK_VALUE_EQUAL_ERR(group_params.numUeSchdPerCellTTI, cell.nMaxSchUePerCell);
            CHECK_VALUE_EQUAL_ERR(group_params.precodingScheme, cell.precoderType);
            CHECK_VALUE_EQUAL_ERR(group_params.receiverScheme, cell.receiverType);
            CHECK_VALUE_EQUAL_ERR(group_params.allocType, cell.allocType);

            CHECK_VALUE_EQUAL_ERR(group_params.columnMajor, cell.colMajChanAccess);
            CHECK_VALUE_EQUAL_ERR(group_params.sinValThr, cell.sinValThr);

            CHECK_VALUE_EQUAL_ERR(group_params.mcsSelSinrCapThr, cell.mcsSelSinrCapThr);
            CHECK_VALUE_EQUAL_ERR(group_params.mcsSelLutType, cell.mcsSelLutType);
            CHECK_VALUE_EQUAL_ERR(group_params.harqEnabledInd, cell.harqEnabledInd);
            CHECK_VALUE_EQUAL_ERR(group_params.mcsSelCqi, cell.mcsSelCqi);

            CHECK_VALUE_EQUAL_ERR(cell_configs[0].blerTarget, cell.blerTarget);
        }
    }

    nanoseconds ts_start = duration_cast<nanoseconds>(system_clock::now().time_since_epoch());

    cumacSchedulerParam &p = group_params;
    NVLOGC_FMT(TAG, "GroupParams-1: nUe={} nCell={} totNumCell={} nPrbGrp={} nBsAnt={} nUeAnt={} W={} sigmaSqrd={} maxNumUePerCell={} nMaxSchdUePerRnd={} betaCoeff={} harqEnabledInd={}",
               p.nUe, p.nCell, p.totNumCell, p.nPrbGrp, p.nBsAnt, p.nUeAnt, p.W, p.sigmaSqrd, p.maxNumUePerCell, p.nMaxSchdUePerRnd, p.betaCoeff, p.harqEnabledInd);
    NVLOGC_FMT(TAG, "GroupParams-2: nActiveUe={} numUeSchdPerCellTTI={} precodingScheme={} receiverScheme={} allocType={} columnMajor={} allocType={} columnMajor={} sinValThr={} mcsSelLutType={} mcsSelSinrCapThr={} mcsSelCqi={}",
               p.nActiveUe, p.numUeSchdPerCellTTI, p.precodingScheme, p.receiverScheme, p.allocType, p.columnMajor, p.allocType, p.columnMajor, p.sinValThr, p.mcsSelLutType, p.mcsSelSinrCapThr, p.mcsSelCqi);

    uint32_t ring_len = task_ring->get_ring_len();
    
    for (int i = 0; i < ring_len; i++)
    {
        cumac_task *task = task_ring->get_buf_addr(i);
        if (task == nullptr)
        {
            NVLOGE_FMT(TAG, AERIAL_CUMAC_CP_EVENT, "Error cumac_task ring length: i={} length={}", i, ring_len);
            return -1;
        }

        // Initiate object at pre-allocated memory
        cumac_task *task_obj = new (task) cumac_task();
        if (task_obj != task)
        {
            NVLOGE_FMT(TAG, AERIAL_CUMAC_CP_EVENT, "{}: {} task_obj={} task={}", __func__, i, (void *)task_obj, (void *)task);
        }

        const int result = initiate_cumac_task(task);
        if (result != 0)
        {
            NVLOGE_FMT(TAG, AERIAL_CUMAC_CP_EVENT, "{}: initiate_cumac_task failed for task[{}]", __func__, i);
            return result;
        }
    }

    nanoseconds ts_end = duration_cast<nanoseconds>(system_clock::now().time_since_epoch());
    NVLOGC_FMT(TAG, "{}: buffer allocated: group_buf_size={} duration={}ns nCells={} nUe={} nPrbGrp={} nBsAnt={} nUeAnt={} betaCoeff={}", __func__,
            group_buf_size, ts_end.count() - ts_start.count(), group_params.nCell, group_params.nUe, group_params.nPrbGrp, group_params.nBsAnt, group_params.nUeAnt, group_params.betaCoeff);
    return 0;
}

int cumac_cp_handler::check_task_buf_size(cumac_task *task)
{
    if (task == nullptr)
    {
        return 0;
    }

    // UE_SELECTION buffers
    CHECK_VALUE_MAX_ERR(task->data_num.cellId, buf_num.cellId);
    CHECK_VALUE_MAX_ERR(task->data_num.prgMsk, buf_num.prgMsk);
    CHECK_VALUE_MAX_ERR(task->data_num.wbSinr, buf_num.wbSinr);

    CHECK_VALUE_MAX_ERR(task->data_num.avgRatesActUe, buf_num.avgRatesActUe);
    CHECK_VALUE_MAX_ERR(task->data_num.cellAssocActUe, buf_num.cellAssocActUe);

    CHECK_VALUE_MAX_ERR(task->data_num.setSchdUePerCellTTI, buf_num.setSchdUePerCellTTI);

    // PRB_ALLOCATION buffers
    CHECK_VALUE_MAX_ERR(task->data_num.cellAssoc, buf_num.cellAssoc);

    CHECK_VALUE_MAX_ERR(task->data_num.postEqSinr, buf_num.postEqSinr);
    CHECK_VALUE_MAX_ERR(task->data_num.sinVal, buf_num.sinVal);

    CHECK_VALUE_MAX_ERR(task->data_num.detMat, buf_num.detMat);
    CHECK_VALUE_MAX_ERR(task->data_num.prdMat, buf_num.prdMat);
    CHECK_VALUE_MAX_ERR(task->data_num.estH_fr, buf_num.estH_fr);

    return 0;
}

int cumac_task_callback_func(cumac_task* task, void* arg) {
    cumac_cp_handler* handler = static_cast<cumac_cp_handler*>(arg);
    handler->cumac_task_callback(task);
    return 0;
}

int cumac_cp_handler::cumac_task_callback(cumac_task *task)
{
    // Send regular scheduler responses for each cell
    for (int cell_id = 0; cell_id < group_params.nCell; cell_id ++) {
        send_sch_tti_response(task, cell_id);
    }
    return 0;
}

#define CUMAC_GPU_ALIGN_BYTES (16)

template <typename T>
int cumac_cp_handler::malloc_cumac_buf(cumac_task *task, T **ptr, uint32_t *num_save, uint32_t num, uint32_t force_host_mem)
{
    if (task->run_in_cpu || force_host_mem) // Allocate CPU memory
    {
        CHECK_CUDA_ERR(cudaMallocHost((void **)ptr, sizeof(T) * num));
    }
    else if (task->group_buf_enabled)
    {
        // Allocate a section from the contiguous group_buf
        *ptr = reinterpret_cast<T *>(task->group_buf + task->group_buf_offset);
        task->group_buf_offset += sizeof(T) * num;
        // Add padding bytes to align
        task->group_buf_offset = (task->group_buf_offset + CUMAC_GPU_ALIGN_BYTES - 1) & ~(CUMAC_GPU_ALIGN_BYTES - 1);
        if (task->group_buf_offset > group_buf_size)
        {
            NVLOGE_FMT(TAG, AERIAL_CUMAC_CP_EVENT, "{}: group_buf size={} exceeds allocated size={}", __func__, task->group_buf_offset, group_buf_size);
        }
    }
    else // Allocate GPU memory
    {
        CHECK_CUDA_ERR(cudaMalloc((void **)ptr, sizeof(T) * num));
    }

    if (num_save != nullptr) // Save the buffer number
    {
        *num_save = num;
    }
    return 0;
}

int cumac_cp_handler::initiate_cumac_task(cumac_task *task)
{
    if (task == nullptr)
    {
        NVLOGE_FMT(TAG, AERIAL_CUMAC_CP_EVENT, "{}: task pointer is NULL!", __func__);
        return -1;
    }

    task->callback_fun = cumac_task_callback_func;
    task->callback_args = this;
    task->cp_handler = this;
    task->cell_num = group_params.nCell;
    task->debug_option = configs.debug_option;
    task->group_buf_enabled = configs.group_buffer_enable;
    task->tti_reqs.resize(task->cell_num);
    task->group_buf_offset = 0;

    // Alloc host-pinned memory for debug log print
    CHECK_CUDA_ERR(cudaMallocHost((void **)&task->debug_buffer, CUMAC_TASK_DEBUG_BUF_MAX_SIZE));
    if (task->debug_buffer == nullptr)
    {
        NVLOGE_FMT(TAG, AERIAL_CUMAC_CP_EVENT, "{}: cudaMallocHost failed for debug_buffer", __func__);
        return -1;
    }

    // Allocate CPU and GPU memory for cell_desc_t: size = sizeof(cell_desc_t) * cell_num
    task->cpu_cell_descs = reinterpret_cast<cell_desc_t *>(malloc(sizeof(cell_desc_t) * task->cell_num));
    if (task->cpu_cell_descs == nullptr)
    {
        NVLOGE_FMT(TAG, AERIAL_CUMAC_CP_EVENT, "{}: malloc failed for cpu_cell_descs", __func__);
        return -1;
    }

    CHECK_CUDA_ERR(cudaMalloc(&task->gpu_cell_descs, sizeof(cell_desc_t) * task->cell_num));
    CHECK_CUDA_ERR(cudaMalloc(&task->gpu_task_info, sizeof(cumac_task_info_t)));

    // Allocate GPU memory for each cell: size = configs.get_max_data_size() for each cell
    CHECK_CUDA_ERR(cudaMalloc(&task->cells_buf, configs.get_max_data_size() * task->cell_num));
    CHECK_CUDA_ERR(cudaMalloc(&task->group_buf, group_buf_size));

    for (int cell_id = 0; cell_id < task->cell_num; cell_id++)
    {
        task->tti_reqs[cell_id].reset();

        cell_desc_t *cell_desc = task->cpu_cell_descs + cell_id;
        cell_desc->home = task->cells_buf + configs.get_max_data_size() * cell_id;
    }

    // Set running in GPU or CPU
    if (configs.run_in_cpu == 1) // Force running in CPU
    {
        task->run_in_cpu = configs.run_in_cpu;
    }
    else if (configs.run_in_cpu == 2) // Run in GPU for even task_id, run in CPU for odd task_id
    {
        task->run_in_cpu = task->task_id & 0x1;
    }
    else // Default: Run in GPU
    {
        task->run_in_cpu = 0;
    }

    // Create CUDA stream for each task if multi-stream is enabled
    if (task->run_in_cpu == 0 && configs.multi_stream_enable)
    {
        CHECK_CUDA_ERR(cudaStreamCreate(&task->strm));
    }

    task->slot_concurrent_enable = configs.slot_concurrent_enable;

    if (group_tv.parsed)
    {
        task->tv = &group_tv;
    }

    // Group parameters
    struct cumac::cumacCellGrpPrms *grpPrms = &task->grpPrms;
    struct cumac::cumacCellGrpUeStatus *cellGrpUeStatus = &task->ueStatus;
    struct cumac::cumacSchdSol *schdSol = &task->schdSol;

    grpPrms->numUeSchdPerCellTTI = group_params.numUeSchdPerCellTTI;
    grpPrms->nUe = group_params.nUe;
    grpPrms->nActiveUe = group_params.nActiveUe; // nActiveUe can change per slot request
    grpPrms->nCell = group_params.nCell;

    grpPrms->nPrbGrp = group_params.nPrbGrp;
    grpPrms->nBsAnt = group_params.nBsAnt;
    grpPrms->nUeAnt = group_params.nUeAnt;
    grpPrms->W = group_params.W;

    grpPrms->sigmaSqrd = group_params.sigmaSqrd;
    grpPrms->Pt_Rbg = 79.4328 / group_params.nPrbGrp;
    grpPrms->Pt_rbgAnt = 79.4328 / group_params.nPrbGrp / group_params.nBsAnt; // 5.38e-43f
    grpPrms->precodingScheme = group_params.precodingScheme; // precoder type: 0 - no precoding, 1 - SVD precoding

    grpPrms->receiverScheme = group_params.receiverScheme; // receiver type: only support 1 - MMSE-IRC
    grpPrms->allocType = group_params.allocType;           // PRB allocation type: 0 - non-consecutive type 0 allocate, 1 - consecutive type 1 allocate
    grpPrms->betaCoeff = group_params.betaCoeff;           // coefficient for balancing cell-center and cell-edge UEs' performance in multi-cell scheduling. Default value is 1.0
    grpPrms->sinValThr = group_params.sinValThr;           // singular value threshold for layer selection, value is in (0, 1). Default value is 0.1

    grpPrms->corrThr = cell_configs[0].corrThr;               // channel vector correlation value threshold for layer selection,  value is in (0, 1). Default value is 0.5
    grpPrms->prioWeightStep = cell_configs[0].prioWeightStep; // step size for UE priority weight increment per TTI if UE does not get scheduled. Default is 100

    grpPrms->harqEnabledInd = group_params.harqEnabledInd;
    grpPrms->mcsSelCqi = group_params.mcsSelCqi;
    grpPrms->mcsSelSinrCapThr = group_params.mcsSelSinrCapThr;
    grpPrms->mcsSelLutType = group_params.mcsSelLutType;

    // Alloc uint8_t** prgMsk pointer array in CPU memory
    CHECK_CUDA_ERR(cudaMallocHost((void **)&grpPrms->prgMsk, group_params.nCell * sizeof(uint8_t *)));
    for (int cIdx = 0; cIdx < group_params.nCell; cIdx++)
    {
        malloc_cumac_buf(task, &grpPrms->prgMsk[cIdx], &buf_num.prgMsk, group_params.nPrbGrp);
    }

    uint32_t prdLen = group_params.nUe * group_params.nPrbGrp * group_params.nBsAnt * group_params.nBsAnt;
    uint32_t detLen = group_params.nUe * group_params.nPrbGrp * group_params.nBsAnt * group_params.nBsAnt;
    uint32_t hLen = group_params.nPrbGrp * group_params.nUe * group_params.nCell * group_params.nBsAnt * group_params.nUeAnt;

    malloc_cumac_buf(task, &grpPrms->cellId, &buf_num.cellId, group_params.nCell);

    malloc_cumac_buf(task, &grpPrms->cellAssoc, &buf_num.cellAssoc, group_params.nCell * group_params.nUe);
    malloc_cumac_buf(task, &grpPrms->cellAssocActUe, &buf_num.cellAssocActUe, group_params.nCell * group_params.nActiveUe);

    CHECK_CUDA_ERR(cudaMemset(grpPrms->cellAssoc, 0, static_cast<size_t>(group_params.nCell) * group_params.nUe));
    CHECK_CUDA_ERR(cudaMemset(grpPrms->cellAssocActUe, 0, static_cast<size_t>(group_params.nCell) * group_params.nActiveUe));

    malloc_cumac_buf(task, &grpPrms->blerTargetActUe, &buf_num.blerTargetActUe, group_params.nActiveUe);
    CHECK_CUDA_ERR(cudaMemcpy(grpPrms->blerTargetActUe, blerTargetActUe, sizeof(float) * group_params.nActiveUe, cudaMemcpyHostToDevice));

    malloc_cumac_buf(task, &grpPrms->wbSinr, &buf_num.wbSinr, group_params.nActiveUe * group_params.nUeAnt);

    malloc_cumac_buf(task, &grpPrms->postEqSinr, &buf_num.postEqSinr, group_params.nActiveUe * group_params.nPrbGrp * group_params.nUeAnt);
    malloc_cumac_buf(task, &grpPrms->sinVal, &buf_num.sinVal, group_params.nUe * group_params.nPrbGrp * group_params.nUeAnt);

    malloc_cumac_buf(task, &grpPrms->prdMat, &buf_num.prdMat, prdLen);
    malloc_cumac_buf(task, &grpPrms->detMat, &buf_num.detMat, detLen);
    malloc_cumac_buf(task, &grpPrms->estH_fr, &buf_num.estH_fr, hLen);

    malloc_cumac_buf(task, &schdSol->setSchdUePerCellTTI, &buf_num.setSchdUePerCellTTI, group_params.nUe);
    if (group_params.allocType == 1) {
        malloc_cumac_buf(task, &schdSol->allocSol, &buf_num.allocSol, group_params.nUe * 2);

        uint32_t pfSize = group_params.nPrbGrp * group_params.numUeSchdPerCellTTI;
        uint32_t pow2N = 2;
        while (pow2N < pfSize)
        {
            pow2N = pow2N << 1;
        }
        malloc_cumac_buf(task, &schdSol->pfMetricArr, &buf_num.pfMetricArr, group_params.nCell * pow2N);
        malloc_cumac_buf(task, &schdSol->pfIdArr, &buf_num.pfIdArr, group_params.nCell * pow2N);
    } else {
        malloc_cumac_buf(task, &schdSol->allocSol, &buf_num.allocSol, group_params.nCell * group_params.nPrbGrp);
        schdSol->pfMetricArr = nullptr;
        schdSol->pfIdArr = nullptr;
    }

    malloc_cumac_buf(task, &schdSol->layerSelSol, &buf_num.layerSelSol, group_params.nUe);
    malloc_cumac_buf(task, &schdSol->mcsSelSol, &buf_num.mcsSelSol, group_params.nUe);

    malloc_cumac_buf(task, &cellGrpUeStatus->avgRatesActUe, &buf_num.avgRatesActUe, group_params.nActiveUe);
    malloc_cumac_buf(task, &cellGrpUeStatus->avgRates, &buf_num.avgRates, group_params.nUe);
    malloc_cumac_buf(task, &cellGrpUeStatus->newDataActUe, &buf_num.newDataActUe, group_params.nActiveUe);

    malloc_cumac_buf(task, &cellGrpUeStatus->tbErrLastActUe, &buf_num.tbErrLastActUe, group_params.nActiveUe);
    malloc_cumac_buf(task, &cellGrpUeStatus->tbErrLast, &buf_num.tbErrLast, group_params.nUe);

    // Allocate memory for pfmSort
    malloc_cumac_buf(task, &task->pfmCellInfo, &buf_num.pfmCellInfo, group_params.nCell);
    malloc_cumac_buf(task, &task->output_pfmSortSol, &buf_num.pfmSortSol, group_params.nCell, 1);

    // Init some static CUDA buffers
    for (uint16_t cell_id = 0; cell_id < grpPrms->nCell; cell_id++)
    {

        if (task->run_in_cpu)
        {
            *(grpPrms->cellId + cell_id) = cell_id;
        }
        else
        {
            CHECK_CUDA_ERR(cudaMemcpy(grpPrms->cellId + cell_id, &cell_id, sizeof(uint16_t), cudaMemcpyHostToDevice));
        }
    }

    // Alloc host-pinned memory for block copy, force in CPU memory
    malloc_cumac_buf(task, &task->input_avgRatesActUe, nullptr, buf_num.avgRatesActUe, 1);
    malloc_cumac_buf(task, &task->input_avgRates, nullptr, buf_num.avgRates, 1);
    malloc_cumac_buf(task, &task->input_tbErrLastActUe, nullptr, buf_num.tbErrLastActUe, 1);
    malloc_cumac_buf(task, &task->input_tbErrLast, nullptr, buf_num.tbErrLast, 1);
    malloc_cumac_buf(task, &task->input_estH_fr, nullptr, buf_num.estH_fr, 1);

    malloc_cumac_buf(task, &task->output_setSchdUePerCellTTI, nullptr, buf_num.setSchdUePerCellTTI, 1);
    malloc_cumac_buf(task, &task->output_allocSol, nullptr, buf_num.allocSol, 1);
    malloc_cumac_buf(task, &task->output_layerSelSol, nullptr, buf_num.layerSelSol, 1);
    malloc_cumac_buf(task, &task->output_mcsSelSol, nullptr, buf_num.mcsSelSol, 1);

    CHECK_CUDA_ERR(cudaMemcpy(&task->gpu_task_info->grpPrms, &task->grpPrms, sizeof(struct cumac::cumacCellGrpPrms), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(&task->gpu_task_info->ueStatus, &task->ueStatus, sizeof(struct cumac::cumacCellGrpUeStatus), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(&task->gpu_task_info->schdSol, &task->schdSol, sizeof(struct cumac::cumacSchdSol), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(&task->gpu_task_info->data_num, &buf_num, sizeof(cumac_buf_num_t), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(&task->gpu_task_info->pfmCellInfo, &task->pfmCellInfo, sizeof(cumac_pfm_cell_info_t*), cudaMemcpyHostToDevice));

    if (task->group_buf_enabled)
    {
        uint8_t**  tmp_prgMsk_array = nullptr;
        uint32_t tmp_size = group_params.nCell * sizeof(uint8_t *);
        malloc_cumac_buf(task, &tmp_prgMsk_array, nullptr, group_params.nCell);
        CHECK_CUDA_ERR(cudaMemcpy(&task->gpu_task_info->grpPrms.prgMsk, &tmp_prgMsk_array, sizeof(uint8_t**), cudaMemcpyHostToDevice));
        CHECK_CUDA_ERR(cudaMemcpy(tmp_prgMsk_array, grpPrms->prgMsk, tmp_size, cudaMemcpyHostToDevice));
    }

    task->init_cumac_modules();

    NVLOGI_FMT(TAG, "{}: group_buf_size={} group_buf_offset={} group_buf_enabled={}", __func__, group_buf_size, task->group_buf_offset, task->group_buf_enabled);
    return 0;
}

void cumac_cp_handler::set_task_ring(nv::lock_free_ring_pool<cumac_task>* ring, sem_t* sem) {
    task_ring = ring;
    task_sem = sem;
}

void cumac_cp_handler::on_config_request(nv_ipc_msg_t& msg) {

    cumac_config_req_t* req = reinterpret_cast<cumac_config_req_t*>(msg.msg_buf);
    cumac_cell_configs_t& cfg = cell_configs[msg.cell_id];

    // Copy cell configs first so we have access to the config data
    cfg = *reinterpret_cast<cumac_cell_configs_t*>(req->body);

    if (configured_cell_num == 0) {
        // Allocate blerTargetActUe for all cells when first cell config is received
        const size_t blerTargetActUe_num = static_cast<size_t>(cfg.nMaxActUePerCell) * static_cast<size_t>(cfg.nMaxCell);
        blerTargetActUe = reinterpret_cast<float*>(malloc(sizeof(float) * blerTargetActUe_num));
        if (blerTargetActUe == nullptr) {
            NVLOGE_FMT(TAG, AERIAL_CUMAC_CP_EVENT, "{}: malloc failed for blerTargetActUe, num={}", __func__, blerTargetActUe_num);
            return;
        }
        for (size_t i = 0; i < blerTargetActUe_num; i++) {
            blerTargetActUe[i] = cfg.blerTarget;
        }
    }

    NVLOGC_FMT(TAG, "{}: cell_id={} nMaxCell={} nMaxPrg={} nPrbPerPrg={} nMaxBsAnt={} nMaxUeAnt={} harqEnabledInd={}",
            __func__, msg.cell_id, cfg.nMaxCell, cfg.nMaxPrg, cfg.nPrbPerPrg, cfg.nMaxBsAnt, cfg.nMaxUeAnt, cfg.harqEnabledInd);

    // TODO: handle re-config
    configured_cell_num++;
    if (configured_cell_num == configs.cell_num) {
        check_config_params();
    }

    // Send response after config
    nv::phy_mac_transport& transp = transport(msg.cell_id);
    nv::phy_mac_msg_desc msg_desc;
    if (transp.tx_alloc(msg_desc) < 0)
    {
        return;
    }

    auto resp = cumac_init_msg_header<cumac_config_resp_t>(&msg_desc, CUMAC_CONFIG_RESPONSE, msg.cell_id);
    resp->error_code = 0;

    NVLOGI_FMT(TAG, "SEND: cell_id={} msg_id=0x{:02X} {}", msg.cell_id, msg_desc.msg_id, get_cumac_msg_name(msg_desc.msg_id));

    transp.tx_send(msg_desc);
    transp.tx_post();
}

void cumac_cp_handler::on_start_request(nv_ipc_msg_t& msg)
 {
    nv::phy_mac_transport& transp = transport(msg.cell_id);
    nv::phy_mac_msg_desc msg_desc;
    if (transp.tx_alloc(msg_desc) < 0)
    {
        return;
    }

    auto resp = cumac_init_msg_header<cumac_start_resp_t>(&msg_desc, CUMAC_START_RESPONSE, msg.cell_id);
    resp->error_code = 0;

    NVLOGI_FMT(TAG, "SEND: cell_id={} msg_id=0x{:02X} {}", msg.cell_id, msg_desc.msg_id, get_cumac_msg_name(msg_desc.msg_id));

    transp.tx_send(msg_desc);
    transp.tx_post();
}

void cumac_cp_handler::on_stop_request(nv_ipc_msg_t& msg)
{
    nv::phy_mac_transport& transp = transport(msg.cell_id);
    nv::phy_mac_msg_desc msg_desc;
    if (transp.tx_alloc(msg_desc) < 0)
    {
        return;
    }

    auto resp = cumac_init_msg_header<cumac_stop_resp_t>(&msg_desc, CUMAC_STOP_RESPONSE, msg.cell_id);
    resp->error_code = 0;

    NVLOGI_FMT(TAG, "SEND: cell_id={} msg_id=0x{:02X} {}", msg.cell_id, msg_desc.msg_id, get_cumac_msg_name(msg_desc.msg_id));

    transp.tx_send(msg_desc);
    transp.tx_post();
}

template <typename T>
int copy_from_ipc_buf(nv::phy_mac_transport &transp, nv_ipc_msg_t &msg, cumac_task *task, T *dst_buf, const char *info, uint32_t &src_offset_in_bytes, uint32_t &dst_offset_in_num, uint32_t num)
{
    if (task->run_in_cpu)
    {
        transp.copy_from_data_buf(msg, src_offset_in_bytes, dst_buf + dst_offset_in_num, num * sizeof(T));
    }
    else if (task->group_buf_enabled == 0)
    {
        uint8_t *src = reinterpret_cast<uint8_t *>(msg.data_buf);
        CHECK_CUDA_ERR(cudaMemcpyAsync(dst_buf + dst_offset_in_num, src + src_offset_in_bytes, num * sizeof(T), cudaMemcpyHostToDevice, task->strm));
    }
    dst_offset_in_num += num;
    return 0;
}

void cumac_cp_handler::cell_copy_task(nv_ipc_msg_t &msg, cumac_task *task)
{
    cumac_sch_tti_req_t &head = *reinterpret_cast<cumac_sch_tti_req_t *>(msg.msg_buf);
    cumac_tti_req_payload_t &req = head.payload;
    uint8_t *src = reinterpret_cast<uint8_t *>(msg.data_buf);
    uint32_t nBlock = group_params.nUe * req.nBsAnt * req.nUeAnt;

    struct cumacCellGrpPrms &grpPrms = task->grpPrms;

    if (req.taskBitMask & (0x1 << CUMAC_TASK_PRB_ALLOCATION)) // multiCellScheduler buffers
    {
        // task->data_num.estH_fr += req.nPrbGrp * nBlock;

        // if (task->tv != nullptr && task->debug_option & DBG_OPT_WAR_COPY_GROUP_TV)
        // {
        // return;
        // }

        // copy_from_ipc_buf(transp, msg, task, grpPrms->estH_fr, "estH_fr", req.offsets.estH_fr, task->data_num.estH_fr, hLen);
        cuComplex(*dst_estH_fr)[grpPrms.nPrbGrp][grpPrms.nCell][grpPrms.nUe][grpPrms.nBsAnt][grpPrms.nUeAnt] = reinterpret_cast<cuComplex(*)[grpPrms.nPrbGrp][grpPrms.nCell][grpPrms.nUe][grpPrms.nBsAnt][grpPrms.nUeAnt]>(task->input_estH_fr);
        cuComplex(*src_estH_fr)[grpPrms.nPrbGrp][grpPrms.nUe][grpPrms.nBsAnt][grpPrms.nUeAnt] = reinterpret_cast<cuComplex(*)[grpPrms.nPrbGrp][grpPrms.nUe][grpPrms.nBsAnt][grpPrms.nUeAnt]>(src + req.offsets.estH_fr);
        for (int prgId = 0; prgId < req.nPrbGrp; prgId++)
        {
            memcpy((*dst_estH_fr)[prgId][msg.cell_id], (*src_estH_fr)[prgId], nBlock * sizeof(cuComplex));
        }

        // cuComplex *base_estH_fr = reinterpret_cast<cuComplex *>(src + req.offsets.estH_fr);
        // for (int prgId = 0; prgId < req.nPrbGrp; prgId++)
        // {
        //     int indexGroup = prgId * group_params.nCell * nBlock + msg.cell_id * nBlock;
        //     int indexCell = prgId * nBlock;
        //     memcpy(task->input_estH_fr + indexGroup, base_estH_fr + indexCell, nBlock * sizeof(cuComplex));
        // }
    }
}

// This function is called in cell_id order
void cumac_cp_handler::on_sch_tti_request(nv_ipc_msg_t& msg, cumac_task* task)
{
    uint16_t cell_id = msg.cell_id;
    sfn_slot_t ss_msg = nv_ipc_get_sfn_slot(&msg);

    nv::phy_mac_transport& transp = transport(cell_id);

    cumac_cell_configs_t& cfg = cell_configs[cell_id];

    cumac_sch_tti_req_t& head = *reinterpret_cast<cumac_sch_tti_req_t*>(msg.msg_buf);
    cumac_tti_req_payload_t& req = head.payload;

    if (task == nullptr)
    {
        NVLOGE_FMT(TAG, AERIAL_CUMAC_CP_EVENT, "SFN {}.{} cell_id={} task == nullptr", head.sfn, head.slot, cell_id);
        return;
    }

    task->taskBitMask = req.taskBitMask;

    task->grpPrms.sigmaSqrd = req.sigmaSqrd; // Use request value; no longer hardcoded to 1.0
    task->grpPrms.nActiveUe += req.nActiveUe;

    struct cumacCellGrpPrms* grpPrms = &task->grpPrms;
    struct cumacCellGrpUeStatus* ueStatus = &task->ueStatus;
    struct cumacSchdSol* schdSol = &task->schdSol;

    CHECK_VALUE_MAX_ERR(cell_id, group_params.nCell - 1);

    if (req.taskBitMask & 0x0F) // Check common parameters for the four 4T4R modules
    {
        // Below parameter validations are for debugging
        CHECK_VALUE_EQUAL_ERR(cell_id, req.cellID);
        CHECK_VALUE_MAX_ERR(grpPrms->numUeSchdPerCellTTI, cfg.nMaxSchUePerCell);
        CHECK_VALUE_MAX_ERR(grpPrms->nUe, grpPrms->numUeSchdPerCellTTI * grpPrms->nCell);
        // CHECK_VALUE_EQUAL_ERR(grpPrms->nActiveUe, req.nActiveUe * grpPrms->nCell);
        CHECK_VALUE_MAX_ERR(req.nActiveUe, cfg.nMaxActUePerCell);
        CHECK_VALUE_EQUAL_ERR(grpPrms->nCell, configs.cell_num);

        CHECK_VALUE_MAX_ERR(grpPrms->nPrbGrp, group_params.nPrbGrp);
        CHECK_VALUE_MAX_ERR(grpPrms->nBsAnt, req.nBsAnt);
        CHECK_VALUE_MAX_ERR(grpPrms->nUeAnt, req.nUeAnt);
        CHECK_VALUE_EQUAL_ERR(grpPrms->W, group_params.W);

        // CHECK_VALUE_EQUAL_ERR(grpPrms->sigmaSqrd, req.sigmaSqrd);
        // CHECK_VALUE_EQUAL_ERR(grpPrms->Pt_Rbg, 0);
        // CHECK_VALUE_EQUAL_ERR(grpPrms->Pt_rbgAnt, 0);
        CHECK_VALUE_EQUAL_ERR(grpPrms->precodingScheme, cfg.precoderType);

        CHECK_VALUE_EQUAL_ERR(grpPrms->receiverScheme, cfg.receiverType);
        CHECK_VALUE_EQUAL_ERR(grpPrms->allocType, cfg.allocType);
        CHECK_VALUE_EQUAL_ERR(grpPrms->betaCoeff, cfg.betaCoeff);
        CHECK_VALUE_EQUAL_ERR(grpPrms->sinValThr, cfg.sinValThr);

        CHECK_VALUE_EQUAL_ERR(grpPrms->corrThr, cfg.corrThr);
        CHECK_VALUE_EQUAL_ERR(grpPrms->prioWeightStep, cfg.prioWeightStep);

        CHECK_PTR_NULL_FATAL(msg.data_buf);
        CHECK_PTR_NULL_FATAL(grpPrms->wbSinr);
        CHECK_PTR_NULL_FATAL(ueStatus->avgRatesActUe);
        CHECK_PTR_NULL_FATAL(grpPrms->prgMsk[cell_id]);
    }

    // Copy data buffers
    uint8_t *src = reinterpret_cast<uint8_t *>(msg.data_buf);

    if (req.taskBitMask & (0x1 << CUMAC_TASK_UE_SELECTION)) // multiCellUeSelection buffers
    {
        if (task->run_in_cpu)
        {
            // Populate cellAssocActUe buffer: (Assume each cell has the same nActiveUe)
            // ue_id_offset = req.nActiveUe * cell_id
            // cell_offset = group_params.nActiveUe * cell_id
            // home_offset_for_cell_cellAssocActUe = cell_offset + ue_id_offset = (group_params.nActiveUe + req.nActiveUe ) * cell_id
            uint32_t total_active_ue = group_params.nCell * req.nActiveUe;
            memset(grpPrms->cellAssocActUe + (total_active_ue + req.nActiveUe) * cell_id, 1, req.nActiveUe);
            memset(grpPrms->cellAssoc + (group_params.numUeSchdPerCellTTI * group_params.nCell + group_params.numUeSchdPerCellTTI) * cell_id, 1, group_params.numUeSchdPerCellTTI);
        }
        else if (task->group_buf_enabled == 0)
        {
            // Populate cellAssocActUe buffer: (Assume each cell has the same nActiveUe)
            // ue_id_offset = req.nActiveUe * cell_id
            // cell_offset = group_params.nActiveUe * cell_id
            // home_offset_for_cell_cellAssocActUe = cell_offset + ue_id_offset = (group_params.nActiveUe + req.nActiveUe ) * cell_id
            uint32_t total_active_ue = group_params.nCell * req.nActiveUe;
            CHECK_CUDA_ERR(cudaMemsetAsync(grpPrms->cellAssocActUe + (total_active_ue + req.nActiveUe) * cell_id, 1, req.nActiveUe, task->strm));
            CHECK_CUDA_ERR(cudaMemsetAsync(grpPrms->cellAssoc + (group_params.numUeSchdPerCellTTI * group_params.nCell + group_params.numUeSchdPerCellTTI) * cell_id, 1, group_params.numUeSchdPerCellTTI, task->strm));
        }

        task->data_num.cellId += 1;
        task->data_num.cellAssocActUe += group_params.nCell * req.nActiveUe; // Actual count this TTI, not configured max
        task->data_num.cellAssoc += group_params.nCell * group_params.numUeSchdPerCellTTI;

        uint32_t prgMsk_num = 0;
        copy_from_ipc_buf(transp, msg, task, grpPrms->prgMsk[cell_id], "prgMsk", req.offsets.prgMsk, prgMsk_num, req.nPrbGrp);
        task->data_num.prgMsk = prgMsk_num;

        copy_from_ipc_buf(transp, msg, task, grpPrms->wbSinr, "wbSinr", req.offsets.wbSinr, task->data_num.wbSinr, req.nActiveUe * req.nUeAnt);

        // Copy to a contiguous CPU buffer for later selection
        memcpy(task->input_avgRatesActUe + task->data_num.avgRatesActUe, src + req.offsets.avgRatesActUe, req.nActiveUe * sizeof(float));
        copy_from_ipc_buf(transp, msg, task, ueStatus->avgRatesActUe, "avgRatesActUe", req.offsets.avgRatesActUe, task->data_num.avgRatesActUe, req.nActiveUe);

        if (task->debug_option & DBG_OPT_PRINT_NVIPC_BUF) // Dump IPC buffers
        {
            NVLOGI_FMT_ARRAY(TAG, "NVIPC_avgRatesActUe", reinterpret_cast<float*>(src + req.offsets.avgRatesActUe), req.nActiveUe);
        }
    }

    if (req.taskBitMask & (0x1 << CUMAC_TASK_PRB_ALLOCATION)) // multiCellScheduler buffers
    {
        // estH_fr data_num
        task->data_num.estH_fr += req.nPrbGrp * group_params.nUe * req.nBsAnt * req.nUeAnt;

        copy_from_ipc_buf(transp, msg, task, grpPrms->postEqSinr, "postEqSinr", req.offsets.postEqSinr, task->data_num.postEqSinr, req.nActiveUe * req.nPrbGrp * req.nUeAnt);
        copy_from_ipc_buf(transp, msg, task, grpPrms->sinVal, "sinVal", req.offsets.sinVal, task->data_num.sinVal, group_params.numUeSchdPerCellTTI * req.nPrbGrp * req.nUeAnt);

        uint32_t prdLen = group_params.numUeSchdPerCellTTI * req.nPrbGrp * req.nBsAnt * req.nBsAnt;
        uint32_t detLen = group_params.numUeSchdPerCellTTI * req.nPrbGrp * req.nUeAnt * req.nUeAnt;

        copy_from_ipc_buf(transp, msg, task, grpPrms->detMat, "detMat", req.offsets.detMat, task->data_num.detMat, detLen);
        copy_from_ipc_buf(transp, msg, task, grpPrms->prdMat, "prdMat", req.offsets.prdMat, task->data_num.prdMat, prdLen);
    }

    if (req.taskBitMask & (0x1 << CUMAC_TASK_MCS_SELECTION)) // mcsSelectionLUT buffers
    {
        // Copy to a contiguous CPU buffer for later selection
        memcpy(task->input_tbErrLastActUe + task->data_num.tbErrLastActUe, src + req.offsets.tbErrLastActUe, req.nActiveUe * sizeof(int8_t));
        copy_from_ipc_buf(transp, msg, task, ueStatus->tbErrLastActUe, "tbErrLastActUe", req.offsets.tbErrLastActUe, task->data_num.tbErrLastActUe, req.nActiveUe);
    }

    // Handle pfmSort task setup
    if (req.taskBitMask & (0x1 << CUMAC_TASK_PFM_SORT))
    {
        if (task->debug_option & DBG_OPT_PRINT_NVIPC_BUF) // Dump IPC buffers
        {
            uint8_t* pfmCellInfo_buf = reinterpret_cast<uint8_t*>(msg.data_buf) + req.offsets.pfmCellInfo;
            NVLOGI_FMT_ARRAY(TAG, "NVIPC_pfmCellInfo", pfmCellInfo_buf, sizeof(cumac_pfm_cell_info_t));
        }
        copy_from_ipc_buf(transp, msg, task, task->pfmCellInfo, "pfmCellInfo", req.offsets.pfmCellInfo, task->data_num.pfmCellInfo, 1);
    }

    if (task->group_buf_enabled) // Copy the whole msg.data_buf to the GPU group buffer
    {
        cell_desc_t *cell_desc = task->cpu_cell_descs + cell_id;
        CHECK_CUDA_ERR(cudaMemcpyAsync(cell_desc->home, src, msg.data_len, cudaMemcpyHostToDevice, task->strm));
        memcpy(&cell_desc->offsets, &req.offsets, sizeof(cumac_tti_req_buf_offsets_t));
    }

    NVLOGI_FMT(TAG, "SFN {}.{} RECV: SCH_TTI.req cell_id={} cellID={} ULDLSch={} nActiveUe={} nBsAnt={} nUeAnt={} group: nCell={} nActiveUe={} nUe={}",
        head.sfn, head.slot, cell_id, req.cellID, req.ULDLSch, req.nActiveUe, req.nBsAnt, req.nUeAnt, grpPrms->nCell, grpPrms->nActiveUe, grpPrms->nUe);
}

void cumac_cp_handler::print_cumac_cp_thrput(uint64_t slot_counter)
{
    for(int cell_id = 0; cell_id < configs.cell_num; cell_id++)
    {
        cumac_cp_thrput_t& thrput = thrputs[cell_id];
        // Console log print per second
        NVLOGC_FMT(TAG, "Cell {:2} | CUMAC {:4} | ERR {:4} | Slots {}",
                   cell_id,
                   thrput.cumac_slots.load(),
                   thrput.error.load(),
                   slot_counter);

        thrput.reset();
    }
}

void cumac_cp_handler::push_cumac_task(sfn_slot_t ss)
{
    cumac_task* task = get_cumac_task(ss);
    struct cumac::cumacCellGrpPrms& grpPrms = task->grpPrms;

    NVLOGI_FMT(TAG, "SFN {}.{} PUSH_TASK: 0x{:X} nUe={} nActiveUe={} numUeSchdPerCellTTI={} nCell={} nPrbGrp={} nBsAnt={} nUeAnt={} precodingScheme={} receiverScheme={} allocType={} prioWeightStep={}", ss.u16.sfn, ss.u16.slot,
               task->taskBitMask, grpPrms.nUe, grpPrms.nActiveUe, grpPrms.numUeSchdPerCellTTI, grpPrms.nCell, grpPrms.nPrbGrp, grpPrms.nBsAnt, grpPrms.nUeAnt, grpPrms.precodingScheme, grpPrms.receiverScheme, grpPrms.allocType, grpPrms.prioWeightStep);

    // Add buffer size check
    task->calculate_output_data_num();
    check_task_buf_size(task);

    task->ts_enqueue = std::chrono::system_clock::now().time_since_epoch().count();
    task_ring->enqueue(task);
    sem_post(task_sem);
}

int cumac_cp_handler::send_sch_tti_response(cumac_task *task, int cell_id)
{
    size_t offset = 0;

    nv::phy_mac_transport& transp = transport(cell_id);

    nv::phy_mac_msg_desc msg;
    msg.data_pool = NV_IPC_MEMPOOL_CPU_DATA;

    cumac_cell_configs_t &cell_cfg = cell_configs[cell_id];
    uint32_t task_mask = task->taskBitMask;

    if (transp.tx_alloc(msg) < 0)
    {
        return -1;
    }

    // Get the request message header and payload
    nv::phy_mac_msg_desc& req_msg = task->tti_reqs[cell_id];
    cumac_sch_tti_req_t& req_head = *reinterpret_cast<cumac_sch_tti_req_t*>(req_msg.msg_buf);
    cumac_tti_req_payload_t& req = req_head.payload;

    cumac_sch_tti_resp_t& resp = *cumac_init_msg_header<cumac_sch_tti_resp_t>(&msg, CUMAC_SCH_TTI_RESPONSE, cell_id);
    resp.sfn = task->ss.u16.sfn;
    resp.slot = task->ss.u16.slot;

    // Per-cell slice length in setSchdUePerCellTTI / allocSol / layer / MCS buffers (nCell * nMaxSch from CONFIG)
    const size_t nMaxSchUePerCell = static_cast<size_t>(task->grpPrms.numUeSchdPerCellTTI);

    if ((task_mask & (0x1u << CUMAC_TASK_UE_SELECTION)) && task->output_setSchdUePerCellTTI != nullptr)
    {
        const uint16_t* row = task->output_setSchdUePerCellTTI + static_cast<size_t>(cell_id) * nMaxSchUePerCell;
        uint32_t nActual = 0;
        for (uint32_t i = 0; i < nMaxSchUePerCell; i++)
        {
            if (row[i] != 0xFFFFu)
            {
                nActual++;
            }
        }
        resp.nUeSchd = static_cast<uint16_t>(nActual);
    }
    else
    {
        // UE selection not run or no CPU copy: keep prior max-slots semantics for L2
        resp.nUeSchd = static_cast<uint16_t>(nMaxSchUePerCell);
    }

    memset(&resp.offsets, INVALID_CUMAC_BUF_OFFSET, sizeof(resp.offsets));

    uint32_t ipc_offset = 0;

    if (task_mask & (0x1 << CUMAC_TASK_UE_SELECTION)) // UE_SELECTION result
    {
        size_t setSchdUePerCellTTI_offset = nMaxSchUePerCell * cell_id;
        transp.copy_to_data_buf(msg, ipc_offset, task->output_setSchdUePerCellTTI + setSchdUePerCellTTI_offset, sizeof(*task->output_setSchdUePerCellTTI) * resp.nUeSchd);
        // Map L1 group ue_id to L2 per cell ue_id
        uint16_t *ue_id_buf = reinterpret_cast<uint16_t *>(reinterpret_cast<uint8_t *>(msg.data_buf) + ipc_offset);
        uint32_t ue_id_base = cell_id * task->grpPrms.nActiveUe / task->grpPrms.nCell;
        for (uint32_t i = 0; i < resp.nUeSchd; i++)
        {
            if (ue_id_buf[i] != 0xFFFFu)
            {
                ue_id_buf[i] = static_cast<uint16_t>(ue_id_buf[i] - ue_id_base);
            }
        }
        resp.offsets.setSchdUePerCellTTI = ipc_offset;
        ipc_offset += resp.nUeSchd * sizeof(*task->output_setSchdUePerCellTTI);
        if (task->debug_option & DBG_OPT_PRINT_NVIPC_BUF) // Dump IPC buffers
        {
            NVLOGI_FMT_ARRAY(TAG, "ARRAY setSchdUePerCellTTI", ue_id_buf, resp.nUeSchd);
        }
    }

    if (task_mask & (0x1 << CUMAC_TASK_PRB_ALLOCATION)) // PRB_ALLOCATION result
    {
        uint32_t allocSol_num = cell_cfg.allocType == 1 ? 2 * resp.nUeSchd : task->grpPrms.nPrbGrp;
        size_t allocSol_offset = cell_cfg.allocType == 1 ? 2 * static_cast<size_t>(nMaxSchUePerCell) * cell_id : static_cast<size_t>(task->grpPrms.nPrbGrp) * cell_id;
        transp.copy_to_data_buf(msg, ipc_offset, task->output_allocSol + allocSol_offset, sizeof(*task->output_allocSol) * allocSol_num);
        resp.offsets.allocSol = ipc_offset;
        ipc_offset += allocSol_num * sizeof(*task->output_allocSol);
    }

    if (task_mask & (0x1 << CUMAC_TASK_LAYER_SELECTION)) // LAYER_SELECTION result
    {
        size_t layerSelSol_offset = static_cast<size_t>(nMaxSchUePerCell) * cell_id;
        transp.copy_to_data_buf(msg, ipc_offset, task->output_layerSelSol + layerSelSol_offset, sizeof(*task->output_layerSelSol) * resp.nUeSchd);
        resp.offsets.layerSelSol = ipc_offset;
        ipc_offset += resp.nUeSchd * sizeof(*task->output_layerSelSol);
    }

    if (task_mask & (0x1 << CUMAC_TASK_MCS_SELECTION)) // MCS_SELECTION result
    {
        size_t mcsSelSol_offset = static_cast<size_t>(nMaxSchUePerCell) * cell_id;
        transp.copy_to_data_buf(msg, ipc_offset, task->output_mcsSelSol + mcsSelSol_offset, sizeof(*task->output_mcsSelSol) * resp.nUeSchd);
        resp.offsets.mcsSelSol = ipc_offset;
        ipc_offset += resp.nUeSchd * sizeof(*task->output_mcsSelSol);
    }

    if (task_mask & (0x1 << CUMAC_TASK_PFM_SORT)) // PFM_SORT result
    {
        transp.copy_to_data_buf(msg, ipc_offset, task->output_pfmSortSol + cell_id, sizeof(*task->output_pfmSortSol));
        resp.offsets.pfmSortSol = ipc_offset;
        ipc_offset += sizeof(*task->output_pfmSortSol);
    }

    msg.data_len = ipc_offset;

    NVLOGI_FMT(TAG, "SFN {}.{} SEND: SCH_TTI.resp cell_id={} msg_len={} data_len={} nActiveUe={} nMaxSchUePerCell={} nUeSchd={} allocSol_offset={} layerSelSol_offset={} mcsSelSol_offset={} setSchdUePerCellTTI_offset={}",
               resp.sfn, resp.slot, cell_id, msg.msg_len, msg.data_len, task->grpPrms.nActiveUe, nMaxSchUePerCell, resp.nUeSchd, resp.offsets.allocSol, resp.offsets.layerSelSol, resp.offsets.mcsSelSol, resp.offsets.setSchdUePerCellTTI);

    transp.tx_send(msg);
    transp.notify(1);

    thrputs[cell_id].cumac_slots ++;

    return 0;
}

sched_slot_data& cumac_cp_handler::get_sched_slot_data(sfn_slot_t ss) {
    return sched_slots[ss.u16.slot & 0x03];
}

void cumac_cp_handler::handle_slot_msg(nv::phy_mac_msg_desc &msg_desc, sfn_slot_t ss_msg)
{
    sfn_slot_t ss_curr = this->ss_curr.load();
    NVLOGI_FMT(TAG, "SFN {}.{} HANDLE: cell_id={} msg_id=0x{:02X} {} SFN {}.{}",
               ss_curr.u16.sfn, ss_curr.u16.slot, msg_desc.cell_id, msg_desc.msg_id, get_cumac_msg_name(msg_desc.msg_id), ss_msg.u16.sfn, ss_msg.u16.slot);

    nv::phy_mac_transport& transp = transport(msg_desc.cell_id);

    if (configured_cell_num < configs.cell_num)
    {
        NVLOGW_FMT(TAG, "SFN {}.{} cell_id={} msg_id=0x{:02X} {} skip before all cells configured",
            ss_msg.u16.sfn, ss_msg.u16.slot, msg_desc.cell_id, msg_desc.msg_id, get_cumac_msg_name(msg_desc.msg_id));
        transp.rx_release(msg_desc);
        return;
    }

    sched_slot_data &slot_data = get_sched_slot_data(ss_msg);
    if (ss_msg.u32 != slot_data.ss_sched.u32)
    {
        // This is the first message of a new slot, reset slot_data and cuda_task buffer
        NVLOGI_FMT(TAG, "SFN {}.{} received: SFN {}.{} cell_id={} msg_id=0x{:02X} {} for new slot", slot_data.ss_sched.u16.sfn, slot_data.ss_sched.u16.slot,
                   ss_msg.u16.sfn, ss_msg.u16.slot, msg_desc.cell_id, msg_desc.msg_id, get_cumac_msg_name(msg_desc.msg_id));
        global_tick ++;
        slot_data.reset_slot_data(ss_msg);
        if (slot_data.task != nullptr)
        {
            slot_data.task->ts_start = transp.get_ts_send(msg_desc);
        }
    }

    // Handle slot messages
    switch (msg_desc.msg_id)
    {
    case CUMAC_SCH_TTI_REQUEST:
        if (slot_data.task != nullptr)
        {
            // Do not release IPC buffer until async copy finished
            slot_data.task->tti_reqs[msg_desc.cell_id] = msg_desc;
            // on_sch_tti_request(msg_desc, slot_data.task);
        }
        else
        {
            // Drop the message if not assigned task buffer successfully
            transp.rx_release(msg_desc);
        }
        break;
    case CUMAC_TTI_END:
        // Start handling next cell after current cell ended
        slot_data.curr_cell_id++;
        if (slot_data.curr_cell_id >= slot_data.cell_num)
        {
            if (slot_data.task != nullptr)
            {
                // SLOT messages ended, create and push cumac_task into task queue
                slot_data.task->ts_last_send = transp.get_ts_send(msg_desc);
                slot_data.task->ts_last_recv = std::chrono::system_clock::now().time_since_epoch().count();
                push_cumac_task(ss_msg);
                slot_data.task = nullptr;
                slot_data.ss_sched = {.u32 = SFN_SLOT_INVALID};
            }

            // Print throughput every second
            if (global_tick > 0 && global_tick % SLOTS_PER_SECOND == 0)
            {
                print_cumac_cp_thrput(global_tick);
            }
        }
        transp.rx_release(msg_desc);
        break;
    default:
        break;
    }
}

void cumac_cp_handler::handle_slot_msg_reorder(nv::phy_mac_msg_desc& msg_desc, sfn_slot_t ss_msg)
{
    sfn_slot_t ss_curr = this->ss_curr.load();
    NVLOGI_FMT(TAG, "SFN {}.{} HANDLE_ORIGIN: cell_id={} msg_id=0x{:02X} {} SFN {}.{}",
        ss_curr.u16.sfn, ss_curr.u16.slot, msg_desc.cell_id, msg_desc.msg_id, get_cumac_msg_name(msg_desc.msg_id), ss_msg.u16.sfn, ss_msg.u16.slot);

    sched_slot_data& slot_data = get_sched_slot_data(ss_msg);
    if (ss_msg.u32 != slot_data.ss_sched.u32) {
        // This is the first message of a new slot, reset slot_data and cuda_task buffer
        NVLOGI_FMT(TAG, "SFN {}.{} received: SFN {}.{} cell_id={} msg_id=0x{:02X} {} for new slot", slot_data.ss_sched.u16.sfn, slot_data.ss_sched.u16.slot,
                   ss_msg.u16.sfn, ss_msg.u16.slot, msg_desc.cell_id, msg_desc.msg_id, get_cumac_msg_name(msg_desc.msg_id));
        slot_data.reset_slot_data(ss_msg);
        slot_data.task->ts_start = transport(msg_desc.cell_id).get_ts_send(msg_desc);
    }

    slot_data.slot_msgs[msg_desc.cell_id].push_msg(msg_desc);

    // Check if current cell_id message arrived
    nv::phy_mac_msg_desc* msg = nullptr;
    while (slot_data.curr_cell_id < slot_data.cell_num && ((msg = slot_data.slot_msgs[slot_data.curr_cell_id].pull_msg()) != nullptr)) {
        handle_slot_msg(msg_desc, ss_msg);
    }
}