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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime_api.h>

#include "cumac_cuda.hpp"
#include "nvlog.hpp"

#define TAG (NVLOG_TAG_BASE_CUMAC_CP + 5) // "CUMCP.TASK"

#define N_THREAD_PER_BLOCK 32

#define offset_of(st, m) ((size_t)&(((st *)0)->m))

#define req_offset(member) ((size_t)&(((cumac_tti_req_buf_offsets_t *)0)->member))

inline cudaError __checkLastCudaError(const char* file, int line)
{
    cudaError lastErr = cudaGetLastError();
    if(lastErr != cudaSuccess)
    {
        NVLOGE_FMT(TAG, AERIAL_CUDA_API_EVENT, "Error at {} line {}: {}", file, line, cudaGetErrorString(lastErr));
    }
    return lastErr;
}

#define checkLastCudaError() __checkLastCudaError(__FILE__, __LINE__)

#define HANDLE_ERROR(x)                                                                 \
    do                                                                                  \
    {                                                                                   \
        if((x) != cudaSuccess) { printf("Error %s line%d\n", __FUNCTION__, __LINE__); } \
    } while(0)
#define HANDLE_NULL(x)

/**
 * Speculation barrier with CTA scope.
 *
 * @note Uses membar.cta to constrain speculative loads within the block.
 */
static __device__ __forceinline__ void speculation_barrier()
{
    asm volatile("membar.cta;" ::: "memory");
}

static __global__ void gpu_copy_avgRates(float *avgRatesActUe, float *avgRates, uint16_t *setSchdUePerCellTTI, uint32_t number, uint32_t max_active_ue)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < number; i += stride)
    {
        uint16_t ue_id = *(setSchdUePerCellTTI + i);
        if (ue_id < max_active_ue)
        {
            speculation_barrier(); // Add barrier to avoid speculative loads
            *(avgRates + i) = *(avgRatesActUe + ue_id);
        }
        else
        {
            *(avgRates + i) = 0.0f;
        }
    }
}

static __global__ void gpu_copy_tbErrLast(int8_t *tbErrLastActUe, int8_t *tbErrLast, uint16_t *setSchdUePerCellTTI, uint32_t number, uint32_t max_active_ue)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < number; i += stride)
    {
        uint16_t ue_id = *(setSchdUePerCellTTI + i);
        if (ue_id < max_active_ue)
        {
            speculation_barrier(); // Add barrier to avoid speculative loads
            *(tbErrLast + i) = *(tbErrLastActUe + ue_id);
        }
        else
        {
            *(tbErrLast + i) = 0;
        }
    }
}

template <typename T>
static __device__ void gpu_memset_element(T *group_data, cell_desc_t *cell_descs, uint32_t data_num, uint32_t cell_num)
{
    uint32_t total_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t cell_id = total_thread_id % cell_num;

    uint32_t numBlocks = gridDim.x;
    uint32_t threadsPerBlock = blockDim.x;

    uint32_t cell_buf_num = data_num / cell_num;

    // Calculate section information more robustly
    uint32_t total_threads = numBlocks * threadsPerBlock;
    uint32_t threads_per_cell = total_threads / cell_num;
    uint32_t section_id = total_thread_id / cell_num;
    uint32_t section_size = (cell_buf_num + threads_per_cell - 1) / threads_per_cell;

    // printf("%s %02u-%02u: data_num=%u cell_buf_num=%u cell_num %u-%u section %u-%u section_size=%u\n", __func__, blockIdx.x, threadIdx.x,
    //     data_num, cell_buf_num, cell_num, cell_id, threads_per_cell, section_id, section_size);

    uint32_t cell_section_offset = section_id * section_size;
    // uint32_t src_home_offset = *reinterpret_cast<uint32_t *>((reinterpret_cast<uint8_t *>(&cell_descs[cell_id].offsets) + offset));
    // T *src_base = reinterpret_cast<T *>(cell_descs[cell_id].home + src_home_offset) + cell_section_offset;
    // T *dst_base = group_data + cell_buf_num * cell_id + cell_section_offset;
    uint32_t section_offset = cell_buf_num * cell_id + cell_section_offset;

    uint32_t group_size = data_num / cell_num;
    uint32_t cell_size = group_size / cell_num;

    for (uint32_t i = 0; i < section_size; i++)
    {
        uint32_t total_offset = section_offset + i;
        if (total_offset >= data_num)
        {
            continue;
        }

        if (total_offset / group_size == (total_offset % group_size) / cell_size)
        {
            *(group_data + total_offset) = 1;
        }
        else
        {
            *(group_data + total_offset) = 0;
        }
    }
}

template <typename T>
static __device__ void gpu_copy_element(T *group_data, cell_desc_t *cell_descs, uint32_t offset, uint32_t data_num, uint32_t cell_num)
{
    uint32_t total_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t cell_id = total_thread_id % cell_num;

    uint32_t numBlocks = gridDim.x;
    uint32_t threadsPerBlock = blockDim.x;

    uint32_t cell_buf_num = data_num / cell_num;

    // Calculate section information more robustly
    uint32_t total_threads = numBlocks * threadsPerBlock;
    uint32_t threads_per_cell = total_threads / cell_num;
    uint32_t section_id = total_thread_id / cell_num;
    uint32_t section_size = (cell_buf_num + threads_per_cell - 1) / threads_per_cell;

    // printf("%s %02u-%02u: data_num=%u cell_buf_num=%u cell_num %u-%u section %u-%u section_size=%u\n", __func__, blockIdx.x, threadIdx.x,
    //     data_num, cell_buf_num, cell_num, cell_id, threads_per_cell, section_id, section_size);

    uint32_t cell_section_offset = section_id * section_size;
    uint32_t src_home_offset = *reinterpret_cast<uint32_t *>((reinterpret_cast<uint8_t *>(&cell_descs[cell_id].offsets) + offset));
    T *src_base = reinterpret_cast<T *>(cell_descs[cell_id].home + src_home_offset) + cell_section_offset;
    T *dst_base = group_data + cell_buf_num * cell_id + cell_section_offset;

    for (uint32_t i = 0; i < section_size; i++)
    {
        if (cell_section_offset + i < cell_buf_num)
        {
            *(dst_base + i) = *(src_base + i);
        }
        // else
        // {
        //     printf("%s %02u-%02u: data_num=%u cell_buf_num=%u src_home_offset=%u cell_num %u-%u section %u-%u - offset: %u+%u=%u skip\n", __func__, blockIdx.x, threadIdx.x,
        //         data_num, cell_buf_num, src_home_offset, cell_num, cell_id, threads_per_cell, section_id, cell_section_offset, i, cell_section_offset + i);
        // }
    }
}

static __device__ void gpu_copy_estH_fr(cumac_task_info_t *task_info, cell_desc_t *cell_descs, uint32_t offset, uint32_t cell_num)
{
    struct cumac::cumacCellGrpPrms& grpPrms =  task_info->grpPrms;
    // cuComplex *group_data = grpPrms.estH_fr;
    float *group_data = reinterpret_cast<float *>(grpPrms.estH_fr);
    uint32_t data_block = grpPrms.nUe * grpPrms.nBsAnt * grpPrms.nUeAnt;
    uint32_t data_num = data_block * grpPrms.nPrbGrp * cell_num;

    uint32_t total_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t cell_id = total_thread_id % cell_num;

    uint32_t numBlocks = gridDim.x;
    uint32_t threadsPerBlock = blockDim.x;

    uint32_t cell_buf_num = data_num / cell_num;

    // Calculate section information more robustly
    uint32_t total_threads = numBlocks * threadsPerBlock;
    uint32_t threads_per_cell = total_threads / cell_num;
    uint32_t section_id = total_thread_id / cell_num;
    uint32_t section_size = (cell_buf_num + threads_per_cell - 1) / threads_per_cell;

    // printf("%s %02u-%02u: data_num=%u cell_buf_num=%u cell_num %u-%u section %u-%u section_size=%u\n", __func__, blockIdx.x, threadIdx.x,
    //        data_num, cell_buf_num, cell_num, cell_id, threads_per_cell, section_id, section_size);

    uint32_t src_home_offset = *reinterpret_cast<uint32_t *>((reinterpret_cast<uint8_t *>(&cell_descs[cell_id].offsets) + offset));
    // cuComplex *src_base = reinterpret_cast<cuComplex *>(cell_descs[cell_id].home + src_home_offset);
    float *src_base = reinterpret_cast<float *>(cell_descs[cell_id].home + src_home_offset);
    // T *dst_base = group_data + cell_buf_num * cell_id + cell_section_offset;

    uint32_t cell_section_offset = section_id * section_size;
    for (uint32_t i = 0; i < section_size; i++)
    {
        uint32_t src_offset = cell_section_offset + i;
        if (src_offset < cell_buf_num)
        {
            uint32_t prgId = src_offset / data_block;
            uint32_t dst_offset = prgId * cell_num * data_block + cell_id * data_block + src_offset % data_block;
            // *(group_data + dst_offset) = *(src_base + src_offset);
            *(group_data + dst_offset * 2) = *(src_base + src_offset * 2);
            *(group_data + dst_offset * 2 + 1) = *(src_base + src_offset * 2 + 1);
        }
    }
}

#if 0
static __device__ void gpu_copy_cuComplex(cuComplex *group_data, cell_desc_t *cell_descs, uint32_t offset, uint32_t data_num, uint32_t cell_num)
{
    uint32_t cell_id = threadIdx.x % cell_num;
    uint32_t numBlocks = gridDim.x;
    uint32_t threadsPerBlock = blockDim.x;

    uint32_t cell_buf_num = data_num / cell_num;

    uint32_t section_num = numBlocks * threadsPerBlock / cell_num; // 16 * 32 / 8 = 16 * 4 = 64
    uint32_t section_id = (blockIdx.x * threadsPerBlock + threadIdx.x) / cell_num;

    uint32_t section_size = (cell_buf_num + section_num - 1) / section_num;

    // printf("%s %02u-%02u: data_num=%u cell_buf_num=%u cell_num %u-%u section %u-%u\n", __func__, blockIdx.x, threadIdx.x,
    //        data_num, cell_buf_num, cell_num, cell_id, section_num, section_id);

    uint32_t cell_section_offset = section_id * section_size;
    uint32_t src_home_offset = *reinterpret_cast<uint32_t *>((reinterpret_cast<uint8_t *>(&cell_descs[cell_id].offsets) + offset));
    cuComplex *src_base = reinterpret_cast<cuComplex *>(cell_descs[cell_id].home + src_home_offset) + cell_section_offset;
    cuComplex *dst_base = group_data + cell_buf_num * cell_id + cell_section_offset;

    for (uint32_t i = 0; i < section_size; i++)
    {
        if (cell_section_offset + i < cell_buf_num)
        {
            // *(dst_base + i) = *(src_base + i);
            cuComplex* src = src_base + i;
            cuComplex* dst = dst_base + i;
            dst->x = src->x;
            dst->y = src->y;
        }
        // else
        // {
        //     printf("%s %02u-%02u: data_num=%u cell_buf_num=%u src_home_offset=%u cell_num %u-%u section %u-%u - offset: %u+%u=%u skip\n", __func__, blockIdx.x, threadIdx.x,
        //         data_num, cell_buf_num, src_home_offset, cell_num, cell_id, section_num, section_id, cell_section_offset, i, cell_section_offset + i);
        // }
    }
}
#endif

static __global__ void gpu_copy_cell_bufs(cumac_task_info_t *task_info, cell_desc_t *cell_descs, uint32_t task_bitmask)
{
    uint32_t cell_num = task_info->grpPrms.nCell;
    // uint32_t cell_id = threadIdx.x % cell_num;
    // CHECK_CUDA_ERR(cudaMemsetAsync(grpPrms->cellAssocActUe + (group_params.nActiveUe + req.nActiveUe) * cell_id, 1, req.nActiveUe, task->strm));
    // CHECK_CUDA_ERR(cudaMemsetAsync(grpPrms->cellAssoc + (group_params.numUeSchdPerCellTTI * group_params.nCell + group_params.numUeSchdPerCellTTI) * cell_id, 1, group_params.numUeSchdPerCellTTI, task->strm));

    // struct cumac::cumacCellGrpPrms &grpPrms = task_info->grpPrms;
    // printf("%s %02u-%02u: nUe=%u nActiveUe=%u numUeSchdPerCellTTI=%u nCell=%u nPrbGrp=%u nBsAnt=%u nUeAnt=%u\n", __func__, blockIdx.x, threadIdx.x,
    //        grpPrms.nUe, grpPrms.nActiveUe, grpPrms.numUeSchdPerCellTTI, grpPrms.nCell, grpPrms.nPrbGrp, grpPrms.nBsAnt, grpPrms.nUeAnt);

    // printf("%s %02u-%02u: cell_num=%u prgMsk=%u wbSinr=%u avgRatesActUe=%u\n", __func__, blockIdx.x, threadIdx.x,
    //     cell_num, task_info->data_num.prgMsk * cell_num, task_info->data_num.wbSinr, task_info->data_num.avgRatesActUe);

    if (task_bitmask & (0x1 << CUMAC_TASK_UE_SELECTION))
    {
        gpu_memset_element<uint8_t>(task_info->grpPrms.cellAssoc, cell_descs, task_info->data_num.cellAssoc, cell_num);
        gpu_memset_element<uint8_t>(task_info->grpPrms.cellAssocActUe, cell_descs, task_info->data_num.cellAssocActUe, cell_num);
        gpu_copy_element(*task_info->grpPrms.prgMsk, cell_descs, req_offset(prgMsk), task_info->data_num.prgMsk * cell_num, cell_num);
        gpu_copy_element(task_info->grpPrms.wbSinr, cell_descs, req_offset(wbSinr), task_info->data_num.wbSinr, cell_num);
        gpu_copy_element(task_info->ueStatus.avgRatesActUe, cell_descs, req_offset(avgRatesActUe), task_info->data_num.avgRatesActUe, cell_num);
    }

    if (task_bitmask & (0x1 << CUMAC_TASK_PRB_ALLOCATION))
    {
        gpu_copy_estH_fr(task_info, cell_descs, req_offset(estH_fr), cell_num);

        gpu_copy_element(task_info->grpPrms.postEqSinr, cell_descs, req_offset(postEqSinr), task_info->data_num.postEqSinr, cell_num);
        gpu_copy_element(task_info->grpPrms.sinVal, cell_descs, req_offset(sinVal), task_info->data_num.sinVal, cell_num);
        // gpu_copy_element(task_info->grpPrms.detMat, cell_descs, req_offset(detMat), task_info->data_num.detMat, cell_num);
        // gpu_copy_element(task_info->grpPrms.prdMat, cell_descs, req_offset(prdMat), task_info->data_num.prdMat, cell_num);
        gpu_copy_element(reinterpret_cast<float*>(task_info->grpPrms.detMat), cell_descs, req_offset(detMat), task_info->data_num.detMat * 2, cell_num);
        gpu_copy_element(reinterpret_cast<float*>(task_info->grpPrms.prdMat), cell_descs, req_offset(prdMat), task_info->data_num.prdMat * 2, cell_num);
    }

    if (task_bitmask & (0x1 << CUMAC_TASK_LAYER_SELECTION))
    {
        // Nothing to do
    }

    if (task_bitmask & (0x1 << CUMAC_TASK_MCS_SELECTION))
    {
        gpu_copy_element(task_info->ueStatus.tbErrLastActUe, cell_descs, req_offset(tbErrLastActUe), task_info->data_num.tbErrLastActUe, cell_num);
    }

    if (task_bitmask & (0x1 << CUMAC_TASK_PFM_SORT))
    {
        gpu_copy_element(task_info->pfmCellInfo, cell_descs, req_offset(pfmCellInfo), task_info->data_num.pfmCellInfo, cell_num);
    }
}

int cumac_copy_cell_to_group(cumac_task *task, uint32_t cuda_block_num)
{
    gpu_copy_cell_bufs<<<cuda_block_num, N_THREAD_PER_BLOCK, 0, task->strm>>>(task->gpu_task_info, task->gpu_cell_descs, task->taskBitMask);
    cudaError lastErr = cudaGetLastError();
    if(lastErr != cudaSuccess)
    {
        NVLOGE_FMT(TAG, AERIAL_CUDA_API_EVENT, "SFN {}.{} {} line {}: {}", task->ss.u16.sfn, task->ss.u16.slot, __func__, __LINE__, cudaGetErrorString(lastErr));
    }
    return 0;
}

int get_cuda_device_id(void)
{
    int num;
    cudaError_t err = cudaGetDeviceCount (&num);
    NVLOGC_FMT(TAG, "{}: err={} num={}", __func__, +err, num);

    if (err == cudaSuccess && num > 0)
    {
        return 0;
    }
    else
    {
        return -1;
    }
}

int cumac_copy_avgRates(cumac_task* task, uint32_t cuda_block_num)
{
    if (task == nullptr)
    {
        return -1;
    }

    uint16_t *setSchdUePerCellTTI = task->schdSol.setSchdUePerCellTTI;
    float *avgRates = task->ueStatus.avgRates;
    float *avgRatesActUe = task->ueStatus.avgRatesActUe;
    uint32_t number = task->data_num.setSchdUePerCellTTI;

    const uint32_t max_active_ue = task->data_num.avgRatesActUe;
    gpu_copy_avgRates<<<cuda_block_num, N_THREAD_PER_BLOCK, 0, task->strm>>>(avgRatesActUe, avgRates, setSchdUePerCellTTI, number, max_active_ue);
    cudaError lastErr = cudaGetLastError();
    if(lastErr != cudaSuccess)
    {
        NVLOGI_FMT(TAG, "SFN {}.{} {} line {}: {}", task->ss.u16.sfn, task->ss.u16.slot, __func__, __LINE__, cudaGetErrorString(lastErr));
    }

    return 0;
}


int cumac_copy_tbErrLast(cumac_task* task, uint32_t cuda_block_num)
{
    if (task == nullptr)
    {
        return -1;
    }

    uint16_t *setSchdUePerCellTTI = task->schdSol.setSchdUePerCellTTI;
    int8_t *tbErrLast = task->ueStatus.tbErrLast;
    int8_t *tbErrLastActUe = task->ueStatus.tbErrLastActUe;
    uint32_t number = task->data_num.setSchdUePerCellTTI;

    const uint32_t max_active_ue = task->data_num.tbErrLastActUe;
    gpu_copy_tbErrLast<<<cuda_block_num, N_THREAD_PER_BLOCK, 0, task->strm>>>(tbErrLastActUe, tbErrLast, setSchdUePerCellTTI, number, max_active_ue);
    cudaError lastErr = cudaGetLastError();
    if(lastErr != cudaSuccess)
    {
        NVLOGI_FMT(TAG, "SFN {}.{} {} line {}: {}", task->ss.u16.sfn, task->ss.u16.slot, __func__, __LINE__, cudaGetErrorString(lastErr));
    }

    return 0;
}
