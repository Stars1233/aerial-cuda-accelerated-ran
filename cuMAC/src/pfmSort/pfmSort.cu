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

#include "pfmSort.cuh"

// #define SCHEDULER_KERNEL_TIME_MEASURE_ 
#ifdef SCHEDULER_KERNEL_TIME_MEASURE_
constexpr uint16_t numRunSchKnlTimeMsr = 1000;
#endif

// cuMAC namespace
namespace cumac {
    pfmSort::pfmSort()
    {
        pCpuDynDesc = std::make_unique<pfmSortDynDescr>();

        pLaunchCfg = std::make_unique<launchCfg_t>();

        CUDA_CHECK_ERR(cudaMalloc((void **)&pGpuDynDesc, sizeof(pfmSortDynDescr)));
        CUDA_CHECK_ERR(cudaMalloc((void **)&gpu_main_in_buf, sizeof(PFM_CELL_INFO_MANAGE)*CUMAC_PFM_MAX_NUM_CELL));
        CUDA_CHECK_ERR(cudaMalloc((void **)&gpu_out_buf, sizeof(cumac_pfm_output_cell_info_t)*CUMAC_PFM_MAX_NUM_CELL));

        pCpuDynDesc->gpu_main_in_buf = gpu_main_in_buf;
        pCpuDynDesc->gpu_out_buf = gpu_out_buf;
    }

    pfmSort::~pfmSort()
    {
        CUDA_CHECK_ERR(cudaFree(pGpuDynDesc));
        CUDA_CHECK_ERR(cudaFree(gpu_main_in_buf));
        CUDA_CHECK_ERR(cudaFree(gpu_out_buf));
    }

    void pfmSort::kernelSelect()
    {
        void* kernelFunc = reinterpret_cast<void*>(pfmSortKernel);

        CUDA_CHECK_ERR(cudaGetFuncBySymbol(&pLaunchCfg->kernelNodeParamsDriver.func, kernelFunc));
  
        // launch geometry
        gridDim  = {numThrdBlk, 1, 1};
        blockDim = {numThrdPerBlk, 1, 1};
  
        // populate kernel parameters
        CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriver   = pLaunchCfg->kernelNodeParamsDriver;
  
        kernelNodeParamsDriver.blockDimX                  = blockDim.x;
        kernelNodeParamsDriver.blockDimY                  = blockDim.y;
        kernelNodeParamsDriver.blockDimZ                  = blockDim.z;
  
        kernelNodeParamsDriver.gridDimX                   = gridDim.x;
        kernelNodeParamsDriver.gridDimY                   = gridDim.y;
        kernelNodeParamsDriver.gridDimZ                   = gridDim.z;
  
        kernelNodeParamsDriver.extra                      = nullptr;
        kernelNodeParamsDriver.sharedMemBytes             = 0;  
    }

    void pfmSort::setup(pfmSortTask* task)
    {
        pCpuDynDesc->task_in_buf = task->gpu_buf;

        strm = task->strm;

        num_cell = task->num_cell;

        numThrdPerBlk   = 1024;
        numThrdBlk      = num_cell*(CUMAC_PFM_NUM_QOS_TYPES_DL+CUMAC_PFM_NUM_QOS_TYPES_UL);

        CUDA_CHECK_ERR(cudaMemcpyAsync(pGpuDynDesc, pCpuDynDesc.get(), sizeof(pfmSortDynDescr), cudaMemcpyHostToDevice, strm));

        // select kernel 
        kernelSelect();
 
        pLaunchCfg->kernelArgs[0]                       = &pGpuDynDesc;
        pLaunchCfg->kernelNodeParamsDriver.kernelParams = &(pLaunchCfg->kernelArgs[0]);
    }

    void pfmSort::run(uint8_t* hSolAddr)
    {
        const CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriver = pLaunchCfg->kernelNodeParamsDriver;

        #ifdef SCHEDULER_KERNEL_TIME_MEASURE_
        cudaEvent_t start, stop;
        CUDA_CHECK_ERR(cudaEventCreate(&start));
        CUDA_CHECK_ERR(cudaEventCreate(&stop));
        float milliseconds = 0;
        CUDA_CHECK_ERR(cudaEventRecord(start));
        for (int exeIdx = 0; exeIdx < numRunSchKnlTimeMsr; exeIdx++) {
        #endif

        CUDA_CHECK_RES(cuLaunchKernel(kernelNodeParamsDriver.func,
                                      kernelNodeParamsDriver.gridDimX,
                                      kernelNodeParamsDriver.gridDimY, 
                                      kernelNodeParamsDriver.gridDimZ,
                                      kernelNodeParamsDriver.blockDimX, 
                                      kernelNodeParamsDriver.blockDimY, 
                                      kernelNodeParamsDriver.blockDimZ,
                                      kernelNodeParamsDriver.sharedMemBytes,
                                      strm,
                                      kernelNodeParamsDriver.kernelParams,
                                      kernelNodeParamsDriver.extra));   
        #ifdef SCHEDULER_KERNEL_TIME_MEASURE_
        }
        CUDA_CHECK_ERR(cudaEventRecord(stop));
        CUDA_CHECK_ERR(cudaEventSynchronize(stop));
        CUDA_CHECK_ERR(cudaEventElapsedTime(&milliseconds, start, stop));
        printf("PFM sorting CUDA kernel execution time = %f ms\n", milliseconds/static_cast<float>(numRunSchKnlTimeMsr));
        #endif 

        CUDA_CHECK_ERR(cudaMemcpyAsync(hSolAddr, gpu_out_buf, sizeof(cumac_pfm_output_cell_info_t)*num_cell, cudaMemcpyDeviceToHost, strm));
    }

    constexpr uint32_t dir = 0; // controls direction of comparator in sorting

    template<typename T1, typename T2, typename T3>
    inline __device__ void pfmBitonicSort(T1* valueArr, T2* uidArr, T3* lidArr, uint32_t n)
    {
        for (uint32_t size = 2; size < n; size *= 2) {
            uint32_t d = dir ^ ((threadIdx.x & (size / 2)) != 0);
       
            for (uint32_t stride = size / 2; stride > 0; stride /= 2) {
                if(threadIdx.x < n/2) {
                    uint32_t pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));

                    T1 t;
                    T2 t_uid;
                    T3 t_lid;

                    if (((valueArr[pos] > valueArr[pos + stride]) || 
                         (valueArr[pos] == valueArr[pos + stride] && uidArr[pos] < uidArr[pos + stride]) ||
                         (valueArr[pos] == valueArr[pos + stride] && uidArr[pos] == uidArr[pos + stride] && lidArr[pos] < lidArr[pos + stride])) == d) {
                        t = valueArr[pos];
                        valueArr[pos] = valueArr[pos + stride];
                        valueArr[pos + stride] = t;
                        t_uid = uidArr[pos];
                        uidArr[pos] = uidArr[pos + stride];
                        uidArr[pos + stride] = t_uid;
                        t_lid = lidArr[pos];
                        lidArr[pos] = lidArr[pos + stride];
                        lidArr[pos + stride] = t_lid;
                    }
                }
                __syncthreads(); 
            }
        }
    
        for (uint32_t stride = n / 2; stride > 0; stride /= 2) {
            if(threadIdx.x < n/2) {
                uint32_t pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));

                T1 t;
                T2 t_uid;
                T3 t_lid;

                if (((valueArr[pos] > valueArr[pos + stride]) || 
                     (valueArr[pos] == valueArr[pos + stride] && uidArr[pos] < uidArr[pos + stride]) ||
                     (valueArr[pos] == valueArr[pos + stride] && uidArr[pos] == uidArr[pos + stride] && lidArr[pos] < lidArr[pos + stride])) == dir) {
                    t = valueArr[pos];
                    valueArr[pos] = valueArr[pos + stride];
                    valueArr[pos + stride] = t;
             
                    t_uid = uidArr[pos];
                    uidArr[pos] = uidArr[pos + stride];
                    uidArr[pos + stride] = t_uid;
                    t_lid = lidArr[pos];
                    lidArr[pos] = lidArr[pos + stride];
                    lidArr[pos + stride] = t_lid;
                }
            }
            __syncthreads(); 
        }
    }

    // CUDA kernel for PFM sorting
    __global__ void pfmSortKernel(pfmSortDynDescr* pDynDescr)
    {
        uint32_t cIdx = blockIdx.x/(CUMAC_PFM_NUM_QOS_TYPES_DL+CUMAC_PFM_NUM_QOS_TYPES_UL);
        uint32_t qosType = blockIdx.x - cIdx*(CUMAC_PFM_NUM_QOS_TYPES_DL+CUMAC_PFM_NUM_QOS_TYPES_UL);

        __shared__ float valueShared[2048];
        __shared__ uint32_t rntiShared[2048];
        __shared__ uint32_t lidShared[2048];
        __shared__ uint32_t nUeLcFound;

        // Initialize valueShared to -1.0
        for (uint32_t idx = threadIdx.x; idx < 2048; idx += blockDim.x) {
            valueShared[idx] = -1.0;
            rntiShared[idx] = CUMAC_INVALID_RNTI;
            lidShared[idx] = 0xFFFF;
        }

        if (threadIdx.x == 0) {
            nUeLcFound = 0;
        }
        __syncthreads();

        cumac_pfm_cell_info_t* cell_data_task = (cumac_pfm_cell_info_t*)(pDynDescr->task_in_buf + sizeof(cumac_pfm_cell_info_t)*cIdx);

        PFM_CELL_INFO_MANAGE* cell_data_main = (PFM_CELL_INFO_MANAGE*)(pDynDescr->gpu_main_in_buf + sizeof(PFM_CELL_INFO_MANAGE)*cIdx);

        if (qosType < 5) { // DL
            for (uint32_t idx = threadIdx.x; idx < cell_data_task->num_ue*CUMAC_PFM_MAX_NUM_LC_PER_UE; idx += blockDim.x) {
                uint32_t ue_idx = idx / CUMAC_PFM_MAX_NUM_LC_PER_UE;
                uint32_t lc_idx = idx - ue_idx*CUMAC_PFM_MAX_NUM_LC_PER_UE;

                uint32_t id_in_main_buff = cell_data_task->ue_info[ue_idx].id;

                if (((cell_data_task->ue_info[ue_idx].flags & 0x01) > 0) && 
                    ((cell_data_task->ue_info[ue_idx].dl_lc_info[lc_idx].flags & 0x01) > 0) && 
                    (cell_data_task->ue_info[ue_idx].dl_lc_info[lc_idx].qos_type == qosType)) {
                    
                    uint32_t storeIdx = atomicAdd(&nUeLcFound, 1);
                    rntiShared[storeIdx] = cell_data_task->ue_info[ue_idx].rnti;
                    lidShared[storeIdx] = lc_idx;

                    float l_temp_ravg;
                    if ((cell_data_task->ue_info[ue_idx].dl_lc_info[lc_idx].flags & 0x02) > 0) {
                        l_temp_ravg = 1.0;
                    } else {
                        l_temp_ravg = static_cast<float>(cell_data_main->ue_info_manage[id_in_main_buff].ravg_dl_lc[lc_idx]) * (1 - CUMAC_PFM_IIR_ALPHA) +
                                        CUMAC_PFM_IIR_ALPHA * static_cast<float>(cell_data_task->ue_info[ue_idx].dl_lc_info[lc_idx].tbs_scheduled) /
                                        cell_data_task->ue_info[ue_idx].num_layers_dl / CUMAC_PFM_SLOT_DURATION;
                    }

                    cell_data_main->ue_info_manage[id_in_main_buff].ravg_dl_lc[lc_idx] = static_cast<uint32_t>(l_temp_ravg);  
                        
                    valueShared[storeIdx] = cell_data_task->ue_info[ue_idx].rcurrent_dl / l_temp_ravg;    
                }
            }
        } else { // UL
            for (uint32_t idx = threadIdx.x; idx < cell_data_task->num_ue*CUMAC_PFM_MAX_NUM_LCG_PER_UE; idx += blockDim.x) {
                uint32_t ue_idx = idx / CUMAC_PFM_MAX_NUM_LCG_PER_UE;
                uint32_t lcg_idx = idx - ue_idx*CUMAC_PFM_MAX_NUM_LCG_PER_UE;

                uint32_t id_in_main_buff = cell_data_task->ue_info[ue_idx].id;

                if (((cell_data_task->ue_info[ue_idx].flags & 0x02) > 0) && 
                    ((cell_data_task->ue_info[ue_idx].ul_lcg_info[lcg_idx].flags & 0x01) > 0) && 
                    cell_data_task->ue_info[ue_idx].ul_lcg_info[lcg_idx].qos_type == (qosType - 5)) {
                    
                    uint32_t storeIdx = atomicAdd(&nUeLcFound, 1);
                    rntiShared[storeIdx] = cell_data_task->ue_info[ue_idx].rnti;
                    lidShared[storeIdx] = lcg_idx;

                    float l_temp_ravg;
                    if ((cell_data_task->ue_info[ue_idx].ul_lcg_info[lcg_idx].flags & 0x02) > 0) {
                        l_temp_ravg = 1.0;
                    } else {
                        l_temp_ravg = static_cast<float>(cell_data_main->ue_info_manage[id_in_main_buff].ravg_ul_lcg[lcg_idx]) * (1 - CUMAC_PFM_IIR_ALPHA) +
                                        CUMAC_PFM_IIR_ALPHA * static_cast<float>(cell_data_task->ue_info[ue_idx].ul_lcg_info[lcg_idx].tbs_scheduled) /
                                        cell_data_task->ue_info[ue_idx].num_layers_ul / CUMAC_PFM_SLOT_DURATION;
                    }
                    cell_data_main->ue_info_manage[id_in_main_buff].ravg_ul_lcg[lcg_idx] = static_cast<uint32_t>(l_temp_ravg);   

                    valueShared[storeIdx] = cell_data_task->ue_info[ue_idx].rcurrent_ul / l_temp_ravg;
                }
            }
        }
        __syncthreads();

        // Sort the PFM values
        uint32_t minPow2 = 2;
        while (minPow2 < nUeLcFound) {
            minPow2 *= 2;
        }
        pfmBitonicSort<float, uint32_t, uint32_t>(valueShared, rntiShared, lidShared, minPow2);

        // Store the sorted results
        cumac_pfm_output_cell_info_t* cell_out = (cumac_pfm_output_cell_info_t*)(pDynDescr->gpu_out_buf + sizeof(cumac_pfm_output_cell_info_t)*cIdx);

        switch (qosType) {
            case 0:
                for (uint32_t idx = threadIdx.x; idx < cell_data_task->num_output_sorted_lc[0]; idx += blockDim.x) {
                    cell_out->dl_gbr_critical[idx].rnti = rntiShared[idx];
                    cell_out->dl_gbr_critical[idx].lc_id = lidShared[idx];
                }
                break;
            case 1:
                for (uint32_t idx = threadIdx.x; idx < cell_data_task->num_output_sorted_lc[1]; idx += blockDim.x) {
                    cell_out->dl_gbr_non_critical[idx].rnti = rntiShared[idx];
                    cell_out->dl_gbr_non_critical[idx].lc_id = lidShared[idx];
                }
                break;
            case 2:
                for (uint32_t idx = threadIdx.x; idx < cell_data_task->num_output_sorted_lc[2]; idx += blockDim.x) {
                    cell_out->dl_ngbr_critical[idx].rnti = rntiShared[idx];
                    cell_out->dl_ngbr_critical[idx].lc_id = lidShared[idx];
                }
                break;
            case 3:
                for (uint32_t idx = threadIdx.x; idx < cell_data_task->num_output_sorted_lc[3]; idx += blockDim.x) {
                    cell_out->dl_ngbr_non_critical[idx].rnti = rntiShared[idx];
                    cell_out->dl_ngbr_non_critical[idx].lc_id = lidShared[idx];
                }
                break;
            case 4:
                for (uint32_t idx = threadIdx.x; idx < cell_data_task->num_output_sorted_lc[4]; idx += blockDim.x) {
                    cell_out->dl_mbr_non_critical[idx].rnti = rntiShared[idx];
                    cell_out->dl_mbr_non_critical[idx].lc_id = lidShared[idx];

                }
                break;
            case 5:
                for (uint32_t idx = threadIdx.x; idx < cell_data_task->num_output_sorted_lc[5]; idx += blockDim.x) {
                    cell_out->ul_gbr_critical[idx].rnti = rntiShared[idx];
                    cell_out->ul_gbr_critical[idx].lcg_id = lidShared[idx];

                }
                break;
            case 6:
                for (uint32_t idx = threadIdx.x; idx < cell_data_task->num_output_sorted_lc[6]; idx += blockDim.x) {
                    cell_out->ul_gbr_non_critical[idx].rnti = rntiShared[idx];
                    cell_out->ul_gbr_non_critical[idx].lcg_id = lidShared[idx];
                }
                break;
            case 7:
                for (uint32_t idx = threadIdx.x; idx < cell_data_task->num_output_sorted_lc[7]; idx += blockDim.x) {
                    cell_out->ul_ngbr_critical[idx].rnti = rntiShared[idx];
                    cell_out->ul_ngbr_critical[idx].lcg_id = lidShared[idx];
                }
                break;
            case 8:
                for (uint32_t idx = threadIdx.x; idx < cell_data_task->num_output_sorted_lc[8]; idx += blockDim.x) {
                    cell_out->ul_ngbr_non_critical[idx].rnti = rntiShared[idx];
                    cell_out->ul_ngbr_non_critical[idx].lcg_id = lidShared[idx];
                }
                break;
            case 9:
                for (uint32_t idx = threadIdx.x; idx < cell_data_task->num_output_sorted_lc[9]; idx += blockDim.x) {
                    cell_out->ul_mbr_non_critical[idx].rnti = rntiShared[idx];
                    cell_out->ul_mbr_non_critical[idx].lcg_id = lidShared[idx];
                }
                break;
        }
    }
}