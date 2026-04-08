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

// #define _GNU_SOURCE
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <stdlib.h>
#include <errno.h>
#include <pthread.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdarg.h>
#include <errno.h>
#include <unistd.h>
#include <termios.h>
#include <stdatomic.h>
#include <sys/queue.h>
#include <sys/epoll.h>
#include <semaphore.h>
#include <iostream>
#include <cuda_runtime.h>
#include <memory>
#include <climits>
#include <random>
#include <cmath>

#include "nv_utils.h"
#include "nv_ipc.h"
#include "nv_ipc_utils.h"
#include "cu_mac_api.h"
#include "nv_lockfree.hpp"

/*
* YAML config path: relative path based on this app build folder
* (1) App binary path: build/cuMAC-CP/examples/L2IntegrationExample/l2_test
* (2) YAML file path: ./cuMAC-CP/examples/L2IntegrationExample/cumcp_nvipc.yaml
*/ 
#define YAML_CONFIG_PATH "./cuMAC-CP/examples/L2IntegrationExample/cumcp_nvipc.yaml"

#define MAX_PATH_LEN (1024)

#define LEN_LOCKFREE_RING_POOL (10)

// Log TAG configured in nvlog
static int TAG = (NVLOG_TAG_BASE_NVIPC + 0);

#define WORKER_THREAD_CORE (5)
#define RECV_THREAD_CORE (6)

#define CHECK_CUDA_ERR(stmt)                                                                                                                                     \
    do                                                                                                                                                           \
    {                                                                                                                                                            \
        cudaError_t result1 = (stmt);                                                                                                                            \
        if (cudaSuccess != result1)                                                                                                                              \
        {                                                                                                                                                        \
            NVLOGW(TAG, "[%s:%d] cuda failed with result1 %s", __FILE__, __LINE__, cudaGetErrorString(result1));                                            \
            cudaError_t result2 = cudaGetLastError();                                                                                                            \
            if (cudaSuccess != result2)                                                                                                                          \
            {                                                                                                                                                    \
                NVLOGW(TAG, "[%s:%d] cuda failed with result2 %s result1 %s", __FILE__, __LINE__, cudaGetErrorString(result2), cudaGetErrorString(result1)); \
                cudaError_t result3 = cudaGetLastError(); /*check for stickiness*/                                                                               \
                if (cudaSuccess != result3)                                                                                                                      \
                {                                                                                                                                                \
                    NVLOGE(TAG, AERIAL_CUDA_API_EVENT, "[%s:%d] cuda failed with result3 %s result2 %s result1 %s",                                          \
                               __FILE__,                                                                                                                         \
                               __LINE__,                                                                                                                         \
                               cudaGetErrorString(result3),                                                                                                      \
                               cudaGetErrorString(result2),                                                                                                      \
                               cudaGetErrorString(result1));                                                                                                     \
                }                                                                                                                                                \
            }                                                                                                                                                    \
        }                                                                                                                                                        \
    } while (0)

typedef struct {
    uint32_t sfn;
    uint32_t slot;
    uint32_t num_ue;
    uint32_t data_size;
    uint8_t* gpu_buf;
    nv_ipc_msg_t recv_msg;
    cudaStream_t strm;
    uint16_t num_output_sorted_lc[10];
} test_task_t;

// global variables (accessible to all threads)
nv_ipc_t* ipc = NULL;
sem_t task_sem;
nv::lock_free_ring_pool<test_task_t>* test_task_ring;

// CUDA stream for cuMAC-CP
cudaStream_t cuStrmCumacCp;

static void* cumacCp_blocking_recv_task(void* arg)
{
    // Set thread name, max string length < 16
    pthread_setname_np(pthread_self(), "cumcp_recv");
    // Set thread schedule policy to SCHED_FIFO and set priority to 80
    nv_set_sched_fifo_priority(80);
    // Set the thread CPU core
    nv_assign_thread_cpu_core(RECV_THREAD_CORE);

    nv_ipc_msg_t recv_msg;

    int num_recv_msg = 0;

    while(num_recv_msg < NUM_TIME_SLOTS)
    {
        NVLOGI(TAG, "%s: wait for incoming messages notification ...", __func__);

        // Wait for notification of incoming cuMAC scheduling message
        ipc->rx_tti_sem_wait(ipc);

        // Dequeue the incoming NVIPC message
        while(ipc->rx_recv_msg(ipc, &recv_msg) >= 0)
        {
            num_recv_msg++; // received a message, increment the counter

            cumac_pfm_tti_req_t* req = (cumac_pfm_tti_req_t*)recv_msg.msg_buf;

            NVLOGC(TAG, "RECV: SFN %u.%u msg_id=0x%02X %s msg_len=%d data_len=%d",
                    req->sfn, req->slot, recv_msg.msg_id, get_cumac_msg_name(recv_msg.msg_id), recv_msg.msg_len, recv_msg.data_len);

            test_task_t* task;
            if ((task = test_task_ring->alloc()) == nullptr) {
                NVLOGW(TAG, "RECV: SFN %u.%u: task process can't catch up with enqueue, drop slot", req->sfn, req->slot);
                continue;
            }

            task->sfn = req->sfn;
            task->slot = req->slot;
            task->num_ue = req->num_ue;
            task->data_size = recv_msg.data_len;
            task->recv_msg = recv_msg;
            memcpy(task->num_output_sorted_lc, req->num_output_sorted_lc, sizeof(uint16_t)*10);
            test_task_ring->enqueue(task);
            sem_post(&task_sem);

            // Do not release the NVIPC message buffer because data is not copied to GPU yet
            // ipc->rx_release(ipc, &recv_msg);
        }
    }

    return NULL;
}

#define dir 0 // controls direction of comparator sorts

template<typename T1, typename T2, typename T3>
inline __device__ void bitonicSort(T1* valueArr, T2* uidArr, T3* lidArr, uint16_t n)
{
    for (int size = 2; size < n; size*=2) {
        int d=dir^((threadIdx.x & (size / 2)) != 0);
       
        for (int stride = size / 2; stride > 0; stride/=2) {
           __syncthreads(); 

           if(threadIdx.x<n/2) {
              int pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));

              T1 t;
              T2 t_uid;
              T3 t_lid;

              if (((valueArr[pos] > valueArr[pos + stride]) || (valueArr[pos] == valueArr[pos + stride] && uidArr[pos] < uidArr[pos + stride])) == d) {
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
        }
    }
    
    for (int stride = n / 2; stride > 0; stride/=2) {
        __syncthreads(); 
        if(threadIdx.x<n/2) {
           int pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));

           T1 t;
           T2 t_uid;
           T3 t_lid;

           if (((valueArr[pos] > valueArr[pos + stride]) || (valueArr[pos] == valueArr[pos + stride] && uidArr[pos] < uidArr[pos + stride])) == dir) {
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
    }

    __syncthreads(); 
}

// CUDA kernel for PFM sorting
__global__ void pfmSort(uint8_t*    main_in_buf, 
                        uint8_t*    task_in_buf, 
                        uint8_t*    out_buf, 
                        uint32_t    num_Ue, 
                        uint8_t     max_num_DL_LC, 
                        uint8_t     max_num_UL_LCG,
                        uint16_t    num_lc_dl_gbr_crtc,
                        uint16_t    num_lc_dl_gbr_non_crtc,
                        uint16_t    num_lc_dl_ngbr_crtc,
                        uint16_t    num_lc_dl_ngbr_non_crtc,
                        uint16_t    num_lc_dl_mbr_non_crtc,
                        uint16_t    num_lcg_ul_gbr_crtc,
                        uint16_t    num_lcg_ul_gbr_non_crtc,
                        uint16_t    num_lcg_ul_ngbr_crtc,
                        uint16_t    num_lcg_ul_ngbr_non_crtc,
                        uint16_t    num_lcg_ul_mbr_non_crtc)
{
    __shared__ float valueShared[2048];
    __shared__ uint16_t rntiShared[2048];
    __shared__ uint16_t lidShared[2048];
    __shared__ int nUeLcFound;

    // Initialize valueShared to -1.0
    valueShared[threadIdx.x] = -1.0;
    rntiShared[threadIdx.x] = 0xFFFF;
    lidShared[threadIdx.x] = 0xFFFF;

    if (threadIdx.x == 0) {
        nUeLcFound = 0;
    }
     __syncthreads();

    if (blockIdx.x < 5) { // DL
        if (threadIdx.x < num_Ue*max_num_DL_LC) {
            uint16_t ue_idx = threadIdx.x / max_num_DL_LC;
            uint16_t lc_idx = threadIdx.x % max_num_DL_LC;
            
            uint16_t qos_type_block = blockIdx.x;

            uint32_t UE_INFO_offset = sizeof(PFM_UE_INFO)*ue_idx;
            PFM_UE_INFO* ue_data_task = (PFM_UE_INFO*)(task_in_buf + UE_INFO_offset);
            PFM_UE_INFO* ue_data_main = (PFM_UE_INFO*)(main_in_buf + UE_INFO_offset);

            uint16_t ue_valid = ue_data_task->flags & 0x01;
            uint16_t ue_scheduled_dl = ue_data_task->flags & 0x02;
            uint16_t lc_valid = ue_data_task->dl_lc_info[lc_idx].flags & 0x01;
            uint16_t qos_type_ueLc = ue_data_task->dl_lc_info[lc_idx].qos_type;
            uint16_t reset_ravg = ue_data_task->dl_lc_info[lc_idx].flags & 0x02;

            if (ue_valid && ue_scheduled_dl && lc_valid && qos_type_ueLc == qos_type_block) {
                int storeIdx = atomicAdd(&nUeLcFound, 1);
                rntiShared[storeIdx] = ue_data_main->rnti;
                lidShared[storeIdx] = lc_idx;

                float l_temp_ravg;
                if (reset_ravg == 1) {
                    l_temp_ravg = 1.0;
                    ue_data_main->dl_lc_info[lc_idx].ravg = 1;
                } else {
                    l_temp_ravg = static_cast<float>(ue_data_main->dl_lc_info[lc_idx].ravg) * (1 - IIR_ALPHA) +
                        IIR_ALPHA * static_cast<float>(ue_data_task->dl_lc_info[lc_idx].tbs_scheduled) /
                        ue_data_task->num_layers_dl / SLOT_DURATION;
                    
                    ue_data_main->dl_lc_info[lc_idx].ravg = static_cast<uint32_t>(l_temp_ravg);  
                }

                float pfm = ue_data_task->rcurrent_dl / l_temp_ravg;
                
                valueShared[storeIdx] = pfm;    
            }
        }
    } else { // UL
        if (threadIdx.x < num_Ue*max_num_UL_LCG) {
            uint32_t ue_idx = threadIdx.x / max_num_UL_LCG;
            uint32_t lcg_idx = threadIdx.x % max_num_UL_LCG;
            
            uint8_t qos_type_block = blockIdx.x - 5;

            uint32_t UE_INFO_offset = sizeof(PFM_UE_INFO)*ue_idx;
            PFM_UE_INFO* ue_data_task = (PFM_UE_INFO*)(task_in_buf + UE_INFO_offset);
            PFM_UE_INFO* ue_data_main = (PFM_UE_INFO*)(main_in_buf + UE_INFO_offset);

            uint16_t ue_valid = ue_data_task->flags & 0x01;
            uint16_t ue_scheduled_ul = ue_data_task->flags & 0x04;
            uint16_t lcg_valid = ue_data_task->ul_lcg_info[lcg_idx].flags & 0x01;
            uint16_t qos_type_ueLcg = ue_data_task->ul_lcg_info[lcg_idx].qos_type;
            uint16_t reset_ravg = ue_data_task->ul_lcg_info[lcg_idx].flags & 0x02;

            if (ue_valid && ue_scheduled_ul && lcg_valid && qos_type_ueLcg == qos_type_block) {
                int storeIdx = atomicAdd(&nUeLcFound, 1);
                rntiShared[storeIdx] = ue_data_main->rnti;
                lidShared[storeIdx] = lcg_idx;

                float l_temp_ravg;
                if (reset_ravg == 1) {
                    l_temp_ravg = 1.0;
                    ue_data_main->ul_lcg_info[lcg_idx].ravg = 1;
                } else {
                    l_temp_ravg = static_cast<float>(ue_data_main->ul_lcg_info[lcg_idx].ravg) * (1 - IIR_ALPHA) +
                        IIR_ALPHA * static_cast<float>(ue_data_task->ul_lcg_info[lcg_idx].tbs_scheduled) /
                        ue_data_task->num_layers_ul / SLOT_DURATION;

                    ue_data_main->ul_lcg_info[lcg_idx].ravg = static_cast<uint32_t>(l_temp_ravg);   
                }

                float pfm = ue_data_task->rcurrent_ul / l_temp_ravg;

                valueShared[storeIdx] = pfm;
            }
        }
    }
    __syncthreads();

    // Sort the PFM values
    uint16_t minPow2 = 2;
    while (minPow2 < nUeLcFound) {
        minPow2 *= 2;
    }
    bitonicSort<float, uint16_t, uint16_t>(valueShared, rntiShared, lidShared, minPow2);

    // Store the sorted results
    uint16_t num_lc_output = 0;
    uint32_t out_buf_offset = 0;
    if (blockIdx.x == 0) {
        num_lc_output = num_lc_dl_gbr_crtc;
        out_buf_offset = 0;
    } else if (blockIdx.x == 1) {
        num_lc_output = num_lc_dl_gbr_non_crtc;
        out_buf_offset = sizeof(PFM_DL_OUTPUT_INFO)*num_lc_dl_gbr_crtc;
    } else if (blockIdx.x == 2) {
        num_lc_output = num_lc_dl_ngbr_crtc;
        out_buf_offset = sizeof(PFM_DL_OUTPUT_INFO)*(num_lc_dl_gbr_crtc + num_lc_dl_gbr_non_crtc);
    } else if (blockIdx.x == 3) {
        num_lc_output = num_lc_dl_ngbr_non_crtc;
        out_buf_offset = sizeof(PFM_DL_OUTPUT_INFO)*(num_lc_dl_gbr_crtc + num_lc_dl_gbr_non_crtc + num_lc_dl_ngbr_crtc);
    } else if (blockIdx.x == 4) {
        num_lc_output = num_lc_dl_mbr_non_crtc;
        out_buf_offset = sizeof(PFM_DL_OUTPUT_INFO)*(num_lc_dl_gbr_crtc + num_lc_dl_gbr_non_crtc + num_lc_dl_ngbr_crtc + num_lc_dl_ngbr_non_crtc);
    } else if (blockIdx.x == 5) {
        num_lc_output = num_lcg_ul_gbr_crtc;
        out_buf_offset = sizeof(PFM_DL_OUTPUT_INFO)*(num_lc_dl_gbr_crtc + num_lc_dl_gbr_non_crtc + num_lc_dl_ngbr_crtc + num_lc_dl_ngbr_non_crtc + num_lc_dl_mbr_non_crtc);
    } else if (blockIdx.x == 6) {
        num_lc_output = num_lcg_ul_gbr_non_crtc;
        out_buf_offset = sizeof(PFM_DL_OUTPUT_INFO)*(num_lc_dl_gbr_crtc + num_lc_dl_gbr_non_crtc + num_lc_dl_ngbr_crtc + num_lc_dl_ngbr_non_crtc + num_lc_dl_mbr_non_crtc) +
            sizeof(PFM_UL_OUTPUT_INFO)*num_lcg_ul_gbr_crtc;
    } else if (blockIdx.x == 7) {
        num_lc_output = num_lcg_ul_ngbr_crtc;
        out_buf_offset = sizeof(PFM_DL_OUTPUT_INFO)*(num_lc_dl_gbr_crtc + num_lc_dl_gbr_non_crtc + num_lc_dl_ngbr_crtc + num_lc_dl_ngbr_non_crtc + num_lc_dl_mbr_non_crtc) +
            sizeof(PFM_UL_OUTPUT_INFO)*(num_lcg_ul_gbr_crtc + num_lcg_ul_gbr_non_crtc);
    } else if (blockIdx.x == 8) {
        num_lc_output = num_lcg_ul_ngbr_non_crtc;
        out_buf_offset = sizeof(PFM_DL_OUTPUT_INFO)*(num_lc_dl_gbr_crtc + num_lc_dl_gbr_non_crtc + num_lc_dl_ngbr_crtc + num_lc_dl_ngbr_non_crtc + num_lc_dl_mbr_non_crtc) +
            sizeof(PFM_UL_OUTPUT_INFO)*(num_lcg_ul_gbr_crtc + num_lcg_ul_gbr_non_crtc + num_lcg_ul_ngbr_crtc);
    } else if (blockIdx.x == 9) {
        num_lc_output = num_lcg_ul_mbr_non_crtc;
        out_buf_offset = sizeof(PFM_DL_OUTPUT_INFO)*(num_lc_dl_gbr_crtc + num_lc_dl_gbr_non_crtc + num_lc_dl_ngbr_crtc + num_lc_dl_ngbr_non_crtc + num_lc_dl_mbr_non_crtc) +
            sizeof(PFM_UL_OUTPUT_INFO)*(num_lcg_ul_gbr_crtc + num_lcg_ul_gbr_non_crtc + num_lcg_ul_ngbr_crtc + num_lcg_ul_ngbr_non_crtc);
    }

    if (threadIdx.x < num_lc_output) {
        if (blockIdx.x < 5) { // DL
            PFM_DL_OUTPUT_INFO* dl_info = (PFM_DL_OUTPUT_INFO*)(out_buf + out_buf_offset);

            dl_info->rnti = rntiShared[threadIdx.x];
            dl_info->lc_id = lidShared[threadIdx.x];
        } else { // UL
            PFM_UL_OUTPUT_INFO* ul_info = (PFM_UL_OUTPUT_INFO*)(out_buf + out_buf_offset);

            ul_info->rnti = rntiShared[threadIdx.x];
            ul_info->lcg_id = lidShared[threadIdx.x];
        }
    }
}

void alloc_mem_test_task(test_task_t *task, cudaStream_t strm)
{
    CHECK_CUDA_ERR(cudaMalloc(&task->gpu_buf, sizeof(PFM_UE_INFO) * MAX_NUM_UE));
    task->strm = strm;
}

int main(int argc, char** argv)
{
    // main cuMAC-CP worker thread

    NVLOGC(TAG, "YAML file: %s", YAML_CONFIG_PATH);

    // Load nvipc configuration from YAML file
    nv_ipc_config_t config;
    load_nv_ipc_yaml_config(&config, YAML_CONFIG_PATH, NV_IPC_MODULE_PRIMARY);

    // Initialize NVIPC interface and connect to the cuMAC-CP
    if ((ipc = create_nv_ipc_interface(&config)) == NULL)
    {
        NVLOGE(TAG, AERIAL_NVIPC_API_EVENT, "%s: create IPC interface failed", __func__);
        return -1;
    }

    // Initialize semaphore for GPU task finish notification
    sem_init(&task_sem, 0, 0);

    // Initialize CUDA stream for cuMAC-CP
    CHECK_CUDA_ERR(cudaStreamCreate(&cuStrmCumacCp));

    // Configure PFM sorting kernel
    int num_blocks = 10; // 5 blocks for DL and 5 blocks for UL
    int num_threads = 1024;

    // Initialize GPU main buffer to store PFM_UE_INFO data
    uint8_t* gpu_main_in_buf;
    uint8_t* gpu_out_buf;
    CHECK_CUDA_ERR(cudaMalloc(&gpu_main_in_buf, sizeof(PFM_UE_INFO) * MAX_NUM_UE));
    CHECK_CUDA_ERR(cudaMalloc(&gpu_out_buf, (sizeof(PFM_DL_OUTPUT_INFO) + sizeof(PFM_UL_OUTPUT_INFO))*MAX_NUM_OUTPUT_SORTED_LC_PER_QOS*5));

    // Initialize lock-free ring pool for tasks
    test_task_ring = new nv::lock_free_ring_pool<test_task_t>("test_task", LEN_LOCKFREE_RING_POOL, sizeof(test_task_t));
    uint32_t ring_len = test_task_ring->get_ring_len();
    for (int i = 0; i < ring_len; i++)
    {
        test_task_t *task = test_task_ring->get_buf_addr(i);
        if (task == nullptr)
        {
            NVLOGE(TAG, AERIAL_CUMAC_CP_EVENT, "Error cumac_task ring length: i=%d length=%d", i, ring_len);
            return -1;
        }

        alloc_mem_test_task(task, cuStrmCumacCp);
    }

    uint8_t max_num_DL_LC = MAX_NUM_LC;
    uint8_t max_num_UL_LCG = MAX_NUM_LCG;

    // Create cuMAC-CP receiver thread
    pthread_t recv_thread_id;
    int ret = pthread_create(&recv_thread_id, NULL, cumacCp_blocking_recv_task, NULL);
    if(ret != 0)
    {
        NVLOGE(TAG, AERIAL_NVIPC_API_EVENT, "%s failed, ret=%d", __func__, ret);
        return -1;
    }

    // Set thread name, max string length < 16
    pthread_setname_np(pthread_self(), "cumcp_worker");
    // Set thread schedule policy to SCHED_FIFO and set priority to 80
    nv_set_sched_fifo_priority(80);
    // Set the thread CPU core
    nv_assign_thread_cpu_core(WORKER_THREAD_CORE);

    // Worker thread to process GPU tasks and send response to L2
    int task_count = 0;
    while(task_count < NUM_TIME_SLOTS) {
        // Wait for GPU processing finish notification by semaphore task_sem
        sem_wait(&task_sem);

        test_task_t* task = test_task_ring->dequeue();
        if (task == nullptr)
        {
            // No task to process
            NVLOGI(TAG, "%s: no task to process", __func__);
            continue;
        }

        // Create CUDA events
        cudaEvent_t startCopyH2D, stopCopyH2D;
        cudaEvent_t startKernel, stopKernel;
        cudaEvent_t startCopyD2H, stopCopyD2H;
        CHECK_CUDA_ERR(cudaEventCreate(&startCopyH2D));
        CHECK_CUDA_ERR(cudaEventCreate(&stopCopyH2D));
        CHECK_CUDA_ERR(cudaEventCreate(&startKernel));
        CHECK_CUDA_ERR(cudaEventCreate(&stopKernel));
        CHECK_CUDA_ERR(cudaEventCreate(&startCopyD2H));
        CHECK_CUDA_ERR(cudaEventCreate(&stopCopyD2H));

        struct timespec slot_start, slot_end;
        clock_gettime(CLOCK_REALTIME, &slot_start);

        // prepare response message and allocate NVIPC buffer
        // Alloc NVIPC buffer: msg_buf will be allocated by default. Set data_pool to get data_buf
        nv_ipc_msg_t send_msg;
        send_msg.data_pool = NV_IPC_MEMPOOL_CPU_DATA;

        // Allocate NVIPC buffer which contains MSG part and DATA part
        if(ipc->tx_allocate(ipc, &send_msg, 0) != 0)
        {
            NVLOGE(TAG, AERIAL_NVIPC_API_EVENT, "%s error: NVIPC memory pool is full", __func__);
            break;
        }

        // MSG part
        cumac_pfm_tti_resp_t* resp = (cumac_pfm_tti_resp_t*)send_msg.msg_buf;
        uint8_t* data_buf = (uint8_t*)send_msg.data_buf;

        resp->header.type_id = CUMAC_SCH_TTI_RESPONSE;
        resp->sfn = task->sfn;
        resp->slot = task->slot;
        memcpy(resp->num_output_sorted_lc, task->num_output_sorted_lc, sizeof(uint16_t)*10);

        uint32_t data_size = 0;
        for (int i = 0; i < 5; i++) {
            resp->offset_output_sorted_lc[i] = data_size;
            data_size += sizeof(PFM_DL_OUTPUT_INFO)*task->num_output_sorted_lc[i];
        }
        for (int i = 5; i < 10; i++) {
            resp->offset_output_sorted_lc[i] = data_size;
            data_size += sizeof(PFM_UL_OUTPUT_INFO)*task->num_output_sorted_lc[i];
        }

        // Update the msg_len and data_len of the NVIPC message header
        send_msg.msg_id = CUMAC_SCH_TTI_RESPONSE;
        send_msg.cell_id = 0;
        send_msg.msg_len = sizeof(cumac_pfm_tti_resp_t);
        send_msg.data_len = data_size;

        clock_gettime(CLOCK_REALTIME, &slot_end);
        int64_t slot_duration = nvlog_timespec_interval(&slot_start, &slot_end);
        NVLOGI(TAG, "cuMAC-CP Slot preparation duration: %ld ns", slot_duration);

        // Copy data from NVIPC buffer to GPU buffer
        CHECK_CUDA_ERR(cudaEventRecord(startCopyH2D));
        for (int rIdx = 0; rIdx < 100; rIdx++) {
            CHECK_CUDA_ERR(cudaMemcpyAsync(task->gpu_buf, task->recv_msg.data_buf, task->recv_msg.data_len, cudaMemcpyHostToDevice, task->strm));
        }
        CHECK_CUDA_ERR(cudaEventRecord(stopCopyH2D));
        CHECK_CUDA_ERR(cudaEventSynchronize(stopCopyH2D));
    
        // Launch PFM sorting kernel
        CHECK_CUDA_ERR(cudaEventRecord(startKernel));
        for (int rIdx = 0; rIdx < 100; rIdx++) {
            pfmSort<<<num_blocks, num_threads, 0, task->strm>>>(gpu_main_in_buf, 
                                                                task->gpu_buf, 
                                                                gpu_out_buf, 
                                                                task->num_ue, 
                                                                max_num_DL_LC, 
                                                                max_num_UL_LCG, 
                                                                task->num_output_sorted_lc[0],
                                                                task->num_output_sorted_lc[1],
                                                                task->num_output_sorted_lc[2],
                                                                task->num_output_sorted_lc[3],
                                                                task->num_output_sorted_lc[4],
                                                                task->num_output_sorted_lc[5],
                                                                task->num_output_sorted_lc[6],
                                                                task->num_output_sorted_lc[7],
                                                                task->num_output_sorted_lc[8],
                                                                task->num_output_sorted_lc[9]);
            CHECK_CUDA_ERR(cudaGetLastError());
        }
        CHECK_CUDA_ERR(cudaEventRecord(stopKernel));
        CHECK_CUDA_ERR(cudaEventSynchronize(stopKernel));

        CHECK_CUDA_ERR(cudaEventRecord(startCopyD2H));
        for (int rIdx = 0; rIdx < 100; rIdx++) {
            CHECK_CUDA_ERR(cudaMemcpyAsync(data_buf, gpu_out_buf, data_size, cudaMemcpyDeviceToHost, task->strm));
        }
        CHECK_CUDA_ERR(cudaEventRecord(stopCopyD2H));
        CHECK_CUDA_ERR(cudaEventSynchronize(stopCopyD2H));

        CHECK_CUDA_ERR(cudaStreamSynchronize(task->strm));

        // calculate timings
        float timeH2D, timeKernel, timeD2H;

        CHECK_CUDA_ERR(cudaEventElapsedTime(&timeH2D, startCopyH2D, stopCopyH2D));
        CHECK_CUDA_ERR(cudaEventElapsedTime(&timeKernel, startKernel, stopKernel));
        CHECK_CUDA_ERR(cudaEventElapsedTime(&timeD2H, startCopyD2H, stopCopyD2H));

        printf("Host to Device copy time: %f microseconds\n", timeH2D*10.0);
        printf("Kernel execution time:     %f microseconds\n", timeKernel*10.0);
        printf("Device to Host copy time: %f microseconds\n", timeD2H*10.0);

        // Destroy CUDA events
        CHECK_CUDA_ERR(cudaEventDestroy(startCopyH2D));
        CHECK_CUDA_ERR(cudaEventDestroy(stopCopyH2D));
        CHECK_CUDA_ERR(cudaEventDestroy(startKernel));
        CHECK_CUDA_ERR(cudaEventDestroy(stopKernel));
        CHECK_CUDA_ERR(cudaEventDestroy(startCopyD2H));
        CHECK_CUDA_ERR(cudaEventDestroy(stopCopyD2H));

        // todo: validate the results

        // Once finished, send cuMAC schedule result to L2
        // Send the message
        NVLOGC(TAG, "SEND: SFN %u.%u msg_id=0x%02X %s msg_len=%d data_len=%d",
                resp->sfn, resp->slot, send_msg.msg_id, get_cumac_msg_name(send_msg.msg_id), send_msg.msg_len, send_msg.data_len);
        if(ipc->tx_send_msg(ipc, &send_msg) < 0)
        {
            NVLOGE(TAG, AERIAL_NVIPC_API_EVENT, "%s error: send message failed", __func__);
            break;
        }

        if(ipc->tx_tti_sem_post(ipc) < 0)
        {
            NVLOGE(TAG, AERIAL_NVIPC_API_EVENT, "%s error: tx notification failed", __func__);
            break;
        }

        // Release the NVIPC message buffer
        ipc->rx_release(ipc, &task->recv_msg);

        // Task is finished, free the task buffer
        test_task_ring->free(task);
        task_count++;
    }

    CHECK_CUDA_ERR(cudaFree(gpu_main_in_buf));
    CHECK_CUDA_ERR(cudaFree(gpu_out_buf));
    for (int i = 0; i < ring_len; i++) {
        test_task_t *task = test_task_ring->get_buf_addr(i);
        if (task == nullptr)
        {
            NVLOGE(TAG, AERIAL_CUMAC_CP_EVENT, "Error cumac_task ring length: i=%d length=%d", i, ring_len);
            continue;
        }
        CHECK_CUDA_ERR(cudaFree(task->gpu_buf));
    }

    int exit_code = 0;
    if(recv_thread_id != 0)
    {
        // Cancel the receiver thread
        NVLOGI(TAG, "%s: canceling receiver thread", __func__);
        if(pthread_cancel(recv_thread_id) != 0)
        {
            NVLOGE(TAG, AERIAL_NVIPC_API_EVENT, "%s pthread_cancel failed, stderr=%s", __func__, strerror(errno));
        }

        // Wait for the receiver thread to finish
        NVLOGI(TAG, "%s: waiting for receiver thread to exit", __func__);
        int ret = pthread_join(recv_thread_id, NULL);
        if(ret != 0)
        {
            NVLOGE(TAG, AERIAL_NVIPC_API_EVENT, "%s pthread_join failed, stderr=%s", __func__, strerror(ret));
            exit_code = -1;
        }
    }

    // Synchronize CUDA device before exiting
    CHECK_CUDA_ERR(cudaDeviceSynchronize());

    return exit_code;
}