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

#include <iostream>
#include <cuda_runtime.h>
#include <memory>
#include <climits>
#include <iostream>
#include <random>
#include <cmath>

/**
 * CUDA error checking macro.
 *
 * Prints file, line, and error string to stderr on failure.
 */
#define CHECK_CUDA_ERR(stmt)                                                                  \
    do                                                                                        \
    {                                                                                         \
        cudaError_t result = (stmt);                                                          \
        if (cudaSuccess != result)                                                            \
        {                                                                                     \
            fprintf(stderr, "[%s:%d] CUDA error: %s\n", __FILE__, __LINE__,                   \
                    cudaGetErrorString(result));                                              \
        }                                                                                     \
    } while (0)

struct inputStruct {
    float* pfm = nullptr;
    float* gbr_limit = nullptr;
    float* mbr_limit = nullptr;
    uint32_t* last_slot_tbs_scheduled = nullptr;
    uint16_t* ue_id = nullptr;
    uint8_t* qos_type = nullptr; // 0 - GBR_CRITICAL , 1 - NGBR_CRITICAL , 2 - GBR_NON_CRITCAL , 3 - NGBR_NON_CRITICAL
    uint8_t* lc_id = nullptr;
    uint8_t* qos_5qi = nullptr;
    uint8_t* priority = nullptr;
    uint8_t* cqiI = nullptr;
};

#define dir 0 // controls direction of comparator sorts

inline __device__ void bitonicSort(float* valueArr, uint16_t* idArr, uint16_t n)
{
    for (int size = 2; size < n; size*=2) {
        int d=dir^((threadIdx.x & (size / 2)) != 0);
       
        for (int stride = size / 2; stride > 0; stride/=2) {
           __syncthreads(); 

           if(threadIdx.x<n/2) {
              int pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));

              float t;
              uint16_t t_id;

              if (((valueArr[pos] > valueArr[pos + stride]) || (valueArr[pos] == valueArr[pos + stride] && idArr[pos] < idArr[pos + stride])) == d) {
                  t = valueArr[pos];
                  valueArr[pos] = valueArr[pos + stride];
                  valueArr[pos + stride] = t;
                  t_id = idArr[pos];
                  idArr[pos] = idArr[pos + stride];
                  idArr[pos + stride] = t_id;
              }
           }
        }
    }
    
    for (int stride = n / 2; stride > 0; stride/=2) {
        __syncthreads(); 
        if(threadIdx.x<n/2) {
           int pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));

           float t;
           uint16_t t_id;

           if (((valueArr[pos] > valueArr[pos + stride]) || (valueArr[pos] == valueArr[pos + stride] && idArr[pos] < idArr[pos + stride])) == dir) {
               t = valueArr[pos];
               valueArr[pos] = valueArr[pos + stride];
               valueArr[pos + stride] = t;
             
               t_id = idArr[pos];
               idArr[pos] = idArr[pos + stride];
               idArr[pos + stride] = t_id;
           }
        }
    }

    __syncthreads(); 
}
 
// CUDA kernel for PFM sorting
__global__ void pfmSort(uint8_t* dataBuffer, int numUe) 
{
    __shared__ uint16_t ueIdsShared[1024];
    __shared__ float valueShared[1024];
    __shared__ uint8_t dataBufferShared[23552];

    ueIdsShared[threadIdx.x] = 0xFFFF;
    valueShared[threadIdx.x] = -1.0;

    for (int idx = threadIdx.x; idx < numUe*23; idx += blockDim.x) {
        dataBufferShared[idx] = dataBuffer[idx];
    }

    if (threadIdx.x < numUe) {
        uint8_t* qos_type = (uint8_t*) (dataBufferShared + numUe*18);
        uint8_t qos = qos_type[threadIdx.x];
        
        uint16_t* ue_id = (uint16_t*) (dataBufferShared + numUe*16);
        ueIdsShared[threadIdx.x] = ue_id[threadIdx.x];
        
        float* pfmPtr = (float*) (dataBufferShared);
        float pfm = pfmPtr[threadIdx.x];
        
        uint8_t* priority = (uint8_t*) (dataBufferShared + numUe*21);
        uint8_t prio = priority[threadIdx.x];
        
        uint32_t* last_slot_tbs_scheduled = (uint32_t*) (dataBufferShared + numUe*12);
        uint32_t last_slot = last_slot_tbs_scheduled[threadIdx.x];
        
        uint8_t* cqiI = (uint8_t*) (dataBufferShared + numUe*22);
        uint8_t cqi = cqiI[threadIdx.x];
        
        valueShared[threadIdx.x] = pfm + (3-qos)*1000.0 + prio*10.0 + last_slot*100.0 + cqi*100.0;
    }
    __syncthreads(); 
    
    bitonicSort(valueShared, ueIdsShared, 1024);

    if (threadIdx.x < numUe) {
        uint16_t* ue_id = (uint16_t*) (dataBuffer + numUe*16);
        ue_id[threadIdx.x] = ueIdsShared[threadIdx.x];
    }
}
 
int main() {
    // number of UEs to test
    int numUe = 1000;
    printf("numUe = %d\n", numUe);

    std::unique_ptr<inputStruct> inputDataPtr = std::make_unique<inputStruct>();

    // allocate host memory
    std::unique_ptr<uint8_t[]> dataBuffer = std::make_unique<uint8_t[]>(23*numUe);

    // assign memory for buffers
    inputDataPtr->pfm                       = (float*) dataBuffer.get();
    inputDataPtr->gbr_limit                 = (float*) (inputDataPtr->pfm + numUe);
    inputDataPtr->mbr_limit                 = (float*) (inputDataPtr->gbr_limit + numUe);
    inputDataPtr->last_slot_tbs_scheduled   = (uint32_t*) (inputDataPtr->mbr_limit + numUe);
    inputDataPtr->ue_id                     = (uint16_t*) (inputDataPtr->last_slot_tbs_scheduled + numUe);
    inputDataPtr->qos_type                  = (uint8_t*) (inputDataPtr->ue_id + numUe);
    inputDataPtr->lc_id                     = (uint8_t*) (inputDataPtr->qos_type + numUe);
    inputDataPtr->qos_5qi                   = (uint8_t*) (inputDataPtr->lc_id + numUe);
    inputDataPtr->priority                  = (uint8_t*) (inputDataPtr->qos_5qi + numUe);
    inputDataPtr->cqiI                      = (uint8_t*) (inputDataPtr->priority + numUe);

    // Allocate device memory
    size_t size = 23*numUe * sizeof(uint8_t);

    uint8_t* d_dataBuffer;
    CHECK_CUDA_ERR(cudaMalloc((void **)&d_dataBuffer, size));
    uint16_t* d_ue_id = (uint16_t*) (d_dataBuffer+numUe*16);

    ////////////////////////////
    // prepare random input data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(0, INT_MAX);

    for (int uIdx = 0; uIdx < numUe; uIdx++) {
        inputDataPtr->qos_type[uIdx] = distrib(gen) % 4;
        inputDataPtr->ue_id[uIdx] = uIdx;
        inputDataPtr->lc_id[uIdx] = uIdx;
        inputDataPtr->qos_5qi[uIdx] = distrib(gen) % 10;
        inputDataPtr->pfm[uIdx] = static_cast<float>(distrib(gen) % 10000);
        inputDataPtr->mbr_limit[uIdx] = 1.0e8f;
        inputDataPtr->gbr_limit[uIdx] = static_cast<float>(distrib(gen) % 100000000);
        inputDataPtr->priority[uIdx] = distrib(gen) % 100;
        inputDataPtr->last_slot_tbs_scheduled[uIdx] = distrib(gen) % 50;
        inputDataPtr->cqiI[uIdx] = distrib(gen) % 16;
    }

    // test input (for a small number of UEs)
    /*
    for (int uIdx = 0; uIdx < numUe; uIdx++) {
        printf("UE ID: %d, qos_type: %d, lc_id: %d, qos_5qi: %d, pfm: %f, mbr_limit: %f, gbr_limit: %f, priority: %d, last_slot_tbs_scheduled: %d, cqiI: %d\n", inputDataPtr->ue_id[uIdx], inputDataPtr->qos_type[uIdx], inputDataPtr->lc_id[uIdx], inputDataPtr->qos_5qi[uIdx], inputDataPtr->pfm[uIdx], inputDataPtr->mbr_limit[uIdx], inputDataPtr->gbr_limit[uIdx], inputDataPtr->priority[uIdx], inputDataPtr->last_slot_tbs_scheduled[uIdx], inputDataPtr->cqiI[uIdx]);
    }
    */
    //////////////////////////
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

    // CUDA kernel layout
    int threadsPerBlock = 1024;
    int blocksPerGrid = 1;

    // perform PFM sorting in GPU
    // step 1: copy data from host to device
    CHECK_CUDA_ERR(cudaEventRecord(startCopyH2D));
    for (int rIdx = 0; rIdx < 1000; rIdx++) {
        CHECK_CUDA_ERR(cudaMemcpy(d_dataBuffer, dataBuffer.get(), size, cudaMemcpyHostToDevice));
    }
    CHECK_CUDA_ERR(cudaEventRecord(stopCopyH2D));
    CHECK_CUDA_ERR(cudaEventSynchronize(stopCopyH2D));
 
    // step 2: launch kernel for PFM sorting in GPU
    CHECK_CUDA_ERR(cudaEventRecord(startKernel));
    for (int rIdx = 0; rIdx < 1000; rIdx++) {
        pfmSort<<<blocksPerGrid, threadsPerBlock>>>(d_dataBuffer, numUe);
        CHECK_CUDA_ERR(cudaGetLastError());
    }
    CHECK_CUDA_ERR(cudaEventRecord(stopKernel));
    CHECK_CUDA_ERR(cudaEventSynchronize(stopKernel));
 
    // step 3: copy result back to host
    CHECK_CUDA_ERR(cudaEventRecord(startCopyD2H));
    for (int rIdx = 0; rIdx < 1000; rIdx++) {
        CHECK_CUDA_ERR(cudaMemcpy(inputDataPtr->ue_id, d_ue_id, numUe*sizeof(uint16_t), cudaMemcpyDeviceToHost));
    }
    CHECK_CUDA_ERR(cudaEventRecord(stopCopyD2H));
    CHECK_CUDA_ERR(cudaEventSynchronize(stopCopyD2H));

    // calculate timings
    float timeH2D, timeKernel, timeD2H;

    CHECK_CUDA_ERR(cudaEventElapsedTime(&timeH2D, startCopyH2D, stopCopyH2D));
    CHECK_CUDA_ERR(cudaEventElapsedTime(&timeKernel, startKernel, stopKernel));
    CHECK_CUDA_ERR(cudaEventElapsedTime(&timeD2H, startCopyD2H, stopCopyD2H));

    std::cout << "Host to Device copy time: " << timeH2D<< " microseconds\n";
    std::cout << "Kernel execution time:     " << timeKernel << " microseconds\n";
    std::cout << "Device to Host copy time: " << timeD2H << " microseconds\n";

    // printf sorted UE IDs
    /*
    printf("Sorted UE IDs: ");
    for (int uIdx = 0; uIdx < numUe; uIdx++) {
        printf("%d ", inputDataPtr->ue_id[uIdx]);
    }
    printf("\n");
    */
 
    // Free memory
    CHECK_CUDA_ERR(cudaFree(d_dataBuffer));

    CHECK_CUDA_ERR(cudaEventDestroy(startCopyH2D));
    CHECK_CUDA_ERR(cudaEventDestroy(stopCopyH2D));
    CHECK_CUDA_ERR(cudaEventDestroy(startKernel));
    CHECK_CUDA_ERR(cudaEventDestroy(stopKernel));
    CHECK_CUDA_ERR(cudaEventDestroy(startCopyD2H));
    CHECK_CUDA_ERR(cudaEventDestroy(stopCopyD2H));

    return 0;
}