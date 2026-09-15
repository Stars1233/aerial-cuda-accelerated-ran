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
#include <string.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#define TAG "NVIPC.TESTMEMCPY"
#include "cuda_driver_utils/cuda_driver_utils.hpp"

#define SIZE (64 * 1024 * 1024)


float cuda_malloc_test(int size, bool up)
{
    cudaEvent_t start, stop;
    int *       a = NULL;
    CUdeviceptr dev_a = 0;
    float       elapsedTime;

    CUDA_DRIVER_CHECK(cuEventCreate(&start, 0));
    CUDA_DRIVER_CHECK(cuEventCreate(&stop, 0));

    a = (int*)malloc(size * sizeof(*a));
    if (a == NULL)
    {
        return 0.0;
    }
    memset(a, 0, size * sizeof(*a));
    CUDA_DRIVER_CHECK(cuMemAlloc(&dev_a, size * sizeof(*a)));

    CUDA_DRIVER_CHECK(cuEventRecord(start, 0));
    for(int i = 0; i < 100; i++)
    {
        if(up)
            CUDA_DRIVER_CHECK(
                cuMemcpyHtoD(dev_a, a, size * sizeof(*a)));
        else
            CUDA_DRIVER_CHECK(
                cuMemcpyDtoH(a, dev_a, size * sizeof(*a)));
    }
    CUDA_DRIVER_CHECK(cuEventRecord(stop, 0));
    CUDA_DRIVER_CHECK(cuEventSynchronize(stop));
    CUDA_DRIVER_CHECK(cuEventElapsedTime(&elapsedTime, start, stop));

    free(a);
    CUDA_DRIVER_CHECK(cuMemFree(dev_a));
    CUDA_DRIVER_CHECK(cuEventDestroy(start));
    CUDA_DRIVER_CHECK(cuEventDestroy(stop));

    return elapsedTime;
}

float cuda_host_register_test(int size, bool up)
{
    cudaEvent_t start, stop;
    int *       a;
    CUdeviceptr dev_a = 0;
    float       elapsedTime;

    CUDA_DRIVER_CHECK(cuEventCreate(&start, 0));
    CUDA_DRIVER_CHECK(cuEventCreate(&stop, 0));

    a = (int*)malloc(size * sizeof(*a));
    if (a == NULL)
    {
        return 0.0;
    }
    memset(a, 0, size * sizeof(*a));

    unsigned int flag = CU_MEMHOSTREGISTER_PORTABLE | CU_MEMHOSTREGISTER_DEVICEMAP;
    CUDA_DRIVER_CHECK(cuMemHostRegister(a, size * sizeof(*a), flag));

    CUDA_DRIVER_CHECK(cuMemAlloc(&dev_a, size * sizeof(*a)));

    CUDA_DRIVER_CHECK(cuEventRecord(start, 0));
    for(int i = 0; i < 100; i++)
    {
        if(up)
            CUDA_DRIVER_CHECK(
                cuMemcpyHtoD(dev_a, a, size * sizeof(*a)));
        else
            CUDA_DRIVER_CHECK(
                cuMemcpyDtoH(a, dev_a, size * sizeof(*a)));
    }
    CUDA_DRIVER_CHECK(cuEventRecord(stop, 0));
    CUDA_DRIVER_CHECK(cuEventSynchronize(stop));
    CUDA_DRIVER_CHECK(cuEventElapsedTime(&elapsedTime, start, stop));
    CUDA_DRIVER_CHECK(cuMemHostUnregister(a));
    free(a);
    CUDA_DRIVER_CHECK(cuMemFree(dev_a));
    CUDA_DRIVER_CHECK(cuEventDestroy(start));
    CUDA_DRIVER_CHECK(cuEventDestroy(stop));

    return elapsedTime;
}

float cuda_host_alloc_test(int size, bool up)
{
    cudaEvent_t start, stop;
    void *      a = NULL;
    CUdeviceptr dev_a = 0;
    float       elapsedTime;

    CUDA_DRIVER_CHECK(cuEventCreate(&start, 0));
    CUDA_DRIVER_CHECK(cuEventCreate(&stop, 0));

    CUDA_DRIVER_CHECK(cuMemAllocHost(&a, size * sizeof(int)));
    CUDA_DRIVER_CHECK(cuMemAlloc(&dev_a, size * sizeof(int)));

    CUDA_DRIVER_CHECK(cuEventRecord(start, 0));
    for(int i = 0; i < 100; i++)
    {
        if(up)
            CUDA_DRIVER_CHECK(
                cuMemcpyHtoD(dev_a, a, size * sizeof(int)));
        else
            CUDA_DRIVER_CHECK(
                cuMemcpyDtoH(a, dev_a, size * sizeof(int)));
    }
    CUDA_DRIVER_CHECK(cuEventRecord(stop, 0));
    CUDA_DRIVER_CHECK(cuEventSynchronize(stop));
    CUDA_DRIVER_CHECK(cuEventElapsedTime(&elapsedTime, start, stop));

    CUDA_DRIVER_CHECK(cuMemFreeHost(a));
    CUDA_DRIVER_CHECK(cuMemFree(dev_a));
    CUDA_DRIVER_CHECK(cuEventDestroy(start));
    CUDA_DRIVER_CHECK(cuEventDestroy(stop));

    return elapsedTime;
}

int main(void)
{
    PrimaryCtxGuard ctx_guard(0);

    float elapsedTime;
    float MB = (float)100 * SIZE * sizeof(int) / 1024 / 1024;

    elapsedTime = cuda_malloc_test(SIZE, true);
    printf("Time using cuMemAlloc:  %3.1f ms", elapsedTime);
    printf("\tMB/s during copy up:  %3.1f\n", MB / (elapsedTime / 1000));

    elapsedTime = cuda_malloc_test(SIZE, false);
    printf("Time using cuMemAlloc:  %3.1f ms", elapsedTime);
    printf("\tMB/s during copy down:  %3.1f\n", MB / (elapsedTime / 1000));

    elapsedTime = cuda_host_alloc_test(SIZE, true);
    printf("Time using cuMemAllocHost:  %3.1f ms", elapsedTime);
    printf("\tMB/s during copy up:  %3.1f\n", MB / (elapsedTime / 1000));

    elapsedTime = cuda_host_alloc_test(SIZE, false);
    printf("Time using cuMemAllocHost:  %3.1f ms", elapsedTime);
    printf("\tMB/s during copy down:  %3.1f\n", MB / (elapsedTime / 1000));

    elapsedTime = cuda_host_register_test(SIZE, true);
    printf("Time using cuMemHostRegister:  %3.1f ms", elapsedTime);
    printf("\tMB/s during copy up:  %3.1f\n", MB / (elapsedTime / 1000));

    elapsedTime = cuda_host_register_test(SIZE, false);
    printf("Time using cuMemHostRegister:  %3.1f ms", elapsedTime);
    printf("\tMB/s during copy down:  %3.1f\n", MB / (elapsedTime / 1000));

}
