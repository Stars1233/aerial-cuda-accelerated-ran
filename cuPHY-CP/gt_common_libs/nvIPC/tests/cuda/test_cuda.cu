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
#include <cuda.h>

#include "test_cuda.h"
#include "nv_ipc_utils.h"

#define TAG (NVLOG_TAG_BASE_NVIPC + 5) // "NVIPC.TESTCUDA"
#include "cuda_driver_utils/cuda_driver_utils.hpp"
#include "cuda_driver_utils/cuda_kernel_utils.cuh"

#define N_BLOCK 64
#define N_THREAD 32;

static const char SUB_VAL = (char) ('A' - 'a');

static __global__ void gpu_to_lower_case(char* str, int length)
{
    int index  = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for(int i = index; i < length; i += stride)
    {
        if(str[i] >= 'A' && str[i] <= 'Z')
        {
            str[i] -= SUB_VAL;
        }
    }
}

// CUfunction handles are context-specific. All callers must use the same deviceId
// across the process lifetime. Must be initialized once from the main thread via
// init_test_cuda_kernels() before any calls to test_cuda_to_lower_case or
// cuda_to_lower_case.
//
// PrimaryCtxGuard is used to set the CUDA context on the calling thread (e.g. the
// receive thread in test_ipc.c). cuCtxSynchronize ensures the async cuLaunchKernel
// completes before PrimaryCtxGuard releases the context at scope exit.
static CUfunction gpu_to_lower_case_func = nullptr;

[[nodiscard]] bool init_test_cuda_kernels()
{
    return resolve_kernel_func<TAG>(&gpu_to_lower_case_func,
                                    reinterpret_cast<const void*>(gpu_to_lower_case),
                                    "gpu_to_lower_case");
}

extern "C" int init_test_cuda_kernels_c(void)
{
    return init_test_cuda_kernels() ? 0 : -1;
}

void test_cuda_to_lower_case(int deviceId, char* str, int length, int gpu)
{
    if(deviceId < 0)
    {
        NVLOGD_FMT(TAG, "{}: deviceId={}, fall back to CPU IPC test", __func__, deviceId);
        gpu = 0;
    }

    NVLOGI_FMT(TAG, "{}: gpu={}", __func__, gpu);
    if(gpu)
    {
        PrimaryCtxGuard ctx_guard(deviceId);

        int nblock  = N_BLOCK;
        int nthread = N_THREAD;
        void* args[] = {&str, &length};
        CUDA_DRIVER_CHECK(cuLaunchKernel(gpu_to_lower_case_func, nblock, 1, 1, nthread, 1, 1, 0, nullptr, args, nullptr));
        CUDA_DRIVER_CHECK(cuCtxSynchronize());
    }
    else
    {
        cpu_to_lower_case(str, length);
    }
}

void cuda_to_lower_case(char* str, int length, int deviceId)
{
    if(deviceId < 0)
    {
        NVLOGC_FMT(TAG, "{}: invalid CUDA deviceId: {}", __func__, deviceId);
        return;
    }

    PrimaryCtxGuard ctx_guard(deviceId);

    int nblock  = N_BLOCK;
    int nthread = N_THREAD;
    void* args[] = {&str, &length};
    CUDA_DRIVER_CHECK(cuLaunchKernel(gpu_to_lower_case_func, nblock, 1, 1, nthread, 1, 1, 0, nullptr, args, nullptr));
    CUDA_DRIVER_CHECK(cuCtxSynchronize());
}
