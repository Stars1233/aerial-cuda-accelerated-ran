/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <cuda.h>
#include <pthread.h>
#include <unistd.h>
#include <assert.h>

#include "nvlog.hpp"
#include "test_kernel.h"
#include "cupti_helper.hpp"
#include "util.hpp"
#include "common_utils.hpp"

#define TAG "CUPHY.CUPTI"

#define CU_CHECK_PHYDRIVER(stmt)                   \
    do                                             \
    {                                              \
        CUresult result = (stmt);                  \
        if(CUDA_SUCCESS != result)                 \
        {                                          \
            printf("[%s:%d] cu failed with %d\n",  \
                   __FILE__,                       \
                   __LINE__,                       \
                   result);                        \
        }                                          \
        assert(CUDA_SUCCESS == result);            \
    } while(0)

uint64_t get_cpu_ns()
{
    struct timespec t;
    int             ret;
    ret = clock_gettime(CLOCK_REALTIME, &t);
    if(ret != 0)
    {
        printf("clock_gettime fail: %d\n",ret);
        exit(1);
    }
    return static_cast<uint64_t>(t.tv_nsec) + static_cast<uint64_t>(t.tv_sec) * 1000000000ULL;
}

struct thread_info
{
    CUcontext cuCtx;
    cudaStream_t stream;
    int initial_delay_us;
    int kernel_delay_us;
    int loop_delay_us;
    int count_max;
    int do_external;
    int do_init;
    int do_memops;
};

void* test_thread(void* p)
{
    nvlog_fmtlog_thread_init();
    thread_info *info = static_cast<thread_info*>(p);

    CU_CHECK_PHYDRIVER(cuCtxSetCurrent(info->cuCtx));

    if (info->do_init) cuphy_cupti_helper_init();
    usleep(info->initial_delay_us);

    if (info->do_memops)
    {
        void *h_x, *d_x;
        CUDA_CHECK(cudaHostAlloc(&h_x, 1<<20, cudaHostAllocPortable));
        CUDA_CHECK(cudaMalloc(&d_x, 1<<20));
        CUDA_CHECK(cudaMemcpyAsync(d_x, h_x, 1<<20, cudaMemcpyHostToDevice, info->stream));
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaFree(d_x));
        CUDA_CHECK(cudaFreeHost(h_x));
    }

    int count = 0;
    while (count < info->count_max) //10000)
    {
        if (info->do_external) cuphy_cupti_helper_push_external_id(count);
        launch_test_kernel(info->stream, count, info->kernel_delay_us, get_cpu_ns());

        usleep(info->loop_delay_us);
        if (info->do_external) cuphy_cupti_helper_pop_external_id();
        count++;
    }

    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess)
    {
        printf("cudaGetLastError returned %d\n",status);
    }

    return 0;
}

void thread_setup(pthread_t& thread, int core_id, thread_info* info)
{
    int ret;
    cpu_set_t cpuset;

    ret = pthread_create(&thread, NULL, test_thread, info);
    if (ret != 0)
    {
        perror("pthread_create");
        exit(1);
    }

    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);

    ret = pthread_setaffinity_np(thread, sizeof(cpuset), &cpuset);
    if (ret != 0)
    {
        perror("pthread_setaffinity_np");
        exit(1);
    }

    if (1)
    {
        struct sched_param schedprm{.__sched_priority = 95};
        ret = pthread_setschedparam(thread, SCHED_FIFO, &schedprm);
        if (ret != 0)
        {
            perror("pthread_setschedparam");
            exit(1);
        }
    }

    ret = pthread_setname_np(thread, "test_thread");
}

int main(void)
{

    char nvlog_yaml_file[1024];
    // Relative path from binary to default nvlog_config.yaml
    std::string relative_path = std::string("../../../../").append(NVLOG_DEFAULT_CONFIG_FILE);
    std::string log_name = "cuphy_cupti_example.log";
    nv_get_absolute_path(nvlog_yaml_file, relative_path.c_str());
    pthread_t log_thread_id = nvlog_fmtlog_init(nvlog_yaml_file, log_name.c_str(), NULL);
    nvlog_fmtlog_thread_init();
    NVLOGC_FMT(TAG,"Starting cuphy_cupti_example");

    cudaSetDevice(0);
    cuphy_cupti_helper_init();

    pthread_t threads[2];

    CUdevice cuDev;

    CUexecAffinityParam affinityPrm;
    affinityPrm.type = CU_EXEC_AFFINITY_TYPE_SM_COUNT;
    affinityPrm.param.smCount.val = 10; // a 224 error can only mean that MPS service is not running
    CU_CHECK_PHYDRIVER(cuDeviceGet(&cuDev, 0));

    thread_info infos[2];
#if CUDA_VERSION >= 13000
    CUctxCreateParams ctxParams{};
    ctxParams.execAffinityParams = &affinityPrm;
    ctxParams.numExecAffinityParams = 1;
    ctxParams.cigParams = nullptr;
    CU_CHECK_PHYDRIVER(cuCtxCreate(&infos[0].cuCtx, &ctxParams, CU_CTX_SCHED_SPIN | CU_CTX_MAP_HOST, cuDev));
    CU_CHECK_PHYDRIVER(cuCtxCreate(&infos[1].cuCtx, &ctxParams, CU_CTX_SCHED_SPIN | CU_CTX_MAP_HOST, cuDev));
#else
    CU_CHECK_PHYDRIVER(cuCtxCreate_v3(&infos[0].cuCtx, &affinityPrm, 1, CU_CTX_SCHED_SPIN | CU_CTX_MAP_HOST, cuDev));
    CU_CHECK_PHYDRIVER(cuCtxCreate_v3(&infos[1].cuCtx, &affinityPrm, 1, CU_CTX_SCHED_SPIN | CU_CTX_MAP_HOST, cuDev));
#endif

    CU_CHECK_PHYDRIVER(cuCtxSetCurrent(infos[0].cuCtx));
    CUDA_CHECK(cudaStreamCreate(&infos[0].stream));
    CU_CHECK_PHYDRIVER(cuCtxSetCurrent(infos[1].cuCtx));
    CUDA_CHECK(cudaStreamCreate(&infos[1].stream));
    infos[0].initial_delay_us = 1000000;
    infos[0].kernel_delay_us = 500000;
    infos[0].loop_delay_us = 2000000;
    infos[0].count_max = 2;
    infos[0].do_external = 1;
    infos[0].do_init = 0;
    infos[0].do_memops = 0;
    infos[1].initial_delay_us = 1200000;
    infos[1].kernel_delay_us = 1000000;
    infos[1].loop_delay_us = 2000000;
    infos[1].count_max = 2;
    infos[1].do_external = 1;
    infos[1].do_init = 0;
    infos[1].do_memops = 1;

    thread_setup(threads[0], 13, &infos[0]);
    thread_setup(threads[1], 15, &infos[1]);

    struct timespec tv;
    int ret = 1;
    while (ret != 0)
    {
        if (clock_gettime(CLOCK_REALTIME, &tv) < 0)
        {
            perror("clock_gettime");
            exit(1);
        }
        tv.tv_sec += 2;

        ret = pthread_timedjoin_np(threads[0], NULL, &tv);
    }
    ret = pthread_join(threads[1],NULL);

    cuphy_cupti_helper_stop();
}
