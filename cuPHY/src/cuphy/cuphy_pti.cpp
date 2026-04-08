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

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <sys/mman.h>
#include <fcntl.h>
#include "nvlog.hpp"
#include "cuphy_pti.hpp"
#include "cuphy_pti_internal.hpp"

#define TAG "CUPHY.PTI"
#define TAG_TIMING "CUPHY.PTI"

// PTI stats
static cuphy_pti_all_stats_t _cuphy_pti_all_stats[CUPHY_PTI_INDEX_MAX];

// Timing kernel data
constexpr int MAX_TIMING_KERNEL_RECORDS = 4;
static uint64_t *d_ptpreg {0};
static uint64_t *hd_ptp_time_ns;
static uint64_t *hd_gpu_time_v0_ns;
static uint64_t *hd_gpu_time_v1_ns;
static uint64_t prev_ptp_time_ns {0};
static uint64_t prev_gpu_time_ns {0};
static uint64_t h_correlation_id[MAX_TIMING_KERNEL_RECORDS];
static double timing_clock_rate[MAX_TIMING_KERNEL_RECORDS];
static double timing_loop_coeff {1.0};
static double timing_clock_rate_internal {1.0};
static std::atomic<int> timing_valid {0};
static std::atomic<int> timing_rd_index {0};
static int timing_wr_index {0};

void cuphy_pti_init(const char* nic_pci_addr)
{
    for (int k=0; k<CUPHY_PTI_INDEX_MAX; k++)
    {
        CHECK_CUDA(cudaHostAlloc(&_cuphy_pti_all_stats[k].dh_gpu_start_times,sizeof(uint64_t)*CUPHY_PTI_ACTIVITIES_MAX,cudaHostAllocPortable));
        CHECK_CUDA(cudaHostAlloc(&_cuphy_pti_all_stats[k].dh_gpu_stop_times,sizeof(uint64_t)*CUPHY_PTI_ACTIVITIES_MAX,cudaHostAllocPortable));
        CHECK_CUDA(cudaMalloc(&_cuphy_pti_all_stats[k].d_cta_counts,sizeof(uint32_t)*CUPHY_PTI_ACTIVITIES_MAX));
        CHECK_CUDA(cudaMemset(_cuphy_pti_all_stats[k].d_cta_counts,0,sizeof(uint32_t)*CUPHY_PTI_ACTIVITIES_MAX));
        memfoot_add_gpu_size(MF_TAG_CUPHY_OTHER, sizeof(uint32_t)*CUPHY_PTI_ACTIVITIES_MAX);
    }

    constexpr uint32_t PTP_REG_ADDR_OFFSET = 0x1040;
    constexpr uint32_t PCIE_BAR0_SIZE = 8192;
    char filename[1024];
    snprintf(filename,1022,"/sys/bus/pci/devices/%s/resource0",nic_pci_addr);

    int fd = open(filename, O_RDWR);
    if (fd < 0)
    {
        printf("Unable to open %s.  Must be run as root.\n",filename);
        exit(1);
    }

    void* m = mmap(0, PCIE_BAR0_SIZE, PROT_WRITE | PROT_READ, MAP_SHARED, fd, 0);
    if (m == reinterpret_cast<void*>(-1))
    {
        char err_string[3000];
        snprintf(err_string,2998,"Unable to mmap PCIE_BAR0 using %s",filename);
        perror(err_string);
        exit(1);
    }

    CHECK_CUDA(cudaHostRegister(m, PCIE_BAR0_SIZE, cudaHostRegisterMapped | cudaHostRegisterPortable | cudaHostRegisterIoMemory));

    uint64_t *h_ptpreg = reinterpret_cast<uint64_t*>(reinterpret_cast<char*>(m) + PTP_REG_ADDR_OFFSET);
    CHECK_CUDA(cudaHostGetDevicePointer(reinterpret_cast<void**>(&d_ptpreg), const_cast<void*>(reinterpret_cast<volatile void*>(h_ptpreg)), 0));

    CHECK_CUDA(cudaMallocHost(&hd_ptp_time_ns, sizeof(uint64_t)*MAX_TIMING_KERNEL_RECORDS));
    CHECK_CUDA(cudaMallocHost(&hd_gpu_time_v0_ns, sizeof(uint64_t)*MAX_TIMING_KERNEL_RECORDS));
    CHECK_CUDA(cudaMallocHost(&hd_gpu_time_v1_ns, sizeof(uint64_t)*MAX_TIMING_KERNEL_RECORDS));
    memset(hd_ptp_time_ns, 0, sizeof(uint64_t)*MAX_TIMING_KERNEL_RECORDS);
    memset(hd_gpu_time_v0_ns, 0, sizeof(uint64_t)*MAX_TIMING_KERNEL_RECORDS);
    memset(hd_gpu_time_v1_ns, 0, sizeof(uint64_t)*MAX_TIMING_KERNEL_RECORDS);
}

__thread int _cuphy_pti_record_index = -1;
void cuphy_pti_set_record_index(int record_index)
{
    _cuphy_pti_record_index = record_index;
}

int cuphy_pti_get_record_index(void)
{
    return _cuphy_pti_record_index;
}

struct cuphy_pti_all_stats_t* _cuphy_pti_get_record_all_activities()
{
    return &_cuphy_pti_all_stats[_cuphy_pti_record_index];
}

void _cuphy_pti_get_record_activity(struct cuphy_pti_activity_stats_t& activity_stats, cuphy_pti_activity_t activity)
{
    activity_stats.dh_gpu_start_time = &_cuphy_pti_all_stats[_cuphy_pti_record_index].dh_gpu_start_times[activity];
    activity_stats.dh_gpu_stop_time = &_cuphy_pti_all_stats[_cuphy_pti_record_index].dh_gpu_stop_times[activity];
    activity_stats.d_cta_count = &_cuphy_pti_all_stats[_cuphy_pti_record_index].d_cta_counts[activity];
}

void cuphy_pti_calibrate_gpu_timer(cudaStream_t stream, uint64_t id)
{
    // Time between timing kernel launches is so large that we assume the previous kernel finished.

    uint64_t ptp_time_ns = hd_ptp_time_ns[timing_wr_index];
    uint64_t gpu_time_v0_ns = hd_gpu_time_v0_ns[timing_wr_index];
    uint64_t gpu_time_v1_ns = hd_gpu_time_v1_ns[timing_wr_index];
    int error = 0;

    if ((gpu_time_v1_ns - gpu_time_v0_ns) < 2048)
    {
        if (prev_ptp_time_ns != 0)
        {
            uint64_t ptp_delta_ns = ptp_time_ns - prev_ptp_time_ns;
            uint64_t gpu_target_ns = prev_gpu_time_ns + static_cast<uint64_t>(ptp_delta_ns * timing_clock_rate_internal);
            error = gpu_time_v1_ns - gpu_target_ns;
            double frac_error = static_cast<double>(error)/(gpu_time_v1_ns - prev_gpu_time_ns);

            timing_clock_rate_internal += timing_loop_coeff*frac_error;
            timing_clock_rate[timing_wr_index] = timing_clock_rate_internal;
            timing_rd_index = timing_wr_index;
            timing_valid = 1;

            NVLOGI_FMT(TAG_TIMING,"{} seconds since last timing update, correlation id {} PTP time {} GPU time {} PCI read time {} clock_rate {} ({} ppm), loop error ticks {}, prev_ptp_time {}, prev_gpu_time {}",
                        static_cast<float>(ptp_time_ns - prev_ptp_time_ns)/1e9,
                        h_correlation_id[timing_wr_index],
                        ptp_time_ns,
                        gpu_time_v1_ns,
                        gpu_time_v1_ns - gpu_time_v0_ns,
                        timing_clock_rate_internal,
                        (timing_clock_rate_internal-1.0) * 1e6,
                        error,
                        prev_ptp_time_ns,
                        prev_gpu_time_ns);

            if (timing_loop_coeff == 1)
            {
                timing_loop_coeff = 0.1;
            }
        }

        timing_wr_index = (timing_wr_index + 1) % MAX_TIMING_KERNEL_RECORDS;
        prev_ptp_time_ns = ptp_time_ns;
        prev_gpu_time_ns = gpu_time_v1_ns;
    }
    else
    {
        NVLOGI_FMT(TAG_TIMING,"PTP time {} GPU time {} PCI read time {} (too large, loop not updated)",
                   ptp_time_ns,
                   gpu_time_v1_ns,
                   gpu_time_v1_ns - gpu_time_v0_ns);
    }


    NVLOGD_FMT(TAG_TIMING,"Launching ptp clock tracker, timing_wr_index {}", timing_wr_index);
    hd_ptp_time_ns[timing_wr_index] = 0;
    hd_gpu_time_v0_ns[timing_wr_index] = 0;
    hd_gpu_time_v1_ns[timing_wr_index] = 0;
    h_correlation_id[timing_wr_index] = id;
    cuphy_pti_launch_snapshot_ptp(d_ptpreg,
                                  &hd_ptp_time_ns[timing_wr_index],
                                  &hd_gpu_time_v0_ns[timing_wr_index],
                                  &hd_gpu_time_v1_ns[timing_wr_index],
                                  stream);
}
