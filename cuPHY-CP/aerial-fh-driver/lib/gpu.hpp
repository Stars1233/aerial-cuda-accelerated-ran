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

#ifndef AERIAL_FH_GPU__
#define AERIAL_FH_GPU__

#include "aerial-fh-driver/api.hpp"

#define ASSERT_CUDA_FH(stmt)                   \
    do                                               \
    {                                                \
        cudaError_t result = (stmt);                 \
        if(cudaSuccess != result)                    \
        {                                            \
            NVLOGE_FMT(TAG, AERIAL_DPDK_API_EVENT, "[{}:{}] cuda failed with {} ", \
                   __FILE__,                         \
                   __LINE__,                         \
                   cudaGetErrorString(result));      \
        }                                            \
    } while(0)

#define ASSERT_CU_FH(stmt)                         \
    do                                             \
    {                                              \
        CUresult result = (stmt);                  \
        if(CUDA_SUCCESS != result)                 \
        {                                          \
            NVLOGE_FMT(TAG, AERIAL_DPDK_API_EVENT, "[{}:{}] cu failed with {} ", \
                   __FILE__,                       \
                   __LINE__,                       \
                   result);                        \
        }                                          \
    } while(0)

namespace aerial_fh
{
class Fronthaul;

class Gpu {
public:
    Gpu(Fronthaul* fhi, GpuId cuda_device_id);
    ~Gpu();
    GpuId       get_cuda_device_id() const;
    std::string get_pci_bus_id() const;

    static std::string cuda_device_id_to_pci_bus_id(GpuId cuda_device_id);
    static int16_t     cuda_device_id_to_dpdk_gpu_id(GpuId cuda_device_id);

protected:
    Fronthaul*  fhi_;
    GpuId       cuda_device_id_{-1};
    std::string pci_bus_id_;
    int16_t     dpdk_gpu_id_{-1};
};

} // namespace aerial_fh

#endif //ifndef AERIAL_FH_GPU__