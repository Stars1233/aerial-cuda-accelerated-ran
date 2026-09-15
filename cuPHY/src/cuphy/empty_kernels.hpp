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

#if !defined(EMPTY_KERNELS_HPP_INCLUDED_)
#define EMPTY_KERNELS_HPP_INCLUDED_

#include "cuphy.h"

template<size_t SZ>
struct testDescr_sz
{
    char tmp[SZ];
};

cuphyStatus_t CUPHYWINAPI internalCuphySetGenericEmptyKernelNodeGridConstantParams(CUDA_KERNEL_NODE_PARAMS* pNodeParams, void** pKernelParams, int ptrArgsCnt, uint16_t descr_size);
cuphyStatus_t CUPHYWINAPI internalCuphySetGenericEmptyKernelNodeParams(CUDA_KERNEL_NODE_PARAMS* pNodeParams, int ptrArgsCnt, void** pKernelParams);
/** @brief: Set kernel node parameters for delay kernel
 *
 * @param[in] pNodeParams: Pointer to CUDA_KERNEL_NODE_PARAMS to populate
 * @param[in] pKernelParams: Pointer to kernel parameters (delay in usec)
 * @return CUPHY_STATUS_SUCCESS or CUPHY_STATUS_INVALID_ARGUMENT or CUPHY_STATUS_INTERNAL_ERROR
 */
cuphyStatus_t CUPHYWINAPI internalCuphySetDelayKernelNodeParams(CUDA_KERNEL_NODE_PARAMS* pNodeParams, void** pKernelParams);
cuphyStatus_t CUPHYWINAPI internalCuphySetEmptyKernelNodeParams(CUDA_KERNEL_NODE_PARAMS* pNodeParams);
cuphyStatus_t CUPHYWINAPI internalCuphySetWorkCancelKernelNodeParams(CUDA_KERNEL_NODE_PARAMS* pNodeParams, void** pKernelParams, uint8_t device_graph_launch);

#endif // !defined(EMPTY_KERNELS_HPP_INCLUDED_)
