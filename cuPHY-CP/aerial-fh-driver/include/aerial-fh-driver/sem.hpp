/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef AERIAL_FH_DRIVER_SEM__
#define AERIAL_FH_DRIVER_SEM__

#include "doca_structs.hpp"
#include <doca_gpunetio.h>

/**
 * Set packet information in the aerial FH GPU semaphore
 *
 * This function writes packet information to the specified index in the semaphore's packet info array.
 * It ensures proper memory ordering by using volatile accesses and a thread fence.
 *
 * @param[in] semaphore_gpu Pointer to aerial FH GPU semaphore structure
 * @param[in] idx Index in the packet info array
 * @param[in] status Status to set for this packet info entry
 * @param[in] num_packets Number of packets
 * @param[in] doca_buf_idx_start Starting index of DOCA buffer
 */
__device__ inline void
aerial_fh_gpu_dev_semaphore_set_packet_info(struct aerial_fh_gpu_semaphore_gpu *semaphore_gpu,
                                             const uint32_t idx,
                                             const enum aerial_fh_gpu_semaphore_status status,
                                             const uint32_t num_packets,
                                             const uint64_t doca_buf_idx_start)
{
	DOCA_GPUNETIO_VOLATILE(semaphore_gpu->pkt_info_gpu[idx].num_packets) = num_packets;
	DOCA_GPUNETIO_VOLATILE(semaphore_gpu->pkt_info_gpu[idx].doca_buf_idx_start) = doca_buf_idx_start;
	__threadfence();
	DOCA_GPUNETIO_VOLATILE(semaphore_gpu->pkt_info_gpu[idx].status) = status;
}

/**
 * Get packet information from aerial FH GPU semaphore if status matches
 *
 * This function checks if the packet info at the specified index has the expected status.
 * If the status matches, it retrieves the packet information.
 *
 * @param[in] semaphore_gpu Pointer to aerial FH GPU semaphore structure
 * @param[in] idx Index in the packet info array
 * @param[in] status Expected status to check
 * @param[out] num_packets Pointer to store the number of packets
 * @param[out] doca_buf_idx_start Pointer to store the starting index of DOCA buffer
 *
 * @return DOCA_SUCCESS if status matches and data retrieved, DOCA_ERROR_NOT_FOUND otherwise
 */
__device__ inline doca_error_t
aerial_fh_gpu_dev_semaphore_get_packet_info_status(struct aerial_fh_gpu_semaphore_gpu *semaphore_gpu,
                                                    const uint32_t idx,
                                                    const enum aerial_fh_gpu_semaphore_status status,
                                                    uint32_t *num_packets,
                                                    uint64_t *doca_buf_idx_start)
{
	if (DOCA_GPUNETIO_VOLATILE(semaphore_gpu->pkt_info_gpu[idx].status) == status) {
		(*num_packets) = DOCA_GPUNETIO_VOLATILE(semaphore_gpu->pkt_info_gpu[idx].num_packets);
		(*doca_buf_idx_start) = DOCA_GPUNETIO_VOLATILE(semaphore_gpu->pkt_info_gpu[idx].doca_buf_idx_start);
		return DOCA_SUCCESS;
	}
	return DOCA_ERROR_NOT_FOUND;
}

/**
 * Set status in the aerial FH GPU semaphore
 *
 * This function updates only the status field of the packet info at the specified index.
 *
 * @param[in] semaphore_gpu Pointer to aerial FH GPU semaphore structure
 * @param[in] idx Index in the packet info array
 * @param[in] status Status to set
 */
__device__ inline void
aerial_fh_gpu_dev_semaphore_set_status(struct aerial_fh_gpu_semaphore_gpu *semaphore_gpu,
                                        const uint32_t idx,
                                        const enum aerial_fh_gpu_semaphore_status status)
{
	DOCA_GPUNETIO_VOLATILE(semaphore_gpu->pkt_info_gpu[idx].status) = status;
}

#endif // AERIAL_FH_DRIVER_SEM__

