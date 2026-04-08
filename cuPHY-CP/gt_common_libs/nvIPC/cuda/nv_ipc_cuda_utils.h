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

#ifndef _NV_IPC_CUDA_UTILS_H_
#define _NV_IPC_CUDA_UTILS_H_

#if defined(__cplusplus)
extern "C" {
#endif

// Check whether CUDA driver and CUDA device exist. Return 0 if exist, else return -1
int cuda_version_check();

// Set page lock/unlock for host memory
int cuda_page_lock(void* phost, size_t size);
int cuda_page_unlock(void* phost);

// Check if a CPU memory buffer is host pinned memory: 1 if yes, 0 if no
int cuda_is_host_pinned_memory(void* phost);

// CUDA memory copy function wrapper
int nv_ipc_memcpy_to_host(void* host, const void* device, size_t size);
int nv_ipc_memcpy_to_device(void* device, const void* host, size_t size);

#if defined(__cplusplus)
} /* extern "C" */
#endif

#endif /* _NV_IPC_CUDA_UTILS_H_ */
