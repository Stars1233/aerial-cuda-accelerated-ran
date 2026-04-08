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

#ifndef NV_IPC_UTILS_H_INCLUDED_
#define NV_IPC_UTILS_H_INCLUDED_

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <time.h>

#include "nv_ipc.h"
#include "nvlog.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifdef NVLOG_C // Change all NVLOGW to NVLOGI for FMT logger
#undef NVLOGW
#define NVLOGW NVLOGI
#endif

#define LOG_TEMP_FILE_PATH "/var/log/aerial"

#define SFN_SLOT_INVALID 0xFFFFFFFF

/** SFN and Slot union */
typedef union
{
    uint32_t u32;          //!< 32-bit representation
    struct
    {
        uint16_t sfn;      //!< Frame Number
        uint16_t slot;     //!< Slot number
    } u16;
} sfn_slot_t;

/**
 * Get SFN and slot from message
 *
 * @param[in] msg IPC message
 * @return SFN/slot union
 */
sfn_slot_t nv_ipc_get_sfn_slot(nv_ipc_msg_t* msg);

/**
 * Set handle ID in message
 *
 * @param[in,out] msg IPC message
 * @param[in] handle_id Handle identifier to set
 */
void nv_ipc_set_handle_id(nv_ipc_msg_t* msg, uint8_t handle_id);

/**
 * Get send timestamp for message
 *
 * @param[in] ipc IPC instance
 * @param[in] msg IPC message
 * @return Timestamp in nanoseconds, -1 on failure
 */
int64_t nv_ipc_get_ts_send(nv_ipc_t* ipc, nv_ipc_msg_t* msg);

/**
 * Send message in loopback mode (SHM only)
 *
 * @param[in] ipc IPC instance
 * @param[in] msg Message to send
 * @return 0 on success, -1 on failure
 */
int nv_ipc_shm_send_loopback(nv_ipc_t* ipc, nv_ipc_msg_t* msg);

/**
 * Poll for received messages (SHM only)
 *
 * @param[in] ipc IPC instance
 * @return Number of messages received, -1 on failure
 */
int nv_ipc_shm_rx_poll(nv_ipc_t* ipc);

/**
 * Check if all necessary data memory pools (cpu_data and cpu_large) are host pinned memory
 *
 * @param[in] ipc IPC instance
 */
void nv_ipc_check_host_pinned_memory(nv_ipc_t* ipc);

/**
 * Get CUDA device count
 *
 * @return Number of CUDA devices
 */
int cuda_get_device_count(void);

/**
 * Check if pointer is device memory
 *
 * @param[in] ptr Pointer to check
 * @return 1 if device pointer, 0 if host pointer
 */
int cuda_is_device_pointer(const void *ptr);

/**
 * Check if pointer is device memory (inline version)
 *
 * @param[in] ptr Pointer to check
 * @return 1 if device pointer, 0 if host pointer
 */
static inline int is_device_pointer(const void *ptr) {
    int is_in_gpu = 0;
#ifdef NVIPC_CUDA_ENABLE
    is_in_gpu = cuda_is_device_pointer(ptr);
#endif
    return is_in_gpu;
}

/**
 * Dump IPC state for debugging
 *
 * @param[in] ipc IPC instance
 * @return 0 on success, -1 on failure
 */
int nv_ipc_dump(nv_ipc_t* ipc);

/**
 * Set callback for IPC reconnection events
 *
 * Used by L2 adapter to cleanup and reset when NVIPC reconnects
 *
 * @param[in] ipc IPC instance
 * @param[in] callback Callback function to invoke on reconnect
 * @param[in] cb_args Arguments to pass to callback
 * @return 0 on success, -1 on failure
 */
int nv_ipc_set_reset_callback(nv_ipc_t* ipc, int (*callback)(void *), void *cb_args);

/**
 * Check if module type is primary
 *
 * @param[in] module_type Module type to check
 * @return 1 if primary, 0 if secondary
 */
int is_module_primary(nv_ipc_module_t module_type);

/**
 * Get primary configuration from shared memory
 *
 * Retrieves configuration saved by primary app
 *
 * @param[in] ipc IPC instance
 * @return Pointer to configuration, NULL on failure
 */
nv_ipc_config_t* nv_ipc_get_primary_config(nv_ipc_t* ipc);

/**
 * Lookup configuration initiated by primary app
 *
 * For secondary apps only: finds and copies primary's config
 *
 * @param[out] cfg Configuration buffer to populate
 * @param[in] prefix Instance name prefix
 * @param[in] module_type Module type
 * @return 0 on success, -1 if not found
 */
int nv_ipc_lookup_config(nv_ipc_config_t* cfg, const char* prefix, nv_ipc_module_t module_type);

/**
 * Dump configuration for debugging
 *
 * @param[in] cfg Configuration to dump
 * @return 0 on success, -1 on failure
 */
int nv_ipc_dump_config(nv_ipc_config_t *cfg);

#if defined(__cplusplus)
} /* extern "C" */
#endif

#ifdef __cplusplus
#include "nvlog.hpp"
#endif

#endif /* NV_IPC_UTILS_H_INCLUDED_ */
