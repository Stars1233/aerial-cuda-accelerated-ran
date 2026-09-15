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

#ifndef CUDA_EVENTS_H
#define CUDA_EVENTS_H

#include <cstdint>
#include <limits>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include "aerial_event_code.h"
#include "cuda_driver_utils/cuda_driver_utils.hpp"

/**
 * @brief Calculate elapsed time between two CUDA events
 * 
 * Returns the time elapsed between events start and end, in milliseconds. The func argument should 
 * generally be set to __func__ from the caller so that any log statements have some information about 
 * the caller. The id argument, if set, is logged as an object identifier for additional source 
 * attribution of an error condition. In the case of an error, the return value is 0.
 * 
 * @param start     Start CUDA event (must be recorded before end event)
 * @param end       End CUDA event (must be recorded after start event)
 * @param func      Caller function name (typically __func__) for logging
 * @param id        Optional object identifier for error attribution (default: max uint64_t = not set)
 * @return float    Elapsed time in milliseconds (0.0f on error)
 */
inline float GetCudaEventElapsedTime(cudaEvent_t start, cudaEvent_t end, const char *func, uint64_t id = std::numeric_limits<uint64_t>::max())
{
    // Check for a pre-existing asynchronous error code so that it can be logged.
    // This error could otherwise be replaced with cudaErrorInvalidResourceHandle if
    // one of the events has not been recorded, so at least log any pre-existing errors
    // to aid debugging.
    // NOTE: After driver API migration, cudaGetLastError() only clears the runtime
    // sticky error state. Errors from driver API calls (cuMemAlloc, cuEventRecord, etc.)
    // are NOT reported here. Remove once all runtime API usage is eliminated.

    cudaError_t ret = cudaGetLastError();  ///< Check for any pre-existing async CUDA errors (Runtime; no Driver equivalent)
    if (ret != cudaSuccess) {
        NVLOGE_FMT(TAG,AERIAL_CUDA_API_EVENT,"{}: Async CUDA error: return status {}",func,cudaGetErrorString(ret));
    }
    float ms;  ///< Elapsed time in milliseconds (output)
    CUresult cuRet = cuEventElapsedTime(&ms, start, end);  ///< Calculate elapsed time between events
    if (cuRet != CUDA_SUCCESS) {
        // Query each event to collect additional information for debugging purposes. Note that since these
        // queries are after the failure of cuEventElapsedTime, they may no longer reflect the state at
        // the time that cuEventElapsedTime failed. For example, it is possible for an event to have
        // completed in the time between the calls to cuEventElapsedTime and cuEventQuery.
        const CUresult qRet1 = cuEventQuery(start);   ///< Query start event status for debugging
        const CUresult qRet2 = cuEventQuery(end);     ///< Query end event status for debugging
        const char* cuRetStr = nullptr;
        const char* qRet1Str = nullptr;
        const char* qRet2Str = nullptr;
        cuGetErrorString(cuRet, &cuRetStr);
        cuGetErrorString(qRet1, &qRet1Str);
        cuGetErrorString(qRet2, &qRet2Str);
        if (id != std::numeric_limits<uint64_t>::max()) {  ///< Log with object ID if provided
            NVLOGW_FMT(TAG,"{}: cuEventElapsedTime error {} follow-up cuEventQuery start {} end {} for Obj {:x}",
                func, cuRetStr ? cuRetStr : "unknown", qRet1Str ? qRet1Str : "unknown", qRet2Str ? qRet2Str : "unknown", id);
        } else {  ///< Log without object ID
            NVLOGW_FMT(TAG,"{}: cuEventElapsedTime error {} follow-up cuEventQuery start {} end {}",
                func, cuRetStr ? cuRetStr : "unknown", qRet1Str ? qRet1Str : "unknown", qRet2Str ? qRet2Str : "unknown");
        }
        return 0.0f;  ///< Return 0 on error
    }
    return ms;  ///< Return elapsed time in milliseconds
}

#endif
