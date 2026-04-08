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

#include <cupti.h>
#include <stdio.h>
#include <unistd.h>
#include <mutex>
#include <pthread.h>
#include <vector>

#include "nvlog.hpp"
#include "cupti_helper.hpp"

#define TAG "CUPHY.CUPTI"

#define CHECK_CUDA(expr_to_check) do {            \
    cudaError_t result = expr_to_check;           \
    if(result != cudaSuccess)                     \
    {                                             \
        NVLOGF_FMT(TAG,                           \
                AERIAL_INTERNAL_EVENT,            \
                "CUDA Runtime Error: {}:{}:{}",   \
                __FILE__,                         \
                __LINE__,                         \
                cudaGetErrorString(result));      \
    }                                             \
} while (0)

#define CUPTI_EXTERNAL_CORRELATION_KIND_AERIAL CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM2

// 8-byte alignment for the buffers
#define ALIGN_SIZE (8)
#define ALIGN_BUFFER(buffer, align)                                                 \
  (((uintptr_t) (buffer) & ((align)-1)) ? ((buffer) + (align) - ((uintptr_t) (buffer) & ((align)-1))) : (buffer))

// Runtime configurable values (set during init)
static uint64_t g_cupti_buffer_size = 0;
static uint16_t g_cupti_num_buffers = 0;

#define MAX_ACTIVITY_PRINTS_PER_SECOND 100000LLU

#define CUPTI_API_CALL(apiFunctionCall)                                                                 \
do                                                                                                      \
{                                                                                                       \
    CUptiResult _status = apiFunctionCall;                                                              \
    if ((_status != CUPTI_SUCCESS) && (_status != CUPTI_ERROR_MAX_LIMIT_REACHED))                       \
    {                                                                                                   \
        const char *pErrorString;                                                                       \
        cuptiGetResultString(_status, &pErrorString);                                                   \
                                                                                                        \
        NVLOGF_FMT(TAG, AERIAL_CUPHY_EVENT, "{}:{}: Error: Function {} failed with error: {}.",         \
                __FILE__, __LINE__, #apiFunctionCall, pErrorString);                                    \
                                                                                                        \
        exit(EXIT_FAILURE);                                                                             \
    }                                                                                                   \
} while (0)

static const char *
GetName(
    const char *pName)
{
    if (pName == NULL)
    {
        return "<null>";
    }

    return pName;
}

static std::string
GetCuptiVersionString(
    const uint32_t version)
{
    // CUDA 13.0+ uses format xxyyzz (major.minor.patch)
    if (version >= 130000)
    {
        const uint32_t major = version / 10000;
        const uint32_t minor = (version / 100) % 100;
        const uint32_t patch = version % 100;
        
        if (patch > 0)
        {
            return fmt::format("CUDA Toolkit {}.{} Update {}", major, minor, patch);
        }
        else
        {
            return fmt::format("CUDA Toolkit {}.{}", major, minor);
        }
    }
    
    // Pre-CUDA 13 incremental API versions
    switch (version)
    {
        case 1:  return "CUDAToolsSDK 4.0";
        case 2:  return "CUDAToolsSDK 4.1";
        case 3:  return "CUDA Toolkit 5.0";
        case 4:  return "CUDA Toolkit 5.5";
        case 5:  return "CUDA Toolkit 6.0";
        case 6:  return "CUDA Toolkit 6.5";
        case 7:  return "CUDA Toolkit 6.5 (with sm_52 support)";
        case 8:  return "CUDA Toolkit 7.0";
        case 9:  return "CUDA Toolkit 8.0";
        case 10: return "CUDA Toolkit 9.0";
        case 11: return "CUDA Toolkit 9.1";
        case 12: return "CUDA Toolkit 10.0, 10.1 or 10.2";
        case 13: return "CUDA Toolkit 11.0";
        case 14: return "CUDA Toolkit 11.1";
        case 15: return "CUDA Toolkit 11.2, 11.3 or 11.4";
        case 16: return "CUDA Toolkit 11.5";
        case 17: return "CUDA Toolkit 11.6";
        case 18: return "CUDA Toolkit 11.8";
        case 19: return "CUDA Toolkit 12.0";
        case 20: return "CUDA Toolkit 12.2";
        case 21: return "CUDA Toolkit 12.3";
        case 22: return "CUDA Toolkit 12.4";
        case 23: return "CUDA Toolkit 12.5";
        case 24: return "CUDA Toolkit 12.6";
        case 26: return "CUDA Toolkit 12.8";
        case 27: return "CUDA Toolkit 12.9";
        case 28: return "CUDA Toolkit 12.9 Update 1";
        default: return "Unknown CUPTI version";
    }
}

static const char *
GetMemoryKindString(
    CUpti_ActivityMemoryKind memoryKind)
{
    switch (memoryKind)
    {
        case CUPTI_ACTIVITY_MEMORY_KIND_UNKNOWN:
            return "UNKNOWN";
        case CUPTI_ACTIVITY_MEMORY_KIND_PAGEABLE:
            return "PAGEABLE";
        case CUPTI_ACTIVITY_MEMORY_KIND_PINNED:
            return "PINNED";
        case CUPTI_ACTIVITY_MEMORY_KIND_DEVICE:
            return "DEVICE";
        case CUPTI_ACTIVITY_MEMORY_KIND_ARRAY:
            return "ARRAY";
        case CUPTI_ACTIVITY_MEMORY_KIND_MANAGED:
            return "MANAGED";
        case CUPTI_ACTIVITY_MEMORY_KIND_DEVICE_STATIC:
            return "DEVICE_STATIC";
        case CUPTI_ACTIVITY_MEMORY_KIND_MANAGED_STATIC:
            return "MANAGED_STATIC";
        default:
            return "<unknown>";
    }
}

static const char *
GetMemoryOperationTypeString(
    CUpti_ActivityMemoryOperationType memoryOperationType)
{
    switch (memoryOperationType)
    {
        case CUPTI_ACTIVITY_MEMORY_OPERATION_TYPE_ALLOCATION:
            return "ALLOC";
        case CUPTI_ACTIVITY_MEMORY_OPERATION_TYPE_RELEASE:
            return "FREE";
        default:
            return "<unknown>";
    }
}

static const char *
GetMemcpyKindString(
    CUpti_ActivityMemcpyKind memcpyKind)
{
    switch (memcpyKind)
    {
        case CUPTI_ACTIVITY_MEMCPY_KIND_UNKNOWN:
            return "UNKNOWN";
        case CUPTI_ACTIVITY_MEMCPY_KIND_HTOD:
            return "HtoD";
        case CUPTI_ACTIVITY_MEMCPY_KIND_DTOH:
            return "DtoH";
        case CUPTI_ACTIVITY_MEMCPY_KIND_HTOA:
            return "HtoA";
        case CUPTI_ACTIVITY_MEMCPY_KIND_ATOH:
            return "AtoH";
        case CUPTI_ACTIVITY_MEMCPY_KIND_ATOA:
            return "AtoA";
        case CUPTI_ACTIVITY_MEMCPY_KIND_ATOD:
            return "AtoD";
        case CUPTI_ACTIVITY_MEMCPY_KIND_DTOA:
            return "DtoA";
        case CUPTI_ACTIVITY_MEMCPY_KIND_DTOD:
            return "DtoD";
        case CUPTI_ACTIVITY_MEMCPY_KIND_HTOH:
            return "HtoH";
        case CUPTI_ACTIVITY_MEMCPY_KIND_PTOP:
            return "PtoP";
        default:
            return "<unknown>";
    }
}


static const char *
GetChannelType(
    CUpti_ChannelType channelType)
{
    switch (channelType)
    {
        case CUPTI_CHANNEL_TYPE_INVALID:
            return "INVALID";
        case CUPTI_CHANNEL_TYPE_COMPUTE:
            return "COMPUTE";
        case CUPTI_CHANNEL_TYPE_ASYNC_MEMCPY:
            return "ASYNC_MEMCPY";
        default:
            return "<unknown>";
    }
}

void PrintActivity(CUpti_Activity *pRecord)
{
  CUpti_ActivityKind activityKind = pRecord->kind;

    switch (activityKind)
    {
        case CUPTI_ACTIVITY_KIND_MEMCPY:
        {
            CUpti_ActivityMemcpy5 *pMemcpyRecord = (CUpti_ActivityMemcpy5 *)pRecord;

            NVLOGI_FMT(TAG, "MEMCPY \"{}\" [ {}, {} ] duration {}, size {}, srcKind {}, dstKind {}, correlationId {},"
                    "deviceId {}, contextId {}, streamId {}, graphId {}, graphNodeId {}, channelId {}, channelType {}",
                    GetMemcpyKindString((CUpti_ActivityMemcpyKind)pMemcpyRecord->copyKind),
                    (unsigned long long)pMemcpyRecord->start,
                    (unsigned long long)pMemcpyRecord->end,
                    (unsigned long long)(pMemcpyRecord->end - pMemcpyRecord->start),
                    (unsigned long long)pMemcpyRecord->bytes,
                    GetMemoryKindString((CUpti_ActivityMemoryKind)pMemcpyRecord->srcKind),
                    GetMemoryKindString((CUpti_ActivityMemoryKind)pMemcpyRecord->dstKind),
                    static_cast<uint32_t>(pMemcpyRecord->correlationId),
                    static_cast<uint32_t>(pMemcpyRecord->deviceId),
                    static_cast<uint32_t>(pMemcpyRecord->contextId),
                    static_cast<uint32_t>(pMemcpyRecord->streamId),
                    static_cast<uint32_t>(pMemcpyRecord->graphId),
                    (unsigned long long)pMemcpyRecord->graphNodeId,
                    static_cast<uint32_t>(pMemcpyRecord->channelID),
                    GetChannelType(pMemcpyRecord->channelType));

            break;
        }

        case CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION:
        {
            CUpti_ActivityExternalCorrelation *pExternalCorrelationRecord = (CUpti_ActivityExternalCorrelation *)pRecord;

            NVLOGI_FMT(TAG, "EXTERNAL_CORRELATION externalKind {}, correlationId {}, externalId {}",
                    static_cast<uint32_t>(pExternalCorrelationRecord->externalKind),
                    static_cast<uint32_t>(pExternalCorrelationRecord->correlationId),
                    static_cast<uint64_t>(pExternalCorrelationRecord->externalId));

            break;
        }

        case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL:
        {
            CUpti_ActivityKernel8 *pKernelRecord = (CUpti_ActivityKernel8 *)pRecord;

            NVLOGI_FMT(TAG, "CONCURRENT_KERNEL [ {}, {} ] duration {}, \"{}\", correlationId {}, "
                    "grid [ {}, {}, {} ], block [ {}, {}, {} ], sharedMemory (static {}, dynamic {}), "
                    "deviceId {}, contextId {}, streamId {}, graphId {}, graphNodeId {}, channelId {}",
                    static_cast<uint64_t>(pKernelRecord->start),
                    static_cast<uint64_t>(pKernelRecord->end),
                    static_cast<int64_t>(pKernelRecord->end - pKernelRecord->start),
                    GetName(pKernelRecord->name),
                    static_cast<uint32_t>(pKernelRecord->correlationId),
                    static_cast<int32_t>(pKernelRecord->gridX),
                    static_cast<int32_t>(pKernelRecord->gridY),
                    static_cast<int32_t>(pKernelRecord->gridZ),
                    static_cast<int32_t>(pKernelRecord->blockX),
                    static_cast<int32_t>(pKernelRecord->blockY),
                    static_cast<int32_t>(pKernelRecord->blockZ),
                    static_cast<int32_t>(pKernelRecord->staticSharedMemory),
                    static_cast<int32_t>(pKernelRecord->dynamicSharedMemory),
                    static_cast<uint32_t>(pKernelRecord->deviceId),
                    static_cast<uint32_t>(pKernelRecord->contextId),
                    static_cast<uint32_t>(pKernelRecord->streamId),
                    static_cast<uint32_t>(pKernelRecord->graphId),
                    static_cast<uint64_t>(pKernelRecord->graphNodeId),
                    static_cast<uint32_t>(pKernelRecord->channelID));

            break;
        }

#if 0
        case CUPTI_ACTIVITY_KIND_RUNTIME:
        {
            // intentionally don't log this type
            NVLOGI_FMT(TAG, "CUPTI Activity CUPTI_ACTIVITY_KIND_RUNTIME not logging any info intentionally");
            break;
        }
#else
        case CUPTI_ACTIVITY_KIND_DRIVER:
        case CUPTI_ACTIVITY_KIND_RUNTIME:
        case CUPTI_ACTIVITY_KIND_INTERNAL_LAUNCH_API:
        {
            //Log all driver/runtime API calls that are enabled (should only be kernel launches and graph launches)

            CUpti_ActivityAPI *pApiRecord = (CUpti_ActivityAPI *)pRecord;
            const char* pName = NULL;
            const char* activity = NULL;

            if (pApiRecord->kind == CUPTI_ACTIVITY_KIND_DRIVER)
            {
                cuptiGetCallbackName(CUPTI_CB_DOMAIN_DRIVER_API, pApiRecord->cbid, &pName);
                activity = "CUPTI_ACTIVITY_KIND_DRIVER";
            }
            else if (pApiRecord->kind == CUPTI_ACTIVITY_KIND_RUNTIME)
            {
                cuptiGetCallbackName(CUPTI_CB_DOMAIN_RUNTIME_API, pApiRecord->cbid, &pName);
                activity = "CUPTI_ACTIVITY_KIND_RUNTIME";
            }
            else
            {
                activity = "CUPTI_ACTIVITY_KIND_INTERNAL_LAUNCH_API";
            }

            NVLOGI_FMT(TAG, "{} [ {}, {} ] duration {}, \"{}\", cbid {}, processId {}, threadId {}, correlationId {}",
                    activity,
                    (unsigned long long)pApiRecord->start,
                    (unsigned long long)pApiRecord->end,
                    (unsigned long long)(pApiRecord->end - pApiRecord->start),
                    GetName(pName),
                    static_cast<uint32_t>(pApiRecord->cbid),
                    static_cast<uint32_t>(pApiRecord->processId),
                    static_cast<uint32_t>(pApiRecord->threadId),
                    static_cast<uint32_t>(pApiRecord->correlationId));

            break;
        }
#endif

        case CUPTI_ACTIVITY_KIND_MEMORY:
        {
            CUpti_ActivityMemory *pMemoryRecord = (CUpti_ActivityMemory *)(void *)pRecord;

            NVLOGI_FMT(TAG, "MEMORY [ {}, {} ] duration {}, size {} bytes, address {}, memoryKind {}, deviceId {}, contextId {}, processId {}",
                    (unsigned long long)pMemoryRecord->start,
                    (unsigned long long)pMemoryRecord->end,
                    (unsigned long long)(pMemoryRecord->end - pMemoryRecord->start),
                    (unsigned long long)pMemoryRecord->bytes,
                    (unsigned long long)pMemoryRecord->address,
                    GetMemoryKindString(pMemoryRecord->memoryKind),
                    static_cast<uint32_t>(pMemoryRecord->deviceId),
                    static_cast<uint32_t>(pMemoryRecord->contextId),
                    static_cast<uint32_t>(pMemoryRecord->processId));

            break;
        }

        case CUPTI_ACTIVITY_KIND_MEMORY2:
        {
            CUpti_ActivityMemory4 *pMemory4Record = (CUpti_ActivityMemory4 *)(void *)pRecord;

            NVLOGI_FMT(TAG, "MEMORY2 {} timestamp {}, size {} bytes, address {}, memoryKind {}, name \"{}\", isAsync {}, streamId {}, deviceId {}, contextId {}, processId {}, correlationId {}",
                    GetMemoryOperationTypeString(pMemory4Record->memoryOperationType),
                    (unsigned long long)pMemory4Record->timestamp,
                    (unsigned long long)pMemory4Record->bytes,
                    (unsigned long long)pMemory4Record->address,
                    GetMemoryKindString(pMemory4Record->memoryKind),
                    GetName(pMemory4Record->name),
                    static_cast<uint32_t>(pMemory4Record->isAsync),
                    static_cast<uint32_t>(pMemory4Record->streamId),
                    static_cast<uint32_t>(pMemory4Record->deviceId),
                    static_cast<uint32_t>(pMemory4Record->contextId),
                    static_cast<uint32_t>(pMemory4Record->processId),
                    static_cast<uint32_t>(pMemory4Record->correlationId));

            break;
        }

        case CUPTI_ACTIVITY_KIND_GRAPH_TRACE:
        {
            CUpti_ActivityGraphTrace2 *pGraphTraceRecord = (CUpti_ActivityGraphTrace2 *)(void *)pRecord;

            NVLOGI_FMT(TAG, "GRAPH_TRACE [ {}, {} ] duration {}, graphId {}, correlationId {}, streamId {}, deviceId [{}, {}], contextId [{}, {}]",
                    (unsigned long long)pGraphTraceRecord->start,
                    (unsigned long long)pGraphTraceRecord->end,
                    (unsigned long long)(pGraphTraceRecord->end - pGraphTraceRecord->start),
                    static_cast<uint32_t>(pGraphTraceRecord->graphId),
                    static_cast<uint32_t>(pGraphTraceRecord->correlationId),
                    static_cast<uint32_t>(pGraphTraceRecord->streamId),
                    static_cast<uint32_t>(pGraphTraceRecord->deviceId),
                    static_cast<uint32_t>(pGraphTraceRecord->endDeviceId),
                    static_cast<uint32_t>(pGraphTraceRecord->contextId),
                    static_cast<uint32_t>(pGraphTraceRecord->endContextId));

            break;
        }

        case CUPTI_ACTIVITY_KIND_DEVICE_GRAPH_TRACE:
        {
            CUpti_ActivityDeviceGraphTrace *pDeviceGraphRecord = (CUpti_ActivityDeviceGraphTrace *)(void *)pRecord;

            NVLOGI_FMT(TAG, "DEVICE_GRAPH_TRACE [DEVICE-LAUNCHED] [ {}, {} ] duration {}, graphId {}, streamId {}, deviceId {}, contextId {}",
                    (unsigned long long)pDeviceGraphRecord->start,
                    (unsigned long long)pDeviceGraphRecord->end,
                    (unsigned long long)(pDeviceGraphRecord->end - pDeviceGraphRecord->start),
                    static_cast<uint32_t>(pDeviceGraphRecord->graphId),
                    static_cast<uint32_t>(pDeviceGraphRecord->streamId),
                    static_cast<uint32_t>(pDeviceGraphRecord->deviceId),
                    static_cast<uint32_t>(pDeviceGraphRecord->contextId));

            break;
        }

        case CUPTI_ACTIVITY_KIND_DEVICE:
        {
            CUpti_ActivityDevice5 *pDeviceRecord = (CUpti_ActivityDevice5 *)(void *)pRecord;

            NVLOGI_FMT(TAG, "DEVICE id {}, computeCapability {}.{}, name \"{}\"",
                    static_cast<uint32_t>(pDeviceRecord->id),
                    static_cast<uint32_t>(pDeviceRecord->computeCapabilityMajor),
                    static_cast<uint32_t>(pDeviceRecord->computeCapabilityMinor),
                    GetName(pDeviceRecord->name));

            break;
        }

        default:
            NVLOGW_FMT(TAG, "CUPTI Activity {} printing not implemented",static_cast<uint32_t>(activityKind));
            break;
    }
}

static std::mutex g_cupti_helper_mutex;
static std::vector<uint8_t*> pBufferEmpty;
static std::vector<uint8_t*> pBufferReady;
static std::vector<size_t> bufferReadyValidSize;
static std::thread* cupti_polling_thread;
static bool cupti_polling_thread_done = false;
static bool cupti_initialized = false;

// Buffer Management Functions
static void CUPTIAPI
BufferRequested(
    uint8_t **ppBuffer,
    size_t *pSize,
    size_t *pMaxNumRecords)
{
    static int request_count = 0;
    request_count++;
    NVLOGI_FMT(TAG,"BufferRequested [#{}] entered", request_count);
    const std::lock_guard<std::mutex> lock(g_cupti_helper_mutex);

    // This callback will be called from whatever thread made the CUDA API call


    // use previously allocated buffer to speedup high priority thread
    uint8_t *pBuffer = nullptr;
    int buf_idx = -1;
    for (int k=0; k<g_cupti_num_buffers; k++)
    {
        if (pBufferEmpty[k] != nullptr)
        {
            pBuffer = pBufferEmpty[k];
            pBufferEmpty[k] = nullptr;
            buf_idx = k;
            break; // out of for loop
        }
    }
    if (pBuffer == nullptr)
    {
        NVLOGF_FMT(TAG,AERIAL_MEMORY_EVENT,"Unable to find pre-allocated cupti buffer");
    }

    *pSize = g_cupti_buffer_size;
    *ppBuffer = pBuffer;
    *pMaxNumRecords = 0;
    NVLOGI_FMT(TAG,"BufferRequested [#{}]: pBuffer={} [{}], size={}, maxNumRecords={}",
               request_count,
               reinterpret_cast<void*>(pBuffer),
               buf_idx,
               *pSize,
               *pMaxNumRecords);
}


void PrintActivityBuffer(uint8_t *pBuffer, size_t validBytes)
{
    CUpti_Activity *pRecord = NULL;
    CUptiResult status = CUPTI_SUCCESS;

    static uint64_t total_records = 0;
    static uint64_t external_correlation_records = 0;
    
    // Rate limiting: max prints per 100ms window
    static constexpr uint64_t MAX_PRINTS_PER_WINDOW = MAX_ACTIVITY_PRINTS_PER_SECOND / 10;
    static constexpr std::chrono::milliseconds WINDOW_DURATION{100};
    
    auto window_start = std::chrono::system_clock::now();
    uint64_t prints_in_current_window = 0;
    
    // Console logging: start after 1 second, then every second
    const auto buffer_start = std::chrono::system_clock::now();
    auto last_console_log = buffer_start;
    static constexpr std::chrono::seconds CONSOLE_LOG_INTERVAL{1};
    static constexpr std::chrono::seconds CONSOLE_LOG_START_DELAY{1};
    uint64_t records_in_current_buffer = 0;
    
    do {
        status = cuptiActivityGetNextRecord(pBuffer, validBytes, &pRecord);
        if (status == CUPTI_SUCCESS)
        {
            total_records++;
            records_in_current_buffer++;
            if (pRecord->kind == CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION)
            {
                external_correlation_records++;
            }
            
            // Rate limiting: check if we need to wait for next window
            if (prints_in_current_window >= MAX_PRINTS_PER_WINDOW)
            {
                const auto now = std::chrono::system_clock::now();
                const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - window_start);
                
                if (elapsed < WINDOW_DURATION)
                {
                    // Sleep until next window starts
                    const auto sleep_duration = WINDOW_DURATION - elapsed;
                    std::this_thread::sleep_for(sleep_duration);
                }
                
                // Start new window
                window_start = std::chrono::system_clock::now();
                prints_in_current_window = 0;
            }
            
            PrintActivity(pRecord);
            prints_in_current_window++;
            
            // Periodic console logging
            const auto now = std::chrono::system_clock::now();
            const auto elapsed_since_start = std::chrono::duration_cast<std::chrono::seconds>(now - buffer_start);
            const auto elapsed_since_last_log = std::chrono::duration_cast<std::chrono::seconds>(now - last_console_log);
            
            if (elapsed_since_start >= CONSOLE_LOG_START_DELAY && elapsed_since_last_log >= CONSOLE_LOG_INTERVAL)
            {
                NVLOGC_FMT(TAG, "PrintActivityBuffer processing: {} records processed, {} seconds elapsed", 
                           records_in_current_buffer, elapsed_since_start.count());
                last_console_log = now;
            }
        }
        else if (status == CUPTI_ERROR_MAX_LIMIT_REACHED)
        {
            //This is the normal condition that exits the loop
            break;
        }
        else
        {
            CUPTI_API_CALL(status);
        }
    } while (1);
    
    NVLOGI_FMT(TAG, "PrintActivityBuffer stats: total_records={}, external_correlation_records={}", 
               total_records, external_correlation_records);
}


static void CUPTIAPI
BufferCompleted(
    CUcontext context,
    uint32_t streamId,
    uint8_t *pBuffer,
    size_t size,
    size_t validSize)
{
    static int first_time = 1;
    static int completion_count = 0;
    if (first_time)
    {
        pthread_t my_pthread = pthread_self();
        pthread_setname_np(my_pthread,"cupti_cb");
        nvlog_fmtlog_thread_init("cupti_cb");
        first_time = 0;
    }

    completion_count++;
    
    // Check for dropped records
    size_t dropped = 0;
    CUPTI_API_CALL(cuptiActivityGetNumDroppedRecords(context, streamId, &dropped));
    if (dropped > 0)
    {
        NVLOGW_FMT(TAG, "WARNING: {} CUPTI activity records were DROPPED!", dropped);
    }

    // Place buffer on ready list
    bool bufferMoved = false;
    int buf_idx = -1;
    {
        const std::lock_guard<std::mutex> lock(g_cupti_helper_mutex);
        for (int k=0; k<g_cupti_num_buffers; k++)
        {
            if (pBufferReady[k] == nullptr)
            {
                pBufferReady[k] = pBuffer;
                bufferReadyValidSize[k] = validSize;
                bufferMoved = true;
                buf_idx = k;
                break; //out of for loop
            }
        }
    }
    
    NVLOGI_FMT(TAG, "BufferCompleted [#{}]: pBuffer={} [{}], size={} validSize={} ({}% full), dropped={}",
               completion_count,
               reinterpret_cast<void*>(pBuffer),
               buf_idx,
               size,
               validSize,
               (validSize * 100) / size,
               dropped);

    if (!bufferMoved)
    {
        NVLOGF_FMT(TAG,AERIAL_MEMORY_EVENT,"Unable to place completed buffer in ready queue - all slots full!");
    }
}

static CUpti_SubscriberHandle subscriber;
static void CuptiCallbackHandler(void* pUserData, CUpti_CallbackDomain domain, CUpti_CallbackId callbackId, const void *pCallbackData)
{
    NVLOGI_FMT(TAG,"CuptiCallbackHandler: pUserData={} domain={} callbackId={}",pUserData,static_cast<uint32_t>(domain),static_cast<uint32_t>(callbackId));
}

void printReadyBuffers()
{
    for (int k=0; k<g_cupti_num_buffers; k++)
    {
        //Check if this buffer is ready to be printed
        uint8_t *pBuffer = nullptr;
        size_t validSize {0};
        if (pBufferReady[k] != nullptr)
        {
            const std::lock_guard<std::mutex> lock(g_cupti_helper_mutex);
            pBuffer = pBufferReady[k];
            validSize = bufferReadyValidSize[k];
            bufferReadyValidSize[k] = 0;
            pBufferReady[k] = nullptr;
        }

        //Print the buffer if it is ready to be printed
        if (validSize > 0)
        {
            NVLOGI_FMT(TAG,"PrintActivityBuffer: pBuffer={} [{}], validSize={}",reinterpret_cast<void*>(pBuffer),k,validSize);
            PrintActivityBuffer(pBuffer, validSize);
        }

        //Clear the buffer if it is ready to be printed
        if (pBuffer)
        {
            
            //Clear the buffer
            memset(pBuffer,0,g_cupti_buffer_size);

            //Search empty list (list of buffers that are not in use) for a place to put the buffer
            {
                const std::lock_guard<std::mutex> lock(g_cupti_helper_mutex);
                int buf_idx = -1;
                for (int n=0; n<g_cupti_num_buffers; n++)
                {
                    if (pBufferEmpty[n] == nullptr)
                    {
                        pBufferEmpty[n] = pBuffer;
                        buf_idx = n;
                        break; // out of for loop
                    }
                }

                //Warn if there is no place to put the buffer
                if (buf_idx == -1)
                {
                    NVLOGF_FMT(TAG,AERIAL_MEMORY_EVENT,"Unable to place activity buffer on empty list");
                }
            }

            
        }
    }
}


#if defined(__cplusplus)
extern "C" {
#endif /* defined(__cplusplus) */

void cupti_stats_polling_worker(void* unused)
{
    nvlog_fmtlog_thread_init("cupti_stats");
    while (cupti_polling_thread_done == false)
    {
        printReadyBuffers();
    }
    printReadyBuffers();
}

void launch_cupti_stats_polling_worker(int32_t cpu_core)
{
    cupti_polling_thread = new std::thread(cupti_stats_polling_worker, nullptr);
    if(cpu_core >= 0)
    {
        NVLOGI_FMT(TAG, "Initializing cupti stats polling thread on core {}", cpu_core);
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(cpu_core, &cpuset);

        auto ret = pthread_setaffinity_np(cupti_polling_thread->native_handle(), sizeof(cpuset), &cpuset);
        if(ret)
        {
            NVLOGF_FMT(TAG, AERIAL_THREAD_API_EVENT, "Failed to set affinity for cupti stats polling thread: ret={}",ret);
        }
    }

    auto ret = pthread_setname_np(cupti_polling_thread->native_handle(), "cupti_stats");
    if(ret != 0)
    {
        NVLOGF_FMT(TAG, AERIAL_THREAD_API_EVENT, "Failed to set cupti_stats_polling_worker thread name: ret={}",ret);
    }
}

void cuphy_cupti_helper_init(uint64_t buffer_size, uint16_t num_buffers)
{
    const std::lock_guard<std::mutex> lock(g_cupti_helper_mutex);

    // Validate parameters
    if (buffer_size == 0) {
        NVLOGF_FMT(TAG, AERIAL_CONFIG_EVENT, "Invalid CUPTI buffer_size: 0");
        exit(EXIT_FAILURE);
    }
    if (num_buffers == 0) {
        NVLOGF_FMT(TAG, AERIAL_CONFIG_EVENT, "Invalid CUPTI num_buffers: 0");
        exit(EXIT_FAILURE);
    }

    g_cupti_buffer_size = buffer_size;
    g_cupti_num_buffers = num_buffers;

    // Resize buffer tracking vectors
    pBufferEmpty.resize(num_buffers, nullptr);
    pBufferReady.resize(num_buffers, nullptr);
    bufferReadyValidSize.resize(num_buffers, 0);

    NVLOGI_FMT(TAG, "CUPTI init with buffer_size={} bytes, num_buffers={}", g_cupti_buffer_size, g_cupti_num_buffers);

    // allocate initial buffers (in the low priority thread)
    for (int k=0; k<g_cupti_num_buffers; k++)
    {
        uint8_t *pBuffer;
        CHECK_CUDA(cudaHostAlloc(reinterpret_cast<void**>(&pBuffer), g_cupti_buffer_size+ALIGN_SIZE, cudaHostAllocPortable));
        if (pBuffer == nullptr)
        {
            NVLOGF_FMT(TAG,AERIAL_MEMORY_EVENT,"Unable to allocate initial cupti buffer");
        }
        pBufferEmpty[k] = ALIGN_BUFFER(pBuffer, ALIGN_SIZE);
        memset(pBufferEmpty[k],0,g_cupti_buffer_size);
    }

    if (1)
    {
        size_t valueSize = sizeof(uint8_t);
        uint8_t value = 1;
        CUPTI_API_CALL(cuptiActivitySetAttribute(CUPTI_ACTIVITY_ATTR_ZEROED_OUT_ACTIVITY_BUFFER, &valueSize, (void *)&value));
    }

    if (1)
    {
        size_t valueSize = sizeof(size_t);
        size_t value = 0;

        value=10;
        CUPTI_API_CALL(cuptiActivitySetAttribute(CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_PRE_ALLOCATE_VALUE, &valueSize, (void *)&value));

        value=16*1024*1024;
        CUPTI_API_CALL(cuptiActivitySetAttribute(CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_SIZE, &valueSize, (void *)&value));

        CUPTI_API_CALL(cuptiActivityGetAttribute(CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_SIZE, &valueSize, (void *)&value));
        printf("CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_SIZE: %ld\n",value);

        CUPTI_API_CALL(cuptiActivityGetAttribute(CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_POOL_LIMIT, &valueSize, (void *)&value));
        printf("CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_POOL_LIMIT: %ld\n",value);

        CUPTI_API_CALL(cuptiActivityGetAttribute(CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_PRE_ALLOCATE_VALUE, &valueSize, (void *)&value));
        printf("CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_PRE_ALLOCATE_VALUE: %ld\n",value);
    }

    CUPTI_API_CALL(cuptiSubscribe(&subscriber, CuptiCallbackHandler, NULL));
    
    // Runtime CUPTI version check
    uint32_t cuptiVersion = 0;
    CUPTI_API_CALL(cuptiGetVersion(&cuptiVersion));
    NVLOGW_FMT(TAG, "CUPTI Runtime Version: {} ({})", cuptiVersion, GetCuptiVersionString(cuptiVersion));
    
#if defined(CUPTI_API_VERSION)
    NVLOGW_FMT(TAG, "CUPTI Compile-Time API Version: {} ({})", 
               static_cast<uint32_t>(CUPTI_API_VERSION), 
               GetCuptiVersionString(static_cast<uint32_t>(CUPTI_API_VERSION)));
#endif
    
    CUPTI_API_CALL(cuptiActivityRegisterCallbacks(BufferRequested, BufferCompleted));
    CUPTI_API_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION));
    CUPTI_API_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_DEVICE));
    CUPTI_API_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
    CUPTI_API_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME));  // Required for EXTERNAL_CORRELATION
    CUPTI_API_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_DRIVER));   // Required for EXTERNAL_CORRELATION (app uses both APIs)
    // CUPTI_API_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_GRAPH_TRACE));
    
    // Enable device graph tracing (CUDA 12.8+)
    // This automatically enables CUPTI_ACTIVITY_KIND_DEVICE_GRAPH_TRACE
    // CUPTI_API_CALL(cuptiActivityEnableDeviceGraph(1));

    //CUPTI_API_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMORY));
    // CUPTI_API_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMORY2));
    //CUPTI_API_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_ENVIRONMENT));
    //CUPTI_API_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONTEXT));
    // CUPTI_API_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY));

    // CUPTI_API_CALL(cuptiActivityFlushPeriod(250)); // time in ms

    //Disable specific activity inherent in Aerial that overwhelms the output
    {
        //Example of disabling cudaEventQuery only
        // CUPTI_API_CALL(cuptiActivityEnableRuntimeApi(CUPTI_RUNTIME_TRACE_CBID_cudaEventQuery_v3020, 0));
        // CUPTI_API_CALL(cuptiActivityEnableDriverApi(CUPTI_DRIVER_TRACE_CBID_cuEventQuery, 0));

        //Disable all runtime API activity
        for (int k=1; k<CUPTI_RUNTIME_TRACE_CBID_SIZE; k++)
        {
            CUPTI_API_CALL(cuptiActivityEnableRuntimeApi(k, 0));
        }
        // Enable all cudaLaunchKernel variants
        CUPTI_API_CALL(cuptiActivityEnableRuntimeApi(CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000, 1));
        CUPTI_API_CALL(cuptiActivityEnableRuntimeApi(CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_ptsz_v7000, 1));
        CUPTI_API_CALL(cuptiActivityEnableRuntimeApi(CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernelExC_v11060, 1));
        CUPTI_API_CALL(cuptiActivityEnableRuntimeApi(CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernelExC_ptsz_v11060, 1));
        // Enable all cudaGraphLaunch variants
        CUPTI_API_CALL(cuptiActivityEnableRuntimeApi(CUPTI_RUNTIME_TRACE_CBID_cudaGraphLaunch_v10000, 1));
        CUPTI_API_CALL(cuptiActivityEnableRuntimeApi(CUPTI_RUNTIME_TRACE_CBID_cudaGraphLaunch_ptsz_v10000, 1));

        //Disable all driver API activity
        for (int k=1; k<CUPTI_DRIVER_TRACE_CBID_SIZE; k++)
        {
            CUPTI_API_CALL(cuptiActivityEnableDriverApi(k, 0));
        }
        // Enable all cuLaunchKernel variants
        CUPTI_API_CALL(cuptiActivityEnableDriverApi(CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel, 1));
        CUPTI_API_CALL(cuptiActivityEnableDriverApi(CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel_ptsz, 1));
        CUPTI_API_CALL(cuptiActivityEnableDriverApi(CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx, 1));
        CUPTI_API_CALL(cuptiActivityEnableDriverApi(CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx_ptsz, 1));
        // Enable all cuGraphLaunch variants
        CUPTI_API_CALL(cuptiActivityEnableDriverApi(CUPTI_DRIVER_TRACE_CBID_cuGraphLaunch, 1));
        CUPTI_API_CALL(cuptiActivityEnableDriverApi(CUPTI_DRIVER_TRACE_CBID_cuGraphLaunch_ptsz, 1));
    }


    NVLOGW_FMT(TAG,"Intialized cuPHY CUPTI");
    launch_cupti_stats_polling_worker(-1);
    cupti_initialized = true;
}

void cuphy_cupti_helper_flush()
{
    CUPTI_API_CALL(cuptiGetLastError());
    CUPTI_API_CALL(cuptiActivityFlushAll(1));
}

void cuphy_cupti_helper_stop()
{
    if (!cupti_initialized) {
        return;
    }
    cuphy_cupti_helper_flush();
    cupti_polling_thread_done = true;
    int ret = pthread_join(cupti_polling_thread->native_handle(), NULL);
    (void)ret; // suppress unused variable warning
}

void cuphy_cupti_helper_push_external_id(uint64_t id)
{
    CUPTI_API_CALL(cuptiActivityPushExternalCorrelationId(CUPTI_EXTERNAL_CORRELATION_KIND_AERIAL, id));
}

void cuphy_cupti_helper_pop_external_id()
{
    uint64_t id{};
    CUPTI_API_CALL(cuptiActivityPopExternalCorrelationId(CUPTI_EXTERNAL_CORRELATION_KIND_AERIAL, &id));
}

#if defined(__cplusplus)
} /* extern "C" */
#endif /* defined(__cplusplus) */