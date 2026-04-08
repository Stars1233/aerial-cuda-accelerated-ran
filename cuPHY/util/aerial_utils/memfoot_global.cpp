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
#include <unistd.h>
#include <cuda_runtime.h>

#include <atomic>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "memfoot_global.h"
#include "memfoot_global.hpp"
#include "nvlog.hpp"

#define TAG (NVLOG_TAG_BASE_UTIL + 1) // "UTIL.MEMFOOT"

#define INT64_ONE_KIB (1024L) // 1 KiB in bytes
#define INT64_ONE_MIB (1024L * 1024L) // 1 MiB in bytes

namespace {
// Define instances of MemFootBase
std::unordered_set<MemFootBase*> cuphy_instance_set;
std::unordered_set<MemFootBase*> phydrv_instance_set;
std::unordered_set<MemFootBase*> other_instance_set;
} // namespace

MemFootBase::MemFootBase(memfoot_module_t module_id)
{
    MemtraceDisableScope md;
    this->module_id = module_id;
    switch (this->module_id)
    {
        case MF_MODULE_CUPHY:
            cuphy_instance_set.insert(this);
            break;
        case MF_MODULE_PHYDRV:
            phydrv_instance_set.insert(this);
            break;
        default:
            other_instance_set.insert(this);
            break;
    }
}

MemFootBase::~MemFootBase()
{
    switch (this->module_id)
    {
        case MF_MODULE_CUPHY:
            cuphy_instance_set.erase(this);
            break;
        case MF_MODULE_PHYDRV:
            phydrv_instance_set.erase(this);
            break;
        default:
            other_instance_set.erase(this);
            break;
    }
}

// Log full file path
#define LOG_FULL_FILE_PATH 1

// For debug, the last tracking code info
std::atomic<const char*> last_file = nullptr; // The last tracking code's caller file name
std::atomic<int> last_log_tag = 0; // The last tracking code's caller log_tag
std::atomic<int> last_line = 0; // The last tracking code's caller line number
std::atomic<const char*> last_caller = nullptr; // The last tracking code's caller function name

// The total actual freed GPU memory size during initialization
static std::atomic<int64_t> total_freed_gpu_size = 0;

// Global memory footprint total size
static std::atomic<int64_t> last_free_size = 0;

static int64_t free_init = 0; // Total GPU free memory size at the first tracking
static int64_t total_init = 0; // Total GPU memory size
static int64_t system_used = 0; // GPU memory size used by system and other applications

// Only call memory tracking during initialization
static std::atomic<int> initiated_finished = 0;

// Mutex for GPU allocation, enable during initialization and disable after printed the memory info
static bool enable_gpu_alloc_mutex = true;
static std::mutex gpu_alloc_mutex;
static void gpu_alloc_mutex_lock()
{
    if(enable_gpu_alloc_mutex)
    {
        gpu_alloc_mutex.lock();
    }
}

static void gpu_alloc_mutex_unlock()
{
    if(enable_gpu_alloc_mutex)
    {
        gpu_alloc_mutex.unlock();
    }
}

// Per module total memfoot info

typedef struct module_info {
    memfoot_module_t module_id;
    char module_name[32];
    std::atomic<int64_t> module_gpu_size;
} module_info_t;

#define MEMFOOT_INFO_PER_MODULE_INIT(module_id, module_name) {module_id, module_name, 0}
static module_info_t module_info_array[MF_MODULE_MAX_NUM] = {
    MEMFOOT_INFO_PER_MODULE_INIT(MF_MODULE_CUPHY, "CUPHY"),
    MEMFOOT_INFO_PER_MODULE_INIT(MF_MODULE_PHYDRV, "PHYDRV"),
    MEMFOOT_INFO_PER_MODULE_INIT(MF_MODULE_FH, "FH"),
    MEMFOOT_INFO_PER_MODULE_INIT(MF_MODULE_OTHER, "OTHER"),
};

// Per tag memfoot info
typedef struct tag_info {
    memfoot_tag_t tag_id;
    memfoot_module_t module_id;
    char tag_name[32];
    std::atomic<int64_t> gpu_alloc_size;
} tag_info_t;

#define MEMFOOT_INFO_PER_TAG_INIT(tag_id, module_id, tag_name) {tag_id, module_id, tag_name, 0}
static tag_info_t tag_info_array[MF_TAG_MAX_NUM] = {
    MEMFOOT_INFO_PER_TAG_INIT(MF_TAG_CUPHY_OTHER, MF_MODULE_CUPHY, "CUPHY-Other"),
    MEMFOOT_INFO_PER_TAG_INIT(MF_TAG_PHYDRV_OTHER, MF_MODULE_PHYDRV, "PHYDRV-Other"),
    MEMFOOT_INFO_PER_TAG_INIT(MF_TAG_FH_DOCA_RX, MF_MODULE_FH, "fh_doca_rx"),
    MEMFOOT_INFO_PER_TAG_INIT(MF_TAG_FH_DOCA_TX, MF_MODULE_FH, "fh_doca_tx"),
    MEMFOOT_INFO_PER_TAG_INIT(MF_TAG_FH_GPU_COMM, MF_MODULE_FH, "fh_gpu_comm"),
    MEMFOOT_INFO_PER_TAG_INIT(MF_TAG_FH_PEER, MF_MODULE_FH, "fh_peer"),
    MEMFOOT_INFO_PER_TAG_INIT(MF_TAG_OTHER, MF_MODULE_OTHER, "other"),
};

// Global grouped memfoot info
typedef struct global_info {
    char group_name[32];
    std::atomic<int64_t> gpu_alloc_size;
    std::atomic<int64_t> cpu_alloc_size;
    std::atomic<int64_t> gpu_overhead;
    std::atomic<int64_t> gpu_implicit;
} global_info_t;

typedef enum {
    MF_GROUP_CUPHY = 0,
    MF_GROUP_PHYDRV = 1,
    MF_GROUP_FH = 2,
    MF_GROUP_CUDA_CTX = 3,
    MF_GROUP_OTHER = 4,
    MF_GROUP_MAX_NUM = 5,
} memfoot_global_group_t;

#define MEMFOOT_INFO_INIT(group_name) {group_name, 0, 0, 0, 0}
static global_info_t global_info_array[MF_GROUP_MAX_NUM] = {
    MEMFOOT_INFO_INIT("cuPHY"),
    MEMFOOT_INFO_INIT("cuphydriver"),
    MEMFOOT_INFO_INIT("aerial-fh-driver"),
    MEMFOOT_INFO_INIT("cuda_context"),
    MEMFOOT_INFO_INIT("other")
};

static std::string get_size_string(int64_t size)
{
    if(size < INT64_ONE_KIB && size > -INT64_ONE_KIB)
    {
        return std::to_string(size) + " Bytes";
    }
    else if(size < INT64_ONE_MIB && size > -INT64_ONE_MIB)
    {
        return std::to_string(size / INT64_ONE_KIB) + " KiB";
    }
    else
    {
        return std::to_string(size / INT64_ONE_MIB) + " MiB";
    }
}

static const char* get_filename(const char* path) {
    if (LOG_FULL_FILE_PATH)
    {
        return path;
    }
    else
    {
        const char* filename = strrchr(path, '/');
        return filename ? filename + 1 : path;
    }
}

// Per tag memfoot size add
void memfoot_add_gpu_size(memfoot_tag_t tag_id, int64_t size)
{
    if (tag_id < 0 || tag_id >= MF_TAG_MAX_NUM)
    {
        NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "Invalid tag_id: {}", static_cast<int>(tag_id));
        return;
    }

    tag_info_t& tag_info = tag_info_array[tag_id];
    memfoot_module_t module_id = tag_info.module_id;
    if (module_id < 0 || module_id >= MF_MODULE_MAX_NUM)
    {
        NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "Invalid module_id: {}", static_cast<int>(module_id));
        return;
    }

    tag_info.gpu_alloc_size.fetch_add(size);
    module_info_array[module_id].module_gpu_size.fetch_add(size);
    NVLOGD_FMT(TAG, "{}: module_id={} tag_id={} size={}={}MiB", __func__,
        static_cast<int>(module_id), static_cast<int>(tag_id), size, size / INT64_ONE_MIB);
}

// GPU size check
void memfoot_global_gpu_size_check(const char* file, int line, int log_tag, const char* func_caller)
{
    memfoot_global_get_mem_info(file, line, log_tag, func_caller);

    // Unlock the GPU allocation mutex
    gpu_alloc_mutex_unlock();
}

// Get system free memory size and update the last free size
void memfoot_global_get_mem_info(const char* file, int line, int log_tag, const char* func_caller)
{
    if (initiated_finished.load())
    {
        // Do not track memory after the initialization is finished
        return;
    }

    gpu_alloc_mutex_lock();

    size_t free_before, total_before;
    if(cudaMemGetInfo(&free_before, &total_before) != cudaSuccess)
    {
        NVLOGE_FMT(TAG, AERIAL_CUDA_API_EVENT, "cudaMemGetInfo failed");
    }
    size_t last_free = last_free_size.load();
    if(last_free == 0)
    {
        // The first time allocation
        free_init = free_before;
        total_init = total_before;
        last_free_size.fetch_add(free_before);
        system_used = total_before - free_before;
        NVLOGI_FMT(TAG, "INIT: total={}={}MiB free={}={}MiB system_used={}={}MiB",
                total_init, total_init / INT64_ONE_MIB, free_init, free_init / INT64_ONE_MIB, system_used, system_used / INT64_ONE_MIB);
        last_free = free_before;
    }

    NVLOGD_FMT(TAG, "CHECK: {} +{} log_tag={} func={} last_free_size={}", get_filename(file), line, nvlog_get_component_name(log_tag), func_caller, last_free);

    int64_t untracked = last_free - free_before;
    if (untracked != 0)
    {
        NVLOGI_FMT(TAG, "MISSED: last:[{} +{} log_tag={} func={} free_size={}] curr:[{} +{} log_tag={} func={} free_size={}] untracked={}",
            get_filename(last_file.load()), last_line.load(), nvlog_get_component_name(last_log_tag.load()), last_caller.load(), last_free,
            get_filename(file), line, nvlog_get_component_name(log_tag), func_caller, free_before, untracked);

        last_free_size.fetch_sub(untracked);
    }

    last_file.store(file);
    last_line.store(line);
    last_caller.store(func_caller);
    last_log_tag.store(log_tag);
}

// Get the global grouped index by the source file path or log_tag ID
static memfoot_global_group_t get_group_index(const char* file, int log_tag)
{
    if(log_tag > 0)
    {
        // Map log_tag ranges to module groups based on NVLOG_TAG_BASE values
        if(log_tag >= NVLOG_TAG_BASE_CUPHY_DRIVER && log_tag < NVLOG_TAG_BASE_L2_ADAPTER)
        {
            // NVLOG_TAG_BASE_CUPHY_DRIVER (200) to NVLOG_TAG_BASE_L2_ADAPTER (300)
            return MF_GROUP_PHYDRV;
        }
        else if(log_tag >= NVLOG_TAG_BASE_CUPHY && log_tag < NVLOG_TAG_BASE_TESTBENCH)
        {
            // NVLOG_TAG_BASE_CUPHY (900) to NVLOG_TAG_BASE_TESTBENCH (1000)
            return MF_GROUP_CUPHY;
        }
        else if(log_tag >= NVLOG_TAG_BASE_FH_DRIVER && log_tag < NVLOG_TAG_BASE_COMPRESSION)
        {
            // NVLOG_TAG_BASE_FH_DRIVER (600) to NVLOG_TAG_BASE_COMPRESSION (700)
            return MF_GROUP_FH;
        }
        else
        {
            // Default to OTHER for unknown tags or fallback to file-based detection
            return MF_GROUP_OTHER;
        }
    }

    if(file == nullptr)
    {
        return MF_GROUP_OTHER;
    }

    if(strstr(file, "cuPHY/") != nullptr)
    {
        return MF_GROUP_CUPHY;
    }
    else if(strstr(file, "cuPHY-CP/cuphydriver") != nullptr)
    {
        return MF_GROUP_PHYDRV;
    }
    else if(strstr(file, "cuPHY-CP/aerial-fh-driver") != nullptr)
    {
        return MF_GROUP_FH;
    }
    else
    {
        return MF_GROUP_OTHER;
    }
}

// Add GPU size to the global grouped memfoot info
int memfoot_global_gpu_size_add(const char* file, int line, int log_tag, const char* func_caller, const char* func_alloc, size_t alloc_size, int err_code)
{
    if (initiated_finished.load())
    {
        // Do not track memory after the initialization is finished
        return 0;
    }

    const char* file_name = get_filename(file);

    size_t free_after, total_after;
    size_t free_before = last_free_size.load();

    if(cudaMemGetInfo(&free_after, &total_after) != cudaSuccess)
    {
        NVLOGE_FMT(TAG, AERIAL_CUDA_API_EVENT, "cudaMemGetInfo failed");
    }

    int64_t used = free_before - free_after;
    int64_t diff = used - alloc_size;
    int64_t implicit = 0;

    if (used != 0)
    {
        // Update the last free size
        last_free_size.fetch_sub(used);
    }

    // Add size to the group's device memory size
    memfoot_global_group_t group_index = MF_GROUP_MAX_NUM;
    if(strstr(func_alloc, "CtxCreate") != nullptr)
    {
        // Match functions: cuCtxCreate, cuCtxCreate_v3, cuGreenCtxCreate. They allocate device memory implicitly
        implicit = used;
        global_info_array[MF_GROUP_CUDA_CTX].gpu_implicit.fetch_add(implicit);
    }
    else if(strstr(func_alloc, "GraphInstantiate") != nullptr)
    {
        // Match functions: cuGraphInstantiate, cudaGraphInstantiate. They allocate device memory implicitly
        implicit = used;
        global_info_array[MF_GROUP_OTHER].gpu_implicit.fetch_add(implicit);
    }
    else if(strstr(func_alloc, "cudaStreamCreate") != nullptr)
    {
        // Match functions: cudaStreamCreateWithFlags, cudaStreamCreateWithPriority. They may allocate device memory implicitly
        implicit = used;
        global_info_array[MF_GROUP_OTHER].gpu_implicit.fetch_add(implicit);
    }
    else if(strstr(func_alloc, "doca_ctx_start") != nullptr)
    {
        // Match function: doca_ctx_start. It allocates device memory implicitly
        implicit = used;
        global_info_array[MF_GROUP_OTHER].gpu_implicit.fetch_add(implicit);
    }
    else
    {
        // Other functions explicitly allocated memory, assign group by log_tag if available, otherwise by source file path
        group_index = get_group_index(file, log_tag);
        global_info_array[group_index].gpu_alloc_size.fetch_add(alloc_size);
        global_info_array[group_index].gpu_overhead.fetch_add(diff);
    }

    NVLOGI_FMT(TAG, "GPU: {} +{} log_tag={} func={} {} free_before={} free_after={} alloc={} implicit={} used={} diff={}",
            file_name, line, nvlog_get_component_name(log_tag), func_caller, func_alloc, free_before, free_after, alloc_size, implicit, used, diff);

    last_file.store(file);
    last_line.store(line);
    last_caller.store(func_caller);
    last_log_tag.store(log_tag);

    // Unlock the GPU allocation mutex
    gpu_alloc_mutex_unlock();

    return err_code;
}

// Add CPU size to the global grouped memfoot info
int memfoot_global_cpu_size_add(const char* file, int line, int log_tag, const char* func_caller, const char* func_alloc, size_t alloc_size, int err_code)
{
    if (initiated_finished.load())
    {
        // Do not track memory after the initialization is finished
        return 0;
    }

    const char* file_name = get_filename(file);

    // Assign group by log_tag if available, otherwise by source file path
    memfoot_global_group_t group_index = get_group_index(file, log_tag);
    global_info_array[group_index].cpu_alloc_size.fetch_add(alloc_size);

    size_t free_after, total_after;
    size_t free_before = last_free_size.load();

    if(cudaMemGetInfo(&free_after, &total_after) != cudaSuccess)
    {
        NVLOGE_FMT(TAG, AERIAL_CUDA_API_EVENT, "cudaMemGetInfo failed");
    }

    int64_t used = free_before - free_after;
    if (used != 0)
    {
        global_info_array[group_index].gpu_implicit.fetch_add(used);
        last_free_size.fetch_sub(used);
    }

    NVLOGI_FMT(TAG, "CPU: {} +{} log_tag={} func={} {} free_before={} free_after={} alloc={} used={}",
            file_name, line, nvlog_get_component_name(log_tag), func_caller, func_alloc, free_before, free_after, alloc_size, used);

    last_file.store(file);
    last_line.store(line);
    last_caller.store(func_caller);
    last_log_tag.store(log_tag);

    // Unlock the GPU allocation mutex
    gpu_alloc_mutex_unlock();
    return err_code;
}

// Log GPU memory free operation (do NOT subtract from tracked size)
int memfoot_global_gpu_size_sub(const char* file, int line, int log_tag, const char* func_caller, const char* func_free, int err_code)
{
    if (initiated_finished.load())
    {
        // Do not track memory after the initialization is finished
        return 0;
    }

    const char* file_name = get_filename(file);

    // free_before was set by memfoot_global_get_mem_info() called before the free function
    size_t free_before = last_free_size.load();
    size_t free_after, total_after;

    // Measure memory after the free function was called
    if(cudaMemGetInfo(&free_after, &total_after) != cudaSuccess)
    {
        NVLOGE_FMT(TAG, AERIAL_CUDA_API_EVENT, "cudaMemGetInfo failed");
    }

    int64_t freed = free_after - free_before;
    if(freed != 0)
    {
        last_free_size.fetch_add(freed);
        total_freed_gpu_size.fetch_add(freed);
    }

    // Log the free operation (but do NOT subtract from tracked size)
    NVLOGI_FMT(TAG, "GPU_FREE: {} +{} log_tag={} func={} {} free_before={} free_after={} freed={}",
            file_name, line, nvlog_get_component_name(log_tag), func_caller, func_free, free_before, free_after, freed);

    last_file.store(file);
    last_line.store(line);
    last_caller.store(func_caller);
    last_log_tag.store(log_tag);

    // Unlock the GPU allocation mutex
    gpu_alloc_mutex_unlock();
    return err_code;
}

// Log CPU memory free operation (do NOT subtract from tracked size)
int memfoot_global_cpu_size_sub(const char* file, int line, int log_tag, const char* func_caller, const char* func_free, int err_code)
{
    if (initiated_finished.load())
    {
        // Do not track memory after the initialization is finished
        return 0;
    }

    const char* file_name = get_filename(file);

    // free_before was set by memfoot_global_get_mem_info() called before the free function
    size_t free_before = last_free_size.load();
    size_t free_after, total_after;

    // Measure memory after the free function was called
    if(cudaMemGetInfo(&free_after, &total_after) != cudaSuccess)
    {
        NVLOGE_FMT(TAG, AERIAL_CUDA_API_EVENT, "cudaMemGetInfo failed");
    }

    int64_t freed = free_after - free_before;
    if(freed != 0)
    {
        last_free_size.fetch_add(freed);
        total_freed_gpu_size.fetch_add(freed);
    }

    // Log the free operation (but do NOT subtract from tracked size)
    NVLOGI_FMT(TAG, "CPU_FREE: {} +{} log_tag={} func={} {} free_before={} free_after={} freed={}",
            file_name, line, nvlog_get_component_name(log_tag), func_caller, func_free, free_before, free_after, freed);

    last_file.store(file);
    last_line.store(line);
    last_caller.store(func_caller);
    last_log_tag.store(log_tag);

    // Unlock the GPU allocation mutex
    gpu_alloc_mutex_unlock();
    return err_code;
}

int64_t get_gpu_size_in_set(std::unordered_set<MemFootBase*>& set)
{
    MemtraceDisableScope md;

    int64_t size_sum = 0;
    for(MemFootBase* ins : set)
    {
        if (ins->getName() == "AccumulatorCellPhy"
          || ins->getName() == "wipAccumAccumMf"
          || ins->getName() == "cuphyChannelsAccumMf")
        {
            // Skip the accumulator MemFoot instances in cuphydriver
            continue;
        }
        size_sum += ins->getGpuSize();
    }
    return size_sum;
}

int64_t print_memfoot_instance_set(std::unordered_set<MemFootBase*>& set, const std::string& module_name)
{
    MemtraceDisableScope md;

    // Aggregate sizes by name
    std::unordered_map<std::string, int64_t> name_size_map;
    std::unordered_map<std::string, int> name_count_map;

    int64_t size_sum = 0;
    for(MemFootBase* ins : set)
    {
        if (ins->getName() == "AccumulatorCellPhy"
          || ins->getName() == "wipAccumAccumMf"
          || ins->getName() == "cuphyChannelsAccumMf")
        {
            // Skip the accumulator MemFoot instances in cuphydriver
            continue;
        }
        const std::string name = ins->getName();
        name_size_map[name] += ins->getGpuSize();
        name_count_map[name]++;
        size_sum += ins->getGpuSize();
    }

    // Print MemFoot grouped by instance name
    for(const auto& [name, total_size] : name_size_map)
    {
        // Each line represents all MemFootBase instances with the same name in the module
        // The "Size in MiB" and "Size in Bytes" both represent the total size of MemFootBase instances with the same name
        const int count = name_count_map[name];
        NVLOGC_FMT(TAG, "  {:20} | {:12} | {:14L} | {:14} |", name, total_size / INT64_ONE_MIB, total_size, count);
    }
    return size_sum;
}

// Print the memory info, should be called after all the memory allocation is done (when all cells are configured)
void memfoot_global_print_all()
{
    // Print once and set the flag when all cells are configured
    if (initiated_finished.fetch_or(1))
    {
        return;
    }

    // Check whether there's explicit memory allocation after last tracking
    memfoot_global_gpu_size_check(__FILE__, __LINE__, TAG, __func__);

    // Disable the GPU allocation after initialization to avoid performance dropping during runtime
    enable_gpu_alloc_mutex = false;

    global_info_t global_total = MEMFOOT_INFO_INIT("Total");
    for (int i = 0; i < MF_GROUP_MAX_NUM; i++)
    {
        global_total.gpu_alloc_size += global_info_array[i].gpu_alloc_size;
        global_total.cpu_alloc_size += global_info_array[i].cpu_alloc_size;
        global_total.gpu_overhead += global_info_array[i].gpu_overhead;
        global_total.gpu_implicit += global_info_array[i].gpu_implicit;
    }

    // Check if GPU global memfoot was enabled by replacing malloc functions with the macros
    bool global_memfoot_enabled = global_total.gpu_alloc_size.load() > 0;

    // Print Global grouped MemFoot size if enabled
    if (global_memfoot_enabled)
    {
        int64_t gpu_used = total_init - last_free_size.load() - system_used;
        int64_t untracked = gpu_used - global_total.gpu_alloc_size - global_total.gpu_implicit - global_total.gpu_overhead + total_freed_gpu_size.load();

        // Print total size in bytes with INFO level log for debug
        NVLOGI_FMT(TAG, "TOTAL: GpuAlloc={} HostAlloc={} Overhead={} Implicit={} GpuUsed={} Freed={} Untracked={}",
            global_total.gpu_alloc_size.load(), global_total.cpu_alloc_size.load(), global_total.gpu_overhead.load(),
            global_total.gpu_implicit.load() , gpu_used, total_freed_gpu_size.load(), untracked);

        // Print Global MemFoot Summary in console
        NVLOGC_FMT(TAG, "===== Global MemFoot Summary (Unit: MiB) ======= GPU Capacity: {:6} MiB ============================", total_init / INT64_ONE_MIB);
        NVLOGC_FMT(TAG, "Category         |  GpuAlloc | HostAlloc |  Overhead |  Implicit |   GpuUsed |  GpuFreed | Untracked |");
        for(int i = 0; i < MF_GROUP_MAX_NUM; i++)
        {
            NVLOGC_FMT(TAG, "{:16} | {:9} | {:9} | {:9} | {:9} |           |           |           |",
                global_info_array[i].group_name, global_info_array[i].gpu_alloc_size / INT64_ONE_MIB, global_info_array[i].cpu_alloc_size / INT64_ONE_MIB,
                global_info_array[i].gpu_overhead / INT64_ONE_MIB, global_info_array[i].gpu_implicit / INT64_ONE_MIB);
        }
        NVLOGC_FMT(TAG, "------------------------------------------------------------------------------------------------------");
        NVLOGC_FMT(TAG, "TOTAL            | {:9} | {:9} | {:9} | {:9} | {:9} | {:9} | {:9} |",
            global_total.gpu_alloc_size / INT64_ONE_MIB, global_total.cpu_alloc_size / INT64_ONE_MIB, global_total.gpu_overhead / INT64_ONE_MIB,
            global_total.gpu_implicit / INT64_ONE_MIB, gpu_used / INT64_ONE_MIB, total_freed_gpu_size.load() / INT64_ONE_MIB, untracked / INT64_ONE_MIB);
        NVLOGC_FMT(TAG, "======================================================================================================");
    }

    // Print Per Module-Tag MemFoot size
    NVLOGC_FMT(TAG, "===== Module-Tag MemFoot Summary ========= GPU Capacity: {:6} MiB =====", total_init / INT64_ONE_MIB);
    NVLOGC_FMT(TAG, "Module/Tag             |  Size in MiB |  Size in Bytes | Instance Count |");
    int64_t module_total = 0;
    for(int module_id = 0; module_id < MF_MODULE_MAX_NUM; module_id++)
    {
        std::string module_name = std::string(module_info_array[module_id].module_name) + " - TOTAL";

        int64_t global_module_size = global_info_array[module_id].gpu_alloc_size.load();

        int64_t module_sum = module_info_array[module_id].module_gpu_size;
        if (module_id == MF_MODULE_CUPHY || module_id == MF_MODULE_PHYDRV)
        {
            std::unordered_set<MemFootBase*>& instance_set = module_id == MF_MODULE_CUPHY ? cuphy_instance_set : phydrv_instance_set;
            module_sum += get_gpu_size_in_set(instance_set);
        }

        if (module_sum == 0)
        {
            // Skip the 0 size module. Currently "OTHER" module doesn't track any memory allocation.
            continue;
        }
        module_total += module_sum;

        // Module-TAG MemFoot format: Name | Size in MiB | Size in Bytes | instance count |
        NVLOGC_FMT(TAG, "-------------------------------------------------------------------------");
        NVLOGC_FMT(TAG, "{:22} | {:12} | {:14L} | Missed: {}", module_name, module_sum / INT64_ONE_MIB, module_sum,
            global_memfoot_enabled > 0 ? get_size_string(global_module_size - module_sum) : "N/A");

        if (module_id == MF_MODULE_CUPHY)
        {
            // Print cuPHY cuphyMemoryFootprint size
            print_memfoot_instance_set(cuphy_instance_set, module_name);
        }
        else if (module_id == MF_MODULE_PHYDRV)
        {
            // Print cuphydriver MemFoot size
            print_memfoot_instance_set(phydrv_instance_set, module_name);
        }

        for (int tag_id = 0; tag_id < MF_TAG_MAX_NUM; tag_id++)
        {
            if (tag_info_array[tag_id].module_id == module_id)
            {
                // One line for each tag in the module.
                // The "Size in MiB" and "Size in Bytes" both represent the total size for the same tag
                int64_t tag_bytes = tag_info_array[tag_id].gpu_alloc_size;
                NVLOGC_FMT(TAG, "  {:20} | {:12} | {:14L} |                |", tag_info_array[tag_id].tag_name, tag_bytes / INT64_ONE_MIB, tag_bytes);
            }
        }
    }
    NVLOGC_FMT(TAG, "-------------------------------------------------------------------------");
    NVLOGC_FMT(TAG, "TOTAL                  | {:12} | {:14L} | Missed: {}", module_total / INT64_ONE_MIB, module_total,
        global_memfoot_enabled ? get_size_string((global_total.gpu_alloc_size.load() - module_total)) : "N/A");
    NVLOGC_FMT(TAG, "=========================================================================");
}
