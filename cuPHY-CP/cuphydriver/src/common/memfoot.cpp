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

#define TAG (NVLOG_TAG_BASE_CUPHY_DRIVER + 29) // "DRV.MEMFOOT"

#include "context.hpp"
#include "constant.hpp"
#include "memfoot.hpp"

bool MemFoot::init(phydriver_handle _pdh, std::string _name, size_t _cpu_obj_size) {
    pdh = _pdh;
    name = _name;
    initialized = true;
    cpu_obj_size = _cpu_obj_size;

    // Currently the CPU regular size of the various MemFoot objects isn't
    // accounted for in the wip_accum_mf object. The difference is due to the line below.
    (StaticConversion<PhyDriverCtx>(pdh).get())->ctx_tot_cpu_regular_memory += cpu_obj_size;

    return true;
}

bool MemFoot::addCpuRegularSize(size_t size)
{
    if(initialized == false)
        return false;

    cpu_regular_size += size;
    if (pdh && (StaticConversion<PhyDriverCtx>(pdh).get())) {
        (StaticConversion<PhyDriverCtx>(pdh).get())->ctx_tot_cpu_regular_memory += size;
    }

    return true;
}

bool MemFoot::addCpuPinnedSize(size_t size)
{
    if(initialized == false)
        return false;

    cpu_pinned_size += size;
    if (pdh && (StaticConversion<PhyDriverCtx>(pdh).get())) {
        (StaticConversion<PhyDriverCtx>(pdh).get())->ctx_tot_cpu_pinned_memory += size;
    }

    return true;
}

bool MemFoot::addGpuPinnedSize(size_t size)
{
    if(initialized == false)
        return false;

    gpu_pinned_size += size;
    if (pdh && (StaticConversion<PhyDriverCtx>(pdh).get())) {
        (StaticConversion<PhyDriverCtx>(pdh).get())->ctx_tot_gpu_pinned_memory += size;
    }

    return true;
}

bool MemFoot::addGpuRegularSize(size_t size)
{
    if(initialized == false)
        return false;

    gpu_regular_size += size;
    if (pdh && (StaticConversion<PhyDriverCtx>(pdh).get())) {
        (StaticConversion<PhyDriverCtx>(pdh).get())->ctx_tot_gpu_regular_memory += size;
    }

    return true;
}

size_t MemFoot::getCpuObjSize() const {
    return cpu_obj_size;
}

size_t MemFoot::getCpuRegularSize() const {
    return cpu_regular_size;
}

size_t MemFoot::getCpuPinnedSize() const {
    return cpu_pinned_size;
}

size_t MemFoot::getGpuPinnedSize() const {
    return gpu_pinned_size;
}

size_t MemFoot::getGpuRegularSize() const {
    return gpu_regular_size;
}

void MemFoot::printMemoryFootprint() {
    NVLOGI_FMT(TAG, "Memory Footprint {} x {}",name.c_str(), items);
    NVLOGI_FMT(TAG, "\tObject size {} Bytes", cpu_obj_size);
    NVLOGI_FMT(TAG, "\tCPU regular memory {} Bytes", cpu_regular_size);
    NVLOGI_FMT(TAG, "\tCPU pinned memory {} Bytes", cpu_pinned_size);
    NVLOGI_FMT(TAG, "\tGPU regular memory {} Bytes", gpu_regular_size);
    NVLOGI_FMT(TAG, "\tGPU pinned memory {} Bytes", gpu_pinned_size);
}

void MemFoot::printGpuMemoryFootprint() {
    //NVLOGC_FMT(TAG, "Total GPU allocated memory from %s (items %d) %.3f MiB\n",name.c_str(), items, gpu_regular_size/(1024*1024.0));
    NVLOGI_FMT(TAG, "GPU allocated memory from {}: {:.3f} MiB.", name, gpu_regular_size/(1024*1024.0));
}

void MemFoot::reset() {
    items = 0;
    cpu_obj_size = 0;
    cpu_regular_size = 0;
    cpu_pinned_size = 0;
    gpu_pinned_size = 0;
    gpu_regular_size = 0;
}

bool MemFoot::addMF(MemFoot& _mf) {

    if(initialized == false)
        return false;

    cpu_obj_size += _mf.cpu_obj_size;
    cpu_regular_size += _mf.cpu_regular_size;
    cpu_pinned_size += _mf.cpu_pinned_size;
    gpu_pinned_size += _mf.gpu_pinned_size;
    gpu_regular_size += _mf.gpu_regular_size;

    items++;

    return true;
}
