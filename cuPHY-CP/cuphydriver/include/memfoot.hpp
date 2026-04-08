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

#ifndef MEMFOOT_H
#define MEMFOOT_H

#include "memfoot_global.hpp"

#include <string>
#include "cuphydriver_api.hpp"

/**
 * @brief Memory footprint tracker for resource accounting
 *
 * Tracks memory allocations across different memory types (CPU regular, CPU pinned,
 * GPU pinned, GPU regular) for a specific component or object. Used throughout
 * cuPHYDriver to measure and report memory usage for debugging and optimization.
 */
class MemFoot : public MemFootBase {
public:
    /**
     * @brief Construct memory footprint tracker
     *
     * Initializes all memory counters to zero and sets uninitialized state.
     */
    MemFoot():MemFootBase(MF_MODULE_PHYDRV)
    {
        items = 0;
        cpu_obj_size = 0;
        cpu_regular_size = 0;
        cpu_pinned_size = 0;
        gpu_pinned_size = 0;
        gpu_regular_size = 0;
        initialized = false;
        pdh=nullptr;
    }

    /**
     * @brief Destroy memory footprint tracker
     */
    ~MemFoot() {}

    /**
     * @brief Initialize memory footprint tracker with name and object size
     *
     * @param _pdh          - Physical driver handle for logging
     * @param _name         - Descriptive name for this memory footprint (e.g., component name)
     * @param _cpu_obj_size - Size of the main CPU object being tracked
     * @return true on success, false if already initialized
     */
    bool init(phydriver_handle _pdh, std::string _name, size_t _cpu_obj_size);
    
    /**
     * @brief Add CPU regular memory allocation to footprint
     *
     * @param size - Size in bytes to add
     * @return true on success
     */
    bool addCpuRegularSize(size_t size);
    
    /**
     * @brief Add CPU pinned memory allocation to footprint
     *
     * @param size - Size in bytes to add
     * @return true on success
     */
    bool addCpuPinnedSize(size_t size);
    
    /**
     * @brief Add GPU pinned memory allocation to footprint
     *
     * @param size - Size in bytes to add (GDR pinned memory)
     * @return true on success
     */
    bool addGpuPinnedSize(size_t size);
    
    /**
     * @brief Add GPU regular memory allocation to footprint
     *
     * @param size - Size in bytes to add (cudaMalloc)
     * @return true on success
     */
    bool addGpuRegularSize(size_t size);

    /**
     * @brief Get total GPU memory size
     *
     * @return Total GPU memory in bytes
     */
    size_t getGpuSize() override
    {
        return gpu_regular_size + gpu_pinned_size;
    }

    /**
     * @brief Get name of this memory footprint
     *
     * @return Name string (component/object identifier)
     */
    std::string getName() override
    {
        return name;
    }

    /**
     * @brief Get CPU object size
     *
     * @return Size of main CPU object in bytes
     */
    size_t getCpuObjSize() const;
    
    /**
     * @brief Get total CPU regular memory size
     *
     * @return Accumulated CPU regular memory in bytes
     */
    size_t getCpuRegularSize() const;
    
    /**
     * @brief Get total CPU pinned memory size
     *
     * @return Accumulated CPU pinned memory in bytes
     */
    size_t getCpuPinnedSize() const;
    
    /**
     * @brief Get total GPU pinned memory size
     *
     * @return Accumulated GPU pinned memory (GDR) in bytes
     */
    size_t getGpuPinnedSize() const;
    
    /**
     * @brief Get total GPU regular memory size
     *
     * @return Accumulated GPU device memory in bytes
     */
    size_t getGpuRegularSize() const;
    
    /**
     * @brief Print complete memory footprint to log
     *
     * Logs all memory categories (CPU obj, CPU regular, CPU pinned, GPU pinned, GPU regular)
     * with human-readable sizes.
     */
    void printMemoryFootprint();
    
    /**
     * @brief Print GPU memory footprint only to log
     *
     * Logs GPU pinned and GPU regular memory usage with human-readable sizes.
     */
    void printGpuMemoryFootprint();
    
    /**
     * @brief Reset all memory counters to zero
     *
     * Preserves name and handle but clears all accumulated sizes.
     */
    void reset();
    
    /**
     * @brief Add another MemFoot's counters to this one
     *
     * Accumulates memory sizes from another MemFoot instance into this one.
     *
     * @param _mf - Memory footprint to add
     * @return true on success
     */
    bool addMF(MemFoot& _mf);

protected:
    phydriver_handle pdh{};                                    ///< Physical driver handle for logging
    std::string name;                                          ///< Descriptive name for this memory footprint
    bool initialized;                                          ///< Flag indicating whether init() has been called
    size_t cpu_obj_size;                                       ///< Size of main CPU object (e.g., sizeof(class))
    size_t cpu_regular_size;                                   ///< Accumulated CPU regular memory (malloc/new)
    size_t cpu_pinned_size;                                    ///< Accumulated CPU pinned memory (cudaHostAlloc)
    size_t gpu_pinned_size;                                    ///< Accumulated GPU pinned memory (GDR buffers)
    size_t gpu_regular_size;                                   ///< Accumulated GPU device memory (cudaMalloc)
    int items;                                                 ///< Number of items tracked (used for counting allocations)
};

#endif
