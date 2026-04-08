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

#ifndef MPSCTX_CLASS_H
#define MPSCTX_CLASS_H

#include <memory>
#include <vector>
#include <array>
#include <string>
#include <tuple>
#include "locks.hpp"
#include "cuphydriver_api.hpp"
#include "fh.hpp"
#include "gpudevice.hpp"
#include "constant.hpp"

/**
 * @brief MPS or Green Context wrapper for GPU resource partitioning
 *
 * Manages CUDA contexts with resource partitioning for multi-stream workloads.
 * Supports both MPS (Multi-Process Service) and CUDA 12.4+ green contexts.
 */
class MpsCtx {
public:
    /**
     * @brief Construct MPS context with SM count 
     *
     * Creates a CUDA context with execution affinity under MPS
     * which can use at most _devSmCount SMs.
     *
     * @param _pdh        - Cuphydriver handle
     * @param _gDev       - GPU device struct handle
     * @param _devSmCount - Number of streaming multiprocessors (SMs) this context can use
     */
    MpsCtx(phydriver_handle _pdh, GpuDevice* _gDev, int _devSmCount);
    
#if CUDA_VERSION >= 12040
    /**
     * @brief Construct green context with explicit resource descriptors (CUDA 12.4+)
     *
     * Creates a CUDA green context provisioned with specified device resources
     *
     * @param _pdh                  - Cuphydriver handle
     * @param _gDev                 - GPU device struct handle
     * @param _resources            - Device resources, currently SM-type or work queues, this green context will be provisioned with.
     * @param _name                 - Name for this green context (useful for logging)
     * @param _print_resources      - Print green context's device resources, if true (default false).
     * @param _use_workqueues       - Green context will also use work queues if true (default false).
     * @param _wq_concurrency_limit - Green context's requested work queue concurrency limit.
     */
    MpsCtx(phydriver_handle _pdh, GpuDevice* _gDev, CUdevResource* _resources, const std::string& _name, bool _print_resources=false, bool _use_workqueues=false, unsigned int _wq_concurrency_limit=2);
#endif
    
    /**
     * @brief Destroy MPS/green context
     *
     * Cleans up CUDA context and associated resources. Ensures proper context
     * destruction to avoid resource leaks.
     */
    ~MpsCtx();

    /**
     * @brief Get cuphydriver handle
     *
     * @return cuphydriver handle associated with this context
     */
    phydriver_handle getPhyDriverHandler(void) const;
    
    /**
     * @brief Get unique identifier
     *
     * @return Return unique ID (timestamp-based)
     */
    uint64_t         getId() const;
    
    /**
     * @brief Set this context's GPU device as current
     *
     * Calls gDev->setDevice() to make this context's GPU the current device
     * for subsequent CUDA operations on the calling thread.
     */
    void             setGpuDevice();
    
    /**
     * @brief Get GPU device struct handle
     *
     * @return Pointer to GPU device struct associated with this context
     */
    GpuDevice*       getGpuDevice();
    
    /**
     * @brief Set this context as active for current thread
     *
     * Makes this CUDA context (MPS or green) the active context for the
     * calling thread. Required before launching kernels or performing
     * CUDA operations on this context.
     */
    void             setCtx();
    
#if CUDA_VERSION >= 12040
    /**
     * @brief Get SM-type resources associated with this green context.
     *
     * Retrieve the CUdevResource SM resources this green context (cuGreenCtx), if created, was provisioned with.
     *
     * @param resource - parameter to be updated with device resources
     */
    void             getResources(CUdevResource* resource) const;
    /**
     * @brief Get access to the green context handle
     *
     * Return the green context handle
     */
    CUgreenCtx       getGreenCtx() const;
    /**
     * @brief Print green context device resource information
     *
     * Print (nvlog) information about the device resources (SM-type, work queues) associated with
     * a green context. No information is printed for an MPS context.
     */
    void             printGreenCtxResourceInfo() const;
    /**
     * @brief Get context ID
     *
     * Return context ID
     */
    unsigned long long getCtxId() const;
#endif

private:
    phydriver_handle          pdh;                             ///< cuphydriver handle
    uint64_t                  id;                              ///< Unique identifier (timestamp-based)
    GpuDevice*                gDev;                            ///< GPU device struct handle
    int                       devSmCount;                      ///< Number of SMs this context (cuCtx) has access to
    CUdevice                  cuDev;                           ///< CUDA device handle
    CUcontext                 cuCtx;                           ///< CUDA context handle
#if CUDA_VERSION >= 12040
    CUgreenCtx                cuGreenCtx{};                    ///< CUDA green context handle (CUDA 12.4+)
    CUdevResourceDesc         cuResourceDesc;                  ///< Resource descriptor for green context (SM allocation, etc.)
    CUdevResource             m_resources[2] = {{}, {}};       ///< Device resources for green context
#endif
    bool                      isGreenContext;                  ///< True if this is a green context, false if MPS context
    bool                      ctxCreated = false;              ///< Flag indicating if context was successfully created
    bool                      ctxDestroyed = false;            ///< Flag indicating if context has been destroyed (prevents double-free)
    std::string               name{};                          ///< Name of green context (used for logging)
};

#endif
