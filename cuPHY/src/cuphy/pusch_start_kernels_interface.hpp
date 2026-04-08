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

#ifndef CUPHY_START_KERNELS_INTERFACE_HPP
#define CUPHY_START_KERNELS_INTERFACE_HPP

#include <cstdint>

#include "cuphy.h"

namespace pusch {

    /**
     * @class IStartKernels
     * @brief interface for waitKernels
     * Parameters setting,
     *
     * Kernel configs launch paramsLaunch
     */
    class IStartKernels {
    public:
        IStartKernels() = default;
        virtual ~IStartKernels() = default;
        IStartKernels(const IStartKernels& waitNodes) = default;
        IStartKernels& operator=(const IStartKernels& waitNodes) = default;
        IStartKernels(IStartKernels&& waitNodes) = default;
        IStartKernels& operator=(IStartKernels&& waitNodes) = default;

        // API

        /**
         * @brief populate pLaunchCfg (Wait), set grid/block dims, kernel args and kernel functions.
         * @param pLaunchCfg - Launch configs to populate.
         * @param puschRxProcMode - SUB vs FULL SLOT
         * @param ppStatDescr - if defined, use it for kernelArgs[0]
         * @param ppDynDescr - if defined, use it for kernelArgs[1]
         */
        virtual void setWaitKernelParams(cuphyPuschRxWaitLaunchCfg_t* pLaunchCfg, uint8_t puschRxProcMode,
                                         void* ppStatDescr, void* ppDynDescr) = 0;

        /**
         * @brief populate pLaunchCfg (DGL), set grid/block dims, kernel args and kernel functions.
         * @param pLaunchCfg - Launch configs to populate.
         * @param enableDeviceGraphLaunch - If true, use deviceGraphLaunchKernel<true>, <false> otherwise.
         * @param ppDynDescr - if defined, use it for kernelArgs[0]
         * @param ppDeviceGraph - if defined, use it for kernelArgs[1]
         */
        virtual void setDeviceGraphLaunchKernelParams(cuphyPuschRxDglLaunchCfg_t* pLaunchCfg, uint8_t enableDeviceGraphLaunch, uint8_t puschRxProcMode,
                                                      void* ppDynDescr, void* ppDeviceGraph) = 0;
    };

} // namespace pusch

#endif //CUPHY_START_KERNELS_INTERFACE_HPP
