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

#ifndef CUPHY_PUSCH_START_KERNELS_HPP
#define CUPHY_PUSCH_START_KERNELS_HPP

#include "pusch_start_kernels_interface.hpp"

namespace pusch {

    /**
     * @class StartKernels
     * @brief actual implementation of @class IStartKernels.
     */
    class StartKernels final : public IStartKernels {
    public:
        /**
         * Setting wait kernels params, using Kernels functions defined in the .cu file.
         */
        void setWaitKernelParams(cuphyPuschRxWaitLaunchCfg_t* pLaunchCfg, uint8_t puschRxFullSlotMode, void* ppStatDescr, void* ppDynDescr) final;
        void setDeviceGraphLaunchKernelParams(cuphyPuschRxDglLaunchCfg_t* pLaunchCfg, uint8_t enableDeviceGraphLaunch, uint8_t puschRxFullSlotMode, void* ppDynDescr, void* ppDeviceGraph) final;
    };

    /**
     * @class NullStartKernels
     * @brief NullObject implementation of IWaitKernels
     */
    class NullStartKernels final : public IStartKernels {
    public:
        void setWaitKernelParams(cuphyPuschRxWaitLaunchCfg_t* pLaunchCfg, uint8_t puschRxFullSlotMode, void* ppStatDescr, void* ppDynDescr) final {}
        void setDeviceGraphLaunchKernelParams(cuphyPuschRxDglLaunchCfg_t* pLaunchCfg, uint8_t enableDeviceGraphLaunch, uint8_t puschRxFullSlotMode, void* ppDynDescr, void* ppDeviceGraph) final {}
    };

} // namespace pusch

#endif //CUPHY_PUSCH_START_KERNELS_HPP
