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

#include <memory>

#include "chest_factory.hpp"

#include "ch_est.hpp"
#include "trtengine_chest.hpp"

namespace ch_est::factory {
    std::unique_ptr<IKernelBuilder> createPuschRxChEstKernelBuilder() {
        return std::make_unique<puschRxChEstKernelBuilder>();
    }

    template <typename CH_EST_TYPE>
    std::pair<std::unique_ptr<IModule>, cuphyStatus_t>
    createPuschRxChEstImpl(IKernelBuilder *kernelBuilder,
                           const cuphyChEstSettings &chEstSettings,
                           const bool earlyHarqModeEnabled,
                           const bool enableCpuToGpuDescrAsyncCpy,
                           gsl_lite::span<uint8_t *> statDescrsCpu,
                           gsl_lite::span<uint8_t *> statDescrsGpu,
                           cudaStream_t strm) {
        std::unique_ptr<IModule> pChEst;
        try
        {
            pChEst = std::make_unique<CH_EST_TYPE>(chEstSettings, earlyHarqModeEnabled);

            /**
             * init does the following:
             * for each tensor pair (desc*, void*), do the same thing:
             * assign to ppStatDescrsCpu and from this one assign to pPuschRkhsPrms
             * There is m_kernelArgsArr member, that is being assigned with the ppStatDescrsGpu arg.
             * Algo type is chEstAlgoType
             */
            pChEst->init(kernelBuilder,
                         enableCpuToGpuDescrAsyncCpy,
                         statDescrsCpu,
                         statDescrsGpu,
                         strm);
        }
        catch(const std::bad_alloc& eba)
        {
            return {nullptr, CUPHY_STATUS_ALLOC_FAILED};
        }
        catch(...)
        {
            return {nullptr, CUPHY_STATUS_INTERNAL_ERROR};
        }
        return {std::move(pChEst), CUPHY_STATUS_SUCCESS};
    }

    std::pair<std::unique_ptr<IModule>, cuphyStatus_t>
    createPuschRxChEst(IKernelBuilder*             kernelBuilder,
                       const cuphyChEstSettings&   chEstSettings,
                       const bool        earlyHarqModeEnabled,
                       const bool        enableCpuToGpuDescrAsyncCpy,
                       gsl_lite::span<uint8_t*>   statDescrsCpu,
                       gsl_lite::span<uint8_t*>   statDescrsGpu,
                       cudaStream_t      strm) {

        // Vanilla chest if no filename for specific chest related configs.
        if (!chEstSettings.puschrxChestFactorySettingsFilename ||
            chEstSettings.puschrxChestFactorySettingsFilename->empty()) {
            return createPuschRxChEstImpl<puschRxChEst>(kernelBuilder,
                                                        chEstSettings,
                                                        earlyHarqModeEnabled,
                                                        enableCpuToGpuDescrAsyncCpy,
                                                        statDescrsCpu,
                                                        statDescrsGpu,
                                                        strm);
        }
        return createPuschRxChEstImpl<TrtEnginePuschRxChEst>(kernelBuilder,
                                                             chEstSettings,
                                                             earlyHarqModeEnabled,
                                                             enableCpuToGpuDescrAsyncCpy,
                                                             statDescrsCpu,
                                                             statDescrsGpu,
                                                             strm);
    }
}
