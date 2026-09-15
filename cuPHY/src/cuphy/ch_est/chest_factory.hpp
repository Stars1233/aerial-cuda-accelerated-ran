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

#ifndef CUPHY_CHEST_FACTORY_HPP
#define CUPHY_CHEST_FACTORY_HPP

#include <memory>

#include <gsl-lite/gsl-lite.hpp>

#include "ch_est/IModule.hpp"
#include "ch_est/ch_est_settings.hpp"

namespace ch_est::factory {

    /**
     * @brief Factory function to instantiate a specific PUSCH-RX kernel builds.
     * @note Currently only instantiating Channel Estimate Kernel Builder.
     * @return unique_ptr of IKernelBuilder, pointing to a specific, concrete implementation.
     */
    [[nodiscard]]
    std::unique_ptr<IKernelBuilder> createPuschRxChEstKernelBuilder();

    /**
     * Create PUSCH-RX specific derived instance.
     * Calls init on the newly created instance.
     * If no puschrxChestFactorySettingsFilename provided as part of the ChannelSettings,
     * a vanilla Channel Estimate implementation is instantiated.
     * Otherwise, TrtEnginePuschRxChEst class type is instantiated.
     * @param kernelBuilder - Kernel builder to pass to the newly created instance.
     * @param chEstSettings - Channel estimate settings to use for creating the instance.
     * @param earlyHarqModeEnabled true/false for early HARQ mode at the time of instantiation.
     * @param enableCpuToGpuDescrAsyncCpy bool (@note - currently always true)
     * @param statDescrsCpu - CPU descriptors
     * @param statDescrsGpu - GPU descriptors
     * @param strm CUDA stream to pass the instance
     * @return pair of <unique_ptr, STATUS> unique_ptr of IModule, pointing to a specific, concrete implementation
     */
    [[nodiscard]]
    std::pair<std::unique_ptr<IModule>, cuphyStatus_t>
    createPuschRxChEst(IKernelBuilder*             kernelBuilder,
                       const cuphyChEstSettings&   chEstSettings,
                       bool                        earlyHarqModeEnabled,
                       bool                        enableCpuToGpuDescrAsyncCpy,
                       gsl_lite::span<uint8_t*>         statDescrsCpu,
                       gsl_lite::span<uint8_t*>         statDescrsGpu,
                       cudaStream_t                strm);
} // namespace ch_est::factory

#endif //  CUPHY_CHEST_FACTORY_HPP
