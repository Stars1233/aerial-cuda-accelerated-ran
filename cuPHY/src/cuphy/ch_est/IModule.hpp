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

#ifndef CUPHY_CHEST_IMODULE_HPP
#define CUPHY_CHEST_IMODULE_HPP

#include <gsl-lite/gsl-lite.hpp>

#include "IGraph_mgr.hpp"
#include "pusch_start_kernels_interface.hpp"
#include "IStream.hpp"

#include "cuphy.h"

namespace ch_est {

    /**
     * @brief IKernelBuilder Contains logic to choose a kernel for IModule
     * @class IKernelBuilder: has interface to init and build/setup kernels for every slot.
     *        The build function populates cuphyPuschRxChEstLaunchCfgs_t with the kernels related information.
     */
    class IKernelBuilder {
    public:
        /// Rule of 5 is needed in baseclass
        IKernelBuilder() = default;
        virtual ~IKernelBuilder() = default;
        IKernelBuilder(const IKernelBuilder& kb) = default;
        IKernelBuilder(IKernelBuilder&& kb) = default;
        IKernelBuilder& operator=(const IKernelBuilder& kb) = default;
        IKernelBuilder& operator=(IKernelBuilder&& kb) = default;

        /**
         * @brief Given Static Descriptors of CPU and GPU, assign this information to internal kernel arguments.
         * cudaMemcpy is used to copy data to the device (CPU -> GPU).
         * This init() call is done during initialize time only.
         * @param ppStatDescrsCpu CPU stats descriptor
         * @param ppStatDescrsGpu GPU stats descriptor
         * @param enableCpuToGpuDescrAsyncCpy true/false to enable Async Copy of descriptors
         * @param strm CUDA stream to use for cudaMemcpy
         */
        virtual void init(gsl_lite::span<uint8_t*> ppStatDescrsCpu,
                          gsl_lite::span<uint8_t*> ppStatDescrsGpu,
                          bool enableCpuToGpuDescrAsyncCpy,
                          cudaStream_t strm)  = 0;

        /**
         * @brief Given the below parameters, run logic to choose kernel and populate the Launch Configs array.
         * The array representing the kernel arguments/functions.
         * This function can run every slot.
         * @param pDrvdUeGrpPrmsCpu UE Groups params in CPU
         * @param pDrvdUeGrpPrmsGpu UE Groups params in GPU
         * @param nUeGrps Number of UE groups
         * @param maxDmrsMaxLen the max of DMRS max length
         * @param enableDftSOfdm bool
         * @param chEstAlgo Actual algorithms to use for kernel builder
         * @param enableUlRxBf bool
         * @param enablePerPrgChEst bool
         * @param pPreEarlyHarqWaitKernelStatusGpu
         * @param pPostEarlyHarqWaitKernelStatusGpu
         * @param waitTimeOutPreEarlyHarqUs microsec
         * @param waitTimeOutPostEarlyHarqUs microsec
         * @param enableCpuToGpuDescrAsyncCpy bool
         * @param ppDynDescrsCpu Dynamic descriptor CPU
         * @param ppDynDescrsGpu Dynamic descriptor GPU
         * @param pStartKernels pointer to implementation of startKernels.
         * @param launchCfgs array of kernels configs
         * @param enableEarlyHarqProc
         * @param enableFrontLoadedDmrsProc
         * @param enableDeviceGraphLaunch
         * @param pSubSlotDeviceGraphExec CUDA Graph Exec
         * @param pFullSlotDeviceGraphExec CUDA Graph Exec
         * @param pWaitKernelLaunchCfgsPreSubSlot CUDA Graph Exec
         * @param pWaitKernelLaunchCfgsPostSubSlot CUDA Graph Exec
         * @param pDglKernelLaunchCfgsPreSubSlot Pre subslot related kernel - Device Graph Launch
         * @param pDglKernelLaunchCfgsPostSubSlot Post subslot related kernel - Device Graph Launch
         * @param strm CUDA stream to use
         * @return cuphyStatus_t for SUCCESS or Failure
         */
        [[nodiscard]]
        virtual cuphyStatus_t
        build(gsl_lite::span<cuphyPuschRxUeGrpPrms_t>         pDrvdUeGrpPrmsCpu,
              gsl_lite::span<cuphyPuschRxUeGrpPrms_t>         pDrvdUeGrpPrmsGpu,
              uint16_t                                   nUeGrps,
              uint8_t                                    maxDmrsMaxLen,
              uint8_t                                    enableDftSOfdm,
              uint8_t                                    chEstAlgo,
              uint8_t                                    enableUlRxBf,
              uint8_t                                    enablePerPrgChEst,
              uint8_t*                                   pPreEarlyHarqWaitKernelStatusGpu,
              uint8_t*                                   pPostEarlyHarqWaitKernelStatusGpu,
              uint16_t                                   waitTimeOutPreEarlyHarqUs,
              uint16_t                                   waitTimeOutPostEarlyHarqUs,
              bool                                       enableCpuToGpuDescrAsyncCpy,
              gsl_lite::span<uint8_t*>                        ppDynDescrsCpu,
              gsl_lite::span<uint8_t*>                        ppDynDescrsGpu,
              pusch::IStartKernels*                      pStartKernels,
              gsl_lite::span<cuphyPuschRxChEstLaunchCfgs_t>   launchCfgs,
              uint8_t                               enableEarlyHarqProc,
              uint8_t                               enableFrontLoadedDmrsProc,
              uint8_t                               enableDeviceGraphLaunch,
              CUgraphExec*                          pSubSlotDeviceGraphExec,
              CUgraphExec*                          pFullSlotDeviceGraphExec,
              cuphyPuschRxWaitLaunchCfg_t*          pWaitKernelLaunchCfgsPreSubSlot,
              cuphyPuschRxWaitLaunchCfg_t*          pWaitKernelLaunchCfgsPostSubSlot,
              cuphyPuschRxDglLaunchCfg_t*           pDglKernelLaunchCfgsPreSubSlot,
              cuphyPuschRxDglLaunchCfg_t*           pDglKernelLaunchCfgsPostSubSlot,
              cudaStream_t                          strm) = 0;
    };

    /**
     * @brief Provides a base class for concrete Channel Estimate modules.
     * The class instance is created/instantiated by createPuschRxChEst() function
     * and is used by PuschRx class type.
     * @class IModule: defines a common interface for CUPHY Channel Estimate modules.
     * @note Currently - oriented around PUSCH Channel Estimate.
     */
    class IModule {
    public:
        /// Rule of 5 is needed in baseclass
        IModule() = default;
        virtual ~IModule() = default;
        IModule(const IModule& im) = default;
        IModule(IModule&& im) = default;
        IModule& operator=(const IModule& im) = default;
        IModule& operator=(IModule&& im) = default;

        /**
         * @brief Initialize channel estimator object and static component descriptor
         * Initialize CPU/GPU stats descriptors from various elements (for example ChannelSettings).
         * @param pKernelBuilder kernel builder abstraction to use for init anb configuring kernels per slot.
         * @param enableCpuToGpuDescrAsyncCpy true/false to enable Async Copy of descriptors
         * @param ppStatDescrsCpu CPU stats descriptor
         * @param ppStatDescrsGpu GPU stats descriptor
         * @param strm CUDA stream to use for cudaMemcpy
         */
        virtual void init(IKernelBuilder*             pKernelBuilder,
                          bool                        enableCpuToGpuDescrAsyncCpy,
                          gsl_lite::span<uint8_t*>         ppStatDescrsCpu,
                          gsl_lite::span<uint8_t*>         ppStatDescrsGpu,
                          cudaStream_t                strm) = 0;

        /**
         * @brief setup object state and dynamic component descriptor in preparation towards execution, per slot
         * @param pKernelBuilder - Kernel builder to use during slot setup time.
         * @param pDrvdUeGrpPrmsCpu UE Groups params in CPU
         * @param pDrvdUeGrpPrmsGpu UE Groups params in GPU
         * @param nUeGrps Number of UE groups
         * @param maxDmrsMaxLen the max of DMRS max length
         * @param pPreEarlyHarqWaitKernelStatusGpu
         * @param pPostEarlyHarqWaitKernelStatusGpu
         * @param waitTimeOutPreEarlyHarqUs microsec
         * @param waitTimeOutPostEarlyHarqUs microsec
         * @param enableCpuToGpuDescrAsyncCpy bool
         * @param ppDynDescrsCpu Dynamic descriptor CPU
         * @param ppDynDescrsGpu Dynamic descriptor GPU
         * @param enableEarlyHarqProc
         * @param enableFrontLoadedDmrsProc
         * @param enableDeviceGraphLaunch
         * @param pSubSlotDeviceGraphExec CUDA Graph Exec
         * @param pFullSlotDeviceGraphExec CUDA Graph Exec
         * @param pWaitKernelLaunchCfgsPreSubSlot CUDA Graph Exec
         * @param pWaitKernelLaunchCfgsPostSubSlot CUDA Graph Exec
         * @param pDglKernelLaunchCfgsPreSubSlot Pre subslot related kernel - Device Graph Launch
         * @param pDglKernelLaunchCfgsPostSubSlot Post subslot related kernel - Device Graph Launch
         * @param strm CUDA stream to use
         */
        [[nodiscard]]
        virtual cuphyStatus_t setup(IKernelBuilder*                       pKernelBuilder,
                                    gsl_lite::span<cuphyPuschRxUeGrpPrms_t>    pDrvdUeGrpPrmsCpu,
                                    gsl_lite::span<cuphyPuschRxUeGrpPrms_t>    pDrvdUeGrpPrmsGpu,
                                    uint16_t                              nUeGrps,
                                    uint8_t                               maxDmrsMaxLen,
                                    uint8_t*                              pPreEarlyHarqWaitKernelStatusGpu,
                                    uint8_t*                              pPostEarlyHarqWaitKernelStatusGpu,
                                    uint16_t                              waitTimeOutPreEarlyHarqUs,
                                    uint16_t                              waitTimeOutPostEarlyHarqUs,
                                    bool                                  enableCpuToGpuDescrAsyncCpy,
                                    gsl_lite::span<uint8_t*>                   ppDynDescrsCpu,
                                    gsl_lite::span<uint8_t*>                   ppDynDescrsGpu,
                                    uint8_t                               enableEarlyHarqProc,
                                    uint8_t                               enableFrontLoadedDmrsProc,
                                    uint8_t                               enableDeviceGraphLaunch,
                                    CUgraphExec*                          pSubSlotDeviceGraphExec,
                                    CUgraphExec*                          pFullSlotDeviceGraphExec,
                                    cuphyPuschRxWaitLaunchCfg_t*          pWaitKernelLaunchCfgsPreSubSlot,
                                    cuphyPuschRxWaitLaunchCfg_t*          pWaitKernelLaunchCfgsPostSubSlot,
                                    cuphyPuschRxDglLaunchCfg_t*           pDglKernelLaunchCfgsPreSubSlot,
                                    cuphyPuschRxDglLaunchCfg_t*           pDglKernelLaunchCfgsPostSubSlot,
                                    cudaStream_t                          strm) = 0;

        /**
         * @brief set early HARQ enabled/disabled
         * @param earlyHarqModeEnabled true/false
         */
        virtual void setEarlyHarqModeEnabled(bool earlyHarqModeEnabled) = 0;


        /**
         * @brief Graph manipulation API
         * @return Reference to Channel Estimate Graph mode interface
         */
        virtual IChestGraphNodes&   chestGraph() = 0;

        /**
         * @brief Early HARQ sub-slot for channel estimate
         * @return Reference to sub-slot channel estimate, early HARQ related
         */
        virtual IChestSubSlotNodes& earlyHarqGraph() = 0;

        /**
         * @brief DMRS sub-slot for channel estimate
         * @return Reference to sub-slot channel estimate, DMRS related
         */
        virtual IChestSubSlotNodes& frontDmrsGraph() = 0;

        /**
         * @brief Stream mode interface to run kernels.
         * Run kernels as captured in the derived/concrete class types.
         * Run Secondary kernels as captured in the derived/concrete class types.
         * @return Reference to Channel estimate stream mode API.
         */
        virtual IChestStream&     chestStream() = 0;

        /**
         * @brief Start Kernels run in the beginning of the slot.
         * @return Reference to Start Kernels API
         */
        virtual pusch::IStartKernels& startKernels() = 0;
    };

} // namespace ch_est

#endif //CUPHY_CHEST_IMODULE_HPP
