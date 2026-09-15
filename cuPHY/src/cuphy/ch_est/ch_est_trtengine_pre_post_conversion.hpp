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

#ifndef CH_EST_TRTENGINE_PRE_POST_CONVERSION_HPP
#define CH_EST_TRTENGINE_PRE_POST_CONVERSION_HPP

#include "trt_engine/trt_engine_interfaces.hpp"

namespace ch_est {

/**
 * @brief concreate implementation of interface IPrePostEnqueueTensorConversion
 *
 * @class ChestPrePostEnqueueTensorConversion: Implementing pre/post kernel, and setup()
 */
class ChestPrePostEnqueueTensorConversion final : public trt_engine::IPrePostEnqueueTensorConversion
{
public:
    struct puschPrePostTensorConversionDynDescr_t final
    {
        cuphyPuschRxUeGrpPrms_t* pDrvdUeGrpPrms{};
        cuphyTensorInfo6_t       tInputInternalBuf{};
        cuphyTensorInfo6_t       tOutputInternalBuf{};
    };

    struct puschPrePostTensorConversionKernelArgs_t final
    {
        puschPrePostTensorConversionDynDescr_t* pDynDescr{};
    };

    struct puschPrePostTensorConversionLaunchCfg_t final
    {
        CUDA_KERNEL_NODE_PARAMS kernelNodeParamsDriver{};
        void*                   kernelArgs[1]{};
    };


    [[nodiscard]]
    cuphyStatus_t preEnqueueConvert([[maybe_unused]] const std::vector<trt_engine::TrtParams>& inputTensorPrms,
                                    [[maybe_unused]] const std::vector<std::vector<int>>&      inputStridesTrt,
                                    [[maybe_unused]] const std::vector<std::vector<int>>&      inputStridesCuphy,
                                    [[maybe_unused]] const std::vector<void*>&                 inputBuffers,
                                    [[maybe_unused]] const std::vector<void*>&                 inputInternalBuf,
                                    [[maybe_unused]] cudaStream_t                              cuStream) final;
    [[nodiscard]]
    cuphyStatus_t postEnqueueConvert([[maybe_unused]] const std::vector<trt_engine::TrtParams>& outputTensorPrms,
                                     [[maybe_unused]] const std::vector<std::vector<int>>&      outputStridesCuphy,
                                     [[maybe_unused]] const std::vector<std::vector<int>>&      outputStridesTrt,
                                     [[maybe_unused]] const std::vector<void*>&                 outputInternalBuf,
                                     [[maybe_unused]] const std::vector<void*>&                 outputBuffers,
                                     [[maybe_unused]] cudaStream_t                              cuStream) final;
    [[nodiscard]]
    cuphyStatus_t setup(gsl_lite::span<cuphyPuschRxUeGrpPrms_t>                    pDrvdUeGrpPrmsCpu,
                        gsl_lite::span<cuphyPuschRxUeGrpPrms_t>                    pDrvdUeGrpPrmsGpu,
                        [[maybe_unused]] const std::vector<std::vector<int>>& inputStridesTrt,
                        [[maybe_unused]] const std::vector<void*>&            inputInternalBuf,
                        [[maybe_unused]] const std::vector<std::vector<int>>& outputStridesTrt,
                        [[maybe_unused]] const std::vector<void*>&            outputInternalBuf,
                        cudaStream_t                                          strm) final;

    [[nodiscard]] const puschPrePostTensorConversionLaunchCfg_t& getPreCfg() const { return m_pPreTensorConversionKernelLaunchCfg; }
    [[nodiscard]] const puschPrePostTensorConversionLaunchCfg_t& getPostCfg() const { return m_pPostTensorConversionKernelLaunchCfg; }
private:
    int*                                                                       m_cuphyStrides{};
    int*                                                                       m_inputStrides{};
    std::vector<int>                                                           m_inputStridesCuphyPusch;
    std::vector<int>                                                           m_outputStridesCuphyPusch;
    cuphy::buffer<puschPrePostTensorConversionDynDescr_t, cuphy::device_alloc> m_pDynDescrGpu;
    puschPrePostTensorConversionKernelArgs_t                                   m_kernelArgs{};
    puschPrePostTensorConversionLaunchCfg_t                                    m_pPreTensorConversionKernelLaunchCfg{};
    puschPrePostTensorConversionLaunchCfg_t                                    m_pPostTensorConversionKernelLaunchCfg{};
};

} // namespace ch_est

#endif //CH_EST_TRTENGINE_PRE_POST_CONVERSION_HPP
