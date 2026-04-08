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

#include <iostream>
#include <memory>
#include <stdexcept>
#include <algorithm>

#include "nvlog.h"
#include "cuphy.h"
#include "util.hpp"
#include "utils.cuh"
#include "cuphy.hpp"
#include "cuphy_channels.hpp"
#include "cuda_array_interface.hpp"
#include "pycuphy_util.hpp"
#include "pycuphy_ldpc.hpp"

namespace pycuphy {


LdpcDerateMatch::LdpcDerateMatch(const bool scrambling, const cudaStream_t cuStream, int fpConfig):
m_cuStream(cuStream) {

    // PUSCH rate match descriptors.
    // Descriptors hold kernel parameters in GPU.
    size_t dynDescrAlignBytes;
    cuphyStatus_t statusGetWorkspaceSize = cuphyPuschRxRateMatchGetDescrInfo(&m_dynDescrSizeBytes, &dynDescrAlignBytes);
    if(statusGetWorkspaceSize != CUPHY_STATUS_SUCCESS) {
        throw cuphy::cuphy_fn_exception(statusGetWorkspaceSize, "cuphyPuschRxRateMatchGetDescrInfo");
    }
    m_dynDescrBufCpu = cuphy::buffer<uint8_t, cuphy::pinned_alloc>(m_dynDescrSizeBytes);
    m_dynDescrBufGpu = cuphy::buffer<uint8_t, cuphy::device_alloc>(m_dynDescrSizeBytes);

    // Create the PUSCH rate match object.
    cuphyStatus_t statusCreate = cuphyCreatePuschRxRateMatch(&m_puschRmHndl, fpConfig, (int)scrambling);
    if(statusCreate != CUPHY_STATUS_SUCCESS) {
        throw cuphy::cuphy_fn_exception(statusCreate, "cuphyCreatePuschRxRateMatch");
    }
}


LdpcDerateMatch::~LdpcDerateMatch() {
    destroy();
}


void LdpcDerateMatch::derateMatch(const std::vector<cuphy::tensor_ref>& llrs,
                                  void** deRmOutput,
                                  PuschParams& puschParams) {

    uint16_t nUeGrps = puschParams.getNumUeGrps();
    uint16_t nUes = puschParams.m_puschDynPrms.pCellGrpDynPrm->nUes;

    std::vector<cuphyTensorPrm_t> inputLlrs(nUeGrps);
    for(int ueGrpIdx = 0; ueGrpIdx < nUeGrps; ueGrpIdx++) {
        inputLlrs[ueGrpIdx].desc = llrs[ueGrpIdx].desc().handle();
        inputLlrs[ueGrpIdx].pAddr = (void*)llrs[ueGrpIdx].addr();
    }

    const PerTbParams* pTbPrmsCpu = puschParams.getPerTbPrmsCpuPtr();
    const PerTbParams* pTbPrmsGpu = puschParams.getPerTbPrmsGpuPtr();

    derateMatch(inputLlrs, deRmOutput, pTbPrmsCpu, pTbPrmsGpu, nUes);
}


void LdpcDerateMatch::derateMatch(const std::vector<cuphyTensorPrm_t>& inputLlrs,
                                  void** deRmOutput,
                                  const PerTbParams* pTbPrmsCpu,
                                  const PerTbParams* pTbPrmsGpu,
                                  int nUes) {
    int nUeGrps = inputLlrs.size();

    uint16_t nSchUes = 0;
    std::vector<uint16_t> schUserIdxsVec(nUes);
    for(int ueIdx = 0; ueIdx < nUes; ++ueIdx) {
        if(pTbPrmsCpu[ueIdx].isDataPresent) {
            schUserIdxsVec[nSchUes] = ueIdx;
            nSchUes++;
        }
    }

    // Launch config holds everything needed to launch kernel using CUDA driver API.
    cuphyPuschRxRateMatchLaunchCfg_t puschRmLaunchCfg;

    // Setup PUSCH rate match object.
    bool enableCpuToGpuDescrAsyncCpy = false;
    cuphyStatus_t puschRmSetupStatus = cuphySetupPuschRxRateMatch(m_puschRmHndl,
                                                                  nSchUes,
                                                                  schUserIdxsVec.data(),
                                                                  pTbPrmsCpu,
                                                                  pTbPrmsGpu,
                                                                  (cuphyTensorPrm_t*)inputLlrs.data(),
                                                                  (cuphyTensorPrm_t*)inputLlrs.data(),
                                                                  deRmOutput,
                                                                  m_dynDescrBufCpu.addr(),
                                                                  m_dynDescrBufGpu.addr(),
                                                                  enableCpuToGpuDescrAsyncCpy,
                                                                  &puschRmLaunchCfg,
                                                                  m_cuStream);
    if(puschRmSetupStatus != CUPHY_STATUS_SUCCESS) {
        throw cuphy::cuphy_fn_exception(puschRmSetupStatus, "cuphySetupPuschRxRateMatch");
    }

    if(!enableCpuToGpuDescrAsyncCpy) {
        CUDA_CHECK(cudaMemcpyAsync(m_dynDescrBufGpu.addr(), m_dynDescrBufCpu.addr(), m_dynDescrSizeBytes, cudaMemcpyHostToDevice, m_cuStream));
    }

    // Run PUSCH rate match.
    // Launch reset buffer kernel first, then main derate matching kernel.
    
    // Launch reset buffer kernel first using the CUDA driver API.
    const CUDA_KERNEL_NODE_PARAMS& resetKernelNodeParamsDriver = puschRmLaunchCfg.resetKernelNodeParamsDriver;
    CU_CHECK_EXCEPTION_PRINTF_VERSION(cuLaunchKernel(resetKernelNodeParamsDriver.func,
                                                     resetKernelNodeParamsDriver.gridDimX,
                                                     resetKernelNodeParamsDriver.gridDimY,
                                                     resetKernelNodeParamsDriver.gridDimZ,
                                                     resetKernelNodeParamsDriver.blockDimX,
                                                     resetKernelNodeParamsDriver.blockDimY,
                                                     resetKernelNodeParamsDriver.blockDimZ,
                                                     resetKernelNodeParamsDriver.sharedMemBytes,
                                                     static_cast<CUstream>(m_cuStream),
                                                     resetKernelNodeParamsDriver.kernelParams,
                                                     resetKernelNodeParamsDriver.extra));
    
    // Launch main derate matching kernel using the CUDA driver API.
    const CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriver = puschRmLaunchCfg.kernelNodeParamsDriver;
    CU_CHECK_EXCEPTION_PRINTF_VERSION(cuLaunchKernel(kernelNodeParamsDriver.func,
                                                     kernelNodeParamsDriver.gridDimX,
                                                     kernelNodeParamsDriver.gridDimY,
                                                     kernelNodeParamsDriver.gridDimZ,
                                                     kernelNodeParamsDriver.blockDimX,
                                                     kernelNodeParamsDriver.blockDimY,
                                                     kernelNodeParamsDriver.blockDimZ,
                                                     kernelNodeParamsDriver.sharedMemBytes,
                                                     static_cast<CUstream>(m_cuStream),
                                                     kernelNodeParamsDriver.kernelParams,
                                                     kernelNodeParamsDriver.extra));
    
    // Launch clamp kernel using the CUDA driver API.
    const CUDA_KERNEL_NODE_PARAMS& clmapKernelNodeParamsDriver = puschRmLaunchCfg.clampKernelNodeParamsDriver;
    CU_CHECK_EXCEPTION_PRINTF_VERSION(cuLaunchKernel(clmapKernelNodeParamsDriver.func,
                                                     clmapKernelNodeParamsDriver.gridDimX,
                                                     clmapKernelNodeParamsDriver.gridDimY,
                                                     clmapKernelNodeParamsDriver.gridDimZ,
                                                     clmapKernelNodeParamsDriver.blockDimX,
                                                     clmapKernelNodeParamsDriver.blockDimY,
                                                     clmapKernelNodeParamsDriver.blockDimZ,
                                                     clmapKernelNodeParamsDriver.sharedMemBytes,
                                                     static_cast<CUstream>(m_cuStream),
                                                     clmapKernelNodeParamsDriver.kernelParams,
                                                     clmapKernelNodeParamsDriver.extra));
}


void LdpcDerateMatch::destroy() {
    cuphyStatus_t status = cuphyDestroyPuschRxRateMatch(m_puschRmHndl);
    if(status != CUPHY_STATUS_SUCCESS) {
        NVLOGE_FMT(NVLOG_PYAERIAL, AERIAL_PYAERIAL_EVENT,
                   "LdpcDerateMatch::destroy() failed to call cuphyDestroyPuschRxRateMatch()");
    }
}


PyLdpcDerateMatch::PyLdpcDerateMatch(const bool scrambling, const uint64_t cuStream):
m_cuStream((cudaStream_t)cuStream),
m_linearAlloc(getBufferSize()),
m_derateMatch(scrambling, (cudaStream_t)cuStream, 3)  // 3 = FP16 in, FP16 out
{
    // Reserve pinned memory for the output addresses.
    CUDA_CHECK(cudaMallocHost(&m_deRmOutput, sizeof(void*) * MAX_N_TBS_SUPPORTED));
}


PyLdpcDerateMatch::~PyLdpcDerateMatch() {
    CUDA_CHECK_NO_THROW(cudaFreeHost(m_deRmOutput));
}


size_t PyLdpcDerateMatch::getBufferSize() const {
    static constexpr  size_t NUM_BYTES_PER_LLR = sizeof(__half);
    return NUM_BYTES_PER_LLR * MAX_N_TBS_SUPPORTED * MAX_N_CBS_PER_TB_SUPPORTED * MAX_N_RM_LLRS_PER_CB + LINEAR_ALLOC_PAD_BYTES;
}


const std::vector<cuda_array_t<__half>>& PyLdpcDerateMatch::derateMatch(const std::vector<cuda_array_t<__half>>& inputLlrs,
                                                                        const std::vector<uint32_t>& tbSizes,
                                                                        const std::vector<float>& codeRates,
                                                                        const std::vector<uint32_t>& rateMatchLengths,
                                                                        const std::vector<uint8_t>& qamMods,
                                                                        const std::vector<uint8_t>& numLayers,
                                                                        const std::vector<uint32_t>& rvs,
                                                                        const std::vector<uint32_t>& ndis,
                                                                        const std::vector<uint32_t>& cinits,
                                                                        const std::vector<uint32_t>& userGroupIdxs) {

    m_linearAlloc.reset();
    int nUeGrps = inputLlrs.size();
    int nUes = tbSizes.size();
    cuphy::buffer<PerTbParams, cuphy::pinned_alloc> tbPrmsCpu(nUes);
    cuphy::buffer<PerTbParams, cuphy::device_alloc> tbPrmsGpu(nUes);
    std::vector<cuphyTensorPrm_t> dInputLlrs(nUeGrps);
    m_pyDeRmOutput.clear();
    m_inputLlrTensors.resize(nUeGrps);

    for(int ueGrpIdx = 0; ueGrpIdx < nUeGrps; ueGrpIdx++) {

        // Convert the input CuPy array to a device tensor.
        m_inputLlrTensors[ueGrpIdx] = deviceFromCudaArray<__half>(
            inputLlrs[ueGrpIdx],
            nullptr,  // Use the same device buffer as no conversion needed.
            CUPHY_R_16F,
            CUPHY_R_16F,
            cuphy::tensor_flags::align_tight,
            m_cuStream);
        dInputLlrs[ueGrpIdx].pAddr = m_inputLlrTensors[ueGrpIdx].addr();
        dInputLlrs[ueGrpIdx].desc = m_inputLlrTensors[ueGrpIdx].desc().handle();
    }

    cuphyLDPCParams ldpcParams;  // Dummy - values not actually used.

    // Count number of layers per UE group.
    std::vector<uint8_t> numUeGrpLayers(nUeGrps, 0);
    std::vector<std::vector<uint32_t>> layerMapArray(nUes);
    for(int ueIdx = 0; ueIdx < nUes; ueIdx++) {
        layerMapArray[ueIdx].resize(numLayers[ueIdx]);
        for(int layerIdx = 0; layerIdx < numLayers[ueIdx]; layerIdx++) {
            layerMapArray[ueIdx][layerIdx] = numUeGrpLayers[userGroupIdxs[ueIdx]];
            numUeGrpLayers[userGroupIdxs[ueIdx]]++;
        }
    }

    for(int ueIdx = 0; ueIdx < nUes; ueIdx++) {
        setPerTbParams(tbPrmsCpu[ueIdx],
                       ldpcParams,
                       tbSizes[ueIdx],
                       codeRates[ueIdx],
                       qamMods[ueIdx],
                       ndis[ueIdx],
                       rvs[ueIdx],
                       rateMatchLengths[ueIdx],
                       cinits[ueIdx],
                       userGroupIdxs[ueIdx],
                       numLayers[ueIdx],
                       numUeGrpLayers[userGroupIdxs[ueIdx]],
                       layerMapArray[ueIdx]);
    }

    // Copy to GPU.
    CUDA_CHECK(cudaMemcpyAsync(tbPrmsGpu.addr(),
                               tbPrmsCpu.addr(),
                               sizeof(PerTbParams) * nUes,
                               cudaMemcpyHostToDevice,
                               m_cuStream));

    // Reserve output buffers.
    const size_t NUM_BYTES_PER_LLR = sizeof(__half);
    for(int ueIdx = 0; ueIdx < nUes; ++ueIdx) {
        size_t nBytesDeRm = NUM_BYTES_PER_LLR * tbPrmsCpu[ueIdx].Ncb_padded * tbPrmsCpu[ueIdx].num_CBs;
        m_deRmOutput[ueIdx] = m_linearAlloc.alloc(nBytesDeRm);
    }

    // Run the derate matching.
    m_pyDeRmOutput.reserve(nUes);
    const PerTbParams* pTbPrmsCpu = tbPrmsCpu.addr();
    const PerTbParams* pTbPrmsGpu = tbPrmsGpu.addr();
    m_derateMatch.derateMatch(dInputLlrs, m_deRmOutput, pTbPrmsCpu, pTbPrmsGpu, nUes);

    for(int ueIdx = 0; ueIdx < nUes; ueIdx++) {
        std::vector<size_t> shape = {tbPrmsCpu[ueIdx].Ncb + 2 * tbPrmsCpu[ueIdx].Zc, tbPrmsCpu[ueIdx].num_CBs};
        std::vector<size_t> strides = {sizeof(__half), sizeof(__half) * tbPrmsCpu[ueIdx].Ncb_padded};
        m_pyDeRmOutput.push_back(deviceToCudaArray<__half>(m_deRmOutput[ueIdx], shape, strides));
    }

    return m_pyDeRmOutput;
}

}  // namespace pycuphy
