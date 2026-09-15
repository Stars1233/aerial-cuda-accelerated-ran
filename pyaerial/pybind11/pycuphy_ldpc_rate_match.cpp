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

#include <iostream>
#include <memory>
#include <stdexcept>
#include <algorithm>
#include <ranges>
#include <bit>
#include "cuphy.h"
#include "util.hpp"
#include "utils.cuh"
#include "cuda_array_interface.hpp"
#include "cuphy_internal.h"
#include "pdsch_dmrs/pdsch_dmrs.hpp"
#include "pycuphy_util.hpp"
#include "pycuphy_ldpc.hpp"
#include "pycuphy_csirs_util.hpp"


namespace py = pybind11;

namespace pycuphy {

LdpcRateMatch::LdpcRateMatch(const EnableScrambling scrambling,
                             const uint16_t nPrbDlBwp,
                             const uint32_t maxNumTbs,
                             const uint32_t maxNumCodeBlocks,
                             const cudaStream_t cuStream):
m_scrambling(scrambling),
m_nPrbDlBwp(nPrbDlBwp),
m_cuStream(cuStream),
m_csiRsReMapper(nPrbDlBwp, cuStream) {

    // Allocate workspace and descriptors for rate matching.
    const auto allocatedWorkspaceSize = cuphyDlRateMatchingWorkspaceSize(static_cast<int>(maxNumTbs));
    m_dRmWorkspace = cuphy::buffer<uint32_t, cuphy::device_alloc>(allocatedWorkspaceSize);
    m_hRmWorkspace = cuphy::buffer<uint32_t, cuphy::pinned_alloc>((2 + 2) * maxNumTbs);

    size_t descSize = 0, allocSize = 0;
    if (cuphyStatus_t status = cuphyDlRateMatchingGetDescrInfo(&descSize, &allocSize); status != CUPHY_STATUS_SUCCESS) {
        NVLOGE_FMT(NVLOG_PYAERIAL, AERIAL_PYAERIAL_EVENT, "cuphyDlRateMatchingGetDescrInfo error {}", status);
        throw std::runtime_error("LdpcRateMatch::LdpcRateMatch: cuphyDlRateMatchingGetDescrInfo error!");
    }
    m_dRmDesc = cuphy::buffer<uint8_t, cuphy::device_alloc>(descSize);
    m_hRmDesc = cuphy::buffer<uint8_t, cuphy::pinned_alloc>(descSize);

    const auto nRmOutputElems = div_round_up<uint32_t>(maxNumTbs * maxNumCodeBlocks * PDSCH_MAX_ER_PER_CB_BITS, sizeof(uint32_t) * CHAR_BIT);
    m_rmOutput = cuphy::make_unique_device<uint32_t>(nRmOutputElems);

    m_ueGrpPrms = cuphy::buffer<PdschUeGrpParams, cuphy::pinned_alloc>(PDSCH_MAX_UE_GROUPS_PER_CELL_GROUP);
    m_ueGrpPrmsDev = cuphy::buffer<PdschUeGrpParams, cuphy::device_alloc>(PDSCH_MAX_UE_GROUPS_PER_CELL_GROUP);
    memset(m_ueGrpPrms.addr(), 0, PDSCH_MAX_UE_GROUPS_PER_CELL_GROUP * sizeof(PdschUeGrpParams));
}

const cuphy::tensor_device& LdpcRateMatch::rateMatch(const cuphy::tensor_device& dInputBits,
                                                     const cuphy::buffer<PdschPerTbParams, cuphy::pinned_alloc>& tbParams,
                                                     const cuphy::buffer<PdschPerTbParams, cuphy::device_alloc>& tbParamsDev,
                                                     const cuphy::buffer<PdschDmrsParams, cuphy::pinned_alloc>& dmrsParams,
                                                     const cuphy::buffer<PdschDmrsParams, cuphy::device_alloc>& dmrsParamsDev,
                                                     const cuphy::buffer<cuphyCsirsRrcDynPrm_t, cuphy::pinned_alloc>& csiRsParams,
                                                     const cuphy::buffer<cuphyCsirsRrcDynPrm_t, cuphy::device_alloc>& csiRsParamsDev,
                                                     const uint32_t numTbs,
                                                     const uint32_t numCsiRsConfigs) {
    // Overprovisioned output buffer.
    uint32_t maxNumCbs = 0;
    for (int i = 0; i < numTbs; i++) {
        maxNumCbs = std::max(maxNumCbs, tbParams[i].num_CBs);
    }
    const auto maxNumOutputElems = div_round_up<uint32_t>(numTbs * maxNumCbs * PDSCH_MAX_ER_PER_CB_BITS, sizeof(uint32_t) * CHAR_BIT);
    m_dOutputTensor = cuphy::tensor_device(m_rmOutput.get(), CUPHY_R_32U, maxNumOutputElems);

    constexpr uint8_t enableModLayerMap = 0;
    constexpr uint32_t numUeGrps = 1;
    run(dInputBits,
        m_dOutputTensor,
        tbParams,
        tbParamsDev,
        dmrsParams,
        dmrsParamsDev,
        csiRsParams,
        csiRsParamsDev,
        numTbs,
        numCsiRsConfigs,
        numUeGrps,
        enableModLayerMap);

    return m_dOutputTensor;
}


void LdpcRateMatch::run(const cuphy::tensor_device& dInputBits,
                        const cuphy::tensor_device& dOutputTensor,
                        const cuphy::buffer<PdschPerTbParams, cuphy::pinned_alloc>& tbParams,
                        const cuphy::buffer<PdschPerTbParams, cuphy::device_alloc>& tbParamsDev,
                        const cuphy::buffer<PdschDmrsParams, cuphy::pinned_alloc>& dmrsParams,
                        const cuphy::buffer<PdschDmrsParams, cuphy::device_alloc>& dmrsParamsDev,
                        const cuphy::buffer<cuphyCsirsRrcDynPrm_t, cuphy::pinned_alloc>& csiRsParams,
                        const cuphy::buffer<cuphyCsirsRrcDynPrm_t, cuphy::device_alloc>& csiRsParamsDev,
                        const uint32_t numTbs,
                        const uint32_t numCsiRsConfigs,
                        const uint32_t numUeGrps,
                        const uint8_t enableModLayerMap) {

    // Allocate launch config struct.
    const std::unique_ptr<cuphyDlRateMatchingLaunchConfig> rmHandle = std::make_unique<cuphyDlRateMatchingLaunchConfig>();

    uint32_t* dRmOutputAddr = nullptr;
    void* dModOutputAddr = nullptr;
    uint8_t enablePrecoding = 0;
    if(enableModLayerMap) {
        dModOutputAddr = dOutputTensor.addr();
        enablePrecoding = std::any_of(dmrsParams.addr(),
                                      dmrsParams.addr() + numTbs,
                                      [](const PdschDmrsParams& prm) { return prm.enablePrcdBf; });
    }
    else {
        dRmOutputAddr = static_cast<uint32_t*>(dOutputTensor.addr());
    }

    // Set the correct tb_idx for each UE grp (first TB of the UE group)
    for (int i = 0; i < numUeGrps; i++) {
        for (int j = 0; j < numTbs; j++) {
            if(dmrsParams[j].ueGrp_idx == i) {
                m_ueGrpPrms[i].tb_idx = j;
                break;
            }
        }
    }
    CUDA_CHECK(cudaMemcpyAsync(m_ueGrpPrmsDev.addr(),
                               m_ueGrpPrms.addr(),
                               sizeof(PdschUeGrpParams) * numUeGrps,
                               cudaMemcpyHostToDevice,
                               m_cuStream));

    // Run CSI-RS mapping if configs are provided, else just set the RE map to nullptr.
    void* reMap = nullptr;
    if (numCsiRsConfigs > 0) {
        reMap = m_csiRsReMapper.run(dmrsParamsDev,
                                    csiRsParams,
                                    csiRsParamsDev,
                                    m_ueGrpPrmsDev,
                                    numTbs,
                                    numCsiRsConfigs,
                                    numUeGrps);
    }

    // Setup DL rate matching object
    constexpr bool interCellBatching = false;
    constexpr bool restructureKernel = false;
    constexpr uint8_t descAsyncCopy = 1;        // Copy descriptor to the GPU during setup.
    constexpr uint8_t postFecRmScramblingProcessing = 0U;
    cuphyPdschStatusOut_t pdschStatusOut{};     // Populated during cuphySetupDlRateMatching, but contents not used here
    cuphyStatus_t status = cuphySetupDlRateMatching(rmHandle.get(),
                                                    &pdschStatusOut,
                                                    static_cast<const uint32_t*>(dInputBits.addr()),
                                                    dRmOutputAddr,
                                                    nullptr,
                                                    dModOutputAddr,
                                                    reMap,
                                                    m_nPrbDlBwp,
                                                    numTbs,
                                                    0,  // Number of layers, not used
                                                    static_cast<uint8_t>(m_scrambling),
                                                    enableModLayerMap,  // Both modulation and layer mapping
                                                    enableModLayerMap,  // either enabled or disabled
                                                    enablePrecoding,
                                                    restructureKernel,
                                                    interCellBatching,
                                                    postFecRmScramblingProcessing, // post-FEC-RM scrambling mode not supported here
                                                    m_hRmWorkspace.addr(),
                                                    m_dRmWorkspace.addr(),  // Explicit H2D copy as part of setup
                                                    const_cast<PdschPerTbParams*>(tbParams.addr()),
                                                    const_cast<PdschPerTbParams*>(tbParamsDev.addr()),
                                                    const_cast<PdschDmrsParams*>(dmrsParamsDev.addr()),
                                                    m_ueGrpPrmsDev.addr(),
                                                    m_hRmDesc.addr(),
                                                    m_dRmDesc.addr(),
                                                    descAsyncCopy,
                                                    m_cuStream);

    if (status != CUPHY_STATUS_SUCCESS) {
        NVLOGE_FMT(NVLOG_PYAERIAL, AERIAL_PYAERIAL_EVENT, "cuphySetupDlRateMatching error {}", status);
        throw std::runtime_error("LdpcRateMatch::run: cuphySetupDlRateMatching error!");
    }

    // Record the number of rate matched output bits per code block.
    uint32_t totNumCbs = 0;
    for (int i = 0; i < numTbs; i++) {
        totNumCbs += tbParams[i].num_CBs;
    }
    m_numRmBitsPerCb.resize(totNumCbs);
    const uint32_t* Er = m_hRmWorkspace.addr() + 2 * numTbs;
    uint32_t cb = 0;
    for(int tbIdx = 0; tbIdx < numTbs; tbIdx++) {
        for (int cbIdx = 0; cbIdx < tbParams[tbIdx].num_CBs; cbIdx++) {
            uint32_t numRmBits = Er[tbIdx * 2 + 1] + ((cbIdx < Er[tbIdx * 2]) ? 0 : tbParams[tbIdx].Nl * tbParams[tbIdx].Qm);
            m_numRmBitsPerCb[cb++] = numRmBits;
        }
    }
    uint32_t maxEr = *std::ranges::max_element(m_numRmBitsPerCb);
    static constexpr uint32_t ELEMENT_SIZE = sizeof(uint32_t) * CHAR_BIT;
    maxEr = div_round_up<uint32_t>(maxEr, ELEMENT_SIZE) * ELEMENT_SIZE;
    if (maxEr > PDSCH_MAX_ER_PER_CB_BITS) {
        NVLOGE_FMT(NVLOG_PYAERIAL, AERIAL_PYAERIAL_EVENT,  "Emax {} but supported maximum Emax is {}", maxEr, PDSCH_MAX_ER_PER_CB_BITS);
        throw std::runtime_error("Emax exceeds max supported!");
    }

    // Run the kernel.
    if(CUresult r = launch_kernel_ex(rmHandle->m_kernelNodeParams[0], m_cuStream, true); r != CUDA_SUCCESS) {
        NVLOGE_FMT(NVLOG_PYAERIAL, AERIAL_PYAERIAL_EVENT, "Rate matching kernel launch failed!");
        throw std::runtime_error("LdpcRateMatch::run: Invalid argument for kernel launch!");
    }
}


CsiRsReMapper::CsiRsReMapper(uint16_t nPrbDlBwp, cudaStream_t cuStream):
m_nPrbDlBwp(nPrbDlBwp),
m_cuStream(cuStream) {

    // Allocate workspace and descriptors for CSI-RS preparation.
    m_dCsiRsOffsets = cuphy::buffer<uint32_t, cuphy::device_alloc>(CUPHY_CSIRS_MAX_NUM_PARAMS + 1);  // Storing one extra offset in the end of the array.
    m_hCsiRsOffsets = cuphy::buffer<uint32_t, cuphy::pinned_alloc>(CUPHY_CSIRS_MAX_NUM_PARAMS + 1);
    m_dCsiRsCellIndex = cuphy::buffer<uint32_t, cuphy::device_alloc>(CUPHY_CSIRS_MAX_NUM_PARAMS);
    m_hCsiRsCellIndex = cuphy::buffer<uint32_t, cuphy::pinned_alloc>(CUPHY_CSIRS_MAX_NUM_PARAMS);

    size_t descSize = 0, allocSize = 0;
    if (cuphyStatus_t status = cuphyPdschCsirsPrepGetDescrInfo(&descSize, &allocSize); status != CUPHY_STATUS_SUCCESS) {
        NVLOGE_FMT(NVLOG_PYAERIAL, AERIAL_PYAERIAL_EVENT, "cuphyPdschCsirsPrepGetDescrInfo error {}", status);
        throw std::runtime_error("LdpcRateMatch::LdpcRateMatch: cuphyPdschCsirsPrepGetDescrInfo error!");
    }
    m_dCsiRsPrepDesc = cuphy::buffer<uint8_t, cuphy::device_alloc>(descSize);
    m_hCsiRsPrepDesc = cuphy::buffer<uint8_t, cuphy::pinned_alloc>(descSize);

    // Allocate RE map array
    const size_t nReMapElems = m_nPrbDlBwp * CUPHY_N_TONES_PER_PRB * OFDM_SYMBOLS_PER_SLOT + OFDM_SYMBOLS_PER_SLOT;
    m_reMap = cuphy::make_unique_device<uint16_t>(nReMapElems);
}

void* CsiRsReMapper::run(const cuphy::buffer<PdschDmrsParams, cuphy::device_alloc>& dmrsParams,
                         const cuphy::buffer<cuphyCsirsRrcDynPrm_t, cuphy::pinned_alloc>& csiRsParams,
                         const cuphy::buffer<cuphyCsirsRrcDynPrm_t, cuphy::device_alloc>& csiRsParamsDev,
                         const cuphy::buffer<PdschUeGrpParams, cuphy::device_alloc>& ueGrpPrms,
                         const uint32_t numTbs,
                         const uint32_t numCsiRsConfigs,
                         const uint32_t numUeGrps) {
    uint32_t offset = 0;
    for(int i = 0; i < numCsiRsConfigs; i++) {
        m_hCsiRsOffsets[i] = offset;
        m_hCsiRsCellIndex[i] = m_nPrbDlBwp; // Cell index 0 since we only have one cell, BWP size

        const uint32_t nRB = csiRsParams[i].nRb;
        const uint8_t row = csiRsParams[i].row;

        if (row == 0 || row > std::size(csirsRowDataNumPorts)) {
            throw std::runtime_error(fmt::format(
                "CsiRsReMapper::run: invalid CSI-RS row index {} against size of csirsRowDataNumPorts {}",
                row,
                std::size(csirsRowDataNumPorts)));
        }

        const uint32_t numElements = nRB * ((row == 1) ? 3 : csirsRowDataNumPorts[row - 1]);
        offset += (numElements + 31) & ~31;
    }
    m_hCsiRsOffsets[numCsiRsConfigs] = offset;

    CUDA_CHECK(cudaMemcpyAsync(m_dCsiRsOffsets.addr(), m_hCsiRsOffsets.addr(),
                              sizeof(uint32_t) * (numCsiRsConfigs + 1),
                              cudaMemcpyHostToDevice, m_cuStream));

    CUDA_CHECK(cudaMemcpyAsync(m_dCsiRsCellIndex.addr(), m_hCsiRsCellIndex.addr(),
                              sizeof(uint32_t) * numCsiRsConfigs,
                              cudaMemcpyHostToDevice, m_cuStream));

    // Create launch config
    const auto launchConfig = std::make_unique<cuphyPdschCsirsPrepLaunchConfig>();

    // Setup CSI-RS preprocessing
    constexpr uint8_t descAsyncCopy = 1;
    constexpr uint16_t numCells = 1;
    cuphyStatus_t status = cuphySetupPdschCsirsPreprocessing(
        launchConfig.get(),
        m_reMap.get(),
        const_cast<cuphyCsirsRrcDynPrm_t*>(csiRsParamsDev.addr()),
        numCsiRsConfigs,
        offset,
        m_dCsiRsOffsets.addr(),
        m_dCsiRsCellIndex.addr(),
        numUeGrps,
        const_cast<PdschUeGrpParams*>(ueGrpPrms.addr()),
        const_cast<PdschDmrsParams*>(dmrsParams.addr()),
        m_nPrbDlBwp,
        numCells,
        m_hCsiRsPrepDesc.addr(),
        m_dCsiRsPrepDesc.addr(),
        descAsyncCopy,
        m_cuStream
    );

    if (status != CUPHY_STATUS_SUCCESS) {
        NVLOGE_FMT(NVLOG_PYAERIAL, AERIAL_PYAERIAL_EVENT, "cuphySetupPdschCsirsPreprocessing error {}", status);
        throw std::runtime_error("CsiRsReMapper::run: cuphySetupPdschCsirsPreprocessing error!");
    }

    // Launch the kernels
    constexpr std::array launchOrder = {2, 0, 1};
    for (const auto i : launchOrder) {
        if (CUresult r = launch_kernel(launchConfig->m_kernelNodeParams[i], m_cuStream); r != CUDA_SUCCESS) {
            NVLOGE_FMT(NVLOG_PYAERIAL, AERIAL_PYAERIAL_EVENT, "CSI-RS preprocessing kernel {} launch failed!", i);
            throw std::runtime_error("CsiRsReMapper::run: Invalid argument for kernel launch!");
        }
    }

    return m_reMap.get();
}


PyLdpcRateMatch::PyLdpcRateMatch(const EnableScrambling scrambling,
                                 const uint16_t nPrbDlBwp,
                                 const uint32_t maxNumTbs,
                                 const uint32_t maxNumCodeBlocks,
                                 const uint64_t cuStream):
m_ldpcRateMatch(scrambling,
                nPrbDlBwp,
                maxNumTbs,
                maxNumCodeBlocks,
                std::bit_cast<cudaStream_t>(cuStream)),
m_nPrbDlBwp(nPrbDlBwp),
m_cuStream(std::bit_cast<cudaStream_t>(cuStream)) {

    m_tbParams = cuphy::buffer<PdschPerTbParams, cuphy::pinned_alloc>(maxNumTbs);
    m_tbParamsDev = cuphy::buffer<PdschPerTbParams, cuphy::device_alloc>(maxNumTbs);

    m_dmrsParams = cuphy::buffer<PdschDmrsParams, cuphy::pinned_alloc>(maxNumTbs);
    m_dmrsParamsDev = cuphy::buffer<PdschDmrsParams, cuphy::device_alloc>(maxNumTbs);

    m_csiRsParams = cuphy::buffer<cuphyCsirsRrcDynPrm_t, cuphy::pinned_alloc>(CUPHY_CSIRS_MAX_NUM_PARAMS);
    m_csiRsParamsDev = cuphy::buffer<cuphyCsirsRrcDynPrm_t, cuphy::device_alloc>(CUPHY_CSIRS_MAX_NUM_PARAMS);

    constexpr size_t nTxBufElems = MAX_N_PRBS_SUPPORTED * CUPHY_N_TONES_PER_PRB * OFDM_SYMBOLS_PER_SLOT * MAX_DL_LAYERS;
    m_txBufHalf = cuphy::make_unique_device<__half2>(nTxBufElems);
}


cuphy::tensor_device PyLdpcRateMatch::getInputTensor(const cuda_array_uint32& inputBits) {
    // Input data is packed to 32-bit integers, assumed done when calling from Python.
    return deviceFromCudaArray<uint32_t>(inputBits,
                                         nullptr,
                                         CUPHY_R_32U,
                                         CUPHY_R_32U,
                                         cuphy::tensor_flags::align_tight,
                                         m_cuStream);
}


void PyLdpcRateMatch::rmModLayerMap(const cuda_array_uint32& inputBits,
                                    cuda_array_complex_float& txBuffer,
                                    const std::vector<py::object>& pdschConfigs,
                                    const std::vector<py::object>& csiRsConfigs) {
    const uint16_t numUeGrps = pdschConfigs.size();

    const cuphy::tensor_device dInputBits = getInputTensor(inputBits);

    // Convert Tx buffer to half precision.
    const cuphy::tensor_device dTxBuffer = deviceFromCudaArray<std::complex<float>>(
        txBuffer,
        m_txBufHalf.get(),
        CUPHY_C_32F,
        CUPHY_C_16F,
        cuphy::tensor_flags::align_tight,
        m_cuStream);

    // Unused parameters.
    constexpr uint32_t slot = 0;
    constexpr uint16_t cellId = 0;

    uint32_t numTbs = 0;
    readDmrsParams(pdschConfigs,
                   slot,
                   cellId,
                   m_nPrbDlBwp,
                   m_txBufHalf.get(),
                   numTbs,
                   m_dmrsParams);

    readTbParams(pdschConfigs,
                 nullptr,  // TB input not used, not needed.
                 numTbs,
                 m_tbParams);

    // Copy TB and DMRS parameters from host to device.
    CUDA_CHECK(cudaMemcpyAsync(m_tbParamsDev.addr(),
                               m_tbParams.addr(),
                               sizeof(PdschPerTbParams) * numTbs,
                               cudaMemcpyHostToDevice,
                               m_cuStream));
    CUDA_CHECK(cudaMemcpyAsync(m_dmrsParamsDev.addr(),
                               m_dmrsParams.addr(),
                               sizeof(PdschDmrsParams) * numTbs,
                               cudaMemcpyHostToDevice,
                               m_cuStream));

    // Handle CSI-RS parameters if provided
    const size_t numCsiRsConfigs = csiRsConfigs.size();

    if (numCsiRsConfigs > CUPHY_CSIRS_MAX_NUM_PARAMS) {
        throw std::runtime_error(fmt::format(
            "PyLdpcRateMatch::rmModLayerMap: "
            "numCsiRsConfigs ({}) exceeds maximum number of configs ({})",
            numCsiRsConfigs,
            CUPHY_CSIRS_MAX_NUM_PARAMS));
    }

    if (numCsiRsConfigs > 0) {
        // Parse CSI-RS parameters
        for (size_t i = 0; i < numCsiRsConfigs; ++i) {
            readCsiRsRrcDynPrms(csiRsConfigs[i], m_csiRsParams[i]);
        }

        // Copy CSI-RS parameters to device
        CUDA_CHECK(cudaMemcpyAsync(m_csiRsParamsDev.addr(),
                                   m_csiRsParams.addr(),
                                   sizeof(cuphyCsirsRrcDynPrm_t) * numCsiRsConfigs,
                                   cudaMemcpyHostToDevice,
                                   m_cuStream));
    }

    // Run rate matching.
    constexpr uint8_t enableModLayerMap = 1;
    m_ldpcRateMatch.run(dInputBits,
                        dTxBuffer,
                        m_tbParams,
                        m_tbParamsDev,
                        m_dmrsParams,
                        m_dmrsParamsDev,
                        m_csiRsParams,
                        m_csiRsParamsDev,
                        numTbs,
                        numCsiRsConfigs,
                        numUeGrps,
                        enableModLayerMap);

    // Convert back to full precision, write output.
    const auto& shape = txBuffer.get_shape();
    txBuffer = deviceToCudaArray<std::complex<float>>(
        dTxBuffer.addr(),
        txBuffer.get_device_ptr(),
        shape,
        CUPHY_C_16F,
        CUPHY_C_32F,
        cuphy::tensor_flags::align_tight,
        m_cuStream);
}


const cuda_array_t<uint32_t>& PyLdpcRateMatch::rateMatch(const cuda_array_uint32& inputBits,
                                                         const std::vector<uint32_t>& tbSizes,
                                                         const std::vector<float>& codeRates,
                                                         const std::vector<uint32_t>& rateMatchLens,
                                                         const std::vector<uint8_t>& modOrders,
                                                         const std::vector<uint8_t>& numLayers,
                                                         const std::vector<uint8_t>& redundancyVersions,
                                                         const std::vector<uint32_t>& cinits) {
    const cuphy::tensor_device dInputTensor = getInputTensor(inputBits);

    // Check that all vector arguments have the same length
    const size_t numTbs = tbSizes.size();
    if (codeRates.size() != numTbs ||
        rateMatchLens.size() != numTbs ||
        modOrders.size() != numTbs ||
        numLayers.size() != numTbs ||
        redundancyVersions.size() != numTbs ||
        cinits.size() != numTbs) {
        throw std::runtime_error("PyLdpcRateMatch::rateMatch: All vector arguments must have the same length!");
    }

    // Set transport block parameters.
    uint32_t maxNumCbs = 0;
    for (int i = 0; i < numTbs; i++) {
        setPdschPerTbParams(
            m_tbParams[i],
            nullptr,  // tbStartAddr not used
            0,        // tbStartOffset not used
            tbSizes[i],
            0,        // cumulativeTbSizePadding, not used here
            codeRates[i],
            rateMatchLens[i],
            modOrders[i],
            0,  // numCodedBits, can be given for verification
            redundancyVersions[i],
            numLayers[i],
            cinits[i]
        );
        maxNumCbs = std::max(maxNumCbs, m_tbParams[i].num_CBs);
    }

    // Copy TB parameters from host to device.
    CUDA_CHECK(cudaMemcpyAsync(m_tbParamsDev.addr(),
                               m_tbParams.addr(),
                               sizeof(PdschPerTbParams) * numTbs,
                               cudaMemcpyHostToDevice,
                               m_cuStream));

    // Run rate matching.
    constexpr uint32_t numCsiRsConfigs = 0;  // We run this with no CSI-RS parameters.
    const cuphy::tensor_device& dOutputTensor = m_ldpcRateMatch.rateMatch(dInputTensor,
                                                                          m_tbParams,
                                                                          m_tbParamsDev,
                                                                          m_dmrsParams,
                                                                          m_dmrsParamsDev,
                                                                          m_csiRsParams,
                                                                          m_csiRsParamsDev,
                                                                          numTbs,
                                                                          numCsiRsConfigs);
    if(dOutputTensor.rank() != 1) {
        throw std::runtime_error("PyLdpcRateMatch::rateMatch: Invalid output data dimensions!");
    }

    // Return to Python.
    const std::vector<uint32_t>& numRmBitsPerCb = m_ldpcRateMatch.getNumRmBitsPerCb();
    const uint32_t maxEr = *std::ranges::max_element(numRmBitsPerCb);

    static constexpr uint32_t ELEMENT_SIZE = sizeof(uint32_t) * CHAR_BIT;
    const std::vector shape = {numTbs * static_cast<size_t>(div_round_up(maxEr, ELEMENT_SIZE) * maxNumCbs)};
    m_rmBits = deviceToCudaArrayPtr<uint32_t>(dOutputTensor.addr(), shape);
    return *m_rmBits;
}


}  // namespace pycuphy
