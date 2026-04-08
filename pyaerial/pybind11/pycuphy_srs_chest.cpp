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

#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "cuphy.h"
#include "cuda_array_interface.hpp"
#include "pycuphy_util.hpp"
#include "tensor_desc.hpp"
#include "pycuphy_srs_chest.hpp"
#include "pycuphy_srs_util.hpp"
#include "srs_rx.hpp"

namespace py = pybind11;

namespace pycuphy {

constexpr uint16_t SRS_BW_TABLE[64][8] =
   {{4,1,4,1,4,1,4,1},
    {8,1,4,2,4,1,4,1},
    {12,1,4,3,4,1,4,1},
    {16,1,4,4,4,1,4,1},
    {16,1,8,2,4,2,4,1},
    {20,1,4,5,4,1,4,1},
    {24,1,4,6,4,1,4,1},
    {24,1,12,2,4,3,4,1},
    {28,1,4,7,4,1,4,1},
    {32,1,16,2,8,2,4,2},
    {36,1,12,3,4,3,4,1},
    {40,1,20,2,4,5,4,1},
    {48,1,16,3,8,2,4,2},
    {48,1,24,2,12,2,4,3},
    {52,1,4,13,4,1,4,1},
    {56,1,28,2,4,7,4,1},
    {60,1,20,3,4,5,4,1},
    {64,1,32,2,16,2,4,4},
    {72,1,24,3,12,2,4,3},
    {72,1,36,2,12,3,4,3},
    {76,1,4,19,4,1,4,1},
    {80,1,40,2,20,2,4,5},
    {88,1,44,2,4,11,4,1},
    {96,1,32,3,16,2,4,4},
    {96,1,48,2,24,2,4,6},
    {104,1,52,2,4,13,4,1},
    {112,1,56,2,28,2,4,7},
    {120,1,60,2,20,3,4,5},
    {120,1,40,3,8,5,4,2},
    {120,1,24,5,12,2,4,3},
    {128,1,64,2,32,2,4,8},
    {128,1,64,2,16,4,4,4},
    {128,1,16,8,8,2,4,2},
    {132,1,44,3,4,11,4,1},
    {136,1,68,2,4,17,4,1},
    {144,1,72,2,36,2,4,9},
    {144,1,48,3,24,2,12,2},
    {144,1,48,3,16,3,4,4},
    {144,1,16,9,8,2,4,2},
    {152,1,76,2,4,19,4,1},
    {160,1,80,2,40,2,4,10},
    {160,1,80,2,20,4,4,5},
    {160,1,32,5,16,2,4,4},
    {168,1,84,2,28,3,4,7},
    {176,1,88,2,44,2,4,11},
    {184,1,92,2,4,23,4,1},
    {192,1,96,2,48,2,4,12},
    {192,1,96,2,24,4,4,6},
    {192,1,64,3,16,4,4,4},
    {192,1,24,8,8,3,4,2},
    {208,1,104,2,52,2,4,13},
    {216,1,108,2,36,3,4,9},
    {224,1,112,2,56,2,4,14},
    {240,1,120,2,60,2,4,15},
    {240,1,80,3,20,4,4,5},
    {240,1,48,5,16,3,8,2},
    {240,1,24,10,12,2,4,3},
    {256,1,128,2,64,2,4,16},
    {256,1,128,2,32,4,4,8},
    {256,1,16,16,8,2,4,2},
    {264,1,132,2,44,3,4,11},
    {272,1,136,2,68,2,4,17},
    {272,1,68,4,4,17,4,1},
    {272,1,16,17,8,2,4,2}};


PySrsChannelEstimator::PySrsChannelEstimator(uint8_t chEstAlgoIdx, uint8_t chEstToL2NormalizationAlgo, float  chEstToL2Constantscaler, uint8_t enableDelayOffsetCorrection, const py::dict& chEstParams, uint64_t cuStream):
m_cuStream(reinterpret_cast<cudaStream_t>(cuStream)),
m_srsChEstimator(reinterpret_cast<cudaStream_t>(cuStream)),
m_tensorPrms(chEstParams, reinterpret_cast<cudaStream_t>(cuStream)),
m_nSrsUes(0) {
    m_srsChEstimator.init(m_tensorPrms.getSrsFilterPrms(),
                          m_tensorPrms.getSrsRkhsPrms(),
                          static_cast<cuphySrsChEstAlgoType_t>(chEstAlgoIdx),
                          chEstToL2NormalizationAlgo, 
                          chEstToL2Constantscaler,
                          enableDelayOffsetCorrection);

    // Pre-allocate device memory for Rx data in 16-bit float format (needs conversion).
    static constexpr size_t nBytes = sizeof(__half) * MAX_N_PRBS_SUPPORTED * CUPHY_N_TONES_PER_PRB * MAX_N_ANTENNAS_SUPPORTED * CUPHY_SRS_MAX_FULL_BAND_CHEST_PER_TTI;
    CUDA_CHECK(cudaMalloc((void**)&m_pRxData, nBytes));
}


PySrsChannelEstimator::~PySrsChannelEstimator() {
    CUDA_CHECK_NO_THROW(cudaFree(m_pRxData));
}


const std::vector<cuda_array_t<std::complex<float>>>& PySrsChannelEstimator::estimate(
        const cuda_array_t<std::complex<float>>& inputData,
        uint16_t nSrsUes,
        uint16_t nCells,
        uint16_t nPrbGrps,
        uint16_t startPrbGrp,
        const std::vector<py::object>& pySrsCellPrms,
        const std::vector<py::object>& pyUeSrsPrms) {

    m_nSrsUes = nSrsUes;

    m_srsReports.resize(nSrsUes);
    m_chEsts.clear();

    // Convert Python structs into cuPHY.
    std::vector<cuphySrsCellPrms_t> srsCellPrms;
    readSrsCellParams(srsCellPrms, pySrsCellPrms);
    std::vector<cuphyUeSrsPrm_t> ueSrsPrms;
    readUeSrsParams(ueSrsPrms, pyUeSrsPrms);

    // Read input data into device memory.
    std::vector<cuphyTensorPrm_t> tDataRx(nCells);
    cuphy::tensor_device deviceRxDataTensor = deviceFromCudaArray<std::complex<float>>(
        inputData,
        m_pRxData,
        CUPHY_C_32F,
        CUPHY_C_16F,
        cuphy::tensor_flags::align_tight,
        m_cuStream);

    // Only one cell supported through the Python API for now.
    tDataRx[0].desc = deviceRxDataTensor.desc().handle();
    tDataRx[0].pAddr = deviceRxDataTensor.addr();

    std::vector<cuphySrsChEstBuffInfo_t> srsChEstBuffInfo = m_srsChEstimator.estimate(tDataRx,
                                                                                      nSrsUes,
                                                                                      nCells,
                                                                                      nPrbGrps,
                                                                                      startPrbGrp,
                                                                                      srsCellPrms,
                                                                                      ueSrsPrms);

    // Create the return values.
    for(int ueIdx=0; ueIdx < nSrsUes; ueIdx++) {
        cuphyTensorPrm_t* pChEst = &srsChEstBuffInfo[ueIdx].tChEstBuffer;
        const ::tensor_desc& tDesc = static_cast<const ::tensor_desc&>(*pChEst->desc);
        const ::tensor_layout_any& tLayout = tDesc.layout();

        size_t nPrbGrpEsts = tLayout.dimensions[0];
        size_t nGnbAnts = tLayout.dimensions[1];
        size_t nUeAnts = tLayout.dimensions[2];

        std::vector shape = {nPrbGrpEsts, nGnbAnts, nUeAnts};
        m_chEsts.push_back(deviceToCudaArray<std::complex<float>>(pChEst->pAddr, shape));
    }

    cuphySrsReport_t* pSrsReports = m_srsChEstimator.getSrsReport();
    CUDA_CHECK(cudaMemcpyAsync(m_srsReports.data(),
                               pSrsReports,
                               sizeof(cuphySrsReport_t) * nSrsUes,
                               cudaMemcpyDeviceToHost,
                               m_cuStream));
    CUDA_CHECK(cudaStreamSynchronize(m_cuStream));
    return m_chEsts;
}


cuda_array_t<float> PySrsChannelEstimator::getRbSnrBuffer() const {
    std::vector shape = {static_cast<size_t>(m_nPrbs), static_cast<size_t>(m_nSrsUes)};
    return deviceToCudaArray<float>(m_srsChEstimator.getRbSnrBuffer(), shape);
}


size_t SrsChannelEstimator::getBufferSize() const {
    static constexpr size_t maxNumSrsUes = CUPHY_SRS_MAX_N_USERS;
    static constexpr size_t extraPadding = maxNumSrsUes * 128; // Upper bound for extra memory required per allocation due to 128 alignment
    static constexpr size_t maxCells = 3;  // TODO: Only one cell currently.
    static constexpr size_t maxSrsReportMem = maxNumSrsUes * sizeof(cuphySrsReport_t) + extraPadding;

    const size_t maxChEstToL2Mem     = maxCells * m_nPrbs * MAX_N_ANTENNAS_SUPPORTED * CUPHY_SRS_MAX_FULL_BAND_SRS_ANT_PORTS_SLOT_PER_CELL * sizeof(float2) * CUPHY_SRS_MAX_FULL_BAND_CHEST_PER_TTI * 2 + extraPadding;
    const size_t maxRbSnrMem         = maxNumSrsUes * m_nPrbs * sizeof(float) + extraPadding;
    const size_t maxRkhsWorkspaceMem = maxCells * CUPHY_SRS_RKHS_WORKSPACE_SIZE_PER_CELL + extraPadding; // TODO: only allocate if RKHS configured in static paramaters

    const size_t nBytesBuffer = maxRbSnrMem + maxSrsReportMem + maxChEstToL2Mem + maxRkhsWorkspaceMem;
    return nBytesBuffer;
}


void SrsChannelEstimator::allocateDescr() {
    size_t statDescrAlignBytes, dynDescrAlignBytes;
    cuphyStatus_t status = cuphySrsChEstGetDescrInfo(&m_statDescrSizeBytes,
                                                     &statDescrAlignBytes,
                                                     &m_dynDescrSizeBytes,
                                                     &dynDescrAlignBytes);
    if(status != CUPHY_STATUS_SUCCESS) {
        throw cuphy::cuphy_fn_exception(status, "cuphySrsChEstGetDescrInfo()");
    }

    m_statDescrBufCpu = std::move(cuphy::buffer<uint8_t, cuphy::pinned_alloc>(m_statDescrSizeBytes));
    m_statDescrBufGpu = std::move(cuphy::buffer<uint8_t, cuphy::device_alloc>(m_statDescrSizeBytes));
    m_dynDescrBufCpu = std::move(cuphy::buffer<uint8_t, cuphy::pinned_alloc>(m_dynDescrSizeBytes));
    m_dynDescrBufGpu = std::move(cuphy::buffer<uint8_t, cuphy::device_alloc>(m_dynDescrSizeBytes));
}


SrsChannelEstimator::SrsChannelEstimator(cudaStream_t cuStream):
m_cuStream(cuStream),
m_linearAlloc(getBufferSize()) {}


SrsChannelEstimator::SrsChannelEstimator(const cuphySrsFilterPrms_t& srsFilterPrms, const cuphySrsRkhsPrms_t& srsRkhsPrms, cuphySrsChEstAlgoType_t chEstAlgoType, uint8_t chEstToL2NormalizationAlgo, float  chEstToL2Constantscaler, uint8_t enableDelayOffsetCorrection, cudaStream_t cuStream):
m_cuStream(cuStream),
m_linearAlloc(getBufferSize()) {
    init(srsFilterPrms, srsRkhsPrms, chEstAlgoType, chEstToL2NormalizationAlgo, chEstToL2Constantscaler, enableDelayOffsetCorrection);
}


void SrsChannelEstimator::init(const cuphySrsFilterPrms_t& srsFilterPrms, const cuphySrsRkhsPrms_t& srsRkhsPrms, cuphySrsChEstAlgoType_t chEstAlgoType, uint8_t chEstToL2NormalizationAlgo, float  chEstToL2Constantscaler, uint8_t enableDelayOffsetCorrection) {

    m_srsFilterPrms = srsFilterPrms;
    m_srsRkhsPrms   = srsRkhsPrms;
    m_chEstAlgoType = chEstAlgoType;
    m_chEstToL2NormalizationAlgo = chEstToL2NormalizationAlgo;
    m_chEstToL2Constantscaler = chEstToL2Constantscaler;
    m_enableDelayOffsetCorrection = enableDelayOffsetCorrection;

    // Allocate descriptors.
    allocateDescr();

    // Create the SRS channel estimator object.
    bool enableCpuToGpuDescrAsyncCpy = true;

    cuphyStatus_t status = cuphyCreateSrsChEst(&m_srsChEstHndl,
                                               &m_srsFilterPrms,
                                               &m_srsRkhsPrms,
                                               m_chEstAlgoType,
                                               m_chEstToL2NormalizationAlgo,
                                               m_chEstToL2Constantscaler,
                                               m_enableDelayOffsetCorrection,
                                               static_cast<uint8_t>(enableCpuToGpuDescrAsyncCpy),
                                               m_statDescrBufCpu.addr(),
                                               m_statDescrBufGpu.addr(),
                                               m_cuStream);
    if(status != CUPHY_STATUS_SUCCESS) {
        throw cuphy::cuphy_fn_exception(status, "cuphyCreateSrsChEst()");
    }
}


SrsChannelEstimator::~SrsChannelEstimator() {
    // Destroy the SRS channel estimation handle.
    cuphyStatus_t status = cuphyDestroySrsChEst(m_srsChEstHndl);
    if(status != CUPHY_STATUS_SUCCESS) {
        NVLOGE_FMT(NVLOG_PYAERIAL, AERIAL_PYAERIAL_EVENT,
                   "SrsChannelEstimator::~SrsChannelEstimator() failed to call cuphyDestroySrsChEst()");
    }
}


const std::vector<cuphySrsChEstBuffInfo_t>& SrsChannelEstimator::estimate(
        const std::vector<cuphyTensorPrm_t>& tDataRx,
        uint16_t nSrsUes,
        uint16_t nCells,
        uint16_t nPrbGrps,
        uint16_t startPrbGrp,
        const std::vector<cuphySrsCellPrms_t>& srsCellPrms,
        const std::vector<cuphyUeSrsPrm_t>& ueSrsPrms) {

    m_nSrsUes = nSrsUes;
    m_nCells = nCells;

    m_linearAlloc.reset();

    m_tSrsChEstVec.clear();
    m_srsChEstBuffInfo.resize(nSrsUes);
    m_chEstCpuBuffVec.resize(nSrsUes);
    m_dChEstToL2InnerVec.resize(nSrsUes);
    m_dChEstToL2Vec.resize(nSrsUes);
    m_chEstToL2Vec.resize(nSrsUes);
    m_srsRbSnrBuffOffsets.resize(nSrsUes);

    // Initializations.
    uint32_t rbSnrBufferSize = nSrsUes * m_nPrbs * sizeof(float);
    m_dSrsRbSnrBuffer = static_cast<float*>(m_linearAlloc.alloc(rbSnrBufferSize));
    for(int ueIdx = 0; ueIdx < nSrsUes; ++ueIdx) {

        uint16_t cellIdx = ueSrsPrms[ueIdx].cellIdx;
        uint16_t nRxAntSrs = srsCellPrms[cellIdx].nRxAntSrs;
        uint8_t prgSize = ueSrsPrms[ueIdx].prgSize;
        uint16_t nPrbGrpsPerHop = SRS_BW_TABLE[ueSrsPrms[ueIdx].configIdx][2 * ueSrsPrms[ueIdx].bandwidthIdx] / prgSize;
        uint16_t nHops = ueSrsPrms[ueIdx].nSyms / ueSrsPrms[ueIdx].nRepetitions;
        uint16_t nAntPorts = ueSrsPrms[ueIdx].nAntPorts;

        m_srsRbSnrBuffOffsets[ueIdx] = ueIdx * m_nPrbs;

        m_tSrsChEstVec.push_back(cuphy::tensor_device(CUPHY_C_32F,
                                                      nPrbGrps,
                                                      nRxAntSrs,
                                                      nAntPorts,
                                                      cuphy::tensor_flags::align_tight));

        m_srsChEstBuffInfo[ueIdx].tChEstBuffer.desc = m_tSrsChEstVec[ueIdx].desc().handle();
        m_srsChEstBuffInfo[ueIdx].tChEstBuffer.pAddr = m_tSrsChEstVec[ueIdx].addr();
        m_srsChEstBuffInfo[ueIdx].startPrbGrp = startPrbGrp;

        // Allocate buffers for ChEst to L2.
        size_t maxChEstSize = nPrbGrpsPerHop * nRxAntSrs * nHops * nAntPorts * sizeof(float2);
        m_chEstCpuBuffVec[ueIdx] = std::move(cuphy::buffer<uint8_t, cuphy::pinned_alloc>(maxChEstSize));
        m_dChEstToL2InnerVec[ueIdx] = m_linearAlloc.alloc(maxChEstSize);
        m_dChEstToL2Vec[ueIdx] = m_linearAlloc.alloc(maxChEstSize);
        m_chEstToL2Vec[ueIdx].pChEstCpuBuff = m_chEstCpuBuffVec[ueIdx].addr();

    }
    m_dSrsReports = static_cast<cuphySrsReport_t*>(m_linearAlloc.alloc(sizeof(cuphySrsReport_t) * nSrsUes));
    m_linearAlloc.memset(0., m_cuStream);

    void* d_workspace = m_linearAlloc.alloc(CUPHY_SRS_RKHS_WORKSPACE_SIZE_PER_CELL * 3);

    // Launch config holds everything needed to launch kernel using CUDA driver API or add it to a graph
    cuphySrsChEstLaunchCfg_t srsChEstLaunchCfg;
    cuphySrsChEstNormalizationLaunchCfg_t srsChEstNormalizationLaunchCfg;

    // Setup function populates dynamic descriptor and launch config. Option to copy descriptors to GPU during setup call.
    bool  enableCpuToGpuDescrAsyncCpy = false;
    cuphyStatus_t setupStatus = cuphySetupSrsChEst(m_srsChEstHndl,
                                                   nSrsUes,
                                                   const_cast<cuphyUeSrsPrm_t*>(ueSrsPrms.data()),
                                                   nCells,
                                                   const_cast<cuphyTensorPrm_t*>(tDataRx.data()),
                                                   const_cast<cuphySrsCellPrms_t*>(srsCellPrms.data()),
                                                   m_dSrsRbSnrBuffer,
                                                   m_srsRbSnrBuffOffsets.data(),
                                                   m_dSrsReports,
                                                   m_srsChEstBuffInfo.data(),
                                                   m_dChEstToL2InnerVec.data(),
                                                   m_dChEstToL2Vec.data(),
                                                   m_chEstToL2Vec.data(),
                                                   d_workspace,
                                                   static_cast<uint8_t>(enableCpuToGpuDescrAsyncCpy),
                                                   m_dynDescrBufCpu.addr(),
                                                   m_dynDescrBufGpu.addr(),
                                                   &srsChEstLaunchCfg,
                                                   &srsChEstNormalizationLaunchCfg,
                                                   m_cuStream);
    if(!enableCpuToGpuDescrAsyncCpy) {
        CUDA_CHECK(cudaMemcpyAsync(m_dynDescrBufGpu.addr(),
                                   m_dynDescrBufCpu.addr(),
                                   m_dynDescrSizeBytes,
                                   cudaMemcpyHostToDevice,
                                   m_cuStream));
    }

    const CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriver = srsChEstLaunchCfg.kernelNodeParamsDriver;
    CUresult srsChEstRunStatus = cuLaunchKernel(kernelNodeParamsDriver.func,
                                                 kernelNodeParamsDriver.gridDimX,
                                                 kernelNodeParamsDriver.gridDimY,
                                                 kernelNodeParamsDriver.gridDimZ,
                                                 kernelNodeParamsDriver.blockDimX,
                                                 kernelNodeParamsDriver.blockDimY,
                                                 kernelNodeParamsDriver.blockDimZ,
                                                 kernelNodeParamsDriver.sharedMemBytes,
                                                 m_cuStream,
                                                 kernelNodeParamsDriver.kernelParams,
                                                 kernelNodeParamsDriver.extra);
    if(srsChEstRunStatus != CUDA_SUCCESS) {
        throw cuphy::cuphy_exception(CUPHY_STATUS_INTERNAL_ERROR);
    }

    return m_srsChEstBuffInfo;
}


} // namespace pycuphy
