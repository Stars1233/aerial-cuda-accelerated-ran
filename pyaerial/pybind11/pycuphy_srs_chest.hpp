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

#ifndef PYCUPHY_SRS_CHEST_HPP
#define PYCUPHY_SRS_CHEST_HPP

#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "cuphy.h"
#include "cuda_array_interface.hpp"
#include "pycuphy_srs_util.hpp"

namespace py = pybind11;

namespace pycuphy {


// This is the interface when called from C++.
class SrsChannelEstimator {
public:
    // Runs init() internally.
    SrsChannelEstimator(const cuphySrsFilterPrms_t& srsFilterPrms, const cuphySrsRkhsPrms_t& srsRkhsPrms, cuphySrsChEstAlgoType_t chEstAlgoType, uint8_t chEstToL2NormalizationAlgo, float  chEstToL2Constantscaler, uint8_t enableDelayOffsetCorrection, cudaStream_t cuStream);

    // When using this, init() needs to be run separately, use this when the filter parameters
    // are not available upon instatiating the object.
    SrsChannelEstimator(cudaStream_t cuStream);
    ~SrsChannelEstimator();

    void init(const cuphySrsFilterPrms_t& srsFilterPrms, const cuphySrsRkhsPrms_t& srsRkhsPrms, cuphySrsChEstAlgoType_t chEstAlgoType, uint8_t chEstToL2NormalizationAlgo, float  chEstToL2Constantscaler, uint8_t enableDelayOffsetCorrection);

    const std::vector<cuphySrsChEstBuffInfo_t>& estimate(const std::vector<cuphyTensorPrm_t>& tDataRx,
                                                         uint16_t nSrsUes,
                                                         uint16_t nCells,
                                                         uint16_t nPrbGrps,
                                                         uint16_t startPrbGrp,
                                                         const std::vector<cuphySrsCellPrms_t>& srsCellPrms,
                                                         const std::vector<cuphyUeSrsPrm_t>& ueSrsPrms);

    // Getters for other estimation results after running estimate().
    cuphySrsReport_t* getSrsReport() const { return m_dSrsReports; }
    float* getRbSnrBuffer() const { return m_dSrsRbSnrBuffer; }
    const std::vector<uint32_t>& getRbSnrBufferOffsets() const { return m_srsRbSnrBuffOffsets; }

private:
    size_t getBufferSize() const;
    void allocateDescr();

    uint16_t m_nSrsUes;
    uint16_t m_nCells;
    static constexpr uint16_t m_nPrbs = 273;

    cuphy::linear_alloc<128, cuphy::device_alloc> m_linearAlloc;

    // Descriptor variables.
    size_t m_statDescrSizeBytes;
    size_t m_dynDescrSizeBytes;
    cuphy::buffer<uint8_t, cuphy::pinned_alloc> m_statDescrBufCpu;
    cuphy::buffer<uint8_t, cuphy::device_alloc> m_statDescrBufGpu;
    cuphy::buffer<uint8_t, cuphy::pinned_alloc> m_dynDescrBufCpu;
    cuphy::buffer<uint8_t, cuphy::device_alloc> m_dynDescrBufGpu;

    // SRS estimator handle.
    cuphySrsChEstHndl_t m_srsChEstHndl;

    cuphySrsFilterPrms_t    m_srsFilterPrms;
    cuphySrsRkhsPrms_t      m_srsRkhsPrms;
    cuphySrsChEstAlgoType_t m_chEstAlgoType;
    uint8_t                 m_chEstToL2NormalizationAlgo;
    float                   m_chEstToL2Constantscaler;
    uint8_t                 m_enableDelayOffsetCorrection;

    cudaStream_t m_cuStream;

    cuphySrsReport_t* m_dSrsReports;

    std::vector<cuphy::tensor_device> m_tSrsChEstVec;
    std::vector<cuphySrsChEstBuffInfo_t> m_srsChEstBuffInfo;

    std::vector<cuphy::buffer<uint8_t, cuphy::pinned_alloc>> m_chEstCpuBuffVec;
    std::vector<void*> m_dChEstToL2InnerVec;
    std::vector<void*> m_dChEstToL2Vec;
    std::vector<cuphySrsChEstToL2_t> m_chEstToL2Vec;

    float* m_dSrsRbSnrBuffer;
    std::vector<uint32_t> m_srsRbSnrBuffOffsets;
};


// This is the interface towards pybind11.
class __attribute__((visibility("default"))) PySrsChannelEstimator {
public:
    PySrsChannelEstimator(uint8_t chEstAlgoIdx, uint8_t chEstToL2NormalizationAlgo, float chEstToL2Constantscaler, uint8_t enableDelayOffsetCorrection, const py::dict& chEstParams, uint64_t cuStream);
    ~PySrsChannelEstimator();

    const std::vector<cuda_array_t<std::complex<float>>>& estimate(const cuda_array_t<std::complex<float>>& inputData,
                                                                   uint16_t nSrsUes,
                                                                   uint16_t nCells,
                                                                   uint16_t nPrbGrps,
                                                                   uint16_t startPrbGrp,
                                                                   const std::vector<py::object>& srsCellPrms,
                                                                   const std::vector<py::object>& ueSrsPrms);

    const std::vector<cuphySrsReport_t>& getSrsReport() const { return m_srsReports; }
    cuda_array_t<float> getRbSnrBuffer() const;
    const std::vector<uint32_t>& getRbSnrBufferOffsets() { return m_srsChEstimator.getRbSnrBufferOffsets(); }

private:
    SrsChannelEstimator m_srsChEstimator;

    uint16_t m_nSrsUes;
    static constexpr uint16_t m_nPrbs = 273;

    // Filter tensors and parameters.
    SrsTensorPrms m_tensorPrms;

    cudaStream_t m_cuStream;

    void* m_pRxData;

    // Outputs.
    std::vector<cuda_array_t<std::complex<float>>> m_chEsts;
    std::vector<cuphySrsReport_t> m_srsReports;

};

} // namespace pycuphy

#endif // PYCUPHY_SRS_CHEST_HPP