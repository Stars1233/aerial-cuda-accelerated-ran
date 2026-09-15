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

#ifndef PYCUPHY_PARAMS_HPP
#define PYCUPHY_PARAMS_HPP

#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "cuphy.h"
#include "cuphy.hpp"
#include "cuphy_api.h"
#include "cuphy_hdf5.hpp"
#include "pusch_utils.hpp"
#include "pycuphy_util.hpp"

namespace py = pybind11;

namespace pycuphy {


// Utilities for debugging.
void printPdschStatPrms(const cuphyPdschStatPrms_t& statPrms);
void printPdschDynPrms(const cuphyPdschDynPrms_t& dynPrms);
void printPdschCellGrpDynPrms(const cuphyPdschCellGrpDynPrm_t& cellGrpDynPrms);
void printPuschStatPrms(const cuphyPuschStatPrms_t& statPrms);
void printPuschDynPrms(const cuphyPuschDynPrms_t& dynPrms);
void printPerTbParams(const char *desciption, const uint16_t nSchUes, const PerTbParams* tbPrms);
void printPuschDrvdUeGrpPrms(const cuphyPuschRxUeGrpPrms_t& puschRxUeGrpPrms);


class PdschParams {

public:
    PdschParams(const py::object& statPrms);
    ~PdschParams();
    void setDynPrms(const py::object& dynPrms);
    void printStatPrms() const;
    void printDynPrms() const;

    cuphyPdschDynPrms_t     m_pdschDynPrms;
    cuphyPdschStatPrms_t    m_pdschStatPrms;

private:

    // Static parameters.
    std::vector<cuphyPdschDbgPrms_t>    m_dbgPrm;
    std::vector<std::string>            m_dbgFilenames;
    std::vector<cuphyCellStatPrm_t>     m_cellStatPrms;
    cuphyTracker_t                      m_tracker;

    // Dynamic parameters.
    cuphyPdschCellGrpDynPrm_t           m_cellGrpDynPrms;
    std::vector<cuphyPdschCellAerialMetrics_t>  m_cellMetrics;
    std::vector<cuphyPdschCellDynPrm_t> m_cellDynPrms;
    std::vector<cuphyPdschUeGrpPrm_t>   m_ueGrpPrms;
    std::vector<cuphyPdschUePrm_t>      m_uePrms;
    std::vector<cuphyPdschCwPrm_t>      m_cwPrms;
    std::vector<cuphyPdschDmrsPrm_t>    m_pdschDmrsPrms;
    std::vector<cuphyCsirsRrcDynPrm_t>  m_csirsPrms;
    std::vector<cuphyPmW_t>             m_pmwPrms;

    cuphyPdschDataIn_t                  m_dataIn;
    cuphyPdschDataIn_t                  m_tbCrcDataIn;
    cuphyPdschDataOut_t                 m_outputData;
    std::vector<cuphyPdschStatusOut_t>  m_outputStatus;
    std::vector<cuphyTensorPrm_t>       m_outputTensorPrm;
    std::vector<cuphy::tensor_device>   m_dataTxTensor;
    std::vector<uint8_t*>               m_tbInputPtr;
};


class PuschParams {

public:
    PuschParams();

    // Python API.
    void setFilters(const py::array& WFreq,
                    const py::array& WFreq4,
                    const py::array& WFreqSmall,
                    const py::array& shiftSeq,
                    const py::array& shiftSeq4,
                    const py::array& unShiftSeq,
                    const py::array& unShiftSeq4,
                    uint64_t cuStream = 0);

    void setStatPrms(const py::object& statPrms);
    void setDynPrms(const py::object& dynPrms);

    // C++ API.
    void setStatPrms(const cuphyPuschStatPrms_t& statPrms);
    void setDynPrms(const cuphyPuschDynPrms_t& dynPrms);

    // Copy from host to device.
    void copyPuschRxUeGrpPrms();
    void copyPerTbPrms();

    // Printing utilities
    void printStatPrms() const;
    void printDynPrms() const;
    void printPerPuschTbParams(const char* desc) const;
    void printDrvdUeGrpPrms() const;

    // Getters.
    uint16_t getNumCells() const { return m_puschDynPrms.pCellGrpDynPrm->nCells; }
    uint16_t getNumUeGrps() const { return m_puschDynPrms.pCellGrpDynPrm->nUeGrps; }
    uint16_t getNumUes() const { return m_puschDynPrms.pCellGrpDynPrm->nUes; }
    uint32_t getMaxNumPrb() const { return m_nMaxPrb; }

    std::vector<cuphy::tensor_device>& getDataTensor() { return m_tDataRx; }

    cuphyPuschRxUeGrpPrms_t* getPuschRxUeGrpPrmsCpuPtr() const { return m_drvdUeGrpPrmsCpu; }
    cuphyPuschRxUeGrpPrms_t* getPuschRxUeGrpPrmsGpuPtr() const { return m_drvdUeGrpPrmsGpu; }

    const cuphyLDPCParams& getLdpcPrms() const { return m_ldpcParams; }
    const PerTbParams* getPerTbPrmsCpuPtr() const { return m_tbPrmsCpu.addr(); }
    const PerTbParams* getPerTbPrmsGpuPtr() const { return m_tbPrmsGpu.addr(); }

    cuphyPuschStatPrms_t    m_puschStatPrms{};
    cuphyPuschDynPrms_t     m_puschDynPrms{};

    void setChestFactorySettingsFilename(const std::string& filename) { m_puschrxChestFactorySettingsFilename = filename; }

private:
    void setDynPrmsPhase1(const py::object& dynPrms);
    void setDynPrmsPhase2(const py::object& dynPrms);
    uint32_t updatePuschRxUeGrpPrms(cudaStream_t cuStream);
    size_t getBufferSize() const;

    // Storage for static parameters when the structures are generated from Python input.
    // Note: These do not get set when instantiated from C++ using cuPHY structures.
    cuphyTracker_t                                  m_tracker;
    std::vector<cuphyCellStatPrm_t>                 m_cellStatPrms;
    std::vector<cuphyPuschCellStatPrm_t>            m_puschCellStatPrms;
    cuphyPuschStatDbgPrms_t                         m_puschStatDbgPrms;
    std::string                                     m_debugFilename;
    std::string                                     m_puschrxChestFactorySettingsFilename;
    cuphy::buffer<uint32_t, cuphy::device_alloc>    m_bSymRxStatus;
    cudaEvent_t                                     m_subSlotCompletedEvent;
    cudaEvent_t                                     m_waitCompletedSubSlotEvent;
    cudaEvent_t                                     m_waitCompletedFullSlotEvent;
    cudaEvent_t                                     m_uciOnPuschCompletedEvent;

    cuphyTensorPrm_t m_tPrmWFreq, m_tPrmWFreq4, m_tPrmWFreqSmall, m_tPrmShiftSeq, m_tPrmUnShiftSeq, m_tPrmShiftSeq4, m_tPrmUnShiftSeq4;
    cuphy::tensor_device m_tWFreq, m_tWFreq4, m_tWFreqSmall, m_tShiftSeq, m_tUnshiftSeq, m_tShiftSeq4, m_tUnshiftSeq4;

    // Storage for dynamic parameters when the structures are generated from Python input.
    // Note: These do not get set when instantiated from C++ using cuPHY structures.
    cuphyPuschStatusOut_t                           m_puschStatusOutput;
    std::vector<cuphyPuschCellDynPrm_t>             m_cellDynPrms;
    cuphyPuschCellGrpDynPrm_t                       m_cellGrpDynPrms;
    std::vector<cuphyPuschUeGrpPrm_t>               m_ueGrpPrms;
    std::vector<std::vector<uint16_t>>              m_uePrmIdxs;
    std::vector<cuphyPuschDmrsPrm_t>                m_dmrsPrms;
    std::vector<cuphyPuschUePrm_t>                  m_uePrms;
    cuphyPuschDynDbgPrms_t                          m_puschDynDbgPrms;
    cuphyPuschDataIn_t                              m_puschDataIn;
    cuphyPuschDataInOut_t                           m_puschDataInOut;
    cuphyPuschDataOut_t                             m_puschDataOut;
    cuphy::buffer<uint8_t*, cuphy::pinned_alloc>    m_bHarqBufferPtrs = std::move(cuphy::buffer<uint8_t*, cuphy::pinned_alloc>(MAX_N_TBS_SUPPORTED));
    cuphy::buffer<float*, cuphy::pinned_alloc>      m_bFoCompensationBufferPtrs = std::move(cuphy::buffer<float*, cuphy::pinned_alloc>(MAX_N_TBS_SUPPORTED));

    // These get set also when instantiated from cuPHY structures.
    uint32_t                            m_nMaxPrb;
    std::vector<cuphy::tensor_device>   m_tDataRx;
    std::vector<cuphyTensorPrm_t>       m_tPrmDataRx;

    // Backend parameters.
    cuphyLDPCParams                                 m_ldpcParams;
    cuphy::buffer<PerTbParams, cuphy::pinned_alloc> m_tbPrmsCpu;
    cuphy::buffer<PerTbParams, cuphy::device_alloc> m_tbPrmsGpu;

    static constexpr uint32_t LINEAR_ALLOC_PAD_BYTES = 128;
    cuphy::linear_alloc<LINEAR_ALLOC_PAD_BYTES, cuphy::device_alloc> m_LinearAlloc;

    size_t m_descrSizeBytes;
    cuphy::buffer<uint8_t, cuphy::pinned_alloc> m_drvdUeGrpPrmsCpuBuf;
    cuphy::buffer<uint8_t, cuphy::device_alloc> m_drvdUeGrpPrmsGpuBuf;
    cuphyPuschRxUeGrpPrms_t* m_drvdUeGrpPrmsCpu;
    cuphyPuschRxUeGrpPrms_t* m_drvdUeGrpPrmsGpu;
    cuphy::buffer<uint8_t, cuphy::pinned_alloc> m_workCancelBuffer;
};



} // namespace pycuphy


#endif // PYCUPHY_PARAMS_HPP
