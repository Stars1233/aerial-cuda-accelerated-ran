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
#include <string>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "cuphy_api.h"
#include "hdf5hpp.hpp"
#include "pycuphy_params.hpp"
#include "pycuphy_pusch.hpp"

namespace py = pybind11;

namespace pycuphy {

PuschPipeline::PuschPipeline(const py::object& statPrms, uint64_t cuStream) {
    m_puschParams.setStatPrms(statPrms);

    if(m_puschParams.m_puschStatPrms.pDbg && m_puschParams.m_puschStatPrms.pDbg->pOutFileName) {
        std::string outFilename = std::string(m_puschParams.m_puschStatPrms.pDbg->pOutFileName);
        if (outFilename != "None") {
            m_debugFile.reset(new hdf5hpp::hdf5_file(hdf5hpp::hdf5_file::create(m_puschParams.m_puschStatPrms.pDbg->pOutFileName)));
        }
    }

    createPuschRx((cudaStream_t)cuStream);
}


PuschPipeline::~PuschPipeline() {
    destroyPuschRx();
}


void PuschPipeline::createPuschRx(cudaStream_t cuStream) {
    cuphyStatus_t status = cuphyCreatePuschRx(&m_puschHandle, &m_puschParams.m_puschStatPrms, cuStream);
    if(status != CUPHY_STATUS_SUCCESS) {
        throw cuphy::cuphy_fn_exception(status, "cuphyCreatePuschRx()");
    }
}


void PuschPipeline::setupPuschRx(const py::object& dynPrms) {

    m_puschParams.setDynPrms(dynPrms);
    cudaStream_t cuStream = m_puschParams.m_puschDynPrms.phase1Stream;

    cuphySetupPuschRx(m_puschHandle, &m_puschParams.m_puschDynPrms, nullptr);
    CUDA_CHECK(cudaStreamSynchronize(cuStream));

    if (m_puschParams.m_puschDynPrms.setupPhase == PUSCH_SETUP_PHASE_1) {

        py::object dataOut = dynPrms.attr("dataOut");

        // If we need UCI output in the future, should have
        // a function after run() to return those value to Python.

        // Return output data size.
        uint32_t* pTotNumTbs          = numpyArrayToPtr<uint32_t>(dataOut.attr("totNumTbs"));
        uint32_t* pTotNumCbs          = numpyArrayToPtr<uint32_t>(dataOut.attr("totNumCbs"));
        uint32_t* pTotNumPayloadBytes = numpyArrayToPtr<uint32_t>(dataOut.attr("totNumPayloadBytes"));
        uint16_t* pTotNumUciSegs      = numpyArrayToPtr<uint16_t>(dataOut.attr("totNumUciSegs"));

        pTotNumTbs[0]          = m_puschParams.m_puschDynPrms.pDataOut->totNumTbs;
        pTotNumCbs[0]          = m_puschParams.m_puschDynPrms.pDataOut->totNumCbs;
        pTotNumPayloadBytes[0] = m_puschParams.m_puschDynPrms.pDataOut->totNumPayloadBytes;
        pTotNumUciSegs[0]      = m_puschParams.m_puschDynPrms.pDataOut->totNumUciSegs;
    }
}


void PuschPipeline::runPuschRx() {

    cuphyStatus_t status = cuphyRunPuschRx(m_puschHandle, PUSCH_RUN_ALL_PHASES);
    if(status != CUPHY_STATUS_SUCCESS) {
        throw cuphy::cuphy_fn_exception(status, "cuphyRunPuschRx()");
    }
}


void PuschPipeline::writeDbgBufSynch() {

    cudaStream_t cuStream = m_puschParams.m_puschDynPrms.phase1Stream;

    CUDA_CHECK(cudaStreamSynchronize(cuStream));

    cuphyStatus_t status = cuphyWriteDbgBufSynch(m_puschHandle, cuStream);
    if(status != CUPHY_STATUS_SUCCESS) {
        throw cuphy::cuphy_fn_exception(status, "cuphyWriteDbgBufSynch()");
    }
}


void PuschPipeline::destroyPuschRx() {
    cuphyStatus_t status = cuphyDestroyPuschRx(m_puschHandle);
    if(status != CUPHY_STATUS_SUCCESS) {
        throw cuphy::cuphy_fn_exception(status, "cuphyDestroyPuschRx()");
    }
}


} // pycuphy
