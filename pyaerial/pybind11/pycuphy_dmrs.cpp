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

#include <bit>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "cuda_array_interface.hpp"
#include "pycuphy_dmrs.hpp"
#include "pycuphy_util.hpp"
#include <gsl-lite/gsl-lite.hpp>


namespace py = pybind11;

namespace pycuphy {


PdschDmrsTx::PdschDmrsTx(const cudaStream_t cuStream):
m_cuStream(cuStream) {
    size_t dynDescrSizeBytes{}, dynDescrAlignBytes{};
    if(cuphyStatus_t status = cuphyPdschDmrsGetDescrInfo(&dynDescrSizeBytes, &dynDescrAlignBytes); status != CUPHY_STATUS_SUCCESS) {
        NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT, "cuphyPdschDmrsGetDescrInfo error {}", status);
        throw std::runtime_error("PdschDmrsTx::PdschDmrsTx: cuphyPdschDmrsGetDescrInfo error!");
    }
    gsl_Expects(dynDescrSizeBytes > 0 && dynDescrAlignBytes > 0);
    dynDescrSizeBytes = ((dynDescrSizeBytes + (dynDescrAlignBytes - 1)) / dynDescrAlignBytes) * dynDescrAlignBytes;
    m_dynDescrBufCpu = cuphy::buffer<uint8_t, cuphy::pinned_alloc>(dynDescrSizeBytes);
    m_dynDescrBufGpu = cuphy::buffer<uint8_t, cuphy::device_alloc>(dynDescrSizeBytes);
}


void PdschDmrsTx::run(const cuphy::buffer<PdschDmrsParams, cuphy::pinned_alloc>& dmrsParams,
                      const cuphy::buffer<PdschDmrsParams, cuphy::device_alloc>& dmrsParamsDev,
                      const uint32_t numTbs) {

    const uint8_t enablePrecoding = std::any_of(dmrsParams.addr(),
                                                dmrsParams.addr() + numTbs,
                                                [](const PdschDmrsParams& prm) { return prm.enablePrcdBf; });

    cuphyPdschDmrsLaunchConfig dmrsLaunchCfg{};
    const cuphyTensorDescriptor_t desc{};  // Dummy since not actually used.
    constexpr uint8_t descAsyncCopy = 1;
    const cuphyStatus_t status = cuphySetupPdschDmrs(&dmrsLaunchCfg,
                                                     const_cast<PdschDmrsParams*>(dmrsParamsDev.addr()),
                                                     numTbs,
                                                     enablePrecoding,
                                                     desc,        // Not used
                                                     nullptr,     // Not used, Tx buffer address is in DMRS params
                                                     m_dynDescrBufCpu.addr(),
                                                     m_dynDescrBufGpu.addr(),
                                                     descAsyncCopy,
                                                     m_cuStream);
    if (status != CUPHY_STATUS_SUCCESS) {
        NVLOGE_FMT(NVLOG_PYAERIAL, AERIAL_PYAERIAL_EVENT, "cuphySetupPdschDmrs error {}", status);
        throw std::runtime_error("PdschDmrsTx::run: cuphySetupPdschDmrs error!");
    }

   if(CUresult dmrsStatus = launch_kernel(dmrsLaunchCfg.m_kernelNodeParams, m_cuStream); dmrsStatus != CUDA_SUCCESS) {
        NVLOGE_FMT(NVLOG_PYAERIAL, AERIAL_PYAERIAL_EVENT, "DMRS kernel launch error {}", dmrsStatus);
        throw std::runtime_error("PdschDmrsTx::run: DMRS kernel failed!");
   }
}


PyPdschDmrsTx::PyPdschDmrsTx(const uint64_t cuStream,
                             const uint32_t maxNumCells,
                             const uint32_t maxNumTbs):
m_dmrsTx(std::bit_cast<cudaStream_t>(cuStream)),
m_cuStream(std::bit_cast<cudaStream_t>(cuStream)) {

    // Allocate TX buffer memory.
    // TODO: This is no longer needed when CuPy supports cp.complex32 (no conversion needed).

    m_txBufHalf.resize(maxNumCells);
    constexpr size_t nElems =  MAX_N_PRBS_SUPPORTED * CUPHY_N_TONES_PER_PRB * OFDM_SYMBOLS_PER_SLOT * MAX_DL_LAYERS;
    for(auto& buf : m_txBufHalf) {
        buf = cuphy::make_unique_device<__half2>(nElems);
    }

    // Buffers get (possibly over-)provisioned to maxNumTbs.
    m_dmrsParams = cuphy::buffer<PdschDmrsParams, cuphy::pinned_alloc>(maxNumTbs);
    m_dmrsParamsDev = cuphy::buffer<PdschDmrsParams, cuphy::device_alloc>(maxNumTbs);
}

void PyPdschDmrsTx::run(std::vector<cuda_array_complex_float>& txBuffers,
                        const uint32_t slot,
                        const std::vector<py::object>& dmrsParams) {

    const uint32_t numTbs = dmrsParams.size();
    const uint32_t numCells = txBuffers.size();

    // DMRS Tx takes half-precision complex float, need conversion here.
    // TODO: Remove when CuPy supports cp.complex32.
    m_txTensors.resize(numCells);
    for(int cellIdx = 0; auto& tensor : m_txTensors) {
        const cuda_array_complex_float& cellTxBuffer = txBuffers[cellIdx];
        tensor = deviceFromCudaArray<std::complex<float>>(
            cellTxBuffer,
            m_txBufHalf[cellIdx].get(),
            CUPHY_C_32F,
            CUPHY_C_16F,
            cuphy::tensor_flags::align_tight,
            m_cuStream);
        cellIdx++;
    }

    // Read DMRS parameters from the Python object.
    // Set TX buffer address in DMRS params.
    readDmrsParams(dmrsParams, slot);

    // Copy DMRS parameters to device.
    CUDA_CHECK(cudaMemcpyAsync(m_dmrsParamsDev.addr(),
                               m_dmrsParams.addr(),
                               sizeof(PdschDmrsParams) * numTbs,
                               cudaMemcpyHostToDevice,
                               m_cuStream));

    // Run DMRS Tx.
    m_dmrsTx.run(m_dmrsParams, m_dmrsParamsDev, numTbs);

    // Convert back to full precision, write output.
    for (int cellIdx = 0; cellIdx < numCells; cellIdx++) {
        const auto& shape = txBuffers[cellIdx].get_shape();
        txBuffers[cellIdx] = deviceToCudaArray<std::complex<float>>(
            m_txTensors[cellIdx].addr(),
            txBuffers[cellIdx].get_device_ptr(),
            shape,
            CUPHY_C_16F,
            CUPHY_C_32F,
            cuphy::tensor_flags::align_tight,
            m_cuStream
        );
    }
}


void PyPdschDmrsTx::readDmrsParams(const std::vector<py::object>& dmrsParams, const uint32_t slot) const {
    const uint32_t numTbs = dmrsParams.size();

    for(int tbIdx = 0; tbIdx < numTbs; tbIdx++) {
        const auto cellIdx = dmrsParams[tbIdx].attr("cell_index_in_cell_group").cast<uint8_t>();
        m_dmrsParams[tbIdx].cell_output_tensor_addr = static_cast<void*>(m_txBufHalf[cellIdx].get());
        m_dmrsParams[tbIdx].cell_index_in_cell_group = cellIdx;

        const auto numDmrsCdmGrps = dmrsParams[tbIdx].attr("num_dmrs_cdm_grps_no_data").cast<uint8_t>();
        if(numDmrsCdmGrps == 3) {
            NVLOGE_FMT(NVLOG_PYAERIAL, AERIAL_PYAERIAL_EVENT, "3 DM-RS CDM groups without data not supported for type-I DMRS!");
            throw std::runtime_error("PyPdschDmrsTx::readDmrsParams: Invalid parameters");
        }

        const auto startSym = dmrsParams[tbIdx].attr("start_sym").cast<uint8_t>();
        const auto numPdschSyms = dmrsParams[tbIdx].attr("num_pdsch_syms").cast<uint8_t>();
        expandDmrsBitmap(dmrsParams[tbIdx].attr("dmrs_syms"),
                         startSym,
                         numPdschSyms,
                         m_dmrsParams[tbIdx].dmrs_sym_loc,
                         m_dmrsParams[tbIdx].data_sym_loc,
                         m_dmrsParams[tbIdx].num_dmrs_symbols,
                         m_dmrsParams[tbIdx].num_data_symbols);

        m_dmrsParams[tbIdx].slot_number = slot;
        m_dmrsParams[tbIdx].cell_id = 0; // Not used.
        m_dmrsParams[tbIdx].symbol_number = startSym;
        m_dmrsParams[tbIdx].resourceAlloc = dmrsParams[tbIdx].attr("resource_alloc").cast<uint8_t>();

        const py::array& rbBitmap = dmrsParams[tbIdx].attr("prb_bitmap");
        readRbBitmap(rbBitmap, m_dmrsParams[tbIdx].rbBitmap);

        m_dmrsParams[tbIdx].num_BWP_PRBs = dmrsParams[tbIdx].attr("num_bwp_prbs").cast<uint16_t>();
        m_dmrsParams[tbIdx].num_Rbs = dmrsParams[tbIdx].attr("num_prbs").cast<uint16_t>();
        m_dmrsParams[tbIdx].start_Rb = dmrsParams[tbIdx].attr("start_prb").cast<uint16_t>();
        m_dmrsParams[tbIdx].beta_dmrs = std::sqrt(numDmrsCdmGrps * 1.0f) * dmrsParams[tbIdx].attr("beta_dmrs").cast<float>();
        m_dmrsParams[tbIdx].beta_qam = 1.0f; // Not used.
        m_dmrsParams[tbIdx].num_layers = dmrsParams[tbIdx].attr("layers").cast<uint8_t>();

        auto dmrsPortsBmsk = dmrsParams[tbIdx].attr("dmrs_ports").cast<uint16_t>();
        for (int i = 0; i < m_dmrsParams[tbIdx].num_layers; i++) {
            m_dmrsParams[tbIdx].port_ids[i] = __builtin_ctz(dmrsPortsBmsk);
            dmrsPortsBmsk ^= (1 << m_dmrsParams[tbIdx].port_ids[i]);
        }

        m_dmrsParams[tbIdx].n_scid = dmrsParams[tbIdx].attr("scid").cast<uint32_t>();
        m_dmrsParams[tbIdx].dmrs_scid = dmrsParams[tbIdx].attr("dmrs_scrm_id").cast<uint32_t>();
        m_dmrsParams[tbIdx].BWP_start_PRB = dmrsParams[tbIdx].attr("bwp_start").cast<uint16_t>();
        m_dmrsParams[tbIdx].ref_point = dmrsParams[tbIdx].attr("ref_point").cast<uint8_t>();

        const py::object& temp = dmrsParams[tbIdx].attr("precoding_matrix");
        if (not (py::str(temp).cast<std::string>() == "None")) {
            m_dmrsParams[tbIdx].enablePrcdBf = 1;
            const py::array_t<std::complex<float>>& pmwArray = temp;
            readPrecodingMatrix(pmwArray, m_dmrsParams[tbIdx].pmW, m_dmrsParams[tbIdx].Np);
        }
        else {
            m_dmrsParams[tbIdx].enablePrcdBf = 0;
            m_dmrsParams[tbIdx].Np = 0;
        }

        m_dmrsParams[tbIdx].ueGrp_idx = 0; // Not used.
        m_dmrsParams[tbIdx].dmrsCdmGrpsNoData1 = (numDmrsCdmGrps == 1);
        m_dmrsParams[tbIdx].nlAbove16 = 0;
    }
}


}  // namespace pycuphy
