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

#include <optional>
#include <numeric>
#include <functional>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "pycuphy_chan_model.hpp"
#include "cuda_array_interface.hpp"

namespace py = pybind11;

namespace pycuphy {

// Helper function to create cuda_array_t for channel output signal
// Avoids code duplication between TdlChanWrapper and CdlChanWrapper
template <typename Tscalar>
cuda_array_t<std::complex<Tscalar>> createRxSignalOutArray(
    void* ptr,
    uint16_t nCell,
    uint16_t nUe,
    uint16_t nRxAnt,
    uint32_t sigLenPerAnt)
{
    if (ptr == nullptr) {
        throw std::runtime_error("Output buffer not allocated. Ensure signal_length_per_ant > 0.");
    }
    if (sigLenPerAnt == 0) {
        throw std::runtime_error("signal_length_per_ant is 0. Cannot create output array.");
    }
    if (nRxAnt == 0) {
        throw std::runtime_error("nRxAnt is 0. Check antenna configuration.");
    }

    // Shape: [nCell, nUe, nRxAnt, sigLenPerAnt]
    std::vector<size_t> shape = {
        static_cast<size_t>(nCell),
        static_cast<size_t>(nUe),
        static_cast<size_t>(nRxAnt),
        static_cast<size_t>(sigLenPerAnt)
    };
    std::vector<size_t> strides = {
        static_cast<size_t>(nUe * nRxAnt * sigLenPerAnt * sizeof(std::complex<Tscalar>)),
        static_cast<size_t>(nRxAnt * sigLenPerAnt * sizeof(std::complex<Tscalar>)),
        static_cast<size_t>(sigLenPerAnt * sizeof(std::complex<Tscalar>)),
        static_cast<size_t>(sizeof(std::complex<Tscalar>))
    };

    return cuda_array_t<std::complex<Tscalar>>(reinterpret_cast<intptr_t>(ptr), shape, strides);
}

/*-------------------------------       OFDM modulation class       -------------------------------*/
template <typename Tscalar, typename Tcomplex>
OfdmModulateWrapper<Tscalar, Tcomplex>::OfdmModulateWrapper(cuphyCarrierPrms_t* cuphyCarrierPrms, py::array_t<std::complex<Tscalar>> freqDataInCpu, uintptr_t streamHandle) :
m_cuStrm((cudaStream_t)streamHandle),
m_externGpuAlloc(0)
{
    // buffer size from config
    m_freqDataInSizeDl = (cuphyCarrierPrms -> N_sc) * (cuphyCarrierPrms -> N_bsLayer) * (cuphyCarrierPrms -> N_symbol_slot);
    m_freqDataInSizeUl = (cuphyCarrierPrms -> N_sc) * (cuphyCarrierPrms -> N_ueLayer) * (cuphyCarrierPrms -> N_symbol_slot);
    // get buffer info from the NumPy array
    py::buffer_info buf = freqDataInCpu.request();
    m_freqDataInCpu = static_cast<Tcomplex*>(buf.ptr);

    // allocate GPU buffer with RAII guard to prevent leak if ofdmModulate ctor throws
    m_freqDataInGpu = nullptr;
    m_ofdmModulateHandle = nullptr;
    {
        size_t allocSize = sizeof(Tcomplex) * std::max(m_freqDataInSizeDl, m_freqDataInSizeUl);
        cudaError_t err = cudaMalloc((void**) &(m_freqDataInGpu), allocSize);
        if (err != cudaSuccess) {
            m_freqDataInGpu = nullptr;
            throw std::runtime_error(
                std::string("cudaMalloc failed for m_freqDataInGpu: ") +
                cudaGetErrorString(err) + " (error " + std::to_string(static_cast<int>(err)) +
                "), requested " + std::to_string(allocSize) + " bytes" +
                " (sizeDl=" + std::to_string(m_freqDataInSizeDl) +
                ", sizeUl=" + std::to_string(m_freqDataInSizeUl) + ")");
        }
    }
    try {
        m_ofdmModulateHandle = new ofdm_modulate::ofdmModulate<Tscalar, Tcomplex>(cuphyCarrierPrms, m_freqDataInGpu, m_cuStrm);
    } catch (...) {
        cudaFree(m_freqDataInGpu);
        m_freqDataInGpu = nullptr;
        throw;
    }
}

template <typename Tscalar, typename Tcomplex>
OfdmModulateWrapper<Tscalar, Tcomplex>::OfdmModulateWrapper(cuphyCarrierPrms_t* cuphyCarrierPrms, uintptr_t freqDataInGpu, uintptr_t streamHandle) :
m_cuStrm((cudaStream_t)streamHandle),
m_externGpuAlloc(1)
{
    m_freqDataInGpu = (Tcomplex*)freqDataInGpu;
    m_ofdmModulateHandle = new ofdm_modulate::ofdmModulate<Tscalar, Tcomplex>(cuphyCarrierPrms, m_freqDataInGpu, m_cuStrm);
}

template <typename Tscalar, typename Tcomplex>
OfdmModulateWrapper<Tscalar, Tcomplex>::~OfdmModulateWrapper()
{
    if (!m_externGpuAlloc) // cudaMalloc internally, need to free GPU memory
    {
        cudaFree(m_freqDataInGpu);
    }
    delete m_ofdmModulateHandle;
}

template <typename Tscalar, typename Tcomplex>
void OfdmModulateWrapper<Tscalar, Tcomplex>::run(py::array_t<std::complex<Tscalar>> freqDataInCpu, uint8_t enableSwapTxRx)
{
    uint32_t freqDataInSize = (enableSwapTxRx ? m_freqDataInSizeUl : m_freqDataInSizeDl);
    if(freqDataInCpu.size() != 0) // new input numpy array, need to copy new data to GPU
    {
        // get buffer info from the NumPy array
        py::buffer_info buf = freqDataInCpu.request();
        Tcomplex* freqDataInCpuNew = static_cast<Tcomplex*>(buf.ptr);
        assert(buf.size == freqDataInSize); // check data size match

        cudaMemcpyAsync(m_freqDataInGpu, freqDataInCpuNew, sizeof(Tcomplex) * freqDataInSize, cudaMemcpyHostToDevice, m_cuStrm);
    }
    else
    {
        if (!m_externGpuAlloc) // use numpy array, need to copy new data to GPU
        {
            cudaMemcpyAsync(m_freqDataInGpu, m_freqDataInCpu, sizeof(Tcomplex) * freqDataInSize, cudaMemcpyHostToDevice, m_cuStrm);
        }
    }
    m_ofdmModulateHandle -> run(enableSwapTxRx, m_cuStrm);
}

/*-------------------------------       OFDM demodulation class       -------------------------------*/
template <typename Tscalar, typename Tcomplex>
OfdmDeModulateWrapper<Tscalar, Tcomplex>::OfdmDeModulateWrapper(cuphyCarrierPrms_t * cuphyCarrierPrms, uintptr_t timeDataInGpu, py::array_t<std::complex<Tscalar>> freqDataOutCpu, bool prach, bool perAntSamp, uintptr_t streamHandle) :
m_cuStrm((cudaStream_t)streamHandle),
m_perAntSamp(perAntSamp),
m_externGpuAlloc(0)
{
    // buffer size from config
    m_freqDataOutSizeDl = (cuphyCarrierPrms -> N_sc) * (cuphyCarrierPrms -> N_ueLayer) * (cuphyCarrierPrms -> N_symbol_slot);
    m_freqDataOutSizeUl = (cuphyCarrierPrms -> N_sc) * (cuphyCarrierPrms -> N_bsLayer) * (cuphyCarrierPrms -> N_symbol_slot);
    if (m_perAntSamp)
    {
        m_freqDataOutSizeDl *= cuphyCarrierPrms -> N_bsLayer;
        m_freqDataOutSizeUl *= cuphyCarrierPrms -> N_ueLayer;
    }
    // get buffer info from the NumPy array
    py::buffer_info buf = freqDataOutCpu.request();
    m_freqDataOutCpu = static_cast<Tcomplex*>(buf.ptr);

    // allocate GPU buffer with RAII guard to prevent leak if ofdmDeModulate ctor throws
    m_freqDataOutGpu = nullptr;
    m_ofdmDeModulateHandle = nullptr;
    {
        size_t allocSize = sizeof(Tcomplex) * std::max(m_freqDataOutSizeDl, m_freqDataOutSizeUl);
        cudaError_t err = cudaMalloc((void**) &(m_freqDataOutGpu), allocSize);
        if (err != cudaSuccess) {
            m_freqDataOutGpu = nullptr;
            throw std::runtime_error(
                std::string("cudaMalloc failed for m_freqDataOutGpu: ") +
                cudaGetErrorString(err) + " (error " + std::to_string(static_cast<int>(err)) +
                "), requested " + std::to_string(allocSize) + " bytes" +
                " (sizeDl=" + std::to_string(m_freqDataOutSizeDl) +
                ", sizeUl=" + std::to_string(m_freqDataOutSizeUl) + ")");
        }
    }
    try {
        m_ofdmDeModulateHandle = new ofdm_demodulate::ofdmDeModulate<Tscalar, Tcomplex>(cuphyCarrierPrms, (Tcomplex*)timeDataInGpu, m_freqDataOutGpu, prach, perAntSamp, m_cuStrm);
    } catch (...) {
        cudaFree(m_freqDataOutGpu);
        m_freqDataOutGpu = nullptr;
        throw;
    }
}

template <typename Tscalar, typename Tcomplex>
OfdmDeModulateWrapper<Tscalar, Tcomplex>::OfdmDeModulateWrapper(cuphyCarrierPrms_t * cuphyCarrierPrms, uintptr_t timeDataInGpu, uintptr_t freqDataOutGpu, bool prach, bool perAntSamp, uintptr_t streamHandle) :
m_cuStrm((cudaStream_t)streamHandle),
m_perAntSamp(perAntSamp),
m_externGpuAlloc(1)
{
    // buffer size from config
    m_freqDataOutSizeDl = (cuphyCarrierPrms -> N_sc) * (cuphyCarrierPrms -> N_ueLayer) * (cuphyCarrierPrms -> N_symbol_slot);
    m_freqDataOutSizeUl = (cuphyCarrierPrms -> N_sc) * (cuphyCarrierPrms -> N_bsLayer) * (cuphyCarrierPrms -> N_symbol_slot);
    if (m_perAntSamp)
    {
        m_freqDataOutSizeDl *= cuphyCarrierPrms -> N_bsLayer;
        m_freqDataOutSizeUl *= cuphyCarrierPrms -> N_ueLayer;
    }
    m_freqDataOutCpu = nullptr;
    m_freqDataOutGpu = (Tcomplex*) freqDataOutGpu;
    m_ofdmDeModulateHandle = new ofdm_demodulate::ofdmDeModulate<Tscalar, Tcomplex>(cuphyCarrierPrms, (Tcomplex*)timeDataInGpu, m_freqDataOutGpu, prach, perAntSamp, m_cuStrm);
}

template <typename Tscalar, typename Tcomplex>
OfdmDeModulateWrapper<Tscalar, Tcomplex>::~OfdmDeModulateWrapper()
{
    if (!m_externGpuAlloc) // cudaMalloc internally, need to free GPU memory
    {
        cudaFree(m_freqDataOutGpu);
    }
    delete m_ofdmDeModulateHandle;
}

template <typename Tscalar, typename Tcomplex>
void OfdmDeModulateWrapper<Tscalar, Tcomplex>::run(py::array_t<std::complex<Tscalar>> freqDataOutCpu, uint8_t enableSwapTxRx)
{
    m_ofdmDeModulateHandle -> run(enableSwapTxRx, m_cuStrm);
    uint32_t freqDataOutSize = (enableSwapTxRx ? m_freqDataOutSizeUl : m_freqDataOutSizeDl);
    if(freqDataOutCpu.size() != 0) // new output numpy array, need to copy new data from GPU
    {
        py::buffer_info buf = freqDataOutCpu.request();
        Tcomplex* freqDataOutCpuNew = static_cast<Tcomplex*>(buf.ptr);
        assert(buf.size == freqDataOutSize); // check data size match

        cudaMemcpyAsync(freqDataOutCpuNew, m_freqDataOutGpu, sizeof(Tcomplex) * freqDataOutSize, cudaMemcpyDeviceToHost, m_cuStrm);
    }
    else if (!m_externGpuAlloc)
    {
        cudaMemcpyAsync(m_freqDataOutCpu, m_freqDataOutGpu, sizeof(Tcomplex) * freqDataOutSize, cudaMemcpyDeviceToHost, m_cuStrm);
    }
    cudaStreamSynchronize(m_cuStrm);
}

/*-------------------------------       TDL channel class       -------------------------------*/
template <typename Tscalar, typename Tcomplex>
TdlChanWrapper<Tscalar, Tcomplex>::TdlChanWrapper(tdlConfig_t* tdlCfg, uint16_t randSeed, uintptr_t streamHandle) :
m_cuStrm((cudaStream_t)streamHandle),
m_runMode(tdlCfg -> runMode),
m_nLink((tdlCfg -> nCell) * (tdlCfg -> nUe)),
m_tdlCfg(tdlCfg)
{
    tdlCfg -> txSigIn = nullptr;
    m_tdlChanHandle = new tdlChan<Tscalar, Tcomplex>(tdlCfg, randSeed, m_cuStrm);
}

template <typename Tscalar, typename Tcomplex>
TdlChanWrapper<Tscalar, Tcomplex>::~TdlChanWrapper()
{
    delete m_tdlChanHandle;
}

template <typename Tscalar, typename Tcomplex>
void TdlChanWrapper<Tscalar, Tcomplex>::run(const cuda_array_t<std::complex<Tscalar>>& txSigIn, float refTime0, uint8_t enableSwapTxRx, uint8_t txColumnMajorInd)
{
    // Update the tx signal pointer in the channel's descriptor (both CPU and GPU)
    m_tdlChanHandle->setTxSigIn(reinterpret_cast<Tcomplex*>(txSigIn.get_device_ptr()));
    m_tdlChanHandle->run(refTime0, enableSwapTxRx, txColumnMajorInd);
    // Output is written to internal buffer, accessible via getRxSignalOutArray()
}

template <typename Tscalar, typename Tcomplex>
cuda_array_t<std::complex<Tscalar>> TdlChanWrapper<Tscalar, Tcomplex>::getRxSignalOutArray(uint8_t enableSwapTxRx)
{
    // TDL config has direct nBsAnt/nUeAnt members
    uint16_t nRxAnt = enableSwapTxRx ? m_tdlCfg->nBsAnt : m_tdlCfg->nUeAnt;

    return createRxSignalOutArray<Tscalar>(
        m_tdlChanHandle->getRxSigOut(),
        m_tdlCfg->nCell,
        m_tdlCfg->nUe,
        nRxAnt,
        m_tdlCfg->sigLenPerAnt
    );
}

template <typename Tscalar, typename Tcomplex>
void TdlChanWrapper<Tscalar, Tcomplex>::dumpCir(py::array_t<std::complex<Tscalar>> cirCpu)
{
    // buffer size from config
    uint32_t timeChanSize = m_tdlChanHandle -> getTimeChanSize();

    // get buffer info from the NumPy array
    py::buffer_info buf = cirCpu.request();
    assert(buf.size == timeChanSize); // check data size match

    // copy CIR
    cudaMemcpyAsync(buf.ptr, m_tdlChanHandle -> getTimeChan(), sizeof(Tcomplex) * timeChanSize, cudaMemcpyDeviceToHost, m_cuStrm);
}

template <typename Tscalar, typename Tcomplex>
void TdlChanWrapper<Tscalar, Tcomplex>::dumpCfrPrbg(py::array_t<std::complex<Tscalar>> cfrPrbg)
{
    // dump CFR on PRBG
    if(m_runMode > 0 && m_runMode < 3)
    {
        // buffer size from config
        uint32_t freqChanPrbgSize = m_tdlChanHandle -> getFreqChanPrbgSize();

        // get buffer info from the NumPy array
        py::buffer_info buf = cfrPrbg.request();
        assert(buf.size == freqChanPrbgSize); // check data size match

        // copy CFR on PRBG
        cudaMemcpyAsync(buf.ptr, m_tdlChanHandle -> getFreqChanPrbg(), sizeof(Tcomplex) * freqChanPrbgSize, cudaMemcpyDeviceToHost, m_cuStrm);
    }
}

template <typename Tscalar, typename Tcomplex>
void TdlChanWrapper<Tscalar, Tcomplex>::dumpCfrSc(py::array_t<std::complex<Tscalar>> cfrSc)
{
    // dump CFR on SC
    if(m_runMode > 1 && m_runMode < 3)
    {
        // buffer size from config
        uint32_t freqChanScSizePerLink = m_tdlChanHandle -> getFreqChanScPerLinkSize();
        Tcomplex ** freqChanSc = m_tdlChanHandle -> getFreqChanScHostArray(); // CFR on SC is saved using pointer of pointers

        // get buffer info from the NumPy array
        py::buffer_info buf = cfrSc.request();

        // copy CFR on SC
        Tcomplex * freqChScCpuOut = static_cast<Tcomplex*>(buf.ptr);
        for (uint16_t linkIdx = 0; linkIdx < m_nLink; linkIdx ++)
        {
            cudaMemcpyAsync(freqChScCpuOut, freqChanSc[linkIdx], sizeof(Tcomplex) * freqChanScSizePerLink, cudaMemcpyDeviceToHost, m_cuStrm);
            freqChScCpuOut += freqChanScSizePerLink; // CPU address for CFR on SC of next link
        }
    }
}

/*-------------------------------       CDL channel class       -------------------------------*/
template <typename Tscalar, typename Tcomplex>
CdlChanWrapper<Tscalar, Tcomplex>::CdlChanWrapper(cdlConfig_t* cdlCfg, uint16_t randSeed, uintptr_t streamHandle) :
m_cuStrm((cudaStream_t)streamHandle),
m_runMode(cdlCfg -> runMode),
m_nLink((cdlCfg -> nCell) * (cdlCfg -> nUe)),
m_cdlCfg(cdlCfg)
{
    m_nBsAnt = std::accumulate(m_cdlCfg -> bsAntSize.begin(), m_cdlCfg -> bsAntSize.end(), 1U, std::multiplies<uint32_t>());
    m_nUeAnt = std::accumulate(m_cdlCfg -> ueAntSize.begin(), m_cdlCfg -> ueAntSize.end(), 1U, std::multiplies<uint32_t>());
    cdlCfg -> txSigIn = nullptr;
    m_cdlChanHandle = new cdlChan<Tscalar, Tcomplex>(cdlCfg, randSeed, m_cuStrm);
}

template <typename Tscalar, typename Tcomplex>
CdlChanWrapper<Tscalar, Tcomplex>::~CdlChanWrapper()
{
    delete m_cdlChanHandle;
}

template <typename Tscalar, typename Tcomplex>
void CdlChanWrapper<Tscalar, Tcomplex>::run(const cuda_array_t<std::complex<Tscalar>>& txSigIn, float refTime0, uint8_t enableSwapTxRx, uint8_t txColumnMajorInd)
{
    // Update the tx signal pointer in the channel's descriptor (both CPU and GPU)
    m_cdlChanHandle->setTxSigIn(reinterpret_cast<Tcomplex*>(txSigIn.get_device_ptr()));
    m_cdlChanHandle->run(refTime0, enableSwapTxRx, txColumnMajorInd);
    // Output is written to internal buffer, accessible via getRxSignalOutArray()
}

template <typename Tscalar, typename Tcomplex>
cuda_array_t<std::complex<Tscalar>> CdlChanWrapper<Tscalar, Tcomplex>::getRxSignalOutArray(uint8_t enableSwapTxRx)
{
    // CDL config uses bsAntSize/ueAntSize vectors - compute antenna count as product of elements
    uint16_t nBsAnt = std::accumulate(m_cdlCfg->bsAntSize.begin(), m_cdlCfg->bsAntSize.end(),
                                       static_cast<uint16_t>(1), std::multiplies<uint16_t>());
    uint16_t nUeAnt = std::accumulate(m_cdlCfg->ueAntSize.begin(), m_cdlCfg->ueAntSize.end(),
                                       static_cast<uint16_t>(1), std::multiplies<uint16_t>());
    uint16_t nRxAnt = enableSwapTxRx ? nBsAnt : nUeAnt;
    return createRxSignalOutArray<Tscalar>(
        m_cdlChanHandle->getRxSigOut(),
        m_cdlCfg->nCell,
        m_cdlCfg->nUe,
        nRxAnt,
        m_cdlCfg->sigLenPerAnt
    );
}

template <typename Tscalar, typename Tcomplex>
void CdlChanWrapper<Tscalar, Tcomplex>::dumpCir(py::array_t<std::complex<Tscalar>> cirCpu)
{
    // buffer size from config
    uint32_t timeChanSize = m_cdlChanHandle -> getTimeChanSize();

    // get buffer info from the NumPy array
    py::buffer_info buf = cirCpu.request();
    assert(buf.size == timeChanSize); // check data size match

    // copy CIR
    cudaMemcpyAsync(buf.ptr, m_cdlChanHandle -> getTimeChan(), sizeof(Tcomplex) * timeChanSize, cudaMemcpyDeviceToHost, m_cuStrm);
}

template <typename Tscalar, typename Tcomplex>
void CdlChanWrapper<Tscalar, Tcomplex>::dumpCfrPrbg(py::array_t<std::complex<Tscalar>> cfrPrbg)
{
    // dump CFR on PRBG
    if(m_runMode > 0 && m_runMode < 3)
    {
        // buffer size from config
        uint32_t freqChanPrbgSize = m_cdlChanHandle -> getFreqChanPrbgSize();

        // get buffer info from the NumPy array
        py::buffer_info buf = cfrPrbg.request();
        assert(buf.size == freqChanPrbgSize); // check data size match

        // copy CFR on PRBG
        cudaMemcpyAsync(buf.ptr, m_cdlChanHandle -> getFreqChanPrbg(), sizeof(Tcomplex) * freqChanPrbgSize, cudaMemcpyDeviceToHost, m_cuStrm);
    }
}

template <typename Tscalar, typename Tcomplex>
void CdlChanWrapper<Tscalar, Tcomplex>::dumpCfrSc(py::array_t<std::complex<Tscalar>> cfrSc)
{
    // dump CFR on SC
    if(m_runMode > 1 && m_runMode < 3)
    {
        // buffer size from config
        uint32_t freqChanScSizePerLink = m_cdlChanHandle -> getFreqChanScPerLinkSize();
        Tcomplex ** freqChanSc = m_cdlChanHandle -> getFreqChanScHostArray(); // CFR on SC is saved using pointer of pointers

        // get buffer info from the NumPy array
        py::buffer_info buf = cfrSc.request();

        // copy CFR on SC
        Tcomplex * freqChScCpuOut = static_cast<Tcomplex*>(buf.ptr);
        for (uint16_t linkIdx = 0; linkIdx < m_nLink; linkIdx ++)
        {
            cudaMemcpyAsync(freqChScCpuOut, freqChanSc[linkIdx], sizeof(Tcomplex) * freqChanScSizePerLink, cudaMemcpyDeviceToHost, m_cuStrm);
            freqChScCpuOut += freqChanScSizePerLink; // CPU address for CFR on SC of next link
        }
    }
}

/*-------------------------------       add Gaussian noise class       -------------------------------*/
template <typename Tscalar, typename Tcomplex>
GauNoiseAdderWrapper<Tscalar, Tcomplex>::GauNoiseAdderWrapper(uint32_t nThreads, int seed, uintptr_t streamHandle) :
m_cuStrm((cudaStream_t)streamHandle)
{
    m_gauNoiseAdder = new GauNoiseAdder<Tcomplex>(nThreads, seed, m_cuStrm);
}

template <typename Tscalar, typename Tcomplex>
GauNoiseAdderWrapper<Tscalar, Tcomplex>::~GauNoiseAdderWrapper()
{
    delete m_gauNoiseAdder;
}

template <typename Tscalar, typename Tcomplex>
void GauNoiseAdderWrapper<Tscalar, Tcomplex>::addNoise(uintptr_t d_signal, uint32_t signalSize, float snr_db)
{
    // Add noise in-place on GPU - no copy to CPU
    m_gauNoiseAdder -> addNoise((Tcomplex*)d_signal, signalSize, snr_db);
}

/*-------------------------------       Stochastic Channel Model class       -------------------------------*/
template <typename Tscalar, typename Tcomplex>
StatisChanModelWrapper<Tscalar, Tcomplex>::StatisChanModelWrapper(
    const SimConfig& sim_config,
    const SystemLevelConfig& system_level_config,
    const LinkLevelConfig& link_level_config,
    const ExternalConfig& external_config,
    uint32_t randSeed,
    uintptr_t streamHandle) :
m_cuStrm((cudaStream_t)streamHandle),
m_randSeed(randSeed)
{
    m_statisChanModelHandle = new statisChanModel<Tscalar, Tcomplex>(
        &sim_config, &system_level_config, &link_level_config, &external_config,
        m_randSeed, m_cuStrm);
    m_cpuOnlyMode = sim_config.cpu_only_mode;
}

// Constructor with just sim_config and system_level_config
template <typename Tscalar, typename Tcomplex>
StatisChanModelWrapper<Tscalar, Tcomplex>::StatisChanModelWrapper(
    const SimConfig& sim_config,
    const SystemLevelConfig& system_level_config,
    uint32_t randSeed,
    uintptr_t streamHandle) :
m_cuStrm((cudaStream_t)streamHandle),
m_randSeed(randSeed)
{
    m_statisChanModelHandle = new statisChanModel<Tscalar, Tcomplex>(
        &sim_config, &system_level_config, nullptr, nullptr,
        m_randSeed, m_cuStrm);
    m_cpuOnlyMode = sim_config.cpu_only_mode;
}

template <typename Tscalar, typename Tcomplex>
StatisChanModelWrapper<Tscalar, Tcomplex>::~StatisChanModelWrapper()
{
    delete m_statisChanModelHandle;
}

template <typename Tscalar, typename Tcomplex>
void StatisChanModelWrapper<Tscalar, Tcomplex>::run(
    float refTime,
    uint8_t continuous_fading,
    py::object activeCell,
    py::object activeUt,
    py::object utNewLoc,
    py::object utNewVelocity,
    py::object cir_coe,
    py::object cir_norm_delay,
    py::object cir_n_taps,
    py::object cfr_sc,
    py::object cfr_prbg) {

    // Convert activeCell to vector
    std::vector<uint16_t> activeCellVec;
    if (!activeCell.is_none()) {
        if (py::isinstance<py::list>(activeCell)) {
            auto cellList = activeCell.cast<py::list>();
            for (auto item : cellList) {
                activeCellVec.push_back(item.cast<uint16_t>());
            }
        }
    }

    // Convert activeUt to nested vector
    std::vector<std::vector<uint16_t>> activeUtVec;
    if (!activeUt.is_none()) {
        if (py::isinstance<py::list>(activeUt)) {
            auto utList = activeUt.cast<py::list>();
            for (auto item : utList) {
                if (py::isinstance<py::list>(item)) {
                    std::vector<uint16_t> utVec;
                    auto innerList = item.cast<py::list>();
                    for (auto ut : innerList) {
                        utVec.push_back(ut.cast<uint16_t>());
                    }
                    activeUtVec.push_back(utVec);
                }
            }
        }
    }

    // Convert utNewLoc to vector of Coordinates
    std::vector<Coordinate> utNewLocVec;
    if (!utNewLoc.is_none()) {
        if (py::isinstance<py::array>(utNewLoc)) {
            auto arr = utNewLoc.cast<py::array_t<float>>();
            auto buf = arr.unchecked<2>();
            for (py::ssize_t i = 0; i < buf.shape(0); i++) {
                Coordinate coord;
                coord.x = buf(i, 0);
                coord.y = buf(i, 1);
                coord.z = buf(i, 2);
                utNewLocVec.push_back(coord);
            }
        }
    }

    // Convert utNewVelocity to vector of float3
    std::vector<float3> utNewVelocityVec;
    if (!utNewVelocity.is_none()) {
        if (py::isinstance<py::array>(utNewVelocity)) {
            auto arr = utNewVelocity.cast<py::array_t<float>>();
            auto buf = arr.unchecked<2>();
            for (py::ssize_t i = 0; i < buf.shape(0); i++) {
                float3 vel;
                vel.x = buf(i, 0);
                vel.y = buf(i, 1);
                vel.z = buf(i, 2);
                utNewVelocityVec.push_back(vel);
            }
        }
    }

    // Convert per-cell array parameters to vectors of pointers
    std::vector<Tcomplex*> cir_coe_ptrs;
    std::vector<uint16_t*> cir_norm_delay_ptrs;
    std::vector<uint16_t*> cir_n_taps_ptrs;
    std::vector<Tcomplex*> cfr_sc_ptrs;
    std::vector<Tcomplex*> cfr_prbg_ptrs;

    // Helper lambda to extract device pointers from list of CuPy arrays
    auto extract_cuda_array_ptrs = [](py::object obj, auto& ptr_vec) {
        if (!obj.is_none() && py::isinstance<py::list>(obj)) {
            auto array_list = obj.cast<py::list>();
            for (auto item : array_list) {
                uintptr_t device_ptr = 0;

                // Handle CuPy arrays directly using __cuda_array_interface__
                if (py::hasattr(item, "__cuda_array_interface__")) {
                    auto array_interface = item.attr("__cuda_array_interface__");
                    if (py::isinstance<py::dict>(array_interface)) {
                        auto interface_dict = array_interface.cast<py::dict>();
                        if (interface_dict.contains("data")) {
                            auto data_info = interface_dict["data"];
                            if (py::isinstance<py::tuple>(data_info)) {
                                auto data_tuple = data_info.cast<py::tuple>();
                                if (data_tuple.size() > 0) {
                                    device_ptr = data_tuple[0].cast<uintptr_t>();
                                }
                            }
                        }
                    }
                }
                // Also try CPU arrays with __array_interface__ as fallback
                else if (py::hasattr(item, "__array_interface__")) {
                    auto array_interface = item.attr("__array_interface__");
                    if (py::isinstance<py::dict>(array_interface)) {
                        auto interface_dict = array_interface.cast<py::dict>();
                        if (interface_dict.contains("data")) {
                            auto data_info = interface_dict["data"];
                            if (py::isinstance<py::tuple>(data_info)) {
                                auto data_tuple = data_info.cast<py::tuple>();
                                if (data_tuple.size() > 0) {
                                    device_ptr = data_tuple[0].cast<uintptr_t>();
                                }
                            }
                        }
                    }
                }
                // Fallback: try pycuphy wrapper with get_device_ptr()
                else if (py::hasattr(item, "get_device_ptr")) {
                    auto ptr_value = item.attr("get_device_ptr")();
                    if (py::isinstance<py::int_>(ptr_value)) {
                        device_ptr = ptr_value.cast<uintptr_t>();
                    }
                }
                // Fallback: try PyTorch-style data_ptr()
                else if (py::hasattr(item, "data_ptr")) {
                    device_ptr = item.attr("data_ptr")().cast<uintptr_t>();
                }

                ptr_vec.push_back(reinterpret_cast<typename std::remove_reference_t<decltype(ptr_vec)>::value_type>(device_ptr));
            }
        }
    };

    // Extract pointers from Python objects:
    // - In GPU mode, prefer __cuda_array_interface__ and device pointers
    // - In CPU-only mode, fall back to __array_interface__ host pointers
    extract_cuda_array_ptrs(cir_coe, cir_coe_ptrs);
    extract_cuda_array_ptrs(cir_norm_delay, cir_norm_delay_ptrs);
    extract_cuda_array_ptrs(cir_n_taps, cir_n_taps_ptrs);
    extract_cuda_array_ptrs(cfr_sc, cfr_sc_ptrs);
    extract_cuda_array_ptrs(cfr_prbg, cfr_prbg_ptrs);

    if (m_cpuOnlyMode == 0) {
        // Debug GPU pointers only in GPU mode
        printf("DEBUG: Extracted GPU pointers from pybind11:\n");
        printf("  cir_coe_ptrs: %zu pointers\n", cir_coe_ptrs.size());
        for (size_t i = 0; i < cir_coe_ptrs.size(); ++i) {
            printf("    Cell %zu: cir_coe=0x%lx\n", i, reinterpret_cast<uintptr_t>(cir_coe_ptrs[i]));
        }
        printf("  cir_norm_delay_ptrs: %zu pointers\n", cir_norm_delay_ptrs.size());
        for (size_t i = 0; i < cir_norm_delay_ptrs.size(); ++i) {
            printf("    Cell %zu: cir_norm_delay=0x%lx\n", i, reinterpret_cast<uintptr_t>(cir_norm_delay_ptrs[i]));
        }
        printf("  cir_n_taps_ptrs: %zu pointers\n", cir_n_taps_ptrs.size());
        for (size_t i = 0; i < cir_n_taps_ptrs.size(); ++i) {
            printf("    Cell %zu: cir_n_taps=0x%lx\n", i, reinterpret_cast<uintptr_t>(cir_n_taps_ptrs[i]));
        }
        printf("  cfr_sc_ptrs: %zu pointers\n", cfr_sc_ptrs.size());
        for (size_t i = 0; i < cfr_sc_ptrs.size(); ++i) {
            printf("    Cell %zu: cfr_sc=0x%lx\n", i, reinterpret_cast<uintptr_t>(cfr_sc_ptrs[i]));
        }
        printf("  cfr_prbg_ptrs: %zu pointers\n", cfr_prbg_ptrs.size());
        for (size_t i = 0; i < cfr_prbg_ptrs.size(); ++i) {
            printf("    Cell %zu: cfr_prbg=0x%lx\n", i, reinterpret_cast<uintptr_t>(cfr_prbg_ptrs[i]));
        }
    }

    // Call the C++ method with vectors of pointers
    m_statisChanModelHandle->run(refTime, continuous_fading, activeCellVec, activeUtVec,
                               utNewLocVec, utNewVelocityVec, cir_coe_ptrs, cir_norm_delay_ptrs,
                               cir_n_taps_ptrs, cfr_sc_ptrs, cfr_prbg_ptrs);
    // sync stream in GPU mode
    if (m_cpuOnlyMode == 0) {
        cudaStreamSynchronize(m_cuStrm);
    }
}

template <typename Tscalar, typename Tcomplex>
void StatisChanModelWrapper<Tscalar, Tcomplex>::run_link_level(
    float refTime0,
    uint8_t continuous_fading,
    uint8_t enableSwapTxRx,
    uint8_t txColumnMajorInd) {

    // Create empty vectors and empty pointer vectors for unused parameters
    std::vector<uint16_t> empty_cells;
    std::vector<std::vector<uint16_t>> empty_uts;
    std::vector<Coordinate> empty_locs;
    std::vector<float3> empty_velocities;
    std::vector<Tcomplex*> empty_cir_coe;
    std::vector<uint16_t*> empty_cir_norm_delay;
    std::vector<uint16_t*> empty_cir_n_taps;
    std::vector<Tcomplex*> empty_cfr_sc;
    std::vector<Tcomplex*> empty_cfr_prbg;

    // Call the run method with all parameters
    m_statisChanModelHandle->run(refTime0, continuous_fading, empty_cells, empty_uts,
                               empty_locs, empty_velocities, empty_cir_coe, empty_cir_norm_delay,
                               empty_cir_n_taps, empty_cfr_sc, empty_cfr_prbg);
}

template <typename Tscalar, typename Tcomplex>
void StatisChanModelWrapper<Tscalar, Tcomplex>::dump_los_nlos_stats(py::array_t<float> lost_nlos_stats) {
    float* stats_ptr = nullptr;
    if (lost_nlos_stats.size() > 0) {
        stats_ptr = lost_nlos_stats.mutable_data();
    }
    m_statisChanModelHandle->dump_los_nlos_stats(stats_ptr);
}

template <typename Tscalar, typename Tcomplex>
void StatisChanModelWrapper<Tscalar, Tcomplex>::dump_pl_sf_stats(
    py::array_t<float> pl_sf,
    py::array_t<int> activeCell,
    py::array_t<int> activeUt) {

    // pl_sf is required
    if (pl_sf.size() == 0) {
        throw std::invalid_argument("pl_sf array cannot be empty");
    }

    float* pl_sf_ptr = pl_sf.mutable_data();

    // Convert numpy arrays to vectors
    std::vector<uint16_t> activeCellVec;
    std::vector<uint16_t> activeUtVec;

    if (activeCell.size() > 0) {
        activeCellVec.reserve(activeCell.size());
        const int* cell_data = activeCell.data();
        for (size_t i = 0; i < activeCell.size(); ++i) {
            activeCellVec.push_back(static_cast<uint16_t>(cell_data[i]));
        }
    }

    if (activeUt.size() > 0) {
        activeUtVec.reserve(activeUt.size());
        const int* ut_data = activeUt.data();
        for (size_t i = 0; i < activeUt.size(); ++i) {
            activeUtVec.push_back(static_cast<uint16_t>(ut_data[i]));
        }
    }
    // Call the C++ method with vectors
    m_statisChanModelHandle->dump_pl_sf_stats(pl_sf_ptr, activeCellVec, activeUtVec);
}

template <typename Tscalar, typename Tcomplex>
void StatisChanModelWrapper<Tscalar, Tcomplex>::dump_pl_sf_ant_gain_stats(
    py::array_t<float> pl_sf_ant_gain,
    py::array_t<int> activeCell,
    py::array_t<int> activeUt) {

    if (pl_sf_ant_gain.size() == 0) {
        throw std::invalid_argument("pl_sf_ant_gain array cannot be empty");
    }

    float* pl_sf_ant_gain_ptr = pl_sf_ant_gain.mutable_data();

    std::vector<uint16_t> activeCellVec;
    std::vector<uint16_t> activeUtVec;

    if (activeCell.size() > 0) {
        activeCellVec.reserve(activeCell.size());
        const int* cell_data = activeCell.data();
        for (size_t i = 0; i < activeCell.size(); ++i) {
            activeCellVec.push_back(static_cast<uint16_t>(cell_data[i]));
        }
    }

    if (activeUt.size() > 0) {
        activeUtVec.reserve(activeUt.size());
        const int* ut_data = activeUt.data();
        for (size_t i = 0; i < activeUt.size(); ++i) {
            activeUtVec.push_back(static_cast<uint16_t>(ut_data[i]));
        }
    }

    m_statisChanModelHandle->dump_pl_sf_ant_gain_stats(pl_sf_ant_gain_ptr, activeCellVec, activeUtVec);
}

template <typename Tscalar, typename Tcomplex>
void StatisChanModelWrapper<Tscalar, Tcomplex>::dump_topology_to_yaml(const std::string& filename) {
    m_statisChanModelHandle->dump_topology_to_yaml(filename);
}

template <typename Tscalar, typename Tcomplex>
void StatisChanModelWrapper<Tscalar, Tcomplex>::saveSlsChanToH5File(std::string_view filename_ending) {
    m_statisChanModelHandle->saveSlsChanToH5File(filename_ending);
}

}