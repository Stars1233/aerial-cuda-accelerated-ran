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

// Channel-model wrapper implementations exposed through nanobind.
#include <cstddef>
#include <cstring>
#include <optional>
#include <numeric>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include "gpu3gppchan_bindings.hpp"
#include "cuda_array_interface.hpp"

namespace nb = nanobind;

namespace gpu3gppchan_bindings {

namespace {

template <typename Container>
uint16_t checkedAntennaCount(const Container& dimensions, const char* fieldName)
{
    size_t count{1};
    constexpr size_t maxCount{std::numeric_limits<uint16_t>::max()};
    for (const auto dimension : dimensions) {
        const size_t value{static_cast<size_t>(dimension)};
        if (value != 0 && count > maxCount / value) {
            throw std::invalid_argument(
                std::string(fieldName) + " product exceeds native uint16 capacity");
        }
        count *= value;
    }
    return static_cast<uint16_t>(count);
}

uint16_t checkedSelectorIndex(const int value, const char* selectorName)
{
    if (value < 0 || value > std::numeric_limits<uint16_t>::max()) {
        throw std::invalid_argument(
            std::string(selectorName) + " values must fit in uint16_t");
    }
    return static_cast<uint16_t>(value);
}

struct OutputBuffer {
    void* data;
    size_t nbytes;
};

size_t checkedSizeProduct(const size_t lhs, const size_t rhs, const char* bufferName)
{
    if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs) {
        throw nb::value_error((std::string(bufferName) + " size overflow").c_str());
    }
    return lhs * rhs;
}

template <typename T>
void validateChannelInput(
    const cuda_array_t<T>& input,
    const size_t expectedElements,
    const uint8_t columnMajor,
    const char* channelName)
{
    if (input.get_size() != expectedElements) {
        throw nb::value_error(
            (std::string(channelName) + " input has "
             + std::to_string(input.get_size()) + " elements; expected "
             + std::to_string(expectedElements)).c_str());
    }
    const bool contiguous = columnMajor != 0
                                ? input.is_f_contiguous()
                                : input.is_c_contiguous();
    if (!contiguous) {
        throw nb::value_error(
            (std::string(channelName) + " input layout does not match tx_column_major_ind")
                .c_str());
    }
}

size_t getTypestrItemSize(const std::string& typestr)
{
    const auto digits = typestr.find_first_of("0123456789");
    if (digits == std::string::npos) {
        throw nb::value_error("Channel output buffer has an invalid typestr");
    }
    return std::stoul(typestr.substr(digits));
}

void validateWritableContiguousInterface(
    const nb::dict& interface, const char* bufferName)
{
    if (!interface.contains("data")) {
        throw nb::value_error(
            (std::string(bufferName) + " is missing a data pointer").c_str());
    }
    const auto data = nb::cast<nb::tuple>(nb::cast<nb::object>(interface["data"]));
    if (data.size() < 2) {
        throw nb::value_error(
            (std::string(bufferName) + " has an invalid data tuple").c_str());
    }
    if (nb::cast<bool>(data[1])) {
        throw nb::value_error(
            (std::string(bufferName) + " must be writable").c_str());
    }

    if (!interface.contains("strides")) {
        return;
    }
    const auto stridesObject = nb::cast<nb::object>(interface["strides"]);
    if (stridesObject.is_none()) {
        return;
    }
    if (!interface.contains("shape") || !interface.contains("typestr")) {
        throw nb::value_error(
            (std::string(bufferName) + " lacks shape or dtype metadata").c_str());
    }

    const auto shape = nb::cast<nb::tuple>(nb::cast<nb::object>(interface["shape"]));
    const auto strides = nb::cast<nb::tuple>(stridesObject);
    if (shape.size() != strides.size()) {
        throw nb::value_error(
            (std::string(bufferName) + " has invalid stride metadata").c_str());
    }

    size_t expectedStride = getTypestrItemSize(
        nb::cast<std::string>(interface["typestr"]));
    for (size_t dimension = shape.size(); dimension-- > 0;) {
        const size_t extent = nb::cast<size_t>(shape[dimension]);
        const size_t stride = nb::cast<size_t>(strides[dimension]);
        if (extent > 1 && stride != expectedStride) {
            throw nb::value_error(
                (std::string(bufferName) + " must be C-contiguous").c_str());
        }
        expectedStride = checkedSizeProduct(
            expectedStride, extent, bufferName);
    }
}

OutputBuffer getOutputBuffer(nb::handle array, const bool cpuOnlyMode)
{
    const char* interfaceName = cpuOnlyMode
                                    ? "__array_interface__"
                                    : "__cuda_array_interface__";
    if (!nb::hasattr(array, interfaceName)) {
        throw std::invalid_argument(
            std::string("Channel output buffer must expose ") + interfaceName);
    }

    auto interface = nb::cast<nb::dict>(nb::getattr(array, interfaceName));
    validateWritableContiguousInterface(interface, "Channel output buffer");
    auto data = nb::cast<nb::tuple>(nb::cast<nb::object>(interface["data"]));
    const auto address = nb::cast<uintptr_t>(data[0]);
    const auto nbytes = nb::cast<size_t>(nb::getattr(array, "nbytes"));
    if (address == 0 && nbytes != 0) {
        throw nb::value_error("Channel output buffer exposes a null data pointer");
    }
    return {
        reinterpret_cast<void*>(address),
        nbytes,
    };
}

template <typename SourceFn, typename SizeFn>
void copyInternalChannelBuffers(
    nb::object arrays,
    const std::vector<std::vector<uint16_t>>& activeUt,
    const bool cpuOnlyMode,
    cudaStream_t stream,
    SourceFn sourceForCell,
    SizeFn expectedBytesForCell,
    const char* bufferName)
{
    if (arrays.is_none()) {
        return;
    }
    if (!nb::isinstance<nb::list>(arrays)) {
        throw std::invalid_argument(std::string(bufferName) + " must be a list or None");
    }
    auto arrayList = nb::cast<nb::list>(arrays);
    if (arrayList.size() != activeUt.size()) {
        throw std::invalid_argument(
            std::string(bufferName) + " must contain one buffer per active cell");
    }

    bool queuedCopy{false};
    for (size_t cellIdx = 0; cellIdx < activeUt.size(); ++cellIdx) {
        auto array = nb::cast<nb::object>(arrayList[cellIdx]);
        const auto output = getOutputBuffer(array, cpuOnlyMode);
        const size_t expectedBytes = expectedBytesForCell(cellIdx);
        if (output.nbytes != expectedBytes) {
            throw std::invalid_argument(
                std::string(bufferName) + " has an unexpected size for active cell "
                + std::to_string(cellIdx));
        }
        if (expectedBytes == 0) {
            continue;
        }
        const void* source = sourceForCell(cellIdx);
        if (source == nullptr) {
            throw std::runtime_error(
                std::string("Internal ") + bufferName + " data is unavailable");
        }
        if (cpuOnlyMode) {
            std::memcpy(output.data, source, expectedBytes);
            continue;
        }
        const auto status = cudaMemcpyAsync(
            output.data, source, expectedBytes, cudaMemcpyDeviceToDevice, stream);
        if (status != cudaSuccess) {
            throw std::runtime_error(
                std::string("Failed to copy internal ") + bufferName + ": "
                + cudaGetErrorString(status));
        }
        queuedCopy = true;
    }
    if (queuedCopy) {
        const auto status = cudaStreamSynchronize(stream);
        if (status != cudaSuccess) {
            throw std::runtime_error(
                std::string("Failed to synchronize after copying internal ") + bufferName
                + ": " + cudaGetErrorString(status));
        }
    }
}

}  // namespace

// Helper function to create cuda_array_t for channel output signal
// Avoids code duplication between TdlChanWrapper and CdlChanWrapper
template <typename Tscalar>
cuda_array_t<std::complex<Tscalar>> createRxSignalOutArray(
    void* ptr,
    size_t nCell,
    size_t nUe,
    size_t nRxAnt,
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
        nCell,
        nUe,
        nRxAnt,
        static_cast<size_t>(sigLenPerAnt)
    };
    std::vector<size_t> strides = {
        nUe * nRxAnt *
            static_cast<size_t>(sigLenPerAnt) * sizeof(std::complex<Tscalar>),
        nRxAnt * static_cast<size_t>(sigLenPerAnt) *
            sizeof(std::complex<Tscalar>),
        static_cast<size_t>(sigLenPerAnt) * sizeof(std::complex<Tscalar>),
        sizeof(std::complex<Tscalar>)
    };

    return cuda_array_t<std::complex<Tscalar>>(reinterpret_cast<intptr_t>(ptr), shape, strides);
}

/*-------------------------------       OFDM modulation class       -------------------------------*/
template <typename Tscalar, typename Tcomplex>
OfdmModulateWrapper<Tscalar, Tcomplex>::OfdmModulateWrapper(cuphyCarrierPrms_t* carrierParams, HostComplexArray<Tscalar> freqDataInCpu, uintptr_t streamHandle) :
m_carrierParams(*carrierParams),
m_freqDataInCpuOwner(freqDataInCpu),
m_cuStrm(reinterpret_cast<cudaStream_t>(streamHandle)),
m_externGpuAlloc(0)
{
    // buffer size from config
    m_freqDataInSizeDl = static_cast<size_t>(m_carrierParams.N_sc) *
        static_cast<size_t>(m_carrierParams.N_bsLayer) *
        static_cast<size_t>(m_carrierParams.N_symbol_slot);
    m_freqDataInSizeUl = static_cast<size_t>(m_carrierParams.N_sc) *
        static_cast<size_t>(m_carrierParams.N_ueLayer) *
        static_cast<size_t>(m_carrierParams.N_symbol_slot);
    const std::size_t requiredCapacity = std::max(m_freqDataInSizeDl, m_freqDataInSizeUl);
    if (m_freqDataInCpuOwner->size() < requiredCapacity) {
        throw std::invalid_argument(
            "freqDataInCpu has " + std::to_string(m_freqDataInCpuOwner->size()) +
            " elements; expected at least " + std::to_string(requiredCapacity));
    }
    // get host pointer from the NumPy array
    m_freqDataInCpu = reinterpret_cast<Tcomplex*>(m_freqDataInCpuOwner->data());

    // allocate GPU buffer
    m_freqDataInGpu = nullptr;
    m_ofdmModulateHandle = nullptr;
    {
        size_t allocSize = sizeof(Tcomplex) * std::max(m_freqDataInSizeDl, m_freqDataInSizeUl);
        cudaError_t err = cudaMalloc(&m_freqDataInGpu, allocSize);
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
        m_ofdmModulateHandle = new ofdm_modulate::ofdmModulate<Tscalar, Tcomplex>(&m_carrierParams, m_freqDataInGpu, m_cuStrm);
    } catch (...) {
        cudaFree(m_freqDataInGpu);
        m_freqDataInGpu = nullptr;
        throw;
    }
}

template <typename Tscalar, typename Tcomplex>
OfdmModulateWrapper<Tscalar, Tcomplex>::OfdmModulateWrapper(cuphyCarrierPrms_t* carrierParams, uintptr_t freqDataInGpu, uintptr_t streamHandle) :
m_carrierParams(*carrierParams),
m_cuStrm(reinterpret_cast<cudaStream_t>(streamHandle)),
m_externGpuAlloc(1)
{
    m_freqDataInGpu = reinterpret_cast<Tcomplex*>(freqDataInGpu);
    m_ofdmModulateHandle = new ofdm_modulate::ofdmModulate<Tscalar, Tcomplex>(&m_carrierParams, m_freqDataInGpu, m_cuStrm);
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
void OfdmModulateWrapper<Tscalar, Tcomplex>::run(std::optional<HostComplexArray<Tscalar>> freqDataInCpu, uint8_t enableSwapTxRx)
{
    const size_t freqDataInSize =
        enableSwapTxRx ? m_freqDataInSizeUl : m_freqDataInSizeDl;
    if(freqDataInCpu.has_value() && freqDataInCpu->size() != 0) // new input numpy array, need to copy new data to GPU
    {
        // The H2D copy is asynchronous. Keep its source alive until the stream
        // has finished using the previous replacement.
        if (m_freqDataInRunOwner.has_value()) {
            cudaError_t err = cudaStreamSynchronize(m_cuStrm);
            if (err != cudaSuccess) {
                throw std::runtime_error(
                    std::string("cudaStreamSynchronize failed before replacing OFDM input: ") +
                    cudaGetErrorString(err));
            }
        }
        m_freqDataInRunOwner = *freqDataInCpu;
        // get host pointer from the NumPy array
        if (m_freqDataInRunOwner->size() != freqDataInSize) {
            throw std::invalid_argument("freqDataInCpu has unexpected element count");
        }
        Tcomplex* freqDataInCpuNew = reinterpret_cast<Tcomplex*>(m_freqDataInRunOwner->data());

        CHECK_CUDAERROR(cudaMemcpyAsync(
            m_freqDataInGpu, freqDataInCpuNew,
            sizeof(Tcomplex) * freqDataInSize, cudaMemcpyHostToDevice, m_cuStrm));
    }
    else
    {
        if (!m_externGpuAlloc) // use numpy array, need to copy new data to GPU
        {
            CHECK_CUDAERROR(cudaMemcpyAsync(
                m_freqDataInGpu, m_freqDataInCpu,
                sizeof(Tcomplex) * freqDataInSize, cudaMemcpyHostToDevice, m_cuStrm));
        }
    }
    m_ofdmModulateHandle -> run(enableSwapTxRx, m_cuStrm);
}

/*-------------------------------       OFDM demodulation class       -------------------------------*/
template <typename Tscalar, typename Tcomplex>
OfdmDeModulateWrapper<Tscalar, Tcomplex>::OfdmDeModulateWrapper(cuphyCarrierPrms_t * carrierParams, uintptr_t timeDataInGpu, HostComplexArray<Tscalar> freqDataOutCpu, bool prach, bool perAntSamp, uintptr_t streamHandle) :
m_carrierParams(*carrierParams),
m_freqDataOutCpuOwner(freqDataOutCpu),
m_cuStrm(reinterpret_cast<cudaStream_t>(streamHandle)),
m_externGpuAlloc(0),
m_perAntSamp(perAntSamp)
{
    // buffer size from config
    m_freqDataOutSizeDl = static_cast<size_t>(m_carrierParams.N_sc) *
        static_cast<size_t>(m_carrierParams.N_ueLayer) *
        static_cast<size_t>(m_carrierParams.N_symbol_slot);
    m_freqDataOutSizeUl = static_cast<size_t>(m_carrierParams.N_sc) *
        static_cast<size_t>(m_carrierParams.N_bsLayer) *
        static_cast<size_t>(m_carrierParams.N_symbol_slot);
    if (m_perAntSamp)
    {
        m_freqDataOutSizeDl *= m_carrierParams.N_bsLayer;
        m_freqDataOutSizeUl *= m_carrierParams.N_ueLayer;
    }
    const std::size_t requiredCapacity = std::max(m_freqDataOutSizeDl, m_freqDataOutSizeUl);
    if (m_freqDataOutCpuOwner->size() < requiredCapacity) {
        throw std::invalid_argument(
            "freqDataOutCpu has " + std::to_string(m_freqDataOutCpuOwner->size()) +
            " elements; expected at least " + std::to_string(requiredCapacity));
    }
    // get host pointer from the NumPy array
    m_freqDataOutCpu = reinterpret_cast<Tcomplex*>(m_freqDataOutCpuOwner->data());

    // allocate GPU buffer
    m_freqDataOutGpu = nullptr;
    m_ofdmDeModulateHandle = nullptr;
    {
        size_t allocSize = sizeof(Tcomplex) * std::max(m_freqDataOutSizeDl, m_freqDataOutSizeUl);
        cudaError_t err = cudaMalloc(&m_freqDataOutGpu, allocSize);
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
        m_ofdmDeModulateHandle = new ofdm_demodulate::ofdmDeModulate<Tscalar, Tcomplex>(
            &m_carrierParams, reinterpret_cast<Tcomplex*>(timeDataInGpu),
            m_freqDataOutGpu, prach, perAntSamp, m_cuStrm);
    } catch (...) {
        cudaFree(m_freqDataOutGpu);
        m_freqDataOutGpu = nullptr;
        throw;
    }
}

template <typename Tscalar, typename Tcomplex>
OfdmDeModulateWrapper<Tscalar, Tcomplex>::OfdmDeModulateWrapper(cuphyCarrierPrms_t * carrierParams, uintptr_t timeDataInGpu, uintptr_t freqDataOutGpu, bool prach, bool perAntSamp, uintptr_t streamHandle) :
m_carrierParams(*carrierParams),
m_cuStrm(reinterpret_cast<cudaStream_t>(streamHandle)),
m_externGpuAlloc(1),
m_perAntSamp(perAntSamp)
{
    // buffer size from config
    m_freqDataOutSizeDl = static_cast<size_t>(m_carrierParams.N_sc) *
        static_cast<size_t>(m_carrierParams.N_ueLayer) *
        static_cast<size_t>(m_carrierParams.N_symbol_slot);
    m_freqDataOutSizeUl = static_cast<size_t>(m_carrierParams.N_sc) *
        static_cast<size_t>(m_carrierParams.N_bsLayer) *
        static_cast<size_t>(m_carrierParams.N_symbol_slot);
    if (m_perAntSamp)
    {
        m_freqDataOutSizeDl *= m_carrierParams.N_bsLayer;
        m_freqDataOutSizeUl *= m_carrierParams.N_ueLayer;
    }
    m_freqDataOutCpu = nullptr;
    m_freqDataOutGpu = reinterpret_cast<Tcomplex*>(freqDataOutGpu);
    m_ofdmDeModulateHandle = new ofdm_demodulate::ofdmDeModulate<Tscalar, Tcomplex>(
        &m_carrierParams, reinterpret_cast<Tcomplex*>(timeDataInGpu),
        m_freqDataOutGpu, prach, perAntSamp, m_cuStrm);
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
void OfdmDeModulateWrapper<Tscalar, Tcomplex>::run(std::optional<HostComplexArray<Tscalar>> freqDataOutCpu, uint8_t enableSwapTxRx)
{
    m_ofdmDeModulateHandle -> run(enableSwapTxRx, m_cuStrm);
    const size_t freqDataOutSize =
        enableSwapTxRx ? m_freqDataOutSizeUl : m_freqDataOutSizeDl;
    if(freqDataOutCpu.has_value() && freqDataOutCpu->size() != 0) // new output numpy array, need to copy new data from GPU
    {
        m_freqDataOutRunOwner = *freqDataOutCpu;
        if (m_freqDataOutRunOwner->size() != freqDataOutSize) {
            throw std::invalid_argument("freqDataOutCpu has unexpected element count");
        }
        Tcomplex* freqDataOutCpuNew = reinterpret_cast<Tcomplex*>(m_freqDataOutRunOwner->data());

        CHECK_CUDAERROR(cudaMemcpyAsync(
            freqDataOutCpuNew, m_freqDataOutGpu,
            sizeof(Tcomplex) * freqDataOutSize, cudaMemcpyDeviceToHost, m_cuStrm));
    }
    else if (!m_externGpuAlloc)
    {
        CHECK_CUDAERROR(cudaMemcpyAsync(
            m_freqDataOutCpu, m_freqDataOutGpu,
            sizeof(Tcomplex) * freqDataOutSize, cudaMemcpyDeviceToHost, m_cuStrm));
    }
    if ((freqDataOutCpu.has_value() && freqDataOutCpu->size() != 0) ||
        !m_externGpuAlloc) {
        CHECK_CUDAERROR(cudaStreamSynchronize(m_cuStrm));
    }
}

/*-------------------------------       TDL channel class       -------------------------------*/
template <typename Tscalar, typename Tcomplex>
TdlChanWrapper<Tscalar, Tcomplex>::TdlChanWrapper(tdlConfig_t* tdlCfg, uint16_t randSeed, uintptr_t streamHandle) :
m_cuStrm(reinterpret_cast<cudaStream_t>(streamHandle)),
m_nLink(static_cast<size_t>(tdlCfg->nCell) * static_cast<size_t>(tdlCfg->nUe)),
m_runMode(tdlCfg -> runMode),
m_tdlCfg(*tdlCfg)
{
    m_tdlCfg.txSigIn = nullptr;
    m_tdlChanHandle = new tdlChan<Tscalar, Tcomplex>(&m_tdlCfg, randSeed, m_cuStrm);
}

template <typename Tscalar, typename Tcomplex>
TdlChanWrapper<Tscalar, Tcomplex>::~TdlChanWrapper()
{
    delete m_tdlChanHandle;
}

template <typename Tscalar, typename Tcomplex>
void TdlChanWrapper<Tscalar, Tcomplex>::run(const cuda_array_t<std::complex<Tscalar>>& txSigIn, float refTime0, uint8_t enableSwapTxRx, uint8_t txColumnMajorInd)
{
    size_t expectedElements = checkedSizeProduct(
        static_cast<size_t>(m_tdlCfg.nCell), static_cast<size_t>(m_tdlCfg.nUe),
        "TDL input");
    const size_t nTxAntennas = enableSwapTxRx != 0
                                   ? static_cast<size_t>(m_tdlCfg.nUeAnt)
                                   : static_cast<size_t>(m_tdlCfg.nBsAnt);
    expectedElements = checkedSizeProduct(expectedElements, nTxAntennas, "TDL input");
    expectedElements = checkedSizeProduct(
        expectedElements, static_cast<size_t>(m_tdlCfg.sigLenPerAnt), "TDL input");
    validateChannelInput(txSigIn, expectedElements, txColumnMajorInd, "TDL");

    // Update the tx signal pointer in the channel's descriptor (both CPU and GPU)
    m_tdlChanHandle->setTxSigIn(reinterpret_cast<Tcomplex*>(txSigIn.get_device_ptr()));
    m_tdlChanHandle->run(refTime0, enableSwapTxRx, txColumnMajorInd);
    // Output is written to internal buffer, accessible via getRxSignalOutArray()
}

template <typename Tscalar, typename Tcomplex>
cuda_array_t<std::complex<Tscalar>> TdlChanWrapper<Tscalar, Tcomplex>::getRxSignalOutArray(uint8_t enableSwapTxRx)
{
    // TDL config has direct nBsAnt/nUeAnt members
    uint16_t nRxAnt = enableSwapTxRx ? m_tdlCfg.nBsAnt : m_tdlCfg.nUeAnt;

    return createRxSignalOutArray<Tscalar>(
        m_tdlChanHandle->getRxSigOut(),
        m_tdlCfg.nCell,
        m_tdlCfg.nUe,
        nRxAnt,
        m_tdlCfg.sigLenPerAnt
    );
}

template <typename Tscalar, typename Tcomplex>
void TdlChanWrapper<Tscalar, Tcomplex>::dumpCir(HostComplexArray<Tscalar> cirCpu)
{
    // buffer size from config
    uint32_t timeChanSize = m_tdlChanHandle -> getTimeChanSize();

    if (cirCpu.size() != timeChanSize) {
        throw std::invalid_argument("cirCpu has unexpected element count");
    }

    // copy CIR
    CHECK_CUDAERROR(cudaMemcpyAsync(
        cirCpu.data(), m_tdlChanHandle->getTimeChan(),
        sizeof(Tcomplex) * timeChanSize, cudaMemcpyDeviceToHost, m_cuStrm));
    CHECK_CUDAERROR(cudaStreamSynchronize(m_cuStrm));
}

template <typename Tscalar, typename Tcomplex>
void TdlChanWrapper<Tscalar, Tcomplex>::dumpCfrPrbg(HostComplexArray<Tscalar> cfrPrbg)
{
    // dump CFR on PRBG
    if(m_runMode > 0 && m_runMode < 3)
    {
        // buffer size from config
        uint32_t freqChanPrbgSize = m_tdlChanHandle -> getFreqChanPrbgSize();

        if (cfrPrbg.size() != freqChanPrbgSize) {
            throw std::invalid_argument("cfrPrbg has unexpected element count");
        }

        // copy CFR on PRBG
        CHECK_CUDAERROR(cudaMemcpyAsync(
            cfrPrbg.data(), m_tdlChanHandle->getFreqChanPrbg(),
            sizeof(Tcomplex) * freqChanPrbgSize, cudaMemcpyDeviceToHost, m_cuStrm));
        CHECK_CUDAERROR(cudaStreamSynchronize(m_cuStrm));
    }
}

template <typename Tscalar, typename Tcomplex>
void TdlChanWrapper<Tscalar, Tcomplex>::dumpCfrSc(HostComplexArray<Tscalar> cfrSc)
{
    // dump CFR on SC
    if(m_runMode > 1 && m_runMode < 3)
    {
        // buffer size from config
        uint32_t freqChanScSizePerLink = m_tdlChanHandle -> getFreqChanScPerLinkSize();
        Tcomplex ** freqChanSc = m_tdlChanHandle -> getFreqChanScHostArray(); // CFR on SC is saved using pointer of pointers
        const size_t expected_size = static_cast<size_t>(m_nLink) * freqChanScSizePerLink;
        if (cfrSc.size() != expected_size) {
            throw std::invalid_argument("cfrSc has unexpected element count");
        }

        // copy CFR on SC
        Tcomplex * freqChScCpuOut = reinterpret_cast<Tcomplex*>(cfrSc.data());
        for (size_t linkIdx = 0; linkIdx < m_nLink; ++linkIdx)
        {
            CHECK_CUDAERROR(cudaMemcpyAsync(
                freqChScCpuOut, freqChanSc[linkIdx],
                sizeof(Tcomplex) * freqChanScSizePerLink,
                cudaMemcpyDeviceToHost, m_cuStrm));
            freqChScCpuOut += freqChanScSizePerLink; // CPU address for CFR on SC of next link
        }
        CHECK_CUDAERROR(cudaStreamSynchronize(m_cuStrm));
    }
}

/*-------------------------------       CDL channel class       -------------------------------*/
template <typename Tscalar, typename Tcomplex>
CdlChanWrapper<Tscalar, Tcomplex>::CdlChanWrapper(cdlConfig_t* cdlCfg, uint16_t randSeed, uintptr_t streamHandle) :
m_cuStrm(reinterpret_cast<cudaStream_t>(streamHandle)),
m_nLink(static_cast<size_t>(cdlCfg->nCell) * static_cast<size_t>(cdlCfg->nUe)),
m_runMode(cdlCfg -> runMode),
m_cdlCfg(*cdlCfg)
{
    m_nBsAnt = checkedAntennaCount(m_cdlCfg.bsAntSize, "bsAntSize");
    m_nUeAnt = checkedAntennaCount(m_cdlCfg.ueAntSize, "ueAntSize");
    m_cdlCfg.txSigIn = nullptr;
    m_cdlChanHandle = new cdlChan<Tscalar, Tcomplex>(&m_cdlCfg, randSeed, m_cuStrm);
}

template <typename Tscalar, typename Tcomplex>
CdlChanWrapper<Tscalar, Tcomplex>::~CdlChanWrapper()
{
    delete m_cdlChanHandle;
}

template <typename Tscalar, typename Tcomplex>
void CdlChanWrapper<Tscalar, Tcomplex>::run(const cuda_array_t<std::complex<Tscalar>>& txSigIn, float refTime0, uint8_t enableSwapTxRx, uint8_t txColumnMajorInd)
{
    size_t expectedElements = checkedSizeProduct(
        static_cast<size_t>(m_cdlCfg.nCell), static_cast<size_t>(m_cdlCfg.nUe),
        "CDL input");
    const size_t nTxAntennas = enableSwapTxRx != 0
                                   ? static_cast<size_t>(m_nUeAnt)
                                   : static_cast<size_t>(m_nBsAnt);
    expectedElements = checkedSizeProduct(expectedElements, nTxAntennas, "CDL input");
    expectedElements = checkedSizeProduct(
        expectedElements, static_cast<size_t>(m_cdlCfg.sigLenPerAnt), "CDL input");
    validateChannelInput(txSigIn, expectedElements, txColumnMajorInd, "CDL");

    // Update the tx signal pointer in the channel's descriptor (both CPU and GPU)
    m_cdlChanHandle->setTxSigIn(reinterpret_cast<Tcomplex*>(txSigIn.get_device_ptr()));
    m_cdlChanHandle->run(refTime0, enableSwapTxRx, txColumnMajorInd);
    // Output is written to internal buffer, accessible via getRxSignalOutArray()
}

template <typename Tscalar, typename Tcomplex>
cuda_array_t<std::complex<Tscalar>> CdlChanWrapper<Tscalar, Tcomplex>::getRxSignalOutArray(uint8_t enableSwapTxRx)
{
    const uint16_t nRxAnt = enableSwapTxRx ? m_nBsAnt : m_nUeAnt;
    return createRxSignalOutArray<Tscalar>(
        m_cdlChanHandle->getRxSigOut(),
        m_cdlCfg.nCell,
        m_cdlCfg.nUe,
        nRxAnt,
        m_cdlCfg.sigLenPerAnt
    );
}

template <typename Tscalar, typename Tcomplex>
void CdlChanWrapper<Tscalar, Tcomplex>::dumpCir(HostComplexArray<Tscalar> cirCpu)
{
    // buffer size from config
    uint32_t timeChanSize = m_cdlChanHandle -> getTimeChanSize();

    if (cirCpu.size() != timeChanSize) {
        throw std::invalid_argument("cirCpu has unexpected element count");
    }

    // copy CIR
    CHECK_CUDAERROR(cudaMemcpyAsync(
        cirCpu.data(), m_cdlChanHandle->getTimeChan(),
        sizeof(Tcomplex) * timeChanSize, cudaMemcpyDeviceToHost, m_cuStrm));
    CHECK_CUDAERROR(cudaStreamSynchronize(m_cuStrm));
}

template <typename Tscalar, typename Tcomplex>
void CdlChanWrapper<Tscalar, Tcomplex>::dumpCfrPrbg(HostComplexArray<Tscalar> cfrPrbg)
{
    // dump CFR on PRBG
    if(m_runMode > 0 && m_runMode < 3)
    {
        // buffer size from config
        uint32_t freqChanPrbgSize = m_cdlChanHandle -> getFreqChanPrbgSize();

        if (cfrPrbg.size() != freqChanPrbgSize) {
            throw std::invalid_argument("cfrPrbg has unexpected element count");
        }

        // copy CFR on PRBG
        CHECK_CUDAERROR(cudaMemcpyAsync(
            cfrPrbg.data(), m_cdlChanHandle->getFreqChanPrbg(),
            sizeof(Tcomplex) * freqChanPrbgSize, cudaMemcpyDeviceToHost, m_cuStrm));
        CHECK_CUDAERROR(cudaStreamSynchronize(m_cuStrm));
    }
}

template <typename Tscalar, typename Tcomplex>
void CdlChanWrapper<Tscalar, Tcomplex>::dumpCfrSc(HostComplexArray<Tscalar> cfrSc)
{
    // dump CFR on SC
    if(m_runMode > 1 && m_runMode < 3)
    {
        // buffer size from config
        uint32_t freqChanScSizePerLink = m_cdlChanHandle -> getFreqChanScPerLinkSize();
        Tcomplex ** freqChanSc = m_cdlChanHandle -> getFreqChanScHostArray(); // CFR on SC is saved using pointer of pointers
        const size_t expected_size = static_cast<size_t>(m_nLink) * freqChanScSizePerLink;
        if (cfrSc.size() != expected_size) {
            throw std::invalid_argument("cfrSc has unexpected element count");
        }

        // copy CFR on SC
        Tcomplex * freqChScCpuOut = reinterpret_cast<Tcomplex*>(cfrSc.data());
        for (size_t linkIdx = 0; linkIdx < m_nLink; ++linkIdx)
        {
            CHECK_CUDAERROR(cudaMemcpyAsync(
                freqChScCpuOut, freqChanSc[linkIdx],
                sizeof(Tcomplex) * freqChanScSizePerLink,
                cudaMemcpyDeviceToHost, m_cuStrm));
            freqChScCpuOut += freqChanScSizePerLink; // CPU address for CFR on SC of next link
        }
        CHECK_CUDAERROR(cudaStreamSynchronize(m_cuStrm));
    }
}

/*-------------------------------       add Gaussian noise class       -------------------------------*/
template <typename Tscalar, typename Tcomplex>
GauNoiseAdderWrapper<Tscalar, Tcomplex>::GauNoiseAdderWrapper(uint32_t nThreads, int seed, uintptr_t streamHandle) :
m_cuStrm(reinterpret_cast<cudaStream_t>(streamHandle))
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
    m_gauNoiseAdder -> addNoise(
        reinterpret_cast<Tcomplex*>(d_signal), signalSize, snr_db);
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
m_simConfig(sim_config),
m_systemLevelConfig(system_level_config),
m_linkLevelConfig(link_level_config),
m_externalConfig(external_config),
m_cuStrm(reinterpret_cast<cudaStream_t>(streamHandle)),
m_randSeed(randSeed)
{
    m_statisChanModelHandle = createStatisChanModelFloat(
        &m_simConfig, &m_systemLevelConfig, &*m_linkLevelConfig, &*m_externalConfig,
        m_randSeed, m_cuStrm);
    m_cpuOnlyMode = m_simConfig.cpu_only_mode;
}

// Constructor with just sim_config and system_level_config
template <typename Tscalar, typename Tcomplex>
StatisChanModelWrapper<Tscalar, Tcomplex>::StatisChanModelWrapper(
    const SimConfig& sim_config,
    const SystemLevelConfig& system_level_config,
    uint32_t randSeed,
    uintptr_t streamHandle) :
m_simConfig(sim_config),
m_systemLevelConfig(system_level_config),
m_cuStrm(reinterpret_cast<cudaStream_t>(streamHandle)),
m_randSeed(randSeed)
{
    m_statisChanModelHandle = createStatisChanModelFloat(
        &m_simConfig, &m_systemLevelConfig, nullptr, nullptr,
        m_randSeed, m_cuStrm);
    m_cpuOnlyMode = m_simConfig.cpu_only_mode;
}

template <typename Tscalar, typename Tcomplex>
StatisChanModelWrapper<Tscalar, Tcomplex>::~StatisChanModelWrapper()
{
    destroyStatisChanModelFloat(m_statisChanModelHandle);
}

template <typename Tscalar, typename Tcomplex>
void StatisChanModelWrapper<Tscalar, Tcomplex>::run(
    float refTime,
    uint8_t continuous_fading,
    nb::object activeCell,
    nb::object activeUt,
    nb::object utNewLoc,
    nb::object utNewVelocity,
    nb::object cir_coe,
    nb::object cir_norm_delay,
    nb::object cir_n_taps,
    nb::object cfr_sc,
    nb::object cfr_prbg) {

    // Convert activeCell to vector
    std::vector<uint16_t> activeCellVec;
    if (!activeCell.is_none()) {
        if (!nb::isinstance<nb::list>(activeCell)) {
            throw nb::type_error("active_cell must be a list or None");
        }
        auto cellList = nb::cast<nb::list>(activeCell);
        for (auto item : cellList) {
            activeCellVec.push_back(nb::cast<uint16_t>(item));
        }
    }

    // Convert activeUt to nested vector
    std::vector<std::vector<uint16_t>> activeUtVec;
    if (!activeUt.is_none()) {
        if (!nb::isinstance<nb::list>(activeUt)) {
            throw nb::type_error("active_ut must be a list of lists or None");
        }
        auto utList = nb::cast<nb::list>(activeUt);
        for (auto item : utList) {
            if (!nb::isinstance<nb::list>(item)) {
                throw nb::type_error("each active_ut entry must be a list");
            }
            std::vector<uint16_t> utVec;
            auto innerList = nb::cast<nb::list>(item);
            for (auto ut : innerList) {
                utVec.push_back(nb::cast<uint16_t>(ut));
            }
            activeUtVec.push_back(std::move(utVec));
        }
    }

    // Convert utNewLoc to vector of Coordinates
    std::vector<Coordinate> utNewLocVec;
    {
        nb::ndarray<float, nb::ndim<2>, nb::c_contig, nb::device::cpu> arr;
        if (!utNewLoc.is_none()) {
            if (!nb::try_cast(utNewLoc, arr)) {
                throw nb::value_error(
                    "ut_new_loc must be a C-contiguous float32 array with shape (N, 3)");
            }
            size_t rows = arr.shape(0);
            size_t cols = arr.shape(1);
            if (cols != 3) {
                throw nb::value_error("ut_new_loc must have shape (N, 3)");
            }
            const float* d = arr.data();
            for (size_t i = 0; i < rows; i++) {
                Coordinate coord;
                coord.x = d[i * cols + 0];
                coord.y = d[i * cols + 1];
                coord.z = d[i * cols + 2];
                utNewLocVec.push_back(coord);
            }
        }
    }

    // Convert utNewVelocity to vector of float3
    std::vector<float3> utNewVelocityVec;
    {
        nb::ndarray<float, nb::ndim<2>, nb::c_contig, nb::device::cpu> arr;
        if (!utNewVelocity.is_none()) {
            if (!nb::try_cast(utNewVelocity, arr)) {
                throw nb::value_error(
                    "ut_new_velocity must be a C-contiguous float32 array with shape (N, 3)");
            }
            size_t rows = arr.shape(0);
            size_t cols = arr.shape(1);
            if (cols != 3) {
                throw nb::value_error("ut_new_velocity must have shape (N, 3)");
            }
            const float* d = arr.data();
            for (size_t i = 0; i < rows; i++) {
                float3 vel;
                vel.x = d[i * cols + 0];
                vel.y = d[i * cols + 1];
                vel.z = d[i * cols + 2];
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

    // Helper lambda to extract device pointers, or host pointers in CPU-only mode.
    const bool allow_host_pointers = (m_cpuOnlyMode != 0);
    const size_t snapshots = static_cast<size_t>(m_simConfig.n_snapshot_per_slot);
    const size_t ueAntennas = m_statisChanModelHandle->getNUeAnt();
    const size_t bsAntennas = m_statisChanModelHandle->getNBsAnt();
    const size_t maxTaps = m_statisChanModelHandle->getEffectiveMaxTaps();
    const auto validateOutputList = [&](nb::object arrays, const char* bufferName,
                                        const auto& expectedBytesForCell,
                                        const char* expectedTypestr) {
        if (arrays.is_none()) {
            return;
        }
        if (!nb::isinstance<nb::list>(arrays)) {
            throw nb::type_error(
                (std::string(bufferName) + " must be provided as a list or None").c_str());
        }
        const auto arrayList = nb::cast<nb::list>(arrays);
        if (arrayList.empty()) {
            return;
        }
        if (arrayList.size() != activeUtVec.size()) {
            throw nb::value_error(
                (std::string(bufferName) + " must contain one buffer per active cell").c_str());
        }
        for (size_t cellIdx = 0; cellIdx < activeUtVec.size(); ++cellIdx) {
            const auto array = nb::cast<nb::object>(arrayList[cellIdx]);
            const auto validateTypestr = [&](nb::object arrayInterface) {
                if (nb::isinstance<nb::dict>(arrayInterface)) {
                    const auto interfaceDict = nb::cast<nb::dict>(arrayInterface);
                    if (interfaceDict.contains("typestr") &&
                        nb::cast<std::string>(interfaceDict["typestr"]) != expectedTypestr) {
                        throw nb::value_error(
                            (std::string(bufferName) + " has incompatible dtype").c_str());
                    }
                }
            };
            if (allow_host_pointers && nb::hasattr(array, "__array_interface__")) {
                const auto arrayInterface = nb::getattr(array, "__array_interface__");
                validateTypestr(arrayInterface);
                validateWritableContiguousInterface(
                    nb::cast<nb::dict>(arrayInterface), bufferName);
            }
            if (!allow_host_pointers && nb::hasattr(array, "__cuda_array_interface__")) {
                const auto arrayInterface = nb::getattr(array, "__cuda_array_interface__");
                validateTypestr(arrayInterface);
                validateWritableContiguousInterface(
                    nb::cast<nb::dict>(arrayInterface), bufferName);
            }
            if (!nb::hasattr(array, "__array_interface__") &&
                !nb::hasattr(array, "__cuda_array_interface__") &&
                nb::hasattr(array, "is_contiguous") &&
                !nb::cast<bool>(nb::getattr(array, "is_contiguous")())) {
                throw nb::value_error(
                    (std::string(bufferName) + " must be C-contiguous").c_str());
            }
            if (!nb::hasattr(array, "nbytes")) {
                throw nb::value_error(
                    (std::string(bufferName) + " must expose nbytes for capacity validation").c_str());
            }
            const auto nbytes = nb::cast<size_t>(nb::getattr(array, "nbytes"));
            if (nbytes < expectedBytesForCell(cellIdx)) {
                throw nb::value_error(
                    (std::string(bufferName) + " is too small for active cell "
                     + std::to_string(cellIdx)).c_str());
            }
        }
    };
    const auto bytesForLinks = [&](const size_t cellIdx, const size_t elementsPerLink,
                                   const size_t elementSize, const char* bufferName) {
        return checkedSizeProduct(
            checkedSizeProduct(activeUtVec[cellIdx].size(), elementsPerLink, bufferName),
            elementSize, bufferName);
    };
    validateOutputList(cir_coe, "cir_coe", [&](const size_t cellIdx) {
        size_t elements = checkedSizeProduct(snapshots, ueAntennas, "cir_coe");
        elements = checkedSizeProduct(elements, bsAntennas, "cir_coe");
        elements = checkedSizeProduct(elements, maxTaps, "cir_coe");
        return bytesForLinks(cellIdx, elements, sizeof(Tcomplex), "cir_coe");
    }, "<c8");
    validateOutputList(cir_norm_delay, "cir_norm_delay", [&](const size_t cellIdx) {
        return bytesForLinks(cellIdx, maxTaps, sizeof(uint16_t), "cir_norm_delay");
    }, "<u2");
    validateOutputList(cir_n_taps, "cir_n_taps", [&](const size_t cellIdx) {
        return bytesForLinks(cellIdx, 1, sizeof(uint16_t), "cir_n_taps");
    }, "<u2");
    const size_t cfrElements = checkedSizeProduct(
        checkedSizeProduct(snapshots, ueAntennas, "cfr"), bsAntennas, "cfr");
    validateOutputList(cfr_prbg, "cfr_prbg", [&](const size_t cellIdx) {
        return bytesForLinks(cellIdx, checkedSizeProduct(
                                 cfrElements, static_cast<size_t>(m_simConfig.n_prbg), "cfr_prbg"),
                             sizeof(Tcomplex), "cfr_prbg");
    }, "<c8");
    const size_t subcarriers = m_simConfig.run_mode == 4
                                   ? static_cast<size_t>(m_simConfig.fft_size)
                                   : checkedSizeProduct(static_cast<size_t>(m_simConfig.n_prb), 12, "cfr_sc");
    validateOutputList(cfr_sc, "cfr_sc", [&](const size_t cellIdx) {
        return bytesForLinks(cellIdx, checkedSizeProduct(cfrElements, subcarriers, "cfr_sc"),
                             sizeof(Tcomplex), "cfr_sc");
    }, "<c8");
    auto extract_array_ptrs = [allow_host_pointers, &activeUtVec](
                                  nb::object obj, auto& ptr_vec, const char* bufferName) {
        if (obj.is_none()) {
            return;
        }
        if (!nb::isinstance<nb::list>(obj)) {
            throw nb::type_error("Channel buffers must be provided as a list or None");
        }
        auto array_list = nb::cast<nb::list>(obj);
        if (!array_list.empty() && array_list.size() != activeUtVec.size()) {
            throw nb::value_error(
                (std::string(bufferName) + " must contain one buffer per active cell").c_str());
        }
        for (auto item : array_list) {
            uintptr_t array_ptr = 0;

            // Handle CuPy arrays directly using __cuda_array_interface__.
            if (nb::hasattr(item, "__cuda_array_interface__")) {
                if (allow_host_pointers) {
                    const auto message = std::string(bufferName)
                                         + " must be host-accessible in CPU-only mode";
                    throw nb::value_error(message.c_str());
                }
                nb::object array_interface = nb::getattr(item, "__cuda_array_interface__");
                if (nb::isinstance<nb::dict>(array_interface)) {
                    auto interface_dict = nb::cast<nb::dict>(array_interface);
                    if (interface_dict.contains("data")) {
                        nb::object data_info = nb::cast<nb::object>(interface_dict["data"]);
                        if (nb::isinstance<nb::tuple>(data_info)) {
                            auto data_tuple = nb::cast<nb::tuple>(data_info);
                            if (data_tuple.size() > 0) {
                                array_ptr = nb::cast<uintptr_t>(data_tuple[0]);
                            }
                        }
                    }
                }
            }
            // Host pointers are valid only when the native model runs on the CPU.
            else if (nb::hasattr(item, "__array_interface__")) {
                if (!allow_host_pointers) {
                    throw nb::type_error(
                        "GPU mode requires buffers that expose __cuda_array_interface__");
                }
                nb::object array_interface = nb::getattr(item, "__array_interface__");
                if (nb::isinstance<nb::dict>(array_interface)) {
                    auto interface_dict = nb::cast<nb::dict>(array_interface);
                    if (interface_dict.contains("data")) {
                        nb::object data_info = nb::cast<nb::object>(interface_dict["data"]);
                        if (nb::isinstance<nb::tuple>(data_info)) {
                            auto data_tuple = nb::cast<nb::tuple>(data_info);
                            if (data_tuple.size() > 0) {
                                array_ptr = nb::cast<uintptr_t>(data_tuple[0]);
                            }
                        }
                    }
                }
            }
            // Fallback: try the native wrapper with get_device_ptr().
            else if (nb::hasattr(item, "get_device_ptr") && !allow_host_pointers) {
                nb::object ptr_value = nb::getattr(item, "get_device_ptr")();
                if (nb::isinstance<nb::int_>(ptr_value)) {
                    array_ptr = nb::cast<uintptr_t>(ptr_value);
                }
            }
            // Fallback: try PyTorch-style data_ptr().
            else if (nb::hasattr(item, "data_ptr") && !allow_host_pointers) {
                if (nb::hasattr(item, "is_cuda") && !nb::cast<bool>(nb::getattr(item, "is_cuda"))) {
                    const auto message = std::string(bufferName)
                                         + " must be CUDA-accessible in GPU mode";
                    throw nb::value_error(message.c_str());
                }
                array_ptr = nb::cast<uintptr_t>(nb::getattr(item, "data_ptr")());
            }
            else {
                throw nb::type_error(
                    "Channel buffer does not expose a pointer valid for the configured mode");
            }

            if (array_ptr == 0) {
                throw nb::value_error("Channel buffer exposes a null data pointer");
            }

            using Pointer = typename std::remove_reference_t<decltype(ptr_vec)>::value_type;
            ptr_vec.push_back(reinterpret_cast<Pointer>(array_ptr));
        }
    };

    // Extract pointers from Python objects:
    // - In GPU mode, prefer __cuda_array_interface__ and device pointers
    // - In CPU-only mode, fall back to __array_interface__ host pointers
    extract_array_ptrs(cir_coe, cir_coe_ptrs, "cir_coe");
    extract_array_ptrs(cir_norm_delay, cir_norm_delay_ptrs, "cir_norm_delay");
    extract_array_ptrs(cir_n_taps, cir_n_taps_ptrs, "cir_n_taps");
    extract_array_ptrs(cfr_sc, cfr_sc_ptrs, "cfr_sc");
    extract_array_ptrs(cfr_prbg, cfr_prbg_ptrs, "cfr_prbg");

    std::vector<nb::object> outputOwners;
    const auto retainOutputOwners = [&outputOwners](const nb::object& arrays) {
        if (arrays.is_none()) {
            return;
        }
        for (const auto item : nb::cast<nb::list>(arrays)) {
            outputOwners.push_back(nb::borrow<nb::object>(item));
        }
    };
    retainOutputOwners(cir_coe);
    retainOutputOwners(cir_norm_delay);
    retainOutputOwners(cir_n_taps);
    retainOutputOwners(cfr_sc);
    retainOutputOwners(cfr_prbg);

#ifdef SLS_DEBUG_
    if (m_cpuOnlyMode == 0) {
        // Debug GPU pointers only in GPU mode
        printf("DEBUG: Extracted GPU pointers from nanobind:\n");
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
#endif

    // Call the C++ method with vectors of pointers
    m_statisChanModelHandle->run(refTime, continuous_fading, activeCellVec, activeUtVec,
                               utNewLocVec, utNewVelocityVec, cir_coe_ptrs, cir_norm_delay_ptrs,
                               cir_n_taps_ptrs, cfr_sc_ptrs, cfr_prbg_ptrs);
    m_externalOutputOwners = std::move(outputOwners);
    m_lastActiveUt = std::move(activeUtVec);
}

template <typename Tscalar, typename Tcomplex>
void StatisChanModelWrapper<Tscalar, Tcomplex>::get_cir(
    nb::object cir_coe,
    nb::object cir_norm_delay,
    nb::object cir_n_taps)
{
    if (m_simConfig.internal_memory_mode < 1) {
        throw std::runtime_error("get_cir requires internal_memory_mode 1 or 2");
    }
    if (m_lastActiveUt.empty()) {
        throw std::runtime_error("get_cir requires a prior run with explicit active_ut");
    }

    const size_t snapshots = static_cast<size_t>(m_simConfig.n_snapshot_per_slot);
    const size_t ueAntennas = m_statisChanModelHandle->getNUeAnt();
    const size_t bsAntennas = m_statisChanModelHandle->getNBsAnt();
    const size_t maxTaps = m_statisChanModelHandle->getEffectiveMaxTaps();
    const auto linksForCell = [this](const size_t cellIdx) {
        return m_lastActiveUt[cellIdx].size();
    };
    const bool cpuOnlyMode = m_cpuOnlyMode != 0;

    copyInternalChannelBuffers(
        cir_coe, m_lastActiveUt, cpuOnlyMode, m_cuStrm,
        [this](const size_t cellIdx) {
            return m_statisChanModelHandle->getCirCoe(static_cast<uint32_t>(cellIdx));
        },
        [&](const size_t cellIdx) {
            size_t elements = checkedSizeProduct(
                linksForCell(cellIdx), snapshots, "cir_coe");
            elements = checkedSizeProduct(elements, ueAntennas, "cir_coe");
            elements = checkedSizeProduct(elements, bsAntennas, "cir_coe");
            elements = checkedSizeProduct(elements, maxTaps, "cir_coe");
            return checkedSizeProduct(elements, sizeof(Tcomplex), "cir_coe");
        },
        "cir_coe");
    copyInternalChannelBuffers(
        cir_norm_delay, m_lastActiveUt, cpuOnlyMode, m_cuStrm,
        [this](const size_t cellIdx) {
            return m_statisChanModelHandle->getCirIndex(static_cast<uint32_t>(cellIdx));
        },
        [&](const size_t cellIdx) {
            return checkedSizeProduct(
                checkedSizeProduct(linksForCell(cellIdx), maxTaps, "cir_norm_delay"),
                sizeof(uint16_t), "cir_norm_delay");
        },
        "cir_norm_delay");
    copyInternalChannelBuffers(
        cir_n_taps, m_lastActiveUt, cpuOnlyMode, m_cuStrm,
        [this](const size_t cellIdx) {
            return m_statisChanModelHandle->getCirNtaps(static_cast<uint32_t>(cellIdx));
        },
        [&](const size_t cellIdx) {
            return checkedSizeProduct(
                linksForCell(cellIdx), sizeof(uint16_t), "cir_n_taps");
        },
        "cir_n_taps");
}

template <typename Tscalar, typename Tcomplex>
void StatisChanModelWrapper<Tscalar, Tcomplex>::get_cfr(
    nb::object cfr_sc,
    nb::object cfr_prbg)
{
    if (m_simConfig.internal_memory_mode != 2) {
        throw std::runtime_error("get_cfr requires internal_memory_mode 2");
    }
    if (m_lastActiveUt.empty()) {
        throw std::runtime_error("get_cfr requires a prior run with explicit active_ut");
    }

    const size_t snapshots = static_cast<size_t>(m_simConfig.n_snapshot_per_slot);
    const size_t ueAntennas = m_statisChanModelHandle->getNUeAnt();
    const size_t bsAntennas = m_statisChanModelHandle->getNBsAnt();
    const auto linksForCell = [this](const size_t cellIdx) {
        return m_lastActiveUt[cellIdx].size();
    };
    const bool cpuOnlyMode = m_cpuOnlyMode != 0;

    if (!cfr_prbg.is_none()) {
        if (m_simConfig.run_mode != 1 && m_simConfig.run_mode != 3) {
            throw std::runtime_error("The last run did not produce PRBG CFR data");
        }
        copyInternalChannelBuffers(
            cfr_prbg, m_lastActiveUt, cpuOnlyMode, m_cuStrm,
            [this](const size_t cellIdx) {
                return m_statisChanModelHandle->getFreqChanPrbg(
                    static_cast<uint32_t>(cellIdx));
            },
            [&](const size_t cellIdx) {
                size_t elements = checkedSizeProduct(
                    linksForCell(cellIdx), snapshots, "cfr_prbg");
                elements = checkedSizeProduct(elements, ueAntennas, "cfr_prbg");
                elements = checkedSizeProduct(elements, bsAntennas, "cfr_prbg");
                elements = checkedSizeProduct(
                    elements, static_cast<size_t>(m_simConfig.n_prbg), "cfr_prbg");
                return checkedSizeProduct(elements, sizeof(Tcomplex), "cfr_prbg");
            },
            "cfr_prbg");
    }
    if (!cfr_sc.is_none()) {
        if (m_simConfig.run_mode != 2 && m_simConfig.run_mode != 3
            && m_simConfig.run_mode != 4) {
            throw std::runtime_error("The last run did not produce subcarrier CFR data");
        }
        const size_t subcarriers = m_simConfig.run_mode == 4
                                       ? static_cast<size_t>(m_simConfig.fft_size)
                                       : checkedSizeProduct(
                                             static_cast<size_t>(m_simConfig.n_prb),
                                             12, "cfr_sc");
        copyInternalChannelBuffers(
            cfr_sc, m_lastActiveUt, cpuOnlyMode, m_cuStrm,
            [this](const size_t cellIdx) {
                return m_statisChanModelHandle->getFreqChanSc(
                    static_cast<uint32_t>(cellIdx));
            },
            [&](const size_t cellIdx) {
                size_t elements = checkedSizeProduct(
                    linksForCell(cellIdx), snapshots, "cfr_sc");
                elements = checkedSizeProduct(elements, ueAntennas, "cfr_sc");
                elements = checkedSizeProduct(elements, bsAntennas, "cfr_sc");
                elements = checkedSizeProduct(elements, subcarriers, "cfr_sc");
                return checkedSizeProduct(elements, sizeof(Tcomplex), "cfr_sc");
            },
            "cfr_sc");
    }
}

template <typename Tscalar, typename Tcomplex>
void StatisChanModelWrapper<Tscalar, Tcomplex>::run_link_level(
    float refTime0,
    uint8_t continuous_fading,
    uint8_t /* enableSwapTxRx */,
    uint8_t /* txColumnMajorInd */) {

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
void StatisChanModelWrapper<Tscalar, Tcomplex>::dump_los_nlos_stats(std::optional<nb::ndarray<float, nb::c_contig, nb::device::cpu>> lost_nlos_stats) {
    float* stats_ptr = nullptr;
    if (lost_nlos_stats.has_value() && lost_nlos_stats->size() > 0) {
        stats_ptr = lost_nlos_stats->data();
    }
    m_statisChanModelHandle->dump_los_nlos_stats(stats_ptr);
}

template <typename Tscalar, typename Tcomplex>
void StatisChanModelWrapper<Tscalar, Tcomplex>::dump_pl_sf_stats(
    nb::ndarray<float, nb::c_contig, nb::device::cpu> pl_sf,
    std::optional<nb::ndarray<int, nb::c_contig, nb::device::cpu>> activeCell,
    std::optional<nb::ndarray<int, nb::c_contig, nb::device::cpu>> activeUt) {

    // pl_sf is required
    if (pl_sf.size() == 0) {
        throw std::invalid_argument("pl_sf array cannot be empty");
    }

    float* pl_sf_ptr = pl_sf.data();

    // Convert numpy arrays to vectors
    std::vector<uint16_t> activeCellVec;
    std::vector<uint16_t> activeUtVec;

    if (activeCell.has_value() && activeCell->size() > 0) {
        activeCellVec.reserve(activeCell->size());
        const int* cell_data = activeCell->data();
        for (size_t i = 0; i < activeCell->size(); ++i) {
            activeCellVec.push_back(checkedSelectorIndex(cell_data[i], "active_cell"));
        }
    }

    if (activeUt.has_value() && activeUt->size() > 0) {
        activeUtVec.reserve(activeUt->size());
        const int* ut_data = activeUt->data();
        for (size_t i = 0; i < activeUt->size(); ++i) {
            activeUtVec.push_back(checkedSelectorIndex(ut_data[i], "active_ut"));
        }
    }
    // Call the C++ method with vectors
    m_statisChanModelHandle->dump_pl_sf_stats(pl_sf_ptr, activeCellVec, activeUtVec);
}

template <typename Tscalar, typename Tcomplex>
void StatisChanModelWrapper<Tscalar, Tcomplex>::dump_pl_sf_ant_gain_stats(
    nb::ndarray<float, nb::c_contig, nb::device::cpu> pl_sf_ant_gain,
    std::optional<nb::ndarray<int, nb::c_contig, nb::device::cpu>> activeCell,
    std::optional<nb::ndarray<int, nb::c_contig, nb::device::cpu>> activeUt) {

    if (pl_sf_ant_gain.size() == 0) {
        throw std::invalid_argument("pl_sf_ant_gain array cannot be empty");
    }

    float* pl_sf_ant_gain_ptr = pl_sf_ant_gain.data();

    std::vector<uint16_t> activeCellVec;
    std::vector<uint16_t> activeUtVec;

    if (activeCell.has_value() && activeCell->size() > 0) {
        activeCellVec.reserve(activeCell->size());
        const int* cell_data = activeCell->data();
        for (size_t i = 0; i < activeCell->size(); ++i) {
            activeCellVec.push_back(checkedSelectorIndex(cell_data[i], "active_cell"));
        }
    }

    if (activeUt.has_value() && activeUt->size() > 0) {
        activeUtVec.reserve(activeUt->size());
        const int* ut_data = activeUt->data();
        for (size_t i = 0; i < activeUt->size(); ++i) {
            activeUtVec.push_back(checkedSelectorIndex(ut_data[i], "active_ut"));
        }
    }

    m_statisChanModelHandle->dump_pl_sf_ant_gain_stats(pl_sf_ant_gain_ptr, activeCellVec, activeUtVec);
}

template <typename Tscalar, typename Tcomplex>
void StatisChanModelWrapper<Tscalar, Tcomplex>::dump_topology_to_yaml(const std::string& filename) {
    m_statisChanModelHandle->dump_topology_to_yaml(filename);
}

template <typename Tscalar, typename Tcomplex>
void StatisChanModelWrapper<Tscalar, Tcomplex>::set_los_override(
    std::optional<nb::ndarray<uint8_t, nb::c_contig, nb::device::cpu>> los_ind)
{
    if (!los_ind.has_value()) {
        clear_los_override();
        return;
    }

    const uint32_t n_links =
        getStatisChanModelNumSiteUtLinks(m_statisChanModelHandle);
    if (los_ind->size() != n_links) {
        throw std::invalid_argument("los_ind must contain exactly " +
                                    std::to_string(n_links) + " entries");
    }
    if (!setStatisChanModelLosOverride(
            m_statisChanModelHandle, los_ind->data(), los_ind->size())) {
        throw std::invalid_argument(
            "los_ind entries must be 0 (NLOS), 1 (LOS), or 255 (model draw)");
    }
}

template <typename Tscalar, typename Tcomplex>
void StatisChanModelWrapper<Tscalar, Tcomplex>::clear_los_override()
{
    if (!setStatisChanModelLosOverride(m_statisChanModelHandle, nullptr, 0)) {
        throw std::runtime_error("failed to clear LOS/NLOS overrides");
    }
}

template <typename Tscalar, typename Tcomplex>
uint32_t StatisChanModelWrapper<Tscalar, Tcomplex>::get_num_site_ut_links() const
{
    return getStatisChanModelNumSiteUtLinks(m_statisChanModelHandle);
}

template <typename Tscalar, typename Tcomplex>
std::vector<LinkParams> StatisChanModelWrapper<Tscalar, Tcomplex>::get_link_params_host() const
{
    const uint32_t n_links = get_num_site_ut_links();
    if (n_links == 0) {
        return {};
    }

    const LinkParams* params =
        getStatisChanModelLinkParamsHost(m_statisChanModelHandle);
    if (params == nullptr) {
        throw std::runtime_error("link parameters are unavailable; call run() first");
    }
    return std::vector<LinkParams>(params, params + n_links);
}

template <typename Tscalar, typename Tcomplex>
std::vector<ClusterParams> StatisChanModelWrapper<Tscalar, Tcomplex>::get_cluster_params_host() const
{
    const uint32_t n_links = get_num_site_ut_links();
    if (n_links == 0) {
        return {};
    }

    const ClusterParams* params =
        getStatisChanModelClusterParamsHost(m_statisChanModelHandle);
    if (params == nullptr) {
        throw std::runtime_error("cluster parameters are unavailable; call run() first");
    }
    return std::vector<ClusterParams>(params, params + n_links);
}

template <typename Tscalar, typename Tcomplex>
void StatisChanModelWrapper<Tscalar, Tcomplex>::saveSlsChanToH5File(std::string_view filename_ending) {
    m_statisChanModelHandle->saveSlsChanToH5File(filename_ending);
}

template class OfdmModulateWrapper<float, cuComplex>;
template class OfdmDeModulateWrapper<float, cuComplex>;
template class TdlChanWrapper<float, cuComplex>;
template class CdlChanWrapper<float, cuComplex>;
template class GauNoiseAdderWrapper<float, cuComplex>;
template class StatisChanModelWrapper<float, cuComplex>;

}  // namespace gpu3gppchan_bindings
