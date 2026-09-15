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

#ifndef GPU3GPPCHAN_PYTHON_NANOBIND_GPU3GPPCHAN_BINDINGS_HPP
#define GPU3GPPCHAN_PYTHON_NANOBIND_GPU3GPPCHAN_BINDINGS_HPP

// Channel-model wrapper declarations exposed through nanobind.

#include <complex>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <cuda_runtime.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include "fading_chan.cuh"  // include the fading channel header
#include "cuda_array_interface.hpp"  // for cuda_array_t

// Add channel models includes
#include "gpu3gppchanApi.hpp"
#include "gpu3gppchanDataset.hpp"

namespace nb = nanobind;

namespace gpu3gppchan_bindings {

// Host-side complex sample buffer passed from NumPy (contiguous, on CPU).
template <typename Tscalar>
using HostComplexArray = nb::ndarray<std::complex<Tscalar>, nb::c_contig, nb::device::cpu>;

/*-------------------------------       OFDM modulation class       -------------------------------*/
/**
 * Python-facing OFDM modulator that owns its native handle and optional host input.
 *
 * @tparam Tscalar Real sample type.
 * @tparam Tcomplex CUDA complex sample type.
 */
template <typename Tscalar, typename Tcomplex>
class OfdmModulateWrapper{
public:
    /**
     * Construct a modulator using contiguous host-frequency input.
     *
     * @param[in] carrierParams Carrier configuration copied by the wrapper.
     * @param[in] freqDataInCpu Host-frequency input retained by the wrapper.
     * @param[in] streamHandle CUDA stream address.
     * @throws std::invalid_argument If the host input is too small.
     * @throws std::runtime_error If allocation or a CUDA operation fails.
     */
    OfdmModulateWrapper(cuphyCarrierPrms_t* carrierParams, HostComplexArray<Tscalar> freqDataInCpu, uintptr_t streamHandle);
    /**
     * Construct a modulator using caller-owned device-frequency input.
     *
     * @param[in] carrierParams Carrier configuration copied by the wrapper.
     * @param[in] freqDataInGpu Device-frequency input address.
     * @param[in] streamHandle CUDA stream address.
     * @throws std::runtime_error If allocation or a CUDA operation fails.
     */
    OfdmModulateWrapper(cuphyCarrierPrms_t* carrierParams, uintptr_t freqDataInGpu, uintptr_t streamHandle);
    /** Destroy the native modulator and its internally owned storage. */
    ~OfdmModulateWrapper();
    OfdmModulateWrapper(const OfdmModulateWrapper&) = delete;
    OfdmModulateWrapper& operator=(const OfdmModulateWrapper&) = delete;
    OfdmModulateWrapper(OfdmModulateWrapper&&) = delete;
    OfdmModulateWrapper& operator=(OfdmModulateWrapper&&) = delete;

    /**
     * Run modulation, optionally replacing the retained host input for this call.
     *
     * @param[in] freqDataInCpu Optional contiguous host-frequency input.
     * @param[in] enableSwapTxRx Whether to swap transmit and receive dimensions.
     * @throws std::invalid_argument If the replacement input size is invalid.
     * @throws std::runtime_error If a CUDA operation fails.
     */
    void run(std::optional<HostComplexArray<Tscalar>> freqDataInCpu = std::nullopt, uint8_t enableSwapTxRx = 0);
    /** Print a prefix of the generated time samples.
     * @param[in] printLen Maximum number of samples to print.
     */
    void printTimeSample(int printLen = 10){ m_ofdmModulateHandle -> printTimeSample(printLen); }
    /** Return the generated time-domain sample address.
     * @return Device address of the generated samples.
     */
    [[nodiscard]] uintptr_t getTimeDataOut(){ return reinterpret_cast<uintptr_t>(m_ofdmModulateHandle -> getTimeDataOut()); }
    /** Return an owner-bearing CUDA-array view of the generated time samples.
     * @return Read-only view whose Python object retains this modulator.
     */
    [[nodiscard]] cuda_array_t<std::complex<Tscalar>> getTimeDataOutArray() {
        return cuda_array_t<std::complex<Tscalar>>(
            static_cast<intptr_t>(getTimeDataOut()),
            {static_cast<size_t>(getTimeDataLen())}, {}, true);
    }
    /** Return the generated time-domain sample count.
     * @return Number of generated samples.
     */
    [[nodiscard]] uint32_t getTimeDataLen(){ return m_ofdmModulateHandle -> getTimeDataLen(); }
    /** Return the generated length of each OFDM symbol.
     * @return Per-symbol sample counts including cyclic prefixes.
     */
    [[nodiscard]] std::vector<uint32_t> getEachSymbolLenWithCP(){ return m_ofdmModulateHandle -> getEachSymbolLenWithCP(); }

private:
    cuphyCarrierPrms_t m_carrierParams;
    std::optional<HostComplexArray<Tscalar>> m_freqDataInCpuOwner;
    std::optional<HostComplexArray<Tscalar>> m_freqDataInRunOwner;
    ofdm_modulate::ofdmModulate<Tscalar, Tcomplex> * m_ofdmModulateHandle;
    cudaStream_t m_cuStrm;
    size_t m_freqDataInSizeDl, m_freqDataInSizeUl;
    Tcomplex* m_freqDataInCpu;
    Tcomplex* m_freqDataInGpu;
    uint8_t m_externGpuAlloc; // indicator for freqDataIn storage type: 0 - internal GPU memory allocation; 1 - external GPU memory allocation
};
extern template class OfdmModulateWrapper<float, cuComplex>;

/*-------------------------------       OFDM demodulation class       -------------------------------*/
/**
 * Python-facing OFDM demodulator that owns its native handle and optional host output.
 *
 * @tparam Tscalar Real sample type.
 * @tparam Tcomplex CUDA complex sample type.
 */
template <typename Tscalar, typename Tcomplex>
class OfdmDeModulateWrapper{
public:
    /**
     * Construct a demodulator that copies frequency output to host memory.
     *
     * @param[in] carrierParams Carrier configuration copied by the wrapper.
     * @param[in] timeDataInGpu Device time-domain input address.
     * @param[out] freqDataOutCpu Host-frequency output retained by the wrapper.
     * @param[in] prach Whether to use PRACH demodulation.
     * @param[in] perAntSamp Whether input samples are per antenna pair.
     * @param[in] streamHandle CUDA stream address.
     * @throws std::invalid_argument If the host output is too small.
     * @throws std::runtime_error If allocation or a CUDA operation fails.
     */
    OfdmDeModulateWrapper(cuphyCarrierPrms_t * carrierParams, uintptr_t timeDataInGpu, HostComplexArray<Tscalar> freqDataOutCpu, bool prach, bool perAntSamp, uintptr_t streamHandle);
    /**
     * Construct a demodulator that writes frequency output to device memory.
     *
     * @param[in] carrierParams Carrier configuration copied by the wrapper.
     * @param[in] timeDataInGpu Device time-domain input address.
     * @param[out] freqDataOutGpu Device-frequency output address.
     * @param[in] prach Whether to use PRACH demodulation.
     * @param[in] perAntSamp Whether input samples are per antenna pair.
     * @param[in] streamHandle CUDA stream address.
     * @throws std::runtime_error If allocation or a CUDA operation fails.
     */
    OfdmDeModulateWrapper(cuphyCarrierPrms_t * carrierParams, uintptr_t timeDataInGpu, uintptr_t freqDataOutGpu, bool prach, bool perAntSamp, uintptr_t streamHandle);
    /** Destroy the native demodulator and its internally owned storage. */
    ~OfdmDeModulateWrapper();
    OfdmDeModulateWrapper(const OfdmDeModulateWrapper&) = delete;
    OfdmDeModulateWrapper& operator=(const OfdmDeModulateWrapper&) = delete;
    OfdmDeModulateWrapper(OfdmDeModulateWrapper&&) = delete;
    OfdmDeModulateWrapper& operator=(OfdmDeModulateWrapper&&) = delete;

    /**
     * Run demodulation, optionally replacing the retained host output for this call.
     *
     * @param[out] freqDataOutCpu Optional contiguous host-frequency output.
     * @param[in] enableSwapTxRx Whether to swap transmit and receive dimensions.
     * @throws std::invalid_argument If the replacement output size is invalid.
     * @throws std::runtime_error If a CUDA operation fails.
     */
    void run(std::optional<HostComplexArray<Tscalar>> freqDataOutCpu = std::nullopt, uint8_t enableSwapTxRx = 0);
    /** Print a prefix of the generated frequency samples.
     * @param[in] printLen Maximum number of samples to print.
     */
    void printFreqSample(int printLen = 10){ m_ofdmDeModulateHandle -> printFreqSample(printLen); }
    /** Return the generated frequency-domain sample address.
     * @return Device address of the generated samples.
     */
    [[nodiscard]] uintptr_t getFreqDataOut(){return reinterpret_cast<uintptr_t>(m_ofdmDeModulateHandle -> getFreqDataOut()); }

private:
    cuphyCarrierPrms_t m_carrierParams;
    std::optional<HostComplexArray<Tscalar>> m_freqDataOutCpuOwner;
    std::optional<HostComplexArray<Tscalar>> m_freqDataOutRunOwner;
    ofdm_demodulate::ofdmDeModulate<Tscalar, Tcomplex> * m_ofdmDeModulateHandle;
    cudaStream_t m_cuStrm;
    size_t m_freqDataOutSizeDl, m_freqDataOutSizeUl;
    Tcomplex *m_freqDataOutCpu;
    Tcomplex *m_freqDataOutGpu;
    uint8_t m_externGpuAlloc; // indicator for freqDataOut storage type: 0 - internal GPU memory allocation; 1 - external GPU memory allocation
    bool m_perAntSamp; // true: input sample is per rx-tx antenna pair; false: input sample is per rx antenna
};
extern template class OfdmDeModulateWrapper<float, cuComplex>;

/*-------------------------------       TDL channel class       -------------------------------*/
/**
 * Python-facing tapped-delay-line channel wrapper.
 *
 * @tparam Tscalar Real sample type.
 * @tparam Tcomplex CUDA complex sample type.
 */
template <typename Tscalar, typename Tcomplex>
class TdlChanWrapper{
public:
    /**
     * Construct a TDL channel.
     *
     * @param[in] tdlCfg TDL configuration copied by the wrapper.
     * @param[in] randSeed Random seed.
     * @param[in] streamHandle CUDA stream address.
     * @throws std::runtime_error If configuration, allocation, or CUDA setup fails.
     */
    TdlChanWrapper(tdlConfig_t* tdlCfg, uint16_t randSeed, uintptr_t streamHandle);
    /** Destroy the native TDL channel. */
    ~TdlChanWrapper();
    TdlChanWrapper(const TdlChanWrapper&) = delete;
    TdlChanWrapper& operator=(const TdlChanWrapper&) = delete;
    TdlChanWrapper(TdlChanWrapper&&) = delete;
    TdlChanWrapper& operator=(TdlChanWrapper&&) = delete;

    /** Reset the channel state. */
    void reset(){ m_tdlChanHandle -> reset(); }
    /**
     * Run the TDL channel.
     *
     * @param[in] txSigIn CUDA-array view of transmit samples.
     * @param[in] refTime0 Reference time in seconds.
     * @param[in] enableSwapTxRx Whether to swap transmit and receive dimensions.
     * @param[in] txColumnMajorInd Whether transmit samples use column-major ordering.
     * @throws std::runtime_error If the input or a CUDA operation is invalid.
     */
    void run(const cuda_array_t<std::complex<Tscalar>>& txSigIn, float refTime0 = 0.0f, uint8_t enableSwapTxRx = 0, uint8_t txColumnMajorInd = 0);
    /** Return a CUDA-array view of receive samples.
     * @param[in] enableSwapTxRx Whether transmit and receive dimensions were swapped.
     * @return Non-owning view that keeps the Python channel object alive.
     * @throws std::runtime_error If native output storage or dimensions are invalid.
     */
    [[nodiscard]] cuda_array_t<std::complex<Tscalar>> getRxSignalOutArray(uint8_t enableSwapTxRx);
    /** Return the time-domain channel response address.
     * @return Device address of the response.
     */
    [[nodiscard]] uintptr_t getTimeChan(){ return reinterpret_cast<uintptr_t>(m_tdlChanHandle -> getTimeChan()); }
    /** Return the subcarrier channel response address.
     * @return Device address of the response.
     */
    [[nodiscard]] uintptr_t getFreqChanSc(){ return reinterpret_cast<uintptr_t>(m_tdlChanHandle -> getFreqChanSc()); }
    /** Return the PRBG channel response address.
     * @return Device address of the response.
     */
    [[nodiscard]] uintptr_t getFreqChanPrbg(){ return reinterpret_cast<uintptr_t>(m_tdlChanHandle -> getFreqChanPrbg()); }
    /** Return the receive-sample address.
     * @return Device address of receive samples.
     */
    [[nodiscard]] uintptr_t getRxSigOut(){ return reinterpret_cast<uintptr_t>(m_tdlChanHandle -> getRxSigOut()); }
    /** Return the antenna-pair receive-sample address.
     * @return Device address of receive samples arranged by antenna pair.
     */
    [[nodiscard]] uintptr_t getRxTimeAntPairSigOut() { return reinterpret_cast<uintptr_t>(m_tdlChanHandle -> getRxTimeAntPairSigOut()); }
    /** Return the time-domain channel size.
     * @return Number of channel elements.
     */
    [[nodiscard]] uint32_t getTimeChanSize(){ return m_tdlChanHandle -> getTimeChanSize(); }
    /** Return the per-link subcarrier channel size.
     * @return Number of channel elements per link.
     */
    [[nodiscard]] uint32_t getFreqChanScPerLinkSize(){ return m_tdlChanHandle -> getFreqChanScPerLinkSize();}
    /** Return the total PRBG channel size.
     * @return Number of PRBG channel elements.
     */
    [[nodiscard]] uint32_t getFreqChanPrbgSize(){ return m_tdlChanHandle -> getFreqChanPrbgSize(); }
    /** Print time-domain channel samples.
     * @param[in] cid Cell index.
     * @param[in] uid UT index.
     * @param[in] printLen Maximum number of samples to print.
     */
    void printTimeChan(uint16_t cid = 0, uint16_t uid = 0, int printLen = 10){ m_tdlChanHandle -> printTimeChan(cid, uid, printLen); }
    /** Print subcarrier channel samples.
     * @param[in] cid Cell index.
     * @param[in] uid UT index.
     * @param[in] printLen Maximum number of samples to print.
     */
    void printFreqScChan(uint16_t cid = 0, uint16_t uid = 0, int printLen = 10){ m_tdlChanHandle -> printFreqScChan(cid, uid, printLen); }
    /** Print PRBG channel samples.
     * @param[in] cid Cell index.
     * @param[in] uid UT index.
     * @param[in] printLen Maximum number of samples to print.
     */
    void printFreqPrbgChan(uint16_t cid = 0, uint16_t uid = 0, int printLen = 10){ m_tdlChanHandle -> printFreqPrbgChan(cid, uid, printLen); }
    /** Print receive samples.
     * @param[in] cid Cell index.
     * @param[in] uid UT index.
     * @param[in] printLen Maximum number of samples to print.
     */
    void printSig(uint16_t cid = 0, uint16_t uid = 0, int printLen = 10){ m_tdlChanHandle -> printSig(cid, uid, printLen); }
    /** Print native GPU-memory usage. */
    void printGpuMemUseMB(){ m_tdlChanHandle -> printGpuMemUseMB(); }
    /** Copy the CIR into `cir` and synchronize before returning.
     * @param[out] cir Contiguous host array sized by `getTimeChanSize()`.
     * @throws std::invalid_argument If the array size is invalid.
     * @throws std::runtime_error If the CUDA transfer fails.
     */
    void dumpCir(HostComplexArray<Tscalar> cir);
    /** Copy PRBG CFR into `cfrPrbg` and synchronize before returning.
     * @param[out] cfrPrbg Contiguous host array sized by `getFreqChanPrbgSize()`.
     * @throws std::invalid_argument If the array size is invalid.
     * @throws std::runtime_error If the CUDA transfer fails.
     */
    void dumpCfrPrbg(HostComplexArray<Tscalar> cfrPrbg);
    /** Copy subcarrier CFR into `cfrSc` and synchronize before returning.
     * @param[out] cfrSc Contiguous host array containing all links.
     * @throws std::invalid_argument If the array size is invalid.
     * @throws std::runtime_error If the CUDA transfer fails.
     */
    void dumpCfrSc(HostComplexArray<Tscalar> cfrSc);

    /**
     * Save TDL data to HDF5 for MATLAB verification.
     *
     * @param[in] padFileNameEnding Optional filename suffix.
     */
    void saveTdlChanToH5File(const std::string& padFileNameEnding = "") {
        m_tdlChanHandle->saveTdlChanToH5File(padFileNameEnding);
    }

private:
    tdlChan<Tscalar, Tcomplex> * m_tdlChanHandle;
    cudaStream_t m_cuStrm;
    size_t m_nLink;
    uint8_t m_runMode;
    tdlConfig_t m_tdlCfg;
};
extern template class TdlChanWrapper<float, cuComplex>;

/*-------------------------------       CDL channel class       -------------------------------*/
/**
 * Python-facing clustered-delay-line channel wrapper.
 *
 * @tparam Tscalar Real sample type.
 * @tparam Tcomplex CUDA complex sample type.
 */
template <typename Tscalar, typename Tcomplex>
class CdlChanWrapper{
public:
    /**
     * Construct a CDL channel.
     *
     * @param[in] cdlCfg CDL configuration copied by the wrapper.
     * @param[in] randSeed Random seed.
     * @param[in] streamHandle CUDA stream address.
     * @throws std::invalid_argument If an antenna product exceeds native capacity.
     * @throws std::runtime_error If configuration, allocation, or CUDA setup fails.
     */
    CdlChanWrapper(cdlConfig_t* cdlCfg, uint16_t randSeed, uintptr_t streamHandle);
    /** Destroy the native CDL channel. */
    ~CdlChanWrapper();
    CdlChanWrapper(const CdlChanWrapper&) = delete;
    CdlChanWrapper& operator=(const CdlChanWrapper&) = delete;
    CdlChanWrapper(CdlChanWrapper&&) = delete;
    CdlChanWrapper& operator=(CdlChanWrapper&&) = delete;

    /** Reset the channel state. */
    void reset(){ m_cdlChanHandle -> reset(); }
    /**
     * Run the CDL channel.
     *
     * @param[in] txSigIn CUDA-array view of transmit samples.
     * @param[in] refTime0 Reference time in seconds.
     * @param[in] enableSwapTxRx Whether to swap transmit and receive dimensions.
     * @param[in] txColumnMajorInd Whether transmit samples use column-major ordering.
     * @throws std::runtime_error If the input or a CUDA operation is invalid.
     */
    void run(const cuda_array_t<std::complex<Tscalar>>& txSigIn, float refTime0 = 0.0f, uint8_t enableSwapTxRx = 0, uint8_t txColumnMajorInd = 0);
    /** Return a CUDA-array view of receive samples.
     * @param[in] enableSwapTxRx Whether transmit and receive dimensions were swapped.
     * @return Non-owning view that keeps the Python channel object alive.
     * @throws std::runtime_error If native output storage or dimensions are invalid.
     */
    [[nodiscard]] cuda_array_t<std::complex<Tscalar>> getRxSignalOutArray(uint8_t enableSwapTxRx);
    /** Return the time-domain channel response address.
     * @return Device address of the response.
     */
    [[nodiscard]] uintptr_t getTimeChan(){ return reinterpret_cast<uintptr_t>(m_cdlChanHandle -> getTimeChan()); }
    /** Return the subcarrier channel response address.
     * @return Device address of the response.
     */
    [[nodiscard]] uintptr_t getFreqChanSc(){ return reinterpret_cast<uintptr_t>(m_cdlChanHandle -> getFreqChanSc()); }
    /** Return the PRBG channel response address.
     * @return Device address of the response.
     */
    [[nodiscard]] uintptr_t getFreqChanPrbg(){ return reinterpret_cast<uintptr_t>(m_cdlChanHandle -> getFreqChanPrbg()); }
    /** Return the receive-sample address.
     * @return Device address of receive samples.
     */
    [[nodiscard]] uintptr_t getRxSigOut(){ return reinterpret_cast<uintptr_t>(m_cdlChanHandle -> getRxSigOut()); }
    /** Return the antenna-pair receive-sample address.
     * @return Device address of receive samples arranged by antenna pair.
     */
    [[nodiscard]] uintptr_t getRxTimeAntPairSigOut() { return reinterpret_cast<uintptr_t>(m_cdlChanHandle -> getRxTimeAntPairSigOut()); }
    /** Return the time-domain channel size.
     * @return Number of channel elements.
     */
    [[nodiscard]] uint32_t getTimeChanSize(){ return m_cdlChanHandle -> getTimeChanSize(); }
    /** Return the per-link subcarrier channel size.
     * @return Number of channel elements per link.
     */
    [[nodiscard]] uint32_t getFreqChanScPerLinkSize(){ return m_cdlChanHandle -> getFreqChanScPerLinkSize();}
    /** Return the total PRBG channel size.
     * @return Number of PRBG channel elements.
     */
    [[nodiscard]] uint32_t getFreqChanPrbgSize(){ return m_cdlChanHandle -> getFreqChanPrbgSize(); }
    /** Print time-domain channel samples.
     * @param[in] cid Cell index.
     * @param[in] uid UT index.
     * @param[in] printLen Maximum number of samples to print.
     */
    void printTimeChan(uint16_t cid = 0, uint16_t uid = 0, int printLen = 10){ m_cdlChanHandle -> printTimeChan(cid, uid, printLen); }
    /** Print subcarrier channel samples.
     * @param[in] cid Cell index.
     * @param[in] uid UT index.
     * @param[in] printLen Maximum number of samples to print.
     */
    void printFreqScChan(uint16_t cid = 0, uint16_t uid = 0, int printLen = 10){ m_cdlChanHandle -> printFreqScChan(cid, uid, printLen); }
    /** Print PRBG channel samples.
     * @param[in] cid Cell index.
     * @param[in] uid UT index.
     * @param[in] printLen Maximum number of samples to print.
     */
    void printFreqPrbgChan(uint16_t cid = 0, uint16_t uid = 0, int printLen = 10){ m_cdlChanHandle -> printFreqPrbgChan(cid, uid, printLen); }
    /** Print receive samples.
     * @param[in] cid Cell index.
     * @param[in] uid UT index.
     * @param[in] printLen Maximum number of samples to print.
     */
    void printSig(uint16_t cid = 0, uint16_t uid = 0, int printLen = 10){ m_cdlChanHandle -> printSig(cid, uid, printLen); }
    /** Print native GPU-memory usage. */
    void printGpuMemUseMB(){ m_cdlChanHandle -> printGpuMemUseMB(); }
    /** Copy the CIR into `cir` and synchronize before returning.
     * @param[out] cir Contiguous host array sized by `getTimeChanSize()`.
     * @throws std::invalid_argument If the array size is invalid.
     * @throws std::runtime_error If the CUDA transfer fails.
     */
    void dumpCir(HostComplexArray<Tscalar> cir);
    /** Copy PRBG CFR into `cfrPrbg` and synchronize before returning.
     * @param[out] cfrPrbg Contiguous host array sized by `getFreqChanPrbgSize()`.
     * @throws std::invalid_argument If the array size is invalid.
     * @throws std::runtime_error If the CUDA transfer fails.
     */
    void dumpCfrPrbg(HostComplexArray<Tscalar> cfrPrbg);
    /** Copy subcarrier CFR into `cfrSc` and synchronize before returning.
     * @param[out] cfrSc Contiguous host array containing all links.
     * @throws std::invalid_argument If the array size is invalid.
     * @throws std::runtime_error If the CUDA transfer fails.
     */
    void dumpCfrSc(HostComplexArray<Tscalar> cfrSc);

    /**
     * Save CDL data to HDF5 for MATLAB verification.
     *
     * @param[in] padFileNameEnding Optional filename suffix.
     */
    void saveCdlChanToH5File(const std::string& padFileNameEnding = "") {
        m_cdlChanHandle->saveCdlChanToH5File(padFileNameEnding);
    }

private:
    cdlChan<Tscalar, Tcomplex> * m_cdlChanHandle;
    cudaStream_t m_cuStrm;
    size_t m_nLink;
    uint16_t m_nBsAnt, m_nUeAnt;
    uint8_t m_runMode;
    cdlConfig_t m_cdlCfg;
};
extern template class CdlChanWrapper<float, cuComplex>;

/*-------------------------------       add Gaussian noise class       -------------------------------*/
/**
 * Python-facing Gaussian-noise adder.
 *
 * @tparam Tscalar Real sample type.
 * @tparam Tcomplex CUDA complex sample type.
 */
template <typename Tscalar, typename Tcomplex>
class GauNoiseAdderWrapper{
public:
    /** Construct a Gaussian-noise generator.
     * @param[in] nThreads Number of CUDA threads used by the native helper.
     * @param[in] seed Random seed.
     * @param[in] streamHandle CUDA stream address.
     * @throws std::runtime_error If native setup fails.
     */
    GauNoiseAdderWrapper(uint32_t nThreads, int seed, uintptr_t streamHandle);
    /** Add noise in place to a device signal.
     * @param[in,out] d_signal Device signal address.
     * @param[in] signalSize Number of complex samples.
     * @param[in] snr_db Target signal-to-noise ratio in decibels.
     * @throws std::runtime_error If the native CUDA operation fails.
     */
    void addNoise(uintptr_t d_signal, uint32_t signalSize, float snr_db);
    /** Destroy the native Gaussian-noise generator. */
    ~GauNoiseAdderWrapper();
    GauNoiseAdderWrapper(const GauNoiseAdderWrapper&) = delete;
    GauNoiseAdderWrapper& operator=(const GauNoiseAdderWrapper&) = delete;
    GauNoiseAdderWrapper(GauNoiseAdderWrapper&&) = delete;
    GauNoiseAdderWrapper& operator=(GauNoiseAdderWrapper&&) = delete;

private:
    GauNoiseAdder<Tcomplex> * m_gauNoiseAdder;
    cudaStream_t m_cuStrm;
};
extern template class GauNoiseAdderWrapper<float, cuComplex>;

/*-------------------------------       Stochastic Channel Model class       -------------------------------*/
/**
 * Python-facing statistical channel-model wrapper.
 *
 * Owns copied configuration and the native statistical model handle.
 *
 * @tparam Tscalar Real sample type.
 * @tparam Tcomplex CUDA complex sample type.
 */
template <typename Tscalar, typename Tcomplex>
class StatisChanModelWrapper{
public:
    /** Construct a statistical model with explicit link and external configuration.
     * @param[in] sim_config Simulation configuration.
     * @param[in] system_level_config System-level topology configuration.
     * @param[in] link_level_config Link-level fading configuration.
     * @param[in] external_config Explicit cells, UTs, panels, and sensing targets.
     * @param[in] randSeed Random seed.
     * @param[in] streamHandle CUDA stream address.
     * @throws std::runtime_error If model construction fails.
     */
    StatisChanModelWrapper(const SimConfig& sim_config,
                      const SystemLevelConfig& system_level_config,
                      const LinkLevelConfig& link_level_config,
                      const ExternalConfig& external_config,
                      uint32_t randSeed,
                      uintptr_t streamHandle);

    /** Construct a statistical model using generated topology defaults.
     * @param[in] sim_config Simulation configuration.
     * @param[in] system_level_config System-level topology configuration.
     * @param[in] randSeed Random seed.
     * @param[in] streamHandle CUDA stream address.
     * @throws std::runtime_error If model construction fails.
     */
    StatisChanModelWrapper(const SimConfig& sim_config,
                      const SystemLevelConfig& system_level_config,
                      uint32_t randSeed,
                      uintptr_t streamHandle);

    /** Statistical wrappers are noncopyable because they uniquely own a native model. */
    StatisChanModelWrapper(const StatisChanModelWrapper&) = delete;
    /** Statistical wrappers are noncopyable because they uniquely own a native model. */
    StatisChanModelWrapper& operator=(const StatisChanModelWrapper&) = delete;

    /** Statistical wrappers are nonmovable because nanobind retains their address. */
    StatisChanModelWrapper(StatisChanModelWrapper&&) = delete;
    /** Statistical wrappers are nonmovable because nanobind retains their address. */
    StatisChanModelWrapper& operator=(StatisChanModelWrapper&&) = delete;

    /** Destroy the native statistical model. */
    ~StatisChanModelWrapper();

    /** Reset native channel state without changing configuration. */
    void reset() { m_statisChanModelHandle->reset(); }

    /** Run one system-level statistical-channel update.
     * @param[in] refTime Reference time in seconds.
     * @param[in] continuous_fading Whether to preserve fading continuity.
     * @param[in] activeCell Optional active-cell indices.
     * @param[in] activeUt Optional active-UT indices.
     * @param[in] utNewLoc Optional updated UT coordinates.
     * @param[in] utNewVelocity Optional updated UT velocities.
     * @param[out] cir_coe Optional CIR coefficient CUDA array.
     * @param[out] cir_norm_delay Optional normalized-delay CUDA array.
     * @param[out] cir_n_taps Optional tap-count CUDA array.
     * @param[out] cfr_sc Optional subcarrier CFR CUDA array.
     * @param[out] cfr_prbg Optional PRBG CFR CUDA array.
     * @throws nanobind::type_error If a buffer does not expose the required interface.
     * @throws nanobind::value_error If a supplied buffer has a null data pointer.
     * @throws std::invalid_argument If CUDA-array metadata is invalid.
     * @throws std::runtime_error If array metadata, topology, or native execution fails.
     */
    void run(float refTime = 0.0f,
             uint8_t continuous_fading = 1,
             nb::object activeCell = nb::none(),
             nb::object activeUt = nb::none(),
             nb::object utNewLoc = nb::none(),
             nb::object utNewVelocity = nb::none(),
             nb::object cir_coe = nb::none(),
             nb::object cir_norm_delay = nb::none(),
             nb::object cir_n_taps = nb::none(),
             nb::object cfr_sc = nb::none(),
             nb::object cfr_prbg = nb::none());

    /** Run one link-level statistical-channel update.
     * @param[in] refTime0 Reference time in seconds.
     * @param[in] continuous_fading Whether to preserve fading continuity.
     * @param[in] enableSwapTxRx Whether to swap transmit and receive dimensions.
     * @param[in] txColumnMajorInd Whether transmit samples use column-major ordering.
     * @throws std::runtime_error If native execution fails.
     */
    void run_link_level(float refTime0 = 0.0f,
                       uint8_t continuous_fading = 1,
                       uint8_t enableSwapTxRx = 0,
                       uint8_t txColumnMajorInd = 0);

    /** Copy internally stored CIR data into caller-owned output buffers.
     * @throws std::runtime_error If no compatible internal CIR is available.
     * @throws std::invalid_argument If output buffers do not match the last run.
     */
    void get_cir(nb::object cir_coe = nb::none(),
                 nb::object cir_norm_delay = nb::none(),
                 nb::object cir_n_taps = nb::none());

    /** Copy internally stored CFR data into caller-owned output buffers.
     * @throws std::runtime_error If no compatible internal CFR is available.
     * @throws std::invalid_argument If output buffers do not match the last run.
     */
    void get_cfr(nb::object cfr_sc = nb::none(),
                 nb::object cfr_prbg = nb::none());

    /** Copy LOS/NLOS counts into an optional caller-provided host array.
     * @param[out] lost_nlos_stats Optional contiguous output array.
     */
    void dump_los_nlos_stats(std::optional<nb::ndarray<float, nb::c_contig, nb::device::cpu>> lost_nlos_stats = std::nullopt);
    /** Copy pathloss and shadow-fading statistics.
     * @param[out] pl_sf Contiguous output array.
     * @param[in] activeCell Optional active-cell indices.
     * @param[in] activeUt Optional active-UT indices.
     * @throws std::invalid_argument If the output array is empty.
     */
    void dump_pl_sf_stats(nb::ndarray<float, nb::c_contig, nb::device::cpu> pl_sf,
                          std::optional<nb::ndarray<int, nb::c_contig, nb::device::cpu>> activeCell = std::nullopt,
                          std::optional<nb::ndarray<int, nb::c_contig, nb::device::cpu>> activeUt = std::nullopt);
    /** Copy pathloss, shadow-fading, and antenna-gain statistics.
     * @param[out] pl_sf_ant_gain Contiguous output array.
     * @param[in] activeCell Optional active-cell indices.
     * @param[in] activeUt Optional active-UT indices.
     * @throws std::invalid_argument If the output array is empty.
     */
    void dump_pl_sf_ant_gain_stats(nb::ndarray<float, nb::c_contig, nb::device::cpu> pl_sf_ant_gain,
                          std::optional<nb::ndarray<int, nb::c_contig, nb::device::cpu>> activeCell = std::nullopt,
                          std::optional<nb::ndarray<int, nb::c_contig, nb::device::cpu>> activeUt = std::nullopt);
    /** Save current topology to YAML.
     * @param[in] filename Output path.
     * @throws std::runtime_error If the native export fails.
     */
    void dump_topology_to_yaml(const std::string& filename);

    /** Set per-link LOS/NLOS overrides for the next system-level run.
     * @param[in] los_ind One value per link: 0=NLOS, 1=LOS, 255=model draw;
     *     `None` clears all overrides.
     * @throws std::invalid_argument If values or link count are invalid.
     * @throws std::runtime_error If the native model rejects the override.
     */
    void set_los_override(
        std::optional<nb::ndarray<uint8_t, nb::c_contig, nb::device::cpu>> los_ind);
    /** Clear all per-link LOS/NLOS overrides.
     * @throws std::runtime_error If the native model rejects the clear request.
     */
    void clear_los_override();

    /** Return the number of site-UT links represented by model snapshots.
     * @return Number of configured site-UT links.
     */
    [[nodiscard]] uint32_t get_num_site_ut_links() const;
    /** Return copies of per-link large-scale parameters.
     * @return Snapshot vector, valid after `run()`.
     * @throws std::runtime_error If the model has links but no snapshot storage.
     */
    [[nodiscard]] std::vector<LinkParams> get_link_params_host() const;
    /** Return copies of per-link cluster parameters.
     * @return Snapshot vector, valid after `run()`.
     * @throws std::runtime_error If the model has links but no snapshot storage.
     */
    [[nodiscard]] std::vector<ClusterParams> get_cluster_params_host() const;

    /**
     * Save SLS channel data to H5 file for debugging
     *
     * @param[in] filename_ending Optional string to append to filename.
     */
    void saveSlsChanToH5File(std::string_view filename_ending = "");

private:
    SimConfig m_simConfig;
    SystemLevelConfig m_systemLevelConfig;
    std::optional<LinkLevelConfig> m_linkLevelConfig;
    std::optional<ExternalConfig> m_externalConfig;
    statisChanModel<Tscalar, Tcomplex> * m_statisChanModelHandle;
    cudaStream_t m_cuStrm;
    uint32_t m_randSeed;
    int m_cpuOnlyMode;
    std::vector<std::vector<uint16_t>> m_lastActiveUt;
    std::vector<nb::object> m_externalOutputOwners;
};
extern template class StatisChanModelWrapper<float, cuComplex>;
} // namespace gpu3gppchan_bindings

#endif // GPU3GPPCHAN_PYTHON_NANOBIND_GPU3GPPCHAN_BINDINGS_HPP
