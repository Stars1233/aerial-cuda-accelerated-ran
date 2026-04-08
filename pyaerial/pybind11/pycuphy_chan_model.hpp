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

#ifndef PYCUPHY_CHAN_MODEL_HPP
#define PYCUPHY_CHAN_MODEL_HPP

#include <optional>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <complex>
#include <cuda_runtime.h>
#include "fading_chan.cuh"  // include the fading channel header
#include "cuda_array_interface.hpp"  // for cuda_array_t

// Add channel models includes
#include "chanModelsApi.hpp"
#include "chanModelsDataset.hpp"

namespace py = pybind11;

namespace pycuphy {
/*-------------------------------       OFDM modulation class       -------------------------------*/
template <typename Tscalar, typename Tcomplex>
class OfdmModulateWrapper{
public:
    OfdmModulateWrapper(cuphyCarrierPrms_t* cuphyCarrierPrms, py::array_t<std::complex<Tscalar>> freqDataInCpu, uintptr_t streamHandle); // constructor using Python array on CPU
    OfdmModulateWrapper(cuphyCarrierPrms_t* cuphyCarrierPrms, uintptr_t freqDataInGpu, uintptr_t streamHandle);  // constructor using GPU memory pointer
    ~OfdmModulateWrapper();

    void run(py::array_t<std::complex<Tscalar>> freqDataInCpu = py::none(), uint8_t enableSwapTxRx = 0); // run function to enable changing input numpy array in run; if not provided, use the memory address in initialization
    void printTimeSample(int printLen = 10){ m_ofdmModulateHandle -> printTimeSample(printLen); }
    uintptr_t getTimeDataOut(){ return reinterpret_cast<uintptr_t>(m_ofdmModulateHandle -> getTimeDataOut()); }
    uint32_t getTimeDataLen(){ return m_ofdmModulateHandle -> getTimeDataLen(); }
    std::vector<uint32_t> getEachSymbolLenWithCP(){ return m_ofdmModulateHandle -> getEachSymbolLenWithCP(); }

private:
    ofdm_modulate::ofdmModulate<Tscalar, Tcomplex> * m_ofdmModulateHandle;
    cudaStream_t m_cuStrm;
    size_t m_freqDataInSizeDl, m_freqDataInSizeUl;
    Tcomplex* m_freqDataInCpu;
    Tcomplex* m_freqDataInGpu;
    uint8_t m_externGpuAlloc; // indicator for freqDataIn storage type: 0 - internal GPU memory allocation; 1 - external GPU memory allocation
};
// explicit instantiation
template class OfdmModulateWrapper<float, cuComplex>;

/*-------------------------------       OFDM demodulation class       -------------------------------*/
template <typename Tscalar, typename Tcomplex>
class OfdmDeModulateWrapper{
public:
    OfdmDeModulateWrapper(cuphyCarrierPrms_t * cuphyCarrierPrms, uintptr_t timeDataInGpu, py::array_t<std::complex<Tscalar>> freqDataOutCpu, bool prach, bool perAntSamp, uintptr_t streamHandle); // constructor using Python array on CPU, timeDataInGpu is always GPU memory address
    OfdmDeModulateWrapper(cuphyCarrierPrms_t * cuphyCarrierPrms, uintptr_t timeDataInGpu, uintptr_t freqDataOutGpu, bool prach, bool perAntSamp, uintptr_t streamHandle); // constructor using GPU memory pointer, timeDataInGpu is always GPU memory address
    ~OfdmDeModulateWrapper();

    void run(py::array_t<std::complex<Tscalar>> freqDataOutCpu = py::none(), uint8_t enableSwapTxRx = 0); // run function to enable changing out numpy array in run, if not provided, use the memory address in initialization
    void printFreqSample(int printLen = 10){ m_ofdmDeModulateHandle -> printFreqSample(printLen); }
    uintptr_t getFreqDataOut(){return reinterpret_cast<uintptr_t>(m_ofdmDeModulateHandle -> getFreqDataOut()); }

private:
    ofdm_demodulate::ofdmDeModulate<Tscalar, Tcomplex> * m_ofdmDeModulateHandle;
    cudaStream_t m_cuStrm;
    size_t m_freqDataOutSizeDl, m_freqDataOutSizeUl;
    Tcomplex *m_freqDataOutCpu;
    Tcomplex *m_freqDataOutGpu;
    uint8_t m_externGpuAlloc; // indicator for freqDataOut storage type: 0 - internal GPU memory allocation; 1 - external GPU memory allocation
    bool m_perAntSamp; // true: input sample is per rx-tx antenna pair; false: input sample is per rx antenna
};
// explicit instantiation
template class OfdmDeModulateWrapper<float, cuComplex>;

/*-------------------------------       TDL channel class       -------------------------------*/
template <typename Tscalar, typename Tcomplex>
class TdlChanWrapper{
public:
    TdlChanWrapper(tdlConfig_t* tdlCfg, uint16_t randSeed, uintptr_t streamHandle);
    ~TdlChanWrapper();

    void reset(){ m_tdlChanHandle -> reset(); }
    void run(const cuda_array_t<std::complex<Tscalar>>& txSigIn, float refTime0 = 0.0f, uint8_t enableSwapTxRx = 0, uint8_t txColumnMajorInd = 0);
    cuda_array_t<std::complex<Tscalar>> getRxSignalOutArray(uint8_t enableSwapTxRx);
    uintptr_t getTimeChan(){ return reinterpret_cast<uintptr_t>(m_tdlChanHandle -> getTimeChan()); }
    uintptr_t getFreqChanSc(){ return reinterpret_cast<uintptr_t>(m_tdlChanHandle -> getFreqChanSc()); }
    uintptr_t getFreqChanPrbg(){ return reinterpret_cast<uintptr_t>(m_tdlChanHandle -> getFreqChanPrbg()); }
    uintptr_t getRxSigOut(){ return reinterpret_cast<uintptr_t>(m_tdlChanHandle -> getRxSigOut()); }
    uintptr_t getRxTimeAntPairSigOut() { return reinterpret_cast<uintptr_t>(m_tdlChanHandle -> getRxTimeAntPairSigOut()); }
    uint32_t getTimeChanSize(){ return m_tdlChanHandle -> getTimeChanSize(); }
    uint32_t getFreqChanScPerLinkSize(){ return m_tdlChanHandle -> getFreqChanScPerLinkSize();}
    uint32_t getFreqChanPrbgSize(){ return m_tdlChanHandle -> getFreqChanPrbgSize(); }
    void printTimeChan(uint16_t cid = 0, uint16_t uid = 0, int printLen = 10){ m_tdlChanHandle -> printTimeChan(cid, uid, printLen); }
    void printFreqScChan(uint16_t cid = 0, uint16_t uid = 0, int printLen = 10){ m_tdlChanHandle -> printFreqScChan(cid, uid, printLen); }
    void printFreqPrbgChan(uint16_t cid = 0, uint16_t uid = 0, int printLen = 10){ m_tdlChanHandle -> printFreqPrbgChan(cid, uid, printLen); }
    void printSig(uint16_t cid = 0, uint16_t uid = 0, int printLen = 10){ m_tdlChanHandle -> printSig(cid, uid, printLen); }
    void printGpuMemUseMB(){ m_tdlChanHandle -> printGpuMemUseMB(); }
    // functions to dump TDL channels to numpy array, need external stream sync
    void dumpCir(py::array_t<std::complex<Tscalar>> cir);
    void dumpCfrPrbg(py::array_t<std::complex<Tscalar>> cfrPrbg);
    void dumpCfrSc(py::array_t<std::complex<Tscalar>> cfrSc);

    /**
    * @brief This function saves the tdl data into h5 file, for verification in matlab using verify_tdl.m
    *
    * @param padFileNameEnding optional ending of h5 file, e.g., tdlChan_1cell1Ue_4x4_A30_dopp10_cfo200_runMode0_FP32_swap0<padFileNameEnding>.h5
    */
    void saveTdlChanToH5File(std::string & padFileNameEnding = nullptr) {m_tdlChanHandle -> saveTdlChanToH5File(padFileNameEnding); };

private:
    tdlChan<Tscalar, Tcomplex> * m_tdlChanHandle;
    cudaStream_t m_cuStrm;
    uint16_t m_nLink;
    uint8_t m_runMode;
    tdlConfig_t* m_tdlCfg;
};
// explicit instantiation
template class TdlChanWrapper<float, cuComplex>;

/*-------------------------------       CDL channel class       -------------------------------*/
template <typename Tscalar, typename Tcomplex>
class CdlChanWrapper{
public:
    CdlChanWrapper(cdlConfig_t* cdlCfg, uint16_t randSeed, uintptr_t streamHandle);
    ~CdlChanWrapper();

    void reset(){ m_cdlChanHandle -> reset(); }
    void run(const cuda_array_t<std::complex<Tscalar>>& txSigIn, float refTime0 = 0.0f, uint8_t enableSwapTxRx = 0, uint8_t txColumnMajorInd = 0);
    cuda_array_t<std::complex<Tscalar>> getRxSignalOutArray(uint8_t enableSwapTxRx);
    uintptr_t getTimeChan(){ return reinterpret_cast<uintptr_t>(m_cdlChanHandle -> getTimeChan()); }
    uintptr_t getFreqChanSc(){ return reinterpret_cast<uintptr_t>(m_cdlChanHandle -> getFreqChanSc()); }
    uintptr_t getFreqChanPrbg(){ return reinterpret_cast<uintptr_t>(m_cdlChanHandle -> getFreqChanPrbg()); }
    uintptr_t getRxSigOut(){ return reinterpret_cast<uintptr_t>(m_cdlChanHandle -> getRxSigOut()); }
    uintptr_t getRxTimeAntPairSigOut() { return reinterpret_cast<uintptr_t>(m_cdlChanHandle -> getRxTimeAntPairSigOut()); }
    uint32_t getTimeChanSize(){ return m_cdlChanHandle -> getTimeChanSize(); }
    uint32_t getFreqChanScPerLinkSize(){ return m_cdlChanHandle -> getFreqChanScPerLinkSize();}
    uint32_t getFreqChanPrbgSize(){ return m_cdlChanHandle -> getFreqChanPrbgSize(); }
    void printTimeChan(uint16_t cid = 0, uint16_t uid = 0, int printLen = 10){ m_cdlChanHandle -> printTimeChan(cid, uid, printLen); }
    void printFreqScChan(uint16_t cid = 0, uint16_t uid = 0, int printLen = 10){ m_cdlChanHandle -> printFreqScChan(cid, uid, printLen); }
    void printFreqPrbgChan(uint16_t cid = 0, uint16_t uid = 0, int printLen = 10){ m_cdlChanHandle -> printFreqPrbgChan(cid, uid, printLen); }
    void printSig(uint16_t cid = 0, uint16_t uid = 0, int printLen = 10){ m_cdlChanHandle -> printSig(cid, uid, printLen); }
    void printGpuMemUseMB(){ m_cdlChanHandle -> printGpuMemUseMB(); }
    // functions to dump CDL channels to numpy array, need external stream sync
    void dumpCir(py::array_t<std::complex<Tscalar>> cir);
    void dumpCfrPrbg(py::array_t<std::complex<Tscalar>> cfrPrbg);
    void dumpCfrSc(py::array_t<std::complex<Tscalar>> cfrSc);

    /**
    * @brief This function saves the cdl data into h5 file, for verification in matlab using verify_cdl.m
    *
    * @param padFileNameEnding optional ending of h5 file, e.g., cdlChan_1cell1Ue_4x4_A30_dopp10_cfo200_runMode0_FP32_swap0<padFileNameEnding>.h5
    */
    void saveCdlChanToH5File(std::string & padFileNameEnding = nullptr) {m_cdlChanHandle -> saveCdlChanToH5File(padFileNameEnding); };

private:
    cdlChan<Tscalar, Tcomplex> * m_cdlChanHandle;
    cudaStream_t m_cuStrm;
    uint16_t m_nLink;
    uint16_t m_nBsAnt, m_nUeAnt;
    uint8_t m_runMode;
    cdlConfig_t* m_cdlCfg;
};
// explicit instantiation
template class CdlChanWrapper<float, cuComplex>;

/*-------------------------------       add Gaussian noise class       -------------------------------*/
template <typename Tscalar, typename Tcomplex>
class GauNoiseAdderWrapper{
public:
    GauNoiseAdderWrapper(uint32_t nThreads, int seed, uintptr_t streamHandle);
    void addNoise(uintptr_t d_signal, uint32_t signalSize, float snr_db);
    ~GauNoiseAdderWrapper();

private:
    GauNoiseAdder<Tcomplex> * m_gauNoiseAdder;
    cudaStream_t m_cuStrm;
};
// explicit instantiation
template class GauNoiseAdderWrapper<float, cuComplex>;

/*-------------------------------       Stochastic Channel Model class       -------------------------------*/
template <typename Tscalar, typename Tcomplex>
class StatisChanModelWrapper{
public:
        StatisChanModelWrapper(const SimConfig& sim_config,
                      const SystemLevelConfig& system_level_config,
                      const LinkLevelConfig& link_level_config,
                      const ExternalConfig& external_config,
                      uint32_t randSeed = 0,
                      uintptr_t streamHandle = 0);

    // Constructor with just sim_config and system_level_config
    StatisChanModelWrapper(const SimConfig& sim_config,
                      const SystemLevelConfig& system_level_config,
                      uint32_t randSeed = 0,
                      uintptr_t streamHandle = 0);

    // Delete copy constructor and assignment operator
    StatisChanModelWrapper(const StatisChanModelWrapper&) = delete;
    StatisChanModelWrapper& operator=(const StatisChanModelWrapper&) = delete;

    // Delete move constructor and assignment operator
    StatisChanModelWrapper(StatisChanModelWrapper&&) = delete;
    StatisChanModelWrapper& operator=(StatisChanModelWrapper&&) = delete;

    ~StatisChanModelWrapper();

    void reset() { m_statisChanModelHandle->reset(); }

    // System level run method
    void run(float refTime = 0.0f,
             uint8_t continuous_fading = 1,
             py::object activeCell = py::none(),
             py::object activeUt = py::none(),
             py::object utNewLoc = py::none(),
             py::object utNewVelocity = py::none(),
             py::object cir_coe = py::none(),
             py::object cir_norm_delay = py::none(),
             py::object cir_n_taps = py::none(),
             py::object cfr_sc = py::none(),
             py::object cfr_prbg = py::none());

    // Link level run method
    void run_link_level(float refTime0 = 0.0f,
                       uint8_t continuous_fading = 1,
                       uint8_t enableSwapTxRx = 0,
                       uint8_t txColumnMajorInd = 0);

    void dump_los_nlos_stats(py::array_t<float> lost_nlos_stats = py::array_t<float>());
    void dump_pl_sf_stats(py::array_t<float> pl_sf,
                          py::array_t<int> activeCell = py::array_t<int>(),
                          py::array_t<int> activeUt = py::array_t<int>());
    void dump_pl_sf_ant_gain_stats(py::array_t<float> pl_sf_ant_gain,
                          py::array_t<int> activeCell = py::array_t<int>(),
                          py::array_t<int> activeUt = py::array_t<int>());
    void dump_topology_to_yaml(const std::string& filename);  // see dumpTopologyToYaml in sls_chan.cuh

    /**
     * Save SLS channel data to H5 file for debugging
     *
     * @param filename_ending Optional string to append to filename
     */
    void saveSlsChanToH5File(std::string_view filename_ending = "");

private:
    statisChanModel<Tscalar, Tcomplex> * m_statisChanModelHandle;
    cudaStream_t m_cuStrm;
    uint32_t m_randSeed;
    int m_cpuOnlyMode;
};
// explicit instantiation
template class StatisChanModelWrapper<float, cuComplex>;
} // namespace pycuphy

#endif // PYCUPHY_CHAN_MODEL_HPP