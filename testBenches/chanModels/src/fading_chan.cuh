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

#if !defined(FADING_CHAN_CUH_INCLUDED_)
#define FADING_CHAN_CUH_INCLUDED_

#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <vector>
#include <cuda_runtime.h>
#include "cuda.h"
#include "cuda_fp16.h"
#include <string>
#include <curand.h>
#include <curand_kernel.h>
#include <chrono>
#include <fstream>
#include <math.h>
#include <random>
#include "cuComplex.h"
#include <type_traits>
#include <typeinfo>

#include "./ofdm_src/ofdmMod.cuh"
#include "./ofdm_src/ofdmDemod.cuh"
#include "./tdl_chan_src/tdl_chan.cuh"
#include "./cdl_chan_src/cdl_chan.cuh"
#include "gauNoiseAdder.cuh"
// Only remaining header from cuPHY tree (cuPHY/examples/common/hdf5hpp.hpp).
// Header-only HDF5 C++ wrapper; no transitive cuPHY/CUDA dependencies.
// TODO: copy into chanModels/src/ to fully decouple from the cuPHY tree.
#include "hdf5hpp.hpp"

template<class T> struct getScalarType {};
float type_convert(getScalarType<cuComplex>);
__half type_convert(getScalarType<__half2>);

template <typename Tcomplex> 
struct fadingChanDynDescr_t
{
    Tcomplex * sigNoiseFree; // noise free signal for input
    Tcomplex * sigNoisy;     // noisy signal for output
    Tcomplex * noise; // noise to be add
    uint32_t sigLenDl; // length of signal in DL
    uint32_t sigLenUl; // length of signal in Ul
    uint32_t seed; // seed for generate Gaussian noise
};

/**
 * @brief fading channel class
 * this connects tdl channel and ofdm class
 *
 * @note Support data type FP16 and FP32 through <typename Tcomplex>
 * valid data types for <typename Tcomplex> are __half2 and cuComplex:
 *      __half2:   using FP16 data format, cuPHY signal should use CUPHY_C_16F
 *      cuComplex: using FP32 data format, cuPHY signal should use CUPHY_C_32F
 * @note the data buffers Tx and freqRx have to be allocated outside, fadingChan do not check their dimensions
 */
template <typename Tcomplex> 
class fadingChan{
public:
    /**
     * @brief Construct a new fading Chan object
     * 
     * @param Tx GPU memory for freq/time-domain tx samples
     * @param freqRx GPU memory for freq rx samples
     * @param fadingMode fading mode, currently support 0: no additional fading, only GauNoiseAdder is applied, 1: TDL, 2: CDL
     * @param randSeed random seed to generate 
     * @param strm cuda stream to run fading chan 
     * @param phyChannType indicator for physical channel type: 0 - PUSCH, 1 - PUCCH, 2 - PRACH
     */
    fadingChan(Tcomplex* Tx, Tcomplex* freqRx, cudaStream_t strm, uint8_t fadingMode = 1, uint16_t randSeed = 0, uint8_t phyChannType = 0);
    ~fadingChan();
    fadingChan(fadingChan const&) = delete;
    fadingChan& operator=(fadingChan const&) = delete;

    /**
     * @brief setup fadingChan class using a TV file
     * 
     * @param inputFile input TV file with chan_pars, carrier_pars and, timeTx (for TDL/CDL testing), freqRx (for no additional fading testing)
     * @param enableSwapTxRx swap tx and rx
     */
    void setup(hdf5hpp::hdf5_file& inputFile, uint8_t enableSwapTxRx = 0);

    /**
     * @brief run fading channel
     * perform ofdm mod, generate tdl time chan, apply tdl chan to tx sample, add nosie, perform ofdm demod
     * 
     * @param refTime0  the time stamp for the start of tx symbol
     * @param targetSNR target SNR for adding noise
     * @param enableSwapTxRx enable swap tx an rx
     */
    void run(float refTime0 = 0.0f, float targetSNR = 0.0f, uint8_t enableSwapTxRx = 0); // run fading channel pipleline

    /**
     * @brief save freq tx,rxNoisy, rxNoiseFree, noise data, and estimated SNR (if exists) to "fadingChanData.h5" file
     */
    void savefadingChanToH5File();

    /**
     * @brief estimate SNR from a specific OFDM symbol and a set of SCs, based on m_freqRxNoiseFree and tempFreqNoisyBuffer 
     * 
     * @param ofdmSymIdx index of OFDM symbol
     * @param startSC start SC index, inclusive
     * @param endSC end SC index, exclusive; total SCs used is [endSC , startSC)
     * 
     * @note report average SNR over all antennas in command line & save SNRs to "SNR.txt" file during savefadingChanToH5File()
     */
    void calSnr(uint16_t ofdmSymIdx, uint16_t startSC, uint16_t endSC);

private:

    /**
     * @brief add noise to rx time samples
     * 
     * @param targetSNR SNR to calculate noise samples, default 0.0f
     */
    void addNoiseFreq(float targetSNR = 0.0f); // add noise

    /**
     * @brief read carrier and channel parameters from TV file into m_carrierPrms and m_tdlCfg
     * 
     * @param inputFile input TV file
     */
    void readCarrierChanPar(hdf5hpp::hdf5_file& inputFile);

    /**
     * @brief read samples from TV
     * for AWGN, read rx freq data from TV into m_freqRx
     * for TDL, read tx freq data from TV m_freqTx
     * 
     * @param inputFile  input TV file
     */
    void read_Xtf(hdf5hpp::hdf5_file& inputFile);

    /**
     * @brief read frequency/time-domain samples from TV for PRACH
     * @param inputFile  input TV file
     */
    void read_Xtf_prach(hdf5hpp::hdf5_file& inputFile);

    /* -----  tdl, ofdm classes ---------*/
    // configuration
    cuphyCarrierPrms_t* m_carrierPrms = nullptr;
    tdlConfig_t* m_tdlCfg = nullptr;
    cdlConfig_t* m_cdlCfg = nullptr;

    // class declaration
    using myTscalar = decltype(type_convert(getScalarType<Tcomplex>{}));
    tdlChan<myTscalar, Tcomplex> * m_tdl_chan = nullptr;  // ptr to tdl channel class
    cdlChan<myTscalar, Tcomplex> * m_cdl_chan = nullptr;  // ptr to cdl channel class
    GauNoiseAdder<Tcomplex> * m_gauNoiseAdder = nullptr;  // ptr to Gaussian noise adder
    ofdm_modulate::ofdmModulate<myTscalar, Tcomplex> * m_ofdmMod = nullptr; // ptr to ofdm modulation class
    ofdm_demodulate::ofdmDeModulate<myTscalar, Tcomplex> * m_ofdmDeMod = nullptr; // ptr to ofdm demodualation class

    /* -----  sample buffers in GPU ---------*/
    Tcomplex * m_Tx;
    Tcomplex * m_timeTx;
    Tcomplex * m_timeRx;
    Tcomplex * m_freqRxNoiseFree;
    Tcomplex * m_freqRxNoisy;
    uint m_freqTxDataSizeDl, m_freqTxDataSizeUl; // length of frequency-domain tx sample
    uint m_timeTxDataSizeDl, m_timeTxDataSizeUl; // length of time-domain tx sample
    uint m_freqRxDataSizeDl, m_freqRxDataSizeUl; // length of frequency-domain rx sample, for adding noise
    
    //rng and noise buffer for adding noise
    Tcomplex * m_noise;
    /**
     * @param  m_phyChannType physical channel type
     * 0: PUSCH
     * 1: PUCCH
     * 2: PRACH
     */
    uint8_t m_phyChannType;

    /**
     * @param  m_prach indicator for PRACH
     * true: PRACH
     * false: non-PRACH
     */
    bool m_prach;

    /**
     * @param  m_fadingMode fading mode, currently support AWGN and TDL
     * 0: AWGN
     * 1: TDL based on TV
     * 2: CDL based on TV
     */
    uint8_t m_fadingMode;
    uint16_t m_randSeed;
    uint8_t m_enableSwapTxRx;
    cudaStream_t m_strm; // stream to perform all processing

    std::vector<float> m_SNR; // for tracking the SNR over time

    // fading channel descriptors
    fadingChanDynDescr_t<Tcomplex> * m_fadingChanDynDescrCpu;
    fadingChanDynDescr_t<Tcomplex> * m_fadingChanDynDescrGpu;
};

// Explicitly instantiate the template to resovle "undefined functions"
template class fadingChan<__half2>;
template class fadingChan<cuComplex>;

#endif // !defined(FADING_CHAN_CUH_INCLUDED_)