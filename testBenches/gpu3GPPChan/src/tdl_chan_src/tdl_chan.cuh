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

#if !defined(TDL_CHAN_CUH_INCLUDED_)
#define TDL_CHAN_CUH_INCLUDED_

#include "../fastFadingCommon.cuh"

/**
 * @brief struct to config tdl model
 * 
 * @todo no additional MIMO antenna correlations are added
 */
struct tdlConfig_t{
    bool useSimplifiedPdp = true; // true for simplified pdp in 38.141, false for 38.901
    char delayProfile = 'A';
    float delaySpread = 30;
    float maxDopplerShift = 5;
    float f_samp = 4096 * 15e3 * 2; // default numerology 1, 4096 * scSpacingHz
    uint16_t nCell = 1; // number of cells
    uint16_t nUe   = 1; // number of UEs
    uint16_t nBsAnt = 4; // number tx antennas
    uint16_t nUeAnt = 4; // number rx antennas
    uint32_t fBatch = 15e3; // update rate of quasi-static channel, will be used if batchLen is not provided
    uint16_t numPath = 48; // number sin waves to super impose, generate TDL CIR
    float cfoHz = 200.0f; // cfo in Hz
    float delay = 0.0f; // delay in second
    uint32_t sigLenPerAnt = 4096; // tx sample length per antenna per ue per cell
    uint16_t N_sc = 68*4*12; // number of Sc
    uint16_t N_sc_Prbg = 4*12; // number of Sc per Prbg
    float    scSpacingHz = 15e3 * 2; // subcarrier spacing in Hz, default numerology 1
    uint8_t  freqConvertType = 1;
    // freqConvertType 0: use first SC for CFR on the Prbg
    // freqConvertType 1: use center SC for CFR on the Prbg, e.g., sc 6 for sc 0,1,2,...,11
    // freqConvertType 2: use last SC for CFR on the Prbg
    // freqConvertType 3: use average SC for CFR on the Prbg
    // freqConvertType 4: use average SC for CFR on the Prbg with removing frequency ramping
    uint8_t scSampling = 1; // whether to only calculate CFR for a subset of Scs, within a Prbg, only Scs for 0:scSampling:N_sc_Prbg-1 wil be calculated; only appliable when not using FFT and freqConvertType = 3 or 4
    uint8_t runMode = 0;
    // runMode 0: time channel and processing tx sig
    // runMode 1: time channel and frequency channel on Prbg only
    // runMode 2: time channel and frequency channel on Sc and Prbg
    uint8_t procSigFreq = 0;
    // procSigFreq 0: process tx samples in time domain (default)
    // procSigFreq 1: process tx samples in freq domain
    uint8_t saveAntPairSample = 0; // save per antenna pair data, to be used to generate genie channel when CFO presents
    std::vector<uint32_t> batchLen = {}; // each elements represents length of samples for a new CIR. If set, will sepersede fBatch
    void * txSigIn = nullptr; // GPU address of tx sample, [nLink, nBatch, nBsAnt, sigLenPerAnt]; E.g., index 1 is for(0,0,0,1);   index ((1*nBatch+2)*nBsAnt+3)*sigLenPerAnt+4 is (linkIdx 1, batchIdx 2, bsAntIdx 3, sigPerAntIdx 4);    linkIdx = cid*nUe+uid
};

/**
 * @brief TDL dynamic descriptor, only has specific params
 * 
 */
template <typename Tscalar, typename Tcomplex> 
struct tdlDynDescr_t
{
    bool LosTap; // 0: first tap is NLOS, 1: first tap is LOS
    uint16_t nTaps; // number of fir Nz taps to add up
    uint16_t nPath; // number of sins to add up
    uint16_t* firNzTapMap;  // tap index for each tap in pdp table
    Tscalar ** thetaRand; // rand phase for real and imag part due to doppler
    float PI_2_nPath; // a constant pi/2/nPath, calculated on CPU
    float PI_4_nPath; // a constant pi/4/nPath, calculated on CPU
};

/**
* @brief TDL channel class
*/
template <typename Tscalar, typename Tcomplex>
class tdlChan : public fastFadingBaseChan<Tscalar, Tcomplex> {
public:
    /**
     * @brief Construct a new tdl Chan object
     * 
     * @param tdlConfig TDL chan configurations
     * @param randSeed random seed to generate tdl channel
     * @param strm cuda stream during config setup
     */
    tdlChan(tdlConfig_t * tdlCfg, uint16_t randSeed, cudaStream_t strm);
    ~tdlChan();

    /**
     * @brief generate tdl time chan and process the tx time signals
     * will generate tdl freq chan if runMode=1 or 2
     * will process tx time samples if sigLenPerAnt > 0 
     * support time correlation so the time stamp should be input
     * 
     * @param refTime0 the time stamp for the start of tx symbol
     * @param enableSwapTxRx: 0: DL case; 1: UL case
     * @param txColumnMajorInd: 0: input sample is row major; 1: input sample is column major
     */
    void run(float refTime0 = 0.0f, uint8_t enableSwapTxRx = 0, uint8_t txColumnMajorInd = 0);

    /**
     * @brief reset channel inintial phase
     * 
     */
    void reset();

    /**
    * @brief This function saves the tdl data into h5 file, for verification in matlab using verify_tdl.m
    * 
    * @param padFileNameEnding optional ending of h5 file, e.g., tdlChan_1cell1Ue_4x4_A30_dopp10_cfo200_runMode0_FP32_swap0<>.h5
    */
    void saveTdlChanToH5File(const std::string & padFileNameEnding = "");

    // print GPU memory usage in MB, only sum of explicit cudaMalloc() calls
    void printGpuMemUseMB();
private:

    /**
     * @brief update the random mag and phase in tdl chann
     * this help control the time domain correlation
     * by default it only runs during tdl chan setup
     */
    void updateTapPathRand();

    // generate TDL time channels
    void genTimeChan() override;

    tdlConfig_t * m_tdlCfg; // only save pointer
    bool m_LosTap; // whether the fm_strmirst tap is LOS
    uint16_t m_nPath;   // number of sins to add up 
    char m_delayProfile; // delay profile
    float * m_pdp; // read power delay profile from tables stored in tdl_pdp_table.h
    float m_delaySpread; 
    std::vector<uint16_t> m_firNzTapMap;  // tap index for each tap in pdp table
    bool m_useSimplifiedPdp; // true for simplified pdp in 38.141, false for 38.901
    uint32_t m_randSizePerLink; // size of random numbers per link
    uint8_t m_numTaps; // numbe of taps in pdp table
    curandGenerator_t m_Rng; // random number generator
    Tscalar ** m_h_deviceThetaRandPerLinkPtr; // per link theta rand ptr saved on host; If not using ptr of ptr, the rand size is too large
    // dynamic descriptor
    tdlDynDescr_t<Tscalar, Tcomplex> * m_tdlDynDescrCpu;
    tdlDynDescr_t<Tscalar, Tcomplex> * m_tdlDynDescrGpu;
};

// Explicitly instantiate the template to resovle "undefined functions"
    template class tdlChan<__half, __half2>;
    template class tdlChan<float, cuComplex>;

/**
 * @brief CUDA kernel to generate time domain channel
 * 
 * @param fastFadingDynDescr fast fading dynamic descriptor
 * @param refTime0 the time stamp for the start of tx symbol
 * @param tdlDynDescr TDL dynamic descriptor
 */
template <typename Tscalar, typename Tcomplex> 
static __global__ void genTdlTimeChanCoeKernel(fastFadingDynDescr_t<Tscalar, Tcomplex> * fastFadingDynDescr, float refTime0, tdlDynDescr_t<Tscalar, Tcomplex> * tdlDynDescr);

#endif // !defined(TDL_CHAN_CUH_INCLUDED_)
