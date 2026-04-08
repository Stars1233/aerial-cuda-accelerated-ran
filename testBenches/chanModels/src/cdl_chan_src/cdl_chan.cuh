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

#if !defined(CDL_CHAN_CUH_INCLUDED_)
#define CDL_CHAN_CUH_INCLUDED_

#include "../fastFadingCommon.cuh"

// Constants
#define PI 3.141592653589793
#define D2PI PI/180.0f

/**
 * @brief struct to config cdl model
 * 
 * @todo no additional MIMO antenna correlations are added
 */
 struct cdlConfig_t{
    char delayProfile = 'A';
    float delaySpread = 30;
    float maxDopplerShift = 10;
    float f_samp = 4096 * 15e3 * 2; // default numerology 1, 4096 * scSpacingHz
    uint16_t nCell = 1; // number of cells
    uint16_t nUe   = 1; // number of UEs
    // bs antenna
    std::vector<uint16_t> bsAntSize = {1,1,1,2,2}; // {M_g,N_g,M,N,P} 3GPP TR 38.901 Section 7.3
    std::vector<float> bsAntSpacing = {1.0f, 1.0f, 0.5f, 0.5f}; // [d_g_h, d_g_v, d_h, d_v] in wavelengths
    std::vector<float> bsAntPolarAngles = {45.0,-45.0}; // BS antenna polarization angles
    uint8_t bsAntPattern = 1; // 0: isotropic; 1: 38.901
    // ue antenna
    std::vector<uint16_t> ueAntSize = {1,1,2,2,1}; // {M_g,N_g,M,N,P} 3GPP TR 38.901 Section 7.3
    std::vector<float> ueAntSpacing = {1.0f, 1.0f, 0.5f, 0.5f}; // [d_g_h, d_g_v, d_h, d_v] in wavelengths
    std::vector<float> ueAntPolarAngles = {0.0, 90.0}; // UE antenna polarization angles
    uint8_t ueAntPattern = 0; // 0: isotropic; 1: 38.901
    std::vector<float> vDirection = {90, 0}; // moving direction, [RxA; RxZ] — RxA and RxZ specify the azimuth and zenith of the direction of travel of the moving UE; moving speed is converted to maxDopplerShift in cdlCfg
    uint32_t fBatch = 15e3; // update rate of quasi-static channel, will be used if batchLen is not provided
    uint16_t numRay = 20; // number of rays per cluster, generate CDL CIR
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
    // runMode 0: CDL time channel and processing tx sig
    // runMode 1: CDL time channel and frequency channel on Prbg only
    // runMode 2: CDL time channel and frequency channel on Sc and Prbg
    uint8_t procSigFreq = 0;
    // procSigFreq 0: process tx samples in time domain (default)
    // procSigFreq 1: process tx samples in freq domain
    uint8_t saveAntPairSample = 0; // save per antenna pair data, to be used to generate genie channel when CFO presents
    std::vector<uint32_t> batchLen = {}; // each elements represents lenghth of samples for a new CIR. If set, will sepercede fbatch
    void * txSigIn = nullptr; // GPU address of tx time signal, DL: [nLink, nBatch, nBsAnt, sigLenPerAnt]; E.g., index 1 is for(0,0,0,1);   index ((1*nBatch+2)*nBsAnt+3)*sigLenPerAnt+4 is (linkIdx 1, batchIdx 2, BsAntIdx 3, sigPerAntIdx 4);    linkIdx = cid*nUe+uid;        UL: nBsAnt -> nUeAnt
};


/**
 * @brief CDL dynamic descriptor, inherited from the basic dynamic descriptor
 * 
 */
template <typename Tscalar, typename Tcomplex> 
struct cdlDynDescr_t
{
    // CDL specific params
    bool LosTap; // 0: first tap is NLOS, 1: first tap is LOS
    uint16_t nTaps; // number of fir Nz taps to add up
    uint16_t nRay; // number of sins to add up

    uint16_t bsAntSize[5];
    float bsAntSpacing[4];
    float bsAntPolarAngles[2];
    uint8_t bsAntPattern; // 0: isotropic; 1: 38.901

    uint16_t ueAntSize[5];
    float ueAntSpacing[4];
    float ueAntPolarAngles[2];
    uint8_t ueAntPattern; // 0: isotropic; 1: 38.901

    float vDirection[2]; // direction [x, y]

    float* dpa;
    float* pcp;
    uint16_t* firNzTapMap;  // tap index for each tap in pdp table
    float * rayOffsetAngles;   //ray offset angles defined in 
    uint16_t** rayCouplingRand;  // ray coupling, random generated
    Tscalar** phiRand;  // initial phase phi, random generated

    Tcomplex * multi5Term;   // GPU address to save muti of first five terms
    Tscalar * rheadVbar;  // GPU address to save 2*pi*(r_head_rx_n_m'*v_head*dopplerHz)
};


template <typename Tscalar, typename Tcomplex>
class cdlChan : public fastFadingBaseChan<Tscalar, Tcomplex> {
public:
    /**
     * @brief Construct a new cdl Chan object
     * 
     * @param cdlConfig CDL chan configurations
     * @param randSeed random seed to generate cdl channel
     * @param strm cuda stream during config setup
     */
    cdlChan(cdlConfig_t * cdlCfg, uint16_t randSeed, cudaStream_t strm);
    ~cdlChan();

    /**
     * @brief generate cdl time chan and process the tx time signals
     * will generate cdl freq chan if runMode=1 or 2
     * will process tx time samples if sigLenPerAnt > 0 
     * support time correlation so the time stamp should be input
     * 
     * @param refTime0 the time stamp for the start of tx symbol
     * @param enableSwapTxRx: 0: DL case; 1: UL case
     * @param txColumnMajorInd: 0: input sample is row major; 1: input sample is column major
     */
    void run(float refTime0 = 0.0f, uint8_t enableSwapTxRx = 0, uint8_t txColumnMajorInd = 0);

    /**
     * @brief reset channel inintial ray coupling and phase, regenerate all term 1~5 in CDL equation 3GPP TS 38.901 7.5-22
     * 
     */
    void reset();

    /**
    * @brief This function saves the cdl data into h5 file, for verification in matlab using verify_cdl.m
    * 
    * @param padFileNameEnding optional ending of h5 file, e.g., cdlChan_1cell1Ue_4x4_A30_dopp10_cfo200_runMode0_FP32_swap0<>.h5
    */
    void saveCdlChanToH5File(std::string & padFileNameEnding = nullptr);

    // print GPU memory usage in MB, only sum of explicit cudaMalloc() calls
    void printGpuMemUseMB();
private:

    /**
     * @brief update the random phase in cdl chann
     * this help control the time domain correlation
     * by default it only runs during cdl chan setup
     */
    void updateTapPathRand();

    /**
     * @brief update the random ray coupling in cdl chann
     * this help control the time domain correlation
     * by default it only runs during cdl chan setup
     */
    void updateRayCouplingRand();

    // generate CDL time channels
    void genTimeChan() override;

    cdlConfig_t * m_cdlCfg; // only save pointer
    bool m_LosTap; // whether the fm_strmirst tap is LOS
    uint16_t m_nRay;   // number of sins to add up 
    char m_delayProfile; // delay profile
    float * m_dpa;  // read Delay Power Angle
    float * m_pcp;  // read Per-Cluster Parameters
    float m_delaySpread; 
    std::vector<uint16_t> m_firNzTapMap;  // tap index for each tap in pdp table
    uint32_t m_randPhiSizePerLink; // size of random phi per link
    uint32_t m_randRayCouplingSizePerLink; // size of random ray coupling per link
    uint8_t m_numTaps; // numbe of taps in dpa table
    curandGenerator_t m_Rng; // random number generator
    Tscalar ** m_h_devicePhiRandPerLinkPtr; // per link phi rand ptr saved on host; If not using ptr of ptr, the rand size is too large
    uint16_t ** m_h_deviceRayCouplingRandPerLinkPtr;  // per link ray coupling rand ptr saved on host; If not using ptr of ptr, the rand size is too large
    // dynamic descriptor
    cdlDynDescr_t<Tscalar, Tcomplex> * m_cdlDynDescrCpu;
    cdlDynDescr_t<Tscalar, Tcomplex> * m_cdlDynDescrGpu;
};

// Explicitly instantiate the template to resovle "undefined functions"
    template class cdlChan<__half, __half2>;
    template class cdlChan<float, cuComplex>;

/**
 * @brief CUDA kernel to generate the long term params: multiplication of five terms
 * 
 * @param fastFadingDynDescr fast fading dynamic descriptor
 * @param refTime0 the time stamp for the start of tx symbol
 * @param cdlDynDescr CDL dynamic descriptor
 */
template <typename Tscalar, typename Tcomplex> 
static __global__ void genCdlMulti5TermsKernel(fastFadingDynDescr_t<Tscalar, Tcomplex> * fastFadingDynDescr, float refTime0, cdlDynDescr_t<Tscalar, Tcomplex> * cdlDynDescr);

/**
 * @brief CUDA kernel to generate time domain channel
 * 
 * @param fastFadingDynDescr fast fading dynamic descriptor
 * @param refTime0 the time stamp for the start of tx symbol
 * @param cdlDynDescr CDL dynamic descriptor
 */
template <typename Tscalar, typename Tcomplex> 
static __global__ void genCdlTimeChanCoeKernel(fastFadingDynDescr_t<Tscalar, Tcomplex> * fastFadingDynDescr, float refTime0, cdlDynDescr_t<Tscalar, Tcomplex> * cdlDynDescr);

#endif // !defined(CDL_CHAN_CUH_INCLUDED_)
