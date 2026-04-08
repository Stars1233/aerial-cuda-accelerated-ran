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

#if !defined(FAST_FADING_COMMON_CUH_INCLUDED_)
#define FAST_FADING_COMMON_CUH_INCLUDED_

#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <vector>
#include "hdf5.h"
#include "cuda.h"
#include "cuda_fp16.h"
#include <string>
#include <curand.h>
#include <curand_kernel.h>
#include <chrono>
#include <fstream>
#include <math.h>
#include <random>
#include <algorithm>
#include <unordered_map>
#include "cuComplex.h"
#include "../chanModelsCommon.h"

#define MAX_NZ_TAPS_ 25 // max number of taps according to 3GPP specs
#define USE_MEMOERY_FFT_SHIFT_ // use fft shift in memeory or multiplication
#define FFTs_PER_BLOCK_CONST_ 1 // FFTs per block when generating freq domain channels on Sc and Prbg, changing this will affect run times, should be a divisor of OFDM_SYMBOLS_PER_SLOT (14)
#define USE_FFT_CAL_CFR 0 // 0: use direct equation to convert CIR to CFR; 1: use FFT
#define CAL_COS_SIN_IN_GPU 0 // 0: cos and sin will be calculated on CPU and copied to GPU; 1: calculated on GPU
// macros for normalization
// By default, no need for post-generation normalization, ensure over long term E{||chanMatrix(ueAntIdx, bsAntIdx, :, batchIdx)||_2^2} = 1 for all (ueAntIdx, bsAntIdx, batchIdx), see detials in "aerial_sdk/5GModel/nr_matlab/channel/genTDL.m or genCDL.m"
// #define ENABLE_NORMALIZATION_ // normalize time chan coes per slot
#define THREADS_PER_BLOCK_NORMALIZATION_ 128 // # of threads per block in time channel normalization


/**
 * @brief fast fading basic dynamic descriptor
 * 
 */
template <typename Tscalar, typename Tcomplex> 
struct fastFadingDynDescr_t
{
    // cell config params
    uint32_t nLink; // number of links (cell <-> ue pair)
    uint16_t nBsAnt; // number of BS antennas
    uint16_t nUeAnt; // number of UE antennas
    uint16_t nBatch; // batch of channel coefficients to genearate
    uint32_t * batchCumLen; // cumulative number of samples for each batch [0, 4096, 8192, 12288] means new CIR/CFR per 4096 tx samples. This can be used to specify 14 OFDM symbols
    float * tBatch;  // batch offsets in time, w.r.t. the start of the first sample
    float cfoHz; // CFO in Hz
    float cfoPhaseSamp; // pre calculated 2*pi*cfoHz/f_samp  
    float maxDopplerShift;
    bool LosTap; // 0: first tap is NLOS, 1: first tap is LOS
    // channel input and output, use DL as example
    uint32_t sigLenPerAnt; // lenght of tx time signal per antenna
    Tcomplex * txSigIn; // GPU address of tx time signal, [nLink, nBatch, nBsAnt, sigLenPerAnt]; E.g., index 1 is for(0,0,0,1);   index ((1*nBatch+2)*nBsAnt+3)*sigLenPerAnt+4 is (linkIdx 1, batchIdx 2, bsAntIdx 3, sigPerAntIdx 4); 
    Tcomplex * rxSigOut; // GPU address of rx time signal, [nLink, nBatch, nUeAnt, sigLenPerAnt]; E.g., index 1 is for(0,0,0,1);   index ((1*nBatch+2)*nUeAnt+3)*sigLenPerAnt+4 is (linkIdx 1, batchIdx 2, ueAntIdx 3, sigPerAntIdx 4); 
    // FIR related params, only store non-zero parameters, as sparse matrix
    Tcomplex * rxSigOutPerAntPair; // GPU address of rx time signal per rx-tx antenna pair, [nLink, nBatch, nUeAnt, nBsAnt, sigLenPerAnt]; E.g., index 1 is for(0,0,0,0,1);   index (((1*nBatch+2)*nUeAnt+3)*nBsAnt+4)*sigLenPerAnt+5 is (linkIdx 1, batchIdx 2, ueAntIdx 3, bsAntIdx 4, sigPerAntIdx 5); only use if saveAntPairSample = 1
    // FIR related params, only store non-zero parameters, as sparse matrix
    float * firNzPw; // non-zero coefficient of FIR, calculated on CPU
    uint16_t * firNzIdx; // non-zero indexes
    Tscalar ** thetaRand; // rand phase for real and imag part due to doppler
    float PI_2_nPath; // a constant pi/2/nPath, calculated on CPU
    float PI_4_nPath; // a constant pi/4/nPath, calculated on CPU
    uint16_t firNzLen; // number of non-zero taps
    uint16_t firMaxLen; // maximum number of taps
    // time channel coefficient
    Tcomplex * timeChan; // GPU memeory address of channel in time domain, index: [nLink, nBatch, nUeAnt, nBsAnt, firNzLen]; E.g., index 1 is for(0,0,0,0,1);   index (((1*nBatch+2)*nUeAnt+3)*nBsAnt+4)*firNzLen+5 is (linkidx 1, batchIdx 2, ueAntIdx 3, bsAntIdx 4, firNzIdx 5); Only non-zero chan coe are stored to save memory;    linkIdx = cid*nUe+uid
    uint32_t timeChanSizePerLink; // length of time domain coes per link
    uint32_t timeChanSize; // total length of time domain coes, timeChanSizePerLink * nLink
    uint16_t procTxSampBlockSample; // sample per block for processing tx sample
    // freq channel coefficient
    uint32_t N_FFT; // FFT size to generate freq channel
    uint16_t N_sc;  // number of Sc
    uint16_t N_Prbg; // number of Prbg, calculated by fastFadingBaseChan class
    uint16_t N_sc_Prbg; // number of Scs per Prbg, will be used to calculate N_Prbg
    uint16_t N_sc_last_Prbg; // number of Scs in the last Prbg, may not have full N_sc_Prbg Scs
    float * scFreqKHz; // sc frequency in KHz, to calculate CFR based on CIR, dim: N_sc
    float * firNzDelayUs2Pi; // fir non-zero delays in us * -2 * pi, dim: firNzLen
    Tcomplex * firNzDelayScFreq2Pi; // fir non-zero delays in us * -2 * pi * sc frequency in Hz, dim: N_sc * firNzLen
    uint8_t freqConvertType;
    // freqConvertType 0: use first SC for CFR on the Prbg
    // freqConvertType 1: use center SC for CFR on the Prbg, e.g., sc 6 for sc 0,1,2,...,11
    // freqConvertType 2: use last SC for CFR on the Prbg
    // freqConvertType 3: use average SC for CFR on the Prbg
    // freqConvertType 4: use average SC for CFR on the Prbg with removing frequency ramping
    uint8_t scSampling;  // whether to only calculate CFR for a subset of Scs, within a Prbg, only Scs for 0:scSampling:N_sc_Prbg-1 wil be calculated; only appliable when not using FFT and freqConvertType = 3 or 4
    uint8_t saveAntPairSample; // save per antenna pair data, to be used to generate genie channel when CFO presents
    float inverseNScPrbg; // for avg conversion of CFR on Sc to Prbg
    float inverseNScLastPrbg; // for avg conversion of CFR on Sc to Prbg on the last Prbg, whihc may not have full N_sc_Prbg Scs
    float cfrPhaseShiftTimeDelay; // CFR phase shift due to delay in time; assuming delay is within in CP
    Tcomplex * cfrBatchRotationCfo; // CFR rotation per batch due to CFO
    Tcomplex ** freqChanSc; // frequency channel on Sc, index: from big-> small: [nLink, nUeAnt, nBsAnt, nSc]; E.g., index 1 is for(0,0,0,1);  index ((1*nUeAnt+2)*nBsAnt+3)*nSc+4 is (linkIdx 1, ueAntIdx 2, bsAntIdx 3, scIdx 4);    linkIdx = cid*nUe+uid
    Tcomplex * freqChanPrbg; // frequency channel on Prbg, index: from big-> small: [nLink, nUeAnt, nBsAnt, nPrbg]; E.g., index 1 is for(0,0,0,1);  index ((1*nUeAnt+2)*nBsAnt+3)*nPrbg+4 is (linkIdx 1, ueAntIdx 2, bsAntIdx 3, prbgIdx 4);    linkIdx = cid*nUe+uid
    // for adding delay
    uint32_t nDelaySample; // delay / T_sample 
};

/**
* @brief fast fading channel basic class
*/
template <typename Tscalar, typename Tcomplex>
class fastFadingBaseChan {
public:
    /**
     * @brief Construct a new fast fading base chan object, only allocate memory of m_fastFadingDynDescrCpu and m_fastFadingDynDescrGpu
     * 
     */
    fastFadingBaseChan();

    /**
     * @brief setup a new fast fading base chan object using internal m_fastFadingDynDescrCpu
        m_fastFadingDynDescrCpu must be initialized in inherited class (e.g., fastFadingBaseChan, cdlChan)
     */
    void setup();
    ~fastFadingBaseChan();

    /**
     * @brief generate time chan and process the tx time signals
     * will generate freq chan if runMode=1 or 2
     * will process tx time samples if sigLenPerAnt > 0 
     * support time correlation so the time stamp should be input
     * 
     * @param refTime0 the time stamp for the start of tx symbol
     * @param enableSwapTxRx: 0: DL case; 1: UL case
     * @param txColumnMajorInd: 0: input sample is row major; 1: input sample is column major
     */
    void genChanProcSig(float refTime0 = 0.0f, uint8_t enableSwapTxRx = 0, uint8_t txColumnMajorInd = 0);

    // obtain channel memory address
    Tcomplex * getTimeChan() {return m_fastFadingDynDescrCpu -> timeChan;}; 
    Tcomplex ** getFreqChanSc() {return m_fastFadingDynDescrCpu -> freqChanSc;}; // this will return the device array of pointers to the address of GPU memory, i-th element corrosponding to the CFR on SC for link i
    Tcomplex ** getFreqChanScHostArray() {return m_h_deviceFreqChanScPerLinkPtr;} // this will return the host array of pointers to the address of GPU memory, i-th element corrosponding to the CFR on SC for link i
    Tcomplex * getFreqChanPrbg() {return m_fastFadingDynDescrCpu -> freqChanPrbg;};
    Tcomplex * getRxSigOut() {return m_fastFadingDynDescrCpu -> rxSigOut;}; // get output signals
    Tcomplex * getRxTimeAntPairSigOut() {return m_fastFadingDynDescrCpu -> rxSigOutPerAntPair;}; //  get rx-txc ant pair output signals
    uint16_t * getFirNzIdx() {return m_fastFadingDynDescrCpu -> firNzIdx;}; // get fir non-zero indexes
    uint32_t getTimeChanSize(){return m_timeChanSize;};
    uint32_t getFreqChanScPerLinkSize(){return m_freqChanScSizePerLink;};
    uint32_t getFreqChanPrbgSize(){return m_freqChanPrbgSize;};
    
    /**
     * @brief Set the input signal pointer for processing
     * Updates CPU descriptor and copies to GPU
     * @param txSigIn GPU pointer to input signal
     */
    void setTxSigIn(Tcomplex* txSigIn);
    
    /**
     * @brief Copy the CPU descriptor to GPU
     * Call after modifying CPU descriptor fields to sync changes to GPU
     */
    void copyDescriptor();

    // for printout samples
    void printTimeChan(uint16_t cid = 0, uint16_t uid = 0, int printLen = 10);
    void printFreqScChan(uint16_t cid = 0, uint16_t uid = 0, int printLen = 10);
    void printFreqPrbgChan(uint16_t cid = 0, uint16_t uid = 0, int printLen = 10);
    void printSig(uint16_t cid = 0, uint16_t uid = 0, int printLen = 10);

protected:
    /**
     * @brief find the thread dimension by scaling the number of tx and rx antenna
     * 
     * @param blockDimX blockIdx.x, firNzLen (max 24) for time chan, or N_Prbg(max 273) for greq chan; 
     * @param nUeAnt number of rx antennas
     * @param nBsAnt number of tx antennas
     * @param scaleUeAnt scaling factor of rx antennas, blockIdx.z = nUeAnt / scaleUeAnt
     * @param scaleBsAnt scaling factor of rx antennas, blockIdx.y = nBsAnt / scaleBsAnt
     */
    inline void findGenFastFadingChanKernelDim(const uint16_t & blockDimX, const uint16_t& nUeAnt, const uint16_t& nBsAnt, uint16_t & scaleUeAnt, uint16_t & scaleBsAnt);
    
    /**
     * @brief calculate m_firNzPw, m_firNzDelayScFreq2Pi, save in CPU and copy to GPU;
     * only called in setup()
     * 
     */
    inline void calCfrRelateParam();

    // generate channels
    virtual void genTimeChan() = 0;  // will throw error when running from the base class
    void genFreqChan();

    // process tx time domain signals
    void processTxSig(uint8_t txColumnMajorInd);
    // save dataset to H5 file
    void saveChanToH5File(hid_t & h5FileHandle, hid_t & complexDataType);

    // dynamic constructor for basic class
    fastFadingDynDescr_t<Tscalar, Tcomplex> * m_fastFadingDynDescrCpu;
    fastFadingDynDescr_t<Tscalar, Tcomplex> * m_fastFadingDynDescrGpu;
    cudaStream_t m_strm; // cuda stream
    uint16_t m_nCell;
    uint16_t m_nUe;
    uint32_t m_nLink; // nCell * nUe, index 1 is cell 0, UE 1
    uint16_t m_nBsAnt;
    uint16_t m_scaleBsAntTimeChan; // launch kernel dimension w.r.t. nBsAnt in time chan generation
    uint16_t m_scaleBsAntFreqChan; // launch kernel dimension w.r.t. nBsAnt in freq chan generation
    uint16_t m_nUeAnt;
    uint16_t m_scaleUeAntTimeChan; // launch kernel dimension w.r.t. nUeAnt in time chan generation
    uint16_t m_scaleUeAntFreqChan; // launch kernel dimension w.r.t. nUeAnt in freq chan generation
    float m_maxDopplerShift;
    float m_cfoHz;  // CFO in Hz
    float m_f_samp;  // sampling frequency
    float m_fBatch;  // update rate of quasi-static channel
    std::vector<uint32_t> m_batchLen = {}; // each elements represents lenghth of samples for a new CIR. If set, will sepercede fbatch 
    float * m_mimoCorrMat; // MIMO correlation matrix, currently not used
    float m_delay;  // time delay in seconds
    uint16_t m_nBatch; // batch of channel coefficients to genearate 
    std::vector<uint32_t> m_batchCumLen; // cumulative number of samples for each batch [0, 4096, 8192, 12288] means 3 batches, new CIR/CFR per 4096 tx samples. This can be used to specify 14 OFDM symbols
    std::vector<float> m_tBatch;  // batch offsets in time, w.r.t. the start of the first sample
    uint32_t m_sigLenPerAnt; // numbe of samples per antenna
    void * m_txSigIn;  // tx signal
    uint64_t m_sigLenTx; // numbe of tx samples, must be smaller than 2^32
    uint64_t m_sigLenRx; // numbe of rx samples, must be smaller than 2^32
    uint32_t m_timeChanSizePerLink; // time channel size per link
    uint64_t m_timeChanSize; // time channel size for all links, must be smaller than 2^32
    uint64_t m_freqChanScSizePerLink; // freq channel on Sc size per link, must be smaller than 2^32
    uint64_t m_freqChanPrbgSize; // freq channel on Prbg size for all links, must be smaller than 2^32
    float m_scSpacingHz; // sc spacing in Hz
    std::vector<float> m_scFreqKHz; // CPU buffer to calculate and store sc frequency in KHz, precaculated to be used on GPU
    std::vector<float> m_firNzPw; // CPU buffer to calculate and store non-zero FIR coefficients
    std::vector<float> m_firNzDelayUs2Pi; // CPU buffer to calculate and store fir non-zero delays * -2 * pi
    std::vector<Tcomplex> m_firNzDelayScFreq2Pi; // CPU buffer to calculate and store fir non-zero delays in us * -2 * pi * sc frequency in Hz
    std::vector<uint16_t> m_firNzIdx; // CPU buffer to calculate and store non-zero FIR indexes
    uint16_t m_firNzLen; // number of non-zero coefficients
    float m_refTime0; // the time stamp for the start of tx symbol
    uint8_t m_runMode; // run mode, control frequency channel generation
    uint8_t m_procSigFreq; // proc tx sample mode
    uint8_t m_enableSwapTxRx; // use DL channel to process UL sample
    uint16_t m_procTxSampBlockSample; // sample per block for processing tx sample
    std::vector<Tcomplex> m_cfrBatchRotationCfo; // CFR rotation per batch due to CFO
    Tcomplex ** m_h_deviceFreqChanScPerLinkPtr; // per link freq chan on sc ptr saved on host; If not using ptr of ptr, the freq sc channel size is too large
    uint16_t m_N_sc;
    uint16_t m_N_sc_Prbg;
    uint8_t m_freqConvertType;
    uint8_t m_scSampling;
    uint8_t m_saveAntPairSample;
    
    uint64_t m_gpuDataUsageByte; // gpu memory usage in Byte
    // launch kernel drivers
    dim3 m_gridDim, m_blockDim;
    void *m_args[3];
    cudaFunction_t m_functionPtr;
};

// Explicitly instantiate the template to resovle "undefined functions"
template class fastFadingBaseChan<__half, __half2>;
template class fastFadingBaseChan<float, cuComplex>;

/**
 * @brief normalize the fast fading channel coes in time domain per TTI, controlled by macro ENABLE_NORMALIZATION_
 * 
 * @param fastFadingDynDescr fast fading dynamic descriptor
 *
 * @note can use thread shuffle to do fast parallel reduction but it requires Kelper arch
 * ref: https://developer.nvidia.com/blog/faster-parallel-reductions-kepler/
 */
 template <typename Tscalar, typename Tcomplex> 
 static __global__ void normalizeTimeChan(fastFadingDynDescr_t<Tscalar, Tcomplex> * fastFadingDynDescr);

/**
 * @brief CUDA kernel to process tx time data using the generated time channel 
 */
template <typename Tscalar, typename Tcomplex> 
static __global__ void processInputKernel_time(fastFadingDynDescr_t<Tscalar, Tcomplex> * fastFadingDynDescr, float refTime0, uint8_t enableSwapTxRx);

/**
 * @brief CUDA kernel to process tx time data using the generated freq channel on Sc, input is row major, output is always row major
 */
 template <typename Tscalar, typename Tcomplex> 
 static __global__ void processInputKernel_freq(fastFadingDynDescr_t<Tscalar, Tcomplex> * fastFadingDynDescr, float refTime0, uint8_t enableSwapTxRx);

 /**
 * @brief CUDA kernel to process tx time data using the generated freq channel on Sc, input is column major, output is always row major
 */
 template <typename Tscalar, typename Tcomplex> 
 static __global__ void processInputKernel_freq_columnMajor(fastFadingDynDescr_t<Tscalar, Tcomplex> * fastFadingDynDescr, float refTime0, uint8_t enableSwapTxRx);
 
/**
 * @brief CUDA kernel to generate freq channel on Sc based on the time channel
 */
template<typename FFT, typename Tscalar, typename Tcomplex>
//__launch_bounds__(FFT::max_threads_per_block)
static __global__ void fast_fading_fft_kernel(fastFadingDynDescr_t<Tscalar, Tcomplex> * fastFadingDynDescr, float refTime0);

/**
 * @brief CUDA kernel to generate freq channel on Prbg based on the time channel, no freq on Sc will be saved
 */
 template<typename FFT, typename Tscalar, typename Tcomplex>
 //__launch_bounds__(FFT::max_threads_per_block)
 static __global__ void fast_fading_fft_kernel_PrgbOnly(fastFadingDynDescr_t<Tscalar, Tcomplex> * fastFadingDynDescr, float refTime0);
 
template<typename Tscalar, typename Tcomplex>
using fftKernelHandle = void (*)(fastFadingDynDescr_t<Tscalar, Tcomplex> * fastFadingDynDescr);

// Choose FFT kernel
template<typename Tscalar, typename Tcomplex, unsigned int FftSize, unsigned int Arch>
fftKernelHandle<Tscalar, Tcomplex> fast_fading_get_fft_param(dim3& block_dim, uint& shared_memory_size, uint8_t& runMode);

 /**
 * @brief CUDA kernel to calculate freq chan on Prbg from freq chan on Sc
  * 
  * @tparam Tscalar scalar data type
  * @tparam Tcomplex complext data type
  * @param fastFadingDynDescr fast fading dynamic descriptor
  */
template<typename Tscalar, typename Tcomplex>
static __global__ void convertSctoPrbg(fastFadingDynDescr_t<Tscalar, Tcomplex> * fastFadingDynDescr);

/**
 * @brief geneerate freq channel on Prbg in runMode 1,
 *        may not need to calculate all Scs, no saving of CFR on Scs
 *        using direct sum of exp(-2*pi*...)
 * 
 * @tparam Tscalar scalar data type
 * @tparam Tcomplex complext data type
 * @param fastFadingDynDescr fast fading dynamic descriptor
 * @param refTime0 the time stamp for the start of tx signal
 */
template <typename Tscalar, typename Tcomplex> 
static __global__ void genFreqChanCoeKernel_runMode1(fastFadingDynDescr_t<Tscalar, Tcomplex> * fastFadingDynDescr, float refTime0);

/**
 * @brief geneerate freq channel on Prbg and Sc in runMode 2 or 3,
 *        need to calculate and save CFR on all Scs
 *        using direct sum of exp(-2*pi*...)
 * 
 * @tparam Tscalar scalar data type
 * @tparam Tcomplex complext data type
 * @param fastFadingDynDescr fast fading dynamic descriptor
 * @param refTime0 the time stamp for the start of tx signal
 */
template <typename Tscalar, typename Tcomplex> 
static __global__ void genFreqChanCoeKernel_runMode2(fastFadingDynDescr_t<Tscalar, Tcomplex> * fastFadingDynDescr, float refTime0);

/**
 * @brief calculate CFR on given frequency with CIR (coefficient and delays)
 * 
 * @tparam Tscalar scalar data type
 * @tparam Tcomplex complext data type
 * @param freqKHz frequency in kHz
 * @param firNzLen number of fir non-zeros taps
 * @param firNzDelayUs2Pi -2*pi*(fir non-zero delays)
 * @param cir fir coefficients
 * @return cuComplex CFR on the given freqKHz
 */
template <typename Tscalar, typename Tcomplex> 
static __device__ cuComplex calCfrbyCir(float freqKHz, uint16_t firNzLen, float * firNzDelayUs2Pi, Tcomplex * cir, float cfrPhaseShift);

/**
 * @brief calculate CFR on given frequency with CIR (coefficient and delays)
 * faster than the above kernel with more GPU memory usage
 * 
 * @tparam Tscalar scalar data type
 * @tparam Tcomplex complext data type
 * @param firNzLen number of fir non-zeros taps
 * @param firNzDelayScFreq2Pi fir non-zero delays in us * -2 * pi * sc frequency in Hz
 * @param cir fir coefficients
 * @return cuComplex CFR based on the given firNzDelayScFreq2Pi
 */

template <typename Tscalar, typename Tcomplex> 
static __device__ cuComplex calCfrbyCir_v2(uint16_t firNzLen, Tcomplex * firNzDelayScFreq2Pi, Tcomplex * cir);
 
#endif // !defined(FAST_FADING_COMMON_CUH_INCLUDED_)