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

#include "fastFadingCommon.cuh"
#include <cufftdx.hpp> // for cuFFTdx library

template <typename Tscalar, typename Tcomplex>
fastFadingBaseChan<Tscalar, Tcomplex>::fastFadingBaseChan()
{
    m_gpuDataUsageByte = 0;  // initialize GPU memory usage in Byte
    // allocate dynamic descriptor
    m_fastFadingDynDescrCpu = new fastFadingDynDescr_t<Tscalar, Tcomplex>;
    CHECK_CUDAERROR(cudaMalloc((void**)&(m_fastFadingDynDescrGpu), sizeof(fastFadingDynDescr_t<Tscalar, Tcomplex>)));
    m_gpuDataUsageByte += sizeof(fastFadingDynDescr_t<Tscalar, Tcomplex>);
}

template <typename Tscalar, typename Tcomplex>
fastFadingBaseChan<Tscalar, Tcomplex>::~fastFadingBaseChan()
{
    cudaFree(m_fastFadingDynDescrCpu -> batchCumLen);
    cudaFree(m_fastFadingDynDescrCpu -> tBatch);
    cudaFree(m_fastFadingDynDescrCpu -> firNzPw);
    cudaFree(m_fastFadingDynDescrCpu -> firNzIdx);
    cudaFree(m_fastFadingDynDescrCpu -> timeChan);

    if(m_sigLenPerAnt)
    {
        cudaFree(m_fastFadingDynDescrCpu -> rxSigOut);
    }

    if (m_fastFadingDynDescrCpu -> saveAntPairSample == 1)
    {
        cudaFree(m_fastFadingDynDescrCpu -> rxSigOutPerAntPair);
    }

    if(m_runMode == 1)
    {
        cudaFree(m_fastFadingDynDescrCpu -> freqChanPrbg);
        cudaFree(m_fastFadingDynDescrCpu -> scFreqKHz);
        cudaFree(m_fastFadingDynDescrCpu -> firNzDelayUs2Pi);
        cudaFree(m_fastFadingDynDescrCpu -> firNzDelayScFreq2Pi);
        cudaFree(m_fastFadingDynDescrCpu -> cfrBatchRotationCfo);
    }
    else if(m_runMode == 2)
    {
        for(uint32_t linkIdx = 0; linkIdx < m_nLink; linkIdx ++)
        {
            cudaFree(m_h_deviceFreqChanScPerLinkPtr[linkIdx]);
        }
        delete m_h_deviceFreqChanScPerLinkPtr;
        cudaFree(m_fastFadingDynDescrCpu -> freqChanSc);
        cudaFree(m_fastFadingDynDescrCpu -> freqChanPrbg);
        cudaFree(m_fastFadingDynDescrCpu -> scFreqKHz);
        cudaFree(m_fastFadingDynDescrCpu -> firNzDelayUs2Pi);
        cudaFree(m_fastFadingDynDescrCpu -> firNzDelayScFreq2Pi);
        cudaFree(m_fastFadingDynDescrCpu -> cfrBatchRotationCfo);
    }

    cudaFree(m_fastFadingDynDescrGpu);
    delete m_fastFadingDynDescrCpu;
}

template <typename Tscalar, typename Tcomplex>
void fastFadingBaseChan<Tscalar, Tcomplex>::calCfrRelateParam()
{
    uint16_t & N_sc = m_N_sc;
    // for CFR shift due to delay
    m_fastFadingDynDescrCpu -> cfrPhaseShiftTimeDelay = - 2.0f * M_PI * m_fastFadingDynDescrCpu -> nDelaySample / m_fastFadingDynDescrCpu -> N_FFT;
    m_cfrBatchRotationCfo.resize(m_nBatch);
    for (uint16_t batchIdx = 0; batchIdx < m_nBatch; batchIdx ++)
    {
        m_cfrBatchRotationCfo[batchIdx].x = cos(2.0f * M_PI * m_batchCumLen[batchIdx] * m_fastFadingDynDescrCpu -> cfoHz / m_f_samp);
        m_cfrBatchRotationCfo[batchIdx].y = sin(2.0f * M_PI * m_batchCumLen[batchIdx] * m_fastFadingDynDescrCpu -> cfoHz / m_f_samp);
    }
    CHECK_CUDAERROR(cudaMalloc((void**)&(m_fastFadingDynDescrCpu -> cfrBatchRotationCfo), sizeof(Tcomplex) * m_nBatch));
    m_gpuDataUsageByte += sizeof(Tcomplex) * m_nBatch;
    cudaMemcpyAsync(m_fastFadingDynDescrCpu -> cfrBatchRotationCfo, m_cfrBatchRotationCfo.data(), sizeof(Tcomplex) * m_nBatch, cudaMemcpyHostToDevice, m_strm);

#if CAL_COS_SIN_IN_GPU
    m_firNzDelayUs2Pi.resize(m_firNzLen);
    m_fastFadingDynDescrCpu -> firNzDelayScFreq2Pi = nullptr;
    // save delays*(-2*pi), dim: m_firNzLen
    for(int firNzIdx = 0; firNzIdx<m_firNzLen; firNzIdx ++)
    {
        m_firNzDelayUs2Pi[firNzIdx] = m_firNzIdx[firNzIdx] / (m_f_samp / 1e6) * (-2.0f) * M_PI;
    }
    CHECK_CUDAERROR(cudaMalloc((void**)&(m_fastFadingDynDescrCpu -> firNzDelayUs2Pi), sizeof(float) * m_firNzLen));
    m_gpuDataUsageByte += sizeof(float) * m_firNzLen;
    cudaMemcpyAsync(m_fastFadingDynDescrCpu -> firNzDelayUs2Pi, m_firNzDelayUs2Pi.data(), sizeof(float) * m_firNzLen, cudaMemcpyHostToDevice, m_strm);

    // sc frequency in KHz, dim: N_sc
    m_scFreqKHz.resize(N_sc);
    for (int scIdx = -N_sc/2 ; scIdx < N_sc/2 ; scIdx++)
    {
        m_scFreqKHz[scIdx + N_sc/2] = (scIdx * m_scSpacingHz - m_fastFadingDynDescrCpu -> cfoHz) * 1e-3;
    }
    CHECK_CUDAERROR(cudaMalloc((void**) &(m_fastFadingDynDescrCpu -> scFreqKHz), sizeof(float) * N_sc));
    m_gpuDataUsageByte += sizeof(float) * N_sc;
    CHECK_CUDAERROR(cudaMemcpyAsync(m_fastFadingDynDescrCpu -> scFreqKHz, m_scFreqKHz.data(),  sizeof(float) * N_sc, cudaMemcpyHostToDevice, m_strm));
#else
    m_fastFadingDynDescrCpu -> firNzDelayUs2Pi = nullptr;
    m_fastFadingDynDescrCpu -> scFreqKHz = nullptr;
    // exp(-2*pi*non-zero delay * scfrequency), dim: N_sc * firNzLen
    m_firNzDelayScFreq2Pi.resize(N_sc * m_firNzLen);
    for (int scIdx = -N_sc/2 ; scIdx < N_sc/2 ; scIdx ++)
    {
        cuComplex cfrPhaseShiftTimeDelay;
        cfrPhaseShiftTimeDelay.x = cos(scIdx * m_fastFadingDynDescrCpu -> cfrPhaseShiftTimeDelay);
        cfrPhaseShiftTimeDelay.y = sin(scIdx * m_fastFadingDynDescrCpu -> cfrPhaseShiftTimeDelay);
        for(int firNzIdx = 0; firNzIdx < m_firNzLen; firNzIdx ++)
        {
            uint32_t saveLoc = (scIdx + N_sc/2) * m_firNzLen + firNzIdx;
            float currentScFreqHz = scIdx * m_scSpacingHz - m_fastFadingDynDescrCpu -> cfoHz;
            cuComplex firNzDelayScFreq2Pi;
            firNzDelayScFreq2Pi.x = cos(currentScFreqHz * m_firNzIdx[firNzIdx] / m_f_samp * (-2.0f) * M_PI);
            firNzDelayScFreq2Pi.y = sin(currentScFreqHz * m_firNzIdx[firNzIdx] / m_f_samp * (-2.0f) * M_PI);

            m_firNzDelayScFreq2Pi[saveLoc].x = firNzDelayScFreq2Pi.x * cfrPhaseShiftTimeDelay.x - firNzDelayScFreq2Pi.y * cfrPhaseShiftTimeDelay.y;
            m_firNzDelayScFreq2Pi[saveLoc].y = firNzDelayScFreq2Pi.x * cfrPhaseShiftTimeDelay.y + firNzDelayScFreq2Pi.y * cfrPhaseShiftTimeDelay.x;
        }
    }
    CHECK_CUDAERROR(cudaMalloc((void**) &(m_fastFadingDynDescrCpu -> firNzDelayScFreq2Pi), sizeof(Tcomplex) * N_sc * m_firNzLen));
    m_gpuDataUsageByte += sizeof(Tcomplex) * N_sc * m_firNzLen;
    CHECK_CUDAERROR(cudaMemcpyAsync(m_fastFadingDynDescrCpu -> firNzDelayScFreq2Pi, m_firNzDelayScFreq2Pi.data(), sizeof(Tcomplex) * N_sc * m_firNzLen, cudaMemcpyHostToDevice, m_strm));
#endif
}


template <typename Tscalar, typename Tcomplex>
void fastFadingBaseChan<Tscalar, Tcomplex>::genChanProcSig(float refTime0, uint8_t enableSwapTxRx, uint8_t txColumnMajorInd)
{
    if (m_refTime0 != refTime0) // different ref time, need to regenerate channnel
    {
        m_refTime0 = refTime0;
        genTimeChan();
        if(m_runMode > 0 && m_runMode < 3) // freq channel required on Sc or/and Prbg
        {
            genFreqChan();
        }
    }

    if(m_sigLenPerAnt) // has input signal
    {
        m_enableSwapTxRx = enableSwapTxRx;
        processTxSig(txColumnMajorInd);
    }
    cudaStreamSynchronize(m_strm);
}

template <typename Tscalar, typename Tcomplex>
void fastFadingBaseChan<Tscalar, Tcomplex>::setup()
{
    if (m_runMode > 2)
    {
        fprintf(stderr, "Error: unsupported runMode %d\n", m_runMode);
        exit(EXIT_FAILURE);
    }

    // copy params to m_fastFadingDynDescrCpu
    m_fastFadingDynDescrCpu -> nLink  = m_nLink;
    m_fastFadingDynDescrCpu -> nBsAnt = m_nBsAnt;
    m_fastFadingDynDescrCpu -> nUeAnt = m_nUeAnt;
    float tSample =  1.0f / (m_f_samp);
    m_fastFadingDynDescrCpu -> cfoHz = m_cfoHz;
    m_fastFadingDynDescrCpu -> cfoPhaseSamp = tSample * 2 * M_PI * m_cfoHz;
    m_fastFadingDynDescrCpu -> maxDopplerShift = m_maxDopplerShift;
    m_fastFadingDynDescrCpu -> nDelaySample = roundf(m_delay / tSample);
    m_fastFadingDynDescrCpu -> firNzLen = m_firNzLen;
    m_fastFadingDynDescrCpu -> firMaxLen = m_firNzIdx.back() + 1;

    // copy nz FIR to GPU
    CHECK_CUDAERROR(cudaMalloc((void**)&(m_fastFadingDynDescrCpu -> firNzPw), sizeof(float) * m_firNzPw.size()));
    m_gpuDataUsageByte += sizeof(float) * m_firNzPw.size();
    CHECK_CUDAERROR(cudaMalloc((void**)&(m_fastFadingDynDescrCpu -> firNzIdx), sizeof(uint16_t) * m_firNzPw.size()));
    m_gpuDataUsageByte += sizeof(uint16_t) * m_firNzPw.size();

    cudaMemcpyAsync(m_fastFadingDynDescrCpu -> firNzPw, m_firNzPw.data(), sizeof(float) * m_firNzPw.size(), cudaMemcpyHostToDevice, m_strm);
    cudaMemcpyAsync(m_fastFadingDynDescrCpu -> firNzIdx, m_firNzIdx.data(), sizeof(uint16_t) * m_firNzPw.size(), cudaMemcpyHostToDevice, m_strm);

    // config the batch
    if (m_batchLen.empty()) // using same number of samples per batch configured by fBatch
    {
        uint32_t nBatchSamp = round(m_f_samp / m_fBatch);
        m_nBatch = max(1, (m_sigLenPerAnt + nBatchSamp - 1) / nBatchSamp);
        m_batchCumLen.resize(m_nBatch + 1);
        m_tBatch.resize(m_nBatch);
        m_batchCumLen[0] = 0;
        for (uint16_t batchIdx = 0; batchIdx < m_nBatch; batchIdx++)
        {
            m_batchCumLen[batchIdx + 1] = min(m_sigLenPerAnt, m_batchCumLen[batchIdx] + nBatchSamp);
            m_tBatch[batchIdx] = tSample * m_batchCumLen[batchIdx];
        }
    }
    else
    {
        m_nBatch = m_batchLen.size();// nBatch from config should be greater than 0
        m_batchCumLen.resize(m_nBatch + 1);
        m_tBatch.resize(m_nBatch);
        m_batchCumLen[0] = 0;
        for (uint16_t batchIdx = 0; batchIdx < m_nBatch; batchIdx++)
        {
            m_batchCumLen[batchIdx + 1] = m_batchCumLen[batchIdx] + m_batchLen[batchIdx];
            m_tBatch[batchIdx] = tSample * m_batchCumLen[batchIdx];
        }
    }
    m_fastFadingDynDescrCpu -> nBatch = m_nBatch;
    CHECK_CUDAERROR(cudaMalloc((void**)&(m_fastFadingDynDescrCpu -> batchCumLen), sizeof(uint32_t) * (m_nBatch + 1)));
    m_gpuDataUsageByte += sizeof(uint32_t) * (m_nBatch + 1);
    cudaMemcpyAsync(m_fastFadingDynDescrCpu -> batchCumLen, m_batchCumLen.data(), sizeof(uint32_t) * (m_nBatch + 1), cudaMemcpyHostToDevice, m_strm);

    CHECK_CUDAERROR(cudaMalloc((void**)&(m_fastFadingDynDescrCpu -> tBatch), sizeof(float) * m_nBatch));
    m_gpuDataUsageByte += sizeof(float) * m_nBatch;
    cudaMemcpyAsync(m_fastFadingDynDescrCpu -> tBatch, m_tBatch.data(), sizeof(float) * m_nBatch, cudaMemcpyHostToDevice, m_strm);

    /*-------------------   setup channel input and output buffers   -------------------*/
    m_sigLenTx     = m_nLink * m_nBsAnt * m_sigLenPerAnt;
    m_sigLenRx     = m_nLink * m_nUeAnt * m_sigLenPerAnt;
    if(m_sigLenPerAnt)
    {
        // find the block dimension for process input signals
        if (m_procSigFreq == 0)  // processing in time domain
        {
            // {m_procTxSampBlockSample, 1024/m_procTxSampBlockSample}
            m_procTxSampBlockSample = 1024;
            for(uint16_t tmp = 1; tmp < m_firNzLen; tmp *=2)
            {
                m_procTxSampBlockSample /= 2;
            }
        }
        else  // processing in frequency domain
        {
            m_procTxSampBlockSample = 1024;  // fixed number of threads per block
        }
    }
    else
    {
        m_procTxSampBlockSample = 0;
    }

    m_fastFadingDynDescrCpu -> sigLenPerAnt = m_sigLenPerAnt;
    m_fastFadingDynDescrCpu -> procTxSampBlockSample = m_procTxSampBlockSample;
    if(m_sigLenPerAnt) // need to perform tx signal processing
    {
        m_fastFadingDynDescrCpu -> txSigIn = reinterpret_cast<Tcomplex*>(m_txSigIn);
        // proc in time domain, does not support swap tx and rx yet
        // proc in freq domain, support tx and rx swap but may be inaccurate when CFO is present
        if (m_procSigFreq == 1)
        {
            ASSERT(m_sigLenPerAnt == m_N_sc * m_nBatch, "processing signal in frequency domain requries sigLenPerAnt equal to N_sc * nBatch");
        }
        CHECK_CUDAERROR(cudaMalloc((void**) &(m_fastFadingDynDescrCpu -> rxSigOut), sizeof(Tcomplex) * m_nLink * m_sigLenPerAnt * std::max(m_nBsAnt, m_nUeAnt)));
        m_gpuDataUsageByte += sizeof(Tcomplex) * m_nLink * m_sigLenPerAnt * std::max(m_nBsAnt, m_nUeAnt);
    }
    else
    {
        m_fastFadingDynDescrCpu -> txSigIn = nullptr;
        m_fastFadingDynDescrCpu -> rxSigOut = nullptr;
    }

    // buffer for output channel
    m_timeChanSizePerLink = m_nBatch * m_nBsAnt * m_nUeAnt * m_firNzLen;
    m_timeChanSize        = m_timeChanSizePerLink * m_nLink;
    m_fastFadingDynDescrCpu -> timeChanSizePerLink = m_timeChanSizePerLink;
    m_fastFadingDynDescrCpu -> timeChanSize        = m_timeChanSize;
    CHECK_CUDAERROR(cudaMalloc((void**)&(m_fastFadingDynDescrCpu -> timeChan), sizeof(Tcomplex) * m_timeChanSize));
    m_gpuDataUsageByte += sizeof(Tcomplex) * m_timeChanSize;


    // freq channel coefficient, only used when m_runMode=1 or 2
    m_fastFadingDynDescrCpu -> N_sc = m_N_sc;
    m_fastFadingDynDescrCpu -> N_sc_Prbg = m_N_sc_Prbg;
    m_fastFadingDynDescrCpu -> N_Prbg = (m_N_sc + m_N_sc_Prbg - 1)/m_N_sc_Prbg;
    m_fastFadingDynDescrCpu -> N_sc_last_Prbg = m_N_sc - m_N_sc_Prbg * (m_fastFadingDynDescrCpu -> N_Prbg - 1);
    m_fastFadingDynDescrCpu -> freqConvertType = m_freqConvertType;
    m_fastFadingDynDescrCpu -> scSampling = m_scSampling;
    m_fastFadingDynDescrCpu -> N_FFT = 4096;
    switch(m_runMode)
    {
        case 0:
            m_freqChanScSizePerLink = 0;
            m_freqChanPrbgSize = 0;
            m_scFreqKHz.resize(0);
            m_firNzDelayUs2Pi.resize(0);
            m_firNzDelayScFreq2Pi.resize(0);
            m_fastFadingDynDescrCpu -> scFreqKHz = nullptr;
            m_fastFadingDynDescrCpu -> firNzDelayUs2Pi = nullptr;
            m_fastFadingDynDescrCpu -> firNzDelayScFreq2Pi = nullptr;
            m_fastFadingDynDescrCpu -> freqChanSc = nullptr;
            m_fastFadingDynDescrCpu -> freqChanPrbg = nullptr;
            break;
        case 1:
            calCfrRelateParam();
            m_fastFadingDynDescrCpu -> inverseNScPrbg = 1.0f / (m_N_sc_Prbg / m_fastFadingDynDescrCpu -> scSampling); // for avg over Sc
            m_fastFadingDynDescrCpu -> inverseNScLastPrbg = 1.0f / (m_fastFadingDynDescrCpu -> N_sc_last_Prbg / m_fastFadingDynDescrCpu -> scSampling); // for avg over Sc
            m_freqChanScSizePerLink = 0;
            m_freqChanPrbgSize = m_nBatch * m_nLink * m_nBsAnt * m_nUeAnt * (m_fastFadingDynDescrCpu -> N_Prbg);

            m_h_deviceFreqChanScPerLinkPtr = nullptr;
            m_fastFadingDynDescrCpu -> freqChanSc = nullptr;
            CHECK_CUDAERROR(cudaMalloc((void**) &(m_fastFadingDynDescrCpu -> freqChanPrbg), sizeof(Tcomplex) * m_freqChanPrbgSize));
            m_gpuDataUsageByte += sizeof(Tcomplex) * m_freqChanPrbgSize;
            break;
        case 2:
            m_fastFadingDynDescrCpu -> N_FFT = 4096;
            calCfrRelateParam();
            m_fastFadingDynDescrCpu -> inverseNScPrbg = 1.0f / (m_N_sc_Prbg / m_fastFadingDynDescrCpu -> scSampling); // for avg over Sc
            m_fastFadingDynDescrCpu -> inverseNScLastPrbg = 1.0f / (m_fastFadingDynDescrCpu -> N_sc_last_Prbg / m_fastFadingDynDescrCpu -> scSampling); // for avg over Sc
            m_freqChanScSizePerLink = m_nBatch * m_nBsAnt * m_nUeAnt * m_N_sc;
            m_freqChanPrbgSize = m_nLink * m_nBatch * m_nBsAnt * m_nUeAnt * (m_fastFadingDynDescrCpu -> N_Prbg);

            CHECK_CUDAERROR(cudaMalloc((void**) &(m_fastFadingDynDescrCpu -> freqChanSc), m_nLink * sizeof(Tcomplex*)));
            m_gpuDataUsageByte += m_nLink * sizeof(Tcomplex*);
            m_h_deviceFreqChanScPerLinkPtr = (Tcomplex **)malloc(m_nLink * sizeof(Tcomplex*));
            for(uint32_t linkIdx = 0; linkIdx < m_nLink; linkIdx ++)
            {
                CHECK_CUDAERROR(cudaMalloc((void**) &(m_h_deviceFreqChanScPerLinkPtr[linkIdx]), sizeof(Tcomplex) * m_freqChanScSizePerLink));
                m_gpuDataUsageByte += sizeof(Tcomplex) * m_freqChanScSizePerLink;
            }
            CHECK_CUDAERROR(cudaMemcpy(m_fastFadingDynDescrCpu -> freqChanSc, m_h_deviceFreqChanScPerLinkPtr, m_nLink * sizeof(Tcomplex*), cudaMemcpyHostToDevice));

            CHECK_CUDAERROR(cudaMalloc((void**) &(m_fastFadingDynDescrCpu -> freqChanPrbg), sizeof(Tcomplex) * m_freqChanPrbgSize));
            m_gpuDataUsageByte += sizeof(Tcomplex) * m_freqChanPrbgSize;
            break;
        default:
            fprintf(stderr, "Error: unsupported run mode %d!", m_runMode);
            exit(EXIT_FAILURE);
    }

    // enable saving per rx-tx antenna pair rx samples
    m_fastFadingDynDescrCpu -> saveAntPairSample = m_saveAntPairSample;
    if (m_fastFadingDynDescrCpu -> saveAntPairSample == 1)
    {
        ASSERT(m_nLink * m_nUeAnt * m_nBsAnt * m_sigLenPerAnt < 4294967296ull, "size of rx samples per antenna pair for all links must be smaller than 4294967296 (2^32)");

        CHECK_CUDAERROR(cudaMalloc((void**) &(m_fastFadingDynDescrCpu -> rxSigOutPerAntPair), sizeof(Tcomplex) * m_nLink * m_nUeAnt * m_nBsAnt * m_sigLenPerAnt));
        m_gpuDataUsageByte += sizeof(Tcomplex) * m_nLink * m_nUeAnt * m_nBsAnt * m_sigLenPerAnt;
    }
    else
    {
        m_fastFadingDynDescrCpu -> rxSigOutPerAntPair = nullptr;
    }

    // assert failure if input network size is too large
    ASSERT(m_nBatch < 65536u, "number of batches must be smaller than 65536 (2^16)");
    ASSERT(m_timeChanSize < 4294967296ull, "size of time channel for all links must be smaller than 4294967296 (2^32)");
    ASSERT(m_sigLenTx < 4294967296ull, "size of tx samples for all links must be smaller than 4294967296 (2^32)");
    ASSERT(m_sigLenRx < 4294967296ull, "size of tx samples for all links must be smaller than 4294967296 (2^32)");
    ASSERT(m_freqChanPrbgSize < 4294967296ull, "size of freq channel on PRBG for all links must be smaller than 4294967296 (2^32)");
    ASSERT(m_freqChanScSizePerLink < 4294967296ull, "size of freq channel on SC per link must be smaller than 4294967296 (2^32)");

    // launch dimensions in genTimeChan() will be (firNzLen, m_nBsAnt/m_scaleBsAntTimeChan, m_nUeAnt/m_scaleBsAntTimeChan) scale to handle mMIMO
    findGenFastFadingChanKernelDim(m_firNzPw.size(), m_nBsAnt, m_nUeAnt, m_scaleBsAntTimeChan, m_scaleUeAntTimeChan);
    // launch dimensions in genFreqChan() will be (N_Prbg, m_nBsAnt/m_scaleBsAntFreqChan, m_nUeAnt/m_scaleUeAntFreqChan) scale to handle mMIMO
    findGenFastFadingChanKernelDim(m_fastFadingDynDescrCpu -> N_Prbg, m_nBsAnt, m_nUeAnt, m_scaleBsAntFreqChan, m_scaleUeAntFreqChan);

    //copy dyndescriptor to GPU
    copyDescriptor();

    // get kernel inputs
    m_refTime0 = -1.0f;  // invalid time
    m_enableSwapTxRx = 0;
    m_args[0] = &m_fastFadingDynDescrGpu;
    m_args[1] = &m_refTime0;
    m_args[2] = &m_enableSwapTxRx;  // can be replaced by other param
}

template <typename Tscalar, typename Tcomplex>
inline void fastFadingBaseChan<Tscalar, Tcomplex>::findGenFastFadingChanKernelDim(const uint16_t & firNzLen, const uint16_t& nUeAnt, const uint16_t& nBsAnt, uint16_t & scaleUeAnt, uint16_t & scaleBsAnt)
{
    for (scaleUeAnt = 1; scaleUeAnt <= nUeAnt; ++scaleUeAnt)
    {
        for (scaleBsAnt = 1; scaleBsAnt <= nBsAnt; ++scaleBsAnt)
        {
            if ( (nUeAnt % scaleUeAnt == 0) && (nBsAnt % scaleBsAnt == 0) && firNzLen*nUeAnt*nBsAnt < 1024*scaleUeAnt*scaleBsAnt)
            {
                return;
            }
        }
    }
}


template <typename Tscalar, typename Tcomplex>
void fastFadingBaseChan<Tscalar, Tcomplex>::processTxSig(uint8_t txColumnMajorInd)
{
    if (m_procSigFreq == 0)  // process tx samples in time domain
    {
        ASSERT(txColumnMajorInd == 0, "processing tx samples in time domain only supports row-major input");
        m_gridDim = {m_nLink, m_nBatch, 1};
        m_blockDim = {m_procTxSampBlockSample, min(m_enableSwapTxRx ? m_nBsAnt : m_nUeAnt, uint32_t(1024 / m_procTxSampBlockSample)), 1};
        // dynamic shared memory size
        uint32_t shareMemBytes = sizeof(Tcomplex) * m_firNzLen * (m_enableSwapTxRx ? m_nBsAnt : m_nUeAnt);
        // __shared__ extern Tcomplex chanCoeLocal[]; // m_firNzLen * blockDim.y

        // static shared memory of uint16_t * MAX_NZ_TAPS_ for
        // __shared__  uint16_t firNzIdx[MAX_NZ_TAPS_];

        cudaGetFuncBySymbol(&m_functionPtr, reinterpret_cast<void*>(processInputKernel_time<Tscalar, Tcomplex>));
        CUresult status = cuLaunchKernel(m_functionPtr, m_gridDim.x, m_gridDim.y, m_gridDim.z, m_blockDim.x, m_blockDim.y, m_blockDim.z, shareMemBytes, m_strm, m_args, nullptr);
        CHECK_CURESULT(status);
    }
    else  // process tx samples in freq domain
    {
        m_gridDim = {m_nLink, m_nBatch, 1};
        m_blockDim = {m_procTxSampBlockSample, 1, 1};

        if (txColumnMajorInd)
        {
            cudaGetFuncBySymbol(&m_functionPtr, reinterpret_cast<void*>(processInputKernel_freq_columnMajor<Tscalar, Tcomplex>));
        }
        else
        {
            cudaGetFuncBySymbol(&m_functionPtr, reinterpret_cast<void*>(processInputKernel_freq<Tscalar, Tcomplex>));
        }
        CUresult status = cuLaunchKernel(m_functionPtr, m_gridDim.x, m_gridDim.y, m_gridDim.z, m_blockDim.x, m_blockDim.y, m_blockDim.z, 0 /*no shared memory*/, m_strm, m_args, nullptr);
        CHECK_CURESULT(status);
    }
}

template <typename Tscalar, typename Tcomplex>
void fastFadingBaseChan<Tscalar, Tcomplex>::printTimeChan(uint16_t cid, uint16_t uid, int printLen)
{
    Tcomplex * tempCpuBuffer = new Tcomplex[printLen];
    cudaMemcpyAsync(tempCpuBuffer, m_fastFadingDynDescrCpu -> timeChan + (cid*m_nUe + uid)*m_timeChanSizePerLink , sizeof(Tcomplex)*printLen, cudaMemcpyDeviceToHost, m_strm);
    cudaStreamSynchronize(m_strm);

    for(int chanIdx=0; chanIdx<printLen; chanIdx++)
    {
        printf("chanIdx %d: %f + %f i \n", chanIdx, static_cast<float>(tempCpuBuffer[chanIdx].x), static_cast<float>(tempCpuBuffer[chanIdx].y));
    }
    printf("Done print time channel! \n");
    delete[] tempCpuBuffer;
}

template <typename Tscalar, typename Tcomplex>
void fastFadingBaseChan<Tscalar, Tcomplex>::printFreqScChan(uint16_t cid, uint16_t uid, int printLen)
{
    if(m_runMode != 2)
    {
        printf("Warning: freq channel on subcarriers not avaiable! \n");
        return;
    }

    Tcomplex * tempCpuBuffer = new Tcomplex[printLen];
    cudaMemcpyAsync(tempCpuBuffer, m_h_deviceFreqChanScPerLinkPtr[cid*m_nUe + uid], sizeof(Tcomplex)*printLen, cudaMemcpyDeviceToHost, m_strm);
    cudaStreamSynchronize(m_strm);

    for(int chanIdx=0; chanIdx<printLen; chanIdx++)
    {
        printf("chanIdx %d: %f + %f i \n", chanIdx, static_cast<float>(tempCpuBuffer[chanIdx].x), static_cast<float>(tempCpuBuffer[chanIdx].y));
    }
    printf("Done print freq channel on subcarriers! \n");
    delete[] tempCpuBuffer;
}

template <typename Tscalar, typename Tcomplex>
void fastFadingBaseChan<Tscalar, Tcomplex>::printFreqPrbgChan(uint16_t cid, uint16_t uid, int printLen)
{
    if(m_runMode != 1 && m_runMode != 2)
    {
        printf("Warning: freq channel on Prbgs not avaiable! \n");
        return;
    }

    Tcomplex * tempCpuBuffer = new Tcomplex[printLen];
    cudaMemcpyAsync(tempCpuBuffer, m_fastFadingDynDescrCpu -> freqChanPrbg + (cid*m_nUe + uid)*m_freqChanPrbgSize/m_nLink, sizeof(Tcomplex)*printLen, cudaMemcpyDeviceToHost, m_strm);
    cudaStreamSynchronize(m_strm);

    for(int chanIdx=0; chanIdx<printLen; chanIdx++)
    {
        printf("chanIdx %d: %f + %f i \n", chanIdx, static_cast<float>(tempCpuBuffer[chanIdx].x), static_cast<float>(tempCpuBuffer[chanIdx].y));
    }
    printf("Done print freq channel on Prbgs! \n");
    delete[] tempCpuBuffer;
}

template <typename Tscalar, typename Tcomplex>
void fastFadingBaseChan<Tscalar, Tcomplex>::copyDescriptor()
{
    cudaMemcpyAsync(m_fastFadingDynDescrGpu, m_fastFadingDynDescrCpu,
                    sizeof(fastFadingDynDescr_t<Tscalar, Tcomplex>),
                    cudaMemcpyHostToDevice, m_strm);
}

template <typename Tscalar, typename Tcomplex>
void fastFadingBaseChan<Tscalar, Tcomplex>::setTxSigIn(Tcomplex* txSigIn)
{
    // Update CPU descriptor
    m_fastFadingDynDescrCpu->txSigIn = txSigIn;

    // Copy descriptor to GPU
    copyDescriptor();
}

template <typename Tscalar, typename Tcomplex>
void fastFadingBaseChan<Tscalar, Tcomplex>::printSig(uint16_t cid, uint16_t uid, int printLen)
{
    Tcomplex * tempCpuBuffer = new Tcomplex[printLen];

    // print tx time singal
    cudaMemcpyAsync(tempCpuBuffer, m_fastFadingDynDescrCpu -> txSigIn + (cid*m_nUe + uid)*(m_enableSwapTxRx ? m_nUeAnt : m_nBsAnt) *m_sigLenPerAnt, sizeof(Tcomplex)*printLen, cudaMemcpyDeviceToHost, m_strm);
    cudaStreamSynchronize(m_strm);

    for(int rxSigIdx=0; rxSigIdx<printLen; rxSigIdx++)
    {
        printf("txSigIdx %d: %f + %f i \n", rxSigIdx, static_cast<float>(tempCpuBuffer[rxSigIdx].x), static_cast<float>(tempCpuBuffer[rxSigIdx].y));
    }
    printf("Done print tx time out signal! \n");

    // print rx time singal
    cudaMemcpyAsync(tempCpuBuffer, m_fastFadingDynDescrCpu -> rxSigOut + (cid*m_nUe + uid)*(m_enableSwapTxRx ? m_nBsAnt : m_nUeAnt)*m_sigLenPerAnt, sizeof(Tcomplex)*printLen, cudaMemcpyDeviceToHost, m_strm);
    cudaStreamSynchronize(m_strm);

    for(int rxSigIdx=0; rxSigIdx<printLen; rxSigIdx++)
    {
        printf("rxSigIdx %d: %f + %f i \n", rxSigIdx, static_cast<float>(tempCpuBuffer[rxSigIdx].x), static_cast<float>(tempCpuBuffer[rxSigIdx].y));
    }
    printf("Done print rx time out signal! \n");
    delete[] tempCpuBuffer;
}

template <typename Tscalar, typename Tcomplex>
void fastFadingBaseChan<Tscalar, Tcomplex>::genFreqChan()
// generate frequency domain chanel for Sc and/or Prbg
{
#if USE_FFT_CAL_CFR
    if (fastFadingDynDescr -> freqConvertType == 4)
    {
        printf("Error: Using FFT to calculate CFR does not support freqConvertType 4!\n");
        exit(EXIT_FAILURE);
    }
    // set up FFT kernel
    uint32_t shared_memory_size = 0;
    ASSERT(mod(m_nUeAnt * m_nBsAnt, FFTs_PER_BLOCK_CONST_) == 0, "m_nUeAnt * m_nBsAnt must be divisble by FFTs_PER_BLOCK_CONST_");
    dim3 gridDim = dim3(m_nLink, m_nBatch, m_nUeAnt * m_nBsAnt / FFTs_PER_BLOCK_CONST_);
    dim3 blockDim; // auto set by cuFFTdx
    const uint32_t cudaDeviceArch = getCudaDeviceArch();
    auto kernelPtr = fast_fading_get_fft_param<Tscalar, Tcomplex>( m_fastFadingDynDescrCpu -> N_FFT, cudaDeviceArch, blockDim, shared_memory_size, m_runMode);
    // launch kernel for freq domain channel per Sc
    cudaFunction_t functionPtr;
    if(m_runMode == 1) // only generate freq channel on Prbg
    {
        shared_memory_size = max(shared_memory_size, static_cast<uint32_t>((m_fastFadingDynDescrCpu -> N_sc)*sizeof(Tcomplex)*FFTs_PER_BLOCK_CONST_)); // ensure conversion of Sc to Prgb has sufficient shared memory
        cudaGetFuncBySymbol(&functionPtr, reinterpret_cast<void*>(kernelPtr));
        CUresult status = cuLaunchKernel(functionPtr, gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z, shared_memory_size, m_strm, m_args, nullptr);
        CHECK_CURESULT(status);
    }
    else if(m_runMode == 2) // generate frequence channel on Sc and Prbg
    {
        cudaGetFuncBySymbol(&functionPtr, reinterpret_cast<void*>(kernelPtr));
        CUresult status = cuLaunchKernel(functionPtr, gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z, shared_memory_size, m_strm, m_args, nullptr);
        CHECK_CURESULT(status);

        // convert freq channel from Sc to Prbg
        gridDim = dim3(m_nLink, m_scaleUeAntFreqChan, m_scaleBsAntFreqChan);
        blockDim = dim3(m_fastFadingDynDescrCpu -> N_Prbg, uint(m_nBsAnt / m_scaleBsAntFreqChan), uint(m_nUeAnt / m_scaleUeAntFreqChan));
        kernelPtr = convertSctoPrbg<Tscalar, Tcomplex>;
        cudaGetFuncBySymbol(&functionPtr, reinterpret_cast<void*>(kernelPtr));
        status = cuLaunchKernel(functionPtr, gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z, 0, m_strm, m_args, nullptr);
        CHECK_CURESULT(status);
    }
#else
    dim3 gridDim = dim3(m_nLink, m_scaleUeAntFreqChan * m_nBatch, m_scaleBsAntFreqChan);
    dim3 blockDim = dim3(m_fastFadingDynDescrCpu -> N_Prbg, uint(m_nBsAnt / m_scaleBsAntFreqChan), uint(m_nUeAnt / m_scaleUeAntFreqChan));
    uint32_t shared_memory_size = sizeof(Tcomplex) * blockDim.y * blockDim.z * m_firNzLen;
    auto kernelPtr = m_runMode == 1 ? genFreqChanCoeKernel_runMode1<Tscalar, Tcomplex> : genFreqChanCoeKernel_runMode2<Tscalar, Tcomplex>;
    cudaFunction_t functionPtr;
    cudaGetFuncBySymbol(&functionPtr, reinterpret_cast<void*>(kernelPtr));
    CUresult status = cuLaunchKernel(functionPtr, gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z, shared_memory_size, m_strm, m_args, nullptr);
#endif
}

template <typename Tscalar, typename Tcomplex>
void fastFadingBaseChan<Tscalar, Tcomplex>::saveChanToH5File(hid_t & h5FileHandle, hid_t & complexDataType)
{
    // assuming all 1D array
    uint8_t rank = 1;
    hsize_t dims[rank] = {0};

    // FIR non zero taps: index and power
    dims[0] = m_firNzPw.size();
    writeHdf5DatasetFromGpu<float>(h5FileHandle, "firNzPw", H5T_IEEE_F32LE, m_fastFadingDynDescrCpu -> firNzPw, dims, rank);
    dims[0] = m_firNzLen;
    writeHdf5DatasetFromGpu<uint16_t>(h5FileHandle, "firNzIdx", H5T_STD_U16LE, m_fastFadingDynDescrCpu -> firNzIdx, dims, rank);

    // batch offsets in time
    dims[0] = m_nBatch;
    writeHdf5DatasetFromGpu<float>(h5FileHandle, "tBatch", H5T_IEEE_F32LE, m_fastFadingDynDescrCpu -> tBatch, dims, rank);
    // cumulative number of samples for each batch [0, 4096, 8192, 12288] means new CIR/CFR per 4096 tx samples. This will be used when specify 14 OFDM symbols
    dims[0] = m_nBatch+1;
    writeHdf5DatasetFromGpu<uint32_t>(h5FileHandle, "batchCumLen", H5T_STD_U32LE, m_fastFadingDynDescrCpu -> batchCumLen, dims, rank);

    // time channel coefficients for all links
    dims[0] = m_timeChanSize;
    writeHdf5DatasetFromGpu<Tcomplex>(h5FileHandle, "timeChan", complexDataType, m_fastFadingDynDescrCpu -> timeChan, dims, rank);
    // frequency channel on Sc if exists
    if(m_runMode == 2)
    {
        dims[0] = m_freqChanScSizePerLink;
        for(uint32_t linkIdx = 0; linkIdx < m_fastFadingDynDescrCpu -> nLink; linkIdx ++)
        {
            std::string freqChanScLinkIdxStr = "freqChanScLink" + std::to_string(linkIdx);
            writeHdf5DatasetFromGpu<Tcomplex>(h5FileHandle, freqChanScLinkIdxStr.c_str(), complexDataType, m_h_deviceFreqChanScPerLinkPtr[linkIdx], dims, rank);
        }
    }
    // frequency channel on prbg if exists
    if(m_runMode > 0 && m_runMode < 3)
    {
        dims[0] = m_freqChanPrbgSize;
        writeHdf5DatasetFromGpu<Tcomplex>(h5FileHandle, "freqChanPrbg", complexDataType, m_fastFadingDynDescrCpu -> freqChanPrbg, dims, rank);

    #if CAL_COS_SIN_IN_GPU
        // CFR related parameters
        dims[0] = m_fastFadingDynDescrCpu -> N_sc;
        writeHdf5DatasetFromGpu<float>(h5FileHandle, "scFreqKHz", H5T_IEEE_F32LE, m_fastFadingDynDescrCpu -> scFreqKHz, dims, rank);

        dims[0] = m_firNzLen;
        writeHdf5DatasetFromGpu<float>(h5FileHandle, "firNzDelayUs2Pi", H5T_IEEE_F32LE, m_fastFadingDynDescrCpu -> firNzDelayUs2Pi, dims, rank);
    #else
        dims[0] = m_firNzLen * m_fastFadingDynDescrCpu -> N_sc;
        writeHdf5DatasetFromGpu<Tcomplex>(h5FileHandle, "firNzDelayScFreq2Pi", complexDataType, m_fastFadingDynDescrCpu -> firNzDelayScFreq2Pi, dims, rank);
    #endif

    }
    // input and output signals if exists
    if(m_sigLenPerAnt)
    {
        dims[0] = m_sigLenPerAnt * (m_enableSwapTxRx ? m_nUeAnt : m_nBsAnt) * m_nLink;
        writeHdf5DatasetFromGpu<Tcomplex>(h5FileHandle, "txSigIn", complexDataType, m_fastFadingDynDescrCpu -> txSigIn, dims, rank);
        dims[0] = m_sigLenPerAnt * (m_enableSwapTxRx ? m_nBsAnt : m_nUeAnt) * m_nLink;
        writeHdf5DatasetFromGpu<Tcomplex>(h5FileHandle, "rxSigOut", complexDataType, m_fastFadingDynDescrCpu -> rxSigOut, dims, rank);

        if (m_fastFadingDynDescrCpu -> saveAntPairSample == 1) // save per antenna sample
        {
            dims[0] = m_sigLenPerAnt * m_nUeAnt * m_nBsAnt * m_nLink;
            writeHdf5DatasetFromGpu<Tcomplex>(h5FileHandle, "rxSigOutPerAntPair", complexDataType, m_fastFadingDynDescrCpu -> rxSigOut, dims, rank);
        }
    }
}

/*----------------------      begin CUDA kernels        -----------------------------*/
template<typename FFT, typename Tscalar, typename Tcomplex>
__launch_bounds__(FFT::max_threads_per_block)
static __global__ void fast_fading_fft_kernel(fastFadingDynDescr_t<Tscalar, Tcomplex> * fastFadingDynDescr, float refTime0)
{
    // GRID(nLink, nBatch, nUeAnt * nBsAnt / FFTs_PER_BLOCK_CONST_)
    // BLOCK(by FFT default)
    using namespace cufftdx;

    // Registers
    cuComplex thread_data[FFT::storage_size];
    uint16_t N_sc_over_2 = fastFadingDynDescr -> N_sc >> 1; // divide by 2
    uint32_t N_FFT = fastFadingDynDescr -> N_FFT;
    uint16_t firNzLen = fastFadingDynDescr -> firNzLen;
    uint16_t firMaxLen = fastFadingDynDescr -> firMaxLen;
    // Local batch id of this FFT in CUDA block, in range [0; FFT::ffts_per_block)
    const uint32_t local_fft_id = threadIdx.y;
    // Global batch id of this FFT in CUDA grid is equal to number of batches per CUDA block (ffts_per_block)
    // times CUDA block id, plus local batch id.
    const uint32_t global_fft_id = ((blockIdx.x * gridDim.y + blockIdx.y) * gridDim.z + blockIdx.z) * FFT::ffts_per_block + local_fft_id;

    // Load freq data from global memory to registers
    const uint32_t freq_offsetSc = fastFadingDynDescr -> N_sc * ((blockIdx.y * gridDim.z + blockIdx.z) * FFT::ffts_per_block + local_fft_id);
    const uint32_t time_offset = firNzLen * global_fft_id;
    const uint32_t stride = FFT::stride;
    uint32_t       index  = time_offset;

    // output buffer
    Tcomplex * freqChanScLink = fastFadingDynDescr -> freqChanSc[blockIdx.x];
    Tcomplex * freqChanPrbg = fastFadingDynDescr -> freqChanPrbg;
    Tcomplex * timeChan = fastFadingDynDescr -> timeChan;

    // get input sinnals
    uint16_t * firNzIdx = fastFadingDynDescr -> firNzIdx;

    // FFT::shared_memory_size bytes of shared memory
    using complex_type = typename FFT::value_type;
    extern __shared__ complex_type shared_mem[];// assuming FFT shared memoery size is much higher than firMaxLen

    // extern __shared__ Tcomplex shared_mem[];
    // Tcomplex * FFT_shared_mem = shared_mem + (FFT::ffts_per_block)*firMaxLen;
    // if(threadIdx.x == 0)
    // {
    //     printf("FFT::storage_size = %d, FFT::elements_per_thread = %d, stride = %d \n ", FFT::storage_size, FFT::elements_per_thread, stride);
    // }
    for(uint32_t resetMemIdx = threadIdx.x; resetMemIdx < firMaxLen; resetMemIdx += blockDim.x)
    {
        shared_mem[firMaxLen * local_fft_id + resetMemIdx].x = 0.0;
        shared_mem[firMaxLen * local_fft_id + resetMemIdx].y = 0.0;
    }
    __syncthreads();

    if(threadIdx.x < firNzLen) // copy NZ taps into the shared, add common cfr ration ref + per batch due to CFO
    {
        float cfoPhaseRef = 2.0f * M_PI * refTime0 * (fastFadingDynDescr -> cfoHz);
        Tcomplex cfrRotationRefCfo = {cos(cfoPhaseRef), sin(cfoPhaseRef)};
        Tcomplex cfrBatchRotationCfo = fastFadingDynDescr -> cfrBatchRotationCfo[blockIdx.y]; // batchIdx = blockIdx.y;
        Tcomplex cfrRotationTotal;
        cfrRotationTotal.x = cfrRotationRefCfo.x * cfrBatchRotationCfo.x - cfrRotationRefCfo.y * cfrBatchRotationCfo.y;
        cfrRotationTotal.y = cfrRotationRefCfo.x * cfrBatchRotationCfo.y + cfrRotationRefCfo.y * cfrBatchRotationCfo.x;
        Tcomplex tmpCopyCir = timeChan[time_offset + threadIdx.x];
        shared_mem[firMaxLen * local_fft_id + firNzIdx[threadIdx.x]].x = tmpCopyCir.x * cfrRotationTotal.x - tmpCopyCir.y * cfrRotationTotal.y;
        shared_mem[firMaxLen * local_fft_id + firNzIdx[threadIdx.x]].y = tmpCopyCir.x * cfrRotationTotal.y + tmpCopyCir.y * cfrRotationTotal.x;
    }
    __syncthreads();

    // Make sure not to go out-of-bounds
    #pragma unroll
    for (uint32_t i = 0; i < FFT::elements_per_thread; i++)
    {
        if (i * stride + threadIdx.x < cufftdx::size_of<FFT>::value)
        {
            #ifdef USE_MEMOERY_FFT_SHIFT_  // use fftshift later
            if(i * stride + threadIdx.x < firMaxLen)
            {
                thread_data[i].x = shared_mem[firMaxLen * local_fft_id + i * stride + threadIdx.x].x;
                thread_data[i].y = shared_mem[firMaxLen * local_fft_id + i * stride + threadIdx.x].y;
            }
            else
            {
                thread_data[i].x = 0.0f;
                thread_data[i].y = 0.0f;
            }
            #else // times 1 or -1 to real part if no fftshift
            if(i * stride + threadIdx.x < firNzLen)
            {
                if(index & 1) // last bit is 1
                {
                    thread_data[i].x =  - shared_mem[firMaxLen * local_fft_id + threadIdx.x].x;
                    thread_data[i].y =  - shared_mem[firMaxLen * local_fft_id + threadIdx.x].y;
                }
                else
                {
                    thread_data[i].x = shared_mem[firMaxLen * local_fft_id + threadIdx.x].x;
                    thread_data[i].y = shared_mem[firMaxLen * local_fft_id + threadIdx.x].y;
                }
            }
            else
            {
                thread_data[i].x = 0.0f;
                thread_data[i].y = 0.0f;
            }
            #endif
            index += stride;
        }
    }

    // Execute IFFT
    FFT().execute(thread_data, shared_mem);

    // Extract per Sc channel coefficients
    index = threadIdx.x;
#pragma unroll
    for (uint32_t i = 0; i < FFT::elements_per_thread; i++)
    {
        if ((i * stride + threadIdx.x) < cufftdx::size_of<FFT>::value)
        {
            thread_data[i].x = thread_data[i].x;
            thread_data[i].y = thread_data[i].y;
            #ifdef USE_MEMOERY_FFT_SHIFT_
            if(index < N_sc_over_2)
            {
                freqChanScLink[freq_offsetSc + index + N_sc_over_2].x = thread_data[i].x;;
                freqChanScLink[freq_offsetSc + index + N_sc_over_2].y = thread_data[i].y;;
            }
            else if(index >= (N_FFT - N_sc_over_2))
            {
                freqChanScLink[freq_offsetSc + index - N_FFT + N_sc_over_2].x = thread_data[i].x;
                freqChanScLink[freq_offsetSc + index - N_FFT + N_sc_over_2].y = thread_data[i].y;
            }
            #else
            if(index >= ((N_FFT >> 1) - N_sc_over_2) && index < ((N_FFT >> 1) + N_sc_over_2) )  // Middle part
            {
                freqChanScLink[freq_offsetSc + index - (N_FFT >> 1) + N_sc_over_2].x = thread_data[i].x;
                freqChanScLink[freq_offsetSc + index - (N_FFT >> 1) + N_sc_over_2].y = thread_data[i].y;
            }
            #endif
            index += stride;
        }
    }
}

template<typename FFT, typename Tscalar, typename Tcomplex>
__launch_bounds__(FFT::max_threads_per_block)
static __global__ void fast_fading_fft_kernel_PrbgOnly(fastFadingDynDescr_t<Tscalar, Tcomplex> * fastFadingDynDescr, float refTime0)
{
    // GRID(nLink, nBatch, nUeAnt * nBsAnt / FFTs_PER_BLOCK_CONST_)
    // BLOCK(by FFT default)
    using namespace cufftdx;

    // Registers
    cuComplex thread_data[FFT::storage_size];
    uint16_t N_sc_over_2 = fastFadingDynDescr -> N_sc >> 1; // divide by 2
    uint16_t N_Prbg = fastFadingDynDescr -> N_Prbg; // number of Prbg
    uint16_t N_sc_Prbg = (threadIdx.x < N_Prbg - 1) ? fastFadingDynDescr -> N_sc_Prbg : fastFadingDynDescr -> N_sc_last_Prbg; // number of Scs per Prbg
    // for the last Prbg, N_sc_Prbg = fastFadingDynDescr -> N_sc_last_Prbg SCs, may be smaller than fastFadingDynDescr -> N_sc_Prbg
    uint32_t N_FFT = fastFadingDynDescr -> N_FFT;
    uint16_t firNzLen = fastFadingDynDescr -> firNzLen;
    uint16_t firMaxLen = fastFadingDynDescr -> firMaxLen;
    uint16_t N_sc = fastFadingDynDescr -> N_sc;
    uint8_t  freqConvertType = fastFadingDynDescr -> freqConvertType;
    float inverseNScPrbg = (threadIdx.x < N_Prbg - 1) ? fastFadingDynDescr -> inverseNScPrbg : fastFadingDynDescr -> inverseNScLastPrbg;
    // Local batch id of this FFT in CUDA block, in range [0; FFT::ffts_per_block)
    const uint32_t local_fft_id = threadIdx.y;
    // Global batch id of this FFT in CUDA grid is equal to number of batches per CUDA block (ffts_per_block)
    // times CUDA block id, plus local batch id.
    const uint32_t global_fft_id = ((blockIdx.x * gridDim.y + blockIdx.y) * gridDim.z + blockIdx.z) * FFT::ffts_per_block + local_fft_id;

    // Load freq data from global memory to registers
    const uint32_t freq_offsetPrbg_global = N_Prbg * global_fft_id;
    const uint32_t time_offset = firNzLen * global_fft_id;
    const uint32_t stride = FFT::stride;
    uint32_t       index  = time_offset;

    // output buffer
    Tcomplex * freqChanPrbg = fastFadingDynDescr -> freqChanPrbg;
    Tcomplex * timeChan = fastFadingDynDescr -> timeChan;

    // get input sinnals
    uint16_t * firNzIdx = fastFadingDynDescr -> firNzIdx;

    // FFT::shared_memory_size bytes of shared memory
    using complex_type = typename FFT::value_type;
    extern __shared__ complex_type shared_mem[];// assuming FFT shared memoery size is much higher than firMaxLen
    // extern __shared__ Tcomplex shared_mem[];
    // Tcomplex * FFT_shared_mem = shared_mem + (FFT::ffts_per_block)*firMaxLen;
    // if(threadIdx.x == 0)
    // {
    //     printf("FFT::storage_size = %d, FFT::elements_per_thread = %d, stride = %d \n ", FFT::storage_size, FFT::elements_per_thread, stride);
    // }
    for(uint32_t resetMemIdx = threadIdx.x; resetMemIdx < firMaxLen; resetMemIdx += blockDim.x)
    {
        shared_mem[firMaxLen * local_fft_id + resetMemIdx].x = 0.0f;
        shared_mem[firMaxLen * local_fft_id + resetMemIdx].y = 0.0f;
    }
    __syncthreads();

    if(threadIdx.x < firNzLen) // copy NZ taps into the shared, add common cfr ration ref + per batch due to CFO
    {
        float cfoPhaseRef = 2.0f * M_PI * refTime0 * (fastFadingDynDescr -> cfoHz);
        Tcomplex cfrRotationRefCfo = {cos(cfoPhaseRef), sin(cfoPhaseRef)};
        Tcomplex cfrBatchRotationCfo = fastFadingDynDescr -> cfrBatchRotationCfo[blockIdx.y]; // batchIdx = blockIdx.y;
        Tcomplex cfrRotationTotal;
        cfrRotationTotal.x = cfrRotationRefCfo.x * cfrBatchRotationCfo.x - cfrRotationRefCfo.y * cfrBatchRotationCfo.y;
        cfrRotationTotal.y = cfrRotationRefCfo.x * cfrBatchRotationCfo.y + cfrRotationRefCfo.y * cfrBatchRotationCfo.x;
        Tcomplex tmpCopyCir = timeChan[time_offset + threadIdx.x];
        shared_mem[firMaxLen * local_fft_id + firNzIdx[threadIdx.x]].x = tmpCopyCir.x * cfrRotationTotal.x - tmpCopyCir.y * cfrRotationTotal.y;
        shared_mem[firMaxLen * local_fft_id + firNzIdx[threadIdx.x]].y = tmpCopyCir.x * cfrRotationTotal.y + tmpCopyCir.y * cfrRotationTotal.x;
    }
    __syncthreads();

    // Make sure not to go out-of-bounds
    #pragma unroll
    for (uint32_t i = 0; i < FFT::elements_per_thread; i++)
    {
        if (i * stride + threadIdx.x < cufftdx::size_of<FFT>::value)
        {
            #ifdef USE_MEMOERY_FFT_SHIFT_  // use fftshift later
            if(i * stride + threadIdx.x < firMaxLen)
            {
                thread_data[i].x = shared_mem[firMaxLen * local_fft_id + i * stride + threadIdx.x].x;
                thread_data[i].y = shared_mem[firMaxLen * local_fft_id + i * stride + threadIdx.x].y;
            }
            else
            {
                thread_data[i].x = 0.0f;
                thread_data[i].y = 0.0f;
            }
            #else // times 1 or -1 to real part if no fftshift
            if(i * stride + threadIdx.x < firNzLen)
            {
                if(index & 1) // last bit is 1
                {
                    thread_data[i].x =  - shared_mem[firMaxLen * local_fft_id + threadIdx.x].x;
                    thread_data[i].y =  - shared_mem[firMaxLen * local_fft_id + threadIdx.x].y;
                }
                else
                {
                    thread_data[i].x = shared_mem[firMaxLen * local_fft_id + threadIdx.x].x;
                    thread_data[i].y = shared_mem[firMaxLen * local_fft_id + threadIdx.x].y;
                }
            }
            else
            {
                thread_data[i].x = 0.0f;
                thread_data[i].y = 0.0f;
            }
            #endif
            index += stride;
        }
    }

    // Execute IFFT
    FFT().execute(thread_data, shared_mem);

    // Extract per Sc channel coefficients
    index = threadIdx.x;
    uint16_t freq_offsetSc_local = (fastFadingDynDescr -> N_sc) * local_fft_id;
    Tcomplex * s_freqSc = reinterpret_cast<Tcomplex*>(shared_mem); // for converting CFR on Sc to Prbg
#pragma unroll
    for (uint32_t i = 0; i < FFT::elements_per_thread; i++)
    {
        if ((i * stride + threadIdx.x) < cufftdx::size_of<FFT>::value)
        {
            #ifdef USE_MEMOERY_FFT_SHIFT_
            if(index < N_sc_over_2)
            {
                s_freqSc[freq_offsetSc_local + (index + N_sc_over_2)].x = thread_data[i].x;
                s_freqSc[freq_offsetSc_local + (index + N_sc_over_2)].y = thread_data[i].y;
            }
            else if(index >= (N_FFT - N_sc_over_2))
            {
                s_freqSc[freq_offsetSc_local + (index - N_FFT + N_sc_over_2)].x = thread_data[i].x;
                s_freqSc[freq_offsetSc_local + (index - N_FFT + N_sc_over_2)].y = thread_data[i].y;
            }
            #else
            if(index >= ((N_FFT >> 1) - N_sc_over_2) && index < ((N_FFT >> 1) + N_sc_over_2) )  // Middle part
            {
                s_freqSc[freq_offsetSc_local + (index - (N_FFT >> 1) + N_sc_over_2)].x = thread_data[i].x;
                s_freqSc[freq_offsetSc_local + (index - (N_FFT >> 1) + N_sc_over_2)].y = thread_data[i].y;
            }
            #endif
            index += stride;
        }
    }
    __syncthreads();

    // convert freq chan on sc to prbg
    if(threadIdx.x < N_Prbg)
    {
        uint32_t sc_avg_offset =  freq_offsetSc_local + threadIdx.x * N_sc_Prbg;
        cuComplex tempSum = {0.0f, 0.0f};

        switch(freqConvertType)
        {
            case 0: // use first SC for CFR on the Prbg
                tempSum.x = s_freqSc[sc_avg_offset].x;
                tempSum.y = s_freqSc[sc_avg_offset].y;
                break;

            case 1: // use center SC for CFR on the Prbg, e.g., sc 6 for sc 0,1,2,...,11
                tempSum.x = s_freqSc[sc_avg_offset + N_sc_Prbg/2].x;
                tempSum.y = s_freqSc[sc_avg_offset + N_sc_Prbg/2].y;
                break;

            case 2: // use last SC for CFR on the Prbg
                tempSum.x = s_freqSc[sc_avg_offset + N_sc_Prbg - 1].x;
                tempSum.y = s_freqSc[sc_avg_offset + N_sc_Prbg - 1].y;
                break;

            case 3: // use average SC for CFR on the Prbg
                // for the last Prbg, inverseNScPrbg = fastFadingDynDescr -> inverseNScLastPrbg
                for(uint16_t scIdx= 0; scIdx< N_sc_Prbg; scIdx++)
                {
                    tempSum.x = tempSum.x + static_cast<float>(s_freqSc[sc_avg_offset].x);
                    tempSum.y = tempSum.y + static_cast<float>(s_freqSc[sc_avg_offset].y);
                    sc_avg_offset++;
                }
                tempSum.x *= inverseNScPrbg;
                tempSum.y *= inverseNScPrbg;
                break;

            // case 4 will never be run in this kernel
            default:
                printf("Error: Invalid freqConvertType %d!\n", freqConvertType);
                break;
        }

        // copy freq chan on Prgb to global memory
        freqChanPrbg[freq_offsetPrbg_global + threadIdx.x].x = tempSum.x;
        freqChanPrbg[freq_offsetPrbg_global + threadIdx.x].y = tempSum.y;
    }
}

// Choose FFT kernel
template<typename Tscalar, typename Tcomplex, uint32_t FftSize, uint32_t Arch>
fftKernelHandle<Tscalar, Tcomplex> fast_fading_get_fft_param(dim3& block_dim, uint& shared_memory_size, uint8_t& runMode)
{
    using namespace cufftdx;

    // use predefined numbers
    using FFT = decltype(Size<FftSize>() + Precision<float>() + Type<fft_type::c2c>()
                                + Direction<fft_direction::forward>()
                                + FFTsPerBlock<FFTs_PER_BLOCK_CONST_>() // + ElementsPerThread<FFT_ELEMENTS_PER_THREAD_CONST_>()
                                + SM<Arch>() + Block());

    // use cuFFTdx configurations
    // Base of the FFT description
    // using FFT_base = decltype(Size<FftSize>() + Precision<Tscalar>() + Type<fft_type::c2c>()
    // + Direction<fft_direction::forward>()
    // /* Notice lack of ElementsPerThread and FFTsPerBlock operators */
    // + SM<Arch>() + Block());
    // // FFT description with suggested FFTs per CUDA block for the default (optimal) elements per thread
    // using FFT = decltype(FFT_base() + FFTsPerBlock<1>());

    block_dim = FFT::block_dim;
    shared_memory_size = FFT::shared_memory_size;

    if(runMode == 1)
    {
        return fast_fading_fft_kernel_PrbgOnly<FFT, Tscalar, Tcomplex>;
    }
    else if(runMode == 2)
    {
        return fast_fading_fft_kernel<FFT, Tscalar, Tcomplex>;
    }
    else
    {
        return nullptr;
    }
}

template<typename Tscalar, typename Tcomplex>
fftKernelHandle<Tscalar, Tcomplex> fast_fading_get_fft_param(const int Nfft, uint32_t cudaDeviceArch, dim3& block_dim, uint& shared_memory_size, uint8_t& runMode)
{
    // current only support cudaDeviceArch = 800, 890, and 900, with Nfft = 4096
    if((Nfft == 4096) && (cudaDeviceArch == 800))
    {
        return fast_fading_get_fft_param<Tscalar, Tcomplex, 4096, 800>(block_dim, shared_memory_size, runMode);
    }
    // else if ((Nfft == 4096) && (cudaDeviceArch == 890))
    // {
    //     return fast_fading_get_fft_param<Tscalar, Tcomplex, 4096, 890>(block_dim, shared_memory_size, runMode);
    // }
    else if ((Nfft == 4096) && (cudaDeviceArch == 900))
    {
        return fast_fading_get_fft_param<Tscalar, Tcomplex, 4096, 900>(block_dim, shared_memory_size, runMode);
    }
    else
    {
        printf("error: Unsupported FFT length %d or cudaDeviceArch %d in frequency channel, please add your Nfft or cudaDeviceArch into fast_fading_get_fft_param and retry \n", Nfft, cudaDeviceArch);
        exit(EXIT_FAILURE);
    }
    return nullptr;
}


template <typename Tscalar, typename Tcomplex>
static __global__ void normalizeTimeChan(fastFadingDynDescr_t<Tscalar, Tcomplex> * fastFadingDynDescr)
{
    // GRID(1,1,1);
    // BLOCK(THREADS_PER_BLOCK_NORMALIZATION_, 1, 1);

    Tcomplex * timeChan = fastFadingDynDescr -> timeChan;
    uint32_t timeChanSize   = fastFadingDynDescr -> timeChanSize;
    uint32_t tid = threadIdx.x;

    // shared memeory for calculate local sums and save normalization coe at localSum[0];
    __shared__ Tscalar localSum[THREADS_PER_BLOCK_NORMALIZATION_];

    // calculate temporary sum for each thread
    localSum[tid] = 0.0f;
    for(uint32_t timeChanIdx = tid; timeChanIdx < timeChanSize; timeChanIdx += blockDim.x)
    {
        localSum[tid] += timeChan[timeChanIdx].x * timeChan[timeChanIdx].x + timeChan[timeChanIdx].y * timeChan[timeChanIdx].y;
    }
    __syncthreads();

    // obtain normalziation coe using parallel reduction
    uint16_t h = THREADS_PER_BLOCK_NORMALIZATION_;
    uint16_t s = ceilf(h*0.5f);
    #pragma unroll
    while(s > 1)
    {
        if(tid < h-s)
        {
            localSum[tid] += localSum[tid + s];
        }
        h = s; s = ceilf(h*0.5f);
        __syncthreads();
    }
    if(tid == 0)
    {
        localSum[0] += localSum[1];
        Tscalar normalizeTarget = (fastFadingDynDescr -> nBsAnt) * (fastFadingDynDescr -> nUeAnt) * (fastFadingDynDescr -> nBatch);
        localSum[0] = sqrt(static_cast<float>(normalizeTarget / localSum[0]));
    }
    __syncthreads();

    // apply normalization
    for(uint32_t timeChanIdx = tid; timeChanIdx < timeChanSize; timeChanIdx += blockDim.x)
    {
        timeChan[timeChanIdx].x *= localSum[0];
        timeChan[timeChanIdx].y *= localSum[0];
    }
}

template <typename Tscalar, typename Tcomplex>
static __device__ cuComplex calCfrbyCir(float freqKHz, uint16_t firNzLen, float * firNzDelayUs2Pi, Tcomplex * cir, float cfrPhaseShift)
{
    cuComplex cfrOnFreqKHz = {0.0f, 0.0f};
    cuComplex tmp_cfr = {0.0f, 0.0f};
    cuComplex tmpExp = {0.0f, 0.0f};
    for (uint16_t firNzIdx = 0; firNzIdx < firNzLen; firNzIdx ++)
    {
        tmpExp.x = cos(firNzDelayUs2Pi[firNzIdx] * freqKHz * 1e-3);
        tmpExp.y = sin(firNzDelayUs2Pi[firNzIdx] * freqKHz * 1e-3);
        tmp_cfr.x += tmpExp.x * float(cir[firNzIdx].x) - tmpExp.y * float(cir[firNzIdx].y);
        tmp_cfr.y += tmpExp.x * float(cir[firNzIdx].y) + tmpExp.y * float(cir[firNzIdx].x);
    }
    // use tmpExp to rotate CFR
    tmpExp.x = cos(cfrPhaseShift);
    tmpExp.y = sin(cfrPhaseShift);
    cfrOnFreqKHz.x = tmp_cfr.x * tmpExp.x - tmp_cfr.y * tmpExp.y;
    cfrOnFreqKHz.y = tmp_cfr.x * tmpExp.y + tmp_cfr.y * tmpExp.x;
    return cfrOnFreqKHz;
}

template <typename Tscalar, typename Tcomplex>
static __device__ cuComplex calCfrbyCir_v2(uint16_t firNzLen, Tcomplex * firNzDelayScFreq2Pi, Tcomplex * cir)
{
    cuComplex cfrOnFreqKHz = {0.0f, 0.0f};
    cuComplex tmpExp = {0.0f, 0.0f};
    for (uint16_t firNzIdx = 0; firNzIdx < firNzLen; firNzIdx ++)
    {
        tmpExp.x = firNzDelayScFreq2Pi[firNzIdx].x;
        tmpExp.y = firNzDelayScFreq2Pi[firNzIdx].y;
        cfrOnFreqKHz.x += tmpExp.x * float(cir[firNzIdx].x) - tmpExp.y * float(cir[firNzIdx].y);
        cfrOnFreqKHz.y += tmpExp.x * float(cir[firNzIdx].y) + tmpExp.y * float(cir[firNzIdx].x);
    }
    return cfrOnFreqKHz;
}

template <typename Tscalar, typename Tcomplex>
static __global__ void genFreqChanCoeKernel_runMode1(fastFadingDynDescr_t<Tscalar, Tcomplex> * fastFadingDynDescr, float refTime0)
{
    // GRID(nLink, m_scaleUeAntFreqChan * m_nBatch, m_scaleBsAntFreqChan); blockIdx.y = scaleUeAntFreqChanIdx * nBatch + batchIdx
    // BLOCK(N_Prbg, nBsAnt/m_scaleBsAntFreqChan, nUeAnt/m_scaleUeAntFreqChan);

    uint32_t cidUidOffset = blockIdx.x; // linkIdx
    uint32_t nBatch = fastFadingDynDescr -> nBatch;
    uint16_t N_sc = fastFadingDynDescr -> N_sc;
    uint16_t N_Prbg = fastFadingDynDescr -> N_Prbg;
    uint16_t N_sc_Prbg = (threadIdx.x < N_Prbg - 1) ? fastFadingDynDescr -> N_sc_Prbg : fastFadingDynDescr -> N_sc_last_Prbg; // number of Scs per Prbg
    uint16_t N_sc_over_2 = fastFadingDynDescr -> N_sc >> 1; // divide by 2
    // for the last Prbg, N_sc_Prbg = fastFadingDynDescr -> N_sc_last_Prbg SCs, may be smaller than fastFadingDynDescr -> N_sc_Prbg
    uint8_t freqConvertType = fastFadingDynDescr -> freqConvertType;
    uint8_t scSampling = fastFadingDynDescr -> scSampling;
    uint16_t nBsAnt = fastFadingDynDescr -> nBsAnt;
    uint16_t nUeAnt = fastFadingDynDescr -> nUeAnt;
    float * scFreqKHz = fastFadingDynDescr -> scFreqKHz;
    Tcomplex * firNzDelayScFreq2Pi = fastFadingDynDescr -> firNzDelayScFreq2Pi;
    float inverseNScPrbg = (threadIdx.x < N_Prbg - 1) ? fastFadingDynDescr -> inverseNScPrbg : fastFadingDynDescr -> inverseNScLastPrbg;
    Tcomplex * timeChan = fastFadingDynDescr -> timeChan;
    Tcomplex * freqChanScLink = fastFadingDynDescr -> freqChanSc[cidUidOffset];
    Tcomplex * freqChanPrbg = fastFadingDynDescr -> freqChanPrbg;
    float cfrPhaseShiftTimeDelay = fastFadingDynDescr -> cfrPhaseShiftTimeDelay;

    // prbg and ant index
    uint16_t prbgIdx  = threadIdx.x;
    uint16_t bsAntIdx = threadIdx.y + blockIdx.z * blockDim.y;
    uint16_t batchIdx = blockIdx.y % nBatch;
    uint16_t ueAntIdx = threadIdx.z + (blockIdx.y / nBatch) * blockDim.z;
    uint32_t prbg_offet = (((cidUidOffset * nBatch + batchIdx) * nUeAnt + ueAntIdx) * nBsAnt + bsAntIdx) * N_Prbg + prbgIdx;
    uint16_t localScOffset = prbgIdx * fastFadingDynDescr -> N_sc_Prbg; // start sc index for a prbg
    // shared mermory for CIR
    extern __shared__ char shareData[]; // all shared memory data pointer
    Tcomplex * s_timeChanLocal = reinterpret_cast<Tcomplex *>(shareData); // firNzLen * nUeAnt * nBsAnt
    // only use shared memory for Tcomplex data type

    __shared__ float s_firNzDelayUs2Pi[MAX_NZ_TAPS_]; // no more than 24 NZ taps based on 3GPP 38.901

    // read CIR to shared memory
    uint16_t firNzLen = fastFadingDynDescr -> firNzLen;
    uint16_t localCirOffset = (threadIdx.z * blockDim.y + threadIdx.y) * firNzLen;

    uint32_t globalCirOffset = (((cidUidOffset * nBatch + batchIdx) * nUeAnt + ueAntIdx) * nBsAnt + bsAntIdx) * firNzLen; // always use the first batch for conversion of CIR to CFR

    for (uint16_t copyIdx = prbgIdx; copyIdx < firNzLen; copyIdx += N_Prbg) // copy NZ taps into the shared, add common cfr ration ref + per batch due to CFO// copy NZ taps into the shared, add common cfr ration ref + per batch due to CFO
    {
        float cfoPhaseRef = 2.0f * M_PI * refTime0 * (fastFadingDynDescr -> cfoHz);
        Tcomplex cfrRotationRefCfo = {cos(cfoPhaseRef), sin(cfoPhaseRef)};
        Tcomplex cfrBatchRotationCfo = fastFadingDynDescr -> cfrBatchRotationCfo[batchIdx];
        Tcomplex cfrRotationTotal;
        cfrRotationTotal.x = cfrRotationRefCfo.x * cfrBatchRotationCfo.x - cfrRotationRefCfo.y * cfrBatchRotationCfo.y;
        cfrRotationTotal.y = cfrRotationRefCfo.x * cfrBatchRotationCfo.y + cfrRotationRefCfo.y * cfrBatchRotationCfo.x;
        Tcomplex tmpCopyCir = timeChan[globalCirOffset + copyIdx];
        s_timeChanLocal[localCirOffset + copyIdx].x = tmpCopyCir.x * cfrRotationTotal.x - tmpCopyCir.y * cfrRotationTotal.y;
        s_timeChanLocal[localCirOffset + copyIdx].y = tmpCopyCir.x * cfrRotationTotal.y + tmpCopyCir.y * cfrRotationTotal.x;

        if(threadIdx.y == 0 && threadIdx.z == 0)
        {
            s_firNzDelayUs2Pi[copyIdx] = fastFadingDynDescr -> firNzDelayUs2Pi[copyIdx];
        }
    }
    __syncthreads();

    // calculate CFR on sc
    cuComplex cfrOnFreqKHz = {0.0f, 0.0f};
    cuComplex tempSum = {0.0f, 0.0f};
#if CAL_COS_SIN_IN_GPU
    float cfrPhaseShift = 0.0f;
#endif
    switch(freqConvertType)
    {
        case 0: // use first SC for CFR on the Prbg
        #if CAL_COS_SIN_IN_GPU
            cfrPhaseShift = cfrPhaseShiftTimeDelay * (localScOffset - N_sc_over_2);
            cfrOnFreqKHz = calCfrbyCir<Tscalar, Tcomplex>(scFreqKHz[localScOffset], firNzLen, s_firNzDelayUs2Pi, s_timeChanLocal + localCirOffset, cfrPhaseShift);
        #else
            cfrOnFreqKHz = calCfrbyCir_v2<Tscalar, Tcomplex>(firNzLen, firNzDelayScFreq2Pi + localScOffset * firNzLen, s_timeChanLocal + localCirOffset);
        #endif
            freqChanPrbg[prbg_offet].x = cfrOnFreqKHz.x;
            freqChanPrbg[prbg_offet].y = cfrOnFreqKHz.y;
            break;

        case 1: // use center SC for CFR on the Prbg, e.g., sc 6 for sc 0,1,2,...,11
            #if CAL_COS_SIN_IN_GPU
                cfrPhaseShift = cfrPhaseShiftTimeDelay * (localScOffset + N_sc_Prbg/2 - N_sc_over_2);
                cfrOnFreqKHz = calCfrbyCir<Tscalar, Tcomplex>(scFreqKHz[localScOffset + N_sc_Prbg/2], firNzLen, s_firNzDelayUs2Pi, s_timeChanLocal + localCirOffset, cfrPhaseShift);
            #else
                cfrOnFreqKHz = calCfrbyCir_v2<Tscalar, Tcomplex>(firNzLen, firNzDelayScFreq2Pi + (localScOffset + N_sc_Prbg/2) * firNzLen, s_timeChanLocal + localCirOffset);
            #endif
            freqChanPrbg[prbg_offet].x = cfrOnFreqKHz.x;
            freqChanPrbg[prbg_offet].y = cfrOnFreqKHz.y;
            break;

        case 2: // use last SC for CFR on the Prbg
            #if CAL_COS_SIN_IN_GPU
                cfrPhaseShift = cfrPhaseShiftTimeDelay * (localScOffset + N_sc_Prbg - 1 - N_sc_over_2);
                cfrOnFreqKHz = calCfrbyCir<Tscalar, Tcomplex>(scFreqKHz[localScOffset + N_sc_Prbg - 1], firNzLen, s_firNzDelayUs2Pi, s_timeChanLocal + localCirOffset, cfrPhaseShift);
            #else
                cfrOnFreqKHz = calCfrbyCir_v2<Tscalar, Tcomplex>(firNzLen, firNzDelayScFreq2Pi + (localScOffset + N_sc_Prbg - 1) * firNzLen, s_timeChanLocal + localCirOffset);
            #endif
            freqChanPrbg[prbg_offet].x = cfrOnFreqKHz.x;
            freqChanPrbg[prbg_offet].y = cfrOnFreqKHz.y;
            break;

        case 3: // use average SC for on for the Prbg
            // for the last Prbg, inverseNScPrbg = fastFadingDynDescr -> inverseNScLastPrbg
            tempSum = {0.0f, 0.0f};
            for(uint16_t scInPrbgIdx = 0; scInPrbgIdx < N_sc_Prbg; scInPrbgIdx += scSampling)
            {
            #if CAL_COS_SIN_IN_GPU
                cfrPhaseShift = cfrPhaseShiftTimeDelay * (localScOffset + scInPrbgIdx - N_sc_over_2);
                cfrOnFreqKHz = calCfrbyCir<Tscalar, Tcomplex>(scFreqKHz[localScOffset + scInPrbgIdx], firNzLen, s_firNzDelayUs2Pi, s_timeChanLocal + localCirOffset, cfrPhaseShift);
            #else
                cfrOnFreqKHz = calCfrbyCir_v2<Tscalar, Tcomplex>(firNzLen, firNzDelayScFreq2Pi + (localScOffset + scInPrbgIdx) * firNzLen, s_timeChanLocal + localCirOffset);
            #endif

                tempSum.x += cfrOnFreqKHz.x; // add Sc together
                tempSum.y += cfrOnFreqKHz.y;
            }
            freqChanPrbg[prbg_offet].x = tempSum.x * inverseNScPrbg;
            freqChanPrbg[prbg_offet].y = tempSum.y * inverseNScPrbg;
            break;

        case 4: // use average SC for on for the Prbg with removing frequency ramping
            // for the last Prbg, inverseNScPrbg = fastFadingDynDescr -> inverseNScLastPrbg
            tempSum = {0.0f, 0.0f};
            for(uint16_t scInPrbgIdx = 0; scInPrbgIdx < N_sc_Prbg; scInPrbgIdx += scSampling)
            {
                // TODO: for imperfect channel, add noise in calculation of CFR, noise should be different across scInPrgbIdx
                // using N_sc_Prbg/2 as the anchor sc within a Prbg
            #if CAL_COS_SIN_IN_GPU
                cfrPhaseShift = cfrPhaseShiftTimeDelay * (localScOffset + N_sc_Prbg/2 - N_sc_over_2);
                cfrOnFreqKHz = calCfrbyCir<Tscalar, Tcomplex>(scFreqKHz[localScOffset + N_sc_Prbg/2], firNzLen, s_firNzDelayUs2Pi, s_timeChanLocal + localCirOffset, cfrPhaseShift);
            #else
                cfrOnFreqKHz = calCfrbyCir_v2<Tscalar, Tcomplex>(firNzLen, firNzDelayScFreq2Pi + (localScOffset + N_sc_Prbg/2) * firNzLen, s_timeChanLocal + localCirOffset);
            #endif

                tempSum.x += cfrOnFreqKHz.x; // add Sc together
                tempSum.y += cfrOnFreqKHz.y;
            }
            freqChanPrbg[prbg_offet].x = tempSum.x * inverseNScPrbg;
            freqChanPrbg[prbg_offet].y = tempSum.y * inverseNScPrbg;
            break;

        default:
            printf("Error: Invalid freqConvertType %d!\n", freqConvertType);
            break;
    }
}

template <typename Tscalar, typename Tcomplex>
static __global__ void genFreqChanCoeKernel_runMode2(fastFadingDynDescr_t<Tscalar, Tcomplex> * fastFadingDynDescr, float refTime0)
{
    // GRID(nLink, m_scaleUeAntFreqChan * m_nBatch, m_scaleBsAntFreqChan); blockIdx.y = scaleUeAntFreqChanIdx * nBatch + batchIdx
    // BLOCK(N_Prbg, nBsAnt/m_scaleBsAntFreqChan, nUeAnt/m_scaleUeAntFreqChan);

    uint32_t cidUidOffset = blockIdx.x; // linkIdx
    uint32_t nBatch = fastFadingDynDescr -> nBatch;
    uint16_t N_sc = fastFadingDynDescr -> N_sc;
    uint16_t N_Prbg = fastFadingDynDescr -> N_Prbg;
    uint16_t N_sc_Prbg = (threadIdx.x < N_Prbg - 1) ? fastFadingDynDescr -> N_sc_Prbg : fastFadingDynDescr -> N_sc_last_Prbg; // number of Scs per Prbg
    uint16_t N_sc_over_2 = fastFadingDynDescr -> N_sc >> 1; // divide by 2
    // for the last Prbg, N_sc_Prbg = fastFadingDynDescr -> N_sc_last_Prbg SCs, may be smaller than fastFadingDynDescr -> N_sc_Prbg
    uint8_t freqConvertType = fastFadingDynDescr -> freqConvertType;
    uint8_t scSampling = fastFadingDynDescr -> scSampling;
    uint16_t nBsAnt = fastFadingDynDescr -> nBsAnt;
    uint16_t nUeAnt = fastFadingDynDescr -> nUeAnt;
    float * scFreqKHz = fastFadingDynDescr -> scFreqKHz;
    Tcomplex * firNzDelayScFreq2Pi = fastFadingDynDescr -> firNzDelayScFreq2Pi;
    float inverseNScPrbg = (threadIdx.x < N_Prbg - 1) ? fastFadingDynDescr -> inverseNScPrbg : fastFadingDynDescr -> inverseNScLastPrbg;
    Tcomplex * timeChan = fastFadingDynDescr -> timeChan;
    Tcomplex * freqChanScLink = fastFadingDynDescr -> freqChanSc[cidUidOffset];
    Tcomplex * freqChanPrbg = fastFadingDynDescr -> freqChanPrbg;
    float cfrPhaseShiftTimeDelay = fastFadingDynDescr -> cfrPhaseShiftTimeDelay;

    // prbg and ant index
    uint16_t prbgIdx  = threadIdx.x;
    uint16_t bsAntIdx = threadIdx.y + blockIdx.z * blockDim.y;
    uint16_t batchIdx = blockIdx.y % nBatch;
    uint16_t ueAntIdx = threadIdx.z + (blockIdx.y / nBatch) * blockDim.z;
    uint32_t prbg_offet = (((cidUidOffset * nBatch + batchIdx) * nUeAnt + ueAntIdx) * nBsAnt + bsAntIdx) * N_Prbg + prbgIdx;
    uint16_t localScOffset = prbgIdx * fastFadingDynDescr -> N_sc_Prbg; // start sc index for a prbg
    uint32_t sc_start_offset  = ((batchIdx * nUeAnt + ueAntIdx) * nBsAnt + bsAntIdx) * N_sc + localScOffset;
    // shared mermory for CIR
    extern __shared__ char shareData[]; // all shared memory data pointer
    Tcomplex * s_timeChanLocal = reinterpret_cast<Tcomplex *>(shareData); // firNzLen * nUeAnt * nBsAnt
    // only use shared memory for Tcomplex data type

    __shared__ float s_firNzDelayUs2Pi[MAX_NZ_TAPS_]; // no more than 24 NZ taps based on 3GPP 38.901

    // read CIR to shared memory
    uint16_t firNzLen = fastFadingDynDescr -> firNzLen;
    uint16_t localCirOffset = (threadIdx.z * blockDim.y + threadIdx.y) * firNzLen;
    uint32_t globalCirOffset = (((cidUidOffset * nBatch + batchIdx) * nUeAnt + ueAntIdx) * nBsAnt + bsAntIdx) * firNzLen;

    for (uint16_t copyIdx = prbgIdx; copyIdx < firNzLen; copyIdx += N_Prbg) // copy NZ taps into the shared, add common cfr ration ref + per batch due to CFO// copy NZ taps into the shared, add common cfr ration ref + per batch due to CFO
    {
        float cfoPhaseRef = 2.0f * M_PI * refTime0 * (fastFadingDynDescr -> cfoHz);
        Tcomplex cfrRotationRefCfo = {cos(cfoPhaseRef), sin(cfoPhaseRef)};
        Tcomplex cfrBatchRotationCfo = fastFadingDynDescr -> cfrBatchRotationCfo[batchIdx];
        Tcomplex cfrRotationTotal;
        cfrRotationTotal.x = cfrRotationRefCfo.x * cfrBatchRotationCfo.x - cfrRotationRefCfo.y * cfrBatchRotationCfo.y;
        cfrRotationTotal.y = cfrRotationRefCfo.x * cfrBatchRotationCfo.y + cfrRotationRefCfo.y * cfrBatchRotationCfo.x;
        Tcomplex tmpCopyCir = timeChan[globalCirOffset + copyIdx];
        s_timeChanLocal[localCirOffset + copyIdx].x = tmpCopyCir.x * cfrRotationTotal.x - tmpCopyCir.y * cfrRotationTotal.y;
        s_timeChanLocal[localCirOffset + copyIdx].y = tmpCopyCir.x * cfrRotationTotal.y + tmpCopyCir.y * cfrRotationTotal.x;

        if(threadIdx.y == 0 && threadIdx.z == 0)
        {
            s_firNzDelayUs2Pi[copyIdx] = fastFadingDynDescr -> firNzDelayUs2Pi[copyIdx];
        }
    }
    __syncthreads();

    // calculate CFR on sc
    cuComplex cfrOnFreqKHz = {0.0f, 0.0f};
    cuComplex tempSum = {0.0f, 0.0f};
#if CAL_COS_SIN_IN_GPU
    float cfrPhaseShift = 0.0f;
#endif
    // calculate CFR on all Scs and save to GPU global memory
    for(uint16_t scInPrbgIdx = 0; scInPrbgIdx < N_sc_Prbg; scInPrbgIdx += scSampling)
    {
        // If freqConvertType == 4, use average SC for on for the Prbg with removing frequency ramping
        // TODO: for imperfect channel, add noise in calculation of CFR, noise should be different across scInPrgbIdx
        // using N_sc_Prbg/2 as the anchor sc within a Prbg
    #if CAL_COS_SIN_IN_GPU
        cfrPhaseShift = cfrPhaseShiftTimeDelay * (localScOffset + (freqConvertType == 4 ? N_sc_Prbg/2 : scInPrbgIdx) - N_sc_over_2);
        cfrOnFreqKHz = calCfrbyCir<Tscalar, Tcomplex>(scFreqKHz[localScOffset + (freqConvertType == 4 ? N_sc_Prbg/2 : scInPrbgIdx)], firNzLen, s_firNzDelayUs2Pi, s_timeChanLocal + localCirOffset, cfrPhaseShift);
    #else
        cfrOnFreqKHz = calCfrbyCir_v2<Tscalar, Tcomplex>(firNzLen, firNzDelayScFreq2Pi + (localScOffset + (freqConvertType == 4 ? N_sc_Prbg/2 : scInPrbgIdx)) * firNzLen, s_timeChanLocal + localCirOffset);
    #endif
        freqChanScLink[sc_start_offset + scInPrbgIdx].x = cfrOnFreqKHz.x;
        freqChanScLink[sc_start_offset + scInPrbgIdx].y = cfrOnFreqKHz.y;
        if (freqConvertType == 3 || freqConvertType == 4)
        {
            tempSum.x += cfrOnFreqKHz.x; // add Sc together
            tempSum.y += cfrOnFreqKHz.y;
        }
    }

    // convert SC CFR to Prbg CFR based on different freqConvertType
    switch (freqConvertType)
    {
        case 0:
            freqChanPrbg[prbg_offet] = freqChanScLink[sc_start_offset];
            break;

        case 1:
            if ((N_sc_Prbg/2) % scSampling == 0)
            {
                freqChanPrbg[prbg_offet] = freqChanScLink[sc_start_offset + N_sc_Prbg/2];
            }
            else // N_sc_Prbg/2 not in 0:scSampling:N_sc_Prbg
            {
            #if CAL_COS_SIN_IN_GPU
                cfrPhaseShift = cfrPhaseShiftTimeDelay * (localScOffset + N_sc_Prbg/2 - N_sc_over_2);
                cfrOnFreqKHz = calCfrbyCir<Tscalar, Tcomplex>(scFreqKHz[localScOffset + N_sc_Prbg/2], firNzLen, s_firNzDelayUs2Pi, s_timeChanLocal + localCirOffset, cfrPhaseShift);
            #else
                cfrOnFreqKHz = calCfrbyCir_v2<Tscalar, Tcomplex>(firNzLen, firNzDelayScFreq2Pi + (localScOffset + N_sc_Prbg/2) * firNzLen, s_timeChanLocal + localCirOffset);
            #endif
                freqChanPrbg[prbg_offet].x = cfrOnFreqKHz.x;
                freqChanPrbg[prbg_offet].y = cfrOnFreqKHz.y;
            }
            break;

        case 2: // N_sc_Prbg-1 not in 0:scSampling:N_sc_Prbg
            if ((N_sc_Prbg-1) % scSampling == 0)
            {
                freqChanPrbg[prbg_offet] = freqChanScLink[sc_start_offset + N_sc_Prbg - 1];
            }
            else
            {
            #if CAL_COS_SIN_IN_GPU
                cfrPhaseShift = cfrPhaseShiftTimeDelay * (localScOffset + N_sc_Prbg - 1 - N_sc_over_2);
                cfrOnFreqKHz = calCfrbyCir<Tscalar, Tcomplex>(scFreqKHz[localScOffset + N_sc_Prbg - 1], firNzLen, s_firNzDelayUs2Pi, s_timeChanLocal + localCirOffset, cfrPhaseShift);
            #else
                cfrOnFreqKHz = calCfrbyCir_v2<Tscalar, Tcomplex>(firNzLen, firNzDelayScFreq2Pi + (localScOffset + N_sc_Prbg - 1) * firNzLen, s_timeChanLocal + localCirOffset);
            #endif
                freqChanPrbg[prbg_offet].x = cfrOnFreqKHz.x;
                freqChanPrbg[prbg_offet].y = cfrOnFreqKHz.y;
            }
            break;

        case 3:
        case 4:
            // for the last Prbg, inverseNScPrbg = fastFadingDynDescr -> inverseNScLastPrbg
            freqChanPrbg[prbg_offet].x = tempSum.x * inverseNScPrbg;
            freqChanPrbg[prbg_offet].y = tempSum.y * inverseNScPrbg;
            break;

        default:
            printf("Error: Invalid freqConvertType %d!\n", freqConvertType);
            break;
    }
}

template<typename Tscalar, typename Tcomplex>
static __global__ void convertSctoPrbg(fastFadingDynDescr_t<Tscalar, Tcomplex> * fastFadingDynDescr)
{
    // GRID(nLink, m_scaleBsAntFreqChan, m_scaleUeAntFreqChan);
    // BLOCK(N_Prbg, nBsAnt/m_scaleBsAntFreqChan, nUeAnt/m_scaleUeAntFreqChan);

    uint32_t cidUidOffset = blockIdx.x; // linkIdx

    uint16_t N_sc = fastFadingDynDescr -> N_sc;
    uint16_t N_Prbg = fastFadingDynDescr -> N_Prbg;
    uint16_t N_sc_Prbg = (threadIdx.x < N_Prbg - 1) ? fastFadingDynDescr -> N_sc_Prbg : fastFadingDynDescr -> N_sc_last_Prbg; // number of Scs per Prbg
    // for the last Prbg, N_sc_Prbg = fastFadingDynDescr -> N_sc_last_Prbg SCs, may be smaller than fastFadingDynDescr -> N_sc_Prbg
    uint8_t freqConvertType = fastFadingDynDescr -> freqConvertType;
    uint16_t nBsAnt = fastFadingDynDescr -> nBsAnt;
    uint16_t nUeAnt = fastFadingDynDescr -> nUeAnt;
    float inverseNScPrbg = (threadIdx.x < N_Prbg - 1) ? fastFadingDynDescr -> inverseNScPrbg : fastFadingDynDescr -> inverseNScLastPrbg;
    Tcomplex * freqChanScLink = fastFadingDynDescr -> freqChanSc[cidUidOffset];
    Tcomplex * freqChanPrbg = fastFadingDynDescr -> freqChanPrbg;

    // prbg and ant index
    uint16_t prbgIdx  = threadIdx.x;
    uint16_t bsAntIdx = threadIdx.y + blockIdx.z * blockDim.y;
    uint16_t ueAntIdx = threadIdx.z + blockIdx.y * blockDim.z;
    uint16_t localScOffset = prbgIdx * fastFadingDynDescr -> N_sc_Prbg; // start sc index for a prbg
    uint32_t sc_avg_offset  = (ueAntIdx * nBsAnt + bsAntIdx) * N_sc + localScOffset;
    uint32_t prbg_offet = ((cidUidOffset * nUeAnt + ueAntIdx) * nBsAnt + bsAntIdx) * N_Prbg + prbgIdx;

    cuComplex tempSum = {0.0f, 0.0f};
    switch(freqConvertType)
    {
        case 0: // use first SC for CFR on the Prbg
            tempSum.x = freqChanScLink[sc_avg_offset].x;
            tempSum.y = freqChanScLink[sc_avg_offset].y;
            break;

        case 1: // use center SC for CFR on the Prbg, e.g., sc 6 for sc 0,1,2,...,11
            tempSum.x = freqChanScLink[sc_avg_offset + N_sc_Prbg/2].x;
            tempSum.y = freqChanScLink[sc_avg_offset + N_sc_Prbg/2].y;
            break;

        case 2: // use last SC for CFR on the Prbg
            tempSum.x = freqChanScLink[sc_avg_offset + N_sc_Prbg - 1].x;
            tempSum.y = freqChanScLink[sc_avg_offset + N_sc_Prbg - 1].y;
            break;

        case 3: // use average SC for on for the Prbg
            // for the last Prbg, inverseNScPrbg = fastFadingDynDescr -> inverseNScLastPrbg
            for(uint16_t scIdx= 0; scIdx< N_sc_Prbg; scIdx++)
            {
                tempSum.x = tempSum.x + static_cast<float>(freqChanScLink[sc_avg_offset].x);
                tempSum.y = tempSum.y + static_cast<float>(freqChanScLink[sc_avg_offset].y);
                sc_avg_offset++;
            }
            tempSum.x *= inverseNScPrbg;
            tempSum.y *= inverseNScPrbg;
            break;

        // case 4 will never be run in this kernel
        default:
            printf("Error: Invalid freqConvertType %d!\n", freqConvertType);
            break;
    }

    freqChanPrbg[prbg_offet].x = tempSum.x;
    freqChanPrbg[prbg_offet].y = tempSum.y;
}

template <typename Tscalar, typename Tcomplex>
static __global__ void processInputKernel_time(fastFadingDynDescr_t<Tscalar, Tcomplex> * fastFadingDynDescr, float refTime0, uint8_t enableSwapTxRx)
{
    // GRID(nLink, nBatch, 1);
    // BLOCK(m_procTxSampBlockSample, DL:nUeAnt/UL:nBsAnt or smaller, 1);
    // timeChan saved in [nLink, nBatch, nUeAnt, nBsAnt, firNzLen]

    // from launch configuration
    uint32_t cidUidOffset = blockIdx.x;
    uint16_t batchIdx = blockIdx.y;
    uint16_t chanCoeIdx = threadIdx.x;

    // from dyn descriptor
    uint16_t firNzLen = fastFadingDynDescr -> firNzLen;
    uint16_t firMaxLen = fastFadingDynDescr -> firMaxLen; // maxmimum delay tap
    uint16_t nBatch  = fastFadingDynDescr -> nBatch;
    uint32_t * batchCumLen = fastFadingDynDescr -> batchCumLen;
    float * tBatch = fastFadingDynDescr -> tBatch;
    uint32_t sigLenPerAnt = fastFadingDynDescr -> sigLenPerAnt;
    Tcomplex * timeChan = fastFadingDynDescr -> timeChan;
    Tcomplex * txSigIn =  fastFadingDynDescr -> txSigIn;
    Tcomplex * rxSigOut = fastFadingDynDescr -> rxSigOut;
    float cfoPhaseRef = 2.0f * M_PI * refTime0 * (fastFadingDynDescr -> cfoHz);
    float cfoPhaseSamp = fastFadingDynDescr -> cfoPhaseSamp;
    uint32_t nDelaySample = fastFadingDynDescr -> nDelaySample;
    uint16_t procTxSampBlockSample = fastFadingDynDescr -> procTxSampBlockSample;
    uint8_t saveAntPairSample = fastFadingDynDescr -> saveAntPairSample;
    Tcomplex * rxSigOutPerAntPair = fastFadingDynDescr -> rxSigOutPerAntPair;

    extern __shared__ char shareData[]; // all shared memory data pointer
    Tcomplex * chanCoeLocal = reinterpret_cast<Tcomplex *>(shareData); // firNzLen * nUeAnt * nBsAnt
    // only use shared memory for Tcomplex data type

    __shared__ uint16_t nBsAnt, nUeAnt;
    __shared__ uint16_t nTxAnt, nRxAnt;
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
    {
        nBsAnt = fastFadingDynDescr -> nBsAnt;
        nUeAnt = fastFadingDynDescr -> nUeAnt;
        if (enableSwapTxRx)  // UL
        {
            nTxAnt = nUeAnt;
            nRxAnt = nBsAnt;
        }
        else
        {
            nTxAnt = nBsAnt;
            nRxAnt = nUeAnt;
        }
    }
    // Use static
    // __shared__ Tcomplex txSigInBlock[BLOCK_SAMPLE_];
    // __shared__ Tcomplex chanCoeLocal[MAX_TX_RX_ANT_ * MAX_NZ_TAPS_];
    __shared__ uint16_t firNzIdx[MAX_NZ_TAPS_]; // no more than 24 NZ taps based on 3GPP 38.901
    if(chanCoeIdx < firNzLen && threadIdx.y == 0)
    {
        firNzIdx[chanCoeIdx] = fastFadingDynDescr -> firNzIdx[chanCoeIdx];
    }
    __syncthreads();

    // start processing tx signals
    for(uint16_t rxAntIdx = threadIdx.y; rxAntIdx < nRxAnt; rxAntIdx += blockDim.y)
    {
        // read input signal and provide output signal
        uint32_t antTxSampOffset, globalTxSampOffset; // antTxSampOffset: tx sample offset for this antenna, i.e., [0, sigLenPerAnt-1]; globalTxSampOffset = antTxSampOffset + offset for current link
        uint32_t antRxSampOffset, globalRxSampOffset; // antRxSampOffset: rx sample offset for this antenna, i.e., [0, sigLenPerAnt-1], consider delay; globalRxSampOffset = antRxSampOffset + offset for current link
        uint32_t globalAntPairRxSampOffset; // globalRxSampOffset = antRxSampOffset + offset for current link / tx
        cuComplex rxSigReg;
        cuComplex rxTxSigRegNoCfo; // rx-tx antenna pair sample without CFO
        Tcomplex tempTxSig, tempChanCoe;
        // copy channel coeficients firNzLen * blockDim.y into shared memory
        for(uint32_t chunkOffset = batchCumLen[batchIdx]; chunkOffset < batchCumLen[batchIdx + 1]; chunkOffset += procTxSampBlockSample) // each chunk for processing
        {
            antTxSampOffset = chunkOffset + chanCoeIdx;
            globalTxSampOffset = antTxSampOffset + cidUidOffset * (enableSwapTxRx ? nUeAnt : nBsAnt) * sigLenPerAnt;
            // get position to write rx sample with delay
            // same with 5GModel using cyclic shift ref: 5GModel/nr_matlab/channel/Channel.m
            // @note it could be the same to just use zero padding at the begining; no impact since it falls into CP position, will be discarded anyway
            if(antTxSampOffset < sigLenPerAnt - nDelaySample) // samples to be shift later by nDelaySample, new position no more than sigLenPerAnt
            {
                antRxSampOffset = antTxSampOffset + nDelaySample;
            }
            else if(antTxSampOffset < sigLenPerAnt) // samples padded to the begin, following cyclicshift
            {
                antRxSampOffset = antTxSampOffset + nDelaySample - sigLenPerAnt;
            }
            else // no opreations needed for samples
            {
                antRxSampOffset = antTxSampOffset;
            }
            if (enableSwapTxRx)
            {
                globalRxSampOffset = rxAntIdx * sigLenPerAnt + antRxSampOffset + cidUidOffset * nBsAnt * sigLenPerAnt;
                globalAntPairRxSampOffset = rxAntIdx * nUeAnt * sigLenPerAnt + antRxSampOffset + cidUidOffset * nBsAnt * nUeAnt * sigLenPerAnt; // will increase by sigLenPerAnt as iterate by tx ant
            }
            else
            {
                globalRxSampOffset = rxAntIdx * sigLenPerAnt + antRxSampOffset + cidUidOffset * nUeAnt * sigLenPerAnt;
                globalAntPairRxSampOffset = rxAntIdx * nBsAnt * sigLenPerAnt + antRxSampOffset + cidUidOffset * nBsAnt * nUeAnt * sigLenPerAnt; // will increase by sigLenPerAnt as iterate by tx ant
            }
            // calculate cfo rotation
            float cfoPhaseTotal = cfoPhaseRef + cfoPhaseSamp * antRxSampOffset;
            cuComplex cfoRotation;
            cfoRotation.x = cos(cfoPhaseTotal);
            cfoRotation.y = sin(cfoPhaseTotal);

            rxSigReg.x = 0.0f; rxSigReg.y = 0.0f;
            for(uint16_t txAntIdx = 0; txAntIdx < (enableSwapTxRx ? nUeAnt : nBsAnt); txAntIdx ++) // add sum of signals from each Tx antenna
            {
                rxTxSigRegNoCfo.x = 0.0f; rxTxSigRegNoCfo.y = 0.0f;
                // read current time chan into shared memory
                uint32_t chanReadOffsetIdx = threadIdx.y * firNzLen; // time chan for current bsAntIdx and part of ueAntIdx [0, 1, ..., blockDim.y-1]
                if(chanCoeIdx < firNzLen)
                {
                    uint32_t gloablCoeMatOffset = 0;
                    if (enableSwapTxRx)  // UL
                    {
                        gloablCoeMatOffset = (((cidUidOffset * nBatch + batchIdx) * nUeAnt + txAntIdx) * nBsAnt + rxAntIdx) * firNzLen; // time chan offset to read firNzLen * blockDim.y * 1 (batch) channel
                    }
                    else  // DL
                    {
                        gloablCoeMatOffset = (((cidUidOffset * nBatch + batchIdx) * nUeAnt + rxAntIdx) * nBsAnt + txAntIdx) * firNzLen; // time chan offset to read firNzLen * blockDim.y * 1 (batch) channel
                    }

                    chanCoeLocal[chanReadOffsetIdx + chanCoeIdx].x = timeChan[gloablCoeMatOffset + chanCoeIdx].x;
                    chanCoeLocal[chanReadOffsetIdx + chanCoeIdx].y = timeChan[gloablCoeMatOffset + chanCoeIdx].y;
                }
                __syncthreads(); // wait for time chan to be read
                // apply the channel coes to tx symbols, add to local register
                if(antTxSampOffset < sigLenPerAnt)
                {
                    for(uint16_t chanSumIdx = 0; chanSumIdx < firNzLen; chanSumIdx++)
                    {
                        if(antTxSampOffset >= firNzIdx[chanSumIdx])
                        {
                            tempTxSig = txSigIn[globalTxSampOffset - firNzIdx[chanSumIdx]]; // current TxSig for sum
                        }
                        else
                        {
                            tempTxSig.x = 0.0f;
                            tempTxSig.y = 0.0f;
                        }

                        tempChanCoe = chanCoeLocal[chanReadOffsetIdx + chanSumIdx]; // current chanCoe for sum

                        // calculate rx-tx ant pair sample before adding CFO, rxSigRegNoCfo += tempTxSig * tempChanCoe
                        rxTxSigRegNoCfo.x = rxTxSigRegNoCfo.x + static_cast<float>(tempTxSig.x * tempChanCoe.x - tempTxSig.y * tempChanCoe.y);
                        rxTxSigRegNoCfo.y = rxTxSigRegNoCfo.y + static_cast<float>(tempTxSig.x * tempChanCoe.y + tempTxSig.y * tempChanCoe.x);
                    }
                }

                if (saveAntPairSample == 1) // add CFO per rx-tx sample, sum to get the actual rx sample
                {
                    cuComplex rxTxSigRegWithCfo;
                    rxTxSigRegWithCfo.x = rxTxSigRegNoCfo.x * cfoRotation.x - rxTxSigRegNoCfo.y * cfoRotation.y;
                    rxTxSigRegWithCfo.y = rxTxSigRegNoCfo.x * cfoRotation.y + rxTxSigRegNoCfo.y * cfoRotation.x;

                    // save the rx-tx antenna pair data sample to global memory
                    if(antRxSampOffset < sigLenPerAnt)
                    {
                        rxSigOutPerAntPair[globalAntPairRxSampOffset].x = rxTxSigRegWithCfo.x;
                        rxSigOutPerAntPair[globalAntPairRxSampOffset].y = rxTxSigRegWithCfo.y;
                    }

                    // sum to get the actual rx sample
                    rxSigReg.x = rxSigReg.x + rxTxSigRegWithCfo.x;
                    rxSigReg.y = rxSigReg.y + rxTxSigRegWithCfo.y;
                }
                else // sum the data before adding CFO, will add CFO rotation later after sum to only use one multiplication
                {
                    rxSigReg.x = rxSigReg.x + rxTxSigRegNoCfo.x;
                    rxSigReg.y = rxSigReg.y + rxTxSigRegNoCfo.y;
                }
                // update gloal channel index
                globalTxSampOffset += sigLenPerAnt;
                globalAntPairRxSampOffset += sigLenPerAnt;
                // if(threadIdx.x + threadIdx.y + threadIdx.z == 0)
                // {
                //     printf("txAntIdx = %d, rxAntIdx = %d, globalTxSampOffset = %d, globalAntPairRxSampOffset = %d\n", txAntIdx, rxAntIdx, globalTxSampOffset, globalAntPairRxSampOffset);
                // }
                __syncthreads(); // wait for the processing of samples to finish
            }
            // save the actual rx data sample to global memory
            if(antRxSampOffset < sigLenPerAnt)
            {
                if (saveAntPairSample == 1) // directly save actual rx sample
                {
                    rxSigOut[globalRxSampOffset].x = rxSigReg.x;
                    rxSigOut[globalRxSampOffset].y = rxSigReg.y;
                }
                else // add CFO and save actual rx sample
                {
                    rxSigOut[globalRxSampOffset].x = rxSigReg.x * cfoRotation.x - rxSigReg.y * cfoRotation.y;
                    rxSigOut[globalRxSampOffset].y = rxSigReg.x * cfoRotation.y + rxSigReg.y * cfoRotation.x;
                }
            }
        }
    }
}

template <typename Tscalar, typename Tcomplex>
static __global__ void processInputKernel_freq(fastFadingDynDescr_t<Tscalar, Tcomplex> * fastFadingDynDescr, float refTime0, uint8_t enableSwapTxRx)
{
    // GRID(nLink, nBatch, 1);
    // BLOCK(m_procTxSampBlockSample, 1, 1);
    // freqChan saved in [nLink, nBatch, nUeAnt, nBsAnt, N_sc]
    // txSigIn saved in  [nLink, nBsAnt, nBatch, N_sc] for DL and [nLink, nUeAnt, nBatch, N_sc] for UL, row major
    // txSigOut saved in [nLink, nUeAnt, nBatch, N_sc] for DL and [nLink, nBsAnt, nBatch, N_sc] for UL, row major

    // from launch configuration
    uint32_t cidUidOffset = blockIdx.x;
    uint16_t batchIdx = blockIdx.y;

    // from dyn descriptor
    uint16_t nBsAnt = fastFadingDynDescr -> nBsAnt;
    uint16_t nUeAnt = fastFadingDynDescr -> nUeAnt;
    uint16_t nBatch  = fastFadingDynDescr -> nBatch;
    uint16_t N_sc = fastFadingDynDescr -> N_sc;
    uint16_t N_Prbg = fastFadingDynDescr -> N_Prbg;
    uint16_t N_sc_Prbg = (threadIdx.x < N_Prbg - 1) ? fastFadingDynDescr -> N_sc_Prbg : fastFadingDynDescr -> N_sc_last_Prbg; // number of Scs per Prbg
    uint32_t sigLenPerAnt = fastFadingDynDescr -> sigLenPerAnt;
    Tcomplex * freqChanSc = fastFadingDynDescr -> freqChanSc[cidUidOffset];
    Tcomplex * txSigIn =  fastFadingDynDescr -> txSigIn;
    Tcomplex * rxSigOut = fastFadingDynDescr -> rxSigOut;

    Tcomplex tmpSig, tmpFreqChan;  // read tx sample and freq chan on Sc
    Tcomplex rxSigReg;  // accumulate for rx sample
    // start processing tx signals
    if (enableSwapTxRx)  // UL
    {
        uint32_t txSampCidUidOffset = cidUidOffset * nUeAnt * nBatch * N_sc;
        uint32_t rxSampCidUidOffset = cidUidOffset * nBsAnt * nBatch * N_sc;
        for(uint16_t bsAntIdx = 0; bsAntIdx < nBsAnt; bsAntIdx ++)
        {
            // read input signal and provide output signal
            for(uint32_t sampleIdx = threadIdx.x; sampleIdx < N_sc; sampleIdx += blockDim.x) // each chunk for processing
            {
                rxSigReg.x = 0.0f; rxSigReg.y = 0.0f;
                #pragma unroll
                for(uint16_t ueAntIdx = 0; ueAntIdx < nUeAnt; ueAntIdx ++) // add sum of signals from each Tx antenna
                {
                    tmpSig = txSigIn[txSampCidUidOffset + (ueAntIdx * nBatch + batchIdx) * N_sc + sampleIdx];
                    tmpFreqChan = freqChanSc[((batchIdx * nUeAnt + ueAntIdx) * nBsAnt + bsAntIdx) * N_sc + sampleIdx];
                    rxSigReg.x = rxSigReg.x + tmpSig.x * tmpFreqChan.x - tmpSig.y * tmpFreqChan.y;
                    rxSigReg.y = rxSigReg.y + tmpSig.x * tmpFreqChan.y + tmpSig.y * tmpFreqChan.x;
                }
                rxSigOut[rxSampCidUidOffset + (bsAntIdx * nBatch + batchIdx) * N_sc + sampleIdx] = rxSigReg;  // save rx sample
            }
        }
    }
    else  // DL
    {
        uint32_t txSampCidUidOffset = cidUidOffset * nBsAnt * nBatch * N_sc;
        uint32_t rxSampCidUidOffset = cidUidOffset * nUeAnt * nBatch * N_sc;
        for(uint16_t ueAntIdx = 0; ueAntIdx < nUeAnt; ueAntIdx ++)
        {
            // read input signal and provide output signal
            for(uint32_t sampleIdx = threadIdx.x; sampleIdx < N_sc; sampleIdx += blockDim.x) // each chunk for processing
            {
                rxSigReg.x = 0.0f; rxSigReg.y = 0.0f;
                #pragma unroll
                for(uint16_t bsAntIdx = 0; bsAntIdx < nBsAnt; bsAntIdx ++) // add sum of signals from each Tx antenna
                {
                    tmpSig = txSigIn[txSampCidUidOffset + (bsAntIdx * nBatch + batchIdx) * N_sc + sampleIdx];
                    tmpFreqChan = freqChanSc[((batchIdx * nUeAnt + ueAntIdx) * nBsAnt + bsAntIdx) * N_sc + sampleIdx];
                    rxSigReg.x = rxSigReg.x + tmpSig.x * tmpFreqChan.x - tmpSig.y * tmpFreqChan.y;
                    rxSigReg.y = rxSigReg.y + tmpSig.x * tmpFreqChan.y + tmpSig.y * tmpFreqChan.x;
                }
                rxSigOut[rxSampCidUidOffset + (ueAntIdx * nBatch + batchIdx) * N_sc + sampleIdx] = rxSigReg;  // save rx sample
            }
        }
    }
}


template <typename Tscalar, typename Tcomplex>
static __global__ void processInputKernel_freq_columnMajor(fastFadingDynDescr_t<Tscalar, Tcomplex> * fastFadingDynDescr, float refTime0, uint8_t enableSwapTxRx)
{
    // GRID(nLink, nBatch, 1);
    // BLOCK(m_procTxSampBlockSample, 1, 1);
    // freqChan saved in [nLink, nBatch, nUeAnt, nBsAnt, N_sc]
    // txSigIn saved in  [nLink, nBsAnt, nBatch, N_sc] for DL and [nLink, nUeAnt, nBatch, N_sc] for UL, column major
    // txSigOut saved in [nLink, nUeAnt, nBatch, N_sc] for DL and [nLink, nBsAnt, nBatch, N_sc] for UL, row major
    // only difference is in reading the tx sample

    // from launch configuration
    uint32_t cidUidOffset = blockIdx.x;
    uint16_t batchIdx = blockIdx.y;

    // from dyn descriptor
    uint16_t nBsAnt = fastFadingDynDescr -> nBsAnt;
    uint16_t nUeAnt = fastFadingDynDescr -> nUeAnt;
    uint16_t nBatch  = fastFadingDynDescr -> nBatch;
    uint16_t N_sc = fastFadingDynDescr -> N_sc;
    uint16_t N_Prbg = fastFadingDynDescr -> N_Prbg;
    uint16_t N_sc_Prbg = (threadIdx.x < N_Prbg - 1) ? fastFadingDynDescr -> N_sc_Prbg : fastFadingDynDescr -> N_sc_last_Prbg; // number of Scs per Prbg
    uint32_t sigLenPerAnt = fastFadingDynDescr -> sigLenPerAnt;
    Tcomplex * freqChanSc = fastFadingDynDescr -> freqChanSc[cidUidOffset];
    Tcomplex * txSigIn =  fastFadingDynDescr -> txSigIn;
    Tcomplex * rxSigOut = fastFadingDynDescr -> rxSigOut;

    Tcomplex tmpSig, tmpFreqChan;  // read tx sample and freq chan on Sc
    Tcomplex rxSigReg;  // accumulate for rx sample
    // start processing tx signals
    if (enableSwapTxRx)  // UL
    {
        uint32_t rxSampCidUidOffset = cidUidOffset * nBsAnt * nBatch * N_sc;
        for(uint16_t bsAntIdx = 0; bsAntIdx < nBsAnt; bsAntIdx ++)
        {
            // read input signal and provide output signal
            for(uint32_t sampleIdx = threadIdx.x; sampleIdx < N_sc; sampleIdx += blockDim.x) // each chunk for processing
            {
                rxSigReg.x = 0.0f; rxSigReg.y = 0.0f;
                #pragma unroll
                for(uint16_t ueAntIdx = 0; ueAntIdx < nUeAnt; ueAntIdx ++) // add sum of signals from each Tx antenna
                {
                    tmpSig = txSigIn[((sampleIdx * nBatch + batchIdx) * nUeAnt + ueAntIdx) * gridDim.x + cidUidOffset];  // gridDim.x = nLink
                    tmpFreqChan = freqChanSc[((batchIdx * nUeAnt + ueAntIdx) * nBsAnt + bsAntIdx) * N_sc + sampleIdx];
                    rxSigReg.x = rxSigReg.x + tmpSig.x * tmpFreqChan.x - tmpSig.y * tmpFreqChan.y;
                    rxSigReg.y = rxSigReg.y + tmpSig.x * tmpFreqChan.y + tmpSig.y * tmpFreqChan.x;
                }
                rxSigOut[rxSampCidUidOffset + (bsAntIdx * nBatch + batchIdx) * N_sc + sampleIdx] = rxSigReg;  // save rx sample
            }
        }
    }
    else  // DL
    {
        uint32_t rxSampCidUidOffset = cidUidOffset * nUeAnt * nBatch * N_sc;
        for(uint16_t ueAntIdx = 0; ueAntIdx < nUeAnt; ueAntIdx ++)
        {
            // read input signal and provide output signal
            for(uint32_t sampleIdx = threadIdx.x; sampleIdx < N_sc; sampleIdx += blockDim.x) // each chunk for processing
            {
                rxSigReg.x = 0.0f; rxSigReg.y = 0.0f;
                #pragma unroll
                for(uint16_t bsAntIdx = 0; bsAntIdx < nBsAnt; bsAntIdx ++) // add sum of signals from each Tx antenna
                {
                    tmpSig = txSigIn[((sampleIdx * nBatch + batchIdx) * nBsAnt + bsAntIdx) * gridDim.x + cidUidOffset];  // gridDim.x = nLink
                    tmpFreqChan = freqChanSc[((batchIdx * nUeAnt + ueAntIdx) * nBsAnt + bsAntIdx) * N_sc + sampleIdx];
                    rxSigReg.x = rxSigReg.x + tmpSig.x * tmpFreqChan.x - tmpSig.y * tmpFreqChan.y;
                    rxSigReg.y = rxSigReg.y + tmpSig.x * tmpFreqChan.y + tmpSig.y * tmpFreqChan.x;
                }
                rxSigOut[rxSampCidUidOffset + (ueAntIdx * nBatch + batchIdx) * N_sc + sampleIdx] = rxSigReg;  // save rx sample
            }
        }
    }
}