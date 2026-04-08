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

#include "cdl_chan.cuh"
#include "cdl_dpa_table.h"


template <typename Tscalar, typename Tcomplex> 
cdlChan<Tscalar, Tcomplex>::cdlChan(cdlConfig_t * cdlCfg, uint16_t randSeed, cudaStream_t strm)
{
    // Assignments without "this->" (directly accessible or local variables)
    m_cdlCfg                 = cdlCfg;
    m_nRay                   = cdlCfg->numRay;
    m_delayProfile           = cdlCfg->delayProfile;
    float tSample =  1.0f / (cdlCfg -> f_samp);
    ASSERT(cdlCfg -> scSampling > 0, "scSampling must be a positive integer");
    ASSERT(cdlCfg -> f_samp == 4096 * cdlCfg -> scSpacingHz, "mismated sampling frequency and SC spacing, f_samp must be equal to 4096 * scSpacingHz");
    ASSERT(cdlCfg -> procSigFreq == 0 || cdlCfg -> runMode == 2, "processing tx samples in frequency domain requires runMode 2");

    // Assignments with "this->" (explicitly referencing base class members)
    this->m_strm              = strm;
    this->m_nCell             = cdlCfg->nCell;
    this->m_nUe               = cdlCfg->nUe;
    this->m_nLink             = this->m_nCell * this->m_nUe;
    this->m_nBsAnt            = std::accumulate(cdlCfg->bsAntSize.begin(), cdlCfg->bsAntSize.end(), 1U, std::multiplies<uint16_t>());
    this->m_nUeAnt            = std::accumulate(cdlCfg->ueAntSize.begin(), cdlCfg->ueAntSize.end(), 1U, std::multiplies<uint16_t>());
    this->m_runMode           = cdlCfg->runMode;
    this->m_maxDopplerShift   = cdlCfg->maxDopplerShift;
    this->m_sigLenPerAnt      = cdlCfg->sigLenPerAnt;
    this->m_delay             = cdlCfg->delay;
    this->m_cfoHz             = cdlCfg->cfoHz;
    this->m_f_samp            = cdlCfg->f_samp;
    this->m_fBatch            = cdlCfg->fBatch;
    this->m_batchLen          = cdlCfg->batchLen;
    this->m_txSigIn           = cdlCfg->txSigIn;
    this->m_mimoCorrMat       = nullptr;                  // TODO: 'Low' correlation for now
    this->m_N_sc              = cdlCfg->N_sc;
    this->m_N_sc_Prbg         = cdlCfg->N_sc_Prbg;
    this->m_scSpacingHz       = cdlCfg->scSpacingHz;
    this->m_freqConvertType   = cdlCfg->freqConvertType;
    this->m_scSampling        = cdlCfg->scSampling;    
    this->m_procSigFreq       = cdlCfg -> procSigFreq;
    this->m_sigLenPerAnt      = cdlCfg -> sigLenPerAnt;
    this->m_saveAntPairSample = cdlCfg -> saveAntPairSample;
    this->m_txSigIn           = reinterpret_cast<Tcomplex*>(cdlCfg -> txSigIn);

    /*-------------------   read dpa and pcp from tables in cdl_dpa_table.h   -------------------*/
    m_delaySpread = cdlCfg -> delaySpread; // customize delay spread
    switch(m_delayProfile)  // read from TS 38.901 Table 7.7.1-1 ~ Table 7.7.1-5
    {
        case 'A':
            m_dpa = DPA_A.data();
            m_pcp = PCP_A.data();
            m_numTaps = DPA_A.size()/6;
            m_LosTap = false;
        break;

        case 'B':
            m_dpa = DPA_B.data();
            m_pcp = PCP_B.data();
            m_numTaps = DPA_B.size()/6;
            m_LosTap = false;
        break;

        case 'C':
            m_dpa = DPA_C.data();
            m_pcp = PCP_C.data();
            m_numTaps = DPA_C.size()/6;
            m_LosTap = false;
        break;

        case 'D':
            m_dpa = DPA_D.data();
            m_pcp = PCP_D.data();
            m_numTaps = DPA_D.size()/6;
            m_LosTap = true;
        break;

        case 'E':
            m_dpa = DPA_E.data();
            m_pcp = PCP_E.data();
            m_numTaps = DPA_E.size()/6;
            m_LosTap = true;
        break;

        default:
            m_dpa = DPA_A.data();
            m_pcp = PCP_A.data();
            m_numTaps = DPA_A.size()/6;
            m_LosTap = false;
        break;
    }
    // TODO: currently does not support CDL-D and CDL-E with LOS path, will added later
    if(m_LosTap)
    {
        printf("ERROR: CDL with LOS path is not supported yet! \n");
        exit(EXIT_FAILURE);
    }
    /*-------------------   calculate FIR based on cdl config   -------------------*/
    // buffer for FIR filter 
    // Example:
    // Assume:
    //   m_numTaps = 4
    //   m_dpa = {
    //       0.1, -10, 0, 0, 0, 0,  // Delay (us), Power (dB), and unused values for tap 1
    //       0.3, -20, 0, 0, 0, 0,  // Delay (us), Power (dB), and unused values for tap 2
    //       0.31, -15, 0, 0, 0, 0, // Delay (us), Power (dB), and unused values for tap 3
    //       1.0, -25, 0, 0, 0, 0   // Delay (us), Power (dB), and unused values for tap 4
    //   };
    //
    //   m_f_samp = 1e6             // Sampling frequency in Hz
    //   m_delaySpread = 1.0        // Scaling factor for delay spread
    //
    // Output:
    //   m_firNzIdx = {1, 3, 10}    // Unique FIR indices corresponding to delays
    //   m_firNzPw = {10^-1, 10^-2, 10^-1.5, 10^-2.5} // individual power for each tap
    //   m_firNzTapMap = {0, 1, 1, 2}
    //     - Maps each original tap to its corresponding FIR index
    //     - Tap indices `1` and `2` will be combined into the same FIR index
    //   m_firNzLen = 3             // Total number of unique non-zero taps in the final CIR
    //
    // Notes:
    //   - The second and third taps are combined into one tap because their delays map to the same FIR index (`3`).
    //   - The final FIR will have **3 unique taps**.

    for(uint8_t tapIdx=0; tapIdx<m_numTaps; tapIdx++)
    {
        this -> m_firNzPw.push_back(pow(10, m_dpa[tapIdx*6 + 1] * 0.1f));
        uint16_t firNzIdx = round(m_dpa[tapIdx*6] * 1e-9 * this -> m_f_samp * m_delaySpread);
        m_firNzTapMap.push_back(firNzIdx);  // tap index for each tap in pdp table
    }
    // unique tap index for CIR
    std::vector<uint16_t> temp = m_firNzTapMap;
    std::sort(temp.begin(), temp.end());
    auto uniqueEnd = std::unique(temp.begin(), temp.end());
    this -> m_firNzIdx.assign(temp.begin(), uniqueEnd);

    std::unordered_map<uint16_t, size_t> valueToIndex;
    for (size_t i = 0; i < this -> m_firNzIdx.size(); ++i) {
        valueToIndex[this -> m_firNzIdx[i]] = i;
    }
    for (auto& elem : m_firNzTapMap) {
        elem = valueToIndex[elem];
    }
    this -> m_firNzLen = this -> m_firNzIdx.size();  // number of nz taps in the final CIR

    // normalize firPW
    float sum_firNzPw = 0.0f;
    for(auto x : this -> m_firNzPw)
    {
        sum_firNzPw += x;
    }
    // normalization targe: 1/sqrt(nRay)
    sum_firNzPw = 1.0f/sum_firNzPw;
    for(int firNzIdx = 0; firNzIdx < m_numTaps; firNzIdx ++)
    {
        // take sqrt to be multiplied for chan coe
        this -> m_firNzPw[firNzIdx] = sqrt((this -> m_firNzPw[firNzIdx] / m_nRay * sum_firNzPw));
    }

    // setup CDL dynamic descrptor
    m_cdlDynDescrCpu = new cdlDynDescr_t<Tscalar, Tcomplex>;
    m_cdlDynDescrCpu -> LosTap = m_LosTap;
    m_cdlDynDescrCpu -> nTaps = m_numTaps;
    m_cdlDynDescrCpu -> nRay = m_nRay;
    // copy antenna seetings from cdlCfg
    for (int i = 0; i < 5; ++i) {
        m_cdlDynDescrCpu -> bsAntSize[i] = cdlCfg -> bsAntSize[i];
        m_cdlDynDescrCpu -> ueAntSize[i] = cdlCfg -> ueAntSize[i];
    }
    for (int i = 0; i < 4; ++i) {
        m_cdlDynDescrCpu -> bsAntSpacing[i] = cdlCfg -> bsAntSpacing[i];
        m_cdlDynDescrCpu -> ueAntSpacing[i] = cdlCfg -> ueAntSpacing[i];
    }
    for (int i = 0; i < 2; ++i) {
        m_cdlDynDescrCpu -> bsAntPolarAngles[i] = cdlCfg -> bsAntPolarAngles[i];
        m_cdlDynDescrCpu -> ueAntPolarAngles[i] = cdlCfg -> ueAntPolarAngles[i];
    }
    m_cdlDynDescrCpu -> bsAntPattern = cdlCfg -> bsAntPattern;
    m_cdlDynDescrCpu -> ueAntPattern = cdlCfg -> ueAntPattern;
    // copy moving direction
    m_cdlDynDescrCpu -> vDirection[0] = cdlCfg -> vDirection[0] * D2PI;
    m_cdlDynDescrCpu -> vDirection[1] = cdlCfg -> vDirection[1] * D2PI;

    // allocate and copy dpa
    CHECK_CUDAERROR(cudaMalloc((void**) &(m_cdlDynDescrCpu -> dpa), m_numTaps * 6 * sizeof(float)));
    CHECK_CUDAERROR(cudaMemcpyAsync(m_cdlDynDescrCpu -> dpa, m_dpa, m_numTaps * 6 * sizeof(float), cudaMemcpyHostToDevice, this -> m_strm));

    // allocate and copy pcp
    std::vector<float> pcpVec(5);
    pcpVec[0] = m_pcp[0];
    pcpVec[1] = m_pcp[1];
    pcpVec[2] = m_pcp[2];
    pcpVec[3] = m_pcp[3];
    pcpVec[4] = 1 / sqrt(pow(10, m_pcp[4] * 0.1f));  // pre calculate 1/sqrt(XPR in linear scale) 
    CHECK_CUDAERROR(cudaMalloc((void**) &(m_cdlDynDescrCpu -> pcp), 5 * sizeof(float)));
    CHECK_CUDAERROR(cudaMemcpyAsync(m_cdlDynDescrCpu -> pcp, pcpVec.data(), 5 * sizeof(float), cudaMemcpyHostToDevice, this -> m_strm));
    
    // allocate and copy rayOffsetAngles
    CHECK_CUDAERROR(cudaMalloc((void**) &(m_cdlDynDescrCpu -> rayOffsetAngles), 20 * sizeof(float)));
    CHECK_CUDAERROR(cudaMemcpyAsync(m_cdlDynDescrCpu -> rayOffsetAngles, rayOffsetAngles.data(), 20 * sizeof(float), cudaMemcpyHostToDevice, this -> m_strm));

    // allocate and copy index map
    CHECK_CUDAERROR(cudaMalloc((void**) &(m_cdlDynDescrCpu -> firNzTapMap), m_numTaps * sizeof(uint16_t)));
    CHECK_CUDAERROR(cudaMemcpyAsync(m_cdlDynDescrCpu -> firNzTapMap, m_firNzTapMap.data(), m_numTaps * sizeof(uint16_t), cudaMemcpyHostToDevice, this -> m_strm));

    // for curand states
    curandCreateGeneratorHost(&m_Rng, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(m_Rng, randSeed); //random seed applied here
    // allocate memory for random phi
    m_randPhiSizePerLink = 4 * m_numTaps * m_nRay;
    CHECK_CUDAERROR(cudaMalloc((void**) &(m_cdlDynDescrCpu -> phiRand), this -> m_nLink * sizeof(Tscalar*)));
    this -> m_gpuDataUsageByte += this -> m_nLink * sizeof(Tscalar*);
    m_h_devicePhiRandPerLinkPtr = (Tscalar **)malloc(this -> m_nLink * sizeof(Tscalar*));
    for(uint32_t linkIdx = 0; linkIdx < this -> m_nLink; linkIdx ++)
    {
        CHECK_CUDAERROR(cudaMalloc((void**) &(m_h_devicePhiRandPerLinkPtr[linkIdx]), m_randPhiSizePerLink * sizeof(Tscalar)));
        this -> m_gpuDataUsageByte += m_randPhiSizePerLink * sizeof(Tscalar);
    }
    CHECK_CUDAERROR(cudaMemcpyAsync(m_cdlDynDescrCpu -> phiRand, m_h_devicePhiRandPerLinkPtr, this -> m_nLink * sizeof(Tscalar*), cudaMemcpyHostToDevice, this -> m_strm));

    // allocate memory for ray coupling
    m_randRayCouplingSizePerLink = m_numTaps * m_nRay * 4;
    CHECK_CUDAERROR(cudaMalloc((void**) &(m_cdlDynDescrCpu -> rayCouplingRand), this -> m_nLink * sizeof(uint16_t*)));
    this -> m_gpuDataUsageByte += this -> m_nLink * sizeof(uint16_t*);
    m_h_deviceRayCouplingRandPerLinkPtr = (uint16_t **)malloc(this -> m_nLink * sizeof(uint16_t*));
    for(uint32_t linkIdx = 0; linkIdx < this -> m_nLink; linkIdx ++)
    {
        CHECK_CUDAERROR(cudaMalloc((void**) &(m_h_deviceRayCouplingRandPerLinkPtr[linkIdx]), m_randRayCouplingSizePerLink * sizeof(uint16_t)));
        this -> m_gpuDataUsageByte += m_randRayCouplingSizePerLink * sizeof(uint16_t);
    }
    CHECK_CUDAERROR(cudaMemcpyAsync(m_cdlDynDescrCpu -> rayCouplingRand, m_h_deviceRayCouplingRandPerLinkPtr, this -> m_nLink * sizeof(uint16_t*), cudaMemcpyHostToDevice, this -> m_strm));

    // allocate long term parameters
    CHECK_CUDAERROR(cudaMalloc((void**) &(m_cdlDynDescrCpu -> multi5Term), this -> m_nLink * this -> m_nUeAnt * this -> m_nBsAnt * m_numTaps * m_nRay * sizeof(Tcomplex)));
    CHECK_CUDAERROR(cudaMalloc((void**) &(m_cdlDynDescrCpu -> rheadVbar), this -> m_nLink * this -> m_nUeAnt * m_numTaps * m_nRay * sizeof(Tscalar)));

    this -> setup();  // call setup in base class

    // copy dyndescriptor to GPU
    CHECK_CUDAERROR(cudaMalloc((void**)&(m_cdlDynDescrGpu), sizeof(cdlDynDescr_t<Tscalar, Tcomplex>)));
    this -> m_gpuDataUsageByte += sizeof(cdlDynDescr_t<Tscalar, Tcomplex>);
    cudaMemcpyAsync(m_cdlDynDescrGpu, m_cdlDynDescrCpu, sizeof(cdlDynDescr_t<Tscalar, Tcomplex>), cudaMemcpyHostToDevice, this -> m_strm);

    // initialization
    reset();

    CHECK_CUDAERROR(cudaStreamSynchronize(this -> m_strm)); // finish constructor
    // check if any errors in creating CDL channel class
    cudaError_t cuda_error = cudaGetLastError();
    if(cuda_error != cudaSuccess) 
    {
        fprintf(stderr, "CUDA error in CDL channel constructor: %s\n", cudaGetErrorString(cuda_error));
        exit(EXIT_FAILURE);
    }
}

template <typename Tscalar, typename Tcomplex> 
cdlChan<Tscalar, Tcomplex>::~cdlChan()
{
    for(uint32_t linkIdx = 0; linkIdx < this -> m_nLink; linkIdx ++)
    {
        cudaFree(m_h_devicePhiRandPerLinkPtr[linkIdx]);
        cudaFree(m_h_deviceRayCouplingRandPerLinkPtr[linkIdx]);
    }
    free(m_h_devicePhiRandPerLinkPtr);
    free(m_h_deviceRayCouplingRandPerLinkPtr);
    cudaFree(m_cdlDynDescrCpu -> phiRand);
    cudaFree(m_cdlDynDescrCpu -> rayCouplingRand);

    cudaFree(m_cdlDynDescrCpu -> dpa);
    cudaFree(m_cdlDynDescrCpu -> pcp);
    cudaFree(m_cdlDynDescrCpu -> firNzTapMap);
    cudaFree(m_cdlDynDescrCpu -> rayOffsetAngles);

    cudaFree(m_cdlDynDescrCpu -> multi5Term);
    cudaFree(m_cdlDynDescrCpu -> rheadVbar);

    curandDestroyGenerator(m_Rng);

    cudaFree(m_cdlDynDescrGpu);
    delete m_cdlDynDescrCpu;
}

template <typename Tscalar, typename Tcomplex> 
void cdlChan<Tscalar, Tcomplex>::updateTapPathRand() // curand does not support half precision yet
{
    curandStatus_t curandResult;
    std::vector<float> hostRandomPhiFloat(m_randPhiSizePerLink);
    if(typeid(Tcomplex) == typeid(__half2)) // need to convert to half precision
    {
        std::vector<Tscalar> hostRandomNumbersHalf(m_randPhiSizePerLink);
        for(uint32_t linkIdx = 0; linkIdx < this -> m_nLink; linkIdx ++)
        {
            curandResult = curandGenerateUniform(m_Rng, hostRandomPhiFloat.data(), m_randPhiSizePerLink); // for phase 
            if (curandResult != CURAND_STATUS_SUCCESS) 
            {
                std::string msg("Could not generate random number phiRand: ");
                throw std::runtime_error(msg);
            }
            for(uint32_t i=0; i<m_randPhiSizePerLink; i++)
            {
                hostRandomPhiFloat[i] = 2 * M_PI * (hostRandomPhiFloat[i] - 0.5);  // random between [-pi, pi)
                hostRandomNumbersHalf[i] = __float2half(hostRandomPhiFloat[i]);
            }
            cudaMemcpyAsync(m_h_devicePhiRandPerLinkPtr[linkIdx], hostRandomNumbersHalf.data(), m_randPhiSizePerLink*sizeof(Tscalar), cudaMemcpyHostToDevice, this -> m_strm);
        }
    }
    else if(typeid(Tcomplex) == typeid(cuComplex))
    {
        for(uint32_t linkIdx = 0; linkIdx < this -> m_nLink; linkIdx ++)
        {
            curandResult = curandGenerateUniform(m_Rng, hostRandomPhiFloat.data(), m_randPhiSizePerLink); // for phase 
            if (curandResult != CURAND_STATUS_SUCCESS) 
            {
                std::string msg("Could not generate random number phiRand: ");
                throw std::runtime_error(msg);
            }
            for(uint32_t i=0; i<m_randPhiSizePerLink; i++)
            {
                hostRandomPhiFloat[i] = 2 * M_PI * (hostRandomPhiFloat[i] - 0.5);  // random between [-pi, pi)
            }
            cudaMemcpyAsync(m_h_devicePhiRandPerLinkPtr[linkIdx], hostRandomPhiFloat.data(), m_randPhiSizePerLink*sizeof(Tscalar), cudaMemcpyHostToDevice, this -> m_strm);
        }
    }
    else
    {
        fprintf(stderr, "Unsupported data type!\n");
        exit(EXIT_FAILURE);
    }
    cudaStreamSynchronize(this -> m_strm);
}

template <typename Tscalar, typename Tcomplex> 
void cdlChan<Tscalar, Tcomplex>::updateRayCouplingRand() // curand does not support half precision yet
{
    curandStatus_t curandResult;
    std::vector<float> hostRandFloat(m_randRayCouplingSizePerLink);
    std::vector<uint16_t> sequence(m_randRayCouplingSizePerLink);  // results of random ray coupling

    // Prepare storage for all permutations
    std::vector<std::vector<uint16_t>> permutations;

    for(uint32_t linkIdx = 0; linkIdx < this -> m_nLink; linkIdx ++)
    {
        // Generate random numbers for shuffling
        curandResult = curandGenerateUniform(m_Rng, hostRandFloat.data(), m_randRayCouplingSizePerLink);
        if (curandResult != CURAND_STATUS_SUCCESS) 
        {
            std::string msg("Could not generate random number RayCouplingRand: ");
            throw std::runtime_error(msg);
        }
        for(uint32_t tapIdx = 0; tapIdx < m_numTaps * 4; tapIdx ++)
        {
            uint16_t* currentSeq = &sequence[tapIdx * m_nRay]; // Pointer to the current tap's section
            // Initialize the sequence 0, 1, ..., m_nRay-1
            std::vector<uint16_t> sequence(m_nRay);
            for (uint16_t j = 0; j < m_nRay; ++j) {
                currentSeq[j] = j;
            }

            // Apply Fisher-Yates Shuffle using the random numbers
            for (int j = m_nRay - 1; j > 0; --j) {
                uint16_t k = static_cast<uint16_t>(hostRandFloat[j + tapIdx * m_nRay] * (j + 1)); // Random index in range [0, j]
                std::swap(currentSeq[j], currentSeq[k]);                       // Swap elements
            }
        }
        cudaMemcpyAsync(m_h_deviceRayCouplingRandPerLinkPtr[linkIdx], sequence.data(), m_randRayCouplingSizePerLink*sizeof(uint16_t), cudaMemcpyHostToDevice, this -> m_strm);
    }
    cudaStreamSynchronize(this -> m_strm);
}

template <typename Tscalar, typename Tcomplex> 
void cdlChan<Tscalar, Tcomplex>::genTimeChan()
{
    // generate time domain channel
    this -> m_args[2] = &m_cdlDynDescrGpu;
    this -> m_gridDim = {this -> m_nLink, this -> m_fastFadingDynDescrCpu->nBatch, 1};
    this -> m_blockDim = {m_numTaps, uint(this -> m_nBsAnt / this -> m_scaleBsAntTimeChan), uint(this -> m_nUeAnt / this -> m_scaleUeAntTimeChan)};
    cudaGetFuncBySymbol(&(this -> m_functionPtr), reinterpret_cast<void*>(genCdlTimeChanCoeKernel<Tscalar, Tcomplex>));

    CUresult status = cuLaunchKernel(
        this->m_functionPtr,          // Kernel function pointer
        this->m_gridDim.x,            // Grid dimension (x)
        this->m_gridDim.y,            // Grid dimension (y)
        this->m_gridDim.z,            // Grid dimension (z)
        this->m_blockDim.x,           // Block dimension (x)
        this->m_blockDim.y,           // Block dimension (y)
        this->m_blockDim.z,           // Block dimension (z)
        0,                            // Shared memory size
        this->m_strm,                 // CUDA stream
        this->m_args,                 // Kernel arguments
        nullptr                       // Extra arguments
    );
    CHECK_CURESULT(status);
    this -> m_args[2] = &(this -> m_enableSwapTxRx);

    // normalize the cdl time channel per TTI if macro ENABLE_NORMALIZATION_ is defined, not used by default
    #ifdef ENABLE_NORMALIZATION_
    this -> m_gridDim = {1,1,1};
    this -> m_blockDim = {THREADS_PER_BLOCK_NORMALIZATION_, 1, 1};
    cudaGetFuncBySymbol(&(this -> m_functionPtr), reinterpret_cast<void*>(normalizeTimeChan<Tscalar, Tcomplex>));
    
    status = cuLaunchKernel(
        this->m_functionPtr,          // Kernel function pointer
        this->m_gridDim.x,            // Grid dimension (x)
        this->m_gridDim.y,            // Grid dimension (y)
        this->m_gridDim.z,            // Grid dimension (z)
        this->m_blockDim.x,           // Block dimension (x)
        this->m_blockDim.y,           // Block dimension (y)
        this->m_blockDim.z,           // Block dimension (z)
        0,                            // Shared memory size
        this->m_strm,                 // CUDA stream
        this->m_args,                 // Kernel arguments
        nullptr                       // Extra arguments
    );
    CHECK_CURESULT(status);
    #endif
}

template <typename Tscalar, typename Tcomplex> 
void cdlChan<Tscalar, Tcomplex>::run(float refTime0, uint8_t enableSwapTxRx, uint8_t txColumnMajorInd)
{
    this -> genChanProcSig(refTime0, enableSwapTxRx, txColumnMajorInd);  // cuda stream synchornize within
}

template <typename Tscalar, typename Tcomplex> 
void cdlChan<Tscalar, Tcomplex>::reset()
{
    this -> m_args[2] = &m_cdlDynDescrGpu;
    updateTapPathRand(); // cuda stream synchornize within
    updateRayCouplingRand(); // cuda stream synchornize within
    this -> m_gridDim = {this -> m_nLink, 1, 1};
    // reduce number of threads per thread block due to high register usage in genCdlMulti5TermsKernel
    if (this -> m_nBsAnt / this -> m_scaleBsAntTimeChan > 1)
    {
    this -> m_blockDim = {m_numTaps, uint(this -> m_nBsAnt / this -> m_scaleBsAntTimeChan / 2), uint(this -> m_nUeAnt / this -> m_scaleUeAntTimeChan)};
    }
    else
    {
        if (this -> m_nUeAnt / this -> m_scaleUeAntTimeChan > 1)
        {
            this -> m_blockDim = {m_numTaps, uint(this -> m_nBsAnt / this -> m_scaleBsAntTimeChan), uint(this -> m_nUeAnt / this -> m_scaleUeAntTimeChan / 2)};
        }
    }
    cudaGetFuncBySymbol(&(this -> m_functionPtr), reinterpret_cast<void*>(genCdlMulti5TermsKernel<Tscalar, Tcomplex>));

    CUresult status = cuLaunchKernel(
        this->m_functionPtr,          // Kernel function pointer
        this->m_gridDim.x,            // Grid dimension (x)
        this->m_gridDim.y,            // Grid dimension (y)
        this->m_gridDim.z,            // Grid dimension (z)
        this->m_blockDim.x,           // Block dimension (x)
        this->m_blockDim.y,           // Block dimension (y)
        this->m_blockDim.z,           // Block dimension (z)
        0,                            // Shared memory size
        this->m_strm,                 // CUDA stream
        this->m_args,                 // Kernel arguments
        nullptr                       // Extra arguments
    );
    CHECK_CURESULT(status);
    this -> m_args[2] = &(this -> m_enableSwapTxRx);
}

template <typename Tscalar, typename Tcomplex> 
void cdlChan<Tscalar, Tcomplex>::saveCdlChanToH5File(std::string & padFileNameEnding)
{
    cudaStreamSynchronize(this -> m_strm);
    std::string outFilename = "cdlChan_" + std::to_string(this -> m_nCell) + "cell" + std::to_string(this -> m_nUe)+ "Ue_" + std::to_string(this -> m_nBsAnt) + "x" + std::to_string(this -> m_nUeAnt) + "_" + m_delayProfile + std::to_string(int(m_delaySpread)) + "_dopp" + std::to_string(int(this -> m_maxDopplerShift)) + "_cfo" + std::to_string(int(this -> m_cfoHz)) + "_runMode" + std::to_string(this -> m_runMode) + "_freqConvert" + std::to_string(this -> m_freqConvertType) + "_scSampling" + std::to_string(this -> m_scSampling) + (typeid(Tcomplex) == typeid(__half2) ? "_FP16" : "_FP32") + "_swap" + std::to_string(this -> m_enableSwapTxRx) + padFileNameEnding + ".h5";
    // Initialize HDF5
    hid_t cdlH5File = H5Fcreate(outFilename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT); // non empty existing file will be overwritten
    if (cdlH5File < 0)
    {
        fprintf(stderr, "Failed to create HDF5 file!\n");
        exit(EXIT_FAILURE);
    }
    
    // Create a compound datatype based on Tcomplex datatype
    hid_t complexDataType = H5Tcreate(H5T_COMPOUND, sizeof(Tcomplex));
    hid_t fp16Type        = generate_native_HDF5_fp16_type();
    if(typeid(Tcomplex) == typeid(__half2))
    {
        H5Tinsert(complexDataType, "re", HOFFSET(__half2, x), fp16Type);
        H5Tinsert(complexDataType, "im", HOFFSET(__half2, y), fp16Type);
    }
    else if(typeid(Tcomplex) == typeid(cuComplex))
    {
        H5Tinsert(complexDataType, "re", HOFFSET(cuComplex, x), H5T_NATIVE_FLOAT);
        H5Tinsert(complexDataType, "im", HOFFSET(cuComplex, y), H5T_NATIVE_FLOAT);
    }
    else
    {
        fprintf(stderr, "Unsupported data type!\n");
        exit(EXIT_FAILURE);
    }
    
    // Create a dataset in the HDF5 file to store the data. TOOD: currently all 1D array
    uint8_t rank = 1;
    hsize_t dims[rank] = {0};

    cudaStreamSynchronize(this -> m_strm); // sync stream to ensure processing done
    
    // save common dataset for fast fading
    this -> saveChanToH5File(cdlH5File, complexDataType);
    // save random numbers used to generate cdl channel, each link saved as a dataset
    dims[0] = m_randPhiSizePerLink;
    for(uint32_t linkIdx = 0; linkIdx < this -> m_nLink; linkIdx ++)
    {
        std::string phiRandLinkIdxStr = "phiRandLink" + std::to_string(linkIdx);
        writeHdf5DatasetFromGpu<Tscalar>(cdlH5File, phiRandLinkIdxStr.c_str(), typeid(Tcomplex) == typeid(__half2) ? fp16Type : H5T_IEEE_F32LE, m_h_devicePhiRandPerLinkPtr[linkIdx], dims, rank);
    }
    // save random ray coupling used to generate cdl channel, each link saved as a dataset
    dims[0] = m_randRayCouplingSizePerLink;
    for(uint32_t linkIdx = 0; linkIdx < this -> m_nLink; linkIdx ++)
    {
        std::string rayCouplingRandLinkIdxStr = "rayCouplingRandLink" + std::to_string(linkIdx);
        writeHdf5DatasetFromGpu<uint16_t>(cdlH5File, rayCouplingRandLinkIdxStr.c_str(), H5T_STD_U16LE, m_h_deviceRayCouplingRandPerLinkPtr[linkIdx], dims, rank);
    }

    dims[0] = m_firNzTapMap.size();
    writeHdf5DatasetFromGpu<uint16_t>(cdlH5File, "firNzTapMap", H5T_STD_U16LE, m_cdlDynDescrCpu -> firNzTapMap, dims, rank);

    // save cdlCfg, data saved on CPU; create a compound data type for cdlConfig_t
    hid_t compType = H5Tcreate(H5T_COMPOUND, sizeof(cdlConfig_t));
    H5Tinsert(compType, "delayProfile", HOFFSET(cdlConfig_t, delayProfile), H5T_NATIVE_CHAR);
    H5Tinsert(compType, "delaySpread", HOFFSET(cdlConfig_t, delaySpread), H5T_IEEE_F32LE);
    H5Tinsert(compType, "maxDopplerShift", HOFFSET(cdlConfig_t, maxDopplerShift), H5T_IEEE_F32LE);
    H5Tinsert(compType, "f_samp", HOFFSET(cdlConfig_t, f_samp), H5T_IEEE_F32LE);
    H5Tinsert(compType, "nCell", HOFFSET(cdlConfig_t, nCell), H5T_STD_U16LE);
    H5Tinsert(compType, "nUe", HOFFSET(cdlConfig_t, nUe), H5T_STD_U16LE);
    H5Tinsert(compType, "bsAntPattern", HOFFSET(cdlConfig_t, bsAntPattern), H5T_NATIVE_UCHAR);
    H5Tinsert(compType, "ueAntPattern", HOFFSET(cdlConfig_t, ueAntPattern), H5T_NATIVE_UCHAR);
    H5Tinsert(compType, "fBatch", HOFFSET(cdlConfig_t, fBatch), H5T_STD_U32LE);
    H5Tinsert(compType, "numRay", HOFFSET(cdlConfig_t, numRay), H5T_STD_U16LE);
    H5Tinsert(compType, "cfoHz", HOFFSET(cdlConfig_t, cfoHz), H5T_IEEE_F32LE);
    H5Tinsert(compType, "delay", HOFFSET(cdlConfig_t, delay), H5T_IEEE_F32LE);
    H5Tinsert(compType, "sigLenPerAnt", HOFFSET(cdlConfig_t, sigLenPerAnt), H5T_STD_U32LE);
    H5Tinsert(compType, "N_sc", HOFFSET(cdlConfig_t, N_sc), H5T_STD_U16LE);
    H5Tinsert(compType, "N_sc_Prbg", HOFFSET(cdlConfig_t, N_sc_Prbg), H5T_STD_U16LE);
    H5Tinsert(compType, "scSpacingHz", HOFFSET(cdlConfig_t, scSpacingHz), H5T_IEEE_F32LE);
    H5Tinsert(compType, "freqConvertType", HOFFSET(cdlConfig_t, freqConvertType), H5T_NATIVE_CHAR);
    H5Tinsert(compType, "scSampling", HOFFSET(cdlConfig_t, scSampling), H5T_NATIVE_CHAR);
    H5Tinsert(compType, "runMode", HOFFSET(cdlConfig_t, runMode), H5T_NATIVE_CHAR);
    H5Tinsert(compType, "procSigFreq", HOFFSET(cdlConfig_t, procSigFreq), H5T_NATIVE_CHAR);
    // H5Tinsert(compType, "batchLen", HOFFSET(cdlConfig_t, batchLen), H5T_STD_U32LE); skip the vector
    // H5Tinsert(compType, "txSigIn", HOFFSET(cdlConfig_t, txSigIn), H5T_STD_U32LE); skip the pointer

    // save cdl config 
    dims[0] = 1;
    hid_t dataspaceId = H5Screate_simple(rank, dims, nullptr);
    hid_t datasetId = H5Dcreate2(cdlH5File, "cdlCfg", compType, dataspaceId, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    herr_t status = H5Dwrite(datasetId, compType, H5S_ALL, H5S_ALL, H5P_DEFAULT, m_cdlCfg);
    H5Dclose(datasetId);
    H5Sclose(dataspaceId);

    // Lambda to write a vector as a dataset into an HDF5 file
    auto writeVectorDataset = [&](const std::string& name,
        const auto& vec,
        hid_t file,
        hid_t type) {
        // Define dimensions for the dataspace
        hsize_t dims[1] = {vec.size()};

        // Create a simple dataspace for the vector
        hid_t space = H5Screate_simple(1, dims, NULL);

        // Create the dataset in the file
        hid_t dataset = H5Dcreate(file,
            name.c_str(),
            type,
            space,
            H5P_DEFAULT,
            H5P_DEFAULT,
            H5P_DEFAULT);

        // Write data to the dataset if the vector is not empty
        if (!vec.empty()) {
        H5Dwrite(dataset,
        type,
        H5S_ALL,
        H5S_ALL,
        H5P_DEFAULT,
        vec.data());
        }

        // Close resources to prevent memory leaks
        H5Dclose(dataset);
        H5Sclose(space);
    };

    // Write TX antenna vectors
    writeVectorDataset("bsAntSize", m_cdlCfg->bsAntSize, cdlH5File, H5T_STD_U16LE); // uint16_t vector
    writeVectorDataset("bsAntSpacing", m_cdlCfg->bsAntSpacing, cdlH5File, H5T_IEEE_F32LE); // float vector
    writeVectorDataset("bsAntPolarAngles", m_cdlCfg->bsAntPolarAngles, cdlH5File, H5T_IEEE_F32LE); // float vector

    // Write RX antenna vectors
    writeVectorDataset("ueAntSize", m_cdlCfg->ueAntSize, cdlH5File, H5T_STD_U16LE); // uint16_t vector
    writeVectorDataset("ueAntSpacing", m_cdlCfg->ueAntSpacing, cdlH5File, H5T_IEEE_F32LE); // float vector
    writeVectorDataset("ueAntPolarAngles", m_cdlCfg->ueAntPolarAngles, cdlH5File, H5T_IEEE_F32LE); // float vector

    // Close HDF5 objects 
    H5Fclose(cdlH5File);
}

template <typename Tscalar, typename Tcomplex> 
void cdlChan<Tscalar, Tcomplex>::printGpuMemUseMB()
{
    float gpuMemUseMB = this -> m_gpuDataUsageByte / 1024.0f / 1024.0f;
    printf("CDL channel class uses %.2f MB GPU memory. \n", gpuMemUseMB);
}

/*----------------------      begin CUDA kernels        -----------------------------*/
// Device function to calculate mAnt, nAnt, pAnt, and d_bar_rx_u
__device__ inline void findAntLocAndDBar(const uint16_t & u, const uint16_t* __restrict__ AntSize, 
    const float* __restrict__ AntSpacing, 
    const float* __restrict__ antPolarAngles, 
    float* __restrict__ d_bar_u, 
    float& zetaAnt) 
{
    // AntSize order: {M_g, N_g, M, N, P}
    // AntSpacing order: {d_g_h, d_g_v, d_h, d_v}
    const uint16_t m = AntSize[2];
    const uint16_t n = AntSize[3];
    const uint16_t p = AntSize[4];
    uint16_t temp = u / p;

    // Calculate d_bar_rx_u (3D vector)
    d_bar_u[0] = 0.0f;
    d_bar_u[1] = (temp % n) * AntSpacing[2];
    d_bar_u[2] = (temp / n) % m * AntSpacing[3];

    zetaAnt = antPolarAngles[u % p] * D2PI;
}

// Device function to calculate A_dB_3D
__device__ inline float calc_A_dB_3D(float & theta, float & phi) 
{
    // Precomputed constants
    constexpr float inv_theta_3dB2 = 1.0f / (65.0f * D2PI * 65.0f * D2PI); // 1 / (theta_3dB^2)
    constexpr float inv_phi_3dB2 = 1.0f / (65.0f * D2PI * 65.0f * D2PI);   // 1 / (phi_3dB^2)
    constexpr float SLA_v = 30.0f;
    constexpr float A_max = 30.0f;

    // Calculate A_dB_theta
    float theta_offset = theta - 90.0f * D2PI;
    float A_dB_theta = -fminf(12.0f * theta_offset * theta_offset * inv_theta_3dB2, SLA_v);

    // Calculate A_dB_phi
    float A_dB_phi = -fminf(12.0f * phi * phi * inv_phi_3dB2, A_max);

    // Combine to calculate A_dB_3D
    return -fminf(-(A_dB_theta + A_dB_phi), A_max);
}

__device__ inline void calc_Field(uint8_t & antPattern, float & theta, float & phi, float & zeta, 
    float & F_theta, float & F_phi) 
{
    float sqrtA = 0.0f;
    float A_dB_3D = 0.0f;

    switch (antPattern) {
        case 0:
            sqrtA = 1.0f;
            break;
        case 1:
            A_dB_3D = 8.0f + calc_A_dB_3D(theta, phi);
            sqrtA = powf(10.0f, A_dB_3D * 0.05f);
            break;
        default:
            return; // silently ignore unsupported patterns
    }

    // Calculate field components
    F_theta = sqrtA * cosf(zeta);
    F_phi = sqrtA * sinf(zeta);
}

template <typename Tscalar, typename Tcomplex> 
static __global__ void genCdlMulti5TermsKernel(fastFadingDynDescr_t<Tscalar, Tcomplex> * fastFadingDynDescr, float refTime0, cdlDynDescr_t<Tscalar, Tcomplex> * cdlDynDescr)
{
    // GRID(nLink, 1, 1);
    // BLOCK(numTaps, nBsAnt/scaleBsAntTimeChan, nUeAnt/scaleUeAntTimeChan);

    uint16_t cidUidOffset = blockIdx.x; // linkIdx
    uint16_t tapIdx = threadIdx.x;

    uint16_t nBsAnt   = fastFadingDynDescr -> nBsAnt;
    uint16_t nUeAnt   = fastFadingDynDescr -> nUeAnt;
    float maxDopplerShift = fastFadingDynDescr -> maxDopplerShift;

    uint16_t nTaps = cdlDynDescr -> nTaps;
    uint16_t nRay = cdlDynDescr -> nRay;
    float * dpa = cdlDynDescr -> dpa;
    float * pcp = cdlDynDescr -> pcp;
    float * rayOffsetAngles = cdlDynDescr -> rayOffsetAngles;
    uint16_t * rayCouplingRand = cdlDynDescr -> rayCouplingRand[cidUidOffset];
    Tscalar * phiRand = cdlDynDescr -> phiRand[cidUidOffset]; // phiRand in [-pi, pi]

    Tcomplex * multi5Term = cdlDynDescr -> multi5Term;   // GPU address to save muti of first five terms
    Tscalar * rheadVbar  = cdlDynDescr -> rheadVbar;  // GPU address to save 2*pi*(r_head_rx_n_m'*v_head*dopplerHz)

    // Declare shared memory for tx and rx parameters (shared across threads in a block)
    __shared__ float c_ASD;
    __shared__ float c_ASA;
    __shared__ float c_ZSD;
    __shared__ float c_ZSA;
    __shared__ float sqrtInvXPR;
    __shared__ uint16_t bsAntSize[5];
    __shared__ float bsAntSpacing[4];
    __shared__ float bsAntPolarAngles[2];
    __shared__ uint8_t bsAntPattern;

    __shared__ uint16_t ueAntSize[5];
    __shared__ float ueAntSpacing[4];
    __shared__ float ueAntPolarAngles[2];
    __shared__ uint8_t ueAntPattern;

    __shared__ float vDirection[2];

    // Load tx and rx antenna parameters into shared memory
    if (threadIdx.y == 0 && threadIdx.z == 0) {
        // Load bsAntSize and ueAntSize
        if (tapIdx < 5) {
            bsAntSize[tapIdx] = cdlDynDescr->bsAntSize[tapIdx];
            ueAntSize[tapIdx] = cdlDynDescr->ueAntSize[tapIdx];
        }

        // Load bsAntSpacing and ueAntSpacing
        if (tapIdx < 4) {
            bsAntSpacing[tapIdx] = cdlDynDescr->bsAntSpacing[tapIdx];
            ueAntSpacing[tapIdx] = cdlDynDescr->ueAntSpacing[tapIdx];
        }

        // Load bsAntPolarAngles, ueAntPolarAngles, and vDirection
        if (tapIdx < 2) {
            bsAntPolarAngles[tapIdx] = cdlDynDescr->bsAntPolarAngles[tapIdx];
            ueAntPolarAngles[tapIdx] = cdlDynDescr->ueAntPolarAngles[tapIdx];
            vDirection[tapIdx] = cdlDynDescr->vDirection[tapIdx];
        }

        // Load bsAntPattern, ueAntPattern, pcp
        if (tapIdx == 0) {
            bsAntPattern = cdlDynDescr->bsAntPattern;
            ueAntPattern = cdlDynDescr->ueAntPattern;

            c_ASD = pcp[1];
            c_ASA = pcp[0];
            c_ZSD = pcp[3];
            c_ZSA = pcp[2];
            sqrtInvXPR = pcp[4];
        }

        // load pcp
        if (tapIdx < 5) {
            bsAntSize[tapIdx] = cdlDynDescr->bsAntSize[tapIdx];
            ueAntSize[tapIdx] = cdlDynDescr->ueAntSize[tapIdx];
        }
    }

    // Synchronize threads to ensure all shared memory is loaded before use
    __syncthreads();

    // Calculate field components
    cuComplex tmpMulti[3];
    for(uint16_t ueAntIdx = threadIdx.z; ueAntIdx < nUeAnt; ueAntIdx += blockDim.z)
    {
        float d_bar_rx_u[3], zetaUeAnt;
        // Call the device function to compute d_bar_rx_u and zetaUeAnt
        findAntLocAndDBar(ueAntIdx, ueAntSize, ueAntSpacing, ueAntPolarAngles, d_bar_rx_u, zetaUeAnt);
        for(uint16_t bsAntIdx = threadIdx.y; bsAntIdx < nBsAnt; bsAntIdx += blockDim.y)
        {
            float d_bar_tx_u[3], zetaBsAnt;
            // Call the device function to compute d_bar_rx_u and zetaUeAnt
            findAntLocAndDBar(bsAntIdx, bsAntSize, bsAntSpacing, bsAntPolarAngles, d_bar_tx_u, zetaBsAnt);
            for(uint16_t rayIdx = 0; rayIdx < nRay; rayIdx++)
            {
                // Load ray coupling indices only once
                uint32_t rayOffset = tapIdx * 4 * nRay + rayIdx;
                float phi_n_m_AOD   = (dpa[tapIdx * 6 + 3] + c_ASD * rayOffsetAngles[rayCouplingRand[rayOffset]]) * D2PI;  // phi_n_AOD = dpa[tapIdx * 6 + 3], idxASD = rayCouplingRand[rayOffset]
                float phi_n_m_AOA   = (dpa[tapIdx * 6 + 2] + c_ASA * rayOffsetAngles[rayCouplingRand[rayOffset + nRay]]) * D2PI;  // phi_n_AOA = dpa[tapIdx * 6 + 2], idxASA = rayCouplingRand[rayOffset + nRay]
                float theta_n_m_ZOD = (dpa[tapIdx * 6 + 5] + c_ZSD * rayOffsetAngles[rayCouplingRand[rayOffset + 2 * nRay]]) * D2PI;  // theta_n_ZOD = dpa[tapIdx * 6 + 5], idxZSD = rayCouplingRand[rayOffset + 2 * nRay]
                float theta_n_m_ZOA = (dpa[tapIdx * 6 + 4] + c_ZSA * rayOffsetAngles[rayCouplingRand[rayOffset + 3 * nRay]]) * D2PI;  // theta_n_ZOA = dpa[tapIdx * 6 + 4], idxZSA = rayCouplingRand[rayOffset + 3 * nRay]

                // Compute Rx field components, term1
                float F_rx_u_theta, F_rx_u_phi;
                calc_Field(ueAntPattern, theta_n_m_ZOA, phi_n_m_AOA, zetaUeAnt, F_rx_u_theta, F_rx_u_phi);

                // term1 * term2, stored as a row vector [tmpMulti[0] tmpMulti[1]]
                // Precompute phiRand offsets
                uint32_t phiRandBase = (tapIdx * nRay + rayIdx) * 4;
                // Inline trigonometric calculations for tmpMulti[0] and tmpMulti[1]
                tmpMulti[0].x = F_rx_u_theta * cosf(phiRand[phiRandBase]) + F_rx_u_phi * sqrtInvXPR * cosf(phiRand[phiRandBase + 2]);
                tmpMulti[0].y = F_rx_u_theta * sinf(phiRand[phiRandBase]) + F_rx_u_phi * sqrtInvXPR * sinf(phiRand[phiRandBase + 2]);
                tmpMulti[1].x = F_rx_u_theta * sqrtInvXPR * cosf(phiRand[phiRandBase + 1]) + F_rx_u_phi * cosf(phiRand[phiRandBase + 3]);
                tmpMulti[1].y = F_rx_u_theta * sqrtInvXPR * sinf(phiRand[phiRandBase + 1]) + F_rx_u_phi * sinf(phiRand[phiRandBase + 3]);
                   
                // Compute Tx field components, term3
                // reuse to reduce registers per thread
                float & F_tx_s_theta = F_rx_u_theta;
                float & F_tx_s_phi = F_rx_u_phi;
                calc_Field(bsAntPattern, theta_n_m_ZOD, phi_n_m_AOD, zetaBsAnt, F_tx_s_theta, F_tx_s_phi);
                // term1 * term2 * term3, store in tmpMulti[2]
                tmpMulti[2].x = tmpMulti[0].x * F_tx_s_theta + tmpMulti[1].x * F_tx_s_phi;
                tmpMulti[2].y = tmpMulti[0].y * F_tx_s_theta + tmpMulti[1].y * F_tx_s_phi;

                // term4 and term5, reuse tmpMulti[0] and tmpMulti[1]
                float r_head_rx_n_m[3];
                r_head_rx_n_m[0] = sinf(theta_n_m_ZOA) * cosf(phi_n_m_AOA);
                r_head_rx_n_m[1] = sinf(theta_n_m_ZOA) * sinf(phi_n_m_AOA);
                r_head_rx_n_m[2] = cosf(theta_n_m_ZOA);

                tmpMulti[0].x = 
                    r_head_rx_n_m[0] * d_bar_rx_u[0] + 
                    r_head_rx_n_m[1] * d_bar_rx_u[1] + 
                    r_head_rx_n_m[2] * d_bar_rx_u[2];
                tmpMulti[0].y = 
                    sinf(theta_n_m_ZOD) * cosf(phi_n_m_AOD) /* r_head_tx_n_m[0] */  * d_bar_tx_u[0] + 
                    sinf(theta_n_m_ZOD) * sinf(phi_n_m_AOD) /* r_head_tx_n_m[1] */ * d_bar_tx_u[1] + 
                    cosf(theta_n_m_ZOD) /* r_head_tx_n_m[2] */ * d_bar_tx_u[2];
                tmpMulti[0].x = (tmpMulti[0].x + tmpMulti[0].y) * 2 * M_PI;  // no need to divide by \lambda_0
                // mutiply term4*term5, store in tmpMulti[1]
                tmpMulti[1].x = cosf(tmpMulti[0].x);
                tmpMulti[1].y = sinf(tmpMulti[0].x);

                // mutiply "term1*term2*term3 (tmpMulti[2])"" * "term4*term5 (tmpMulti[1])", store in tmpMulti[0]
                tmpMulti[0].x = tmpMulti[1].x * tmpMulti[2].x - tmpMulti[1].y * tmpMulti[2].y;
                tmpMulti[0].y = tmpMulti[1].x * tmpMulti[2].y + tmpMulti[1].y * tmpMulti[2].x;
                uint32_t globalSaveIdx = (((cidUidOffset * nUeAnt + ueAntIdx) * nBsAnt + bsAntIdx) * nTaps + tapIdx) * nRay + rayIdx; 
                multi5Term[globalSaveIdx].x = tmpMulti[0].x;
                multi5Term[globalSaveIdx].y = tmpMulti[0].y;

                // reuse tmpMulti[0].x for tmpRheadVbar;
                tmpMulti[0].x = 2 * M_PI * maxDopplerShift * (
                    sinf(vDirection[0]) * cosf(vDirection[1]) * r_head_rx_n_m[0] +
                    sinf(vDirection[0]) * sinf(vDirection[1]) * r_head_rx_n_m[1] +
                    cosf(vDirection[0]) * r_head_rx_n_m[2]
                );
                
                globalSaveIdx = ((cidUidOffset * nUeAnt + ueAntIdx) * nTaps + tapIdx) * nRay + rayIdx;
                rheadVbar[globalSaveIdx] = tmpMulti[0].x;
            }
        }
    }
}

template <typename Tscalar, typename Tcomplex>
__global__ void genCdlTimeChanCoeKernel(
    fastFadingDynDescr_t<Tscalar, Tcomplex> *fastFadingDynDescr,
    float refTime0,
    cdlDynDescr_t<Tscalar, Tcomplex> *cdlDynDescr
) 
{
    // GRID(nLink, nBatch, 1);
    // BLOCK(numTaps, nBsAnt/scaleBsAntTimeChan, nUeAnt/scaleUeAntTimeChan);

    uint16_t cidUidOffset = blockIdx.x; // linkIdx
    uint16_t batchIdx = blockIdx.y;
    uint16_t tapIdx = threadIdx.x;

    uint16_t nBsAnt = fastFadingDynDescr->nBsAnt;
    uint16_t nUeAnt = fastFadingDynDescr->nUeAnt;
    uint16_t firNzLen = fastFadingDynDescr->firNzLen;
    float *firNzPw = fastFadingDynDescr->firNzPw;
    Tcomplex *timeChan = fastFadingDynDescr->timeChan;
    float timeStamp = fastFadingDynDescr->tBatch[batchIdx] + refTime0;

    uint16_t nTaps = cdlDynDescr->nTaps;
    uint16_t nRay = cdlDynDescr->nRay;
    uint16_t *firNzTapMap = cdlDynDescr->firNzTapMap;
    Tcomplex *multi5Term = cdlDynDescr->multi5Term; // GPU address to save muti of first five terms
    Tscalar *rheadVbar = cdlDynDescr->rheadVbar; // GPU address to save 2*pi*(r_head_rx_n_m'*v_head*dopplerHz)

    // initialize all timeChan to 0
    for (uint16_t ueAntIdx = threadIdx.z; ueAntIdx < nUeAnt; ueAntIdx += blockDim.z) {
        for (uint16_t bsAntIdx = threadIdx.y; bsAntIdx < nBsAnt; bsAntIdx += blockDim.y) {
            if(tapIdx < firNzLen)
            {
                uint32_t globalChanIdx = (((cidUidOffset * gridDim.y + batchIdx) * nUeAnt + ueAntIdx) * nBsAnt + bsAntIdx) * firNzLen + tapIdx;
                timeChan[globalChanIdx].x = 0.0f;
                timeChan[globalChanIdx].y = 0.0f;
            }
        }
    }
    __syncthreads();

    cuComplex multi5TermRead;
    cuComplex rheadVbarExp;
    cuComplex tempChanCoe;
    for (uint16_t ueAntIdx = threadIdx.z; ueAntIdx < nUeAnt; ueAntIdx += blockDim.z) {
        for (uint16_t bsAntIdx = threadIdx.y; bsAntIdx < nBsAnt; bsAntIdx += blockDim.y) {
            tempChanCoe.x = 0.0f;
            tempChanCoe.y = 0.0f;
            uint32_t globalMulti5TermIdx;
            uint32_t globalRheadVbarIdx;
            for (uint16_t rayIdx = 0; rayIdx < nRay; rayIdx++) {
                globalMulti5TermIdx = (((cidUidOffset * nUeAnt + ueAntIdx) * nBsAnt + bsAntIdx) * nTaps + tapIdx) * nRay + rayIdx;
                multi5TermRead.x = multi5Term[globalMulti5TermIdx].x;
                multi5TermRead.y = multi5Term[globalMulti5TermIdx].y;

                globalRheadVbarIdx = ((cidUidOffset * nUeAnt + ueAntIdx) * nTaps + tapIdx) * nRay + rayIdx;
                float tmpPhase = float(rheadVbar[globalRheadVbarIdx]) * timeStamp;
                rheadVbarExp.x = cosf(tmpPhase);
                rheadVbarExp.y = sinf(tmpPhase);

                tempChanCoe.x += (multi5TermRead.x * rheadVbarExp.x - multi5TermRead.y * rheadVbarExp.y);
                tempChanCoe.y += (multi5TermRead.x * rheadVbarExp.y + multi5TermRead.y * rheadVbarExp.x);
            }
            tempChanCoe.x *= firNzPw[tapIdx];
            tempChanCoe.y *= firNzPw[tapIdx];

            // Perform atomic addition on the x and y components of timeChan
            uint32_t globalChanIdx = (((cidUidOffset * gridDim.y + batchIdx) * nUeAnt + ueAntIdx) * nBsAnt + bsAntIdx) * firNzLen + firNzTapMap[tapIdx];
            atomicAdd(&(timeChan[globalChanIdx].x), tempChanCoe.x);
            atomicAdd(&(timeChan[globalChanIdx].y), tempChanCoe.y);
        }
    }
}
