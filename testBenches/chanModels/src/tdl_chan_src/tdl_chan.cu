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

#include "tdl_chan.cuh"
#include "tdl_pdp_table.h"


template <typename Tscalar, typename Tcomplex> 
tdlChan<Tscalar, Tcomplex>::tdlChan(tdlConfig_t * tdlCfg, uint16_t randSeed, cudaStream_t strm)
{
    // Assignments without "this->" (directly accessible or local variables)
    m_tdlCfg                 = tdlCfg;
    m_nPath                  = tdlCfg->numPath;
    m_useSimplifiedPdp       = tdlCfg->useSimplifiedPdp;  // true for simplified PDP in 38.141, false for 38.901
    m_delayProfile           = tdlCfg->delayProfile;
    float tSample =  1.0f / (tdlCfg -> f_samp);
    ASSERT(tdlCfg -> scSampling > 0, "scSampling must be a positive integer");
    ASSERT(tdlCfg -> f_samp == 4096 * tdlCfg -> scSpacingHz, "mismated sampling frequency and SC spacing, f_samp must be equal to 4096 * scSpacingHz");
    ASSERT(tdlCfg -> procSigFreq == 0 || tdlCfg -> runMode == 2, "processing tx samples in frequency domain requires runMode 2");

    // Assignments with "this->" (explicitly referencing base class members)
    this->m_strm              = strm;
    this->m_nCell             = tdlCfg->nCell;
    this->m_nUe               = tdlCfg->nUe;
    this->m_nLink             = this->m_nCell * this->m_nUe;
    this->m_nBsAnt            = tdlCfg->nBsAnt;
    this->m_nUeAnt            = tdlCfg->nUeAnt;
    this->m_runMode           = tdlCfg->runMode;
    this->m_maxDopplerShift   = tdlCfg->maxDopplerShift;
    this->m_sigLenPerAnt      = tdlCfg->sigLenPerAnt;
    this->m_delay             = tdlCfg->delay;
    this->m_cfoHz             = tdlCfg->cfoHz;
    this->m_f_samp            = tdlCfg->f_samp;
    this->m_fBatch            = tdlCfg->fBatch;
    this->m_batchLen          = tdlCfg->batchLen;
    this->m_txSigIn           = tdlCfg->txSigIn;
    this->m_mimoCorrMat       = nullptr;                  // TODO: 'Low' correlation for now
    this->m_N_sc              = tdlCfg->N_sc;
    this->m_N_sc_Prbg         = tdlCfg->N_sc_Prbg;
    this->m_scSpacingHz       = tdlCfg->scSpacingHz;
    this->m_freqConvertType   = tdlCfg->freqConvertType;
    this->m_scSampling        = tdlCfg->scSampling;    
    this->m_procSigFreq       = tdlCfg -> procSigFreq;
    this->m_sigLenPerAnt      = tdlCfg -> sigLenPerAnt;
    this->m_saveAntPairSample = tdlCfg -> saveAntPairSample;
    this->m_txSigIn           = reinterpret_cast<Tcomplex*>(tdlCfg -> txSigIn);

    /*-------------------   read pdp from tables in tdl_pdp_table.h   -------------------*/
    if(m_useSimplifiedPdp) // set delay profile based on TS 38.141
    {
        switch(m_delayProfile)
        {
            case 'A': // TLDA30
                m_delaySpread = 30.0f;
                m_pdp = &pdp_38141_const[0];
                m_numTaps = 12;
                m_LosTap = false;
            break;
            case 'B': // TLDB100
                m_delaySpread = 100.0f;
                m_pdp = &pdp_38141_const[0] + 12*2;
                m_numTaps = 12;
                m_LosTap = false;
            break;
            case 'C': // TDLC300
                m_delaySpread = 300.0f;
                m_pdp = &pdp_38141_const[0] + 24*2;
                m_numTaps = 12;
                m_LosTap = false;
            break;
            default: // TLDA30 default
                m_delaySpread = 30.0f;
                m_pdp = &pdp_38141_const[0];
                m_numTaps = 12;
                m_LosTap = false;
            break;
        }
    }
    else // set delay profile based on TS 38.901
    {
        m_delaySpread = tdlCfg -> delaySpread; // customize delay spread
        switch(m_delayProfile)
        {
            case 'A':
                m_pdp = &pdp_38901_const[0];
                m_numTaps = 23;
                m_LosTap = false;
            break;

            case 'B':
                m_pdp = &pdp_38901_const[0] + 23*2;
                m_numTaps = 23;
                m_LosTap = false;
            break;

            case 'C':
                m_pdp = &pdp_38901_const[0] + 46*2;
                m_numTaps = 24;
                m_LosTap = false;
            break;

            case 'D':
                m_pdp = &pdp_38901_const[0] + 70*2;
                m_numTaps = 14;
                m_LosTap = true;
            break;

            case 'E':
                m_pdp = &pdp_38901_const[0] + 84*2;
                m_numTaps = 14;
                m_LosTap = true;
            break;

            default:
                m_pdp = &pdp_38901_const[0];
                m_numTaps = 23;
                m_LosTap = false;
            break;
        }
    }
    // TODO: currently does not support TDL-D and TDL-E with LOS path, will added later
    if(m_LosTap)
    {
        printf("ERROR: TDL with LOS path is not supported yet! \n");
        exit(EXIT_FAILURE);
    }
    /*-------------------   calculate FIR based on tdl config   -------------------*/
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
        this -> m_firNzPw.push_back(pow(10, m_pdp[tapIdx *2 + 1] * 0.1f));
        uint16_t firNzIdx = round(m_pdp[tapIdx*2] * 1e-9 * this -> m_f_samp * m_delaySpread);
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
    // normalization targe: 1/sqrt(nPath)
    sum_firNzPw = 1.0f/(sum_firNzPw * m_nPath);
    for(int firNzIdx = 0; firNzIdx < m_numTaps; firNzIdx ++)
    {
        // take sqrt to be multiplied for chan coe
        this -> m_firNzPw[firNzIdx] = sqrt((this -> m_firNzPw[firNzIdx]) * sum_firNzPw);
    }

    // setup TDL dynamic descrptor
    m_tdlDynDescrCpu = new tdlDynDescr_t<Tscalar, Tcomplex>;
    m_tdlDynDescrCpu -> LosTap = m_LosTap;
    m_tdlDynDescrCpu -> nPath = m_nPath;
    m_tdlDynDescrCpu -> nTaps = m_numTaps;
    m_tdlDynDescrCpu -> PI_4_nPath = M_PI/4.0f/(m_tdlDynDescrCpu -> nPath); // calculate a constant pi/4/nPath on CPU
    m_tdlDynDescrCpu -> PI_2_nPath = M_PI/2.0f/(m_tdlDynDescrCpu -> nPath); // calculate a constant pi/2/nPath on CPU
    // allocate and copy index map
    CHECK_CUDAERROR(cudaMalloc((void**) &(m_tdlDynDescrCpu -> firNzTapMap), m_numTaps * sizeof(uint16_t)));
    CHECK_CUDAERROR(cudaMemcpyAsync(m_tdlDynDescrCpu -> firNzTapMap, m_firNzTapMap.data(), m_numTaps * sizeof(uint16_t), cudaMemcpyHostToDevice, this -> m_strm));

    // for curand states
    curandCreateGeneratorHost(&m_Rng, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(m_Rng, randSeed); //random seed applied here
    m_randSizePerLink = this -> m_nBsAnt * this -> m_nUeAnt * m_numTaps * m_nPath;
    CHECK_CUDAERROR(cudaMalloc((void**) &(m_tdlDynDescrCpu -> thetaRand), this -> m_nLink * sizeof(Tscalar*)));
    this -> m_gpuDataUsageByte += this -> m_nLink * sizeof(Tscalar*);
    m_h_deviceThetaRandPerLinkPtr = (Tscalar **)malloc(this -> m_nLink * sizeof(Tscalar*));
    for(uint32_t linkIdx = 0; linkIdx < this -> m_nLink; linkIdx ++)
    {
        CHECK_CUDAERROR(cudaMalloc((void**) &(m_h_deviceThetaRandPerLinkPtr[linkIdx]), 2 * m_randSizePerLink * sizeof(Tscalar)));
        this -> m_gpuDataUsageByte += 2 * m_randSizePerLink * sizeof(Tscalar);
    }
    CHECK_CUDAERROR(cudaMemcpy(m_tdlDynDescrCpu -> thetaRand, m_h_deviceThetaRandPerLinkPtr, this -> m_nLink * sizeof(Tscalar*), cudaMemcpyHostToDevice));

    this -> setup();  // call setup in base class

    //copy dyndescriptor to GPU
    CHECK_CUDAERROR(cudaMalloc((void**)&(m_tdlDynDescrGpu), sizeof(tdlDynDescr_t<Tscalar, Tcomplex>)));
    this -> m_gpuDataUsageByte += sizeof(tdlDynDescr_t<Tscalar, Tcomplex>);
    cudaMemcpyAsync(m_tdlDynDescrGpu, m_tdlDynDescrCpu, sizeof(tdlDynDescr_t<Tscalar, Tcomplex>), cudaMemcpyHostToDevice, this -> m_strm);

    // initialization
    reset();

    cudaStreamSynchronize(this -> m_strm); // finish constructor
    // check if any errors in creating TDL channel class
    cudaError_t cuda_error = cudaGetLastError();
    if(cuda_error != cudaSuccess) 
    {
        fprintf(stderr, "CUDA error in TDL channel constructor: %s\n", cudaGetErrorString(cuda_error));
        exit(EXIT_FAILURE);
    }
}

template <typename Tscalar, typename Tcomplex> 
tdlChan<Tscalar, Tcomplex>::~tdlChan()
{
    for(uint32_t linkIdx = 0; linkIdx < this -> m_nLink; linkIdx ++)
    {
        cudaFree(m_h_deviceThetaRandPerLinkPtr[linkIdx]);
    }
    free(m_h_deviceThetaRandPerLinkPtr);
    cudaFree(m_tdlDynDescrCpu -> thetaRand);
    cudaFree(m_tdlDynDescrCpu -> firNzTapMap);
    curandDestroyGenerator(m_Rng);

    cudaFree(m_tdlDynDescrGpu);
    delete m_tdlDynDescrCpu;
}

template <typename Tscalar, typename Tcomplex> 
void tdlChan<Tscalar, Tcomplex>::updateTapPathRand() // curand does not support half precision yet
{
    curandStatus_t curandResult;
    std::vector<float> hostRandomNumbersFloat(2 * m_randSizePerLink);
    if(typeid(Tcomplex) == typeid(__half2)) // need to convert to half precision
    {
        std::vector<Tscalar> hostRandomNumbersHalf(2 * m_randSizePerLink);
        for(uint32_t linkIdx = 0; linkIdx < this -> m_nLink; linkIdx ++)
        {
            curandResult = curandGenerateUniform(m_Rng, hostRandomNumbersFloat.data(), 2*m_randSizePerLink); // for phase 
            if (curandResult != CURAND_STATUS_SUCCESS) 
            {
                std::string msg("Could not generate random number thetaRand: ");
                throw std::runtime_error(msg);
            }
            for(uint32_t i=0; i<2*m_randSizePerLink; i++)
            {
                hostRandomNumbersHalf[i] = __float2half(hostRandomNumbersFloat[i]);
            }
            cudaMemcpyAsync(m_h_deviceThetaRandPerLinkPtr[linkIdx], hostRandomNumbersHalf.data(), 2*m_randSizePerLink*sizeof(Tscalar), cudaMemcpyHostToDevice, this -> m_strm);
        }
    }
    else if(typeid(Tcomplex) == typeid(cuComplex))
    {
        for(uint32_t linkIdx = 0; linkIdx < this -> m_nLink; linkIdx ++)
        {
            curandResult = curandGenerateUniform(m_Rng, hostRandomNumbersFloat.data(), 2*m_randSizePerLink); // for phase 
            if (curandResult != CURAND_STATUS_SUCCESS) 
            {
                std::string msg("Could not generate random number thetaRand: ");
                throw std::runtime_error(msg);
            }
            cudaMemcpyAsync(m_h_deviceThetaRandPerLinkPtr[linkIdx], hostRandomNumbersFloat.data(), 2*m_randSizePerLink*sizeof(Tscalar), cudaMemcpyHostToDevice, this -> m_strm);
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
void tdlChan<Tscalar, Tcomplex>::genTimeChan()
{
    // generate time domain channel
    this -> m_args[2] = &m_tdlDynDescrGpu;
    this -> m_gridDim = {this -> m_nLink, this -> m_fastFadingDynDescrCpu->nBatch, 1};
    this -> m_blockDim = {m_numTaps, uint(this -> m_nBsAnt / this -> m_scaleBsAntTimeChan), uint(this -> m_nUeAnt / this -> m_scaleUeAntTimeChan)};
    cudaGetFuncBySymbol(&(this -> m_functionPtr), reinterpret_cast<void*>(genTdlTimeChanCoeKernel<Tscalar, Tcomplex>));

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

    // normalize the tdl time channel per TTI if macro ENABLE_NORMALIZATION_ is defined, not used by default
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
    cudaStreamSynchronize(this -> m_strm); // finish constructor
}

template <typename Tscalar, typename Tcomplex> 
void tdlChan<Tscalar, Tcomplex>::run(float refTime0, uint8_t enableSwapTxRx, uint8_t txColumnMajorInd)
{
    this -> genChanProcSig(refTime0, enableSwapTxRx, txColumnMajorInd);  // cuda stream synchornize within
}

template <typename Tscalar, typename Tcomplex> 
void tdlChan<Tscalar, Tcomplex>::reset()
{
    updateTapPathRand(); // cuda stream synchornize within
}

template <typename Tscalar, typename Tcomplex> 
void tdlChan<Tscalar, Tcomplex>::saveTdlChanToH5File(std::string & padFileNameEnding)
{
    cudaStreamSynchronize(this -> m_strm);
    std::string outFilename = "tdlChan_" + std::to_string(this -> m_nCell) + "cell" + std::to_string(this -> m_nUe)+ "Ue_" + std::to_string(this -> m_nBsAnt) + "x" + std::to_string(this -> m_nUeAnt) + "_" + m_delayProfile + std::to_string(int(m_delaySpread)) + "_dopp" + std::to_string(int(this -> m_maxDopplerShift)) + "_cfo" + std::to_string(int(this -> m_cfoHz)) + "_runMode" + std::to_string(this -> m_runMode) + "_freqConvert" + std::to_string(this -> m_freqConvertType) + "_scSampling" + std::to_string(this -> m_scSampling) + (typeid(Tcomplex) == typeid(__half2) ? "_FP16" : "_FP32") + "_swap" + std::to_string(this -> m_enableSwapTxRx) + padFileNameEnding + ".h5";
    // Initialize HDF5
    hid_t tdlH5File = H5Fcreate(outFilename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT); // non empty existing file will be overwritten
    if (tdlH5File < 0)
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
    this -> saveChanToH5File(tdlH5File, complexDataType);
    // save random numbers used to generate tdl channel, each link saved as a dataset
    dims[0] = m_randSizePerLink * 2;
    for(uint32_t linkIdx = 0; linkIdx < this -> m_nLink; linkIdx ++)
    {
        std::string thetaRandLinkIdxStr = "thetaRandLink" + std::to_string(linkIdx);
        writeHdf5DatasetFromGpu<Tscalar>(tdlH5File, thetaRandLinkIdxStr.c_str(), typeid(Tcomplex) == typeid(__half2) ? fp16Type : H5T_IEEE_F32LE, m_h_deviceThetaRandPerLinkPtr[linkIdx], dims, rank);
    }

    dims[0] = m_firNzTapMap.size();
    writeHdf5DatasetFromGpu<uint16_t>(tdlH5File, "firNzTapMap", H5T_STD_U16LE, m_tdlDynDescrCpu -> firNzTapMap, dims, rank);

    // save tdlCfg, data saved on CPU; create a compound data type for tdlConfig_t
    hid_t compType = H5Tcreate(H5T_COMPOUND, sizeof(tdlConfig_t));
    H5Tinsert(compType, "useSimplifiedPdp", HOFFSET(tdlConfig_t, useSimplifiedPdp), H5T_NATIVE_CHAR);
    H5Tinsert(compType, "delayProfile", HOFFSET(tdlConfig_t, delayProfile), H5T_NATIVE_CHAR);
    H5Tinsert(compType, "delaySpread", HOFFSET(tdlConfig_t, delaySpread), H5T_IEEE_F32LE);
    H5Tinsert(compType, "maxDopplerShift", HOFFSET(tdlConfig_t, maxDopplerShift), H5T_IEEE_F32LE);
    H5Tinsert(compType, "f_samp", HOFFSET(tdlConfig_t, f_samp), H5T_IEEE_F32LE);
    H5Tinsert(compType, "nCell", HOFFSET(tdlConfig_t, nCell), H5T_STD_U16LE);
    H5Tinsert(compType, "nUe", HOFFSET(tdlConfig_t, nUe), H5T_STD_U16LE);
    H5Tinsert(compType, "nBsAnt", HOFFSET(tdlConfig_t, nBsAnt), H5T_STD_U16LE);
    H5Tinsert(compType, "nUeAnt", HOFFSET(tdlConfig_t, nUeAnt), H5T_STD_U16LE);
    H5Tinsert(compType, "fBatch", HOFFSET(tdlConfig_t, fBatch), H5T_STD_U32LE);
    H5Tinsert(compType, "numPath", HOFFSET(tdlConfig_t, numPath), H5T_STD_U16LE);
    H5Tinsert(compType, "cfoHz", HOFFSET(tdlConfig_t, cfoHz), H5T_IEEE_F32LE);
    H5Tinsert(compType, "delay", HOFFSET(tdlConfig_t, delay), H5T_IEEE_F32LE);
    H5Tinsert(compType, "sigLenPerAnt", HOFFSET(tdlConfig_t, sigLenPerAnt), H5T_STD_U32LE);
    H5Tinsert(compType, "N_sc", HOFFSET(tdlConfig_t, N_sc), H5T_STD_U16LE);
    H5Tinsert(compType, "N_sc_Prbg", HOFFSET(tdlConfig_t, N_sc_Prbg), H5T_STD_U16LE);
    H5Tinsert(compType, "scSpacingHz", HOFFSET(tdlConfig_t, scSpacingHz), H5T_IEEE_F32LE);
    H5Tinsert(compType, "freqConvertType", HOFFSET(tdlConfig_t, freqConvertType), H5T_NATIVE_CHAR);
    H5Tinsert(compType, "scSampling", HOFFSET(tdlConfig_t, scSampling), H5T_NATIVE_CHAR);
    H5Tinsert(compType, "runMode", HOFFSET(tdlConfig_t, runMode), H5T_NATIVE_CHAR);
    H5Tinsert(compType, "procSigFreq", HOFFSET(tdlConfig_t, procSigFreq), H5T_NATIVE_CHAR);
    // H5Tinsert(compType, "batchLen", HOFFSET(tdlConfig_t, batchLen), H5T_STD_U32LE); skip the vector
    // H5Tinsert(compType, "txSigIn", HOFFSET(tdlConfig_t, txSigIn), H5T_STD_U32LE); skip the pointer

    // save tdl config 
    dims[0] = 1;
    hid_t dataspaceId = H5Screate_simple(rank, dims, nullptr);
    hid_t datasetId = H5Dcreate2(tdlH5File, "tdlCfg", compType, dataspaceId, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    herr_t status = H5Dwrite(datasetId, compType, H5S_ALL, H5S_ALL, H5P_DEFAULT, m_tdlCfg);
    H5Dclose(datasetId);
    H5Sclose(dataspaceId);

    // Close HDF5 objects 
    H5Fclose(tdlH5File);
}

template <typename Tscalar, typename Tcomplex> 
void tdlChan<Tscalar, Tcomplex>::printGpuMemUseMB()
{
    float gpuMemUseMB = this -> m_gpuDataUsageByte / 1024.0f / 1024.0f;
    printf("TDL channel class uses %.2f MB GPU memory. \n", gpuMemUseMB);
}

/*----------------------      begin CUDA kernels        -----------------------------*/
template <typename Tscalar, typename Tcomplex> 
static __global__ void genTdlTimeChanCoeKernel(fastFadingDynDescr_t<Tscalar, Tcomplex> * fastFadingDynDescr, float refTime0, tdlDynDescr_t<Tscalar, Tcomplex> * tdlDynDescr)
{
    // GRID(nLink, nBatch, 1);
    // BLOCK(nTaps, nBsAnt/scaleBsAntTimeChan, nUeAnt/scaleUeAntTimeChan);

    uint16_t cidUidOffset = blockIdx.x; // linkIdx
    uint16_t batchIdx = blockIdx.y;
    uint16_t tapIdx = threadIdx.x;

    uint16_t nBsAnt   = fastFadingDynDescr -> nBsAnt;
    uint16_t nUeAnt   = fastFadingDynDescr -> nUeAnt;
    uint16_t firNzLen = fastFadingDynDescr -> firNzLen;
    uint16_t nPath = tdlDynDescr -> nPath;
    uint16_t nTaps = tdlDynDescr -> nTaps;
    uint16_t * firNzTapMap = tdlDynDescr -> firNzTapMap;
    Tcomplex * timeChan = fastFadingDynDescr -> timeChan;
    Tscalar * thetaRand = tdlDynDescr -> thetaRand[cidUidOffset]; // thetaRand in (0.0f, 1.0f]
    float PI_4_nPath = tdlDynDescr -> PI_4_nPath; // a constant pi/4/nPath
    float PI_2_nPath = tdlDynDescr -> PI_2_nPath; // a constant pi/2/nPath
    float timeStamp = fastFadingDynDescr -> tBatch[batchIdx] + refTime0;
    float maxDopplerShift = fastFadingDynDescr -> maxDopplerShift;

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

    // tx and rx ant index increamental increase
    float alpha_0 = PI_4_nPath * (tapIdx+1) / (nTaps+2);
    for(uint16_t ueAntIdx = threadIdx.z; ueAntIdx < nUeAnt; ueAntIdx += blockDim.z)
    {
        for(uint16_t bsAntIdx = threadIdx.y; bsAntIdx < nBsAnt; bsAntIdx += blockDim.y)
        {
            uint32_t pathOffset = nPath * (( ueAntIdx*nBsAnt + bsAntIdx)*blockDim.x + tapIdx);
            uint32_t globalChanIdx = (((cidUidOffset*gridDim.y + batchIdx)*nUeAnt + ueAntIdx)*nBsAnt + bsAntIdx)*firNzLen + firNzTapMap[tapIdx]; // [nLink, nBatch, nUeAnt, nBsAnt, firNzLen]
            cuComplex tempChanCoe = {0.0f, 0.0f}; // for superimpose of nPath sins
            
            float freqReal, freqImag; // for doppler of real and imag
            for(uint16_t pathIdx = 0; pathIdx < nPath; pathIdx++)
            {
                // calculate doppler frequency
                float tmpDoppler = PI_2_nPath * (pathIdx + 0.5f);

                freqReal = maxDopplerShift * cos(tmpDoppler + alpha_0);
                freqImag = maxDopplerShift * cos(tmpDoppler - alpha_0);
                // calculate channel per path
                tempChanCoe.x = tempChanCoe.x + cos(2 * M_PI * (timeStamp * freqReal + static_cast<float>(thetaRand[(pathOffset + pathIdx)*2])));
                tempChanCoe.y = tempChanCoe.y + cos(2 * M_PI * (timeStamp * freqImag + static_cast<float>(thetaRand[(pathOffset + pathIdx)*2 + 1])));
                // TODO for TDL-D and TDL-E, the first tap is LOS
                // if((tdlDynDescr -> LosTap) && (tapIdx==0)) // first tap, LOS in TDL-D and TLD-E
            }
            // multiple by FIR and save to gloabl memory; firNzPw has already been normalized
            atomicAdd(&(timeChan[globalChanIdx].x), fastFadingDynDescr -> firNzPw[tapIdx] * tempChanCoe.x);
            atomicAdd(&(timeChan[globalChanIdx].y), fastFadingDynDescr -> firNzPw[tapIdx] * tempChanCoe.y);
        }
    }
}