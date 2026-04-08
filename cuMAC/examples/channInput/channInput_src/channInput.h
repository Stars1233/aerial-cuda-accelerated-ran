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

#ifndef CHANNINPUT_H_INCLUDED_
#define CHANNINPUT_H_INCLUDED_

#pragma once

// Unit test for generate input 
// Read channel from Asim channel file

#include "api.h"
#include "cumac.h"
#include "parameters.h"
#include "H5Cpp.h"

#define m_rawChannMaxDim 5
#define m_outChannMaxDim 5
#define USE_SNR_BASED_SCALING 1  // switch between snrBasedScaling and autoScaling

using namespace cumac;

/**
 * @brief Output channel descriptor
 * 
 * @tparam Tcomplex template for complex number
 */
template <typename Tcomplex> 
struct outChanDescr_t
{
    uint nPrbGrp;   // number of PRBG
    uint nUe;       // number of UE
    uint nCell;     // number of cell
    uint nBsAnt;    // number of BS antenna
    uint nUeAnt;    // number of UE antenna
    uint outChanSize; // number of channel coes, i.e., nPrbGrp * nUe * nCell * nBsAnt * nUeAnt
    Tcomplex * chanPtr; // GPU buffer for chan
    float sigmaSqrd;  // noise level
    float scalingFactor; // scaling factor for channel coe; noise will be AFTER_SCALING_SIGMA_CONST^2, only used in const scaling
};


template <typename inChan_T, typename outChan_T> class channInput{
public:
    channInput(outChan_T* cpuOutChannData, const cumacCellGrpPrms* cellGrpPrms, const cumacSimParam* simParam, std::string iFileName = NULL, cudaStream_t strm = 0, float amplifyCoe = 1.0, uint8_t asimChanFlag = 0);
    ~channInput();
    
    uint16_t* getRawChannDim(); // read raw channel dimension
    uint16_t* getOutChannDim(); // read cuMAC channel dimention
    void setCellsIndex(uint16_t * cellsID); // set cooperative cells ID, by default [0..m_numCells]
    void setCellBias(float * biasGain); // preset cell biased that some cells have higher channel coe 

    // void setup();
    void run(int TTI_index, bool channRenew = false); // obtain the channel for current TTI, reuse TTI if input channel has not been changed
    void printRawChann(short printRawSize = 10); // print first of rawChannel
    void printCpuOutChann(short printOutSize = 10); // print first of outChannel
    void printGpuOutChann(short printOutSize = 10); // print first of 
    void printCellAssoc(); //print cellAssociation
    outChan_T * getCpuOutChann(); // get pointer to CPU channel out buffer
    outChan_T * getGpuOutChann(); // get pointer to GPU channel out buffer
    float * getBiasGain(); // get bias gain
    float * getLongTermDataRate(); // get long term data rates for the UEs
    float getNoiseSqrd() {return m_sigmaSqrd;}; // get updated noise variance
    uint8_t* getCellAssoc(){return cellAssoc;} // get cell association from input channel

private:
    uint16_t * m_cellsIDs; // indexes of the coopeartive cells
    uint16_t m_numCells; // number of cells
    uint16_t * m_ueIDs; // indexes of ues
    float * m_longTermDataRate; // long term data rates from input channel
    uint16_t m_numUes; // number of ues
    float * m_biasGain; // bias gain on cell level
    int m_chanMode; // how to generate channel
    cudaStream_t m_cuStream; // stream for transfer
    outChan_T * m_GPUestH_fr; // buffer in GPU memory
    float m_amplifyCoe; // to amplify the CSI for half precision, can be related to noise
    float m_sigmaSqrd; // noise variance

    // Raw channel data from MAT or h5 file
    std::string m_iFileName; //input file name
    uint16_t * m_rawChannDim; // Dimension of raw channel from MAT or h5 file
    int m_rawChannSize; // total number of data points of raw channel
    int m_rawChannPerCell; // total number of raw data points per cell
    inChan_T * m_rawChannData; // raw channel from MAT or h5 file, also buffer for preprocessing channel data
    // Output channel data to cuMAC
    uint16_t * m_outChannDim; // Dimension of output channel to cuMAC, consistent to cumacCellGrpPrms parameters & format in network.cpp

    int m_outChannSize; // total number of data points of ouput channel
    outChan_T* m_CpuOutChannData; // output address in GPU for channel data

    // channel descriptors
    outChanDescr_t<outChan_T> * m_chanDescrCpu; // chann descritor on CPU
    outChanDescr_t<outChan_T> * m_chanDescrGpu; // chann descritor on GPU

    // cellAssociation
    uint8_t * cellAssoc;

    // lauch kernel parameters
    void * m_args[1]; // launch kernel arguments
    dim3 m_gridDim, m_blockDim; // grid and block dims
    cudaFunction_t m_functionPtr; // functino ptr

    void extractRawChannelData(int TTI_offset = 0); // read raw channel data from file
    
    // different ways to read raw channel
    void genRandomChann(); // generate random chann with bias per cell
    void readMatFile(int TTI_offset = 0); // if MATLAB save by ('-v7.3'), mat can be directly read by H5
    void readCsvFile(int TTI_offset = 0); // read from csv with real and imag separation
    void readDatFile(int TTI_offset = 0); // TODO: read from dat file
    // void readAsimFile(int TTI_offset = 0); // read ASIM file following it's data structure
    // preprocessing to output channel
    void preProcessing(); // translate raw channel to cuMAC data
    void permuteChannDim(); // permute dimension for cuMAC use

    /**
     * @brief auto scale the chan coe and noise power to ensure precision
     * @note their ratio are kept
     */
    void autoScaling();

    /**
     * @brief using a const scaling factor based on SCALING_NOISE_SIGMA_CONST to scale channel and noise
     * @note their ratio are kept
     */
    void snrBasedScaling();
    // I/O between CPU and GPU
    void transferToGPU(); // transfer data to GPU
    void transferToCPU(); // transfer data to GPU

};

/**
 * @brief CUDA kernel for auto scaling
 *      gridDim = {1,1,1};
 *      blockDim = {nUe, 1, 1};
 *
 * @param chanDescr  channel descriptor
 */
template <typename outChan_T> 
static __global__ void autoScalingKernel(outChanDescr_t<outChan_T> * chanDescr); // for GPU autoscaling

/**
 * @brief CUDA kernel for const scaling
 *      gridDim = {1,1,1};
 *      blockDim = {1024, 1, 1};
 *
 * @param chanDescr  channel descriptor
 */
template <typename outChan_T> 
static __global__ void snrBasedScalingKernel(outChanDescr_t<outChan_T> * chanDescr); // for GPU const scaling


// template __global__ void autoScalingKernel<cuComplex>(outChanDescr_t<cuComplex> * chanDescr); // for GPU autoscaling
// template __global__ void autoScalingKernel<__half2>(outChanDescr_t<__half2> * chanDescr); // for GPU autoscaling

// Explicitly instantiate the template to resovle "undefined functions"
template class channInput<cuComplex, cuComplex>; 
template class channInput<cuComplex, __half2>;

#endif