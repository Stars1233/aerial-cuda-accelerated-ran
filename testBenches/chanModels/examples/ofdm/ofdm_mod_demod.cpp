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

#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <cmath>
#include <numeric>
#include <vector>
#include "hdf5hpp.hpp"
#include "chanModelsCommon.h"

#include "ofdmMod.cuh"
#include "ofdmDemod.cuh"

/**
 * @brief read caerrier info from h5 file data set element
 * 
 * @param CarrierPrms struct to hold carrier info
 * @param dset_elem dataset element
 */
inline void carrier_pars_from_dataset_elem(cuphyCarrierPrms * CarrierPrms, const hdf5hpp::hdf5_dataset_elem& dset_elem)
{
    CarrierPrms -> N_sc                   = dset_elem["N_sc"].as<uint16_t>();
    CarrierPrms -> N_FFT                  = dset_elem["N_FFT"].as<uint16_t>();
    CarrierPrms -> N_bsLayer              = dset_elem["N_bsLayer"].as<uint16_t>();
    CarrierPrms -> N_ueLayer              = dset_elem["N_ueLayer"].as<uint16_t>();
    CarrierPrms -> id_slot                = dset_elem["id_slot"].as<uint16_t>();
    CarrierPrms -> id_subFrame            = dset_elem["id_subFrame"].as<uint16_t>();
    CarrierPrms -> mu                     = dset_elem["mu"].as<uint16_t>();
    CarrierPrms -> cpType                 = dset_elem["cpType"].as<uint16_t>();
    CarrierPrms -> f_c                    = dset_elem["f_c"].as<uint32_t>();
    CarrierPrms -> f_samp                 = dset_elem["f_samp"].as<uint32_t>();
    CarrierPrms -> N_symbol_slot          = dset_elem["N_symbol_slot"].as<uint16_t>();
    CarrierPrms -> kappa_bits             = dset_elem["kappa_bits"].as<uint16_t>();
    CarrierPrms -> ofdmWindowLen          = dset_elem["ofdmWindowLen"].as<uint16_t>();
    CarrierPrms -> rolloffFactor          = dset_elem["rolloffFactor"].as<float>();
}


/**
 * @brief check ofdm results
 * if TV is used, we will compare both time and freq samples
 * otherwise, only compare freq tx with freq rx, whether they match
 * 
 * @param dataIn CPU buffer for comparison
 * @param dataOut CPU buffer for comparison
 * @param compareLen length of comparison
 * @return true if dataIn match with dataOut with tolerance
 * @return false if dataIn does match with dataOut with tolerance
 */
template<typename Tscalar, typename Tcomplex>
bool checkOfdmRes(Tcomplex * dataIn, Tcomplex * dataOut, int compareLen)
{
    const Tscalar tolerance = static_cast<Tscalar>(0.001f);
    for(int i=0; i<compareLen; i++)
    {
        auto approxEq = [&](Tscalar a, Tscalar b) {
            Tscalar diff = std::fabs(float(a) - float(b));
            Tscalar m = std::max(std::fabs(float(a)), std::fabs(float(b)));
            if (m <= std::numeric_limits<Tscalar>::epsilon()) {
                return diff <= tolerance;
            }
            Tscalar ratio = (diff >= tolerance) ? (Tscalar)(diff / m) : diff;
            return ratio <= tolerance;
        };
        if(!(approxEq(dataIn[i].x, dataOut[i].x) && approxEq(dataIn[i].y, dataOut[i].y)))
        {
            printf("input and output samples do not match! starting %d\n", i);
            printf("In: %f + %f i, out %f + %f i\n", float(dataIn[i].x), float(dataIn[i].y), float(dataOut[i].x), float(dataOut[i].y));
            return false;
        }
    }
    return true;
}

/**
 * @brief printout usage message
 * 
 */
void usage() {
    std::cout << "OFDM modulation and demodulation test [options]" << std::endl;
    std::cout << "  Arguments:" << std::endl;
    printf("  -i  [input TV file name to read carrier pars, freq tx/rx smaples]\n");
    printf("  -r  [random seed (default 0)]\n");
    printf("  -n  [number of TTIs (default 10)]\n");
    printf("  -p  [enable half precision (default disable)]\n");
    printf("  -d  [enable debug to print sample channel and save TDL coes in H5 file (default disable)]\n");
    printf("  -s  [enable swap of tx and rx (default disable)]\n");
    printf("  -h  display usage information \n");
    std::cout << "Example (seed 0, 10 iterations, random samples, using FP32): ./ofdm_mod_demod_ex" << std::endl;
    std::cout << "Example (seed 0, 10 iterations, using TV, using FP32): ./ofdm_mod_demod_ex -i <path to TV> " << std::endl;
    std::cout << "Example (seed 0, 10 iterations, random samples, using FP16): ./ofdm_mod_demod_ex -p" << std::endl;
    std::cout << "Example (seed 10, 100 iterations, random samples, using FP16): ./ofdm_mod_demod_ex -i <path to TV> -p -r 10 -n 100" << std::endl;
}

/**
 * @brief main test function for OFDM modulation and demodulation
 * 
 * @tparam Tcomplex Template for complex number, cuPHY tensor type and the scalar type will be automatically decided
 * @param inputFileName input TV file name to read carrier pars, channel pars, freq rx smaples
 * @param nTti number of iteration to measure time
 * @param randSeed random seed, used for TDL and generating noise
 * @param enableHalfPrecision true for use FP16, false for use FP32
 * @param debugInd true for use FP16, false for use FP32
 */
template<typename Tscalar, typename Tcomplex>
void test_OFDM(std::string inputFileName, int nTti = 10, uint16_t randSeed = 0, bool & enableHalfPrecision = false, uint8_t & enableSwapTxRx = 0,bool & debugInd = false)
{
    bool prach = 0;
    cudaStream_t cuMainStrm;
    cudaStreamCreate(&cuMainStrm);

    cuphyCarrierPrms * CarrierPrms = new cuphyCarrierPrms;
    /* ----   Config carrierPrms based on TV ---------*/
    if(!inputFileName.empty())
    {
        hdf5hpp::hdf5_file input_file = hdf5hpp::hdf5_file::open(inputFileName.c_str());
        hdf5hpp::hdf5_dataset dset_gnb  = input_file.open_dataset("carrier_pars");
        carrier_pars_from_dataset_elem(CarrierPrms, dset_gnb[0]);
    }

    uint  blocks = (enableSwapTxRx ? (CarrierPrms -> N_ueLayer) : (CarrierPrms -> N_bsLayer)) * (CarrierPrms -> N_symbol_slot);
    CarrierPrms -> ofdmWindowLen = 0;
    const uint  freqDataSize = (CarrierPrms -> N_sc) * blocks;

    Tcomplex* freqDataInGpuPtr = nullptr;
    Tcomplex* freqDataOutGpuPtr = nullptr;
    CHECK_CUDAERROR(cudaMalloc(&freqDataInGpuPtr, sizeof(Tcomplex) * freqDataSize));
    CHECK_CUDAERROR(cudaMalloc(&freqDataOutGpuPtr, sizeof(Tcomplex) * freqDataSize));

    ofdm_modulate::ofdmModulate<Tscalar, Tcomplex> * ofdmMod = new ofdm_modulate::ofdmModulate<Tscalar, Tcomplex>(CarrierPrms, freqDataInGpuPtr, cuMainStrm);

    const uint timeDataSize = ofdmMod -> getTimeDataLen();
    Tcomplex * freqDataInCpu = new Tcomplex[freqDataSize];
    Tcomplex * freqDataOutCpu = new Tcomplex[freqDataSize];

    // printf("timeDataSize = %d \n", timeDataSize);
    Tcomplex * timeDataOutCpu = new Tcomplex[timeDataSize];
    Tcomplex * timeDataOutCpu_ref = new Tcomplex[timeDataSize];

    if(!inputFileName.empty())
    {
        // Read input HDF5 file to read rate-matching output.
        printf("Using TV to test OFDM, time and frequency domain ref check enabled \n");
        hdf5hpp::hdf5_file input_file = hdf5hpp::hdf5_file::open(inputFileName.c_str());
        hdf5hpp::hdf5_dataset Xtf_dataset = input_file.open_dataset("X_tf");
        Xtf_dataset.read(freqDataInCpu);
        cudaMemcpyAsync(freqDataInGpuPtr, freqDataInCpu, sizeof(Tcomplex)*freqDataSize, cudaMemcpyHostToDevice, cuMainStrm);

        // read reference time domain data
        hdf5hpp::hdf5_dataset Xt_dataset = input_file.open_dataset("X_t");
        Xt_dataset.read(timeDataOutCpu_ref);//freqDataInGpu.addr());        
    }
    else
    {
        printf("Using random input to test OFDM, only frequency domain ref check enabled \n");
        curandGenerator_t curandGen;
        CHECK_CURANDERROR(curandCreateGenerator(&curandGen, CURAND_RNG_PSEUDO_DEFAULT));
        CHECK_CURANDERROR(curandSetStream(curandGen, cuMainStrm));
        if constexpr (std::is_same_v<Tcomplex, cuComplex>) {
            CHECK_CURANDERROR(curandGenerateNormal(curandGen, (float*)freqDataInGpuPtr, freqDataSize * 2, 0.0f, sqrtf(0.5f)));
        } else {
            float* d_tmpBuf = nullptr;
            CHECK_CUDAERROR(cudaMalloc((void**)&d_tmpBuf, freqDataSize * 2 * sizeof(float)));
            CHECK_CURANDERROR(curandGenerateNormal(curandGen, d_tmpBuf, freqDataSize * 2, 0.0f, sqrtf(0.5f)));
            std::vector<float> h_floats(freqDataSize * 2);
            CHECK_CUDAERROR(cudaMemcpy(h_floats.data(), d_tmpBuf, freqDataSize * 2 * sizeof(float), cudaMemcpyDeviceToHost));
            cudaFree(d_tmpBuf);
            std::vector<__half> h_halves(freqDataSize * 2);
            for (uint32_t i = 0; i < freqDataSize * 2; ++i) {
                h_halves[i] = __float2half(h_floats[i]);
            }
            CHECK_CUDAERROR(cudaMemcpy(freqDataInGpuPtr, h_halves.data(), freqDataSize * 2 * sizeof(__half), cudaMemcpyHostToDevice));
        }
        CHECK_CURANDERROR(curandDestroyGenerator(curandGen));
    }
    cudaStreamSynchronize(cuMainStrm);

    // get time signal from OFDM modulate for OFDM demodulation
    Tcomplex * timeDataInGpu = ofdmMod -> getTimeDataOut();

    ofdm_demodulate::ofdmDeModulate<Tscalar, Tcomplex> * ofdmDeMod = new ofdm_demodulate::ofdmDeModulate<Tscalar, Tcomplex>(CarrierPrms, timeDataInGpu, freqDataOutGpuPtr, prach, 0 /*perAntSamp*/, cuMainStrm);

    // measure time
    float elapsedTime;
    std::vector<float> measureTime;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for(int sta=0; sta<nTti; sta++)
    {
        cudaEventRecord(start, cuMainStrm);

        // OFDM modulation
        ofdmMod -> run(enableSwapTxRx, cuMainStrm);
        
        // OFDM demodulation
        ofdmDeMod -> run(enableSwapTxRx, cuMainStrm);

        if(debugInd)
        {
            ofdmMod -> printTimeSample();
            ofdmMod -> saveOfdmModToH5File(cuMainStrm);
            ofdmDeMod -> printFreqSample();
            ofdmDeMod -> saveOfdmDemodToH5File(cuMainStrm);
        }

        cudaEventRecord(stop, cuMainStrm);
        CHECK_CUDAERROR(cudaEventSynchronize(start));
        CHECK_CUDAERROR(cudaEventSynchronize(stop));
        cudaDeviceSynchronize();
        cudaEventElapsedTime(&elapsedTime, start, stop);
        measureTime.push_back(elapsedTime);
    }

    /*-----------------        check results match       --------------*/
    // check whether ofdm modulation input vs ofdm demodulation out 
    CHECK_CUDAERROR(cudaMemcpy(freqDataInCpu, freqDataInGpuPtr, freqDataSize * sizeof(Tcomplex), cudaMemcpyDeviceToHost));
    CHECK_CUDAERROR(cudaMemcpy(freqDataOutCpu, ofdmDeMod -> getFreqDataOut(), freqDataSize * sizeof(Tcomplex), cudaMemcpyDeviceToHost));
    if(checkOfdmRes<Tscalar, Tcomplex>(freqDataInCpu, freqDataOutCpu, freqDataSize))
    {
        printf("OFDM test PASS, frequency input and output samples match \n");
    }
    else
    {
        printf("OFDM test FAIL, frequency input and output samples do not match \n");
    }

    // checkofdm modulation output vs ref TV 
    if(!inputFileName.empty())
    {
        CHECK_CUDAERROR(cudaMemcpy(timeDataOutCpu, timeDataInGpu, timeDataSize * sizeof(Tcomplex), cudaMemcpyDeviceToHost));
        if(checkOfdmRes<Tscalar, Tcomplex>(timeDataOutCpu_ref, timeDataOutCpu, timeDataSize))
        {
            printf("OFDM test PASS, time input and output samples match \n");
        }
        else
        {
            printf("OFDM test FAIL, time input and output samples do not match \n");
        }
    }

    // print running time info
    printf("Total OFDM modulation and demodulation time (include add and remove CP) cost: %f millisecond (avg over %ld runs) \n", std::reduce(measureTime.begin(), measureTime.end())/float(measureTime.size()), measureTime.size());
    
    // release allocated buffers
    delete[] freqDataInCpu;
    delete[] freqDataOutCpu;
    delete[] timeDataOutCpu;
    delete[] timeDataOutCpu_ref;

    cudaFree(freqDataInGpuPtr);
    cudaFree(freqDataOutGpuPtr);
    delete CarrierPrms;
    delete ofdmMod;
    delete ofdmDeMod;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(cuMainStrm);
}

int main(int argc, char* argv[])
{  
    std::string inputFileName; // input TV name
    uint16_t randSeed = 0; // random seeds, used for TDL and generating noise
    int nTti = 10;  // number of TTIs
    bool enableHalfPrecision = false; // true for use FP16, false for use FP32
    uint8_t enableSwapTxRx = 0; // 0 for no swap, 1 for swap Tx and Rx
    bool debugInd   = false; // true for debug info, false for no debug info
    // arguments parser
    // inputFilename fadingMode randSeed nTti phyChanType enableHalfPrecision debugInd
    int iArg = 1;
    // read options from CLI
    while(iArg < argc) {
      if('-' == argv[iArg][0]) {
        switch(argv[iArg][1]) {
            case 'i':
                if(++iArg >= argc)
                {
                    fprintf(stderr, "ERROR: Invalid input files \n");
                    exit(1);
                }
                inputFileName.assign(argv[iArg++]);
                break;
            case 'r': // random seed 
                if((++iArg >= argc) || (1 != sscanf(argv[iArg], "%hu", &randSeed)) || (randSeed < 0))
                {
                    fprintf(stderr, "ERROR: Invalid input random seed.\n");
                    exit(1);
                } 
                ++iArg;
                break;
            case 'n': // number of TTI
                if((++iArg >= argc) || (1 != sscanf(argv[iArg], "%u", &nTti)) || (nTti < 0))
                {
                fprintf(stderr, "ERROR: Invalid number of TTIs.\n");
                exit(1);
                } 
                ++iArg;
                break;
            case 'p': // enable half precision
                enableHalfPrecision = true;
                ++iArg;
                break;
            case 's': // enable swap of tx and rx
                enableSwapTxRx = 1;
                ++iArg;
                break;
            case 'd': // enable debug
                debugInd = true;
                ++iArg;
                break;
            case 'h': // print help usage
                usage();
                exit(0);
                break;
            default:
                fprintf(stderr, "ERROR: Unknown option: %s\n", argv[iArg]);
                usage();
                exit(1);
                break;
            }
        }
        else
        {
            fprintf(stderr, "ERROR: Invalid command line argument: %s\n", argv[iArg]);
            usage();
            exit(1);
        }
    }

    // start testing based on TV, iter, and precision 
    if(enableHalfPrecision)
    {
        printf("OFDM mod and demod test: Using 16 bits precision, %s, random seed = %d, %d iterations \n", inputFileName.empty() ? "random input signals" : "using TV", randSeed, nTti);
        test_OFDM<__half, __half2>(inputFileName, nTti, randSeed, enableHalfPrecision, enableSwapTxRx, debugInd);        
    }
    else
    {
        printf("OFDM mod and demod test: Using 32 bits precision, %s, random seed = %d, %d iterations \n",  inputFileName.empty() ? "random input signals" : "using TV", randSeed, nTti);
        test_OFDM<float, cuComplex>(inputFileName, nTti, randSeed, enableHalfPrecision, enableSwapTxRx, debugInd); 
    }

    return 0;
}
