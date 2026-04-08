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

#include "ofdmMod.cuh"
#include <cufftdx.hpp>

using namespace ofdm_modulate;

/**
 * @brief main kernel for ofdm modulation
 * 
 * @tparam FFT FFT configurations, see cuFFTdx documents for detals
 * @tparam Tscalar scalar template, must match with Tcomplex
 * @tparam Tcomplex comlex template, must match with Tscalar
 * @param ofdmModdynDescpr ofdm demodulation dynamic descriptor
 * 
 * @param GridDim m_ofdmDeModdynDescprCpu -> N_bsLayer or N_ueLayer, cuphyCarrierPrms -> N_symbol_slot / OFDM_FFTs_PER_BLOCK_CONST_, 1
 * @param BlockDim defuallt set by cuFFTdx
 */
template<typename FFT, typename Tscalar, typename Tcomplex>
__launch_bounds__(FFT::max_threads_per_block)
static __global__ void ofdmMod_ifft_kernel(ofdmModDynDescr_t<Tscalar, Tcomplex> * ofdmModdynDescpr)
{
    using namespace cufftdx;
    // Registers
    cuComplex thread_data[FFT::storage_size];
    Tcomplex * freqDataIn = ofdmModdynDescpr -> freqDataIn;
    Tcomplex * timeDataOut = ofdmModdynDescpr -> timeDataOut;
    uint N_sc_over_2 = ofdmModdynDescpr -> N_sc >> 1; // divide by 2
    uint N_IFFT = ofdmModdynDescpr -> N_IFFT;
    // Local batch id of this FFT in CUDA block, in range [0; FFT::ffts_per_block)
    const unsigned int local_fft_id = threadIdx.y;
    // Global batch id of this FFT in CUDA grid is equal to number of batches per CUDA block (ffts_per_block)
    // times CUDA block id, plus local batch id.
    const unsigned int global_fft_id = (blockIdx.x * gridDim.y + blockIdx.y) * FFT::ffts_per_block + local_fft_id;
    // blockIdx.x = N_layer, blockIdx.y = N_symbol_slot (14 or 12) / OFDM_FFTs_PER_BLOCK_CONST_ 

    /*-------------------   load data into shared memoery for IFFT-------------------*/
    // Load freq data from global memory to registers
    const unsigned int freq_offset = ofdmModdynDescpr -> N_sc * global_fft_id;
    const unsigned int CP_current = ofdmModdynDescpr -> cpInfo[blockIdx.y * FFT::ffts_per_block + threadIdx.y]; // CP length for current OFDM symbol
    const unsigned int CP_offset  = ofdmModdynDescpr -> cpInfo[blockIdx.y * FFT::ffts_per_block + threadIdx.y + (ofdmModdynDescpr -> N_symbol_slot)]; // CP_offset in this layer
    const unsigned int time_offset = cufftdx::size_of<FFT>::value * global_fft_id + CP_offset + (ofdmModdynDescpr -> cpInfo[(ofdmModdynDescpr -> N_symbol_slot) * 2 - 1]) * blockIdx.x; // FFT size + CP offset in this layer + CP offset in previous layers
    const unsigned int stride = FFT::stride;
    unsigned int       index  = threadIdx.x;

    for (unsigned int i = 0; i < FFT::elements_per_thread; i++) 
    {
        // Make sure not to go out-of-bounds
        if ((i * stride + threadIdx.x) < cufftdx::size_of<FFT>::value) 
        {
            #ifdef USE_MEMOERY_FFT_SHIFT_ // perform ifftshift first
            if(index < N_sc_over_2) // CUPHY_N_TONES_PER_PRB = 12
            {
                thread_data[i].x = freqDataIn[index + freq_offset + N_sc_over_2].x;
                thread_data[i].y = freqDataIn[index + freq_offset + N_sc_over_2].y; // first half
            }
            else if( index >= (N_IFFT - N_sc_over_2))
            {
                thread_data[i].x = freqDataIn[index + freq_offset - N_IFFT + N_sc_over_2].x;
                thread_data[i].y = freqDataIn[index + freq_offset - N_IFFT + N_sc_over_2].y; // second half
            }
            else // zero otherwise
            {
                thread_data[i].x = 0.0f; 
                thread_data[i].y = 0.0f; 
            }
            #else // no ifftshift 
            if(index >= ((N_IFFT >> 1) - N_sc_over_2) && index < ((N_IFFT >> 1) + N_sc_over_2) ) // Middl part
            {
                thread_data[i] = freqDataIn[index + freq_offset - ((N_IFFT >> 1) - N_sc_over_2)]; 
            }
            else // zero otherwise
            {
                thread_data[i].x = 0.0f; 
                thread_data[i].y = 0.0f; 
            }
            #endif

            index += stride;
        }
        // printf("IFFT in: global_fft_id=%d, threadIdx.x=%d, threadIdx.y=%d, i=%d, (i * stride + threadIdx.x) = %d: thread_data[i].x = %f, thread_data[i].y = %f \n", global_fft_id, threadIdx.x, threadIdx.y, i, (i * stride + threadIdx.x), float(thread_data[i].x), float(thread_data[i].y));
    }

    // FFT::shared_memory_size bytes of shared memory
    using complex_type = typename FFT::value_type;
    extern __shared__ complex_type shared_mem[];

    // Execute IFFT
    FFT().execute(thread_data, shared_mem);

    /*-------------------   Add CPs  -------------------*/
    index = time_offset + threadIdx.x;
#pragma unroll
    for (unsigned int i = 0; i < FFT::elements_per_thread; i++) 
    {
        if ((i * stride + threadIdx.x) < cufftdx::size_of<FFT>::value) 
        {
            // normalization
            thread_data[i].x = thread_data[i].x * ofdmModdynDescpr -> sqrt_N_IFFT_inverse; // normalize by sqrt(N_IFFT)
            thread_data[i].y = thread_data[i].y * ofdmModdynDescpr -> sqrt_N_IFFT_inverse; // normalize by sqrt(N_IFFT)
            
            // real part
            #ifdef USE_MEMOERY_FFT_SHIFT_ // no change due to ifftshift first
            timeDataOut[index].x = thread_data[i].x;
            timeDataOut[index].y = thread_data[i].y;

            if((i * stride + threadIdx.x) >= N_IFFT - CP_current) // copy CP
            {
                timeDataOut[index - N_IFFT].x = thread_data[i].x;
                timeDataOut[index - N_IFFT].y = thread_data[i].y;
            }
            #else // times 1 or -1 to real part due to no ifftshift
            if(index & 1) // last bit is 1
            {
                timeDataOut[index].x = - thread_data[i].x;   
                timeDataOut[index].y = - thread_data[i].y;
                
                if((i * stride + threadIdx.x) >= N_IFFT - CP_current) // copy CP
                {
                    timeDataOut[index - N_IFFT].x = - thread_data[i].x;
                    timeDataOut[index - N_IFFT].y = - thread_data[i].y;
                }
            }
            else
            {
                timeDataOut[index].x = thread_data[i].x; 
                timeDataOut[index].y = thread_data[i].y; 

                if((i * stride + threadIdx.x) >= N_IFFT - CP_current) // copy CP
                {
                    timeDataOut[index - N_IFFT].x = thread_data[i].x; 
                    timeDataOut[index - N_IFFT].y = thread_data[i].y; 
                }
            }
            #endif
            index += stride;
        }
        // printf("IFFT out: global_fft_id=%d, threadIdx.x=%d, threadIdx.y=%d, i=%d, (i * stride + threadIdx.x) = %d: thread_data[i].x = %f, thread_data[i].y = %f \n", global_fft_id, threadIdx.x, threadIdx.y, i, (i * stride + threadIdx.x), float(thread_data[i].x), float(thread_data[i].y));
    }
}


/**
 * @brief Apply windowing using raised cosine if needed
 * 
 * @param ofdmModdynDescpr ofdm dynamic descriptor
 * @todo not tested yet
 */
template<typename Tscalar, typename Tcomplex>
static __global__ void applyWindow(ofdmModDynDescr_t<Tscalar, Tcomplex> * ofdmModdynDescpr)
{
    // blockIdx.x = layerIdx, threadIdx.y = ofdmIdx, threadIdx.x = [0 ~ ofdmModdynDescpr -> ofdmWindowLen -1];

    uint N_IFFT = ofdmModdynDescpr -> N_IFFT;
    const unsigned int ofdmSymId = threadIdx.y;
    const unsigned int CP_current = ofdmModdynDescpr -> cpInfo[ofdmSymId]; // CP length for current OFDM symbol
    const unsigned int CP_offset  = ofdmModdynDescpr -> cpInfo[ofdmSymId + (ofdmModdynDescpr -> N_symbol_slot)]; // CP_offset in this layer
    const unsigned int block_offset = N_IFFT*ofdmSymId + CP_offset + (N_IFFT * (ofdmModdynDescpr -> N_symbol_slot) + ofdmModdynDescpr -> cpInfo[(ofdmModdynDescpr -> N_symbol_slot) * 2 - 1]) * blockIdx.x; // FFT size + CP offset in this layer + CP offset in previous layers

    /*---------    Apply windowing  ------------ */
    if(ofdmModdynDescpr -> ofdmWindowLen > 2) // need to apply CP
    {
        uint index;
        Tcomplex temp_windowing;
        // get suffix 
        if(ofdmSymId < (ofdmModdynDescpr -> N_symbol_slot - 1)) 
        {
            index = block_offset + threadIdx.x; // FFT start of symbol (ofdmSymId-1) for suffix
            temp_windowing.x =  ofdmModdynDescpr -> ofdmWindowCoe[threadIdx.x] * (ofdmModdynDescpr -> timeDataOut[index]).x;
            temp_windowing.y =  ofdmModdynDescpr -> ofdmWindowCoe[threadIdx.x] * (ofdmModdynDescpr -> timeDataOut[index]).y;

            index = index + N_IFFT;
        }
        else // i.e., ofdmSymId = ofdmModdynDescpr -> N_symbol_slot - 1
        // only apply prefix window to the first symbol, Fix me if multiple slots simulated together
        {
            (ofdmModdynDescpr -> timeDataOut[threadIdx.x]).x = ofdmModdynDescpr -> ofdmWindowCoe[ofdmModdynDescpr -> ofdmWindowLen - 1 - threadIdx.x] * (ofdmModdynDescpr -> timeDataOut[index]).x; 
            (ofdmModdynDescpr -> timeDataOut[threadIdx.x]).y = ofdmModdynDescpr -> ofdmWindowCoe[ofdmModdynDescpr -> ofdmWindowLen - 1 - threadIdx.x] * (ofdmModdynDescpr -> timeDataOut[index]).y; 
        }
    }
}

template<typename Tscalar, typename Tcomplex>
using ifftKernelHandle = void (*)(ofdmModDynDescr_t<Tscalar, Tcomplex> * ofdmModdynDescpr);

// Choose IFFT kernel
template<typename Tscalar, typename Tcomplex, unsigned int FftSize, unsigned int Arch>
ifftKernelHandle<Tscalar, Tcomplex> ofdmMod_get_ifft_param(dim3& block_dim, uint& shared_memory_size) 
{ 
    using namespace cufftdx;

    // use predefined numbers
    using FFT = decltype(Size<FftSize>() + Precision<float>() + Type<fft_type::c2c>()
                                + Direction<fft_direction::inverse>()
                                + FFTsPerBlock<OFDM_FFTs_PER_BLOCK_CONST_>() // + ElementsPerThread<8>()
                                + SM<Arch>() + Block());
    
    // use cuFFTdx configurations
    // Base of the FFT description
    // using FFT_base = decltype(Size<FftSize>() + Precision<Tscalar>() + Type<fft_type::c2c>()
    //                             + Direction<fft_direction::inverse>()
    //                             /* Notice lack of ElementsPerThread and FFTsPerBlock operators */
    //                             + SM<Arch>() + Block());
    // // FFT description with suggested FFTs per CUDA block for the default (optimal) elements per thread
    // using FFT = decltype(FFT_base() + FFTsPerBlock<1>());

    block_dim = FFT::block_dim;
    shared_memory_size = FFT::shared_memory_size;

    return ofdmMod_ifft_kernel<FFT, Tscalar, Tcomplex>;
 }

 /**
  * @brief get ifft kernel handles
  * 
  * @param Nifft IFFT size
  * @param cudaDeviceArch GPU device arch
  * @param block_dim auto config by cuFFTdx
  * @param shared_memory_size auto config by cuFFTdx
  * @return fftKernelHandle<Tscalar, Tcomplex>
  * 
  * @note To conserve memeory, only selected IFFT size and cudaDeviceArch are added. If your Nifft and cudaDeviceArch are not in the below list, please add them and retry the build
  */
template<typename Tscalar, typename Tcomplex>
ifftKernelHandle<Tscalar, Tcomplex> ofdmMod_get_ifft_param(const int Nifft, unsigned int cudaDeviceArch, dim3& block_dim, uint& shared_memory_size) 
{ 
    // current only support cudaDeviceArch = 800, 890, and 900
    if(cudaDeviceArch == 800)
    {
        switch(Nifft) 
        {
            case 256:
                return ofdmMod_get_ifft_param<Tscalar, Tcomplex, 256, 800>(block_dim, shared_memory_size);
                break;
            case 512:
                return ofdmMod_get_ifft_param<Tscalar, Tcomplex, 512, 800>(block_dim, shared_memory_size);
                break;
            case 1024:
                return ofdmMod_get_ifft_param<Tscalar, Tcomplex, 1024, 800>(block_dim, shared_memory_size);
                break;
            case 2048:
                return ofdmMod_get_ifft_param<Tscalar, Tcomplex, 2048, 800>(block_dim, shared_memory_size);
                break;
            case 4096:
                return ofdmMod_get_ifft_param<Tscalar, Tcomplex, 4096, 800>(block_dim, shared_memory_size);
                break;
            default:
                printf("Unsupported IFFT length %d or cudaDeviceArch %d in OFDM modulation, please add your Nifft or cudaDeviceArch into ofdmMod_get_ifft_param and retry\n", Nifft, cudaDeviceArch); 
                exit(EXIT_FAILURE);
        }
    }
    else if(cudaDeviceArch == 890)
    {
        switch(Nifft) 
        {
            case 256:
                return ofdmMod_get_ifft_param<Tscalar, Tcomplex, 256, 890>(block_dim, shared_memory_size);
                break;
            case 512:
                return ofdmMod_get_ifft_param<Tscalar, Tcomplex, 512, 890>(block_dim, shared_memory_size);
                break;
            case 1024:
                return ofdmMod_get_ifft_param<Tscalar, Tcomplex, 1024, 890>(block_dim, shared_memory_size);
                break;
            case 2048:
                return ofdmMod_get_ifft_param<Tscalar, Tcomplex, 2048, 890>(block_dim, shared_memory_size);
                break;
            case 4096:
                return ofdmMod_get_ifft_param<Tscalar, Tcomplex, 4096, 890>(block_dim, shared_memory_size);
                break;
            default:
                printf("Unsupported IFFT length %d or cudaDeviceArch %d in OFDM modulation, please add your Nifft or cudaDeviceArch into ofdmMod_get_ifft_param and retry\n", Nifft, cudaDeviceArch); 
                exit(EXIT_FAILURE);
        }
    }
    else if(cudaDeviceArch == 900)
    {
        switch(Nifft) 
        {
            case 256:
                return ofdmMod_get_ifft_param<Tscalar, Tcomplex, 256, 900>(block_dim, shared_memory_size);
                break;
            case 512:
                return ofdmMod_get_ifft_param<Tscalar, Tcomplex, 512, 900>(block_dim, shared_memory_size);
                break;
            case 1024:
                return ofdmMod_get_ifft_param<Tscalar, Tcomplex, 1024, 900>(block_dim, shared_memory_size);
                break;
            case 2048:
                return ofdmMod_get_ifft_param<Tscalar, Tcomplex, 2048, 900>(block_dim, shared_memory_size);
                break;
            case 4096:
                return ofdmMod_get_ifft_param<Tscalar, Tcomplex, 4096, 900>(block_dim, shared_memory_size);
                break;
            default:
                printf("Unsupported IFFT length %d or cudaDeviceArch %d in OFDM modulation, please add your Nifft or cudaDeviceArch into ofdmMod_get_ifft_param and retry\n", Nifft, cudaDeviceArch); 
                exit(EXIT_FAILURE);
        }
    }
    else
    {
        printf("Unsupported IFFT length %d or cudaDeviceArch %d in OFDM modulation, please add your Nifft or cudaDeviceArch into ofdmMod_get_ifft_param and retry\n", Nifft, cudaDeviceArch); 
        exit(EXIT_FAILURE);
    }
}

template <typename Tscalar, typename Tcomplex> 
ofdmModulate<Tscalar, Tcomplex>::ofdmModulate(cuphyCarrierPrms_t * cuphyCarrierPrms, Tcomplex * freqDataIn, cudaStream_t strm) :
m_cuphyCarrierPrms(cuphyCarrierPrms),
m_enableSwapTxRx(0)
{
    uint mu = cuphyCarrierPrms -> mu;
    uint N_symbol_slot = cuphyCarrierPrms -> N_symbol_slot;
    //m_N_IFFT = cuphyCarrierPrms -> N_IFFT;
    m_ofdmModdynDescprCpu = new ofdmModDynDescr_t<Tscalar, Tcomplex>;
    m_ofdmModdynDescprCpu -> N_IFFT = cuphyCarrierPrms -> N_FFT;
    m_ofdmModdynDescprCpu -> sqrt_N_IFFT_inverse = 1.0f/sqrt(cuphyCarrierPrms -> N_FFT);
    m_ofdmModdynDescprCpu -> N_sc = cuphyCarrierPrms -> N_sc;
    m_ofdmModdynDescprCpu -> N_bsLayer = cuphyCarrierPrms -> N_bsLayer;
    m_ofdmModdynDescprCpu -> N_ueLayer = cuphyCarrierPrms -> N_ueLayer;
    m_ofdmModdynDescprCpu -> mu = mu;
    m_ofdmModdynDescprCpu -> N_symbol_slot = N_symbol_slot;
    uint symbol0IdxPerSubFrame = (cuphyCarrierPrms -> id_slot) * N_symbol_slot;

    /* ----------------  CP info ---------------------- */
    uint16_t cpInfoLen = (N_symbol_slot << 1);
    m_cpInfoCpu = new uint16_t[cpInfoLen]; // [CP info, accumCP] 
    cudaMalloc((void**)&m_cpInfoGpu, sizeof(uint16_t) * cpInfoLen);
    // calculate CP length
    float T_c_over_T_samp = float(cuphyCarrierPrms->f_samp)/float(cuphyCarrierPrms->f_c);
    if(cuphyCarrierPrms -> cpType == 0) // normal CP length
    {
        uint16_t lenCP0 = (((144 >> mu) + 16) << (cuphyCarrierPrms -> kappa_bits))*T_c_over_T_samp; //(144+16)/2048*Nifft;
        uint16_t lenCP1 = ((144 >> mu) << (cuphyCarrierPrms -> kappa_bits))*T_c_over_T_samp; // 144/2048*Nifft;
        
        for(uint8_t symbolIdx = 0; symbolIdx < N_symbol_slot; symbolIdx++)
        {
            m_cpInfoCpu[symbolIdx] = lenCP1;
        }

        if(mu == 0 && symbol0IdxPerSubFrame == 0)
        {
            m_cpInfoCpu[0] = lenCP0;
            m_cpInfoCpu[7] = lenCP0;
        } // check number of OFDM symbols per layer
        else if(mu != 0 && (symbol0IdxPerSubFrame == 0 || symbol0IdxPerSubFrame == (7 << mu)))
        {
            m_cpInfoCpu[0] = lenCP0;
        }
    }
    else
    {
        if(mu != 2)
        {
            printf("Error! Extended CP only applible in numerology 2! \n");
            exit(1);
        }
        uint lenCP1 = ((512 >> mu) << (cuphyCarrierPrms -> kappa_bits))*T_c_over_T_samp;
        for(uint8_t symbolIdx = 0; symbolIdx < N_symbol_slot; symbolIdx++)
        {
            m_cpInfoCpu[symbolIdx] = lenCP1;
        }
    }
    
    // calculate accumulative CP
    m_cpInfoCpu[N_symbol_slot] = m_cpInfoCpu[0];
    for(uint8_t symbolIdx = 1; symbolIdx < N_symbol_slot; symbolIdx++)
    {
        m_cpInfoCpu[symbolIdx + N_symbol_slot] = m_cpInfoCpu[symbolIdx] + m_cpInfoCpu[symbolIdx + N_symbol_slot -1];
    }
    // copy CP info
    m_ofdmModdynDescprCpu -> cpInfo = m_cpInfoGpu;
    cudaMemcpy(m_cpInfoGpu, m_cpInfoCpu, sizeof(uint16_t) * cpInfoLen, cudaMemcpyHostToDevice);

    m_freqDataLenDl = (cuphyCarrierPrms -> N_bsLayer) * (cuphyCarrierPrms -> N_symbol_slot) * (cuphyCarrierPrms -> N_sc);
    m_freqDataLenUl = (cuphyCarrierPrms -> N_ueLayer) * (cuphyCarrierPrms -> N_symbol_slot) * (cuphyCarrierPrms -> N_sc);
    m_ofdmModdynDescprCpu -> freqDataIn = freqDataIn;
    m_timeDataLenDl = ((m_ofdmModdynDescprCpu -> N_IFFT) * (cuphyCarrierPrms -> N_symbol_slot) + m_cpInfoCpu[cpInfoLen - 1]) * (m_ofdmModdynDescprCpu -> N_bsLayer);
    m_timeDataLenUl = ((m_ofdmModdynDescprCpu -> N_IFFT) * (cuphyCarrierPrms -> N_symbol_slot) + m_cpInfoCpu[cpInfoLen - 1]) * (m_ofdmModdynDescprCpu -> N_ueLayer);
    cudaMalloc((void**)&(m_ofdmModdynDescprCpu -> timeDataOut), sizeof(Tcomplex)*max(m_timeDataLenDl, m_timeDataLenUl));

    // calculate the number of samples per OFDM symbol
    m_eachSymbolLenWithCP.resize(N_symbol_slot);
    for(uint8_t symbolIdx = 0; symbolIdx < N_symbol_slot; symbolIdx++)
    {
        m_eachSymbolLenWithCP[symbolIdx] = m_cpInfoCpu[symbolIdx] + m_ofdmModdynDescprCpu -> N_IFFT; 
    }

    /* ----------------  kernel launch config ---------------------- */
    // copy dynamic descriptor to GPU
    cudaMalloc((void**)&m_ofdmModdynDescprGpu, sizeof(ofdmModDynDescr_t<Tscalar, Tcomplex>));
    cudaMemcpy(m_ofdmModdynDescprGpu, m_ofdmModdynDescprCpu, sizeof(ofdmModDynDescr_t<Tscalar, Tcomplex>), cudaMemcpyHostToDevice);

    // OFDM modulation kernel launch config
    m_pOfdmModCfg = new ofdmLaunchCfg_t;

    // set up kernel
    using namespace cufftdx;
    uint shared_memory_size = 0;
    dim3 block_dim;
    const uint cudaDeviceArch = getCudaDeviceArch();
    auto kernelPtr = ofdmMod_get_ifft_param<Tscalar, Tcomplex>( m_ofdmModdynDescprCpu -> N_IFFT, cudaDeviceArch, block_dim, shared_memory_size);
    m_pOfdmModCfg->kernelArgs[0] = &m_ofdmModdynDescprGpu;
    m_pOfdmModCfg->kernelArgs[1] = &m_enableSwapTxRx;

    CUDA_KERNEL_NODE_PARAMS& ofdmModKernelNodeParams = m_pOfdmModCfg->kernelNodeParamsDriver;
    CHECK_CUDAERROR(cudaGetFuncBySymbol(&ofdmModKernelNodeParams.func, reinterpret_cast<void*>(kernelPtr)));
    // ofdmModKernelNodeParams.func = kernelPtr;
    ofdmModKernelNodeParams.blockDimX = block_dim.x;
    ofdmModKernelNodeParams.blockDimY = block_dim.y;
    ofdmModKernelNodeParams.blockDimZ = block_dim.z;

    ofdmModKernelNodeParams.gridDimX = m_ofdmModdynDescprCpu -> N_bsLayer;
    ofdmModKernelNodeParams.gridDimY = cuphyCarrierPrms -> N_symbol_slot / OFDM_FFTs_PER_BLOCK_CONST_;
    ofdmModKernelNodeParams.gridDimZ = 1;
    ofdmModKernelNodeParams.sharedMemBytes = shared_memory_size;
    ofdmModKernelNodeParams.kernelParams = &(m_pOfdmModCfg->kernelArgs[0]);
    ofdmModKernelNodeParams.extra = nullptr;

    /* ----------------  OFDM windowning ---------------------- */
    // calculate window: OFDM Raised Cosine Window
    // NOT USED FOR NOW
    uint ofdmWindowLen = cuphyCarrierPrms -> ofdmWindowLen;
    m_ofdmModdynDescprCpu -> ofdmWindowLen = ofdmWindowLen;
    float rolloffFactor = cuphyCarrierPrms -> rolloffFactor;
    if(ofdmWindowLen > 3)
    {
        m_ofdmWindowCpu = new Tscalar[ofdmWindowLen];
        cudaMalloc((void**)&m_ofdmWindowGpu, sizeof(Tscalar)*ofdmWindowLen);
        m_ofdmWindowCpu[0] = 1;
        m_ofdmWindowCpu[ofdmWindowLen - 1] = 0;
        float step = 1.0f/(ofdmWindowLen-1);
        for(int windowIdx=1; windowIdx < ofdmWindowLen-1; windowIdx++)
        {
            float t_over_T = windowIdx * step;
            m_ofdmWindowCpu[windowIdx] = sin(M_PI * t_over_T)/(M_PI * t_over_T) * cos(M_PI * t_over_T * rolloffFactor) / (1 - 4 * rolloffFactor * rolloffFactor * t_over_T * t_over_T);
        }
        cudaMemcpy(m_ofdmWindowGpu, m_ofdmWindowCpu, sizeof(Tscalar)*ofdmWindowLen, cudaMemcpyHostToDevice);
        m_ofdmModdynDescprCpu -> ofdmWindowCoe = m_ofdmWindowGpu;

        // apply windowing lauch config
        m_pWindowCfg = new ofdmLaunchCfg_t;
        CUDA_KERNEL_NODE_PARAMS& windowKernelNodeParams = m_pWindowCfg->kernelNodeParamsDriver;
        m_pOfdmModCfg->kernelArgs[0] = &m_ofdmModdynDescprGpu;
        CHECK_CUDAERROR(cudaGetFuncBySymbol(&windowKernelNodeParams.func, reinterpret_cast<void*>(applyWindow<Tscalar, Tcomplex>)));
        // windowKernelNodeParams.func = kernelPtr;
        windowKernelNodeParams.blockDimX = m_ofdmModdynDescprCpu -> ofdmWindowLen;
        windowKernelNodeParams.blockDimY = m_ofdmModdynDescprCpu -> N_symbol_slot;
        windowKernelNodeParams.blockDimZ = 1;

        windowKernelNodeParams.gridDimX = m_ofdmModdynDescprCpu -> N_bsLayer;
        windowKernelNodeParams.gridDimY = 1;
        windowKernelNodeParams.gridDimZ = 1;
        windowKernelNodeParams.sharedMemBytes = 0;
        windowKernelNodeParams.kernelParams = &(m_pOfdmModCfg->kernelArgs[0]);
        windowKernelNodeParams.extra = nullptr;
    }
    else
    {
        m_ofdmModdynDescprCpu -> ofdmWindowCoe = nullptr;
        m_pWindowCfg = nullptr;
    }

    // pre load IFFT kernel to avoid first run timing
    run(m_enableSwapTxRx, strm);
}

template <typename Tscalar, typename Tcomplex> 
ofdmModulate<Tscalar, Tcomplex>::~ofdmModulate()
{
    delete[] m_cpInfoCpu;
    cudaFree(m_cpInfoGpu);

    if(m_ofdmModdynDescprCpu -> ofdmWindowLen)
    {
        delete[] m_ofdmWindowCpu;
        cudaFree(m_ofdmWindowGpu);
        delete m_pWindowCfg;
    }
    cudaFree(m_ofdmModdynDescprGpu);
    cudaFree(m_ofdmModdynDescprCpu -> timeDataOut);
    delete m_ofdmModdynDescprCpu;
    delete m_pOfdmModCfg;
}

template <typename Tscalar, typename Tcomplex> 
void ofdmModulate<Tscalar, Tcomplex>::run(uint8_t enableSwapTxRx, cudaStream_t strm)
{
    m_enableSwapTxRx = enableSwapTxRx;
    // launch ofdm modulation kernel
    CUDA_KERNEL_NODE_PARAMS& ofdmModKernelNodeParams = m_pOfdmModCfg->kernelNodeParamsDriver;
    ofdmModKernelNodeParams.gridDimX = m_enableSwapTxRx ? m_ofdmModdynDescprCpu -> N_ueLayer : m_ofdmModdynDescprCpu -> N_bsLayer;
    CUresult runStatus = cuLaunchKernel(ofdmModKernelNodeParams.func,
                                        ofdmModKernelNodeParams.gridDimX,
                                        ofdmModKernelNodeParams.gridDimY, 
                                        ofdmModKernelNodeParams.gridDimZ,
                                        ofdmModKernelNodeParams.blockDimX, 
                                        ofdmModKernelNodeParams.blockDimY, 
                                        ofdmModKernelNodeParams.blockDimZ,
                                        ofdmModKernelNodeParams.sharedMemBytes,
                                        strm,
                                        ofdmModKernelNodeParams.kernelParams,
                                        ofdmModKernelNodeParams.extra);
    CHECK_CURESULT(runStatus);

    /**
     * @todo apply windowing effect, not used for now
     * 
     */
    // launch windowing kernel
    // if(m_pWindowCfg) // not nullptr
    // {
    //     const CUDA_KERNEL_NODE_PARAMS& windowKernelNodeParams = m_pWindowCfg->kernelNodeParamsDriver;
    //     CUresult runStatus = cuLaunchKernel(windowKernelNodeParams.func,
    //                                         windowKernelNodeParams.gridDimX,
    //                                         windowKernelNodeParams.gridDimY, 
    //                                         windowKernelNodeParams.gridDimZ,
    //                                         windowKernelNodeParams.blockDimX, 
    //                                         windowKernelNodeParams.blockDimY, 
    //                                         windowKernelNodeParams.blockDimZ,
    //                                         windowKernelNodeParams.sharedMemBytes,
    //                                         strm,
    //                                         windowKernelNodeParams.kernelParams,
    //                                         windowKernelNodeParams.extra);
    //     CHECK_CURESULT(runStatus);
    // }
}

template <typename Tscalar, typename Tcomplex> 
void ofdmModulate<Tscalar, Tcomplex>::printTimeSample(int printLen)
{
    Tcomplex * temp_CPU_buffer = new Tcomplex[printLen];

    cudaMemcpy(temp_CPU_buffer, m_ofdmModdynDescprCpu -> timeDataOut, printLen * sizeof(Tcomplex), cudaMemcpyDeviceToHost);

    for (int index=0; index< printLen; index++)
    {
        printf("index: %d: %1.4e + %1.4e  i\n", index, float(temp_CPU_buffer[index].x), float(temp_CPU_buffer[index].y));
    }
    printf("Done printing output time domain signal from GPU \n");

    delete[] temp_CPU_buffer;
}

template <typename Tscalar, typename Tcomplex> 
void ofdmModulate<Tscalar, Tcomplex>::saveOfdmModToH5File(cudaStream_t strm)
{
    std::string outFilename = m_enableSwapTxRx ? "ofdm_mod_swap1.h5" : "ofdm_mod_swap0.h5";
    // Initialize HDF5
    hid_t ofdmModH5File = H5Fcreate(outFilename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT); // non empty existing file will be overwritten
    if (ofdmModH5File < 0)
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
        H5Tinsert(complexDataType, "re", HOFFSET(cuComplex, x), H5T_IEEE_F32LE);
        H5Tinsert(complexDataType, "im", HOFFSET(cuComplex, y), H5T_IEEE_F32LE);
    }
    else
    {
        fprintf(stderr, "Unsupported data type!\n");
        exit(EXIT_FAILURE);
    }
       
    // Create a dataset in the HDF5 file to store the data. TOOD: currently all 1D array
    uint8_t rank = 1;
    hsize_t dims[rank] = {0};

    cudaStreamSynchronize(strm); // sync stream to ensure processing done
    
    dims[0] = m_enableSwapTxRx ? m_freqDataLenUl : m_freqDataLenDl;
    writeHdf5DatasetFromGpu<Tcomplex>(ofdmModH5File, "freqDataIn", complexDataType, m_ofdmModdynDescprCpu -> freqDataIn, dims, rank);

    dims[0] = m_enableSwapTxRx ? m_timeDataLenUl : m_timeDataLenDl;
    writeHdf5DatasetFromGpu<Tcomplex>(ofdmModH5File, "timeDataOut", complexDataType, m_ofdmModdynDescprCpu -> timeDataOut, dims, rank);

    // Define the compound datatype for the struct
    hid_t compType = H5Tcreate(H5T_COMPOUND, sizeof(cuphyCarrierPrms_t));
    H5Tinsert(compType, "N_sc", HOFFSET(cuphyCarrierPrms, N_sc), H5T_STD_U16LE);
    H5Tinsert(compType, "N_FFT", HOFFSET(cuphyCarrierPrms, N_FFT), H5T_STD_U16LE);
    H5Tinsert(compType, "N_bsLayer", HOFFSET(cuphyCarrierPrms, N_bsLayer), H5T_STD_U16LE);
    H5Tinsert(compType, "N_ueLayer", HOFFSET(cuphyCarrierPrms, N_ueLayer), H5T_STD_U16LE);
    H5Tinsert(compType, "id_slot", HOFFSET(cuphyCarrierPrms, id_slot), H5T_STD_U16LE);
    H5Tinsert(compType, "id_subFrame", HOFFSET(cuphyCarrierPrms, id_subFrame), H5T_STD_U16LE);
    H5Tinsert(compType, "mu", HOFFSET(cuphyCarrierPrms, mu), H5T_STD_U16LE);
    H5Tinsert(compType, "cpType", HOFFSET(cuphyCarrierPrms, cpType), H5T_STD_U16LE);
    H5Tinsert(compType, "f_c", HOFFSET(cuphyCarrierPrms, f_c), H5T_STD_U32LE);
    H5Tinsert(compType, "T_c", HOFFSET(cuphyCarrierPrms, T_c), H5T_IEEE_F32LE);
    H5Tinsert(compType, "f_samp", HOFFSET(cuphyCarrierPrms, f_samp), H5T_STD_U32LE);
    H5Tinsert(compType, "N_symbol_slot", HOFFSET(cuphyCarrierPrms, N_symbol_slot), H5T_STD_U16LE);
    H5Tinsert(compType, "k_const", HOFFSET(cuphyCarrierPrms, k_const), H5T_STD_U16LE);
    H5Tinsert(compType, "kappa_bits", HOFFSET(cuphyCarrierPrms, kappa_bits), H5T_STD_U16LE);
    H5Tinsert(compType, "ofdmWindowLen", HOFFSET(cuphyCarrierPrms, ofdmWindowLen), H5T_STD_U32LE);
    H5Tinsert(compType, "rolloffFactor", HOFFSET(cuphyCarrierPrms, rolloffFactor), H5T_IEEE_F32LE);
    H5Tinsert(compType, "N_samp_slot", HOFFSET(cuphyCarrierPrms, N_samp_slot), H5T_STD_U32LE);

    // PRACH parameters
    H5Tinsert(compType, "N_u_mu", HOFFSET(cuphyCarrierPrms, N_u_mu), H5T_STD_U32LE);
    H5Tinsert(compType, "startRaSym", HOFFSET(cuphyCarrierPrms, startRaSym), H5T_STD_U32LE);
    H5Tinsert(compType, "delta_f_RA", HOFFSET(cuphyCarrierPrms, delta_f_RA), H5T_STD_U32LE);
    H5Tinsert(compType, "N_CP_RA", HOFFSET(cuphyCarrierPrms, N_CP_RA), H5T_STD_U32LE);
    H5Tinsert(compType, "K", HOFFSET(cuphyCarrierPrms, K), H5T_STD_U32LE);
    H5Tinsert(compType, "k1", HOFFSET(cuphyCarrierPrms, k1), H5T_STD_I32LE);
    H5Tinsert(compType, "kBar", HOFFSET(cuphyCarrierPrms, kBar), H5T_STD_U32LE);
    H5Tinsert(compType, "N_u", HOFFSET(cuphyCarrierPrms, N_u), H5T_STD_U32LE);
    H5Tinsert(compType, "L_RA", HOFFSET(cuphyCarrierPrms, L_RA), H5T_STD_U32LE);
    H5Tinsert(compType, "n_slot_RA_sel", HOFFSET(cuphyCarrierPrms, n_slot_RA_sel), H5T_STD_U32LE);
    H5Tinsert(compType, "N_rep", HOFFSET(cuphyCarrierPrms, N_rep), H5T_STD_U32LE);

    dims[0] = 1;
    hid_t dataspaceId = H5Screate_simple(rank, dims, nullptr);
    hid_t datasetId = H5Dcreate2(ofdmModH5File, "cuphyCarrierPrms", compType, dataspaceId, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    herr_t status = H5Dwrite(datasetId, compType, H5S_ALL, H5S_ALL, H5P_DEFAULT, m_cuphyCarrierPrms);
    H5Dclose(datasetId);
    H5Sclose(dataspaceId);

    // Close HDF5 objects 
    H5Fclose(ofdmModH5File);
}