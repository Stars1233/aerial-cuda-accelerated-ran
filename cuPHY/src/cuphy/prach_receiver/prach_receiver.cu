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

#include "cuphy_internal.h"
#include "cuphy.hpp"
#include "prach_receiver.hpp"
#include <iostream>
#include "tensor_desc.hpp"
#include "type_convert.hpp"

#include <cufftdx.hpp>

#define NUM_THREAD 256

using namespace cuphy_i;

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

/*VCAST_DONT_INSTRUMENT_START*/
template<typename Tcomplex>
__device__ Tcomplex complex_mult2(Tcomplex num1, Tcomplex num2);

template<>
__device__ cuFloatComplex complex_mult2(cuFloatComplex num1, cuFloatComplex num2) {
    return cuCmulf(num1, num2);
};

template<>
__device__ __half2 complex_mult2(__half2 num1, __half2 num2) {
    return __hmul2(num1, num2);
};
/*VCAST_DONT_INSTRUMENT_END*/

/** @brief: Do coherent combining for repetitive preambles samples and calculate the correlation between 
 *          the received (averaged) preamble and local reference preamble.
 */
 template<typename Tcomplex, typename Tscalar>
 __global__ void prach_compute_correlation(const PrachInternalDynParamPerOcca* d_dynParam,
                                            const PrachDeviceInternalStaticParamPerOcca* d_staticParam,
                                            uint max_ant_u,
                                            uint16_t nOccaProc) {

    int batchIndex = blockIdx.x / max_ant_u;
    int ant_u_idx = blockIdx.x - batchIndex * max_ant_u;

    uint16_t occaPrmStaticIdx = d_dynParam[batchIndex].occaPrmStatIdx;

    const PrachDeviceInternalStaticParamPerOcca& staticOccaParam = d_staticParam[occaPrmStaticIdx];
    const PrachParams* prach_params = &(staticOccaParam.prach_params);

    int N_ant = prach_params->N_ant;
    if(staticOccaParam.enableUlRxBf)
    {
        N_ant = d_dynParam[batchIndex].nUplinkStreams;
    }
    const int Nfft = prach_params->Nfft;
    const int uCount = prach_params->uCount;

    if(threadIdx.x >= Nfft || ant_u_idx >= N_ant*uCount)
        return;

    const int L_RA = prach_params->L_RA;
    const int N_rep = prach_params->N_rep;
    const int N_nc = prach_params->N_nc;
    const int kBar = prach_params->kBar;

    Tcomplex * d_fft = (Tcomplex *)(staticOccaParam.prach_workspace_buffer);

    // O-RAN FH sends 144 or 864 samples instead of 139 or 839 samples
    const int L_ORAN = (L_RA == 139) ? 144 : 864;

    if (threadIdx.x < L_RA) {
        const __half2* d_prach_rx = d_dynParam[batchIndex].dataRx;
        const __half2* d_y_u_ref = staticOccaParam.d_y_u_ref;

        int uIdx = ant_u_idx % uCount;
        int antIdx = ant_u_idx / uCount;
        int idx_y_ref = uIdx * L_RA + threadIdx.x;
        Tscalar x = d_y_u_ref[idx_y_ref].x;
        Tscalar y = -d_y_u_ref[idx_y_ref].y;
        Tcomplex y_ref_val = make_complex<Tcomplex>::create(x, y);
        int rep_start = 0;
        int step = 0;
        // for each non-coherent combining group
        for (int idxNc = 0; idxNc < N_nc; idxNc ++) {
            Tcomplex y_rx_val = make_complex<Tcomplex>::create(0.0, 0.0); 
            if (idxNc < N_nc-1) {
                step = N_rep/N_nc;
            }
            else {
                step = N_rep - (N_nc-1)*N_rep/N_nc;
            }
            // average over repetitive preambles
            for (int idxRep = rep_start; idxRep < rep_start + step; idxRep ++) {
                // int idx_y_rx = (antIdx *N_rep + idxRep) * L_RA + threadIdx.x;
                // Need to skip the first kBar guard subcarriers for O-RAN FH samples
                int idx_y_rx = (antIdx *N_rep + idxRep) * L_ORAN + threadIdx.x + kBar;
                y_rx_val.x = ((Tscalar) d_prach_rx[idx_y_rx].x) + y_rx_val.x;
                y_rx_val.y = ((Tscalar) d_prach_rx[idx_y_rx].y) + y_rx_val.y;
            }
            y_rx_val.x = ((Tscalar) y_rx_val.x)/((Tscalar) step);
            y_rx_val.y = ((Tscalar) y_rx_val.y)/((Tscalar) step);
            rep_start = rep_start + step;
            // freq domain multiplication
            Tcomplex z_u  = complex_mult2<Tcomplex>(y_rx_val, y_ref_val);
            int idx_fft = (ant_u_idx*N_nc+idxNc)*Nfft + threadIdx.x;
            d_fft[idx_fft] = z_u;
        }            
    }
    else {
        for (int idxNc = 0; idxNc < N_nc; idxNc ++) {
            int idx_fft = (ant_u_idx*N_nc+idxNc)*Nfft + threadIdx.x;
            // pad zero
            d_fft[idx_fft] = make_complex<Tcomplex>::create(0.0, 0.0);
        }
    }
}

template<class FFT>
__launch_bounds__(FFT::max_threads_per_block)
__global__ void block_fft_kernel(cuFloatComplex** d_fft, uint32_t numFfts) {
    using namespace cufftdx;

    using complex_type = typename FFT::value_type;

    const unsigned int global_fft_idx = blockIdx.x * FFT::ffts_per_block + threadIdx.y;
    if (global_fft_idx >= numFfts) {
        return;
    }
    complex_type* data = (complex_type*)d_fft[global_fft_idx];

    // Local array and copy data into it
    complex_type thread_data[FFT::storage_size];

    const int stride = size_of<FFT>::value / FFT::elements_per_thread;
    static_assert(size_of<FFT>::value % FFT::elements_per_thread == 0, "FFT size must be divisible by elements per thread");

    int index = threadIdx.x;

    for (int i = 0; i < FFT::storage_size; i++) {
        thread_data[i].x = data[index + i * stride].x;
        thread_data[i].y = data[index + i * stride].y;
    }

    extern __shared__ complex_type shared_mem[];

    // Execute FFT
    FFT().execute(thread_data, shared_mem);

    for (int i = 0; i < FFT::storage_size; i++) {
        // Save results
        data[index + i * stride].x = thread_data[i].x;
        data[index + i * stride].y = thread_data[i].y;
    }
}

template<typename Tscalar, unsigned int FftSize, unsigned int Arch>
FftKernelHandle prach_get_fft_param(dim3& block_dim, uint& shared_memory_size, uint32_t& ffts_per_block) {

    using namespace cufftdx;

    // We benchmark to determine the optimal elements per thread for each supported FFT
    // size, so a case must be added if adding newly supported sizes. We could also
    // leave out the ElementsPerThread parameter and use cuFFTDx's heuristics, but
    // as of cufftDx <= 1.6.0, heuristics choose 32 EPT for 1024 FFTs and 16 EPT is ~12% faster.
    static_assert(FftSize == 256 || FftSize == 512 || FftSize == 1024, "Unsupported FFT size");
    constexpr unsigned int EPT = (FftSize == 1024) ? 16U :
                                 (FftSize == 512) ? 8U :
                                 (FftSize == 256) ? 16U :
                                 32U;
    constexpr unsigned int FPB = (FftSize == 256) ? 8U :
                                 (FftSize == 1024) ? 2U :
                                 1U;
    using FFT = decltype(Size<FftSize>() + Precision<Tscalar>() + Type<fft_type::c2c>()
                        + Direction<fft_direction::inverse>() + FFTsPerBlock<FPB>()
                        + ElementsPerThread<EPT>() + SM<Arch>() + Block());

    block_dim = FFT::block_dim;
    shared_memory_size = FFT::shared_memory_size;
    ffts_per_block = FFT::ffts_per_block;
    return block_fft_kernel<FFT>;
 }

 /*VCAST_DONT_INSTRUMENT_START*/
 template<typename Tscalar, unsigned int FftSize>
FftKernelHandle prach_get_fft_param_for_arch(unsigned int cudaDeviceArch, dim3& block_dim, uint& shared_memory_size, uint32_t& ffts_per_block) {
    switch(cudaDeviceArch) {
        // All SM supported by cuFFTDx
        case  700: return prach_get_fft_param<Tscalar, FftSize,  700>(block_dim, shared_memory_size, ffts_per_block);
        case  750: return prach_get_fft_param<Tscalar, FftSize,  750>(block_dim, shared_memory_size, ffts_per_block);
        case  800: return prach_get_fft_param<Tscalar, FftSize,  800>(block_dim, shared_memory_size, ffts_per_block);
        case  860: return prach_get_fft_param<Tscalar, FftSize,  860>(block_dim, shared_memory_size, ffts_per_block);
        case  870: return prach_get_fft_param<Tscalar, FftSize,  870>(block_dim, shared_memory_size, ffts_per_block);
        case  890: return prach_get_fft_param<Tscalar, FftSize,  890>(block_dim, shared_memory_size, ffts_per_block);
        case  900: return prach_get_fft_param<Tscalar, FftSize,  900>(block_dim, shared_memory_size, ffts_per_block);
        case 1000: return prach_get_fft_param<Tscalar, FftSize,  1000>(block_dim, shared_memory_size, ffts_per_block);
        case 1200: return prach_get_fft_param<Tscalar, FftSize,  1200>(block_dim, shared_memory_size, ffts_per_block);
        case 1210: return prach_get_fft_param<Tscalar, FftSize,  1210>(block_dim, shared_memory_size, ffts_per_block);
        default: 
            NVLOGE_FMT(NVLOG_PRACH, AERIAL_CUDA_API_EVENT, "Unsupported CUDA device architecture: {} for FFT size: {}", cudaDeviceArch, FftSize);
            return nullptr;
    }
}
/*VCAST_DONT_INSTRUMENT_END*/

 template<typename Tscalar>
FftKernelHandle prach_get_fft_param(unsigned int Nfft, unsigned int cudaDeviceArch, dim3& block_dim, uint& shared_memory_size, uint32_t& ffts_per_block) {

    if(Nfft == 256) {
        return prach_get_fft_param_for_arch<Tscalar, 256>(cudaDeviceArch, block_dim, shared_memory_size, ffts_per_block);
    }
    else if(Nfft == 1024) {
        return prach_get_fft_param_for_arch<Tscalar, 1024>(cudaDeviceArch, block_dim, shared_memory_size, ffts_per_block);
    }
    else {
        NVLOGE_FMT(NVLOG_PRACH, AERIAL_CUDA_API_EVENT, "Unsupported FFT size: {}", Nfft);
        return nullptr;
    }
 }

 /** @brief: Do non-coherent combining and find the power, peak value and peak location 
 *          for each preamble zone. 
 */
 template<typename Tcomplex, typename Tscalar>
 __global__ void prach_compute_pdp(const PrachInternalDynParamPerOcca* d_dynParam,
                                    const PrachDeviceInternalStaticParamPerOcca* d_staticParam,
                                    int zoneSizeExt) {

    int batchIndex = blockIdx.y;

    uint16_t occaPrmStaticIdx = d_dynParam[batchIndex].occaPrmStatIdx;

    const PrachDeviceInternalStaticParamPerOcca& staticOccaParam = d_staticParam[occaPrmStaticIdx];
    const PrachParams* prach_params = &(staticOccaParam.prach_params);

    int N_ant = prach_params->N_ant;
    const int uCount = prach_params->uCount;
    const int Nfft = prach_params->Nfft;
    const int N_nc = prach_params->N_nc;
    const int fft_elements = N_ant * Nfft * uCount * N_nc;
    
    if(staticOccaParam.enableUlRxBf)
    {
        N_ant = d_dynParam[batchIndex].nUplinkStreams;
    }

    int NzonePerBlock = NUM_THREAD/zoneSizeExt;
    int zoneIdxInBlock = threadIdx.x/zoneSizeExt;
    int global_idxZone = blockIdx.x * NzonePerBlock + zoneIdxInBlock;
    int antIdx = global_idxZone / CUPHY_PRACH_RX_NUM_PREAMBLE;

    if(antIdx >= N_ant)
        return;
       
    Tcomplex * d_fft = (Tcomplex *)(staticOccaParam.prach_workspace_buffer);
    prach_pdp_t<Tscalar> * d_pdp = (prach_pdp_t<Tscalar> * )(d_fft + fft_elements);
 
    const int L_RA = prach_params->L_RA;
    const int N_CS = prach_params->N_CS;

    int zoneSize = (N_CS*Nfft+L_RA-1)/L_RA;
    
    __shared__ Tscalar local_power[NUM_THREAD];
    __shared__ Tscalar local_max[NUM_THREAD];
    __shared__ int local_loc[NUM_THREAD];

    int zoneSearchGap = (int) Nfft/L_RA;    
    int C_v = 0;
    int zone_start = 0;
    int prmbCount = global_idxZone & (CUPHY_PRACH_RX_NUM_PREAMBLE-1); // & => mod
    int NzonePerU = L_RA / N_CS;
    int uIdx = prmbCount / NzonePerU;
    int tIdx = threadIdx.x;
    int idxInZone = tIdx & (zoneSizeExt-1); // & => mod

    // copy abs(dfft)^2 to shared memory for each zone
    if (idxInZone < zoneSize) {
        C_v = (prmbCount % NzonePerU) * N_CS;
        zone_start = (C_v*Nfft+L_RA-1)/L_RA;                       
        zone_start = (Nfft - zone_start) & (Nfft-1);  // & => mod
        // compute abs()^2 and do non-coherent combining
        Tscalar val = 0.0;        
        for (int idxNc = 0; idxNc < N_nc; idxNc ++) {
            int idx_fft = ((antIdx * uCount + uIdx) * N_nc + idxNc) * Nfft + ((zone_start + idxInZone - zoneSearchGap + Nfft) & (Nfft-1)); //& => mod
            Tscalar x = ((Tscalar) d_fft[idx_fft].x)/((Tscalar) L_RA);
            Tscalar y = ((Tscalar) d_fft[idx_fft].y)/((Tscalar) L_RA);
            val = x*x + y*y + val;           
        }
        val = val/Tscalar(N_nc);      

        local_power[tIdx] = val;
        local_max[tIdx] = val;
        local_loc[tIdx] = idxInZone;        
    }
    else {
        local_power[tIdx] = 0;
        local_max[tIdx] = 0;
        local_loc[tIdx] = 0; 
    }
    __syncthreads();

    // compute sum and find max/loc for each zone
    for (unsigned int s=zoneSizeExt/2; s>0; s>>=1) {
        if (idxInZone < s) {
            local_power[tIdx] = local_power[tIdx] + local_power[tIdx + s];
            if (local_max[tIdx] < local_max[tIdx + s]) {
                local_max[tIdx] = local_max[tIdx + s];
                local_loc[tIdx] = local_loc[tIdx + s];
            }
        }
        __syncthreads();
    }

    // copy results from shared memory back to d_pdp
    if (idxInZone == 0) {
        local_power[tIdx] = local_power[tIdx]/((Tscalar)zoneSize);
        int pdp_index = antIdx * CUPHY_PRACH_RX_NUM_PREAMBLE + prmbCount;
        d_pdp[pdp_index].power = local_power[tIdx];
        d_pdp[pdp_index].max = local_max[tIdx];
        d_pdp[pdp_index].loc = local_loc[tIdx] - zoneSearchGap;   
    }
 }


 /** @brief: Estimate noise power and detect preambles based on threshold.
 */
 template<typename Tcomplex, typename Tscalar>
 __global__ void prach_search_pdp(const PrachInternalDynParamPerOcca* d_dynParam,
                                const PrachDeviceInternalStaticParamPerOcca* d_staticParam,
                                uint32_t * __restrict__  num_detectedPrmb_addr_arr,
                                uint32_t * __restrict__  prmbIndex_estimates_addr_arr,
                                float * __restrict__ prmbDelay_estimates_addr_arr,
                                float * __restrict__ prmbPower_estimates_addr_arr,
                                float * __restrict__ interference_addr_arr) {
    uint batchIndex = blockIdx.x;

    uint16_t occaPrmStaticIdx = d_dynParam[batchIndex].occaPrmStatIdx;
    uint16_t occaPrmDynIdx = d_dynParam[batchIndex].occaPrmDynIdx;

    const PrachDeviceInternalStaticParamPerOcca& staticOccaParam = d_staticParam[occaPrmStaticIdx];
    const PrachParams* prach_params = &(staticOccaParam.prach_params);

    int N_ant = prach_params->N_ant;
    const int uCount = prach_params->uCount;
    const int Nfft = prach_params->Nfft;
    const int N_nc = prach_params->N_nc;
    const int pdp_elements = N_ant * CUPHY_PRACH_RX_NUM_PREAMBLE;
    const int fft_elements = N_ant * Nfft * uCount * N_nc;
    
    if(staticOccaParam.enableUlRxBf)
    {
        N_ant = d_dynParam[batchIndex].nUplinkStreams;
    }
       
    Tcomplex * d_fft = (Tcomplex *)(staticOccaParam.prach_workspace_buffer);
    prach_pdp_t<Tscalar> * d_pdp = (prach_pdp_t<Tscalar> * )(d_fft + fft_elements);
    prach_det_t<Tscalar> * d_det = (prach_det_t<Tscalar> *)(d_pdp + pdp_elements);
    float* d_ant_rssi = (float*)(d_det + 1); 
    float* d_rssiLin = d_ant_rssi + N_ant;

    uint32_t* num_detectedPrmb_addr = num_detectedPrmb_addr_arr + occaPrmDynIdx;
    uint32_t* prmbIndex_estimates_addr = prmbIndex_estimates_addr_arr + occaPrmDynIdx * CUPHY_PRACH_RX_NUM_PREAMBLE;
    float* prmbDelay_estimates_addr = prmbDelay_estimates_addr_arr + occaPrmDynIdx * CUPHY_PRACH_RX_NUM_PREAMBLE;
    float* prmbPower_estimates_addr = prmbPower_estimates_addr_arr + occaPrmDynIdx * CUPHY_PRACH_RX_NUM_PREAMBLE;

    float* interference_addr = interference_addr_arr + occaPrmDynIdx;
      
    const int delta_f_RA = prach_params->delta_f_RA;
    int detIdx = 0, prmbIdx = threadIdx.x;

    Tscalar thr0 = d_dynParam[batchIndex].thr0;

    __shared__ Tscalar np1, thr1, np2, thr2;
    __shared__ int shared_detIdx;
    __shared__ struct  local_pdp_t
                        {
                            Tscalar power;
                            int cnt;
                        } local_pdp [CUPHY_PRACH_RX_NUM_PREAMBLE];

    if (prmbIdx == 0) {
        shared_detIdx = 0;
    }

    // average pdp over antennas
    if (threadIdx.x < CUPHY_PRACH_RX_NUM_PREAMBLE) {        
        int maxLoc = d_pdp[prmbIdx].loc;
        Tscalar maxVal = d_pdp[prmbIdx].max;
        for (int antIdx = 1; antIdx < N_ant; antIdx ++) {
            int global_index = antIdx * CUPHY_PRACH_RX_NUM_PREAMBLE + prmbIdx;
            d_pdp[prmbIdx].power = d_pdp[prmbIdx].power + d_pdp[global_index].power;
            d_pdp[prmbIdx].max = d_pdp[prmbIdx].max + d_pdp[global_index].max;
            if (((Tscalar) d_pdp[global_index].max) > maxVal) {
                maxVal = d_pdp[global_index].max;
                maxLoc = d_pdp[global_index].loc;
            }                                 
        }
        d_pdp[prmbIdx].power = d_pdp[prmbIdx].power/((Tscalar) N_ant);
        d_pdp[prmbIdx].max = d_pdp[prmbIdx].max/((Tscalar) N_ant);
        d_pdp[prmbIdx].loc = maxLoc;            
        local_pdp[prmbIdx].power = d_pdp[prmbIdx].power; 
        local_pdp[prmbIdx].cnt = 1;        
    }
    __syncthreads();

    // calculate the sum of power over all preamble indices
    for (unsigned int s=CUPHY_PRACH_RX_NUM_PREAMBLE/2; s>0; s>>=1) {
        // Overrunning array 
        if (prmbIdx < s && (prmbIdx + s) < CUPHY_PRACH_RX_NUM_PREAMBLE) {
            local_pdp[prmbIdx].power = local_pdp[prmbIdx].power + local_pdp[prmbIdx + s].power;
            local_pdp[prmbIdx].cnt = local_pdp[prmbIdx].cnt + local_pdp[prmbIdx + s].cnt;
        }
        __syncthreads();
    }

    // calculate the average power and update threshold "thr1"
    if (prmbIdx == 0) {
        np1 = ((Tscalar) local_pdp[0].power)/((Tscalar) local_pdp[0].cnt);
        thr1 = thr0*np1;
    }    
    __syncthreads();

    // find the preamble indices with peak < thr1 (as noise)and record their power
    if (((Tscalar) d_pdp[prmbIdx].max) < thr1) {
        local_pdp[prmbIdx].power = d_pdp[prmbIdx].power;
        local_pdp[prmbIdx].cnt = 1; 
    }
    else
    {
        local_pdp[prmbIdx].power = 0;
        local_pdp[prmbIdx].cnt = 0;
    }   
    __syncthreads();

    // calculate sum of noise power
    for (unsigned int s=CUPHY_PRACH_RX_NUM_PREAMBLE/2; s>0; s>>=1) {
        if (prmbIdx < s) {
            local_pdp[prmbIdx].power = local_pdp[prmbIdx].power + local_pdp[prmbIdx + s].power;
            local_pdp[prmbIdx].cnt = local_pdp[prmbIdx].cnt + local_pdp[prmbIdx + s].cnt;
        }
        __syncthreads();
    }

    // calculate the average noise power and update threshold thr2
    if (prmbIdx == 0) {
        np2 = ((Tscalar) local_pdp[0].power)/((Tscalar)local_pdp[0].cnt);
        if(np2 == (Tscalar)0 || np1 == (Tscalar)0)
        {
            *interference_addr = -100;
        }
        else
        {
            *interference_addr = 10*log10((float)np2);
        }

        Tscalar thr2_min = (*d_rssiLin >= 1 ?  1 : *d_rssiLin) * 1e-2;
        if (((Tscalar) np2*thr0) > thr2_min) 
            thr2 = np2*thr0;
        else
            thr2 = thr2_min;
    }
    __syncthreads();
  
    // find the preamble indices with peak > thr2  (as detected)
    if (((Tscalar) d_pdp[prmbIdx].max) > ((Tscalar) thr2)) {       
        detIdx = atomicAdd(&shared_detIdx, 1);        
        //TBD: may change d_det from struct of array to array of struct to make memory access faster
        d_det->power[detIdx] = d_pdp[prmbIdx].max;
        d_det->prmbIdx[detIdx] = prmbIdx;
        d_det->loc[detIdx] = (d_pdp[prmbIdx].loc > 0)?d_pdp[prmbIdx].loc:0;                                       
    }    
    __syncthreads();

    // pass the dection results 
    if (prmbIdx == 0) {                              
        d_det->Ndet = shared_detIdx;
        * num_detectedPrmb_addr = d_det->Ndet;        
        for (int i = 0; i < d_det->Ndet; i++) {
            prmbIndex_estimates_addr[i] = (uint32_t) d_det->prmbIdx[i];
            prmbDelay_estimates_addr[i] = ((Tscalar) d_det->loc[i])/(((Tscalar) Nfft)*((Tscalar) delta_f_RA));  
            prmbPower_estimates_addr[i] = (Tscalar) d_det->power[i];    
        }
    }   
    __syncthreads();
 }

/** @brief: Compute average power for each antenna and average power over all antennas
 */
 template<typename Tcomplex, typename Tscalar>
 __global__ void prach_compute_rssi(const PrachInternalDynParamPerOcca* d_dynParam,
                                    const PrachDeviceInternalStaticParamPerOcca* d_staticParam,
                                    float* __restrict__ d_rssiDbArray,
                                    uint max_l_oran_ant)
 {
    __shared__ bool isLastBlockDone;

    int gid = threadIdx.x + blockIdx.x * blockDim.x;
    uint batchIndex = blockIdx.y;

    uint16_t occaPrmStaticIdx = d_dynParam[batchIndex].occaPrmStatIdx;
    uint16_t occaPrmDynIdx = d_dynParam[batchIndex].occaPrmDynIdx;

    const PrachDeviceInternalStaticParamPerOcca& staticOccaParam = d_staticParam[occaPrmStaticIdx];
    const PrachParams* prach_params = &(staticOccaParam.prach_params);

    const int N_rep = prach_params->N_rep;
    int N_ant = prach_params->N_ant;
    const int uCount = prach_params->uCount;
    const int Nfft = prach_params->Nfft;
    const int N_nc = prach_params->N_nc;
    const int pdp_elements = N_ant * CUPHY_PRACH_RX_NUM_PREAMBLE;
    const int fft_elements = N_ant * Nfft * uCount * N_nc;
    
    if(staticOccaParam.enableUlRxBf)
    {
        N_ant = d_dynParam[batchIndex].nUplinkStreams;
    }
    const int L_RA = prach_params->L_RA;

    // O-RAN FH sends 144 or 864 samples instead of 139 or 839 samples
    const int L_ORAN = ((L_RA == 139) ? 144 : 864) * N_rep;

    // align L_ORAN so that same warp doesn't have samples for two different antennas
    // this allows us to use shuffle reduction
    unsigned int align_l_oran = ((L_ORAN + 31) >> 5) << 5;
    
    Tcomplex * d_fft = (Tcomplex *)(staticOccaParam.prach_workspace_buffer);
    prach_pdp_t<Tscalar> * d_pdp = (prach_pdp_t<Tscalar> * )(d_fft + fft_elements);
    prach_det_t<Tscalar> * d_det = (prach_det_t<Tscalar> *)(d_pdp + pdp_elements);
    float* d_ant_rssi = (float*)(d_det + 1); 
    float* d_rssiLin = d_ant_rssi + N_ant;
    unsigned int* d_count = (unsigned int*)(d_rssiLin + 1);

    const __half2* d_prach_rx = d_dynParam[batchIndex].dataRx;
    float* d_rssiDb = d_rssiDbArray + occaPrmDynIdx;
    
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    int threadId = cta.thread_rank();

    if (threadId == 0) 
    {
        isLastBlockDone = false;
    }

    cta.sync();

    // Handle to tile in thread block
    cg::thread_block_tile<32> tile = cg::tiled_partition<32>(cta);

    int antIdx = gid / align_l_oran;
    int idxOran = gid -  antIdx * align_l_oran;

    float absRx = 0.0f;

    if(idxOran < L_ORAN && antIdx < N_ant)
    {
        __half2 rx = d_prach_rx[antIdx * L_ORAN + idxOran];
        absRx = rx.x * rx.x + rx.y * rx.y;
    }

    // shuffle redux
    absRx = cg::reduce(tile, absRx, cg::plus<float>());
    cg::sync(tile);
    if (tile.thread_rank() == 0 && antIdx < N_ant) 
    {
      atomicAdd(&d_ant_rssi[antIdx], absRx);
    }

    if (threadId == 0)
    {
        // make sure d_ant_rssi values are updated before we modify d_count
        __threadfence();

        unsigned int value = atomicInc(d_count, gridDim.x);
        isLastBlockDone = (value == (gridDim.x - 1));
    }

    // make sure each thread reads correct value of isLastBlockDone
    cta.sync();

    // number of antennas expected to be <= 32
    assert(N_ant <= warpSize);

    if (isLastBlockDone) 
    {
        // take average of d_ant_rssi values
        if(threadId < warpSize)
        {
            absRx = 0.0f;
            if(threadId < N_ant)
            {
                absRx = d_ant_rssi[threadId];
                if(absRx == 0)
                {
                    d_ant_rssi[threadId] = -100;
                }
                else
                {
                    d_ant_rssi[threadId] = 10*log10(absRx/L_ORAN);
                }
            }

            // shuffle redux over all antenna power
            absRx = cg::reduce(tile, absRx, cg::plus<float>());
            cg::sync(tile);

            if (threadId == 0) 
            {
                if(absRx == 0)
                {
                    d_rssiDb[0] = -100;
                }
                else
                {
                    absRx = absRx/(L_ORAN * N_ant);
                    d_rssiDb[0] = 10*log10(absRx);
                }

                *d_rssiLin = absRx;
            }
        }
    }
 }

 template<typename Tcomplex, typename Tscalar>
 __global__ void memsetRssi(const PrachInternalDynParamPerOcca* d_dynParam,
                            const PrachDeviceInternalStaticParamPerOcca* d_staticParam,
                            uint16_t maxAntenna,
                            uint16_t nOccaProc)
 {
     int index = threadIdx.x + blockIdx.x * blockDim.x;
     int batchIndex = index / (maxAntenna + 2);
     if(batchIndex >= nOccaProc)
        return;

    uint16_t occaPrmStaticIdx = d_dynParam[batchIndex].occaPrmStatIdx;

    const PrachDeviceInternalStaticParamPerOcca& staticOccaParam = d_staticParam[occaPrmStaticIdx];
    const PrachParams* prach_params = &(staticOccaParam.prach_params);

    const int N_ant = prach_params->N_ant;

    int antIndex = index - batchIndex * (maxAntenna + 2);

    if(antIndex >= N_ant + 2)
        return;

    const int uCount = prach_params->uCount;
    const int Nfft = prach_params->Nfft;
    const int N_nc = prach_params->N_nc;
    const int pdp_elements = N_ant * CUPHY_PRACH_RX_NUM_PREAMBLE;
    const int fft_elements = N_ant * Nfft * uCount * N_nc;
    
    Tcomplex * d_fft = (Tcomplex *)(staticOccaParam.prach_workspace_buffer);
    prach_pdp_t<Tscalar> * d_pdp = (prach_pdp_t<Tscalar> * )(d_fft + fft_elements);
    prach_det_t<Tscalar> * d_det = (prach_det_t<Tscalar> *)(d_pdp + pdp_elements);
    uint* d_ant_rssi = (uint*)(d_det + 1); 

    d_ant_rssi[antIndex] = 0;
 }

  template<typename Tcomplex, typename Tscalar>
 __global__ void memcpyRssi(float* ant_rssi_addr, 
                            const PrachInternalDynParamPerOcca* d_dynParam,
                            const PrachDeviceInternalStaticParamPerOcca* d_staticParam,
                            uint16_t maxAntenna,
                            uint16_t nOccaProc)
 {
     int index = threadIdx.x + blockIdx.x * blockDim.x;
     int batchIndex = index / maxAntenna;
     if(batchIndex >= nOccaProc)
        return;

    uint16_t occaPrmStaticIdx = d_dynParam[batchIndex].occaPrmStatIdx;

    const PrachDeviceInternalStaticParamPerOcca& staticOccaParam = d_staticParam[occaPrmStaticIdx];
    const PrachParams* prach_params = &(staticOccaParam.prach_params);

    int N_ant = prach_params->N_ant;

    int antIndex = index - batchIndex * maxAntenna;

    if(antIndex >= N_ant)
        return;

    const int uCount = prach_params->uCount;
    const int Nfft = prach_params->Nfft;
    const int N_nc = prach_params->N_nc;
    const int pdp_elements = N_ant * CUPHY_PRACH_RX_NUM_PREAMBLE;
    const int fft_elements = N_ant * Nfft * uCount * N_nc;
    
    Tcomplex * d_fft = (Tcomplex *)(staticOccaParam.prach_workspace_buffer);
    prach_pdp_t<Tscalar> * d_pdp = (prach_pdp_t<Tscalar> * )(d_fft + fft_elements);
    prach_det_t<Tscalar> * d_det = (prach_det_t<Tscalar> *)(d_pdp + pdp_elements);
    float* d_ant_rssi = (float*)(d_det + 1); 

    uint16_t occaPrmDynIdx = d_dynParam[batchIndex].occaPrmDynIdx;
    float* pinned_ant_rssi = ant_rssi_addr + occaPrmDynIdx * MAX_N_ANTENNAS_SUPPORTED;

    pinned_ant_rssi[antIndex] = d_ant_rssi[antIndex];
  }

 template<typename Tscalar>
 static cuphyStatus_t init_fftinfo_array(
     std::array<FftInfo, PRACH_NUM_SUPPORTED_FFT_SIZES> &fftInfo, cuFloatComplex **h_fftPointers, uint16_t nOccaProc, uint cudaDeviceArch)
 {
     constexpr uint32_t maxFftsPerOccasion = MAX_N_ANTENNAS_SUPPORTED * PRACH_MAX_NUM_PREAMBLES * PRACH_NUM_NON_COHERENT_COMBINING_GROUPS;
     for (uint32_t i = 0; i < PRACH_NUM_SUPPORTED_FFT_SIZES; ++i) {
         cuFloatComplex **ptr = h_fftPointers + i * nOccaProc * maxFftsPerOccasion;
         fftInfo[i].data = ptr;
         FftKernelHandle kernelPtr = prach_get_fft_param<Tscalar>(
             PRACH_SUPPORTED_FFT_SIZES[i], cudaDeviceArch, fftInfo[i].block_dim,
             fftInfo[i].shared_memory_size, fftInfo[i].ffts_per_block);
         if (kernelPtr == nullptr) {
             return CUPHY_STATUS_ARCH_MISMATCH;
         }
         fftInfo[i].kernelPtr = kernelPtr;
         fftInfo[i].numFfts = 0;
     }
     return CUPHY_STATUS_SUCCESS;
 }

 template<typename Tcomplex>
 static cuphyStatus_t refresh_fftinfo_array(
     std::array<FftInfo, PRACH_NUM_SUPPORTED_FFT_SIZES> &fftInfo,
     const PrachInternalDynParamPerOcca* h_dynParam,
     const PrachInternalStaticParamPerOcca* h_staticParam,
     uint16_t nOccaProc,
     const std::vector<char>& activeOccasions = std::vector<char>())
 {
     // Reset FFT counters
     for (uint32_t j = 0; j < PRACH_NUM_SUPPORTED_FFT_SIZES; ++j) {
         fftInfo[j].numFfts = 0;
     }

     const int maxFftsPerOccasion = MAX_N_ANTENNAS_SUPPORTED * PRACH_MAX_NUM_PREAMBLES * PRACH_NUM_NON_COHERENT_COMBINING_GROUPS;
     const int maxFftsPerFftNode = maxFftsPerOccasion * nOccaProc;
     for(int i = 0; i < nOccaProc; ++i)
     {
         uint16_t occaPrmStaticIdx = h_dynParam[i].occaPrmStatIdx;
         if (activeOccasions.size() > 0 && !activeOccasions[occaPrmStaticIdx]) {
             continue;
         }
         const PrachInternalStaticParamPerOcca& staticOccaParam = h_staticParam[occaPrmStaticIdx];
         Tcomplex * d_fft = (Tcomplex *)(staticOccaParam.prach_workspace_buffer.addr());
         const PrachParams& prach_params = staticOccaParam.prach_params;
         bool foundValidFftSize = false;
         for (uint32_t j = 0; j < PRACH_NUM_SUPPORTED_FFT_SIZES; ++j) {
             if (prach_params.Nfft != PRACH_SUPPORTED_FFT_SIZES[j]) {
                 continue;
             }
             foundValidFftSize = true;
             const int numFfts = prach_params.N_ant * prach_params.uCount * prach_params.N_nc;
             FftInfo& info = fftInfo[j];
             if(info.numFfts + numFfts >= maxFftsPerFftNode) {
                 NVLOGE_FMT(NVLOG_PRACH, AERIAL_CUDA_API_EVENT, "CUPHY_ERROR: {} ({}) Max number of FFTs per FFT node exceeded for FFT size: {}", __FILE__,
                        __LINE__, prach_params.Nfft);
                 return CUPHY_STATUS_INVALID_ARGUMENT;
             }
             for (int k = 0; k < numFfts; ++k) {
                 info.data[info.numFfts + k] = reinterpret_cast<cuFloatComplex *>(
                     d_fft + k * prach_params.Nfft);
             }
             info.numFfts += numFfts;
             break;
         }
         if (!foundValidFftSize) {
             NVLOGE_FMT(NVLOG_PRACH, AERIAL_CUDA_API_EVENT, "CUPHY_ERROR: {} ({}) No valid FFT size found for occasion {}, FFT size: {}", __FILE__,
                         __LINE__, i, prach_params.Nfft);
             return CUPHY_STATUS_INVALID_ARGUMENT;
         }
     }
     return CUPHY_STATUS_SUCCESS;
 }

template<typename Tcomplex, typename Tscalar>
cuphyStatus_t launch_templated_prach_receiver_kernels(const PrachInternalDynParamPerOcca* d_dynParam,
                                            const PrachDeviceInternalStaticParamPerOcca* d_staticParam,
                                            const PrachInternalDynParamPerOcca* h_dynParam,
                                            const PrachInternalStaticParamPerOcca* h_staticParam,
                                            cuFloatComplex **h_fftPointers,
                                            std::array<FftInfo, PRACH_NUM_SUPPORTED_FFT_SIZES> &fftInfo,
                                            uint32_t* num_detectedPrmb_addr,
                                            uint32_t* prmbIndex_estimates_addr,
                                            float* prmbDelay_estimates_addr,
                                            float* prmbPower_estimates_addr,
                                            float* ant_rssi_addr,
                                            float* rssi_addr,
                                            float* interference_addr,
                                            uint16_t nOccaProc,
                                            uint16_t maxAntenna,
                                            uint max_l_oran_ant,
                                            uint max_ant_u,
                                            uint max_nfft,
                                            int max_zoneSizeExt,
                                            uint cudaDeviceArch,
                                            cudaStream_t strm) {

    // intialize d_ant_rssi, d_rssiLin, d_count with 0
    dim3 block_dim(128, 1, 1);
    dim3 grid_dim((nOccaProc * (maxAntenna + 2) + block_dim.x - 1) / block_dim.x);

    memsetRssi<Tcomplex, Tscalar><<<grid_dim, block_dim, 0, strm>>>(d_dynParam, d_staticParam, maxAntenna, nOccaProc);

    grid_dim = dim3((max_l_oran_ant + block_dim.x - 1) / block_dim.x, nOccaProc);
    prach_compute_rssi<Tcomplex, Tscalar><<<grid_dim, block_dim, 0, strm>>>(d_dynParam, d_staticParam, rssi_addr, max_l_oran_ant);

    grid_dim = dim3((nOccaProc * maxAntenna + block_dim.x - 1) / block_dim.x);
    memcpyRssi<Tcomplex, Tscalar><<<grid_dim, block_dim, 0, strm>>>(ant_rssi_addr, d_dynParam, d_staticParam, maxAntenna, nOccaProc);

    block_dim = dim3(max_nfft);
    grid_dim = dim3(max_ant_u * nOccaProc);
    prach_compute_correlation<Tcomplex, Tscalar><<<grid_dim, block_dim, 0, strm>>>(d_dynParam, d_staticParam, max_ant_u, nOccaProc);

    // Check for any launch failures in the initial kernels
    {
        cudaError_t cuda_error = cudaGetLastError();
        if (cudaSuccess != cuda_error) {
            NVLOGE_FMT(NVLOG_PRACH, AERIAL_CUDA_API_EVENT, "CUDA Error: {}", cudaGetErrorString(cuda_error));
            return CUPHY_STATUS_INTERNAL_ERROR;
        }
    }

    cuphyStatus_t status = init_fftinfo_array<Tscalar>(fftInfo, h_fftPointers, nOccaProc, cudaDeviceArch);
    if (status != CUPHY_STATUS_SUCCESS) {
        NVLOGE_FMT(NVLOG_PRACH, AERIAL_CUDA_API_EVENT, "init_fftinfo_array failed: {}", (int) status);
        return status;
    }
    status = refresh_fftinfo_array<Tcomplex>(fftInfo, h_dynParam, h_staticParam, nOccaProc);
    if (status != CUPHY_STATUS_SUCCESS) {
        NVLOGE_FMT(NVLOG_PRACH, AERIAL_CUDA_API_EVENT, "refresh_fftinfo_array failed: {}", (int) status);
        return status;
    }

    for (const auto &info : fftInfo) {
        if (info.numFfts == 0) {
            continue;
        }
        void *kernelArgs[2] = {(void *)&info.data, (void *)&info.numFfts};
        dim3 grid_dim((info.numFfts + info.ffts_per_block - 1) / info.ffts_per_block);

        cudaError_t error = cudaLaunchKernel(reinterpret_cast<void*>(info.kernelPtr), grid_dim, info.block_dim,
                                                (void**)(kernelArgs), info.shared_memory_size, strm);
        if (cudaSuccess != error) {
            NVLOGE_FMT(NVLOG_PRACH, AERIAL_CUDA_API_EVENT, "CUDA Error: {}", cudaGetErrorString(error));
            return CUPHY_STATUS_INTERNAL_ERROR;
        }
    }

    assert(max_zoneSizeExt <= 512);
    block_dim = dim3(max_zoneSizeExt > NUM_THREAD ? max_zoneSizeExt : NUM_THREAD);
    int Nzone = block_dim.x/max_zoneSizeExt;
    grid_dim = dim3((maxAntenna * CUPHY_PRACH_RX_NUM_PREAMBLE + Nzone-1) / Nzone, nOccaProc);
    prach_compute_pdp<Tcomplex, Tscalar><<<grid_dim, block_dim, 0, strm>>>(d_dynParam, d_staticParam, max_zoneSizeExt);

    dim3 search_pdp_block_dim(CUPHY_PRACH_RX_NUM_PREAMBLE);
    dim3 search_pdp_grid_dim(nOccaProc);
    prach_search_pdp<Tcomplex, Tscalar><<<search_pdp_grid_dim, search_pdp_block_dim, 0, strm>>>(d_dynParam, d_staticParam,
                                                    num_detectedPrmb_addr,
                                                    prmbIndex_estimates_addr,
                                                    prmbDelay_estimates_addr,
                                                    prmbPower_estimates_addr,
                                                    interference_addr);

    cudaError_t cuda_error = cudaGetLastError();
    if (cudaSuccess != cuda_error) {
        NVLOGE_FMT(NVLOG_PRACH, AERIAL_CUDA_API_EVENT, "CUDA Error: {}", cudaGetErrorString(cuda_error));
        return CUPHY_STATUS_INTERNAL_ERROR;
    }

    return CUPHY_STATUS_SUCCESS;
}

cuphyStatus_t cuphyPrachCreateGraph(cudaGraph_t* graph, cudaGraphExec_t* graphInstance, std::vector<cudaGraphNode_t>& nodes,  cudaStream_t strm,
                                    const PrachInternalDynParamPerOcca* d_dynParam,
                                    const PrachDeviceInternalStaticParamPerOcca* d_staticParam,
                                    const PrachInternalStaticParamPerOcca* h_staticParam,
                                    cuFloatComplex **h_fftPointers,
                                    std::array<FftInfo, PRACH_NUM_SUPPORTED_FFT_SIZES> &fftInfo,
                                    uint32_t* num_detectedPrmb_addr,
                                    uint32_t* prmbIndex_estimates_addr,
                                    float* prmbDelay_estimates_addr,
                                    float* prmbPower_estimates_addr,
                                    float* ant_rssi_addr,
                                    float* rssi_addr,
                                    float* interference_addr,
                                    uint16_t nTotCellOcca,
                                    uint16_t nMaxOccasions,
                                    uint16_t maxAntenna,
                                    uint max_l_oran_ant,
                                    uint max_ant_u,
                                    uint max_nfft,
                                    int max_zoneSizeExt,
                                    std::vector<char>& activeOccasions,
                                    uint cudaDeviceArch)
{
    using Tcomplex = cuFloatComplex;
    using Tscalar = float;

    if (!max_zoneSizeExt) {
        NVLOGE_FMT(NVLOG_PRACH, AERIAL_CUDA_API_EVENT, "CUPHY_ERROR: {} ({}) max_zoneSizeExt is zero", __FILE__,
                   __LINE__);
        throw cuphy::cuphy_exception(CUPHY_STATUS_INVALID_ARGUMENT);
    }

    CUDA_CHECK_EXCEPTION(cudaGraphCreate(graph, 0));

    {
        dim3 block_dim(128, 1, 1);
        cudaKernelNodeParams kernelNodeParams = {0};
        // Update: nMaxOccasions -> nOccaProc
        void *kernelArgs[4] = {(void *)&d_dynParam, (void *)&d_staticParam, &maxAntenna,
                            &nTotCellOcca};

        kernelNodeParams.func = (void *)memsetRssi<Tcomplex, Tscalar>;
        // Update: nMaxOccasions -> nOccaProc
        kernelNodeParams.gridDim = dim3((nTotCellOcca * (maxAntenna + 2) + block_dim.x - 1) / block_dim.x, 1, 1);
        kernelNodeParams.blockDim = block_dim;
        kernelNodeParams.sharedMemBytes = 0;
        kernelNodeParams.kernelParams = (void **)kernelArgs;
        kernelNodeParams.extra = NULL;

        CUDA_CHECK_EXCEPTION(
        cudaGraphAddKernelNode(&nodes[GraphNodeType::MemsetRSSI], *graph, nullptr,
                                0, &kernelNodeParams));
    }

    {
        dim3 block_dim(128, 1, 1);
        cudaKernelNodeParams kernelNodeParams = {0};
        void *kernelArgs[4] = {(void *)&d_dynParam, (void *)&d_staticParam, &rssi_addr, &max_l_oran_ant};

        kernelNodeParams.func = (void *)prach_compute_rssi<Tcomplex, Tscalar>;
        // Update: nMaxOccasions -> nOccaProc
        kernelNodeParams.gridDim = dim3((max_l_oran_ant + block_dim.x - 1) / block_dim.x, nTotCellOcca, 1);
        kernelNodeParams.blockDim = block_dim;
        kernelNodeParams.sharedMemBytes = 0;
        kernelNodeParams.kernelParams = (void **)kernelArgs;
        kernelNodeParams.extra = NULL;

        CUDA_CHECK_EXCEPTION(
        cudaGraphAddKernelNode(&nodes[GraphNodeType::ComputeRSSI], *graph, &nodes[GraphNodeType::MemsetRSSI],
                                1, &kernelNodeParams));
    }

    {
        dim3 block_dim(128, 1, 1);
        cudaKernelNodeParams kernelNodeParams = {0};
        // Update: nMaxOccasions -> nOccaProc
        void *kernelArgs[5] = {(void *)&ant_rssi_addr, (void *)&d_dynParam, &d_staticParam, &maxAntenna, &nMaxOccasions};

        kernelNodeParams.func = (void *)memcpyRssi<Tcomplex, Tscalar>;
        // Update: nMaxOccasions -> nOccaProc
        kernelNodeParams.gridDim = dim3((nMaxOccasions * maxAntenna + block_dim.x - 1) / block_dim.x, 1, 1);
        kernelNodeParams.blockDim = block_dim;
        kernelNodeParams.sharedMemBytes = 0;
        kernelNodeParams.kernelParams = (void **)kernelArgs;
        kernelNodeParams.extra = NULL;

        CUDA_CHECK_EXCEPTION(
        cudaGraphAddKernelNode(&nodes[GraphNodeType::MemcpyRSSI], *graph, &nodes[GraphNodeType::ComputeRSSI],
                                1, &kernelNodeParams));
    }

    {
        cudaKernelNodeParams kernelNodeParams = {0};
        // Update: nMaxOccasions -> nOccaProc
        void *kernelArgs[4] = {(void *)&d_dynParam, &d_staticParam, &max_ant_u, &nMaxOccasions};

        kernelNodeParams.func = (void *)prach_compute_correlation<Tcomplex, Tscalar>;
        // Update: nMaxOccasions -> nOccaProc
        kernelNodeParams.gridDim = dim3(max_ant_u * nMaxOccasions);
        kernelNodeParams.blockDim = dim3(max_nfft);
        kernelNodeParams.sharedMemBytes = 0;
        kernelNodeParams.kernelParams = (void **)kernelArgs;
        kernelNodeParams.extra = NULL;

        CUDA_CHECK_EXCEPTION(
        cudaGraphAddKernelNode(&nodes[GraphNodeType::ComputeCorrelationNode], *graph, nullptr,
                                0, &kernelNodeParams));
    }

    cuphyStatus_t status = init_fftinfo_array<Tscalar>(fftInfo, h_fftPointers, nMaxOccasions, cudaDeviceArch);
    if (status != CUPHY_STATUS_SUCCESS) {
        NVLOGE_FMT(NVLOG_PRACH, AERIAL_CUDA_API_EVENT, "init_fftinfo_array failed: {}", (int) status);
        return status;
    }

    const int maxFftsPerOccasion = MAX_N_ANTENNAS_SUPPORTED * PRACH_MAX_NUM_PREAMBLES * PRACH_NUM_NON_COHERENT_COMBINING_GROUPS;
    const int maxFftsPerFftNode = maxFftsPerOccasion * nMaxOccasions;
    for(int i = 0; i < nMaxOccasions; ++i)
    {
        if(activeOccasions[i] != 0) // Only add nodes for configured occasions
        {
            const PrachInternalStaticParamPerOcca& staticOccaParam = h_staticParam[i];
            Tcomplex * d_fft = (Tcomplex *)(staticOccaParam.prach_workspace_buffer.addr());
            const PrachParams& prach_params = staticOccaParam.prach_params;
            bool foundValidFftSize = false;
            for (int j = 0; j < PRACH_NUM_SUPPORTED_FFT_SIZES; ++j) {
                if (prach_params.Nfft != PRACH_SUPPORTED_FFT_SIZES[j]) {
                    continue;
                }
                foundValidFftSize = true;
                const int numFfts = prach_params.N_ant * prach_params.uCount * prach_params.N_nc;
                FftInfo& info = fftInfo[j];
                if(info.numFfts + numFfts >= maxFftsPerFftNode) {
                    NVLOGE_FMT(NVLOG_PRACH, AERIAL_CUDA_API_EVENT, "CUPHY_ERROR: {} ({}) Max number of FFTs per FFT node exceeded for FFT size: {}", __FILE__,
                           __LINE__, prach_params.Nfft);
                    return CUPHY_STATUS_INVALID_ARGUMENT;
                }
                for (int k = 0; k < numFfts; ++k) {
                    info.data[info.numFfts + k] = reinterpret_cast<cuFloatComplex *>(
                        d_fft + k * prach_params.Nfft);
                }
                info.numFfts += numFfts;
                break;
            }
            if (!foundValidFftSize) {
                NVLOGE_FMT(NVLOG_PRACH, AERIAL_CUDA_API_EVENT, "CUPHY_ERROR: {} ({}) No valid FFT size found for occasion {}, FFT size: {}", __FILE__,
                           __LINE__, i, prach_params.Nfft);
                return CUPHY_STATUS_INVALID_ARGUMENT;
            }
        }
    }

    for (int i = 0; i < PRACH_NUM_SUPPORTED_FFT_SIZES; ++i) {
        auto &info = fftInfo[i];
        cudaKernelNodeParams kernelNodeParams = {0};
        void *kernelArgs[2] = {(void *)&info.data, (void *)&info.numFfts};
        kernelNodeParams.func = (void*)info.kernelPtr;
        dim3 grid((info.numFfts + info.ffts_per_block - 1) / info.ffts_per_block);
        // If numFfts == 0, we will disable this node, but we still set a valid grid dimension.
        // If the node were to run with this configuration, the single CTA would early-exit when checking
        // the numFfts parameter. We add the node so that we always have PRACH_NUM_SUPPORTED_FFT_SIZES
        // FFT nodes in the graph.
        if (grid.x == 0) { grid.x = 1; }
        kernelNodeParams.gridDim = grid;
        kernelNodeParams.blockDim = info.block_dim;
        kernelNodeParams.sharedMemBytes = info.shared_memory_size;
        kernelNodeParams.kernelParams = (void **)kernelArgs;
        kernelNodeParams.extra = NULL;

        CUDA_CHECK_EXCEPTION(
        cudaGraphAddKernelNode(&nodes[GraphNodeType::FFTNode + i], *graph, &nodes[GraphNodeType::ComputeCorrelationNode],
                                1, &kernelNodeParams));
    }

    {
        assert(max_zoneSizeExt <= 512);
        dim3 block_dim(max_zoneSizeExt > NUM_THREAD ? max_zoneSizeExt : NUM_THREAD);
        int Nzone = block_dim.x/max_zoneSizeExt;

        cudaKernelNodeParams kernelNodeParams = {0};
        void *kernelArgs[3] = {(void *)&d_dynParam, &d_staticParam, &max_zoneSizeExt};

        kernelNodeParams.func = (void *)prach_compute_pdp<Tcomplex, Tscalar>;
        // Update: nMaxOccasions -> nOccaProc
        kernelNodeParams.gridDim = dim3((maxAntenna * CUPHY_PRACH_RX_NUM_PREAMBLE + Nzone-1) / Nzone, nTotCellOcca);
        kernelNodeParams.blockDim = block_dim;
        kernelNodeParams.sharedMemBytes = 0;
        kernelNodeParams.kernelParams = (void **)kernelArgs;
        kernelNodeParams.extra = NULL;

        CUDA_CHECK_EXCEPTION(
        cudaGraphAddKernelNode(&nodes[GraphNodeType::ComputePDPNode], *graph, &nodes[GraphNodeType::FFTNode],
            PRACH_NUM_SUPPORTED_FFT_SIZES, &kernelNodeParams));
    }

    {
        cudaKernelNodeParams kernelNodeParams = {0};
        void *kernelArgs[7] = {(void *)&d_dynParam, &d_staticParam,
                            &num_detectedPrmb_addr, &prmbIndex_estimates_addr, &prmbDelay_estimates_addr,
                            &prmbPower_estimates_addr, &interference_addr};

        kernelNodeParams.func = (void *)prach_search_pdp<Tcomplex, Tscalar>;
        // Update: nMaxOccasions -> nOccaProc
        kernelNodeParams.gridDim = dim3(nMaxOccasions);
        kernelNodeParams.blockDim =dim3(CUPHY_PRACH_RX_NUM_PREAMBLE);
        kernelNodeParams.sharedMemBytes = 0;
        kernelNodeParams.kernelParams = (void **)kernelArgs;
        kernelNodeParams.extra = NULL;

        CUDA_CHECK_EXCEPTION(
        cudaGraphAddKernelNode(&nodes[GraphNodeType::SearchPDPNode], *graph, &nodes[GraphNodeType::ComputePDPNode],
                                1, &kernelNodeParams));
    }

    CUDA_CHECK_EXCEPTION(cudaGraphInstantiate(graphInstance, *graph, NULL, NULL, 0));

    // Disable and FFT nodes for which numFFTs is 0. We included the nodes to simplify the
    // graph update logic by keeping the number of nodes static.
    for (int i = 0; i < PRACH_NUM_SUPPORTED_FFT_SIZES; ++i) {
        if (fftInfo[i].numFfts == 0) {
            CUDA_CHECK_EXCEPTION(cudaGraphNodeSetEnabled(*graphInstance, nodes[GraphNodeType::FFTNode + i], 0));
        }
    }

    return CUPHY_STATUS_SUCCESS;
}

cuphyStatus_t cuphyPrachUpdateGraph(cudaGraphExec_t graphInstance, std::vector<cudaGraphNode_t>& nodes,
                                    const PrachInternalDynParamPerOcca* d_dynParam,
                                    const PrachDeviceInternalStaticParamPerOcca* d_staticParam,
                                    const PrachInternalDynParamPerOcca* h_dynParam,
                                    const PrachInternalStaticParamPerOcca* h_staticParam,
                                    cuFloatComplex **h_fftPointers,
                                    std::array<FftInfo, PRACH_NUM_SUPPORTED_FFT_SIZES> &fftInfo,
                                    uint32_t* num_detectedPrmb_addr,
                                    uint32_t* prmbIndex_estimates_addr,
                                    float* prmbDelay_estimates_addr,
                                    float* prmbPower_estimates_addr,
                                    float* ant_rssi_addr,
                                    float* rssi_addr,
                                    float* interference_addr,
                                    uint32_t*& prev_num_detectedPrmb_addr,
                                    uint32_t*& prev_prmbIndex_estimates_addr,
                                    float*& prev_prmbDelay_estimates_addr,
                                    float*& prev_prmbPower_estimates_addr,
                                    float*& prev_ant_rssi_addr,
                                    float*& prev_rssi_addr,
                                    float*& prev_interference_addr,
                                    uint16_t nTotCellOcca,
                                    uint16_t& nPrevOccaProc,
                                    uint16_t nOccaProc,
                                    uint16_t maxAntenna,
                                    uint max_l_oran_ant,
                                    uint max_ant_u,
                                    uint max_nfft,
                                    int max_zoneSizeExt,
                                    std::vector<char>& activeOccasions,
                                    std::vector<char>& prevActiveOccasions)
{
    using Tcomplex = cuFloatComplex;
    using Tscalar = float;

    // During graph creation, we created one FFT node for each FFT size, even if there are no
    // corresponding occasions. If the active occasion set has changed, then we need to update the
    // kernel parameters for at least one FFT node. We only update kernel parameters rather than
    // disable nodes in the case that we have no FFTs of a given size. The FFT kernel in that
    // case will check its block index against the number of FFTs (0) and early-exit.
    std::array<bool, PRACH_NUM_SUPPORTED_FFT_SIZES> fftInfoChanged = {false};
    bool someFftInfoChanged = false;
    for(int i = 0; i < nOccaProc; ++i)
    {
        uint16_t occaPrmStaticIdx = h_dynParam[i].occaPrmStatIdx;
        if(activeOccasions[occaPrmStaticIdx] == prevActiveOccasions[occaPrmStaticIdx]) {
            continue;
        }
        const PrachInternalStaticParamPerOcca& staticOccaParam = h_staticParam[occaPrmStaticIdx];
        const PrachParams& prach_params = staticOccaParam.prach_params;
        for(int j = 0; j < PRACH_NUM_SUPPORTED_FFT_SIZES; ++j) {
            if(prach_params.Nfft == PRACH_SUPPORTED_FFT_SIZES[j]) {
                fftInfoChanged[j] = true;
                someFftInfoChanged = true;
                break;
            }
        }
        prevActiveOccasions[occaPrmStaticIdx] = activeOccasions[occaPrmStaticIdx];
    }

    if (someFftInfoChanged) {
        std::array<uint32_t, PRACH_NUM_SUPPORTED_FFT_SIZES> numPrevFfts;
        for (int i = 0; i < PRACH_NUM_SUPPORTED_FFT_SIZES; ++i) {
            numPrevFfts[i] = fftInfo[i].numFfts;
        }

        // We need to refresh fftInfo to update at least one graph node
        cuphyStatus_t status = refresh_fftinfo_array<Tcomplex>(fftInfo, h_dynParam, h_staticParam, nOccaProc, activeOccasions);
        if (status != CUPHY_STATUS_SUCCESS) {
            return status;
        }
        for (int i = 0; i < PRACH_NUM_SUPPORTED_FFT_SIZES; ++i) {
            // Restrict kernel node updates to only the nodes that have actually changed.
            if (! fftInfoChanged[i]) {
                continue;
            }

            const auto &info = fftInfo[i];
            // If we need to disable the node, we do so without also updating the kernel parameters.
            if (info.numFfts == 0) {
                CUDA_CHECK_EXCEPTION(cudaGraphNodeSetEnabled(graphInstance, nodes[GraphNodeType::FFTNode + i], 0));
                continue;
            }

            cudaKernelNodeParams kernelNodeParams = {0};
            void *kernelArgs[2] = {(void *)&info.data, (void *)&info.numFfts};
            kernelNodeParams.func = (void*)info.kernelPtr;
            // info.numFfts > 0 due to the above node-disable check.
            dim3 grid((info.numFfts + info.ffts_per_block - 1) / info.ffts_per_block);
            kernelNodeParams.gridDim = grid;
            kernelNodeParams.blockDim = info.block_dim;
            kernelNodeParams.sharedMemBytes = info.shared_memory_size;
            kernelNodeParams.kernelParams = (void **)kernelArgs;
            kernelNodeParams.extra = NULL;

            if (numPrevFfts[i] == 0) {
                // Enable the node; it previously had 0 FFTs and was thus disabled.
                CUDA_CHECK_EXCEPTION(cudaGraphNodeSetEnabled(graphInstance, nodes[GraphNodeType::FFTNode + i], 1));
            }

            CUDA_CHECK_EXCEPTION(
                cudaGraphExecKernelNodeSetParams(graphInstance, nodes[GraphNodeType::FFTNode + i], &kernelNodeParams));
        }
    }

    if(nPrevOccaProc != nOccaProc || prev_num_detectedPrmb_addr != num_detectedPrmb_addr ||
        prev_prmbIndex_estimates_addr != prmbIndex_estimates_addr || prev_prmbDelay_estimates_addr != prmbDelay_estimates_addr ||
        prev_prmbPower_estimates_addr != prmbPower_estimates_addr || prev_ant_rssi_addr != ant_rssi_addr ||
        prev_rssi_addr != rssi_addr || prev_interference_addr != interference_addr)
    {
        if(nPrevOccaProc != nOccaProc)
        {
            dim3 block_dim(128, 1, 1);
            cudaKernelNodeParams kernelNodeParams = {0};
            // Update: nMaxOccasions -> nOccaProc
            void *kernelArgs[4] = {(void *)&d_dynParam, (void *)&d_staticParam, &maxAntenna,  &nOccaProc};

            kernelNodeParams.func = (void *)memsetRssi<Tcomplex, Tscalar>;
            // Update: nMaxOccasions -> nOccaProc
            kernelNodeParams.gridDim = dim3((nOccaProc * (maxAntenna + 2) + block_dim.x - 1) / block_dim.x, 1, 1);
            kernelNodeParams.blockDim = block_dim;
            kernelNodeParams.sharedMemBytes = 0;
            kernelNodeParams.kernelParams = (void **)kernelArgs;
            kernelNodeParams.extra = NULL;

            CUDA_CHECK_EXCEPTION(
            cudaGraphExecKernelNodeSetParams(graphInstance, nodes[GraphNodeType::MemsetRSSI],
                                            &kernelNodeParams));
        }

        if(nPrevOccaProc != nOccaProc || rssi_addr != prev_rssi_addr)
        {
            dim3 block_dim(128, 1, 1);
            cudaKernelNodeParams kernelNodeParams = {0};
            void *kernelArgs[4] = {(void *)&d_dynParam, (void *)&d_staticParam, &rssi_addr, &max_l_oran_ant};

            kernelNodeParams.func = (void *)prach_compute_rssi<Tcomplex, Tscalar>;
            // Update: nMaxOccasions -> nOccaProc
            kernelNodeParams.gridDim = dim3((max_l_oran_ant + block_dim.x - 1) / block_dim.x, nOccaProc, 1);
            kernelNodeParams.blockDim = block_dim;
            kernelNodeParams.sharedMemBytes = 0;
            kernelNodeParams.kernelParams = (void **)kernelArgs;
            kernelNodeParams.extra = NULL;

            CUDA_CHECK_EXCEPTION(
            cudaGraphExecKernelNodeSetParams(graphInstance, nodes[GraphNodeType::ComputeRSSI],
                                            &kernelNodeParams));
        }

        if(nPrevOccaProc != nOccaProc || ant_rssi_addr != prev_ant_rssi_addr)
        {
            dim3 block_dim(128, 1, 1);
            cudaKernelNodeParams kernelNodeParams = {0};
            // Update: nMaxOccasions -> nOccaProc
            void *kernelArgs[5] = {(void *)&ant_rssi_addr, (void *)&d_dynParam, &d_staticParam, &maxAntenna, &nOccaProc};

            kernelNodeParams.func = (void *)memcpyRssi<Tcomplex, Tscalar>;
            // Update: nMaxOccasions -> nOccaProc
            kernelNodeParams.gridDim = dim3((nOccaProc * maxAntenna + block_dim.x - 1) / block_dim.x, 1, 1);
            kernelNodeParams.blockDim = block_dim;
            kernelNodeParams.sharedMemBytes = 0;
            kernelNodeParams.kernelParams = (void **)kernelArgs;
            kernelNodeParams.extra = NULL;

            CUDA_CHECK_EXCEPTION(
            cudaGraphExecKernelNodeSetParams(graphInstance, nodes[GraphNodeType::MemcpyRSSI],
                                            &kernelNodeParams));
        }

        if(nPrevOccaProc != nOccaProc || ant_rssi_addr != prev_ant_rssi_addr)
        {
            cudaKernelNodeParams kernelNodeParams = {0};
            // Update: nMaxOccasions -> nOccaProc
            void *kernelArgs[4] = {(void *)&d_dynParam, &d_staticParam, &max_ant_u, &nOccaProc};

            kernelNodeParams.func = (void *)prach_compute_correlation<Tcomplex, Tscalar>;
            // Update: nMaxOccasions -> nOccaProc
            kernelNodeParams.gridDim = dim3(max_ant_u * nOccaProc);
            kernelNodeParams.blockDim = dim3(max_nfft);
            kernelNodeParams.sharedMemBytes = 0;
            kernelNodeParams.kernelParams = (void **)kernelArgs;
            kernelNodeParams.extra = NULL;

            CUDA_CHECK_EXCEPTION(
            cudaGraphExecKernelNodeSetParams(graphInstance, nodes[GraphNodeType::ComputeCorrelationNode], &kernelNodeParams));
        }

        if(nPrevOccaProc != nOccaProc)
        {
            assert(max_zoneSizeExt <= 512);
            dim3 block_dim(max_zoneSizeExt > NUM_THREAD ? max_zoneSizeExt : NUM_THREAD);
            int Nzone = block_dim.x/max_zoneSizeExt;

            cudaKernelNodeParams kernelNodeParams = {0};
            void *kernelArgs[3] = {(void *)&d_dynParam, &d_staticParam, &max_zoneSizeExt};

            kernelNodeParams.func = (void *)prach_compute_pdp<Tcomplex, Tscalar>;
            // Update: nMaxOccasions -> nOccaProc
            kernelNodeParams.gridDim = dim3((maxAntenna * CUPHY_PRACH_RX_NUM_PREAMBLE + Nzone-1) / Nzone, nOccaProc);
            kernelNodeParams.blockDim = block_dim;
            kernelNodeParams.sharedMemBytes = 0;
            kernelNodeParams.kernelParams = (void **)kernelArgs;
            kernelNodeParams.extra = NULL;

            CUDA_CHECK_EXCEPTION(
                cudaGraphExecKernelNodeSetParams(graphInstance, nodes[GraphNodeType::ComputePDPNode],
                                                &kernelNodeParams));
        }

        if(nPrevOccaProc != nOccaProc || prev_num_detectedPrmb_addr != num_detectedPrmb_addr ||
        prev_prmbIndex_estimates_addr != prmbIndex_estimates_addr || prev_prmbDelay_estimates_addr != prmbDelay_estimates_addr ||
        prev_prmbPower_estimates_addr != prmbPower_estimates_addr || prev_interference_addr != interference_addr)
        {
            cudaKernelNodeParams kernelNodeParams = {0};
            void *kernelArgs[7] = {(void *)&d_dynParam, &d_staticParam,
                                &num_detectedPrmb_addr, &prmbIndex_estimates_addr, &prmbDelay_estimates_addr,
                                &prmbPower_estimates_addr, &interference_addr};

            kernelNodeParams.func = (void *)prach_search_pdp<Tcomplex, Tscalar>;
            // Update: nMaxOccasions -> nOccaProc
            kernelNodeParams.gridDim = dim3(nOccaProc);
            kernelNodeParams.blockDim =dim3(CUPHY_PRACH_RX_NUM_PREAMBLE);
            kernelNodeParams.sharedMemBytes = 0;
            kernelNodeParams.kernelParams = (void **)kernelArgs;
            kernelNodeParams.extra = NULL;

            CUDA_CHECK_EXCEPTION(
                cudaGraphExecKernelNodeSetParams(graphInstance, nodes[GraphNodeType::SearchPDPNode],
                                                &kernelNodeParams));
        }

        nPrevOccaProc = nOccaProc;
        prev_num_detectedPrmb_addr = num_detectedPrmb_addr;
        prev_prmbIndex_estimates_addr = prmbIndex_estimates_addr;
        prev_prmbDelay_estimates_addr = prmbDelay_estimates_addr;
        prev_prmbPower_estimates_addr = prmbPower_estimates_addr;
        prev_ant_rssi_addr = ant_rssi_addr;
        prev_rssi_addr = rssi_addr;
        prev_interference_addr = interference_addr;
    }

    return CUPHY_STATUS_SUCCESS;
}

cuphyStatus_t cuphyPrachLaunchGraph(cudaGraphExec_t graphInstance, cudaStream_t strm)
{
    CUDA_CHECK_EXCEPTION(cudaGraphLaunch(graphInstance, strm));
    return CUPHY_STATUS_SUCCESS;
}

cuphyStatus_t cuphyPrachReceiver(const PrachInternalDynParamPerOcca* d_dynParam,
                        const PrachDeviceInternalStaticParamPerOcca* d_staticParam,
                        const PrachInternalDynParamPerOcca* h_dynParam,
                        const PrachInternalStaticParamPerOcca* h_staticParam,
                        cuFloatComplex **h_fftPointers,
                        std::array<FftInfo, PRACH_NUM_SUPPORTED_FFT_SIZES> &fftInfo,
                        uint32_t* num_detectedPrmb_addr,
                        uint32_t* prmbIndex_estimates_addr,
                        float* prmbDelay_estimates_addr,
                        float* prmbPower_estimates_addr,
                        float* ant_rssi_addr,
                        float* rssi_addr,
                        float* interference_addr,
                        uint16_t nOccaProc,
                        uint16_t maxAntenna,
                        uint max_l_oran_ant,
                        uint max_ant_u,
                        uint max_nfft,
                        int max_zoneSizeExt,
                        uint cudaDeviceArch,
                        cudaStream_t strm) {

    return launch_templated_prach_receiver_kernels<cuFloatComplex, float>(d_dynParam,
                                                                    d_staticParam,
                                                                    h_dynParam,
                                                                    h_staticParam,
                                                                    h_fftPointers,
                                                                    fftInfo,
                                                                    num_detectedPrmb_addr,
                                                                    prmbIndex_estimates_addr,
                                                                    prmbDelay_estimates_addr,
                                                                    prmbPower_estimates_addr,
                                                                    ant_rssi_addr,
                                                                    rssi_addr,
                                                                    interference_addr,
                                                                    nOccaProc,
                                                                    maxAntenna,
                                                                    max_l_oran_ant,
                                                                    max_ant_u,
                                                                    max_nfft,
                                                                    max_zoneSizeExt,
                                                                    cudaDeviceArch,
                                                                    strm);

    // return CUPHY_STATUS_NOT_SUPPORTED;
}
