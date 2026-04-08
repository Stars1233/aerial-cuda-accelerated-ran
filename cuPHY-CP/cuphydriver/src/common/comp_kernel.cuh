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

#pragma once
#include "cuphydriver_api.hpp"
#include "QAM_comp.cuh"

#define COMPRESSION_THREADS 576

template <int unique_bitwidth = 0>
__global__ void kernel_compress(compression_params params)
{
    // This block ordering is not the best for input data locality,
    // but hopefully the first symbol will finish ASAP.
    const uint8_t antenna_id = blockIdx.x;
    const int     cell_id    = blockIdx.y;
    const int     symbol_id  = blockIdx.z; 
    
    const int prbs_per_symbol = params.max_num_prb_per_symbol[cell_id];
    const int in_prb_offset = antenna_id * SLOT_NUM_SYMS * prbs_per_symbol + symbol_id * prbs_per_symbol;
    const int out_prb_offset = symbol_id * params.num_antennas[cell_id] * prbs_per_symbol + antenna_id * prbs_per_symbol;
    // coverity[CUDA_INITIATION_OBJECT_DEVICE_THREAD_BLOCK:SUPPRESS]
    // coverity[initiation_object_device_thread_block]
    __shared__ __align__(16) uint8_t sm[COMPRESSION_THREADS * 15]; // (threads / 32) * 10 * 48 bytes
    // coverity[CUDA_INITIATION_OBJECT_DEVICE_THREAD_BLOCK:SUPPRESS]
    // coverity[initiation_object_device_thread_block]
    __shared__ uint8_t* sm_prb_ptr[(COMPRESSION_THREADS / 32) * 10]; // 10 prbs per warp

    uint8_t**  prb_ptrs     = params.prb_ptrs[cell_id];
    const float   beta      = params.beta[cell_id];
    uint8_t *input_buffer   = params.input_ptrs[cell_id];
    uint8_t num_antennas    = params.num_antennas[cell_id];
    uint8_t comptype        = params.comp_meth[cell_id];


    // If a non-zero unique_bitwidth is passed, it is used, and the kernel becomes specialized
    const int32_t compression_bitwidths = unique_bitwidth ? unique_bitwidth : params.bit_width[cell_id];

    if (antenna_id >= num_antennas)
        return;
    
    __half *iptr = &reinterpret_cast<__half *>(input_buffer)[in_prb_offset * 24];
    prb_ptrs += out_prb_offset;

    if(comptype == 0)
    {
        scale_compress_fixed(iptr, prb_ptrs, beta, prbs_per_symbol, compression_bitwidths, threadIdx.x, COMPRESSION_THREADS, sm, sm_prb_ptr);
    } 
    else
    {
        scale_compress_blockFP(iptr, prb_ptrs, beta, prbs_per_symbol, compression_bitwidths, threadIdx.x, COMPRESSION_THREADS, sm, sm_prb_ptr);
    }
    
}

template <class Impl>
__global__ void kernel_mod_compression(compression_params params)
{
    // This block ordering is not the best for input data locality,
    // but hopefully the first symbol will finish ASAP.
    const uint8_t antenna_id = blockIdx.y;
    const int     cell_id    = blockIdx.z;
    const int     symbol_id  = threadIdx.z;
        
    const int prbs_per_symbol = params.max_num_prb_per_symbol[cell_id];
    const int in_prb_offset = antenna_id * SLOT_NUM_SYMS * prbs_per_symbol + symbol_id * prbs_per_symbol;
    const int out_prb_offset = symbol_id * params.num_antennas[cell_id] * prbs_per_symbol + antenna_id * prbs_per_symbol;

    uint8_t**  prb_ptrs     = params.prb_ptrs[cell_id];
    uint8_t *input_buffer   = params.input_ptrs[cell_id];
    uint8_t num_antennas    = params.num_antennas[cell_id];

    if (antenna_id >= num_antennas)
        return;
    
    __half *iptr = &reinterpret_cast<__half *>(input_buffer)[in_prb_offset * 24];
    prb_ptrs += out_prb_offset;

    // One warp per list. blockDim.x = 32, blockDim.y = # warps, threadIdx.y = warp id.
    const int32_t list_id = blockIdx.x * blockDim.y + threadIdx.y;
    int32_t nlists = params.mod_compression_config[cell_id]->num_messages_per_list[antenna_id][symbol_id];
    
    // If there are no items (sections) for a given antenna and symbol (nlists == 0), we also exit here.
    if (list_id >= nlists)
        return;
    const int32_t prbid = threadIdx.x / 3; // 3 Threads per PRB, 10 PRBs per warp.
    const int32_t laneid = threadIdx.x % 3;
    if (threadIdx.x >= 30)
    {
        return; // Don't need the last 2 threads of each warp
    }

    float2  scaler             = params.mod_compression_config[cell_id]->scaling[antenna_id][symbol_id][list_id];
    int32_t nprbs              = params.mod_compression_config[cell_id]->nprbs_per_list[antenna_id][symbol_id][list_id];
    int32_t prb_start          = params.mod_compression_config[cell_id]->prb_start_per_list[antenna_id][symbol_id][list_id];
    QamListParam list_param    = params.mod_compression_config[cell_id]->params_per_list[antenna_id][symbol_id][list_id];
    QamPrbParam prb_param      = params.mod_compression_config[cell_id]->prb_params_per_list[antenna_id][symbol_id][list_id];


    // List-specific pointers and parameters
    const half *input = iptr + (24 * prb_start); 
    //prb_ptrs += prb_start;
    //uint8_t *output = *(prb_ptrs + prb_start); 
    uint8_t *output = nullptr;
    QamPrbParam  prb_par = prb_param;
    QamListParam lparam = list_param;
    const int32_t n = nprbs;
    float invscaler0 = (1.0f / (scaler.x * sqrtf(2.0f)));
    float invscaler1 = (1.0f / (scaler.y * sqrtf(2.0f)));
    float f2i_fact = lparam.get_f2i_fact();

    // Get the shift value and bits per element from the list-specific parameter
    float shift_value = lparam.get_shift();
    int compbits = lparam.get_bits_per_value();

    // Loop on all the PRBs in the list, 10 prbs at a time
    for (int32_t prbloop = 0; prbloop < n; prbloop += 10)
    {
        uint32_t prbindex = prbloop + prbid;
        if (prbindex >= n)
            return;
        // Read PRB parameter
        output = *(prb_ptrs + prb_start + prbindex);
 
        // Call the main processing routine for these 4 REs
        Impl::process_vec4(input, output, prb_par, lparam, prbindex, invscaler0, invscaler1,
                           f2i_fact, shift_value, compbits, laneid);
    }
}
