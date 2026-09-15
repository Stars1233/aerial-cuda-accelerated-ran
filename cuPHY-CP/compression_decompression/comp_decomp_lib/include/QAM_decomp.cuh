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

#include <cuda.h>
#include <cuda_fp16.h>
#include "QAM_param.cuh"
#include "QAM_packing.cuh"

template <class Impl>
__global__ void decompress_QAM_lists(const uint8_t *const *__restrict__ list_inputs,    // Input data, compressed bytes
                                     const QamListParam *__restrict__ list_params,      // Per-list parameters
                                     const QamPrbParam *const *__restrict__ prb_params, // PRB parameters, for each list
                                     float2 *scalers,                                   // PRB scaling, for each list
                                     half **__restrict__ list_outputs,                  // FP16 PRB outputs
                                     const int32_t *__restrict__ nprbs,                 // Number of PRBs in each list
                                     int32_t nlists)                                    // Number of PRB lists
{
    // One warp per list. blockDim.x = 32, blockDim.y = # warps, threadIdx.y = warp id.
    const int32_t listid = blockIdx.x * blockDim.y + threadIdx.y;
    if (listid >= nlists)
        return;
    const int32_t prbid = threadIdx.x / 3; // 3 Threads per PRB, 10 PRBs per warp.
    const int32_t laneid = threadIdx.x % 3;
    if (threadIdx.x >= 30)
    {
        return; // Don't need the last 2 threads of each warp
    }

    // List-specific pointers and parameters
    const uint8_t *input = list_inputs[listid];
    half *output = list_outputs[listid];
    const QamPrbParam *params = prb_params[listid];
    QamListParam lparam = list_params[listid];
    const int32_t n = nprbs[listid];
    float scaler0 = (scalers[listid].x * sqrtf(2.0f));
    float scaler1 = (scalers[listid].y * sqrtf(2.0f));
    float i2f_fact = lparam.get_i2f_fact();

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
        QamPrbParam prb_par = params[prbindex];

        // Call the main processing routine for these 4 REs
        Impl::process_vec4(input, output, prb_par, lparam, prbindex, scaler0, scaler1,
                           i2f_fact, shift_value, compbits, laneid);
    }
}

struct QAM_Decomp
{
    static __host__ __device__ void process_vec4(const uint8_t *input,
                                                 half *output,
                                                 const QamPrbParam &prb_par,
                                                 const QamListParam &lparam,
                                                 const int32_t &prbindex,
                                                 const float &scaler0,
                                                 const float &scaler1,
                                                 const float &i2f_fact,
                                                 const float &shift_value,
                                                 const int &compbits,
                                                 const int &laneid)
    {
        constexpr int threads_per_prb = 3;
        uint32_t index = prbindex * threads_per_prb + laneid;

        // Input offset. Compbits is the number of bits per value,
        // but also the number of bytes per thread which has 8 values
        int offset = index * compbits;

        // Unpack the 4 REs from the compressed format using 2, 3 or 4 bits per value
        int2 viq[4];
        unpackQamInput_x4(input + offset, viq, compbits);

        // Get PRB masks from the PRB parameter
        uint32_t mask0 = prb_par.get_mask<0>() >> ((threads_per_prb-laneid-1) * 4);
        uint32_t mask1 = prb_par.get_mask<1>() >> ((threads_per_prb-laneid-1) * 4);

        // Get the 12-bit mask of which REs must be shifted
        uint32_t shiftmask = lparam.get_shift_mask(mask0, mask1);

        // Write the 4 REs as uint4, to write 4 consecutive pairs per thread = 8 x FP16
        union u128
        {
            uint4 v4;
            half vh[8];
        } vout;

        // Loop on the REs for this thread
        for (int i = 0; i < 4; i++)
        {
            // Apply shift and scaling
            const int idx = 3-i;
            bool needshift = shiftmask & (1 << idx);
            bool inmask0   = mask0     & (1 << idx);
            float scaler = inmask0 ? scaler0 : scaler1;
            float shift = needshift ? shift_value : 0.0f;
            float tmpi = (float)viq[idx].x * i2f_fact + shift;
            float tmpq = (float)viq[idx].y * i2f_fact + shift;
            vout.vh[2 * i    ] = (half)(tmpi * scaler);
            vout.vh[2 * i + 1] = (half)(tmpq * scaler);
        }
        reinterpret_cast<uint4 *>(output)[index] = vout.v4;
    }

    // GPU kernel launcher (Driver API)
    static CUresult gpu_decompress_QAM_lists(CUfunction kernel_func,
                                             const uint8_t *const *__restrict__ list_inputs,    // Input data, compressed bytes
                                             const QamListParam *__restrict__ list_params,      // Per-list parameters
                                             const QamPrbParam *const *__restrict__ prb_params, // PRB parameters, for each list
                                             float2 *scalers,                                   // PRB scaling, for each list
                                             half **__restrict__ list_outputs,                  // FP16 PRB outputs
                                             const int32_t *__restrict__ nprbs,                 // Number of PRBs in each list
                                             int32_t nlists)                                    // Number of PRB lists
    {
        const int nwarps = 8;
        dim3 threads(32, nwarps);
        dim3 blocks((nlists + nwarps - 1) / nwarps);
        auto a0 = list_inputs;
        auto a1 = list_params;
        auto a2 = prb_params;
        auto a3 = scalers;
        auto a4 = list_outputs;
        auto a5 = nprbs;
        auto a6 = nlists;
        void* args[] = {&a0, &a1, &a2, &a3, &a4, &a5, &a6};
        return cuLaunchKernel(kernel_func, blocks.x, blocks.y, blocks.z,
                              threads.x, threads.y, threads.z, 0, nullptr, args, nullptr);
    }

    // CPU decompressor
    static void cpu_decompress_QAM_lists(const uint8_t *const *__restrict__ list_inputs,    // Input data, compressed bytes
                                         const QamListParam *__restrict__ list_params,      // Per-list parameters
                                         const QamPrbParam *const *__restrict__ prb_params, // PRB parameters, for each list
                                         float2 *scalers,                                   // PRB scaling, for each list
                                         half **__restrict__ list_outputs,                  // FP16 PRB outputs
                                         const int32_t *__restrict__ nprbs,                 // Number of PRBs in each list
                                         int32_t nlists)                                    // Number of PRB lists
    {
        // Loop on all the lists (could parallelize with OpenMP)
        for (int listid = 0; listid < nlists; listid++)
        {
            // List-specific pointers and parameters
            const uint8_t *input = list_inputs[listid];
            half *output = list_outputs[listid];
            const QamPrbParam *params = prb_params[listid];
            QamListParam lparam = list_params[listid];
            float scaler0 = (scalers[listid].x * sqrtf(2.0f));
            float scaler1 = (scalers[listid].y * sqrtf(2.0f));
            float i2f_fact = lparam.get_i2f_fact();

            // Get the shift value and bits per element from the list-specific parameter
            float shift_value = lparam.get_shift();
            int compbits = lparam.get_bits_per_value();

            // Loop on all the PRBs of the list
            for (int prbid = 0; prbid < nprbs[listid]; prbid++)
            {
                // Read the PRB parameters
                QamPrbParam prb_par = params[prbid];
                // Loop on 3 vector of 4 REs (like the GPU) to cover all the REs of the PRB
                for (int ivec = 0; ivec < 3; ivec++)
                {

                    process_vec4(input, output, prb_par, lparam, prbid, scaler0, scaler1,
                                 i2f_fact, shift_value, compbits, ivec);
                }
            }
        }
    }
};
