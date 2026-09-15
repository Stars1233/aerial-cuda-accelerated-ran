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

#include <cuda_fp16.h>
#include "gpu_packing.h"

// ****************************************************************************
// scale_compress_blockFP:
// -----------------------
// Power-scale and compress FP16 PRBs using the blockFP compression
// input: FP16 input array, must be aligned on 128-bit boundary
// prb_ptrs: Individual PRB pointers, must be 16-bit aligned.
//           Each PRB must have a size of (3 * compbits + 1) bytes,
//           except if no compression is applied (compbits=16), then 48 bytes per PRB.
//           A nullptr value means the PRB is invalid.
// beta: Power-scaling factor.
// nprb: Number of PRBs to compress (some may be invalid, based on the prb_ptr value)
// compbits: Number of compressed bit of the blockFP format.
//           If compbits is 16, no compression is performed.
// tid: Thread ID participating in the compression (threads must be coalesced).
// nthreads: Number of threads participating in the compression, one or more full warps.
// sm: shared memory pointer to accelerate data packing, need 16 bytes per thread.
// sm_prb_ptr: shared memory pointer to accelerate pointer sharing,
//             size = (nthreads / 4) * sizeof (void *)


__device__ inline void scale_compress_blockFP(const half *__restrict__ input,
                                              uint8_t **__restrict__ prb_ptrs,
                                              const float beta,
                                              int32_t nprb,
                                              int32_t compbits,
                                              int32_t tid,
                                              int32_t nthreads,
                                              uint8_t *sm,
                                              uint8_t **sm_prb_ptr)
{
    // Use 3 threads per PRB, 8 values per thread, 10 PRBs per warp.
    // Threads 30, 31 of each warp aren't doing any useful work
    int warpId = tid / 32;
    int warpLane = tid % 32;
    int activeThread = warpLane < 30;
    int laneid = warpLane % 3;
    int prbid = warpLane / 3 + warpId * 10;
    int ctaprbs = (nthreads / 32) * 10;
    if (!activeThread)
        prbid--; // Threads (30,31) will load the same input as threads (27,28)

    int4 *input_vec = (int4 *)input;
    uint8_t *prefetch_prb_ptr = nullptr;
    int4 prefetch_vin;

    // Prefetch first value. Only reading if the PRB is valid
    if (prbid < nprb)
    {
        prefetch_vin = input_vec[prbid * 3 + laneid];
        prefetch_prb_ptr = prb_ptrs[prbid];
    }

    // Loop on all the PRBs.
    // If nprb and nthreads are known at compile time, this loop can be optimized out or unrolled.
    for (int prbloop = 0; prbloop < nprb; prbloop += ctaprbs)
    {
        // Input data: Using uint4, to read 4 consecutive pairs per thread = 8 x FP16
        union u128
        {
            int4 v4;
            half vh[8];
        } vin;
        vin.v4 = prefetch_vin;
        uint8_t *prb_ptr = prefetch_prb_ptr;
        // Prefetch the inputs for the next iteration
        int nextprb = prbloop + prbid + ctaprbs;
        if (nextprb < nprb)
        {
            prefetch_vin = input_vec[nextprb * 3 + laneid];
            prefetch_prb_ptr = prb_ptrs[nextprb];
        }

        bool valid_prb = prb_ptr != nullptr;
        uint32_t valid_mask = __ballot_sync(~0, valid_prb); // Valid PRB mask for the whole warp

        if (valid_mask)
        {
            // Write the PRB pointer in shared memory (instead of shuffles later)
            if (laneid == 0 && activeThread)
                sm_prb_ptr[prbid] = prb_ptr;

            // Special case with no compression, pass-through
            if (compbits == 16)
            {
                writeUncompressed<3>(valid_mask, activeThread, prbid, tid, laneid, vin.v4, sm, sm_prb_ptr);
            }
            else
            {
                // Apply the scaling factor in FP32, and convert to integers
                int32_t vi[4], vq[4];
                for (int i = 0; i < 4; i++)
                {
                    vi[i] = (int)((float)vin.vh[2 * i] * beta);
                    vq[i] = (int)((float)vin.vh[2 * i + 1] * beta);
                }

                int32_t shift;

                // Min / max of all the values
                int32_t minV = min(vi[0], vq[0]);
                int32_t maxV = max(vi[0], vq[0]);
                for (int i = 1; i < 4; i++)
                {
                    minV = min(minV, min(vi[i], vq[i]));
                    maxV = max(maxV, max(vi[i], vq[i]));
                }
                maxV = max(maxV, abs(minV) - 1);

                // Global max on 3 threads
                int32_t tmp1 = __shfl_down_sync(~0, maxV, 1);
                int32_t tmp2 = __shfl_down_sync(~0, maxV, 2);
                maxV = max(maxV, max(tmp1, tmp2));

                // Lowest thread for this PRB has the right value
                maxV = __shfl_sync(~0, maxV, warpLane - laneid);

                // Find the right shift so that the max value will fit in (compbits-1) bits
                // __clz return type changed in CUDA 13.2+ from signed to unsigned for ARM64 systems, thus the explicit cast
                shift = max(0, 33 - (int)__clz(maxV) - compbits); // shift is between 0 and 15 = 4 bits

                // Shift all the values to remove the exponent
                for (int i = 0; i < 4; i++)
                {
                    vi[i] >>= shift;
                    vq[i] >>= shift;
                }

                // Pack the bytes in shared memory and write to the output.
                packOutput<3>(valid_mask, activeThread, prbid, laneid, tid, vi, vq, shift, compbits, sm, sm_prb_ptr);
            }
        }
    }
}




// ****************************************************************************
// Block Floating Point decompression and scaling
// Similar approach to the compression code, even though there is no intra-warp reduction.
// The 8 values per thread x 3 threads is kept in order to easily unpack the data
// at various bit rates, with always a multiple of 8 bits per thread.
// Output must be 128-bit aligned (8x 16-bit values in an int4).
//
// input: Compressed input containing packed PRBs, with (3 * compbits + 1) bytes per PRB,
//         except if no compression is applied (compbits=16), then 48 bytes per PRB.
//         If no compression is applied, the input array must be aligned on 128-bit.
//         If inplace is true, each output PRB overwrites its input, the output parameter is ignored.
// output: FP16 output array, must be aligned on 128-bit boundary
// beta: Scaling factor.
// nprb: Number of PRBs to decompress
// compbits: Number of compressed bit of the blockFP format (16 = uncompressed)
// tid: Thread ID participating in the decompression (threads must be coalesced).
// nthreads: Number of threads participating in the compression, one or more full warps.

template <bool inplace = false>
__device__ inline void decompress_scale_blockFP(uint8_t *input, __half *output, float beta,
                                                const int32_t nprb, const int32_t compbits,
                                                const int32_t tid, const int32_t nthreads)
{
    constexpr int NUM_THREADS_PER_PRB = 3;
    constexpr int WARP_SIZE = 32;

    // Use consecutive groups of 3 threads per PRB, 10 PRBs per warp.
    const int32_t laneid = tid % NUM_THREADS_PER_PRB;
    const int32_t prbid = tid / NUM_THREADS_PER_PRB;

    // The last two threads per warp are inactive.
    if (tid >= NUM_THREADS_PER_PRB*(WARP_SIZE/NUM_THREADS_PER_PRB))
        return;

    // Loop on all the PRBs, 3 threads per PRB, 10 PRBs per warp
    for (int prbloop = prbid; prbloop < nprb; prbloop += WARP_SIZE/NUM_THREADS_PER_PRB)
    {
        int32_t vi[4], vq[4];
        int32_t shift;
        int32_t prbstride = (inplace || (compbits == 16)) ? 48 : 3 * compbits + 1;
        unpackInput(input, prbloop, prbstride, laneid, vi, vq, shift, compbits);

        // Expand the values back to 32-bit integers
        // shift left first then right to propagate the sign bits
        for (int i = 0; i < 4; i++)
        {
            vi[i] = (vi[i] << (32 - compbits)) >> (32 - compbits - shift);
            vq[i] = (vq[i] << (32 - compbits)) >> (32 - compbits - shift);
        }

        // We'll be writing vectors of 128-bit = 8 x half values
        union u128
        {
            int4 v4;
            half vh[8];
        } vec;
        int4 *output_vec = inplace ? (int4 *)input : (int4 *)output;

        // Apply beta scaling factor in FP32, then convert to FP16
        for (int i = 0; i < 4; i++)
        {
            vec.vh[2 * i] = (half)((float)vi[i] * beta);
            vec.vh[2 * i + 1] = (half)((float)vq[i] * beta);
        }

        output_vec[prbloop * 3 + laneid] = vec.v4;
    }
}
