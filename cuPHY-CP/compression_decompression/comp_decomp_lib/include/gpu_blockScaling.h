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

#include "gpu_packing.h"

// ****************************************************************************
// Block Scaling compression.
// See ORAN-WG4.CUS.0-v01.00.pdf, A.2 page 138/139
// "blockScaler can be chosen smaller than Inverse(maxValue)"
// Here, we're chosing the blockScaler to be the inverse of the next power of 2
// above maxValue. This way, the quantization can be done with a shift, just
// like for the blockFP compression.

// Device function using 4 threads per PRB, one PRB is 12 pairs of (I,Q) = 24 values
// Each value I and Q is 16-bit integer.
// Input must be 128-bit aligned (8x 16-bit values in an int4).
// The device functions must be called with full warps (nthreads = multiple of 32)
//
// Implementation notes:
// Using 3 threads computing 8 values per thread allows:
//   - Good instruction level parallelism
//   - Efficent packing of the compressed data, thread output is always a mutliple of 8 bits.
//   - Good balance between local and intra-warp reduction for min/max.
// Compute-bound while using only 75% of the threads.
// But using 30 threads (10 PRBs per warp) seems to generate more instructions...
// 14-bit compression generates quite a lot of shared memory bank conflicts.
// Might want to pad between compressed PRBs? But calling kernel should support it.

template <bool network_order_input, bool inplace = false, bool fullScaling = false>
__device__ inline void compress_blockScaling(int4 *input, unsigned char *output,
                                             int nprb, int compbits, int tid, int nthreads)
{
    // Use 4 threads per PRB (3 active threads), 4 pairs of (I,Q) per thread = 8 values
    int laneid = tid & 3;
    int prbid = tid / 4;
    int loader = min(2, laneid); // Thread 3 will load same data as thread 2

    // Input data: Using uint4, to read 4 consecutive pairs per thread = 8 shorts
    union u128
    {
        int4 v4;
        short vs[8];
    } vin;

    // Loop on all the PRBs, using all the threads, 4 threads per PRB
    // Threads with laneid=3 aren't used, but kept for warp shuffles.
    for (int prbloop = 0; prbloop < nprb; prbloop += nthreads / 4)
    {
        if (prbloop + prbid < nprb)
            vin.v4 = input[(prbloop + prbid) * 3 + loader];

        // Byte-swap from network order to little endian.
        if (network_order_input)
        {
            vin.v4.x = __byte_perm(vin.v4.x, vin.v4.x, 0x2301);
            vin.v4.y = __byte_perm(vin.v4.y, vin.v4.y, 0x2301);
            vin.v4.z = __byte_perm(vin.v4.z, vin.v4.z, 0x2301);
            vin.v4.w = __byte_perm(vin.v4.w, vin.v4.w, 0x2301);
        }

        // Convert the data from 16-bit to 32-bit integers
        int vi[4], vq[4];
        for (int i = 0; i < 4; i++)
        {
            vi[i] = vin.vs[2 * i];
            vq[i] = vin.vs[2 * i + 1];
        }

        // Min / max of all the values
        int minV = min(vi[0], vq[0]);
        int maxV = max(vi[0], vq[0]);
        for (int i = 1; i < 4; i++)
        {
            minV = min(minV, min(vi[i], vq[i]));
            maxV = max(maxV, max(vi[i], vq[i]));
        }

        // Global min max across 4 threads (thread 3 has the same values as thread 2)
        minV = min(minV, __shfl_xor_sync(0xffffffff, minV, 1, 4));
        maxV = max(maxV, __shfl_xor_sync(0xffffffff, maxV, 1, 4));
        minV = min(minV, __shfl_xor_sync(0xffffffff, minV, 2, 4));
        maxV = max(maxV, __shfl_xor_sync(0xffffffff, maxV, 2, 4));
        maxV = max(maxV, abs(minV) - 1);

        int iscal;
        if (fullScaling)
        {
            // Scaling is done in floating point, factor =  128.0f / iscal, with iscal between 0 and 128.
            // Computing the scaling factor on the fly. A shared memory LUT could result in bank conflicts,
            // and a constant memory LUT is not good as accesses could be non-uniform within a warp.
            // This increases the dynamic range, but can also reduce accuracy (+-1 errors)
            iscal = (maxV + 0xff) >> 8; // Adding 0xff to round up -> [0 ~ 128]
            float fscal = __fdividef(128.0f, max(1.0f, (float)iscal));
            for (int i = 0; i < 4; i++)
            {
                float fi = vi[i] * fscal;
                float fq = vq[i] * fscal;
                vi[i] = (int)fi >> (16 - compbits);
                vq[i] = (int)fq >> (16 - compbits);
            }
        }
        else
        {
            // The blockScale parameter can be chosen < 1 / maxV
            // We can choose blockScale as the inverse of the next power of 2 >= maxV.
            // This way, the scaling is done with a simple shift, and blockScaler has a single bit set.
            // Find the left shift that will bring the maxV to 16 bits
            // __clz return type changed in CUDA 13.2+ from signed to unsigned for ARM64 systems, thus the explicit cast
            int shift = min(7, (int)__clz(maxV) - 17);

            // Shift all the values left, round to nearest,
            // then shift right to meet the required number of bits
            for (int i = 0; i < 4; i++)
            {
                vi[i] = (vi[i] << shift) >> (16 - compbits);
                vq[i] = (vq[i] << shift) >> (16 - compbits);
            }
            // Transform the shift into a fixed point scaler
            iscal = 1 << (7 - shift);
        }

        // Write to the output, valid threads only
        if ((laneid < 3) && (prbloop + prbid < nprb))
        {
            if (inplace)
                // In-place : Write back in input using same 48 byte slot.
                packOutput<true>((unsigned char *)input, prbloop + prbid, 48, laneid, vi, vq, iscal, compbits);
            else
                // Out-of place: Contiguous compressed PRBs, using (3 * compbits + 1) bytes per PRB
                packOutput<false>(output, prbloop + prbid, 3 * compbits + 1, laneid, vi, vq, iscal, compbits);
        }
    }
}

// ****************************************************************************
// Block Scaling decompression
// Similar approach to the compression code, even though there is no intra-warp reduction.
// The 8 values per thread x 3 threads is kept in order to easily unpack the data
// at various bit rates, with always a multiple of 8 bits per thread.
// Output must be 128-bit aligned (8x 16-bit values in an int4).

template <bool network_order_output, bool inplace = false>
__device__ inline void decompress_blockScaling(unsigned char *input, int4 *output,
                                               int nprb, int compbits, int tid, int nthreads)
{
    // Use 4 threads per PRB (3 active threads), 4 pairs of (I,Q) per thread
    int laneid = tid & 3;
    int prbid = tid / 4;

    // We don't need threads with lane id 3.
    if (laneid == 3)
        return;

    // Loop on all the PRBs, 4 threads per PRB
    for (int prbloop = prbid; prbloop < nprb; prbloop += nthreads / 4)
    {
        int vi[4], vq[4];
        int iscal;
        if (inplace)
            unpackInput<true>(input, prbloop, 48, laneid, vi, vq, iscal, compbits);
        else
            unpackInput<false>(input, prbloop, 3 * compbits + 1, laneid, vi, vq, iscal, compbits);

        // Sign extend the values
        for (int i = 0; i < 4; i++)
        {
            vi[i] = (vi[i] << (32 - compbits)) >> (32 - compbits);
            vq[i] = (vq[i] << (32 - compbits)) >> (32 - compbits);
        }

        // Even though the iscal factor might be a power of 2
        // (compression done with preciseScaling = false),
        // we have to assume the scaler can be anything during decompression.
        // The multiplication by the fixed point i.fffffff blockscaler
        // is an integer multiplication followed with 7-bit right shift.
        if (compbits >= 9)
            for (int i = 0; i < 4; i++)
            {
                vi[i] = (vi[i] * iscal) >> (compbits + 7 - 16);
                vq[i] = (vq[i] * iscal) >> (compbits + 7 - 16);
            }
        else
            for (int i = 0; i < 4; i++)
            {
                vi[i] = (vi[i] * iscal) << (16 - 7 - compbits);
                vq[i] = (vq[i] * iscal) << (16 - 7 - compbits);
            }

        // Pack the values into 16-bit IQ pairs, into a single int4
        int4 vec;
        vec.x = (vq[0] << 16) | (vi[0] & 0xffff);
        vec.y = (vq[1] << 16) | (vi[1] & 0xffff);
        vec.z = (vq[2] << 16) | (vi[2] & 0xffff);
        vec.w = (vq[3] << 16) | (vi[3] & 0xffff);
        if (network_order_output)
        {
            vec.x = __byte_perm(vec.x, vec.x, 0x2301);
            vec.y = __byte_perm(vec.y, vec.y, 0x2301);
            vec.z = __byte_perm(vec.z, vec.z, 0x2301);
            vec.w = __byte_perm(vec.w, vec.w, 0x2301);
        }
        if (inplace)
            ((int4 *)input)[prbloop * 3 + laneid] = vec;
        else
            output[prbloop * 3 + laneid] = vec;
    }
}
