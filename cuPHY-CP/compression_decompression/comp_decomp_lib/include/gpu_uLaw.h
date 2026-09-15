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
// u-Law compand / decompand

// This implementation minimizes divergence:
// Example of mu-Law companding from 16-bit input to 12-bit compressed, with mu=8
// Maximum output amplitude with 12-bit compression (including 1 sign bit) = 2047
// Input value is first clamped to 32767.
// 3 possible ranges for the values to be compressed:
// Range 0 : Values in [    0 :  8191] : result = (value >> 3) + 0 * 512
// Range 1 : Values in [ 8192 : 16383] : result = (value >> 4) + 1 * 512
// Range 2 : Values in [16384 : 32767] : result = (value >> 5) + 2 * 512

__device__ inline int compand(int value, int mantissaBits)
{
    value = min(value, 32767);
    // __clz return type changed in CUDA 13.2+ from signed to unsigned for ARM64 systems, thus the explicit cast
    int vrange = max(0, 19 - (int)__clz(value));
    int shift = 15 - mantissaBits + vrange;
    int fact = 1 << (mantissaBits - 3);
    return ((value >> shift) + vrange * fact);
}

__device__ inline int decompand(int value, int mantissaBits)
{
    // vrange is extracted from the 2 most significant bits
    int vrange = max(0, (value >> (mantissaBits - 3)) - 1);
    int shift = 15 - mantissaBits + vrange;
    int fact = 1 << (mantissaBits - 3);
    return ((value - vrange * fact) << shift);
}

// ****************************************************************************
// u-Law compression
// Device function using 4 threads per PRB, one PRB is 12 pairs of (I,Q) = 24 values
// Each value I and Q is 16-bit integer.
// Input must be 128-bit aligned (8x 16-bit values in an int4).
// The device functions must be called with full warps (nthreads = multiple of 32)

template <bool network_order_input, bool inplace = false>
__device__ inline void compress_uLaw(int4 *input, unsigned char *output,
                                     int nprb, int compbits, int tid, int nthreads)
{
    // Use 4 threads per PRB (3 active threads), 4 pairs of (I,Q) per thread = 8 values
    int laneid = tid & 3;
    int prbid = tid / 4;
    int loader = min(2, laneid); // Thread 3 will load same data as thread 2
    int signmask = 1 << (compbits - 1);

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
        int vi[4], vq[4], si[4], sq[4];
        for (int i = 0; i < 4; i++)
        {
            vi[i] = vin.vs[2 * i];
            vq[i] = vin.vs[2 * i + 1];
        }

        // Capture the signs and switch to absolute value
        for (int i = 0; i < 4; i++)
        {
            si[i] = (vi[i] >> 31) & signmask;
            sq[i] = (vq[i] >> 31) & signmask;
            vi[i] = abs(vi[i]);
            vq[i] = abs(vq[i]);
        }

        // Max absolute value
        int maxV = max(vi[0], vq[0]);
        for (int i = 1; i < 4; i++)
            maxV = max(maxV, max(vi[i], vq[i]));

        // Global  max across 4 threads (thread 3 has the same values as thread 2)
        maxV = max(maxV, __shfl_xor_sync(0xffffffff, maxV, 1, 4));
        maxV = max(maxV, __shfl_xor_sync(0xffffffff, maxV, 2, 4));

        // Compute shift based on global max
        // __clz return type changed in CUDA 13.2+ from signed to unsigned for ARM64 systems, thus the explicit cast
        int shift = (int)__clz(maxV) - 17; // Looking for values up to 2^15 not included
        shift = max(0, shift);
        shift = min(7, shift);

        // Left-shift and compand all the values, add the sign
        for (int i = 0; i < 4; i++)
        {
            vi[i] <<= shift;
            vq[i] <<= shift;
            vi[i] = compand(vi[i], compbits - 1) + si[i];
            vq[i] = compand(vq[i], compbits - 1) + sq[i];
        }

        // Write to the output, valid threads only
        if ((laneid < 3) && (prbloop + prbid < nprb))
        {
            if (inplace)
                // In-place : Write back in input using same 48 byte slot.
                packOutput<true>((unsigned char *)input, prbloop + prbid, 48, laneid, vi, vq, shift, compbits);
            else
                // Out-of place: Contiguous compressed PRBs, using (3 * compbits + 1) bytes per PRB
                packOutput<false>(output, prbloop + prbid, 3 * compbits + 1, laneid, vi, vq, shift, compbits);
        }
    }
}

// ****************************************************************************
// u-Law Decompression
// Similar approach to the compression code.
// The 8 values per thread x 3 threads is kept in order to easily unpack the data
// at various bit rates, with always a multiple of 8 bits per thread.
// Output must be 128-bit aligned (8x 16-bit values in an int4).

template <bool network_order_output, bool inplace = false>
__device__ inline void decompress_uLaw(unsigned char *input, int4 *output, int nprb, int compbits, int tid, int nthreads)
{
    // Use 4 threads per PRB (3 active threads), 4 pairs of (I,Q) per thread
    int laneid = tid & 3;
    int prbid = tid / 4;
    int mmask = (1 << (compbits - 1)) - 1;

    // We don't need threads with lane id 3.
    if (laneid == 3)
        return;

    // Loop on all the PRBs, 4 threads per PRB
    for (int prbloop = prbid; prbloop < nprb; prbloop += nthreads / 4)
    {
        int vi[4], vq[4];
        int shift;
        if (inplace)
            unpackInput<true>(input, prbloop, 48, laneid, vi, vq, shift, compbits);
        else
            unpackInput<false>(input, prbloop, 3 * compbits + 1, laneid, vi, vq, shift, compbits);

        for (int i = 0; i < 4; i++)
        {
            // Decompand the value without the sign bit, shift, re-apply the sign
            int ineg = (vi[i] != (vi[i] & mmask));
            int qneg = (vq[i] != (vq[i] & mmask));
            vi[i] = decompand(vi[i] & mmask, compbits - 1) >> shift;
            vq[i] = decompand(vq[i] & mmask, compbits - 1) >> shift;
            if (ineg)
                vi[i] = -vi[i];
            if (qneg)
                vq[i] = -vq[i];
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
