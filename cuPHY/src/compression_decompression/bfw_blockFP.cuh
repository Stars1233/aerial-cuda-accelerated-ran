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

#include <cstdint>
#include <type_traits>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include "tensor_desc.hpp"  // Include for complex_from_scalar type definition

#include "bfw_packing.cuh"

namespace cg = cooperative_groups;

// ****************************************************************************
// bfw_scale_compress_blockFP:
// -----------------------
// Power-scale and compress beam forming weights using the blockFP compression.
// Each thread processes 4 antennas IQ pairs (8 values)
// input: FP32 (or FP16) shared memory input array, must be 16-byte aligned,
//        no padding allowed, contains N_ANT * N_LAYER contiguous complex values.
//        The stride between 2 PRBs is sm_stride values (typically (N_ANT + 1))
// output: Transposed and compressed output containing packed PRBs.
//        The dimension of the output is (N_ANT, ngrps, N_LAYERS)
//        The compressed size is (N_ANT / 4 * compbits + 1) bytes per PRB.
// Uncompressed output:
//        If compbits is 16, the output is uncompressed 16-bit integers
//        If compbits is 32, the data is written in the original FP format (debug)
// beta: Power-scaling factor.
// compbits: Number of compressed bits for the blockFP format.
//        If compbits is either 16 or 32, no compression is performed.
//        Supported values: 6,7,8,9,10,11,12,13,14,15,16,32
// tid: Linear thread ID

// N_ANT : Number of antennas, power of 2 between 4 and 128.
// N_LAYERS: Number of layers
// N_THREADS: Number of threads participating in the compression, multiple of 32.

template <typename TCompute, int SM_LDIM, int N_ANT, int N_LAYERS, int MAX_PRBG_PER_LAYER, int N_THREADS>
__device__ inline void bfw_scale_compress_blockFP(
    typename complex_from_scalar<TCompute>::type* smem, // Shared memory input pointer for the antennas
    uint8_t* __restrict__ output,                       // Output pointer for the first antenna
    float   beta,                                       // Scaling factor
    uint8_t compbits,                                   // Number of bits. If >= 16, pass-through uncompressed
    int16_t beamIdOffset,                               // Starting offset for dynamic beam IDs
    int32_t tid,                                        // 1D thread rank
    int32_t ngrps)                                      // Number of PRB groups
{
    static_assert(N_THREADS % 32 == 0);
    static_assert(N_ANT == 4 || N_ANT == 8 || N_ANT == 16 || N_ANT == 32 || N_ANT == 64);

    using TComplex = typename complex_from_scalar<TCompute>::type;

    // Special debug mode to write uncompressed scaled FP values.
    if(compbits == 32)
    {
        // In this mode, each thread processes one antenna, fully coalesced.
        static_assert(N_THREADS % N_ANT == 0);
        int           prb_id         = tid / N_ANT;
        int           ant_id         = tid % N_ANT;
        constexpr int layers_per_cta = N_THREADS / N_ANT;
        TComplex*     output_ptr     = reinterpret_cast<TComplex*>(output);
        // Loop on all the layers
        for(int prb_loop = 0; prb_loop < N_LAYERS; prb_loop += layers_per_cta)
        {
            int cur_prb = prb_loop + prb_id;
            if(N_LAYERS % layers_per_cta == 0 || cur_prb < N_LAYERS)
            {
                int output_index = cur_prb * ngrps * N_ANT + ant_id;
                // Scale and transpose from (ANT, LAYERS, PRB_GRP) -> (ANT, PRB_GRP, LAYERS)
                TComplex v = smem[cur_prb * SM_LDIM + ant_id];
                v.x *= beta;
                v.y *= beta;
                output_ptr[output_index] = v;
            }
        }
        return;
    }

    // Each thread works on 4 IQ pairs
    constexpr int32_t threads_per_prb = N_ANT / 4;
    constexpr int32_t prbs_per_cta    = N_THREADS / threads_per_prb;

    // Split the block into sub-warp partitions working on the same PRB
    auto tile_prb = cg::tiled_partition<threads_per_prb>(cg::this_thread_block());
    int  tx_prb   = tile_prb.thread_rank();
    int  ty_prb   = tile_prb.meta_group_rank();

    // Loop on all the PRBs, using all the threads
    for(int prb_loop = 0; prb_loop < N_LAYERS; prb_loop += prbs_per_cta)
    {
        // Load 4 consecutive IQ pairs per thread, scale and convert to 32-bit integers
        int vi[4], vq[4];
        if(prb_loop + ty_prb < N_LAYERS)
        {
            for(int i = 0; i < 4; i++)
            {
                TComplex v = smem[(prb_loop + ty_prb) * SM_LDIM + 4 * tx_prb + i];

                vi[i] = (int)((float)v.x * beta);
                vq[i] = (int)((float)v.y * beta);
            }
        }

        int32_t shift;

        if(compbits < 16) // Compression
        {
            // Min / max of all the local values
            int vmin = min(min(min(vi[0], vq[0]), min(vi[1], vq[1])),
                           min(min(vi[2], vq[2]), min(vi[3], vq[3])));
            int vmax = max(max(max(vi[0], vq[0]), max(vi[1], vq[1])),
                           max(max(vi[2], vq[2]), max(vi[3], vq[3])));

            // Absolute max across all the PRB
            vmax = max(vmax, abs(vmin) - 1);
            vmax = cg::reduce(tile_prb, vmax, cg::greater<int>());

            // Find the right shift so that the max value will fit in (compbits-1) bits
            shift = max(0, 33 - __clz(vmax) - compbits); // shift is between 0 and 15 = 4 bits

            // Shift all the values to remove the exponent
            for(int i = 0; i < 4; i++)
            {
                vi[i] >>= shift;
                vq[i] >>= shift;
            }
            if(beamIdOffset >= 0){
                shift |= ((beamIdOffset + (prb_loop + ty_prb)*MAX_PRBG_PER_LAYER + blockIdx.x) << 8); // Populate beam ID in bytes 1 & 2 of bundle header
            } else {
                shift |= 0x00FFFF00; // TODO: clean this up
            }
        }

        // The shared memory is used to stage the packed compressed data, it is used
        // in-place, so each PRB is written at the same offset as when uncompressed.
        tile_prb.sync();

        uint8_t* prb_u8 = reinterpret_cast<uint8_t*>(smem + (prb_loop + ty_prb) * SM_LDIM);
        if(prb_loop + ty_prb < N_LAYERS)
            packPRB(prb_u8, tx_prb, vi, vq, shift, compbits);

        // Now work at the warp level
        // Each warp will transpose and write all the PRBs that it has produced
        auto tile32 = cg::tiled_partition<32>(cg::this_thread_block());
        auto tx     = tile32.thread_rank();
        auto ty     = tile32.meta_group_rank();

        constexpr int prbs_per_warp = 32 * 4 / N_ANT; // 32 threads x 4 antennas per thread

        // Number of bytes for one compressed PRB
        int32_t compbytes = (compbits == 16) ? N_ANT * 4 : 2 * N_ANT / 8 * compbits + 1;
        compbytes += 2*(beamIdOffset>=0); // Add 2 bytes for beam ID in bundle header

        // Loop on all the PRBs produced by the warp
        for(int i = 0; i < prbs_per_warp; i++)
        {
            int iprb = prb_loop + ty * prbs_per_warp + i;
            if(iprb < N_LAYERS)
            {
                // Transpose to an output (N_ANT, ngrps, N_LAYERS)
                uint8_t* compVals = reinterpret_cast<uint8_t*>(smem + iprb * SM_LDIM);
                uint32_t out_offset = iprb * ngrps * compbytes;

                // Copy compressed bytes to global memory
                for (int b = tx; b < compbytes; b += 32) {
                    output[out_offset + b] = compVals[b];
                }
            }
        }
    }
}

// ****************************************************************************
// Block Floating Point decompression and scaling
// Similar approach to the compression code.
// Each thread decompresses 4 consecutive IQ pairs (8 values)

// input: Compressed input containing packed PRBs, with (N_ANT * compbits + 1) bytes per PRB,
//         except if no compression is applied (compbits=16), then N_ANT * 2 bytes per PRB,
//         or if compbits=32 (no compression FP pass-through)
// output: Complex FP output array
// beta: Scaling factor.
// nprb: Number of PRBs to decompress
// compbits: Number of compressed bit of the blockFP format (16 = uncompressed)
// tid: Thread ID participating in the decompression (threads must be coalesced).
// nthreads: Number of threads participating in the compression, one or more full warps.

template <typename TCompute, int N_ANT, int N_THREADS>
__device__ inline void bfw_decompress_blockFP(
    const uint8_t*                                input,         // Compressed input data
    typename complex_from_scalar<TCompute>::type* output,        // Output pointer
    uint32_t                                      output_stride, // Output stride in complex values
    float                                         invbeta,       // Scaling factor
    const uint32_t                                nprb,          // Number of PRB bundles to decompress
    const uint8_t                                 compbits,      // Compressed bits
    bool                                          packed_beam_id, // Whether the beam_id is packed in the compressed data
    const uint32_t                                tid)
{
    static_assert(N_THREADS % 32 == 0);
    static_assert(N_ANT == 4 || N_ANT == 8 || N_ANT == 16 || N_ANT == 32 || N_ANT == 64);

    using TComplex = typename complex_from_scalar<TCompute>::type;

    // compbits=32: Special debug mode, floating-point pass-through
    if(compbits == 32)
    {
        static_assert(N_THREADS % N_ANT == 0);
        int             prb_id       = tid / N_ANT;
        int             ant_id       = tid % N_ANT;
        constexpr int   prbs_per_cta = N_THREADS / N_ANT;
        const TComplex* input_ptr    = reinterpret_cast<const TComplex*>(input);
        // Loop on all the PRBs
        for(int iprb = prb_id; iprb < nprb; iprb += prbs_per_cta)
        {
            TComplex v = input_ptr[iprb * N_ANT + ant_id];
            v.x        = (TCompute)((float)v.x * invbeta);
            v.y        = (TCompute)((float)v.y * invbeta);

            output[iprb * output_stride + ant_id] = v;
        }
        return;
    }

    // One thread works on 4 antenna IQ values
    constexpr uint32_t threads_per_prb = N_ANT / 4;
    constexpr uint32_t prbs_per_cta    = N_THREADS / threads_per_prb;

    // Split the block into sub-warp partitions working on the same PRB
    auto tile_prb = cg::tiled_partition<threads_per_prb>(cg::this_thread_block());
    int  tx_prb   = tile_prb.thread_rank();
    int  ty_prb   = tile_prb.meta_group_rank();

    // Loop on all the prbs to be decompress
    for(int iprb = ty_prb; iprb < nprb; iprb += prbs_per_cta)
    {
        int32_t  vi[4], vq[4];
        int32_t  shift;
        uint32_t compbytes = (compbits == 16) ? 4 * N_ANT : 2 * N_ANT / 8 * compbits + 1;
        compbytes += 2*(packed_beam_id);
        uint32_t offset    = iprb * compbytes;
        unpackInput(input + offset, tx_prb, vi, vq, shift, packed_beam_id, compbits);

        // Extract just the shift value (lower 8 bits) when decompressing
        // as the beam ID information is irrelevant for decompression
        shift = shift & 0xFF;
        
        // Expand the values back to 32-bit integers
        // shift left first then right to propagate the sign bits
        for(int i = 0; i < 4; i++)
        {
            vi[i] = (vi[i] << (32 - compbits)) >> (32 - compbits - shift);
            vq[i] = (vq[i] << (32 - compbits)) >> (32 - compbits - shift);
        }

        // Apply beta scaling factor in FP32, then convert to FP16,
        // write the values to the output
        uint32_t output_index = iprb * output_stride + 4 * tx_prb;
        TComplex v;
        for(int i = 0; i < 4; i++)
        {
            v.x = (TCompute)((float)vi[i] * invbeta);
            v.y = (TCompute)((float)vq[i] * invbeta);

            output[output_index + i] = v;
        }
    }
}
