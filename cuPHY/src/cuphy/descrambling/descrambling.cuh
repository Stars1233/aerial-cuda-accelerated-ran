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

#if !defined(SEQUENCE_CUH_INCLUDED_)
#define SEQUENCE_CUH_INCLUDED_

#include <cuda_runtime.h>
#include "cuphy.h"
#include "cuphy_clmad_util.cuh"
#include "cuphy_internal.h"
#include "descrambling.hpp"
#include "GOLD_2_32_P_LUT.h"
#include "GOLD_1_SEQ_LUT.h"

namespace descrambling
{
// Fibonacci LFSR for second polynomial for Gold sequence generation
CUDA_BOTH inline uint32_t fibonacciLFSR2(uint32_t& state, uint32_t n, uint32_t resInit = 0)
{
    uint32_t res = resInit;
    // x^{31} + x^3 + x^2 + x + 1
    for(int i = 0; i < n; i++)
    {
        uint32_t bit = (state) ^ ((state >> 1)) ^ ((state >> 2)) ^ (state >> 3);
        bit          = bit & 1;
        res >>= 1;
        res ^= (state & 1) << 31; //(% 32);
        state >>= 1;
        state ^= (bit << 30);
    }
    return res;
}

// Fibonacci LFSR for second polynomial for Gold sequence generation, specialized for n=32. This function
// does not accept resInit because with n=32 the initial result will be fully shifted out.
CUDA_BOTH inline uint32_t fibonacciLFSR2_n32(uint32_t& state)
{
    // x^{31} + x^3 + x^2 + x + 1
    // This specialized version uses a closed-form expression to generate 32 bits of output and
    // to update 32 bits of state information. To do so, we first define recurrence relations for
    // the iterative approach shown in fibonacciLFSR2(). Let r(b,i) be the result at bit b before
    // iteration i. Thus, r(b,0) is bit b of the input result (unused here since all input result
    // bits will be shifted out) and r(b,32) is bit b of the final output. Thus, we have
    //
    //              /  s(0,i)       b = 31
    //   r(b,i+1) = |
    //              \  r(b+1,i)     otherwise
    //
    // where s(b,i) is the bth bit of state before iteration i.
    //
    // Applying this recurrence relation to solve for r(b,32), we get
    //   r(0,32) = r(1,31) = ... = r(31,1) = s(0,0)
    //   r(1,32) = r(31,2) = s(0,1)
    //   ...
    //   r(31,32) = s(0,31)
    //
    // We now need the recurrence relations for s(b,i).
    //
    //              /  0                                               b = 31
    //   s(b,i+1) = |  s(31,i) ^ s(0,i) ^ s(1,i) ^ s(2,i) ^ s(3,i)     b = 30
    //              \  s(b+1,i)                                        otherwise
    //
    // Thus,
    //   s(0,1) = s(1,0)
    //   ...
    //   s(0,30) = s(30,0)
    //   s(0,31) = s(30,1) = s(31,0) ^ s(0,0) ^ s(1,0) ^ s(2,0) ^ s(3,0)
    // The above gives us all we need to compute res. The first 30 bits are
    // directly copied from state and bit 31 is computed as above for s(0,31).
    //
    // For state, first introduce the notation sx(b1:b2,i), which is
    //   sx(b1:b2,i) = s(b1,i) ^ s(b1+1,i) ^ ... ^ s(b2-1,i) ^ s(b2,i)
    // This lets us write, e.g., s(0,31) as s(31,0) ^ sx(0:3,0).
    //
    // Thus, we have
    //   s(31,32) = 0
    //   s(30,32) = s(31,31) ^ sx(0:3,31) = s(30,1) ^ s(30,2) ^ s(30,3) ^ s(30,4)
    //            = s(31,0) ^ sx(0:3,0) ^ sx(0:3,1) ^ sx(0:3,2) ^ sx(0:3,3)
    //            = s(31,0) ^ sx(0:3,0) ^ sx(0:3,1) ^ sx(0:3,2) ^ sx(0:3,3)
    //            = s(31,0) ^ sx(0:3,0) ^ sx(1:4,0) ^ sx(2:5,0) ^ sx(3:6,0)
    //   s(29,32) = s(30,31) = s(31,30) ^ sx(0:3,30)
    //            = s(30,0) ^ s(30,1) ^ s(30,2) ^ s(30,3)
    //            = sx(30:31,0) ^ sx(0:3,0) ^ sx(1:4,0) ^ sx(2:5,0)
    //   s(28,32) = s(30,30) = sx(29:31,0) ^ sx(0:3,0) ^ sx(1:4,0)
    //   s(27,32) = s(30,29) = sx(28:31,0) ^ sx(0:3,0)
    //   s(26,32) = s(30,28) = sx(27:30,0)
    //   ...
    //   s(0,32) = sx(1:4,0)
    // For s(:,32), the first 27 bits are just the XOR of the four bits "above" the
    // output bit in index. For bits 27-30, we additionally need to account for
    // the impact of the feedback.

    const uint32_t s_0_1_2_3 = (state) ^ ((state >> 1)) ^ ((state >> 2)) ^ (state >> 3);
    const uint32_t s_0_1_2_3_31 = s_0_1_2_3 ^ (state >> 31);
    const uint32_t res = (state & 0x7FFFFFFF) | ((s_0_1_2_3_31 & 0x1) << 31);

    state =
        // Bits 0-26
        ((s_0_1_2_3 >> 1) & 0x7FFFFFF) |
        // Bit 27
        (((s_0_1_2_3 & 0x10000000) >> 1) ^ ((s_0_1_2_3 & 0x1) << 27)) |
        // Bit 28
        (((s_0_1_2_3 & 0x20000000) >> 1) ^ ((s_0_1_2_3 & 0x1) << 28) ^ ((s_0_1_2_3 & 0x2) << 27)) |
        // Bit 29
        (((s_0_1_2_3 & 0x40000000) >> 1) ^ ((s_0_1_2_3 & 0x1) << 29) ^ ((s_0_1_2_3 & 0x2) << 28) ^ ((s_0_1_2_3 & 0x4) << 27)) |
        // Bit 30
        (((s_0_1_2_3 & 0x80000000) >> 1) ^ ((s_0_1_2_3 & 0x1) << 30) ^ ((s_0_1_2_3 & 0x2) << 29) ^ ((s_0_1_2_3 & 0x4) << 28) ^ ((s_0_1_2_3 & 0x8) << 27));
        // Bit 31 is 0

    return res;
}

// Fibonacci LFSR for second polynomial for Gold sequence generation
CUDA_BOTH inline uint32_t fibonacciLFSR2_1bit(uint32_t& state)
{
    uint32_t res = state;
    // x^{31} + x^3 + x^2 + x + 1
    uint32_t bit = (state) ^ ((state >> 1)) ^ ((state >> 2)) ^ (state >> 3);
    bit          = bit & 1;
    state >>= 1;
    state ^= (bit << 30);
    res ^= (state >> 30) << 31;
    return res;
}
// Fibonacci LFSR for first polynomial for Gold sequence generation
// inverted output
CUDA_BOTH inline uint32_t fibonacciLFSR1(uint32_t& state, uint32_t n)
{
    uint32_t res = 0;

    // x^{31} + x^3 + 1
    for(int i = 0; i < n; i++)
    {
        uint32_t bit = (state) ^ (state >> 3);
        bit          = bit & 1;
        res >>= 1;
        res ^= (state & 1) << 31; //(% 32);

        state >>= 1;
        state ^= (bit << 30);
    }
    return res;
}

// Fibonacci LFSR for first polynomial for Gold sequence generation
// inverted output. This version is specialized to only generate
// 32 output bits.
CUDA_BOTH inline uint32_t fibonacciLFSR1_n32(uint32_t& state)
{
    // See the comment block in fibonacciLFSR2_n32() for an overview of how the recurrence
    // relations yield the closed form expressions below. The result is basically identical,
    // except the polynomial here is x^{31} + x^3 + 1 versus x^{31} + x^3 + x^2 + x + 1, so
    // there are fewer terms in the recurrence relations.
    // The recurrence relation for s(b,i) is:
    //
    //              /  0                                               b = 31
    //   s(b,i+1) = |  s(31,i) ^ s(0,i) ^ s(3,i)                       b = 30
    //              \  s(b+1,i)                                        otherwise
    //
    // Thus, we have
    //   s(31,32) = 0
    //   s(30,32) = s(31,31) ^ s(0,31) ^ s(3,31) = s(30,1) ^ s(30,4)
    //            = s(31,0) ^ s(0,0) ^ s(3,0) ^ s(31,3) ^ s(0,3) ^ s(3,3)
    //            = s(31,0) ^ s(0,0) ^ s(3,0) ^ s(3,0) ^ s(6,0)
    //            = s(31,0) ^ s(0,0) ^ s(6,0)
    //   s(29,32) = s(30,31) = s(30,0) ^ s(2,0) ^ s(5,0)
    //   s(28,32) = s(30,30) = s(29,0) ^ s(1,0) ^ s(4,0)
    //   s(27,32) = s(30,29) = s(28,0) ^ s(31,0) ^ s(0,0) ^ s(3,0)
    //   s(26,32) = s(30,28) = s(27,0) ^ s(30,0)
    //   ...
    //   s(0,32) = s(1,0) ^ s(4,0)
    const uint32_t s_0_3 = (state) ^ (state >> 3);
    const uint32_t s_0_3_31 = s_0_3 ^ (state >> 31);
    const uint32_t res = (state & 0x7FFFFFFF) | ((s_0_3_31 & 0x1) << 31);

    state =
        // Bits 0-26
        ((s_0_3 >> 1) & 0x7FFFFFF) |
        // Bit 27
        (((s_0_3 & 0x10000000) >> 1) ^ ((s_0_3 & 0x1) << 27)) |
        // Bit 28
        (((s_0_3 & 0x20000000) >> 1) ^ ((s_0_3 & 0x2) << 27)) |
        // Bit 29
        (((s_0_3 & 0x40000000) >> 1) ^ ((s_0_3 & 0x4) << 27)) |
        // Bit 30
        (((s_0_3 & 0x80000000) >> 1) ^ ((s_0_3 & 0x1) << 30) ^ ((s_0_3 & 0x8) << 27));
        // Bit 31 is 0

    return res;
}

// Galois LFSR for Gold sequence generation
// computes between 1 and 32 bits, n must be at most 32
CUDA_BOTH inline uint32_t galois31LFSRWord(uint32_t state, uint32_t galoisMask, uint32_t n = 31)
{
    uint32_t res = 0;

    uint32_t msbMask = (1 << 30);
    uint32_t bit;
    uint32_t pred;
#pragma unroll
    for(int i = 0; i < n; i++)
    {
        bit  = (msbMask & state);
        pred = bit != 0;
        state <<= 1;
        state ^= pred * galoisMask;
        res ^= pred << i;
    }
    return res;
}

// Galois LFSR for Gold sequence generation
// Computes 31 bits using the galois mask 0xF. Note that this is a device-only function due
// to the use of __brev().
#ifdef __CUDACC__
__device__ inline uint32_t galois31MaskLFSRWord(uint32_t state)
{
    // We start with the following recurrence relations for the state (s) and result (r):
    //              /  s(b-1,i)                                        b > 3
    //   s(b,i+1) = |  s(b-1,i) ^ s(30,i)                              b = 1, 2, 3
    //              \  s(30,i)                                         b = 0
    //
    //              /  r(b,i)                                          b != i
    //   r(b,i+1) = |
    //              \  s(30,i)                                         otherwise
    // The second branch of r does not include r(i,0) because r(i,0) is zero for all i.
    // We do not actually need to compute the full new state, but we will need to solve
    // for some state bits to fully compute the result.
    //
    // The bits of the result (res) are thus as follows:
    //   r(0,32) = r(0,31) = ... = r(0,1) = s(30,0)
    //   r(1,32) = r(1,2) = s(30,1) = s(29,0)
    //   r(2,32) = r(2,3) = s(30,2) = s(28,0)
    //   ...
    //   r(27,32) = r(27,28) = s(30,27) = s(3,0)
    //   r(28,32) = r(28,29) = s(30,28) = s(3,1) = s(2,0) ^ s(30,0)
    //   r(29,32) = r(29,30) = s(30,29) = s(3,2) = s(2,1) ^ s(30,1) = s(1,0) ^ s(30,0) ^ s(29,0)
    //   r(30,32) = r(30,31) = s(30,30) = s(3,3) = s(2,2) ^ s(30,2) = s(1,1) ^ s(30,1) ^ s(30,2)
    //            = s(0,0) ^ s(30,0) ^ s(29,0) ^ s(28,0)
    //   r(31,32) = 0 because we only generate 31 output bits (n=31)
    const uint32_t rev_state = __brev(state);
    const uint32_t res =
        ((rev_state >> 1) & 0xFFFFFFF) |
        // bit 28 - s(2,0) ^ s(30,0)
        (((state & 0x4) << 26) ^ ((state & 0x40000000) >> 2)) |
        // bit 29 - s(1,0) ^ s(30,0) ^ s(29,0)
        (((state & 0x2) << 28) ^ ((state & 0x40000000) >> 1) ^ (state & 0x20000000)) |
        // bit 30 - s(0,0) ^ s(30,0) ^ s(29,0) ^ s(28,0)
        (((state & 0x1) << 30) ^ ((state & 0x40000000) >> 0) ^ ((state & 0x20000000) << 1) ^ ((state & 0x10000000) << 2));
        // bit 31 is 0
    return res;
}
#endif // __CUDACC__

// Mutiply by POLY_B  = 0x8000000F
CUDA_BOTH inline uint32_t polyBMulHigh31(uint32_t a)
{
    uint32_t prodHi = (a >> 30) ^ (a >> 29) ^ (a >> 28) ^ a;
    return prodHi;
}

CUDA_BOTH inline uint32_t polyMulHigh31(uint32_t a, uint32_t b)
{
    uint32_t prodHi = 0;
#pragma unroll
    for(int i = 1; i < 32; i++)
    {
        uint32_t pred = ((b >> i) & 1);
        prodHi ^= (pred * a) >> (31 - i);
    }
    return prodHi;
}
/*
CUDA_BOTH inline uint32_t mulModPoly31(uint32_t a,
                                       uint32_t pow,
                                       uint32_t poly)
{
    // a moduloe POLY_2, 31 BITs
    uint32_t crc = a ^ (a >= poly) * poly;
    uint32_t r = 1;
    uint32_t y = crc;
    while(pow > 1)
    {
        if(pow & 1)
            r = mulModPoly<uint32_t, 31>(r, y, poly);
        y = mulModPoly<uint32_t, 31>(y, y, poly);
        pow >>= 1;
    }

    return mulModPoly<uint32_t, 31>(r, y, poly);
}
*/

CUDA_BOTH inline uint32_t mulModPoly31LUT(uint32_t a,
                                          uint32_t b,
                                          uint32_t poly)
{
#if CUPHY_CLMAD_AVAILABLE
    if(poly == POLY_2)
    {
        cuphy_clmad::u64_u32x2 abX;
        abX.u64 = cuphy_clmad::clmul_lo(static_cast<uint64_t>(a),
                                        static_cast<uint64_t>(b));

        const uint32_t pX = abX.u32[0] & 0x7FFFFFFFu;
        const uint32_t cX = __funnelshift_rc(abX.u32[0], abX.u32[1], 31);
        const uint32_t resA =
            cuphy_clmad::opt_reduction<31,
                                        31,
                                        cuphy_clmad::qplusX_gold31,
                                        cuphy_clmad::gastrX_gold31>(cX);

        return resA ^ pX;
    }
#endif

    uint32_t prod = 0;
    // a moduloe POLY_2, 31 BITs
    uint32_t crc = a ^ (a >= POLY_2) * POLY_2;
#pragma unroll
    for(int i = 0; i < 31; i++)
    {
        prod ^= (crc & 1) * b;
        b = (b << 1) ^ (b & (1 << (30)) ? poly : 0);
        crc >>= 1;
    }

    return prod;
}

// Little-endian 31-bit Modular GF2 polynomial multiplication by monomials
// using coalesced precomputed x^{32i}, x^{32i +8}, x^{32i + 16}, x^{32i + 24}
// values
CUDA_BOTH inline uint32_t mulModPoly31_Coalesced(const uint32_t  a,
                                                 const uint32_t* table,
                                                 uint32_t        tableWordOffset,
                                                 uint32_t        poly)
{
    uint32_t     prod    = 0;
    uint32_t     msbMask = (1UL << 31);
    unsigned int offset  = 0;

#pragma unroll
    for(int bitsProcessed = 0; bitsProcessed < sizeof(uint32_t) * 8; bitsProcessed += BITS_PROCESSED_PER_LUT_ENTRY)
    {
        uint32_t inputByte = a >> (bitsProcessed)&BITS_PROCESSED_PER_LUT_ENTRY_MASK;
        for(unsigned bit = 0; bit < BITS_PROCESSED_PER_LUT_ENTRY; bit++)
        {
            uint32_t pred  = ((inputByte >> (bit)) & 1);
            uint32_t pprod = table[(offset)] * pred;
            for(unsigned shift = 0; shift < bit; shift++)
            {
                pprod <<= 1;
                uint32_t pred = (pprod & msbMask) == 0;
                pprod ^= (poly * pred);
            }
            prod ^= pprod;
        }
        offset += tableWordOffset;
    }

    return prod;
}
#ifdef __CUDACC__
// Compute 32 bits of the Gold sequence starting from bit floor(n / 32)
__device__ inline uint32_t gold32(uint32_t seed2, uint32_t n)
{
    uint32_t prod2;
    uint32_t output1 = GOLD_1_SEQ_LUT[n / WORD_SIZE];

    //    uint32_t state1 = 0x40000000;         // reverse of 0x1
    uint32_t state2 = __brev(seed2) >> 1; // reverse 31 bits

    //state2 = polyMulHigh31(state2, POLY_2);
    state2 = polyBMulHigh31(state2);
    prod2  = mulModPoly31LUT(state2,
                            GOLD_2_32_P_LUT[(n) / WORD_SIZE],
                            POLY_2);

    uint32_t fstate2 = galois31MaskLFSRWord(prod2);

    uint32_t output2 = fibonacciLFSR2_1bit(fstate2);

    //    return output1 ^ output2;
    return output1 ^ output2;
}

// Compute 32 bits of the Gold sequence starting from bit n
__device__ inline uint32_t gold32n(uint32_t seed2, uint32_t n)
{
    uint32_t prod2;

    //    uint32_t state1 = 0x40000000;         // reverse of 0x1
    uint32_t state2 = __brev(seed2) >> 1; // reverse 31 bits
    uint32_t fstate1 = GOLD_1_SEQ_LUT[n / WORD_SIZE] & 0x7FFFFFFF;

    //state2 = polyMulHigh31(state2, POLY_2);
    state2 = polyBMulHigh31(state2);
    
    prod2  = mulModPoly31LUT(state2,
                            GOLD_2_32_P_LUT[(n) / WORD_SIZE],
                            POLY_2);

    uint32_t fstate2 = galois31MaskLFSRWord(prod2);

    uint32_t output2 = fibonacciLFSR2_n32(fstate2);
    output2          = (output2 >> (n % 32));
    output2 |= (n % 32) ? (fstate2 << (32 - (n % 32))) : 0;

    uint32_t seq1f   = fibonacciLFSR1_n32(fstate1);
    seq1f            = (seq1f >> (n % 32));
    seq1f |= (n % 32) ? (fstate1 << (32 - (n % 32))) : 0;

    return seq1f ^ output2;
}

#endif
} // namespace descrambling

#endif
