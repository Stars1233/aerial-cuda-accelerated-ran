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

#if !defined(LDPC2_C2V_X2_CUH_INCLUDED_)
#define LDPC2_C2V_X2_CUH_INCLUDED_

// Classes/structs for LDPC min-sum decode kernels that work with a pair
// of codewords per CTA.

#include "ldpc2.cuh"
#include "ldpc2_c2v.cuh"

namespace ldpc2
{

////////////////////////////////////////////////////////////////////////
// LDPC_MICRO_PACK_ARGMIN: master toggle for the x2 packed-argmin min-sum
// scan and the FMA-based leave-one-out expansion reformulation. Default 0:
// every TU that does NOT explicitly set it sees the VERBATIM baseline code
// paths (init_row/update/finalize/get_storage/extract_pair) -- so the other
// x2 min-sum decoders (algos 35/39/51/52/55/56) are byte-for-byte unchanged.
// A TU that wants the packed scan (the BG1/Z384/mb4 ASLO_E5M5 kernel) sets it
// to 1 BEFORE including this header, which also enables the sub-toggles
// (LDPC_MICRO_ABS_MASK_CONST / _PARTIAL_SIGN / _PREDIFF / _SPLIT_REDUCE and
// LDPC2_X2_FUSE_NORM). The packed scan truncates min-sum magnitudes to 5
// mantissa bits (E5M5), so it is intentionally opt-in and isolated.
#ifndef LDPC_MICRO_PACK_ARGMIN
#define LDPC_MICRO_PACK_ARGMIN 0
#endif

#if LDPC_MICRO_PACK_ARGMIN
#if defined(LDPC_MICRO_ABS_MASK_CONST) && LDPC_MICRO_ABS_MASK_CONST
////////////////////////////////////////////////////////////////////////
// (A) Abs+truncate mask (0x7FE0 per fp16 lane) for the x2 packed-magnitude
// min-sum scan. Held in CONSTANT memory on purpose: the front-end (cicc)
// constant-folds a plain immediate -- and even a `mov reg,0x7FE07FE0`
// inline-asm -- straight back into every OR-fused packing LOP3, so ptxas
// re-stages it with a fresh `MOV R,0x7FE07FE0` on the binding integer pipe
// ~60 times per kernel. A constant-memory value is opaque to cicc (the host
// may rewrite it), so it is loaded ONCE and held resident across the unrolled
// scan, removing those re-stagings from the integer pipe.
// [[maybe_unused]]: not every TU that includes this header instantiates the
// x2 APP-based row-context constructor (-Werror=all-warnings is on).
[[maybe_unused]] static __constant__ uint32_t g_ldpc2_x2_abs_mask = 0x7FE07FE0u;
#endif

////////////////////////////////////////////////////////////////////////
// LDPC_MICRO_XOR_SIGNPROD: shallow XOR-tree row sign product for the
// high-degree (>=16) packed-argmin core scan. Only meaningful when
// LDPC_MICRO_PACK_ARGMIN (and the split-reduce core branch) is enabled;
// default 0 so every other TU sees the verbatim serial-OR + POPC path.
//
// Baseline product path: the per-column sign bits are accumulated into the
// two packed sign words through a SERIAL inline-asm OR chain (~9 chained
// LOP3s per word), then folded (XOR), masked, and run through two POPCs
// whose parity becomes the row sign product -- a long dependency chain that
// gates finalize() -> normalize -> (min1-min0) -> every per-column
// reconstruct/APP-add of the degree-19 core row.
//
// This toggle computes the product directly as parity-of-signs over the RAW
// APP words: sprod = (app[0] ^ app[1] ^ ... ^ app[D-1]) & 0x80008000.
// Parity is linear over GF(2), so bit 15 (CW0) / bit 31 (CW1) of the XOR
// fold equal the popcount parities of the packed sign words for any input:
// BIT-IDENTICAL to signs_pair_t::sign_product_mask(). A balanced XOR tree is
// ~4 LOP3 levels (ptxas merges pairs into 3-input LOP3.LUT 0x96) issued
// straight off the APP loads, so the product is ready long before the
// VIMNMX min/2nd-min reduction it joins in finalize -- the entire serial
// sign-accumulate + POPC product chain leaves the critical path.
//
// The packed sign words are still built (the compressed storage and the
// next iteration's subtract pass need them), but through balanced OR trees
// (OR is associative/commutative -- same final words) that only feed the
// get_storage() pack, off the row's exposed latency chain.
#ifndef LDPC_MICRO_XOR_SIGNPROD
#define LDPC_MICRO_XOR_SIGNPROD 0
#endif

////////////////////////////////////////////////////////////////////////
// Measured and rejected here (branches removed; see the internal report):
//
//   * folding the per-column sign application into the APP accumulate as a
//     multiply by an exact +/-1.0 fp16x2 selector (was LDPC_MICRO_PM1_APPLY).
//     The fp16 FMA pipe is the top-utilized pipeline in the kernels that use
//     this header, so folding the sign into it starves the bottleneck.
//
//   * packed-integer leave-one-out reconstruction with raw (unnormalized)
//     compressed storage and consume-side normalization (was
//     LDPC_MICRO_RAW_SELECT). It moves work onto the INT/ALU datapath that
//     already hosts the scan and the sign bookkeeping, which on GB203 is the
//     scarce resource.
//
//   * running the packed-argmin running-min/2nd-min scan on the fp16 pipe
//     (HMNMX2) instead of the integer pipe (VIMNMX) (was
//     LDPC_MICRO_FP_MINMAX). Bit-identical -- the packed patterns are
//     positive finite fp16 so IEEE ordering matches unsigned integer
//     ordering -- but it lengthens the exposed serial min/2nd-min chain.

// The packed-argmin scan's min/max primitives. These operate on the packed
// (|app| & 0x7FE0) | colindex words, which are positive finite fp16 bit
// patterns, so unsigned integer ordering is the correct ordering.
#define LDPC2_X2_SCAN_MINU(a, b) __vminu2((a), (b))
#define LDPC2_X2_SCAN_MAXU(a, b) __vmaxu2((a), (b))

////////////////////////////////////////////////////////////////////////
// LDPC2_X2_FUSE_NORM: where the min-sum normalization (params.norm) is
// applied. Only meaningful when LDPC_MICRO_PACK_ARGMIN is enabled.
//   =1 (default): min0/min1 are kept UN-normalized through finalize(); norm
//      is folded into the per-column APP update as __hfma2(inc, norm, app),
//      and get_storage(norm) applies it once when packing the stored message.
//   =0 (normalize-once): min0/min1 are normalized ONCE in finalize(), so the
//      single (normalized) min1m0 serves BOTH this iteration's extract and the
//      stored delta -- get_storage() just packs and the per-column APP update
//      becomes a plain __hadd2.
// A TU sets it to 0 BEFORE including this header to scope the change.
#ifndef LDPC2_X2_FUSE_NORM
#define LDPC2_X2_FUSE_NORM 1
#endif

////////////////////////////////////////////////////////////////////////
// app_combine_x2()
// Accumulate the extracted (already sign-applied) C2V message `inc` into the
// APP value `app`. With FUSE_NORM the extracted message is un-normalized, so
// this is a fused multiply-add by norm; with FUSE_NORM==0 the message is
// already normalized (finalize() scaled min0/min1), so it is a plain add.
// [[nodiscard]]: the returned value IS the update -- there is no output
// parameter and no side effect, so a discarded call silently drops the
// accumulate for that column.
[[nodiscard]] __device__ __forceinline__
word_t app_combine_x2(word_t app, word_t inc, const __half2& norm)
{
    word_t r;
#if LDPC2_X2_FUSE_NORM
    r.f16x2 = __hfma2(inc.f16x2, norm, app.f16x2);
#else
    (void)norm;
    r.f16x2 = __hadd2(app.f16x2, inc.f16x2);
#endif
    return r;
}
#endif // LDPC_MICRO_PACK_ARGMIN

////////////////////////////////////////////////////////////////////////
// signs_pair_high_degree_split
// Sign bit storage and management for "2 codewords at a time", for high
// degree (greater than 10) rows. Internally, signs for each codeword
// are "split" (i.e. stored in two different words). This helps when
// using the same data structure for all rows - the compiler can
// (hopefully) determine that the second word is not used for lower
// degree rows.
//
//  Bit
//  31 30 29 28 27 26 25 24 23 22 21 20 19 18 17 16   15 14 13 12 11 10  9  8  7  6  5  4  3  2  1  0
//
// The first 10 bits of the stored signs will be in the first signs word,
// ordered as follows:
//
//  Word 1 Sign Bits                                  Word 0 Sign Bits
//  ------------------------------------------------|------------------------------------------------
//   9  8  7  6  5  4  3  2  1  0  .  .  .  .  .  . |  9  8  7  6  5  4  3  2  1  0  .  .  .  .  .  .
//                              ^                                                 ^
//                   Row Index  |                                      Row Index  |
//
//  Word 1 Sign Bits                                  Word 0 Sign Bits
//  ------------------------------------------------|------------------------------------------------
//  18 17 16 15 14 13 12 11 10  .  .  .  .  .  .  . | 18 17 16 15 14 13 12 11 10  .  .  .  .  .  .  .
//        ^                                                  ^
//        | Row Index                                        |  Row Index
struct signs_pair_high_degree_split
{
    //------------------------------------------------------------------
    // Data
    uint32_t signs_0_9;
    uint32_t signs_10_;
    //------------------------------------------------------------------
    // signs_pair_high_degree_split()
    signs_pair_high_degree_split() = default;
    //------------------------------------------------------------------
    // signs_pair_high_degree_split()
    __device__
    signs_pair_high_degree_split(uint32_t s0_9, uint32_t s10_) :
        signs_0_9(s0_9),
        signs_10_(s10_)
    {
    }
    //------------------------------------------------------------------
    // init_row()
    // Initialize the internal sign bit representation with a pair of
    // words.
    __device__
    void init_row(word_t v0, word_t v1)
    {
        word_t s0 = fp16x2_sign_mask(v0);
        word_t s1 = fp16x2_sign_mask(v1);
        signs_0_9 = (s0.u32 >> 9) | (s1.u32 >> 8);
        signs_10_ = 0;
    }
    //------------------------------------------------------------------
    // update()
    // Update the internal sign bit representation for two individual
    // codewords with the contents of the high and low halves of 'v'.
    __device__
    void update(word_t v, int idx)
    {
        word_t sv = fp16x2_sign_mask(v);
        if(idx < 10)
        {
            uint32_t shifted = sv.u32 >> (9-idx);
            asm("or.b32 %0, %0, %1;" : "+r"(signs_0_9) : "r"(shifted));
        }
        else
        {
            uint32_t shifted = sv.u32 >> (18-idx);
            asm("or.b32 %0, %0, %1;" : "+r"(signs_10_) : "r"(shifted));
        }
    }
    //------------------------------------------------------------------
    // sign_product_mask()
    // Returns a value with bits 15 and 31 set or cleared, based on the
    // product of the internal stored signs for each of the two
    // codewords represented. All other bits will be zero.
    __device__
    uint32_t sign_product_mask() const
    {
        // The per-codeword sign product is the PARITY of that codeword's
        // column sign bits. The 19 sign bits per codeword are split across
        // signs_0_9 (cols 0-9) and signs_10_ (cols 10-18). The original form
        // shifted the two words into disjoint fields so a single popcount
        // could span both. But parity is linear over GF(2):
        //   parity(A ^ B) == parity(A) ^ parity(B)   (the carry/overlap term
        // vanishes mod 2), so we can fold the two words with ONE xor and let
        // each codeword's region overlap -- the popcount-parity is unchanged.
        // This drops the shift+merge gather to a single xor ahead of the two
        // hardware POPCs, shortening the finalize->min0-sign->extract_pair
        // dependency chain. Bit-identical to the disjoint-gather form: the
        // signs_10_ scratch bits [0:6] and [16:22] are never written (only
        // cols 10-18 land at bits 7-15 / 23-31) and init to 0, so the xor
        // never folds a stray set bit into either codeword's parity region.
        const uint32_t folded = signs_0_9 ^ signs_10_;
        // CW1 region is bits 22-31; >>22 isolates it with no mask (low bits
        // shift out), so its popcount needs only a shift, not an AND.
        const uint32_t CW0_pop = (unsigned int)__popc(folded & 0x0000FFC0u);
        const uint32_t CW1_pop = (unsigned int)__popc(folded >> 22);
        return (((CW0_pop << 15) & 0x00008000u) | (CW1_pop << 31));
    }
    //------------------------------------------------------------------
    // sign_mask()
    // Returns a mask with the appropriate sign bits set at bits 15 and
    // 31, based on the stored sign bits for the given index pair.
    __device__
    uint32_t sign_mask(int idx) const
    {
        const uint32_t smask = (idx < 10) ? (signs_0_9 << (9-idx)) : (signs_10_ << (18-idx));
        return (smask & 0x80008000);
    }
    //------------------------------------------------------------------
    // finalize()
    __device__
    void finalize()
    {
    }
};

////////////////////////////////////////////////////////////////////////
// signs_pair_high_degree_split_fp
// Sign bit storage and management for "2 codewords at a time", for high
// degree (greater than 10) rows. Internally, signs for each codeword
// are "split" (i.e. stored in two different words). This helps when
// using the same data structure for all rows - the compiler can
// (hopefully) determine that the second word is not used for lower
// degree rows.
// This structure uses floatiing point (fp) operations to update sign
// bits, instead of bitwise logical operations.
//
//  Bit
//  31 30 29 28 27 26 25 24 23 22 21 20 19 18 17 16   15 14 13 12 11 10  9  8  7  6  5  4  3  2  1  0
//
// The first 10 bits of the stored signs will be in the first signs word,
// ordered as follows:
//
//  Word 1 Sign Bits                                  Word 0 Sign Bits
//  ------------------------------------------------|------------------------------------------------
//   9  8  7  6  5  4  3  2  1  0  .  .  .  .  .  . |  9  8  7  6  5  4  3  2  1  0  .  .  .  .  .  .
//                              ^                                                 ^
//                   Row Index  |                                      Row Index  |
//
//  Word 1 Sign Bits                                  Word 0 Sign Bits
//  ------------------------------------------------|------------------------------------------------
//  18 17 16 15 14 13 12 11 10  .  .  .  .  .  .  . | 18 17 16 15 14 13 12 11 10  .  .  .  .  .  .  .
//        ^                                                  ^
//        | Row Index                                        |  Row Index
struct signs_pair_high_degree_split_fp
{
    //------------------------------------------------------------------
    // Data
    uint32_t signs_0_9;
    uint32_t signs_10_;
    //------------------------------------------------------------------
    // signs_pair_high_degree_split_fp()
    signs_pair_high_degree_split_fp() = default;
    //------------------------------------------------------------------
    // signs_pair_high_degree_split_fp()
    __device__
    signs_pair_high_degree_split_fp(uint32_t s0_9, uint32_t s10_) :
        signs_0_9(s0_9),
        signs_10_(s10_)
    {
    }
    //------------------------------------------------------------------
    // init_row()
    // Initialize the internal sign bit representation with a pair of
    // words.
    __device__
    void init_row(word_t v0, word_t v1)
    {
        signs_0_9 = update_signs_fp_pair(0, v0, 0);
        signs_0_9 = update_signs_fp_pair(signs_0_9, v1, 1);
        signs_10_ = 0;
    }
    //------------------------------------------------------------------
    // update()
    // Update the internal sign bit representation for two individual
    // codewords with the contents of the high and low halves of 'v'.
    __device__
    void update(word_t v, int idx)
    {
        if(idx < 10) // compile time branch with unrolled loops...
        {
            signs_0_9 = update_signs_fp_pair(signs_0_9, v, idx);
        }
        else
        {
            signs_10_ = update_signs_fp_pair(signs_10_, v, idx - 10);
        }
    }
    //------------------------------------------------------------------
    // sign_product_mask()
    // Returns a value with bits 15 and 31 set or cleared, based on the
    // product of the internal stored signs for each of the two
    // codewords represented. All other bits will be zero.
    __device__
    uint32_t sign_product_mask() const
    {
        // Gather signs from 0-9 and 10+
        const uint32_t CW0_signs = (signs_0_9 & 0x0000FFC0) | (signs_10_ << 16);
        const uint32_t CW1_signs = (signs_0_9 >> 16)        | (signs_10_ & 0xFFC00000);
        const uint32_t CW0_sign_prod = (unsigned int)__popc(CW0_signs) & 0x1;
        const uint32_t CW1_sign_prod = (unsigned int)__popc(CW1_signs) & 0x1;

        return (CW0_sign_prod << 15) | (CW1_sign_prod << 31);
    }
    //------------------------------------------------------------------
    // sign_mask()
    // Returns a mask with the appropriate sign bits set at bits 15 and
    // 31, based on the stored sign bits for the given index pair.
    __device__
    uint32_t sign_mask(int idx) const
    {
        const uint32_t smask = (idx < 10) ? (signs_0_9 << (9-idx)) : (signs_10_ << (18-idx));
        return (smask & 0x80008000);
    }
    //------------------------------------------------------------------
    // finalize()
    __device__
    void finalize()
    {
        // Shift left so that signs bits for index 0 are at bit 6.
        signs_0_9 <<= 6;
        // Shift left so that sign bits for index 10 are at bit 7.
        signs_10_ <<= 7;
    }
};

////////////////////////////////////////////////////////////////////////
// signs_pair_low_degree
// Sign bit storage and management for "2 codewords at a time", for low
// degree (less than or equal to 10) rows. Internally, signs are stored
// in a single word.
//
//  Bit
//  31 30 29 28 27 26 25 24 23 22 21 20 19 18 17 16   15 14 13 12 11 10  9  8  7  6  5  4  3  2  1  0
//
// Up to 10 bits of the stored signs will be stored in a single word,
// ordered as follows:
//
//  Word 1 Sign Bits                                  Word 0 Sign Bits
//  ------------------------------------------------|------------------------------------------------
//   9  8  7  6  5  4  3  2  1  0  .  .  .  .  .  . |  9  8  7  6  5  4  3  2  1  0  .  .  .  .  .  .
//                              ^                                                 ^
//                   Row Index  |                                      Row Index  |
struct signs_pair_low_degree
{
    //------------------------------------------------------------------
    // Data
    uint32_t signs_0_9;
    //------------------------------------------------------------------
    // signs_pair_low_degree()
    signs_pair_low_degree() = default;
    //------------------------------------------------------------------
    // signs_pair_low_degree()
    __device__
    signs_pair_low_degree(uint32_t u) : signs_0_9(u) {}
    //------------------------------------------------------------------
    // init_row()
    // Initialize the internal sign bit representation with a pair of
    // words.
    __device__
    void init_row(word_t v0, word_t v1)
    {
        word_t s0 = fp16x2_sign_mask(v0);
        word_t s1 = fp16x2_sign_mask(v1);
        signs_0_9 = (s0.u32 >> 9) | (s1.u32 >> 8);
    }
    //------------------------------------------------------------------
    // update()
    // Update the internal sign bit representation for two individual
    // codewords with the contents of the high and low halves of 'v'.
    __device__
    void update(word_t v, int idx)
    {
        word_t sv = fp16x2_sign_mask(v);
        uint32_t shifted = sv.u32 >> (9-idx);
        asm("or.b32 %0, %0, %1;" : "+r"(signs_0_9) : "r"(shifted));
    }
    //------------------------------------------------------------------
    // sign_product_mask()
    // Returns a value with bits 15 and 31 set or cleared, based on the
    // product of the internal stored signs for each of the two
    // codewords represented. All other bits will be zero.
    __device__
    uint32_t sign_product_mask() const
    {
        const uint32_t CW0_sign_prod = (unsigned int)__popc(signs_0_9 & 0x0000FFC0) & 0x1;
        const uint32_t CW1_sign_prod = (unsigned int)__popc(signs_0_9 & 0xFFC00000) & 0x1;
        return (CW0_sign_prod << 15) | (CW1_sign_prod << 31);
    }
    //------------------------------------------------------------------
    // sign_mask()
    // Returns a mask with the appropriate sign bits set at bits 15 and
    // 31, based on the stored sign bits for the given index pair.
    __device__
    uint32_t sign_mask(int idx) const
    {
        const uint32_t smask = (signs_0_9 << (9-idx));
        return (smask & 0x80008000);
    }
    //------------------------------------------------------------------
    // finalize()
    __device__
    void finalize()
    {
    }
};

////////////////////////////////////////////////////////////////////////
// signs_pair_low_degree_fp
// Sign bit storage and management for "2 codewords at a time", for low
// degree (less than or equal to 10) rows. Internally, signs are stored
// in a single word. The signs storage update uses floating point (fp)
// operations instead of bitwise logical operations.
//
//  Bit
//  31 30 29 28 27 26 25 24 23 22 21 20 19 18 17 16   15 14 13 12 11 10  9  8  7  6  5  4  3  2  1  0
//
// Up to 10 bits of the stored signs will be stored in a single word,
// ordered as follows:
//
//  Word 1 Sign Bits                                  Word 0 Sign Bits
//  ------------------------------------------------|------------------------------------------------
//   9  8  7  6  5  4  3  2  1  0  .  .  .  .  .  . |  9  8  7  6  5  4  3  2  1  0  .  .  .  .  .  .
//                              ^                                                 ^
//                   Row Index  |                                      Row Index  |
struct signs_pair_low_degree_fp
{
    //------------------------------------------------------------------
    // Data
    uint32_t signs_0_9;
    //------------------------------------------------------------------
    // signs_pair_low_degree_fp()
    signs_pair_low_degree_fp() = default;
    //------------------------------------------------------------------
    // signs_pair_low_degree_fp()
    __device__
    signs_pair_low_degree_fp(uint32_t u) : signs_0_9(u) {}
    //------------------------------------------------------------------
    // init_row()
    // Initialize the internal sign bit representation with a pair of
    // words.
    __device__
    void init_row(word_t v0, word_t v1)
    {
        signs_0_9 = update_signs_fp_pair(0, v0, 0);
        signs_0_9 = update_signs_fp_pair(signs_0_9, v1, 1);
    }
    //------------------------------------------------------------------
    // update()
    // Update the internal sign bit representation for two individual
    // codewords with the contents of the high and low halves of 'v'.
    __device__
    void update(word_t v, int idx)
    {
        signs_0_9 = update_signs_fp_pair(signs_0_9, v, idx);
    }
    //------------------------------------------------------------------
    // sign_product_mask()
    // Returns a value with bits 15 and 31 set or cleared, based on the
    // product of the internal stored signs for each of the two
    // codewords represented. All other bits will be zero.
    __device__
    uint32_t sign_product_mask() const
    {
        const uint32_t CW0_sign_prod = (unsigned int)__popc(signs_0_9 & 0x0000FFC0) & 0x1;
        const uint32_t CW1_sign_prod = (unsigned int)__popc(signs_0_9 & 0xFFC00000) & 0x1;
        return (CW0_sign_prod << 15) | (CW1_sign_prod << 31);
    }
    //------------------------------------------------------------------
    // sign_mask()
    // Returns a mask with the appropriate sign bits set at bits 15 and
    // 31, based on the stored sign bits for the given index pair.
    __device__
    uint32_t sign_mask(int idx) const
    {
        const uint32_t smask = (signs_0_9 << (9-idx));
        return (smask & 0x80008000);
    }
    //------------------------------------------------------------------
    // finalize()
    __device__
    void finalize()
    {
        // Shift left such that index 0 appears at bit 6.
        signs_0_9 <<= 6;
    }
};

////////////////////////////////////////////////////////////////////////
// sign_mgr_pair_src
// This class assumes that the signs word will, after the finalize()
// function is called, store the "source" signs (e.g. the sign of the
// input value to the compressed C2V update function). In general,
// this requires that the product of the signs be stored in the min0
// and min1 values, and the output sign can be obtained via the xor
// of the "src" sign and the sign product.
// T: APP type (__half, __half2, ...)
// TClamp: Boolean value to indicate whether the min0 and min1 values
//         are clamped in the finalize() function, to protect against
//         NaN values that can occur when APP values become infinite.
//         If TClamp is false, no clamping is performed, and it is
//         assumed that input values are constrained such that infinite
//         values will not occur (e.g., by clamping values at input).
template <bool TClamp>
struct sign_mgr_pair_src
{
    //------------------------------------------------------------------
    // finalize()
    template <class TSignsPair>
    static
    __device__
    void finalize(const TSignsPair& s_pair,
                  word_t&           min0,
                  word_t&           min1)
    {
        const uint32_t sign_prod_mask = s_pair.sign_product_mask();

        // We will store the product of all signs in the sign bits of
        // min0 and min1. When retrieving the value, we can then take
        // the xor of the desired input sign bit (retrieved from the
        // signs field) with the min0 sign to get the desired output sign.
        min0.u32 = (min0.u32 & 0x7FFF7FFF) | sign_prod_mask;
        min1.u32 = (min1.u32 & 0x7FFF7FFF) | sign_prod_mask;

        if(TClamp)
        {
            // Clamp to +/- FP16_max to avoid subtracting +/-Inf during
            // the next iteration. (-Inf - (-Inf) = NaN, Inf - Inf = NaN.)
            // In this case, min1 and min0 are signed values, so we must
            // clamp to +/- FP16_max.
            min0 = clamp_to_half_max(min0);
            min1 = clamp_to_half_max(min1);
        }
    }
    //------------------------------------------------------------------
    // finalize_mask()
    // As finalize(), but with the row sign-product mask (bits 15/31 only)
    // already computed by the caller (e.g. the LDPC_MICRO_XOR_SIGNPROD
    // XOR-tree product). Same store-side semantics: the product of all
    // signs is written into the sign bits of min0 and min1.
    static
    __device__
    void finalize_mask(uint32_t sign_prod_mask,
                       word_t&  min0,
                       word_t&  min1)
    {
        min0.u32 = (min0.u32 & 0x7FFF7FFF) | sign_prod_mask;
        min1.u32 = (min1.u32 & 0x7FFF7FFF) | sign_prod_mask;

        if(TClamp)
        {
            min0 = clamp_to_half_max(min0);
            min1 = clamp_to_half_max(min1);
        }
    }
    //------------------------------------------------------------------
    // apply_sign()
    template <class TSignsPair>
    static
    __device__
    word_t apply_sign(const TSignsPair& s_pair,
                      word_t            v,
                      int               index)
    {
        word_t out;
        out.u32 = s_pair.sign_mask(index) ^ v.u32;
        return out;
    }
};

////////////////////////////////////////////////////////////////////////
// cC2V_storage_x2_low_degree
// Compressed check-to-variable data storage for 2 codewords at a time
// implementations (rows with low degree only). In this case, "low
// degree" refers to 10 or less.
//
//  Bit
//  31 30 29 28 27 26 25 24 23 22 21 20 19 18 17 16   15 14 13 12 11 10  9  8  7  6  5  4  3  2  1  0
//
// The first 10 bits of the stored signs will be in the first signs word,
// ordered as follows:
//
//  Word 1 Sign Bits                                  Word 0 Sign Bits
//  ------------------------------------------------|------------------------------------------------
//   9  8  7  6  5  4  3  2  1  0  .  .  .  .  .  . |  9  8  7  6  5  4  3  2  1  0  .  .  .  .  .  .
//                              ^                                                 ^
//                   Row Index  |                                      Row Index  |
//
// Index of min0 will be 5 bits or less (max row degree is 19). The min0_index value for each word will
// be stored in the low bits of each half word.
//
//  Word 1 Min0 Index                                 Word 0 Min0 Index
//  ------------------------------------------------|------------------------------------------------
//   .  .  .  .  .  .  .  .  .  .  .  x  x  x  x  x |  .  .  .  .  .  .  .  .  .  .  .  x  x  x  x  x
struct cC2V_storage_x2_low_degree
{
    word_t   min0;
    word_t   min1;
    uint32_t signs_0_9_min0_index;
    //------------------------------------------------------------------
    // Typedef for struct to hold sign bits
    // TODO: Make different kernels with different sign managers to allow
    // performance comparisons.
    // fp sign management currently slightly faster on sm_120.
    // The packed-argmin stack is designed and verified against the bitwise
    // sign store; keep it there.
#if LDPC_MICRO_PACK_ARGMIN
    typedef signs_pair_low_degree signs_pair_t;
#else
    typedef signs_pair_low_degree_fp signs_pair_t;
#endif
    //------------------------------------------------------------------
    // cC2V_storage_x2_low_degree()
    cC2V_storage_x2_low_degree() = default;
    //------------------------------------------------------------------
    // cC2V_storage_x2_low_degree()
    __device__
    cC2V_storage_x2_low_degree(word_t m0,
                               word_t m1,
                               word_t m0_index,
                               const signs_pair_t& s_pair) :
        min0(m0),
        min1(m1),
        signs_0_9_min0_index(s_pair.signs_0_9 | m0_index.u32)
    {
    }
    //------------------------------------------------------------------
    // init()
    __device__
    void init()
    {
        min0.u32 = 0; min1.u32 = 0; signs_0_9_min0_index = 0;
    }
    //------------------------------------------------------------------
    // get_signs_pair()
    __device__
    signs_pair_t get_signs_pair() const
    {
        return signs_pair_t(signs_0_9_min0_index & 0xFFC0FFC0);
    }
    //------------------------------------------------------------------
    // get_min0_index_pair()
    __device__
    word_t get_min0_index_pair() const
    {
        word_t w;
        w.u32 = (signs_0_9_min0_index & 0x001F001F);
        return w;
    }
#if 0
    //------------------------------------------------------------------
    // print()
    __device__
    void print() const
    {
        printf("min1 = [%f %f], min0 = [%f %f], index = [%i %i], signs = [0x%X 0x%X]\n",
               __high2float(min1.f16x2),
               __low2float(min1.f16x2),
               __high2float(min0.f16x2),
               __low2float(min0.f16x2),
               (signs_0_9_min0_index >> 16) & 0x1F,
               (signs_0_9_min0_index >> 0)  & 0x1F,
               (signs_0_9_min0_index >> 22),
               (signs_0_9_min0_index >> 6)  & 0x3FF);
    }
#endif
};

////////////////////////////////////////////////////////////////////////
// cC2V_storage_x2_high_degree_split
// Compressed check-to-variable data storage for 2 codewords at a time
// implementations (rows with high degree only). In this case, "high
// degree" refers to greater than 10 (i.e. the first 3 rows of BG1 only).
//
// "Split" refers to the fact that the sign bits for a single codeword
// are stored in two separate machine words.
//
// Storage of the first 10 sign bits, and the min0 index will be
// identical to the cC2V_storage_x2_low_degree case.
//
// The remainder of the stored signs will be in the second signs word,
// ordered as follows:
//
//  Word 1 Sign Bits                                  Word 0 Sign Bits
//  ------------------------------------------------|------------------------------------------------
//  18 17 16 15 14 13 12 11 10  .  .  .  .  .  .  . | 18 17 16 15 14 13 12 11 10  .  .  .  .  .  .  .
//        ^                                                  ^
//        | Row Index                                        |  Row Index
struct cC2V_storage_x2_high_degree_split
{
    word_t    min0;
    word_t    min1;
    uint32_t  signs_0_9_min0_index;
    uint32_t  signs_10_;
    //------------------------------------------------------------------
    // Typedef for struct to hold sign bits
    // The packed-argmin stack is designed and verified against the bitwise
    // sign store; keep it there.
#if LDPC_MICRO_PACK_ARGMIN
    typedef signs_pair_high_degree_split signs_pair_t;
#else
    typedef signs_pair_high_degree_split_fp signs_pair_t;
#endif
    //------------------------------------------------------------------
    // cC2V_storage_x2_high_degree_split()
    cC2V_storage_x2_high_degree_split() = default;
    //------------------------------------------------------------------
    // cC2V_storage_x2_high_degree_split()
    __device__
    cC2V_storage_x2_high_degree_split(word_t m0,
                                      word_t m1,
                                      word_t m0_index,
                                      const signs_pair_t& s_pair) :
        min0(m0),
        min1(m1),
        signs_0_9_min0_index(s_pair.signs_0_9 | m0_index.u32),
        signs_10_(s_pair.signs_10_)
    {
    }
    //------------------------------------------------------------------
    // init()
    __device__
    void init()
    {
        min0.u32 = 0; min1.u32 = 0; signs_0_9_min0_index = 0; signs_10_ = 0;
    }
    //------------------------------------------------------------------
    // get_signs_pair()
    __device__
    signs_pair_t get_signs_pair() const
    {
        return signs_pair_t(signs_0_9_min0_index & 0xFFC0FFC0, signs_10_);
    }
    //------------------------------------------------------------------
    // get_min0_index_pair()
    __device__
    word_t get_min0_index_pair() const
    {
        word_t w;
        w.u32 = (signs_0_9_min0_index & 0x001F001F);
        return w;
    }
#if 0
    //------------------------------------------------------------------
    // print()
    __device__
    void print() const
    {
        printf("min1 = [%f %f], min0 = [%f %f], index = [%i %i], signs = [0x%X 0x%X]\n",
               __high2float(min1.f16x2),
               __low2float(min1.f16x2),
               __high2float(min0.f16x2),
               __low2float(min0.f16x2),
               (signs_0_9_min0_index >> 16) & 0x1F,
               (signs_0_9_min0_index >> 0)  & 0x1F,
               (signs_0_9_min0_index >> 22) | ((signs_10_ >> 23) << 10),
               ((signs_0_9_min0_index >> 6) & 0x3FF) | ((signs_10_ << 16) >> 22));
    }
#endif
};

//------------------------------------------------------------------
// Template structure to provide type TA if the TIndex is less than
// TAIfLessThan, and TB otherwise.
template <int   TIndex,
          int   TAIfLessThan,
          class TA,
          class TB> struct if_less_than
{
    typedef typename std::conditional<(TIndex < TAIfLessThan), TA, TB>::type type;
};

////////////////////////////////////////////////////////////////////////
// c2v_storage_prenorm_packed
// Trait marking a compressed C2V storage type as a 2-word PRE-NORM
// E5M5 pack (ALGO202 compressed shared tail; see
// cC2V_storage_x2_tail_packed in ldpc2_c2v_cache_split_band.cuh). Such a
// storage keeps the pre-normalization sign-product-signed E5M5
// magnitudes -- whose low 5 mantissa bits per half are zero by
// construction (packed-argmin index strip, then a bit-15/31-only sign
// OR from a NON-CLAMPING sign manager) -- and hides the row's per-edge
// sign bits and argmin index in those free bits. The row context
// provides, keyed on this trait only (every other storage type
// instantiates the verbatim baseline paths):
//   * a pre-norm capture in finalize()/finalize_sprod(),
//   * a get_storage() branch that packs the captured pre-norm words,
//   * a norm-aware from-storage constructor that reconstructs the
//     normalized min0 / (min1 - min0) with the IDENTICAL __hmul2 /
//     __hsub2 sequence finalize() applies, so the reconstructed
//     context -- and therefore the APP trajectory -- is bit-identical.
template <class... Ts> struct c2v_make_void { typedef void type; };
template <class T, class = void>
struct c2v_storage_prenorm_packed : std::false_type {};
template <class T>
struct c2v_storage_prenorm_packed<T, typename c2v_make_void<typename T::prenorm_packed_tag>::type> : std::true_type {};

////////////////////////////////////////////////////////////////////////
// Structure with expanded, in-register C2V representation (as opposed
// to one compressed for storage to global/shared memory), for temporary
// use in processing a single parity node.
template <class TSignMgr, class TMinSumUpdate, class TStorage>
struct cC2V_row_context<__half2, TSignMgr, TMinSumUpdate, TStorage>
{
    //------------------------------------------------------------------
    typedef TSignMgr                         sign_mgr_t;
    typedef TStorage                         storage_t;
    // Infer the type for storage of sign bits from the compressed
    // storage type. (High degree rows may require 2 words, whereas low
    // degree rows may only need one.)
    typedef typename storage_t::signs_pair_t signs_pair_t;
    //------------------------------------------------------------------
    __device__
    cC2V_row_context() {}
#if LDPC_MICRO_PACK_ARGMIN
#if defined(LDPC_MICRO_SPLIT_REDUCE) && LDPC_MICRO_SPLIT_REDUCE && \
    (CUDART_VERSION >= 11000) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    //------------------------------------------------------------------
    // partial_minsum(): packed argmin-min-sum over columns [S, E). Produces
    // (m0,m1) = the two smallest packed (|app|<<5 | colindex) values. Same
    // packing/intrinsics as init_row/update; col 0 uses the inline abs mask
    // (no index OR), others fold the compile-time column index. See (D).
    //
    // PAIRWISE merge (shorter exposed chain): the columns after the
    // first two are consumed two at a time. Each pair is pre-sorted (mn<=mx),
    // then merged into the running top-2 with the standard two-sorted-list merge:
    //     new_m0 = min(m0, mn)
    //     new_m1 = min( m1, min( max(m0,mn), mx ) )
    // The 2nd-min CANDIDATE  min(max(m0,mn), mx)  is built OFF the m1 chain
    // (it depends only on the m0 chain + the pair's own inputs), so the running
    // 2nd-min m1 advances by exactly ONE __vminu2 per column-PAIR instead of
    // per column -- ~halving the VIMNMX dependency-chain depth that gates this
    // latency-sensitive degree-19 core scan. (sm120 has no native 3-input packed
    // u16x2 min -- VIMNMX3 is not emitted -- so the win is the favorable
    // ASSOCIATION, not a fused op; total VIMNMX count is unchanged.) min/max are
    // over the packed values, so the smaller column index still wins on
    // magnitude ties: (m0,m1,argmin) are BIT-IDENTICAL to the linear scan for
    // any input (the packed values are distinct, so the top-2 is
    // order-independent).
    template <int S, int E, bool PREPARED_LAST = false>
    static __device__ __forceinline__
    void partial_minsum(const word_t* app, uint32_t abs_mask, word_t& m0, word_t& m1)
    {
        word_t av0, av1;
        if constexpr(PREPARED_LAST && (S == (E - 1)))
        {
            // PREPARED_LAST stages the immutable final edge as a
            // comparison-ready magnitude plus an out-of-range sign-bearing
            // sentinel.
            av0.u32 = app[S].u32;
        }
        else
        {
            av0.u32 = (S == 0) ? (app[0].u32 & 0x7FE07FE0u)
                               : ((app[S].u32 & abs_mask) | static_cast<uint32_t>(S | (S << 16)));
        }
        if constexpr(PREPARED_LAST && ((S + 1) == (E - 1)))
        {
            av1.u32 = app[S + 1].u32;
        }
        else
        {
            av1.u32 = (app[S + 1].u32 & abs_mask) | static_cast<uint32_t>((S + 1) | ((S + 1) << 16));
        }
        m0.u32 = LDPC2_X2_SCAN_MINU(av0.u32, av1.u32);
        m1.u32 = LDPC2_X2_SCAN_MAXU(av0.u32, av1.u32);
#if defined(LDPC_PAIRWISE_MINSUM) && LDPC_PAIRWISE_MINSUM
        int i = S + 2;
        #pragma unroll
        for(; i + 1 < E; i += 2)
        {
            word_t x, y;
            if constexpr(PREPARED_LAST)
            {
                x.u32 = (i == (E - 1)) ? app[i].u32
                                       : ((app[i].u32 & abs_mask) | static_cast<uint32_t>(i | (i << 16)));
                y.u32 = ((i + 1) == (E - 1)) ? app[i + 1].u32
                                             : ((app[i + 1].u32 & abs_mask) | static_cast<uint32_t>((i + 1) | ((i + 1) << 16)));
            }
            else
            {
                x.u32 = (app[i].u32     & abs_mask) | static_cast<uint32_t>(i       | (i       << 16));
                y.u32 = (app[i + 1].u32 & abs_mask) | static_cast<uint32_t>((i + 1) | ((i + 1) << 16));
            }
            uint32_t mn   = LDPC2_X2_SCAN_MINU(x.u32, y.u32);
            uint32_t mx   = LDPC2_X2_SCAN_MAXU(x.u32, y.u32);
            uint32_t t    = LDPC2_X2_SCAN_MAXU(m0.u32, mn); // loser of (m0, mn) [m0 chain]
            uint32_t cand = LDPC2_X2_SCAN_MINU(t, mx);      // pair's 2nd-min cand [OFF m1 chain]
            m0.u32        = LDPC2_X2_SCAN_MINU(m0.u32, mn);  // running min   : 1 min / pair
            m1.u32        = LDPC2_X2_SCAN_MINU(m1.u32, cand);// running 2nd-min: 1 min / pair
        }
        // Odd leftover column (E-S odd): one linear insertion step.
        if(i < E)
        {
            word_t avp;
            if constexpr(PREPARED_LAST)
            {
                avp.u32 = (i == (E - 1)) ? app[i].u32
                                         : ((app[i].u32 & abs_mask) | static_cast<uint32_t>(i | (i << 16)));
            }
            else
            {
                avp.u32 = (app[i].u32 & abs_mask) | static_cast<uint32_t>(i | (i << 16));
            }
            word_t lo; lo.u32  = LDPC2_X2_SCAN_MAXU(avp.u32, m0.u32);
            m0.u32             = LDPC2_X2_SCAN_MINU(avp.u32, m0.u32);
            m1.u32             = LDPC2_X2_SCAN_MINU(lo.u32,  m1.u32);
        }
#else
        #pragma unroll
        for(int i = S + 2; i < E; ++i)
        {
            word_t avp;
            if constexpr(PREPARED_LAST)
            {
                avp.u32 = (i == (E - 1)) ? app[i].u32
                                         : ((app[i].u32 & abs_mask) | static_cast<uint32_t>(i | (i << 16)));
            }
            else
            {
                avp.u32 = (app[i].u32 & abs_mask) | static_cast<uint32_t>(i | (i << 16));
            }
            word_t lo; lo.u32  = LDPC2_X2_SCAN_MAXU(avp.u32, m0.u32);
            m0.u32             = LDPC2_X2_SCAN_MINU(avp.u32, m0.u32);
            m1.u32             = LDPC2_X2_SCAN_MINU(lo.u32,  m1.u32);
        }
#endif
    }
    //------------------------------------------------------------------
    // merge_minsum(): merge two top-2 results into one. min is associative and
    // the packed column index in the low 5 bits keeps the smaller-index winner
    // on magnitude ties, so any merge order reproduces the linear scan exactly.
    static __device__ __forceinline__
    void merge_minsum(word_t m0a, word_t m1a, word_t m0b, word_t m1b, word_t& m0, word_t& m1)
    {
        word_t lo;   lo.u32   = LDPC2_X2_SCAN_MAXU(m0a.u32, m0b.u32);
        m0.u32                = LDPC2_X2_SCAN_MINU(m0a.u32, m0b.u32);
        word_t m1ab; m1ab.u32 = LDPC2_X2_SCAN_MINU(m1a.u32, m1b.u32);
        m1.u32                = LDPC2_X2_SCAN_MINU(lo.u32,  m1ab.u32);
    }
#if defined(LDPC_MICRO_XOR_SIGNPROD) && LDPC_MICRO_XOR_SIGNPROD
    //------------------------------------------------------------------
    // xor_tree(): balanced XOR reduction of the raw APP words [S, E).
    // XOR is associative/commutative, so any tree shape produces the same
    // word; bits 15/31 of the result are the CW0/CW1 sign parities (the
    // row sign products). ptxas merges adjacent levels into 3-input
    // LOP3.LUT(0x96), so depth is ~log3..log2(E-S) off the APP loads.
    template <int S, int E>
    static __device__ __forceinline__
    uint32_t xor_tree(const word_t* app)
    {
        if constexpr((E - S) == 1)
        {
            return app[S].u32;
        }
        else
        {
            constexpr int M = S + (E - S) / 2;
            return xor_tree<S, M>(app) ^ xor_tree<M, E>(app);
        }
    }
    //------------------------------------------------------------------
    // sign_or_tree(): balanced OR reduction of the repositioned per-column
    // sign masks (app[i] & 0x80008000) >> (BASE - i) for i in [S, E) --
    // exactly the terms signs_pair_t::init_row()/update() accumulate
    // serially (BASE = 9 for the signs_0_9 word, 18 for signs_10_).
    // OR is associative/commutative: the final packed word is identical.
    template <int S, int E, int BASE>
    static __device__ __forceinline__
    uint32_t sign_or_tree(const word_t* app)
    {
        if constexpr((E - S) == 1)
        {
            return (app[S].u32 & 0x80008000u) >> (BASE - S);
        }
        else
        {
            constexpr int M = S + (E - S) / 2;
            return sign_or_tree<S, M, BASE>(app) | sign_or_tree<M, E, BASE>(app);
        }
    }
#endif // LDPC_MICRO_XOR_SIGNPROD
#endif
#endif // LDPC_MICRO_PACK_ARGMIN
    //------------------------------------------------------------------
    // Constructor
    // Initialize a row context from the storage structure (which is
    // typically as small as possible to conserve space and bandwidth)
    template <int ROW_DEGREE>
    __device__
    cC2V_row_context(const storage_t& s,
                     std::integral_constant<int, ROW_DEGREE>) :
        min0(s.min0),
#if !(LDPC_MICRO_PACK_ARGMIN && defined(LDPC_MICRO_PREDIFF) && LDPC_MICRO_PREDIFF)
        min1(s.min1),
#endif
        min0_index(s.get_min0_index_pair()),
        signs_pair(s.get_signs_pair())
    {
#if LDPC_MICRO_PACK_ARGMIN
#if defined(LDPC_MICRO_PREDIFF) && LDPC_MICRO_PREDIFF
        // (C) The storage's second magnitude field holds the PRE-DIFFERENCED
        // (normalized) leave-one-out delta (m1 - m0), already computed on the
        // store side by get_storage(). Loading it directly -- instead of
        // recomputing __hsub2(min1, min0) here -- is bit-identical (same
        // operands, same rounding) but removes the subtract from the per-row
        // subtract pass AND avoids holding min1 live alongside min0/min1m0 at
        // the degree-19 working-set peak, trimming the register high-water.
        // The min1 member is unused on this (subtract) path -- left uninit.
        min1m0.u32 = s.min1.u32;
#else
        // Precompute (min1 - min0) once so extract_pair() can reconstruct
        // the leave-one-out magnitude (idx==argmin ? min1 : min0) as a
        // single FP-pipe FMA instead of an integer LOP3 select. Storage
        // here holds the normalized message, so the difference is the
        // normalized (min1 - min0).
        min1m0.f16x2 = __hsub2(min1.f16x2, min0.f16x2);
#endif
#endif // LDPC_MICRO_PACK_ARGMIN
    }
#if LDPC_MICRO_PACK_ARGMIN
    //------------------------------------------------------------------
    // Constructor (pre-norm packed storage only; see
    // c2v_storage_prenorm_packed). Reconstructs the normalized context
    // from the 2-word pre-norm pack:
    //   * word a = pre-norm signed min0 (E5M5, low 5 bits/half zero)
    //              | argmin index (bits 2..0) | signs of edges 0..1
    //              (bits 4..3);
    //   * word b = pre-norm signed min1 | signs of edges 2..6 (bits 4..0).
    // The magnitude masks recover EXACTLY the pre-norm words finalize()
    // captured (their low 5 bits are zero by construction), and the
    // __hmul2/__hmul2/__hsub2 sequence below is operand-for-operand the
    // one finalize() applied before get_storage() packed -- so min0 /
    // min1m0 here are bit-identical to the baseline 3-word storage's
    // stored values, for any input. The rebuilt signs word has exactly
    // the (degree <= 7) bits 6..12 per half the baseline
    // get_signs_pair() mask would keep; sign_mask()/extract_pair()
    // consume it identically.
    template <int ROW_DEGREE>
    __device__
    cC2V_row_context(const storage_t& s,
                     const __half2&   norm,
                     std::integral_constant<int, ROW_DEGREE>)
    {
        static_assert(c2v_storage_prenorm_packed<storage_t>::value,
                      "norm-aware storage constructor is for pre-norm packed storage only");
        static_assert(ROW_DEGREE <= 7,
                      "pre-norm pack holds 3 index bits + 7 sign bits per codeword");
        word_t m0pn, m1pn;
        m0pn.u32             = s.a.u32 & 0xFFE0FFE0u;
        m1pn.u32             = s.b.u32 & 0xFFE0FFE0u;
        min0.f16x2           = __hmul2(m0pn.f16x2, norm);
        word_t m1n;
        m1n.f16x2            = __hmul2(m1pn.f16x2, norm);
        min1m0.f16x2         = __hsub2(m1n.f16x2, min0.f16x2);
        min0_index.u32       = s.a.u32 & 0x00070007u;
        signs_pair           = typename storage_t::signs_pair_t(
                                   ((s.a.u32 & 0x00180018u) << 3) |
                                   ((s.b.u32 & 0x001F001Fu) << 8));
    }
#endif // LDPC_MICRO_PACK_ARGMIN
    //------------------------------------------------------------------
    // Constructor
    // Initialize a row context from a sequence of APP values,
    // proceeding through the sequence and updating the min-sum
    // representation.
    template <int ROW_DEGREE, int MAX_WORDS>
    __device__
    explicit cC2V_row_context(const __half2&                          norm,
                              word_t                                  (&app)[MAX_WORDS],
                              std::integral_constant<int, ROW_DEGREE>)
    {
#if LDPC_MICRO_PACK_ARGMIN
        // (A) Abs+truncate mask, materialized ONCE for the whole unrolled scan.
        // From __constant__ memory it is opaque to cicc, so ptxas holds it
        // resident in one register; the OR-fused packing LOP3s read that
        // register instead of ptxas re-emitting MOV R,0x7FE07FE0 per column.
        // When A is off this is the literal, which folds back inline per column
        // -- so the toggle lives entirely here.
#if defined(LDPC_MICRO_ABS_MASK_CONST) && LDPC_MICRO_ABS_MASK_CONST
        const uint32_t abs_mask = g_ldpc2_x2_abs_mask;
#else
        const uint32_t abs_mask = 0x7FE07FE0u;
#endif
#if defined(LDPC_MICRO_SPLIT_REDUCE) && LDPC_MICRO_SPLIT_REDUCE && \
    (CUDART_VERSION >= 11000) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
        // (D) Two-way split min-sum reduction. For high-degree rows the single
        // linear running-min is a long VIMNMX dependency chain; split it into
        // two independent half-length reductions and merge the two (min0,min1)
        // pairs. min is associative/commutative and the argmin index packed in
        // the low 5 bits makes VIMNMX keep the smaller index on magnitude ties
        // exactly as the linear scan does, so min0/min1/argmin are bit-identical
        // for any input. RE-A/B'd on the merged loader-3 + {6,10}-interleave
        // base: the split-point optimum shifted from 10 down to HALF==9
        // (measured 8:1.243 < 9:1.2698 > 10:1.2692 > 11:1.253) -- the interleave
        // changed the core scan's instruction schedule, so the most-balanced
        // split (9/10) now narrowly edges the old 10/9. Bit-identical for any
        // HALF (associative min + packed-index tie-break).
        if constexpr(ROW_DEGREE >= 16)
        {
            // Core (degree-19) two-minimum reduction. Split the running top-2 into
            // LDPC2_X2_CORE_SPLIT independent partial reductions merged by an
            // associative merge tree; the packed argmin index in the low 5 bits
            // keeps (min0,min1,argmin) BIT-IDENTICAL for any input (associative
            // min + index tie-break). Width / split point / emission order are
            // pure scheduling knobs (trade VIMNMX dep-chain depth against extra
            // merge ops on the un-saturated integer pipe). Default (SPLIT=2,
            // HALF=9) reproduces the baseline reduction exactly.
#ifndef LDPC2_X2_CORE_SPLIT
#define LDPC2_X2_CORE_SPLIT 2
#endif
#if defined(LDPC_MICRO_XOR_SIGNPROD) && LDPC_MICRO_XOR_SIGNPROD
            // (E) Row sign product via a balanced XOR tree over the raw APP
            // words (parity-of-signs == XOR of sign bits: bit-identical to
            // sign_product_mask()'s popcount parities), ready ~4 LOP3 levels
            // off the APP loads instead of at the end of the serial OR chain
            // + POPC fold. The packed sign words (still required by the
            // stored message / next iteration's subtract) are built with
            // balanced OR trees of the same repositioned terms -- identical
            // final words, but consumed only by the get_storage() pack, so
            // they leave the row's exposed finalize->extract critical path.
            const uint32_t sprod = xor_tree<0, ROW_DEGREE>(app) & 0x80008000u;
            signs_pair.signs_0_9 = sign_or_tree<0, 10, 9>(app);
            signs_pair.signs_10_ = sign_or_tree<10, ROW_DEGREE, 18>(app);
#else
            signs_pair.init_row(app[0], app[1]);
            #pragma unroll
            for(int i = 2; i < ROW_DEGREE; ++i)
            {
                signs_pair.update(app[i], i);
            }
#endif // LDPC_MICRO_XOR_SIGNPROD
#if LDPC2_X2_CORE_SPLIT == 4
            constexpr int Q1 = ROW_DEGREE / 4;
            constexpr int Q2 = ROW_DEGREE / 2;
            constexpr int Q3 = (3 * ROW_DEGREE) / 4;
            word_t m0a, m1a, m0b, m1b, m0c, m1c, m0d, m1d;
            partial_minsum<0,  Q1>(app, abs_mask, m0a, m1a);
            partial_minsum<Q1, Q2>(app, abs_mask, m0b, m1b);
            partial_minsum<Q2, Q3>(app, abs_mask, m0c, m1c);
            partial_minsum<Q3, ROW_DEGREE>(app, abs_mask, m0d, m1d);
            word_t q00, q01, q10, q11;
            merge_minsum(m0a, m1a, m0b, m1b, q00, q01);
            merge_minsum(m0c, m1c, m0d, m1d, q10, q11);
            merge_minsum(q00, q01, q10, q11, min0, min1);
#elif LDPC2_X2_CORE_SPLIT == 3
            constexpr int T1 = ROW_DEGREE / 3;
            constexpr int T2 = (2 * ROW_DEGREE) / 3;
            word_t m0a, m1a, m0b, m1b, m0c, m1c;
            partial_minsum<0,  T1>(app, abs_mask, m0a, m1a);
            partial_minsum<T1, T2>(app, abs_mask, m0b, m1b);
            partial_minsum<T2, ROW_DEGREE>(app, abs_mask, m0c, m1c);
            word_t q00, q01;
            merge_minsum(m0a, m1a, m0b, m1b, q00, q01);
            merge_minsum(q00, q01, m0c, m1c, min0, min1);
#else
#ifndef LDPC2_X2_CORE_HALF
#define LDPC2_X2_CORE_HALF 9
#endif
            constexpr int HALF = LDPC2_X2_CORE_HALF;
            word_t m0a, m1a, m0b, m1b;
#if defined(LDPC2_X2_CORE_SWAP) && LDPC2_X2_CORE_SWAP
            partial_minsum<HALF, ROW_DEGREE>(app, abs_mask, m0b, m1b);
            partial_minsum<0, HALF>(app, abs_mask, m0a, m1a);
#else
            partial_minsum<0, HALF>(app, abs_mask, m0a, m1a);
            partial_minsum<HALF, ROW_DEGREE>(app, abs_mask, m0b, m1b);
#endif
            merge_minsum(m0a, m1a, m0b, m1b, min0, min1);
#endif
#if defined(LDPC_MICRO_XOR_SIGNPROD) && LDPC_MICRO_XOR_SIGNPROD
            finalize_sprod(norm, sprod);
#else
            finalize(norm, ROW_DEGREE);
#endif
        }
        else
        {
#if defined(LDPC_MICRO_XOR_SIGNPROD) && LDPC_MICRO_XOR_SIGNPROD && \
    defined(LDPC_MICRO_XOR_SIGNPROD_LOWDEG) && LDPC_MICRO_XOR_SIGNPROD_LOWDEG
            // (E'') LDPC_MICRO_XOR_SIGNPROD_LOWDEG: extend the core branch's
            // XOR-tree row sign product and balanced OR-tree sign-word build
            // to LOW-degree compressed rows (ALGO202's compressed tail rows,
            // degree 6..7). Replaces, per row,
            //   * the two-POPC parity fold of sign_product_mask() (2x
            //     LOP3+POPC, plus the shift/merge) with the XOR-tree parity
            //     over the raw v2c words -- identical bits 15/31 by GF(2)
            //     linearity, ready ~log3(D) LOP3 levels off the subtract
            //     results instead of at the end of the serial sign-OR chain,
            //     shortening the finalize->pack/extract critical path; and
            //   * the serial init_row/update sign accumulation with a
            //     balanced OR tree of the SAME repositioned terms
            //     ((app[i] & 0x80008000) >> (9-i)) -- OR is associative and
            //     commutative, so the stored signs_0_9 word is identical.
            // partial_minsum<0, D> is the same packed-argmin insertion scan
            // init_row/update run (same values, same order), so
            // min0/min1/argmin are bit-identical too: the APP trajectory is
            // unchanged; only instruction scheduling differs.
            // Guarded by its own toggle (default off) so every other TU
            // keeps the verbatim serial path below.
            if constexpr(ROW_DEGREE <= 10)
            {
#if defined(LDPC_MICRO_PREPARED_LAST_EDGE) && LDPC_MICRO_PREPARED_LAST_EDGE
                // Prepared words carry the final edge's two signs in bit 4
                // of each halfword. Move them to the native sign positions
                // independently of the comparison dependency chain.
                const uint32_t ext_sprod = app[ROW_DEGREE - 1].u32 << 11;
                const uint32_t sprod =
                    (xor_tree<0, ROW_DEGREE - 1>(app) ^ ext_sprod) & 0x80008000u;
                signs_pair = signs_pair_t(sign_or_tree<0, ROW_DEGREE - 1, 9>(app));
                partial_minsum<0, ROW_DEGREE, true>(app, abs_mask, min0, min1);
                finalize_sprod(norm, sprod);
#else
                const uint32_t sprod = xor_tree<0, ROW_DEGREE>(app) & 0x80008000u;
                signs_pair = signs_pair_t(sign_or_tree<0, ROW_DEGREE, 9>(app));
                partial_minsum<0, ROW_DEGREE>(app, abs_mask, min0, min1);
                finalize_sprod(norm, sprod);
#endif
            }
            else
            {
                init_row(app[0], app[1], abs_mask);
                #pragma unroll
                for(int i = 2; i < ROW_DEGREE; ++i)
                {
                    update(app[i], i, abs_mask);
                }
                finalize(norm, ROW_DEGREE);
            }
#else
            init_row(app[0], app[1], abs_mask);
            #pragma unroll
            for(int i = 2; i < ROW_DEGREE; ++i)
            {
                update(app[i], i, abs_mask);
            }
            finalize(norm, ROW_DEGREE);
#endif // LDPC_MICRO_XOR_SIGNPROD_LOWDEG
        }
#else
        // Initialize with first two values
        init_row(app[0], app[1], abs_mask);
        // Update min-sum with the rest of the values
        #pragma unroll
        for(int i = 2; i < ROW_DEGREE; ++i)
        {
            update(app[i], i, abs_mask);
        }
        // Post-process row context to prepare for extraction
        finalize(norm, ROW_DEGREE);
#endif
#else // !LDPC_MICRO_PACK_ARGMIN -- verbatim baseline
        // Initialize with first two values
        init_row(app[0], app[1]);
        // Update min-sum with the rest of the values
        #pragma unroll
        for(int i = 2; i < ROW_DEGREE; ++i)
        {
            update(app[i], i);
        }
        // Post-process row context to prepare for extraction
        finalize(norm, ROW_DEGREE);
#endif // LDPC_MICRO_PACK_ARGMIN
    }
    //------------------------------------------------------------------
    // init_row()
    // Initialize row context min-sum fields with the first pair of
    // sequence values
#if LDPC_MICRO_PACK_ARGMIN
    __device__
    void init_row(word_t v0, word_t v1, uint32_t abs_mask)
    {
        // Track min0/min1 as MAGNITUDES (the running-min's sign is discarded
        // later; sign_mgr_t::finalize() overwrites the sign bits with the row's
        // sign product). Unsigned 16-bit ordering of an fp16 magnitude bit-
        // pattern equals magnitude ordering, so __vminu2/__vmaxu2 (VIMNMX2,
        // integer pipe) compute the running min/2nd-min exactly.
#if CUDART_VERSION >= 11000 && __CUDA_ARCH__ >= 800
        // ARGMIN-PACKING with FUSED ABS: a single AND mask (0x7FE0 per lane)
        // clears the sign bit (abs) AND the low 5 mantissa bits in one LOP3,
        // then the column index is OR'd into the freed low bits. Unsigned
        // 16-bit ordering of the resulting pattern equals magnitude ordering
        // with the index as tie-break, so __vminu2/__vmaxu2 propagate BOTH the
        // running min MAGNITUDE (high 11 bits) AND its arg-min INDEX (low 5
        // bits) together. The index is recovered in finalize(); magnitudes
        // become 5-bit-mantissa.
        word_t av0, av1;
        av0.u32               = (v0.u32 & 0x7FE07FE0);                 // |v0|, column index 0
        av1.u32               = (v1.u32 & abs_mask) | 0x00010001;      // |v1|, column index 1
        min0.u32              = LDPC2_X2_SCAN_MINU(av0.u32, av1.u32);
        min1.u32              = LDPC2_X2_SCAN_MAXU(av0.u32, av1.u32);
#else
        (void)abs_mask;
        word_t av0 = fp16x2_abs(v0);
        word_t av1 = fp16x2_abs(v1);
        word_t cZeros, cOnes;
        cZeros.u16x2 = ushort2{0, 0};
        cOnes.u16x2  = ushort2{1, 1};
        word_t absv0_lt_absv1 = hset2_bm_lt(av0, av1);
        min0                  = select_from_mask(absv0_lt_absv1, av0, av1);
        min1                  = select_from_mask(absv0_lt_absv1, av1, av0);
        min0_index            = select_from_mask(absv0_lt_absv1, cZeros, cOnes);
#endif
        signs_pair.init_row(v0, v1);
    }
#else // !LDPC_MICRO_PACK_ARGMIN -- verbatim baseline
    __device__
    void init_row(word_t v0, word_t v1)
    {
        word_t cZeros, cOnes;
        cZeros.u16x2 = ushort2{0, 0};
        cOnes.u16x2 = ushort2{1, 1};

        word_t absv0_lt_absv1 = hset2_bm_lt(fp16x2_abs(v0), fp16x2_abs(v1));
        min0                  = select_from_mask(absv0_lt_absv1, v0, v1);
        min1                  = select_from_mask(absv0_lt_absv1, v1, v0);
        min0_index            = select_from_mask(absv0_lt_absv1, cZeros, cOnes);
        signs_pair.init_row(v0, v1);
    }
#endif // LDPC_MICRO_PACK_ARGMIN
    //------------------------------------------------------------------
    // update_old()
    // Update the internal representation with a word containing a new
    // value for each codeword.
    __device__
    void update_old(word_t v, int idx)
    {
        word_t idx_pair;

        idx_pair.u16x2           = ushort2{static_cast<unsigned short>(idx),
                                           static_cast<unsigned short>(idx)};
        word_t absv_lt_absmin0   = hset2_bm_lt(fp16x2_abs(v), fp16x2_abs(min0));            // |v| < |min0|?
        min0_index               = select_from_mask(absv_lt_absmin0, idx_pair, min0_index);
#if CUDART_VERSION >= 11000 && __CUDA_ARCH__ >= 800
        // Compute loser's abs magnitude before overwriting min0.
        // max(|v|, |min0|) = |element that does NOT become the new min0|.
        // This explicitly maps to HMNMX2(!PT) and avoids the
        // select_from_mask + abs decomposition the compiler may otherwise choose.
        word_t loser_abs;
        loser_abs.f16x2          = __hmax2(__habs2(v.f16x2), __habs2(min0.f16x2));
        min0                     = select_from_mask(absv_lt_absmin0, v, min0);
        min1.f16x2               = __hmin2(loser_abs.f16x2, __habs2(min1.f16x2));
#else
        word_t maybe_min1        = select_from_mask(absv_lt_absmin0, min0, v);
        min0                     = select_from_mask(absv_lt_absmin0, v, min0);
        word_t abstmp_lt_absmin1 = hset2_bm_lt(fp16x2_abs(maybe_min1), fp16x2_abs(min1));
        min1                     = select_from_mask(abstmp_lt_absmin1, maybe_min1, min1);
#endif
        signs_pair.update(v, idx);
    }
    //------------------------------------------------------------------
    // update()
    // Update the internal representation with a word containing a new
    // value for each codeword.
#if LDPC_MICRO_PACK_ARGMIN
    __device__
    void update(word_t v, int idx, uint32_t abs_mask)
    {
#if CUDART_VERSION >= 11000 && __CUDA_ARCH__ >= 800
        // ARGMIN-PACKING with FUSED ABS (see init_row): one AND mask does abs
        // (clear sign) AND truncate (clear low 5 mantissa bits); the compile-
        // time column index is OR'd into the freed low bits. __vmaxu2/__vminu2
        // then carry the running min/2nd-min MAGNITUDE (high 11 bits) and the
        // arg-min INDEX (low 5 bits) in one SIMD op each.
        word_t avp;
        avp.u32                  = (v.u32 & abs_mask) | static_cast<uint32_t>(idx | (idx << 16));
        word_t loser_abs;
        loser_abs.u32            = LDPC2_X2_SCAN_MAXU(avp.u32, min0.u32);
        min0.u32                 = LDPC2_X2_SCAN_MINU(avp.u32, min0.u32);
        min1.u32                 = LDPC2_X2_SCAN_MINU(loser_abs.u32, min1.u32);
#else
        (void)abs_mask;
        word_t av;
        av.f16x2                 = __habs2(v.f16x2);
        word_t idx_pair;
        idx_pair.u16x2           = ushort2{static_cast<unsigned short>(idx),
                                           static_cast<unsigned short>(idx)};
        word_t absv_lt_absmin0   = hset2_bm_lt(av, min0);                                   // |v| < min0 ?
        min0_index               = select_from_mask(absv_lt_absmin0, idx_pair, min0_index);
        word_t loser             = select_from_mask(absv_lt_absmin0, min0, av);
        min0                     = select_from_mask(absv_lt_absmin0, av, min0);
        word_t loser_lt_min1     = hset2_bm_lt(loser, min1);
        min1                     = select_from_mask(loser_lt_min1, loser, min1);
#endif
        signs_pair.update(v, idx);
    }
#else // !LDPC_MICRO_PACK_ARGMIN -- verbatim baseline
    __device__
    void update(word_t v, int idx)
    {
        word_t idx_pair;

        idx_pair.u16x2           = ushort2{static_cast<unsigned short>(idx),
                                           static_cast<unsigned short>(idx)};
        word_t absv_lt_absmin0   = hset2_bm_lt(fp16x2_abs(v), fp16x2_abs(min0));            // |v| < |min0|?
        min0_index               = select_from_mask(absv_lt_absmin0, idx_pair, min0_index); // Update min0_index if |v| < |min0|
#if CUDART_VERSION >= 11000 && __CUDA_ARCH__ >= 800
        // Compute loser's abs magnitude before overwriting min0.
        // max(|v|, |min0|) = |element that does NOT become the new min0|.
        // This explicitly maps to HMNMX2(!PT) and avoids the
        // select_from_mask + abs decomposition the compiler may otherwise choose.
        word_t loser_abs;
        loser_abs.f16x2          = __hmax2(__habs2(v.f16x2), __habs2(min0.f16x2));
        min0                     = select_from_mask(absv_lt_absmin0, v, min0);
        min1.f16x2               = __hmin2(loser_abs.f16x2, __habs2(min1.f16x2));
#else
        word_t maybe_min1        = select_from_mask(absv_lt_absmin0, min0, v);
        min0                     = select_from_mask(absv_lt_absmin0, v, min0);
        word_t abstmp_lt_absmin1 = hset2_bm_lt(fp16x2_abs(maybe_min1), fp16x2_abs(min1));
        min1                     = select_from_mask(abstmp_lt_absmin1, maybe_min1, min1);
#endif
        signs_pair.update(v, idx);
    }
#endif // LDPC_MICRO_PACK_ARGMIN
    //------------------------------------------------------------------
    // finalize()
#if LDPC_MICRO_PACK_ARGMIN
    __device__
    void finalize(const __half2& norm, int row_degree)
    {
#if LDPC2_X2_FUSE_NORM
        (void)norm;  // norm folded into the APP add / get_storage instead
#endif
        // Normalization is folded into the APP add (via __hfma2) and applied
        // once when packing the stored message (get_storage(norm)) when
        // FUSE_NORM; otherwise applied once below. min0/min1 hold magnitudes.
#if CUDART_VERSION >= 11000 && __CUDA_ARCH__ >= 800
        // ARGMIN-PACKING wrap-up: recover the arg-min column index from the
        // low 5 bits of the packed running-min, then strip the index bits so
        // min0/min1 hold clean (5-bit-mantissa) magnitudes for the APP math.
        min0_index.u32           = (min0.u32 & 0x001F001F);
        min0.u32                &= 0xFFE0FFE0;
        min1.u32                &= 0xFFE0FFE0;
#endif
        // Allow the storage class for signs to update after all values
        // have been incorporated (the fp-based sign store repositions its
        // accumulated bits here; skipping it corrupts every row sign).
        signs_pair.finalize();

        // Adjust signs of min0, min1 values (optionally)
        sign_mgr_t::finalize(signs_pair, min0, min1);

        if constexpr (c2v_storage_prenorm_packed<storage_t>::value)
        {
            // Pre-norm packed storage: capture the sign-product-signed
            // E5M5 magnitudes BEFORE normalization overwrites them; the
            // get_storage() pack embeds signs/index in their (zero) low
            // 5 bits per half. Dead (never read) for every other storage.
            pn_min0_ = min0;
            pn_min1_ = min1;
        }
#if !LDPC2_X2_FUSE_NORM
        // Normalize-once: scale min0/min1 here (norm > 0, so the row sign
        // product just applied is preserved). The single normalized min1m0
        // below then serves both this iteration's extract AND the stored delta.
        min0.f16x2 = __hmul2(min0.f16x2, norm);
        min1.f16x2 = __hmul2(min1.f16x2, norm);
#endif
        // Precompute (min1 - min0) for the FMA-based leave-one-out
        // reconstruction in extract_pair(). min0/min1 share the same
        // (sign-product) sign here.
        min1m0.f16x2 = __hsub2(min1.f16x2, min0.f16x2);
    }
#if defined(LDPC_MICRO_XOR_SIGNPROD) && LDPC_MICRO_XOR_SIGNPROD
    //------------------------------------------------------------------
    // finalize_sprod()
    // As the packed finalize() above, but consumes the caller-precomputed
    // row sign-product mask (XOR-tree parity, bits 15/31) instead of
    // deriving it from the packed sign words via POPC. Identical argmin
    // recovery, index strip, sign application, normalization, and
    // (min1 - min0) precompute -- bit-identical output for any input.
    __device__
    void finalize_sprod(const __half2& norm, uint32_t sign_prod_mask)
    {
#if LDPC2_X2_FUSE_NORM
        (void)norm;  // norm folded into the APP add / get_storage instead
#endif
#if CUDART_VERSION >= 11000 && __CUDA_ARCH__ >= 800
        // ARGMIN-PACKING wrap-up (see finalize()).
        min0_index.u32           = (min0.u32 & 0x001F001F);
        min0.u32                &= 0xFFE0FFE0;
        min1.u32                &= 0xFFE0FFE0;
#endif
        sign_mgr_t::finalize_mask(sign_prod_mask, min0, min1);

        if constexpr (c2v_storage_prenorm_packed<storage_t>::value)
        {
            // Pre-norm capture (see finalize()).
            pn_min0_ = min0;
            pn_min1_ = min1;
        }
#if !LDPC2_X2_FUSE_NORM
        // Normalize-once (see finalize()).
        min0.f16x2 = __hmul2(min0.f16x2, norm);
        min1.f16x2 = __hmul2(min1.f16x2, norm);
#endif
        min1m0.f16x2 = __hsub2(min1.f16x2, min0.f16x2);
    }
#endif // LDPC_MICRO_XOR_SIGNPROD
#else // !LDPC_MICRO_PACK_ARGMIN -- verbatim baseline
    __device__
    void finalize(const __half2& norm, int row_degree)
    {
        // Apply normalization to both values (min1 and min0)
        // TODO: fuse with add of update to APP value
        min0.f16x2 = __hmul2(min0.f16x2, norm);
        min1.f16x2 = __hmul2(min1.f16x2, norm);

        // Allow the storage class for signs to update after all values
        // have been incorporated.
        signs_pair.finalize();

        // Adjust signs of min0, min1 values (optionally)
        sign_mgr_t::finalize(signs_pair, min0, min1);
    }
#endif // LDPC_MICRO_PACK_ARGMIN
    //------------------------------------------------------------------
    // get_storage()
    // Initialize a storage structure from the internal min-sum
    // representation
#if LDPC_MICRO_PACK_ARGMIN
    // The stored message is normalized here (norm * min), so the next
    // iteration's subtract pass sees the scaled values.
    __device__
    storage_t get_storage(const __half2& norm) const
    {
        if constexpr (c2v_storage_prenorm_packed<storage_t>::value)
        {
            // Pre-norm packed storage: pack the finalize()-captured
            // pre-norm signed E5M5 magnitudes with the argmin index and
            // per-edge signs folded into their zero low 5 bits per half
            // (storage ctor does the bit packing). The normalized values
            // are reconstructed bit-identically by the norm-aware
            // from-storage constructor.
            (void)norm;
            return storage_t(pn_min0_, pn_min1_, min0_index, signs_pair);
        }
        else
        {
#if !LDPC2_X2_FUSE_NORM
        // Normalize-once: min0/min1 (and min1m0) were already scaled by norm in
        // finalize(), so storage is a pure pack -- no multiply on the FP pipe.
        (void)norm;
    #if defined(LDPC_MICRO_PREDIFF) && LDPC_MICRO_PREDIFF
        return storage_t(min0, min1m0, min0_index, signs_pair);
    #else
        return storage_t(min0, min1, min0_index, signs_pair);
    #endif
#else
        word_t m0, m1;
        m0.f16x2 = __hmul2(min0.f16x2, norm);
        m1.f16x2 = __hmul2(min1.f16x2, norm);
#if defined(LDPC_MICRO_PREDIFF) && LDPC_MICRO_PREDIFF
        // (C) Store m0 and the pre-differenced (normalized) leave-one-out delta.
        word_t m1m0;
        m1m0.f16x2 = __hsub2(m1.f16x2, m0.f16x2);
        return storage_t(m0, m1m0, min0_index, signs_pair);
#else
        return storage_t(m0, m1, min0_index, signs_pair);
#endif
#endif // LDPC2_X2_FUSE_NORM
        }
    }
#else // !LDPC_MICRO_PACK_ARGMIN -- verbatim baseline
    __device__
    storage_t get_storage() const
    {
        return storage_t(min0, min1, min0_index, signs_pair);
    }
#endif // LDPC_MICRO_PACK_ARGMIN
    //------------------------------------------------------------------
    // extract_pair()
    // Extract a pair of values using the internal representation. The
    // returned word will contain two values, with the low word having
    // the sequence value with the lower index.
#if LDPC_MICRO_PACK_ARGMIN
    __device__
    word_t extract_pair(int idx) const
    {
        word_t idx_pair;
        idx_pair.u16x2       = ushort2{static_cast<unsigned short>(idx),
                                       static_cast<unsigned short>(idx)};
        // Leave-one-out magnitude reconstruction on the FP pipe:
        //   ex = (idx==argmin) ? min1 : min0  ==  min0 + mask*(min1-min0)
        // The arg-min selector is produced as a 1.0/0.0 fp16x2 and folded
        // with the precomputed (min1-min0) into a single __hfma2.
        word_t mask_eq       = hset2_bf_eq(idx_pair, min0_index);         // 1.0 @ argmin, else 0.0
        word_t ex_value;
        ex_value.f16x2       = __hfma2(mask_eq.f16x2, min1m0.f16x2, min0.f16x2);
        return sign_mgr_t::apply_sign(signs_pair, ex_value, idx);
    }
    //------------------------------------------------------------------
    // extract_signed_magnitude()
    // (B) Reconstruct the leave-one-out message carrying ONLY the row sign
    // product S (already baked into min0/min1 by finalize()). The caller
    // re-applies the per-column input sign sigma_i from the message value it
    // holds (app[i]), so this omits apply_sign()'s per-column packed-sign
    // reposition shift. Identical magnitude/argmin math to extract_pair().
    __device__
    word_t extract_signed_magnitude(int idx) const
    {
        word_t idx_pair;
        idx_pair.u16x2       = ushort2{static_cast<unsigned short>(idx),
                                       static_cast<unsigned short>(idx)};
        word_t mask_eq       = hset2_bf_eq(idx_pair, min0_index);
        word_t ex_value;
        ex_value.f16x2       = __hfma2(mask_eq.f16x2, min1m0.f16x2, min0.f16x2);
        return ex_value;
    }
#else // !LDPC_MICRO_PACK_ARGMIN -- verbatim baseline
    __device__
    word_t extract_pair(int idx) const
    {
        word_t idx_pair;
        // Create a fp16x2 word with the index value replicated, and
        // use denormalized float comparisons to compare the index to
        // the min0_index for each codeword
        idx_pair.u16x2       = ushort2{static_cast<unsigned short>(idx),
                                       static_cast<unsigned short>(idx)};
        word_t idx_is_min0   = hequ_bm(idx_pair, min0_index);             // idx == min0_index?
        word_t ex_value      = select_from_mask(idx_is_min0, min1, min0); // ex_value = (idx_is_min0) ? min1 : min0
        return sign_mgr_t::apply_sign(signs_pair, ex_value, idx);
    }
#endif // LDPC_MICRO_PACK_ARGMIN
    //------------------------------------------------------------------
    // Data
    word_t       min0;        // (fp16x2 for 2 codewords)
    word_t       min1;        // (fp16x2 for 2 codewords)
#if LDPC_MICRO_PACK_ARGMIN
    word_t       min1m0;      // (min1 - min0), precomputed for FMA select
    // Pre-norm signed E5M5 magnitudes, captured by finalize() ONLY for
    // pre-norm packed storage types (c2v_storage_prenorm_packed); never
    // written or read for any other storage, so they are dead members
    // there (SROA/DCE leaves no trace in the generated code).
    word_t       pn_min0_;
    word_t       pn_min1_;
#endif
    word_t       min0_index;  // index of min 0 for each of 2 codewords
    signs_pair_t signs_pair;  // 1 or 2 words of sign bits, depending on
                              // row degree
};

template <class TRowContext>
class cC2V_row_proc<__half2, TRowContext>
{
private:
    //------------------------------------------------------------------
    typedef TRowContext row_context_t;
    typedef __half2     app_t;
public:
    //------------------------------------------------------------------
    // process_row()
    template <int ROW_DEGREE,
              int UPDATE_ROW_DEGREE,
              bool IS_FIRST = false,
              bool IS_LAST = false,
              class TStorage>
    __device__
    static void process_row(word_t          (&app)[ROW_DEGREE],
                            TStorage&       row_storage,
                            const __half2&  norm)
    {
        // IS_FIRST (first BP iteration): compressed C2V row storage is still
        // zero, so the decompress-and-subtract of the previous iteration's
        // C2V is a bit-exact no-op (x - (+/-0) == x in fp16, and the all-zero
        // packed storage decompresses to +0.0 for every column). Skip it at
        // compile time -- this also elides the C2V decompression entirely.
        if constexpr (!IS_FIRST)
        {
            if constexpr (c2v_storage_prenorm_packed<TStorage>::value)
            {
                // Pre-norm packed storage: the from-storage context
                // reconstruction needs norm (it re-applies finalize()'s
                // exact __hmul2/__hsub2 sequence -- bit-identical values).
                app_sub_prev_iter_prenorm<ROW_DEGREE, UPDATE_ROW_DEGREE>(app, row_storage, norm);
            }
            else
            {
                app_sub_prev_iter<ROW_DEGREE, UPDATE_ROW_DEGREE>(app, row_storage);
            }
        }
        // IS_LAST (final BP iteration): the packed C2V written by app_update is
        // read only by the NEXT iteration's app_sub_prev_iter; on the final
        // iteration there is none, so the pack (get_storage) is provably dead.
        app_update<ROW_DEGREE, UPDATE_ROW_DEGREE, IS_LAST>(row_storage, norm, app);
    }
private:
    //------------------------------------------------------------------
    // app_sub_prev_iter()
    template <int ROW_DEGREE, int UPDATE_ROW_DEGREE, class TStorage>
    __device__
    static void app_sub_prev_iter(word_t          (&app)[ROW_DEGREE],
                                  const TStorage& row_storage)
    {
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Initialize a row processing context using data from the
        // previous iteration (which may be stored in registers or
        // global memory).
        row_context_t rc(row_storage, std::integral_constant<int, ROW_DEGREE>{});
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Update the APP values with the decrement from the previous
        // iteration. We don't need to update APP values for extension
        // nodes.
        #pragma unroll
        for(int i = 0; i < UPDATE_ROW_DEGREE; ++i)
        {
            // Operate on a pair of values at a time
            word_t dec = rc.extract_pair(i);
            app[i].f16x2 = __hsub2(app[i].f16x2, dec.f16x2);
        }
    }
    //------------------------------------------------------------------
    // app_sub_prev_iter_prenorm()
    // As app_sub_prev_iter(), for pre-norm packed storage: the context
    // is reconstructed through the norm-aware constructor (bit-identical
    // normalized min0 / min1m0 / index / signs -- see the constructor),
    // then the per-column update loop is the IDENTICAL extract_pair +
    // __hsub2 sequence. (Template: instantiated only for storage types
    // with the c2v_storage_prenorm_packed trait, which exist only in
    // LDPC_MICRO_PACK_ARGMIN translation units.)
    template <int ROW_DEGREE, int UPDATE_ROW_DEGREE, class TStorage>
    __device__
    static void app_sub_prev_iter_prenorm(word_t          (&app)[ROW_DEGREE],
                                          const TStorage& row_storage,
                                          const __half2&  norm)
    {
        row_context_t rc(row_storage, norm, std::integral_constant<int, ROW_DEGREE>{});
        #pragma unroll
        for(int i = 0; i < UPDATE_ROW_DEGREE; ++i)
        {
            word_t dec = rc.extract_pair(i);
            app[i].f16x2 = __hsub2(app[i].f16x2, dec.f16x2);
        }
    }
    //------------------------------------------------------------------
    // app_update()
    template <int ROW_DEGREE,
              int UPDATE_ROW_DEGREE,
              bool IS_LAST = false,
              class TStorage>
    __device__
    static void app_update(TStorage&            row_storage,
                           const __half2&       norm,
                           word_t               (&app)[ROW_DEGREE])
    {
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Construct a row context using the updated APP values (with
        // values from the previous iteration subtracted). This will
        // create a min-sum representation of the APP values.
        row_context_t rc(norm,
                         app,
                         std::integral_constant<int, ROW_DEGREE>{});
#if LDPC_MICRO_PACK_ARGMIN
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Packed-argmin core: get_storage() takes norm (it packs the
        // normalized message when FUSE_NORM, else just packs the already-
        // normalized min0/min1). Issue the pack first so its scaling overlaps
        // the APP adds. app_combine_x2 folds norm via __hfma2 when FUSE_NORM,
        // else is a plain __hadd2 (matching the baseline below).
        // IS_LAST: the packed storage is read only by the next iteration's
        // subtract -- dead on the final iteration. The extract loop below reads
        // `rc` (not row_storage), so eliding the pack is bit-identical and lets
        // ptxas DCE the entire normalize-and-pack sequence.
        if constexpr (!IS_LAST)
        {
#if defined(LDPC2_A107_CORE_EMIT_LEAD) && LDPC2_A107_CORE_EMIT_LEAD
            if constexpr(ROW_DEGREE != 19)
            {
                row_storage = rc.get_storage(norm);
            }
#else
            row_storage = rc.get_storage(norm);
#endif
        }
#if defined(LDPC_MICRO_PARTIAL_SIGN) && LDPC_MICRO_PARTIAL_SIGN
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // (B) Partial signed-domain reconstruction. On this (add) pass app[i]
        // still holds the variable-to-check message v_i, so its sign IS the
        // per-column input sign sigma_i. For columns whose packed-sign
        // reposition shift is non-zero (and not in the small kept set), XOR
        // app[i]'s sign bit into the row-sign-product magnitude
        // (extract_signed_magnitude) instead of repositioning the stored
        // sigma_i (extract_pair's sign_mask shift). Bit-identical to
        // extract_pair(); drops the per-column variable shift on the
        // extract->combine->store critical path. RE-A/B'd on the merged
        // base (loader-3 + {6,10} interleave): the keep-set optimum moved from
        // {0,7,14} to {0,5,10,15} (4 evenly-spaced anchors at multiples of 5).
        // Measured latency: {0,7,14} 1.2698 < {0,5,10,15} 1.2719 > {0,4,8,12,16}
        // 1.2715 (5 anchors saturates) and > {0,6,11,16} 1.2684 (position, not
        // just count, matters). More extract_pair anchors shorten more columns'
        // app[i]-critical chains (inc is independent of app[i]) at the cost of an
        // off-path reposition shift each; 6 total extract_pair columns ({0,5,10,
        // 15} + free zero-shift {9,18}) is the latency sweet spot on this base.
        // Keep set {0,5,10,15} + the two free zero-shift columns (9,18).
        // The sign bit is read before app_combine_x2 overwrites app[i].
#if defined(LDPC_MICRO_PARTIAL_SIGN_ALL) && LDPC_MICRO_PARTIAL_SIGN_ALL
            //- - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            // (B') Partial signed-domain reconstruction for EVERY column.
            // On this (add) pass app[i] still holds the v2c message v_i, whose
            // sign IS the per-column input sign sigma_i, and the stored sign
            // for column i equals that sign, so sign_mask(i) == (app[i] &
            // 0x80008000) for ALL i. Reconstructing the leave-one-out sign by
            // XOR-ing app[i]'s sign into the row-sign-product magnitude is thus
            // bit-identical to extract_pair(i) for every column -- including
            // the {0,7,14} + zero-shift {9,18} columns the keep-set routes
            // through apply_sign(). Doing it for all columns (1) drops the 3
            // per-row reposition shifts the keep-set still pays, and (2) ends
            // signs_pair's live range at get_storage() (no add-loop reader),
            // trimming the degree-19 register high-water on this 1-CTA/SM
            // latency-bound core. This is the register-relief that PRECEDES the
            // {6,10} fused interleave -- freeing per-thread register width so
            // ptxas can overlap the two independent dependency chains deeper.
#if defined(LDPC2_A107_CORE_EMIT_LEAD) && LDPC2_A107_CORE_EMIT_LEAD
        if constexpr((ROW_DEGREE == 19) && (UPDATE_ROW_DEGREE == 19))
        {
            #pragma unroll
            for(int i = 0; i < LDPC2_A107_CORE_EMIT_LEAD; ++i)
            {
                word_t inc = rc.extract_signed_magnitude(i);
                inc.u32    = inc.u32 ^ (app[i].u32 & 0x80008000u);
                app[i]     = app_combine_x2(app[i], inc, norm);
            }
            if constexpr(!IS_LAST)
            {
                row_storage = rc.get_storage(norm);
            }
            #pragma unroll
            for(int i = LDPC2_A107_CORE_EMIT_LEAD; i < UPDATE_ROW_DEGREE; ++i)
            {
                word_t inc = rc.extract_signed_magnitude(i);
                inc.u32    = inc.u32 ^ (app[i].u32 & 0x80008000u);
                app[i]     = app_combine_x2(app[i], inc, norm);
            }
        }
        else
#endif
        {
            #pragma unroll
            for(int i = 0; i < UPDATE_ROW_DEGREE; ++i)
            {
                word_t inc = rc.extract_signed_magnitude(i);
                inc.u32    = inc.u32 ^ (app[i].u32 & 0x80008000u);
                app[i]     = app_combine_x2(app[i], inc, norm);
            }
        }
#else
        #pragma unroll
        for(int i = 0; i < UPDATE_ROW_DEGREE; ++i)
        {
            const int shift_i = (i < 10) ? (9 - i) : (18 - i);
            if((shift_i != 0) && (i != 0) && (i != 5) && (i != 10) && (i != 15))
            {
                word_t inc = rc.extract_signed_magnitude(i);
                inc.u32    = inc.u32 ^ (app[i].u32 & 0x80008000u);
                app[i]     = app_combine_x2(app[i], inc, norm);
            }
            else
            {
                word_t inc = rc.extract_pair(i);
                app[i]     = app_combine_x2(app[i], inc, norm);
            }
        }
#endif
#else
        #pragma unroll
        for(int i = 0; i < UPDATE_ROW_DEGREE; ++i)
        {
            word_t inc = rc.extract_pair(i);
            app[i] = app_combine_x2(app[i], inc, norm);
        }
#endif
#else
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Extract values from the new row context and update the APP
        // values.
        #pragma unroll
        for(int i = 0; i < UPDATE_ROW_DEGREE; ++i)
        {
            word_t inc = rc.extract_pair(i);
            // TODO: Fuse multiply-add of normalization in finalize()
            // here
            app[i].f16x2 = __hadd2(app[i].f16x2, inc.f16x2);
        }
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Store the compressed representation in row storage (dead on the
        // final iteration -- see IS_LAST note above).
        if constexpr (!IS_LAST)
        {
            row_storage = rc.get_storage();
        }
#endif
    }
};

////////////////////////////////////////////////////////////////////////
// cC2V_index
// Specialization for two codewords at a time as independent "high" and
// "low" fp16 values in a __half2 APP data type.
template <int                                 BG,
          class                               TSignManager,
          class                               TMinSumUpdate,
          template <typename, int> class      TAPPLoader,
          template <typename, int, int> class TAPPWriter>
struct cC2V_index<__half2, BG, TSignManager, TMinSumUpdate, TAPPLoader, TAPPWriter>
{
    //typedef C2V_storage_t<__half2, 4>                                             c2v_storage_t;
    //typedef cC2V_row_context<__half2, TSignManager, TMinSumUpdate, c2v_storage_t> row_context_t;
    typedef __half2                                                               app_t;
    //------------------------------------------------------------------
    // app_sub_prev_iter()
    template <int CHECK_IDX, class TRowContext>
    __device__
    void app_sub_prev_iter(word_t             (&app)[app_num_words<__half2, BG, CHECK_IDX>::value],
                           const TRowContext& rc)
    {
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Update the APP values with the decrement from the previous
        // iteration. We don't need to update APP values for extension
        // nodes.
        #pragma unroll
        for(int i = 0; i < update_row_degree<BG, CHECK_IDX>::value; ++i)
        {
            // Operate on a pair of values at a time
            word_t dec = rc.extract_pair(i);

            // TODO: Fuse normalization from prev iteration?
            app[i].f16x2 = __hsub2(app[i].f16x2, dec.f16x2);
        }
    }
    //------------------------------------------------------------------
    // app_update()
    template <int CHECK_IDX, class TRowContext>
    __device__
    void app_update(word_t             (&app)[app_num_words<__half2, BG, CHECK_IDX>::value],
                    const TRowContext& rc)
    {
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Extract values from the new row context and update the APP
        // values.
        #pragma unroll
        for(int i = 0; i < update_row_degree<BG, CHECK_IDX>::value; ++i)
        {
            word_t inc = rc.extract_pair(i);
            // TODO: Fuse multiply-add of normalization in finalize()
            // here
            app[i].f16x2 = __hadd2(app[i].f16x2, inc.f16x2);
        }
    }
    //------------------------------------------------------------------
    // process_row()
    template <int CHECK_IDX, class TKernelParams, class TC2VStorage>
    __device__
    void process_row(const TKernelParams& params,
                     word_t               (&app)[app_num_words<__half2, BG, CHECK_IDX>::value],
                     int                  (&app_addr)[row_degree<BG, CHECK_IDX>::value],
                     TC2VStorage&         c2v_storage,
                     int                  smem_offset)
    {
        typedef cC2V_row_context<__half2,
                                 TSignManager,
                                 TMinSumUpdate,
                                 TC2VStorage> row_context_t;
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Load APP values into registers
        typedef TAPPLoader<__half2, row_degree<BG, CHECK_IDX>::value> app_loader_t;
        app_loader_t::load(app, app_addr, smem_offset);
        {
            //- - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            // Initialize a row processing context using data from the
            // previous iteration (which may be stored in registers or
            // global memory).
            row_context_t rc(c2v_storage,
                             std::integral_constant<int, row_degree<BG, CHECK_IDX>::value>{});

            //- - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            // Update the APP values with the decrement from the previous
            // iteration.
            app_sub_prev_iter<CHECK_IDX>(app, rc);
        }
        {
            //- - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            // Construct a row context using the updated APP values (with
            // values from the previous iteration subtracted). This will
            // create a min-sum representation of the APP values.
            row_context_t rcNew(params.norm.f16x2,
                                app,
                                std::integral_constant<int, row_degree<BG, CHECK_IDX>::value>{});

            //- - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            // Store compressed representation in member variable. It may
            // (or may not) separately be saved to global memory.
            c2v_storage = rcNew.get_storage();

            //- - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            // Extract values from the new row context and update the APP
            // values.
            app_update<CHECK_IDX>(app, rcNew);
        }
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Write output values to shared_memory
        typedef TAPPWriter<__half2,
                           row_degree       <BG, CHECK_IDX>::value,
                           update_row_degree<BG, CHECK_IDX>::value> app_writer_t;
        app_writer_t::write_non_ext(app, app_addr, smem_offset);
    }
    //------------------------------------------------------------------
    // load_shared_strided()
    //__device__
    //void load_shared_strided(const LDPC_kernel_params& params,
    //                         int                       checkIdx,
    //                         int                       threadByteOffset,
    //                         int                       strideBytes)
    //{
    //    if((BG != 1) || checkIdx >= 4)
    //    {
    //        #pragma unroll
    //        for(int i = 0; i < c2v_storage_t::NUM_WORDS_SMALL; ++i)
    //        {
    //            c2v_storage.v.words[i] = smem_address_as<word_t>(threadByteOffset + (i * strideBytes));
    //        }
    //    }
    //    else
    //    {
    //        #pragma unroll
    //        for(int i = 0; i < c2v_storage_t::NUM_WORDS; ++i)
    //        {
    //            c2v_storage.v.words[i] = smem_address_as<word_t>(threadByteOffset + (i * strideBytes));
    //        }
    //    }
    //}
    //------------------------------------------------------------------
    // store_shared_strided()
    //__device__
    //void store_shared_strided(const LDPC_kernel_params& params,
    //                          int                       checkIdx,
    //                          int                       threadByteOffset,
    //                          int                       strideBytes)
    //{
    //    if((BG != 1) || checkIdx >= 4)
    //    {
    //        #pragma unroll
    //        for(int i = 0; i < c2v_storage_t::NUM_WORDS_SMALL; ++i)
    //        {
    //            write_shared_word(c2v_storage.v.words[i], threadByteOffset + (i * strideBytes));
    //        }
    //    }
    //    else
    //    {
    //        #pragma unroll
    //        for(int i = 0; i < c2v_storage_t::NUM_WORDS; ++i)
    //        {
    //            write_shared_word(c2v_storage.v.words[i], threadByteOffset + (i * strideBytes));
    //        }
    //    }
    //}
};

//------------------------------------------------------------------
// "Core" rows in the 5G base graphs are the first 4 rows, and for
// each base graph these have the highest row degree (19 for BG1,
// 10 for BG2).
template <int BG> struct core_storage_x2;
template <> struct core_storage_x2<1>
{
    typedef cC2V_storage_x2_high_degree_split type;
};
template <> struct core_storage_x2<2>
{
    typedef cC2V_storage_x2_low_degree type;
};

////////////////////////////////////////////////////////////////////////
// box_plus_row_proc_x2
// Row processor for a box plus implementation that processes two
// codewords at a time. The low fp16 values of each word in the APP
// array have the values for one codeword, and the high fp16 values
// correspond to the second codeword.
template <class TBoxPlusOp>
class box_plus_row_proc_x2
{
public:
    //typedef TC2VStorage storage_t;
    //------------------------------------------------------------------
    // init()
    //__device__ static void init(storage_t& s) { s.init(); }
    //------------------------------------------------------------------
    // process_row()
    template <int ROW_DEGREE, int UPDATE_ROW_DEGREE, class TStorage>
    __device__
    static void process_row(word_t          (&app)[row_num_words<__half, ROW_DEGREE>::value],
                            TStorage&       row_storage,
                            const __half2&  norm)
    {
        app_sub_prev_iter<ROW_DEGREE, UPDATE_ROW_DEGREE>(app, row_storage);
        app_update<ROW_DEGREE, UPDATE_ROW_DEGREE>(row_storage, norm, app);
    }
private:
    //------------------------------------------------------------------
    // app_sub_prev_iter()
    template <int ROW_DEGREE, int UPDATE_ROW_DEGREE, class TStorage>
    __device__
    static void app_sub_prev_iter(word_t          (&app)[ROW_DEGREE],
                                  const TStorage& row_storage)
    {
        #pragma unroll
        for(int i = 0; i < UPDATE_ROW_DEGREE; ++i)
        {
            app[i].f16x2 = __hsub2(app[i].f16x2, row_storage.v.w[i].f16x2);
        }
    }
    //------------------------------------------------------------------
    // app_update()
    template <int ROW_DEGREE, int UPDATE_ROW_DEGREE, class TStorage>
    __device__
    static void app_update(TStorage&            row_storage,
                           const __half2&       norm,
                           word_t               (&app)[ROW_DEGREE])
    {
        word_t bp_update_seq[UPDATE_ROW_DEGREE];  // temporary storage

        //typedef box_plus_seq_gen<TBoxPlusOp, ROW_DEGREE, UPDATE_ROW_DEGREE> box_plus_seq_gen_t;

        //box_plus_seq_gen_t::generate(bp_update_seq, app);
        //#pragma unroll
        //for(int i = 0; i < UPDATE_ROW_DEGREE; ++i)
        //{
        //    // Apply normalization and store for next iteration
        //    row_storage.v.w[i].f16x2 = __hmul2(bp_update_seq[i].f16x2, norm);
        //    // Update APP
        //    app[i].f16x2 = __hadd2(row_storage.v.w[i].f16x2, app[i].f16x2);
        //}
    }
};

} // namespace ldpc2

#endif // !defined(LDPC2_C2V_X2_CUH_INCLUDED_)
