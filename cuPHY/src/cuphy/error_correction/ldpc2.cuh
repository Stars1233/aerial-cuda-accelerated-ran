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

#if !defined(LDPC2_CUH_INCLUDED_)
#define LDPC2_CUH_INCLUDED_

#include <stdint.h>
#include "nrLDPC_templates.cuh"
#include "cuphy_kernel_util.cuh"
#include "ldpc2.hpp"

// Macros to allow "global" setting of volatile for asm statements. In
// theory, not setting volatile might allow the compiler more freedom,
// but in some tests, code generation was better with volatile throughout.
#define LDPC2_ASM asm volatile
#define LDPC2_ASM_VOLATILE asm volatile

namespace ldpc2
{

union half_word_t
{
    uint16_t u16;
    __half   f16;
};

////////////////////////////////////////////////////////////////////////
// app_max_words
// Provides the maximum number of words (registers) required to store APP
// variables as a function of APP type (fp32 or fp16) and base graph
// index (1 or 2)
template <typename T, int BG> struct app_max_words;
template <int BG>
struct app_max_words<float, BG>  { static const int value = max_row_degree<BG>::value; };
template <int BG>
struct app_max_words<__half, BG> { static const int value = round_up_t<max_row_degree<BG>::value, 2>::value; };
// For __half2, two codewords are being processed at a time, using the
// (independent) high and low halves of a word. Therefore, the maximum
// number of words in this case is equal to the row degree.
template <int BG>
struct app_max_words<__half2, BG>  { static const int value = max_row_degree<BG>::value; };

////////////////////////////////////////////////////////////////////////
// app_num_words
// Provides the number of words (registers) required to store APP
// variables as a function of APP type (fp32 or fp16), base graph index,
// and parity node row.
// For the float type, we store 1 APP value in each register. For __half
// we store 2 APP values in a __half2 structure.
template <typename T, int BG, int CHECK_IDX> struct app_num_words;
template <int BG, int CHECK_IDX>
struct app_num_words<float, BG, CHECK_IDX>  { static const int value = row_degree<BG, CHECK_IDX>::value; };
template <int BG, int CHECK_IDX>
struct app_num_words<__half, BG, CHECK_IDX> { static const int value = div_round_up_t<row_degree<BG, CHECK_IDX>::value, 2>::value; };
// For __half2, two codewords are being processed at a time, using the
// (independent) high and low halves of a word. Therefore, the maximum
// number of words in this case is equal to the row degree.
template <int BG, int CHECK_IDX>
struct app_num_words<__half2, BG, CHECK_IDX> { static const int value = row_degree<BG, CHECK_IDX>::value; };
template <int BG, int CHECK_IDX>
struct app_num_words<__nv_fp8_e4m3, BG, CHECK_IDX> { static const int value = div_round_up_t<row_degree<BG, CHECK_IDX>::value, 4>::value; };
template <int BG, int CHECK_IDX>
struct app_num_words<__nv_fp8_e5m2, BG, CHECK_IDX> { static const int value = div_round_up_t<row_degree<BG, CHECK_IDX>::value, 4>::value; };


////////////////////////////////////////////////////////////////////////
// row_num_words
// Provides the number of words (registers) required to store APP
// variables as a function of APP type (fp32 or fp16) and row degree.
// For the float type, we store 1 APP value in each register. For __half
// we store 2 APP values in a __half2 structure.
template <typename T, int ROW_DEGREE> struct row_num_words;
template <int ROW_DEGREE>
struct row_num_words<float, ROW_DEGREE>   { static const int value = ROW_DEGREE; };
template <int ROW_DEGREE>
struct row_num_words<__half, ROW_DEGREE>  { static const int value = div_round_up_t<ROW_DEGREE, 2>::value; };
template <int ROW_DEGREE>
struct row_num_words<__half2, ROW_DEGREE> { static const int value = ROW_DEGREE; };
template <int ROW_DEGREE>
struct row_num_words<__nv_fp8_e5m2, ROW_DEGREE>  { static const int value = div_round_up_t<ROW_DEGREE, 4>::value; };
template <int ROW_DEGREE>
struct row_num_words<__nv_fp8_e4m3, ROW_DEGREE>  { static const int value = div_round_up_t<ROW_DEGREE, 4>::value; };

////////////////////////////////////////////////////////////////////////
// signbit
__device__ inline
int signbit(__half h)
{
    half_word_t hw;
    hw.f16 = h;
    return ((hw.u16 & 0x00008000) >> 15);
}

////////////////////////////////////////////////////////////////////////
// llr_hard_decision
__device__ inline
int llr_hard_decision(__half h)
{
#if 0
    // h => 0   => 0
    // h < 0    => 1
    return signbit(h);
#else
    // We choose to interpret a zero value as a 1 bit so that the LDPC decoder
    // does not have all zero hard outputs for degenerate cases of all zero
    // inputs.
    // h > 0    => 0
    // h <= 0   => 1
    return (h > static_cast<__half>(0.0)) ? 0 : 1;
#endif
}

////////////////////////////////////////////////////////////////////////
// llr_hard_decision
__device__ inline
int llr_hard_decision(__nv_fp8_e5m2 q)
{
    // q > 0    => 0
    // q <= 0   => 1
    // zero representation: S.00000.00
    // In the absence of an instruction to compare an fp8 value to zero,
    // we interpret the storage as a signed integer. Negative values
    // and zero will have a 1 for the hard decision.
    // The fp8 storage type is unsigned char, and converting to a signed
    // value when the destination type cannot hold the value is
    // implementation defined. Instead or relying on what is probably
    // the most common implementation (i.e. keeping the same 2s complement
    // representation), we copy to a signed integer and rely on the
    // compiler to make this efficient. (Tests with compiler explorer
    // confirm that the generated code is efficient.)
    static_assert(sizeof(q.__x) == sizeof(int8_t));
    int8_t i;
    memcpy(&i, &q.__x, sizeof(i));
    return (i > 0) ? 0 : 1;
}

////////////////////////////////////////////////////////////////////////
// llr_hard_decision
__device__ inline
int llr_hard_decision(__nv_fp8_e4m3 q)
{
    // q > 0    => 0
    // q <= 0   => 1
    // zero representation: S.0000.000
    // In the absence of an instruction to compare an fp8 value to zero,
    // we interpret the storage as a signed integer. Negative values
    // and zero will have a 1 for the hard decision.
    // The fp8 storage type is unsigned char, and converting to a signed
    // value when the destination type cannot hold the value is
    // implementation defined. Instead or relying on what is probably
    // the most common implementation (i.e. keeping the same 2s complement
    // representation), we copy to a signed integer and rely on the
    // compiler to make this efficient. (Tests with compiler explorer
    // confirm that the generated code is efficient.)
    static_assert(sizeof(q.__x) == sizeof(int8_t));
    int8_t i;
    memcpy(&i, &q.__x, sizeof(i));
    return (i > 0) ? 0 : 1;
}

////////////////////////////////////////////////////////////////////////
// llr_hard_decision
__device__ inline
int llr_hard_decision(int8_t i)
{
    return (i > 0) ? 0 : 1;
}

////////////////////////////////////////////////////////////////////////
// word_sign_mask()
// Returns a word with the sign bit of the input value, and all other
// bits 0.
__device__ inline
word_t word_sign_mask(word_t w)
{
    word_t out;
    out.u32 = w.u32 & 0x80000000;
    return out;
}

////////////////////////////////////////////////////////////////////////
// word_pair_sign_mask()
// Returns a word with the sign bits of the two fp16 input values, and
// all other bits 0.
__device__ inline
word_t word_pair_sign_mask(word_t w)
{
    word_t out;
    out.u32 = w.u32 & 0x80008000;
    return out;
}

////////////////////////////////////////////////////////////////////////
// word_pair_abs()
__device__ inline
word_t word_pair_abs(word_t w)
{
    word_t out;
    out.u32 = w.u32 & 0x7FFF7FFF;
    return out;
}

////////////////////////////////////////////////////////////////////////
// half_word_sign_mask()
// Returns a word with the sign bit of the input value, and all other
// bits 0.
__device__ inline
half_word_t half_word_sign_mask(half_word_t w)
{
    half_word_t out;
    out.u16 = w.u16 & 0x00008000;
    return out;
}

////////////////////////////////////////////////////////////////////////
// half_word_sign_mask()
// Returns a word with the sign bit of the input value, and all other
// bits 0.
__device__ inline
word_t half_word_sign_mask(word_t w)
{
    word_t out;
    out.u32 = (w.u32 & 0x00008000);
    return out;
}

////////////////////////////////////////////////////////////////////////
// set_high_zero()
__device__ inline
word_t set_high_zero(word_t win)
{
    word_t out;
    LDPC2_ASM("prmt.b32 %0, %1, 0, 0x5410;\n"
              : "=r"(out.u32)
              : "r"(win.u32));
    return out;
}

////////////////////////////////////////////////////////////////////////
// write_shared_int2()
__device__ inline
void write_shared_int2(int2 i2, int offset)
{
    union u
    {
        int2     i2;
        uint64_t u64;
    };
    u val;
    val.i2 = i2;
    LDPC2_ASM_VOLATILE("st.shared.b64 [%0], %1;\n" :: "r"(offset), "l"(val.u64) : "memory");
}

////////////////////////////////////////////////////////////////////////
// write_shared_int4()
__device__ inline
void write_shared_int4(int4 i4, int offset)
{
    LDPC2_ASM_VOLATILE("st.shared.v4.s32 [%0], { %1, %2, %3, %4 };\n" :: "r"(offset), "r"(i4.x), "r"(i4.y), "r"(i4.z), "r"(i4.w) : "memory");
}

////////////////////////////////////////////////////////////////////////
// write_shared_word()
__device__ inline
void write_shared_word(word_t w, int offset)
{
    LDPC2_ASM_VOLATILE("st.shared.b32 [%0], %1;\n" :: "r"(offset), "r"(w.u32) : "memory");
}

////////////////////////////////////////////////////////////////////////
// write_shared_word_low()
__device__ inline
void write_shared_word_low(word_t w, int offset)
{
    half_word_t wh;
    wh.u16 = w.u32;
    LDPC2_ASM_VOLATILE("st.shared.b16 [%0], %1;\n" :: "r"(offset), "h"(wh.u16) : "memory");
}

////////////////////////////////////////////////////////////////////////
// write_shared_word_high()
__device__ inline
void write_shared_word_high(word_t w, int offset)
{
    half_word_t wh;
    wh.u16 = w.u32 >> 16;
    LDPC2_ASM_VOLATILE("st.shared.b16 [%0], %1;\n" :: "r"(offset), "h"(wh.u16) : "memory");
}

__device__ inline
//void write_shared_uchar(unsigned char c, int offset)
void write_shared_uchar(uint32_t u, int offset)
{
#if 1
    //uint32_t u = c;
    // Passing an unsigned char and casting to a 32-bit value (for use with
    // inline assembly) can cause the compiler to emit AND with 0xFF before
    // writing to shared memory.
    // TODO: Stop using integer shared memory offsets and use proper
    // shared memory pointers.
    LDPC2_ASM_VOLATILE("st.shared.u8 [%0], %1;\n" :: "r"(offset), "r"(u) : "memory");
#else
    intptr_t i = offset;
    unsigned char* dst = (unsigned char*)__cvta_generic_to_shared((void*)i);
    *dst = c;
#endif
}

////////////////////////////////////////////////////////////////////////
// h0_h0()
// Utility function to attempt to coerce the compiler into using the
// .H0_H0 swizzle on instructions that support it (HSET2, HSETP2, ...)
__device__ inline
word_t h0_h0(word_t w_in)
{
    word_t w_out;
    w_out.f16x2 = __low2half2(w_in.f16x2);
    return w_out;
}

////////////////////////////////////////////////////////////////////////
// h0_h0()
// Utility function to attempt to coerce the compiler into using the
// .H0_H0 swizzle on instructions that support it (HSET2, HSETP2, ...)
// Note: input must be greater than or equal to 0 and less than or
// equal to USHRT_MAX.
__device__ inline
word_t h0_h0(int i)
{
    word_t w_out, w_in;
    w_in.f16x2.x = static_cast<unsigned short>(i);
    w_out.f16x2 = __low2half2(w_in.f16x2);
    return w_out;
}

////////////////////////////////////////////////////////////////////////
// h0_h0()
// Utility function to attempt to coerce the compiler into using the
// .H0_H0 swizzle on instructions that support it (HSET2, HSETP2, ...)
// Note: input must be greater than or equal to 0 and less than or
// equal to USHRT_MAX.
__device__ inline
word_t h0_h0(unsigned int u)
{
    word_t w_out, w_in;
    w_in.f16x2.x = static_cast<unsigned short>(u);
    w_out.f16x2 = __low2half2(w_in.f16x2);
    return w_out;
}

////////////////////////////////////////////////////////////////////////
// h1_h1()
// Utility function to attempt to coerce the compiler into using the
// .H1_H1 swizzle on instructions that support it (HSET2, HSETP2, ...)
__device__ inline
word_t h1_h1(word_t w_in)
{
    word_t w_out;
    w_out.f16x2 = __high2half2(w_in.f16x2);
    return w_out;
}

__device__ inline
word_t to_word(short2 s2)
{
    word_t w_out;
    w_out.i16x2 = s2;
    return w_out;
}

////////////////////////////////////////////////////////////////////////
// hset2_bf_lt()
// fp16 pairwise 'less than' comparison, with (fp16)1.0 output for true
// and (fp16)0.0 output for false.
__device__ inline
word_t hset2_bf_lt(word_t inA, word_t inB)
{
    word_t ret;
    LDPC2_ASM("set.lt.f16x2.f16x2 %0, %1, %2;"
              : "=r"(ret.u32)                // outputs
              : "r"(inA.u32), "r"(inB.u32)); // inputs
    return ret;
}

////////////////////////////////////////////////////////////////////////
// hset2_bf_le()
// fp16 pairwise 'less than or equal' comparison, with (fp16)1.0 output for true
// and (fp16)0.0 output for false.
__device__ inline
word_t hset2_bf_le(word_t inA, word_t inB)
{
    word_t ret;
    LDPC2_ASM("set.le.f16x2.f16x2 %0, %1, %2;"
              : "=r"(ret.u32)                // outputs
              : "r"(inA.u32), "r"(inB.u32)); // inputs
    return ret;
}

////////////////////////////////////////////////////////////////////////
// hset2_bf_gt()
// fp16 pairwise 'greater than' comparison, with (fp16)1.0
// output for true and (fp16)0.0 output for false.
__device__ inline
word_t hset2_bf_gt(word_t inA, word_t inB)
{
    word_t ret;
    LDPC2_ASM("set.gt.f16x2.f16x2 %0, %1, %2;"
              : "=r"(ret.u32)                // outputs
              : "r"(inA.u32), "r"(inB.u32)); // inputs
    return ret;
}

////////////////////////////////////////////////////////////////////////
// hset2_bf_ge()
// fp16 pairwise 'greater than or equal' comparison, with (fp16)1.0
// output for true and (fp16)0.0 output for false.
__device__ inline
word_t hset2_bf_ge(word_t inA, word_t inB)
{
    word_t ret;
    LDPC2_ASM("set.ge.f16x2.f16x2 %0, %1, %2;"
              : "=r"(ret.u32)                // outputs
              : "r"(inA.u32), "r"(inB.u32)); // inputs
    return ret;
}

////////////////////////////////////////////////////////////////////////
// hset2_bf_eq()
// fp16 pairwise 'equal' comparison, with (fp16)1.0 output for true
// and (fp16)0.0 output for false. (".bf" => boolean-float result, as
// opposed to the ".bm" bitmask variants.) Used to build a 1.0/0.0
// selector for an FMA-based leave-one-out reconstruction, keeping the
// per-column magnitude select off the saturated integer/LOP3 pipe.
__device__ inline
word_t hset2_bf_eq(word_t inA, word_t inB)
{
    word_t ret;
    LDPC2_ASM("set.equ.f16x2.f16x2 %0, %1, %2;"
              : "=r"(ret.u32)                // outputs
              : "r"(inA.u32), "r"(inB.u32)); // inputs
    return ret;
}

////////////////////////////////////////////////////////////////////////
// hfma2()
// (A * B) + c
// Denormalized number support is maintained (i.e. no FTZ)
__device__ inline
word_t hfma2(word_t inA, word_t inB, word_t inC)
{
    word_t ret;
    LDPC2_ASM("fma.rn.f16x2 %0, %1, %2, %3;"
              : "=r"(ret.u32)                              // outputs
              : "r"(inA.u32), "r"(inB.u32), "r"(inC.u32)); // inputs
    return ret;
}

////////////////////////////////////////////////////////////////////////
// hneg2()
// Negates both fp16 values of a half2
__device__ inline
word_t hneg2(word_t inA)
{
    word_t ret;
    LDPC2_ASM("neg.f16x2 %0, %1;"
              : "=r"(ret.u32)     // outputs
              : "r"(inA.u32));    // inputs
    return ret;
}

////////////////////////////////////////////////////////////////////////
// hadd2()
__device__ inline
word_t hadd2(word_t inA, word_t inB)
{
    word_t ret;
    LDPC2_ASM("add.f16x2 %0, %1, %2;"
              : "=r"(ret.u32)                // outputs
              : "r"(inA.u32), "r"(inB.u32)); // inputs
    return ret;
}

////////////////////////////////////////////////////////////////////////
// half2_has_inf()
inline __device__
bool half2_has_inf(const __half2 h)
{
    __half low = __low2half(h);
    __half high = __high2half(h);
    return (__hisinf(low) | __hisinf(high));
}

////////////////////////////////////////////////////////////////////////
// half2_has_nan()
inline __device__
bool half2_has_nan(const __half2 h)
{
    __half low = __low2half(h);
    __half high = __high2half(h);
    return (__hisnan(low) | __hisnan(high));
}

////////////////////////////////////////////////////////////////////////
// hset2_bm_lt()
// Performs unordered less than comparison of two halves of input half2
// words. The output is 0x0000 0000, 0xFFFF 0000, 0x0000 FFFF, or
// 0xFFFF FFFF, where 0xFFFF represents "less than" and 0x0000
// represents "not less than."
static inline __device__ word_t hset2_bm_lt(word_t a, word_t b)
{
    word_t wOut;
#if ((__CUDACC_VER_MAJOR__ == 10) && (__CUDACC_VER_MINOR__ >= 2)) || (__CUDACC_VER_MAJOR__ >= 11)
    LDPC2_ASM("{\n"
              "set.ltu.u32.f16x2 %0, %1, %2;\n" // compare [a_hi|a_lo] to [b_hi|b_lo]
              "}\n"
              : "=r"(wOut.u32)
              : "r"(a.u32), "r"(b.u32));
#else
    LDPC2_ASM("{\n"
              ".reg .pred a0_lt_b0, a1_lt_b1;\n\t\t"
              "mov.u32 %0, 0xFFFFFFFF;\n\t\t"                    // initializing assuming true for both comparisons
              "setp.ltu.f16x2 a0_lt_b0|a1_lt_b1, %1, %2;\n\t\t"  // compare [a1 a0] to [b1 b0] --> [a1_lt_b1 a0_lt_b1]
              "@!a0_lt_b0 prmt.b32 %0, %0, 0, 0x3254;\n\t"       // set out0 to 0 if the comparison was false
              "@!a1_lt_b1 prmt.b32 %0, %0, 0, 0x5410;\n\t"       // set out1 to 0 if the comparison was false
              "}\n"
              : "=r"(wOut.u32)
              : "r"(a.u32), "r"(b.u32));
#endif
    return wOut;
}

////////////////////////////////////////////////////////////////////////
// hset2_bm_gt()
// Performs unordered greater than comparison of two halves of input half2
// words. The output is 0x0000 0000, 0xFFFF 0000, 0x0000 FFFF, or
// 0xFFFF FFFF, where 0xFFFF represents "greater than" and 0x0000
// represents "not greater than."
static inline __device__ word_t hset2_bm_gt(word_t a, word_t b)
{
    word_t wOut;
#if ((__CUDACC_VER_MAJOR__ == 10) && (__CUDACC_VER_MINOR__ >= 2)) || (__CUDACC_VER_MAJOR__ >= 11)
    LDPC2_ASM("{\n"
              "set.gtu.u32.f16x2 %0, %1, %2;\n" // compare [a_hi|a_lo] to [b_hi|b_lo]
              "}\n"
              : "=r"(wOut.u32)
              : "r"(a.u32), "r"(b.u32));
#else
    LDPC2_ASM("{\n"
              ".reg .pred a0_gt_b0, a1_gt_b1;\n\t\t"
              "mov.u32 %0, 0xFFFFFFFF;\n\t\t"                    // initializing assuming true for both comparisons
              "setp.gtu.f16x2 a0_gt_b0|a1_gt_b1, %1, %2;\n\t\t"  // compare [a1 a0] to [b1 b0] --> [a1_gt_b1 a0_gt_b1]
              "@!a0_gt_b0 prmt.b32 %0, %0, 0, 0x3254;\n\t"       // set out0 to 0 if the comparison was false
              "@!a1_gt_b1 prmt.b32 %0, %0, 0, 0x5410;\n\t"       // set out1 to 0 if the comparison was false
              "}\n"
              : "=r"(wOut.u32)
              : "r"(a.u32), "r"(b.u32));
#endif
    return wOut;
}

////////////////////////////////////////////////////////////////////////
// hequ_bm()
// Performs (unordered) equality comparison of two halves of input half2
// words. The output is 0x0000 0000, 0xFFFF 0000, 0x0000 FFFF, or
// 0xFFFF FFFF, where 0xFFFF represents "equal" and 0x0000
// represents "not equal."
static inline __device__ word_t hequ_bm(word_t a, word_t b)
{
    word_t wOut;
#if ((__CUDACC_VER_MAJOR__ == 10) && (__CUDACC_VER_MINOR__ >= 2)) || (__CUDACC_VER_MAJOR__ >= 11)
    LDPC2_ASM("{\n"
              "set.equ.u32.f16x2 %0, %1, %2;\n" // compare [a_hi|a_lo] to [b_hi|b_lo]
              "}\n"
              : "=r"(wOut.u32)
              : "r"(a.u32), "r"(b.u32));
#else
    LDPC2_ASM("{\n"
              ".reg .pred a0_lt_b0, a1_lt_b1;\n\t\t"
              "mov.u32 %0, 0xFFFFFFFF;\n\t\t"                    // initializing assuming true for both comparisons
              "setp.equ.f16x2 a0_lt_b0|a1_lt_b1, %1, %2;\n\t\t"  // compare [a1 a0] to [b1 b0] --> [a1_lt_b1 a0_lt_b1]
              "@!a0_lt_b0 prmt.b32 %0, %0, 0, 0x3254;\n\t"       // set out0 to 0 if the comparison was false
              "@!a1_lt_b1 prmt.b32 %0, %0, 0, 0x5410;\n\t"       // set out1 to 0 if the comparison was false
              "}\n"
              : "=r"(wOut.u32)
              : "r"(a.u32), "r"(b.u32));
#endif
    return wOut;
}

////////////////////////////////////////////////////////////////////////
// update_signs_fp_pair()
__device__ inline
uint32_t update_signs_fp_pair(uint32_t s_input, word_t value_pair, int idx_pair)
{
    word_t A, B, C, z, res;
    z.u32 = 0;
    C.u32 = s_input;
    // Sets word_t A to have a pair of fp16 values:
    // - 1.0 if the hi/lo value is less than zero
    // - 0.0 otherwise
    A = hset2_bf_le(value_pair, z);
    B.u32 = 0x00010001 << idx_pair;
    // Perform multiplication with B as a denormalized float, and add the
    // result to the input value C.
    res = hfma2(A, B, C);
    return res.u32;
}
////////////////////////////////////////////////////////////////////////
// update_signs_fp_low()
__device__ inline uint32_t update_signs_fp_low(uint32_t s_input, word_t h0, int idx_pair)
{
    word_t A, B, C, z, res;
    z.u32 = 0;
    C.u32 = s_input;
    A = hset2_bf_le(h0, z);
    B.u32 = 0x00000001 << idx_pair;
    res = hfma2(A, B, C);
    return res.u32;
}

////////////////////////////////////////////////////////////////////////
// dp_signs_packed_operand
// Constant value for inline PTX for varying numbers of valid inputs
template <int COUNT> struct dp_signs_packed_operand;
template <> struct dp_signs_packed_operand<4> { static const unsigned int value = 0xF8FCFEFF; };
template <> struct dp_signs_packed_operand<3> { static const unsigned int value = 0x00FCFEFF; };
template <> struct dp_signs_packed_operand<2> { static const unsigned int value = 0x0000FEFF; };
template <> struct dp_signs_packed_operand<1> { static const unsigned int value = 0x000000FF; };

////////////////////////////////////////////////////////////////////////
// dp_signs_packed_operand
// Constant value for inline PTX for varying numbers of valid inputs
template <int COUNT> struct dp_signs_spread_operand;
template <> struct dp_signs_spread_operand<4> { static const unsigned int value = 0x80E0F8FE; };
template <> struct dp_signs_spread_operand<3> { static const unsigned int value = 0x00E0F8FE; };
template <> struct dp_signs_spread_operand<2> { static const unsigned int value = 0x0000F8FE; };
template <> struct dp_signs_spread_operand<1> { static const unsigned int value = 0x000000FE; };

////////////////////////////////////////////////////////////////////////
// dp_signs_packed()
// Extraction of signs using Victor Podlozhnyuk's SIMD formulation
// (PRMT + DP4A).
// The PRMT instruction indices will sign extend the source bit if the
// high bit of the index values is set. Combining this with a signed
// dot product allows us to extract 4 sign bits with only 2 instructions.
// PRMT:
// The sign bits of the 4 input half values are at bytes 7, 5, 3, and 1.
// Setting the high bit (8) of each index yields 8+7, 8+5, 8+3, 8+1, or
// F, D, B, 9.
// DP4A:
// The PRMT output will be either 0xFF or 0x0000, interpreted as a
// signed 8-bit value. To set bits 3, 2, 1, and 0 in the output, we
// multiply by -8 (0xF8), -4 (0xFC), -2 (0xFE), and -1 (0xFF)
template <int COUNT>
inline
__device__ uint32_t dp_signs_packed(word_t a0, word_t a1)
{
    uint32_t result = 0;
    LDPC2_ASM("\t"
              "{\n\t\t"
              ".reg .s32 sx;\n\t\t"
              "prmt.b32 sx, %1, %2, 0xFDB9;\n\t\t"
              "dp4a.s32.s32 %0, sx, %3, 0;\n\t\t"
              "}\n\t"       :
              "=r"(result)  :
              "r"(a0.i32),
              "r"(a1.i32),
              "n"(dp_signs_packed_operand<COUNT>::value));
    return result;
}

////////////////////////////////////////////////////////////////////////
// dp_signs_spread()
// Extraction of signs using Victor Podlozhnyuk's SIMD formulation
// (PRMT + DP4A).
// The PRMT instruction indices will sign extend the source bit if the
// high bit of the index values is set. Combining this with a signed
// dot product allows us to extract 4 sign bits with only 2 instructions.
// PRMT:
// The sign bits of the 4 input half values are at bytes 7, 5, 3, and 1.
// Setting the high bit (8) of each index yields 8+7, 8+5, 8+3, 8+1, or
// F, D, B, 9.
// DP4A:
// The PRMT output will be either 0xFF or 0x0000, interpreted as a
// signed 8-bit value. To set bits 7, 5, 3, and 1 in the output, we
// multiply by -128 (0x80), -32 (0xE0), -8 (0xF8), and -2 (0xFE)
template <int COUNT>
inline
__device__ uint32_t dp_signs_spread(word_t a0, word_t a1)
{
    uint32_t result = 0;
    LDPC2_ASM("\t"
              "{\n\t\t"
              ".reg .s32 sx;\n\t\t"
              "prmt.b32 sx, %1, %2, 0xFDB9;\n\t\t"
              "dp4a.s32.s32 %0, sx, %3, 0;\n\t\t"
              "}\n\t"       :
              "=r"(result)  :
              "r"(a0.i32),
              "r"(a1.i32),
              "n"(dp_signs_spread_operand<COUNT>::value));
    return result;
}

// Function to generate a lop3 PTX instruction, performing
// a logical operation on 3 inputs.
template <int LUT>
__device__
word_t lop3(word_t a, word_t b, word_t c)
{
    word_t out;
    LDPC2_ASM("lop3.b32 %0, %1, %2, %3, %4;\n" :
              "=r"(out.u32) :
              "r"(a.u32), "r"(b.u32), "r"(c.u32), "n"(LUT));
    return out;
}

// Function to generate a lop3 PTX instruction, performing
// a logical operation on 3 inputs.
template <int LUT>
__device__
uint32_t lop3(uint32_t a, uint32_t b, uint32_t c)
{
    uint32_t out;
    LDPC2_ASM("lop3.b32 %0, %1, %2, %3, %4;\n" :
              "=r"(out) :
              "r"(a), "r"(b), "r"(c), "n"(LUT));
    return out;
}


////////////////////////////////////////////////////////////////////////
// select_from_mask()
// Returns a word with bits selected from inputs 'a' and 'b', using bits
// from the input 'mask.' If bit[i] in mask is 1, out[i] will be equal
// to a[i]. If bit[i] is 0, out will be equal to b[i].
// Commonly Used in conjunction with with fp16x2 comparison instructions
// in bitmask (".bm") mode, where comparison output will have 0xFFFF or
// 0x0000 for each high and low 16-bit value in the word.
static inline __device__ word_t select_from_mask(word_t mask, word_t a, word_t b)
{
#if 1
    // Slightly better performance observed in some cases with the C
    // version, compared to the LOP3 version. Unclear why...
    word_t wOut;
    wOut.u32 = (a.u32 & mask.u32) | (b.u32 & ~mask.u32);
    return wOut;
#else
    // LOP3 for masked select
    // (Note that equivalent C code, in some cases, does not seem to
    // result in a single LOP3 instruction. Unclear why...)
    //  A   B   C | (A & C) | (B & ~C)
    // ---+---+---+-------------------
    //  0   0   0 |   0       C is not set: choose B
    //  0   0   1 |   0       C is set:     choose A
    //  0   1   0 |   1       C is not set: choose B
    //  0   1   1 |   0       C is set:     choose A
    //  1   0   0 |   0       C is not set: choose B
    //  1   0   1 |   1       C is set:     choose A
    //  1   1   0 |   1       C is not set: choose B
    //  1   1   1 |   1       C is set:     choose A
    // 11100100 --> 0xE4
    return lop3<0xE4>(a, b, mask);
#endif
}

////////////////////////////////////////////////////////////////////////
// clamp_pos_to_half_max()
// Clamps a each value of a pair of fp16 values to the maximum fp16
// value. (In other words, if a value is +Inf it is converted to
// fp16_max.) Assumes input values are positive.
static inline __device__ word_t clamp_pos_to_half_max(word_t wIn)
{
    //bool hasInf = half2_has_inf(wIn.f16x2);
    word_t wOut, hmax_hmax, bmask;
    hmax_hmax.u32 = 0x7BFF7BFF;
    bmask         = hset2_bm_lt(wIn, hmax_hmax);
    // expecting single LOP3 instruction here...
    wOut.u32      = (wIn.u32 & bmask.u32) | (hmax_hmax.u32 & ~bmask.u32);

    //if(hasInf)
    //{
    //     float2 in = __half22float2(wIn.f16x2);
    //     float2 out = __half22float2(wOut.f16x2);
    //     printf("threadIdx.x = %u: in = (%f, %f), out = (%f, %f)\n", threadIdx.x, in.x, in.y, out.x, out.y);
    //}

    return wOut;
}

////////////////////////////////////////////////////////////////////////
// abs_clamp_half_max()
// Given an input word with fp16x2 values, returns a word with fp16x2
// values, where each value is min(abs(in), FP16_MAX).
__device__ inline
word_t abs_clamp_half_max(word_t wIn)
{
    word_t wOut, hmax_hmax;
    // FP16_MAX
    hmax_hmax.u32 = 0x7BFF7BFF;
    #if __CUDA_ARCH__ >= 800
        //__hmax2(), __habs2() introduced in CUDA 11, only defined for ARCH > 80
        wOut.f16x2 = __hmin2(__habs2(wIn.f16x2), hmax_hmax.f16x2);
    #else
        word_t bmask;
        bmask         = hset2_bm_lt(wIn, hmax_hmax);
        // expecting single LOP3 instruction here...
        wOut.u32      = (wIn.u32 & bmask.u32) | (hmax_hmax.u32 & ~bmask.u32);
    #endif
    return wOut;
}

////////////////////////////////////////////////////////////////////////
// clamp_signed()
// Clamps each fp16 value in the input __half2 value to be in the range
// [-clampValue, +clampValue]
__device__ inline
__half2 clamp_signed(__half2 a, __half2 clampValue)
{
    #if __CUDA_ARCH__ >= 860
        word_t a_w, clamp_w, out;
        a_w.f16x2 = a;
        clamp_w.f16x2 = clampValue;
        LDPC2_ASM("min.xorsign.abs.f16x2 %0, %1, %2;\n"
                  : "=r"(out.u32)
                  : "r"(a_w.u32),
                  "r"(clamp_w.u32));
        return out.f16x2;
    #elif __CUDA_ARCH__ >= 800
        //__hmax2(), __habs2() introduced in CUDA 11, only defined for ARCH > 80
        __half2 c = __hmin2(a, clampValue);
        c = __hmax2(c, __hneg2(clampValue));
        return c;
    #else
        word_t aW, clampW, clampNegW;
        aW.f16x2        = a;
        clampW.f16x2    = clampValue;
        clampNegW.f16x2 = __hneg2(clampValue);

        word_t lt_clamp     = hset2_bm_lt(aW, clampW);
        word_t c            = select_from_mask(lt_clamp, aW, clampW);
        word_t gt_neg_clamp = hset2_bm_gt(c, clampNegW);
        c                   = select_from_mask(gt_neg_clamp, c, clampNegW);
        return c.f16x2;
    #endif
}

////////////////////////////////////////////////////////////////////////
// fp16x2_abs()
__device__ inline
word_t fp16x2_abs(word_t w)
{
    word_t out;
    // abs.f16x2 not in PTX until v6.5 (CUDA 10.2)
    // The compiler will (hopefully) fuse abs calls on f16x2 values into
    // the instruction modifier for those instructions that support
    // it.
#if ((__CUDACC_VER_MAJOR__ == 10) && (__CUDACC_VER_MINOR__ >= 2))  || (__CUDACC_VER_MAJOR__ >= 11)
    LDPC2_ASM("abs.f16x2 %0, %1;\n\t\t"
              : "=r"(out.u32)
              : "r"(w.u32));
#else
    out.u32 = w.u32 & 0x7FFF7FFF;
#endif
    return out;
}

////////////////////////////////////////////////////////////////////////
// fp16x2_sign_mask()
// Returns a word with the sign bits of the two fp16 input values, and
// all other bits 0.
__device__ inline
word_t fp16x2_sign_mask(word_t w)
{
    word_t out;
    out.u32 = w.u32 & 0x80008000;
    return out;
}

////////////////////////////////////////////////////////////////////////
// extract_low_high()
// Ouput will have the low 16 bits from a and the high 16 bits from b
// [a.hi, a.lo], [b.hi, b.lo] --> [b.hi, a.lo]
__device__ inline
word_t extract_low_high(word_t a, word_t b)
{
    word_t wOut;
    LDPC2_ASM("prmt.b32 %0, %1, %2, 0x7610;\n" : "=r"(wOut.u32) : "r"(a.u32), "r"(b.u32));
    return wOut;
}

////////////////////////////////////////////////////////////////////////
// extract_high_low()
// Ouput will have the low 16 bits from a and the high 16 bits from b
// [a.hi, a.lo], [b.hi, b.lo] --> [b.lo, a.hi]
__device__ inline
word_t extract_high_low(word_t a, word_t b)
{
    word_t wOut;
    LDPC2_ASM("prmt.b32 %0, %1, %2, 0x3254;\n" : "=r"(wOut.u32) : "r"(a.u32), "r"(b.u32));
    return wOut;
}

__device__ inline
word_t rotate_right_8b(word_t a)
{
    word_t wOut;
    LDPC2_ASM("prmt.b32 %0, %1, %1, 0x0321;\n" : "=r"(wOut.u32) : "r"(a.u32), "r"(a.u32));
    return wOut;
}

////////////////////////////////////////////////////////////////////////
// swap_high_low()
__device__ inline
word_t swap_high_low(word_t a)
{
    word_t wOut;
#if 0
    // alternate approach: funnel shift...
    LDPC2_ASM("prmt.b32 %0, %1, %1, 0x5432;\n" : "=r"(wOut.u32) : "r"(a.u32));
#else
    wOut.f16x2 = __lowhigh2highlow(a.f16x2);
#endif
    return wOut;
};

////////////////////////////////////////////////////////////////////////
// duplicate_low()
__device__ inline
word_t duplicate_low(word_t a)
{
    word_t wOut;
    wOut.f16x2 = __half2half2(a.f16);
    return wOut;
}

////////////////////////////////////////////////////////////////////////
// pack_word_pair()
template <int ROW_DEGREE>
void __device__ pack_word_pair(word_t       (&w_dst)[ROW_DEGREE],
                               const word_t (&w_src_low) [div_round_up_t<ROW_DEGREE, 2>::value],
                               const word_t (&w_src_high)[div_round_up_t<ROW_DEGREE, 2>::value])
{
    #pragma unroll
    for(int i = 0; i < (ROW_DEGREE / 2); ++i)
    {
        w_dst[(i*2) + 0].f16x2 = __lows2half2(w_src_low[i].f16x2,
                                              w_src_high[i].f16x2);
        w_dst[(i*2) + 1].f16x2 = __highs2half2(w_src_low[i].f16x2,
                                               w_src_high[i].f16x2);
    }
    // Handle the last item separately for odd row degrees
    if(((ROW_DEGREE / 2) * 2) < ROW_DEGREE)
    {
        w_dst[ROW_DEGREE - 1].f16x2 = __lows2half2(w_src_low[ROW_DEGREE / 2].f16x2,
                                                   w_src_high[ROW_DEGREE / 2].f16x2);
    }
}

////////////////////////////////////////////////////////////////////////
// unpack_word_pair()
template <int ROW_DEGREE>
void __device__ unpack_word_pair(word_t         (&w_low) [div_round_up_t<ROW_DEGREE, 2>::value],
                                 word_t         (&w_high)[div_round_up_t<ROW_DEGREE, 2>::value],
                                 const word_t   (&w_src)[ROW_DEGREE],
                                 unsigned short pad_value)
{
    #pragma unroll
    for(int i = 0; i < (ROW_DEGREE / 2); ++i)
    {
        w_low[i].f16x2 = __lows2half2(w_src[(i*2) + 0].f16x2,
                                      w_src[(i*2) + 1].f16x2);
        w_high[i].f16x2 = __highs2half2(w_src[(i*2) + 0].f16x2,
                                        w_src[(i*2) + 1].f16x2);
    }
    // For odd row degrees, place the pad value in the high half
    // of the last word.
    if(((ROW_DEGREE / 2) * 2) < ROW_DEGREE)
    {
        __half2_raw pad;
        pad.x = pad.y = pad_value;
        w_low[ROW_DEGREE / 2].f16x2 = __lows2half2(w_src[ROW_DEGREE - 1].f16x2, pad);
        w_high[ROW_DEGREE / 2].f16x2 = __highs2half2(w_src[ROW_DEGREE - 1].f16x2, pad);
    }
}

////////////////////////////////////////////////////////////////////////
// clamp_to_half_max()
// Clamps a each value of a pair of fp16 values to the maximum fp16
// value, preserving the sign of the input. (In other words, if a value
// is +Inf it is converted to +fp16_max, and if it is -Inf it is
// converted to -fp16_max.)
static inline __device__ word_t clamp_to_half_max(word_t wIn)
{
    //bool hasInf = half2_has_inf(wIn.f16x2);

    word_t wOut, hmax_hmax, ltmask, signMask;
    //uint32_t inf_inf = 0x7C007C00;
    hmax_hmax.u32      = 0x7BFF7BFF;
    ltmask             = hset2_bm_lt(fp16x2_abs(wIn), hmax_hmax);
    // expecting single LOP3 instruction here...
    wOut.u32           = (wIn.u32 & ltmask.u32) | (hmax_hmax.u32 & ~ltmask.u32);
    // +/-Inf values will have the hmax value, but WITHOUT the correct
    // sign. Correct that here.
    // (Finite values will already have the correct sign, but the or
    // operation below will not disturb that.)
    signMask           = fp16x2_sign_mask(wIn);
    wOut.u32           |= signMask.u32;

    //if(hasInf)
    //{
    //     float2 in = __half22float2(wIn.f16x2);
    //     float2 out = __half22float2(wOut.f16x2);
    //     printf("threadIdx.x = %u: in = (%f, %f), out = (%f, %f)\n", threadIdx.x, in.x, in.y, out.x, out.y);
    //}

    return wOut;
}

////////////////////////////////////////////////////////////////////////
// half2_sort()
// Compare two half-words in v1_v0 (loword = v0, hiword = v1) and
// optionally swap values so that on return, abs(loword(v0_v1)) < abs(hiword(v0_v1))
// and set min0_index to indicate whether the low word (0) or high word
// (1) was smaller in absolute value.
__device__ inline
void half2_sort(word_t& v1_v0, int& min0_index)
{
    // Duplicate the high word - expecting generated code to use swizzle
    word_t v1_v1 = h1_h1(v1_v0);

#if ((__CUDACC_VER_MAJOR__ == 10) && (__CUDACC_VER_MINOR__ >= 2)) || (__CUDACC_VER_MAJOR__ >= 11)
    LDPC2_ASM("\t"
              "{\n\t\t"
              ".reg .pred h0_lt_h1, q;\n\t\t"
              ".reg .f16x2 abs_v1_v0, abs_v1_v1;\n\t\t"
              "abs.f16x2 abs_v1_v0, %0;\n\t\t"                         // abs_v1_v0 = [abs(v1)  abs(v0) ] (expecting |value| modifier in setp)
              "abs.f16x2 abs_v1_v1, %2;\n\t\t"                         // abs_v1_v1 = [abs(v1)  abs(v1) ] (expecting |value| modifier in setp)

              "setp.ltu.f16x2 h0_lt_h1|q, abs_v1_v0, abs_v1_v1;\n\t\t" // compare [v1 v0] to [v1 v1] --> [q h0_lt_h1]
              "selp.b32 %1, 0, 1, h0_lt_h1;\n\t\t"                     // min0_index = (h0_lt_h1) ? 0 : 1
              "@!h0_lt_h1 prmt.b32 %0, %0, %0, 0x5432;\n\t\t"          // if(!h0_lt_h1) swap(v0, v1)
              "}\n"
              : "+r"(v1_v0.u32), "=r"(min0_index)
              : "r"(v1_v1.u32));
#else
    // abs.f16x2 not in PTX until v6.5 (CUDA 10.2)
    word_t abs_v1_v0, abs_v1_v1;
    abs_v1_v0 = word_pair_abs(v1_v0);
    abs_v1_v1 = word_pair_abs(v1_v1);

    LDPC2_ASM("\t"
              "{\n\t\t"
              ".reg .pred h0_lt_h1, q;\n\t\t"
              "setp.ltu.f16x2 h0_lt_h1|q, %3, %2;\n\t\t"      // compare [v1 v0] to [v1 v1] --> [q h0_lt_h1]
              "selp.b32 %1, 0, 1, h0_lt_h1;\n\t\t"            // min0_index = (h0_lt_h1) ? 0 : 1
              "@!h0_lt_h1 prmt.b32 %0, %0, %0, 0x5432;\n\t\t" // if(!h0_lt_h1) swap(v0, v1)
              "}\n"
              : "+r"(v1_v0.u32), "=r"(min0_index)
              : "r"(abs_v1_v1.u32), "r"(abs_v1_v0.u32));
#endif
}

////////////////////////////////////////////////////////////////////////
// half2_min_sum_update_low_prmt()
//
// Updating min0, min1, and min0_index for fp16 (single codeword at a time):
// ------------------------------------------------------------------------
// Partition the positive real numbers into 3  possible regions for an
// incoming value (assuming that min0 and min1 have been initialized
// correctly based on the first two values):
//
//       A                 B                    C
// -|----------|-------------------------|--------------
//  0       abs(min0)                 abs(min1)
// Output     min1          min0        abs(value) < abs(min0)    abs(value) < abs(min1)
//            ---------  ----------     ----------------------    ----------------------
// case A:    min0       new_value                T                        T
// case B:    new_value  min0                     F                        T
// case C:    min1       min0                     F                        F
//
// Furthermore, assume that we have a register with min0 in the lo word
// and min1 in the hi word, and that the new value used to update is
// in the lo word of the 'value' register. (The approach can be modified
// to work with a value in the hi word by modifying the indices.) If we
// view these two registers with byte numbering as used by the PRMT
// instruction in index mode, we have:
//      7   6   5   4      3  2  1  0
//    [   _     value ] [  min1  min0]
//
// case A:  0x 1 0 5 4
// case B:  0x 5 4 1 0
// case C:  0x 3 2 1 0    (no change to input min1/min0 register)
//
// Assume: registers with [min1 min0], [idx], [min0_idx], [value]
//         .reg .pred ltmin0, ltmin1
//         .reg b32 prmt_idx;
//         setp.ltu.f16x2 ltmin0|ltmin1, |value|, min0_min1;  // Compare abs(value) to min0 and min1, setting predicates ltmin0 and ltmin1
// @ltmin0  mov.b32 min0_idx, idx;                            // if abs(value) < min0, replace min0_idx
// @ltmin1  selp.u32 prmt_idx, 0x1054, 0x5410, ltmin0;        // select between A, B permutations based on ltmin0 (only if abs(value) < abs(min1))
// @ltmin1  prmt.b32 min0_min1, min0_min1, value, prmt_idx;
//
__device__ inline
void half2_min_sum_update_low_prmt(word_t& min1_min0, int& min0_index, word_t new_value, int idx)
{
    // Duplicate the low word (expecting instruction swizzle to avoid an explicit instruction)
    word_t v = h0_h0(new_value);
#if ((__CUDACC_VER_MAJOR__ == 10) && (__CUDACC_VER_MINOR__ >= 2)) || (__CUDACC_VER_MAJOR__ >= 11)
    LDPC2_ASM("\t"
              "{\n\t\t"
              ".reg .pred ltmin0, ltmin1;\n\t\t"
              ".reg .f16x2 abs_min1_min0, abs_v;\n\t\t"
              ".reg .u32 prmt_idx;\n\t\t"
              "abs.f16x2 abs_min1_min0, %0;\n\t\t"                        // abs_min1_min0 = [abs(min1)  abs(min0) ] (expect |value| modifier in setp)
              "abs.f16x2 abs_v, %2;\n\t\t"                                // abs_v         = [abs(v)     abs(v)    ] (expect |value| modifier in setp)
              "setp.ltu.f16x2 ltmin0|ltmin1, abs_v, abs_min1_min0;\n\t\t" // ltmin0 = abs(v) < abs(min0)
                                                                          // ltmin1 = abs(v) < abs(min1)
              "@ltmin0  mov.b32 %1, %3;\n\t\t"                            // if abs(v) < min0, replace min0_idx
              "@ltmin1  selp.u32 prmt_idx, 0x1054, 0x5410, ltmin0;\n\t\t" // select between A, B permutations based on ltmin0 (only if abs(v) < abs(min1))
              "@ltmin1  prmt.b32 %0, %0, %2, prmt_idx;\n\t"               //
              "}\n"
              : "+r"(min1_min0.u32),
                "+r"(min0_index)
              : "r"(v.u32),
                "r"(idx));
#else
    // abs.f16x2 not in PTX until v6.5 (CUDA 10.2)
    word_t abs_min1_min0, abs_v;
    abs_min1_min0 = word_pair_abs(min1_min0);
    abs_v         = word_pair_abs(v);
    LDPC2_ASM("\t"
              "{\n\t\t"
              ".reg .pred ltmin0, ltmin1;\n\t\t"
              ".reg .u32 prmt_idx;\n\t\t"
              "setp.ltu.f16x2 ltmin0|ltmin1, %4, %5;\n\t\t"               // ltmin0 = abs(v) < abs(min0)
                                                                          // ltmin1 = abs(v) < abs(min1)
              "@ltmin0  mov.b32 %1, %3;\n\t\t"                            // if abs(v) < min0, replace min0_idx
              "@ltmin1  selp.u32 prmt_idx, 0x1054, 0x5410, ltmin0;\n\t\t" // select between A, B permutations based on ltmin0 (only if abs(v) < abs(min1))
              "@ltmin1  prmt.b32 %0, %0, %2, prmt_idx;\n\t"               //
              "}\n"
              : "+r"(min1_min0.u32),
                "+r"(min0_index)
              : "r"(v.u32),
                "r"(idx),
                "r"(abs_v.u32),
                "r"(abs_min1_min0.u32));
#endif
}

////////////////////////////////////////////////////////////////////////
// half2_min_sum_update_high_prmt()
// See the description above for half2_min_sum_update_low_prmt(), but replace
// the source with this pair of registers:
//      7   6  5   4      3  2  1  0
//    [ value    -  ]  [  min1  min0]
//
// case A:  0x 1 0 7 6
// case B:  0x 7 6 1 0
// case C:  0x 3 2 1 0    (no change to input min1/min0 register)
__device__ inline
void half2_min_sum_update_high_prmt(word_t& min1_min0, int& min0_index, word_t new_value, int idx)
{
    // Duplicate the high word (expecting instruction swizzle)
    word_t v = h1_h1(new_value);
#if ((__CUDACC_VER_MAJOR__ == 10) && (__CUDACC_VER_MINOR__ >= 2)) || (__CUDACC_VER_MAJOR__ >= 11)
    LDPC2_ASM("\t"
              "{\n\t\t"
              ".reg .pred ltmin0, ltmin1;\n\t\t"
              ".reg .f16x2 abs_min1_min0, abs_v;\n\t\t"
              ".reg .u32 prmt_idx;\n\t\t"
              "abs.f16x2 abs_min1_min0, %0;\n\t\t"                        // abs_min1_min0 = [abs(min1)  abs(min0) ] (expect |value| modifier in setp)
              "abs.f16x2 abs_v, %2;\n\t\t"                                // abs_v         = [abs(v)     abs(v)    ] (expect |value| modifier in setp)
              "setp.ltu.f16x2 ltmin0|ltmin1, abs_v, abs_min1_min0;\n\t\t" // ltmin0 = abs(v) < abs(min0)
                                                                          // ltmin1 = abs(v) < abs(min1)
              "@ltmin0  mov.b32 %1, %3;\n\t\t"                            // if abs(v) < min0, replace min0_idx
              "@ltmin1  selp.u32 prmt_idx, 0x1076, 0x7610, ltmin0;\n\t\t" // select between A, B permutations based on ltmin0 (only if abs(v) < abs(min1))
              "@ltmin1  prmt.b32 %0, %0, %2, prmt_idx;\n\t"               //
              "}\n"
              : "+r"(min1_min0.u32),  // 0
                "+r"(min0_index)      // 1
              : "r"(v.u32),           // 2
                "r"(idx));            // 3
#else
    // abs.f16x2 not in PTX until v6.5 (CUDA 10.2)
    word_t abs_min1_min0, abs_v;
    abs_min1_min0 = word_pair_abs(min1_min0);
    abs_v         = word_pair_abs(v);
    LDPC2_ASM("\t"
              "{\n\t\t"
              ".reg .pred ltmin0, ltmin1;\n\t\t"
              ".reg .u32 prmt_idx;\n\t\t"
              "setp.ltu.f16x2 ltmin0|ltmin1, %4, %5;\n\t\t"               // ltmin0 = abs(v) < abs(min0)
                                                                          // ltmin1 = abs(v) < abs(min1)
              "@ltmin0  mov.b32 %1, %3;\n\t\t"                            // if abs(v) < min0, replace min0_idx
              "@ltmin1  selp.u32 prmt_idx, 0x1076, 0x7610, ltmin0;\n\t\t" // select between A, B permutations based on ltmin0 (only if abs(v) < abs(min1))
              "@ltmin1  prmt.b32 %0, %0, %2, prmt_idx;\n\t"               //
              "}\n"
              : "+r"(min1_min0.u32),
                "+r"(min0_index)
              : "r"(v.u32),
                "r"(idx),
                "r"(abs_v.u32),
                "r"(abs_min1_min0.u32));
#endif
}

////////////////////////////////////////////////////////////////////////
// smem_address_as()
// Returns a typed value obtained by loading an element of the template
// type at the given address in shared memory. (Note that the address is
// not the same as an array index - the address would be obtained by
// multiplying the index by the size of the element.)
template <typename T> inline __device__ T smem_address_as(int oset);
template <>           inline __device__ int2 smem_address_as(int oset)
{
    union u
    {
        int2     i2;
        uint64_t u64;
    };
    u val;
    LDPC2_ASM_VOLATILE("ld.shared.b64 %0, [%1];\n" : "=l"(val.u64) : "r"(oset) : "memory");
    return val.i2;
}
template <>           inline __device__ float smem_address_as(int oset)
{
    float f;
    LDPC2_ASM_VOLATILE("ld.shared.f32 %0, [%1];\n" : "=f"(f) : "r"(oset) : "memory");
    return f;
}
template <>           inline __device__ word_t smem_address_as(int oset)
{
    word_t w;
    LDPC2_ASM_VOLATILE("ld.shared.u32 %0, [%1];\n" : "=r"(w.u32) : "r"(oset) : "memory");
    return w;
}
template <>           inline __device__ half_word_t smem_address_as(int oset)
{
    half_word_t w;
    LDPC2_ASM_VOLATILE("ld.shared.b16 %0, [%1];\n" : "=h"(w.u16) : "r"(oset) : "memory");
    return w;
}
template <>           inline __device__ int4 smem_address_as(int oset)
{
    int4     i4;
    LDPC2_ASM_VOLATILE("ld.shared.b64 {%0, %1, %2, %3}, [%4];\n"      :
                       "=r"(i4.x), "=r"(i4.y), "=r"(i4.z), "=r"(i4.w) :
                       "r"(oset)                                      :
                       "memory");
    return i4;
}
template <>           inline __device__ __nv_fp8_storage_t smem_address_as(int oset)
{
    uint32_t           u;
    // Note: There is no constraint letter for 8-bit wide PTX registers. PTX
    // instructions accepting 8-bit wide types permit operands to be wider than
    // the instruction type size.
    // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#restricted-use-of-sub-word-sizes
    LDPC2_ASM_VOLATILE("ld.shared.u8 %0, [%1];\n"      :
                       "=r"(u) :
                       "r"(oset) :
                       "memory");
    return static_cast<__nv_fp8_storage_t>(u);
}

////////////////////////////////////////////////////////////////////////
// gmem_address_as()
// Returns a typed value obtained by loading an element of the template
// type at the given address in global memory.
template <typename T> inline __device__ T gmem_address_as(const char* base, int oset);
template <>           inline __device__ float gmem_address_as(const char* base, int oset)
{
    float f;
    LDPC2_ASM_VOLATILE("ld.global.f32 %0, [%1];\n" : "=f"(f) : "l"(base + oset) : "memory");
    return f;
}
template <>           inline __device__ word_t gmem_address_as(const char* base, int oset)
{
    word_t w;
    LDPC2_ASM_VOLATILE("ld.global.u32 %0, [%1];\n" : "=r"(w.u32) : "l"(base + oset) : "memory");
    return w;
}
template <>           inline __device__ half_word_t gmem_address_as(const char* base, int oset)
{
    half_word_t w;
    LDPC2_ASM_VOLATILE("ld.global.u16 %0, [%1];\n" : "=h"(w.u16) : "l"(base + oset) : "memory");
    return w;
}

////////////////////////////////////////////////////////////////////////
// smem_increment()
// Increase the value at the given shmem address by the given increment
template <typename T> inline __device__ void smem_increment(int oset, T value);
template <>           inline __device__ void smem_increment(int oset, float inc)
{
    LDPC2_ASM_VOLATILE("\t"
                       "{\n\t\t"
                       ".reg .f32 smem_value;\n\t\t"
                       "ld.shared.f32 smem_value, [%0];\n\t\t"
                       "add.f32 smem_value, smem_value, %1;\n\t\t"
                       "st.shared.f32 [%0], smem_value;\n\t"
                       "}\n"
                       :
                       : "r"(oset), "f"(inc)
                       : "memory");
}

////////////////////////////////////////////////////////////////////////
// smem_decrement()
// Decrease the value at the given shmem address by the given decrement
template <typename T> inline __device__ void smem_decrement(int oset, T value);
template <>           inline __device__ void smem_decrement(int oset, float value)
{
    LDPC2_ASM_VOLATILE("\t"
                       "{\n\t\t"
                       ".reg .f32 smem_value;\n\t\t"
                       "ld.shared.f32 smem_value, [%0];\n\t\t"
                       "sub.f32 smem_value, smem_value, %1;\n\t\t"
                       "st.shared.f32 [%0], smem_value;\n\t"
                       "}\n"
                       :
                       : "r"(oset), "f"(value)
                       : "memory");
}

////////////////////////////////////////////////////////////////////////
// muladd_lo_u32()
// uint32_t multiply-add, discards overflow
__device__ inline
uint32_t muladd_lo_u32(uint32_t a, uint32_t b, uint32_t c)
{
    uint32_t d;
    LDPC2_ASM("mad.lo.u32 %0, %1, %2, %3;\n" :
              "=r"(d)                        :
              "r"(a), "r"(b) ,"r"(c));
    return d;
}

////////////////////////////////////////////////////////////////////////
// muladd_lo_s32()
// int32_t multiply-add, discards overflow
__device__ inline
int32_t muladd_lo_s32(int32_t a, int32_t b, int32_t c)
{
    uint32_t d;
    LDPC2_ASM("mad.lo.s32 %0, %1, %2, %3;\n" :
              "=r"(d)                        :
              "r"(a), "r"(b) ,"r"(c));
    return d;
}

////////////////////////////////////////////////////////////////////////
// ldpc_traits<>
template <typename T> struct ldpc_traits;

////////////////////////////////////////////////////////////////////////
// ldpc_traits<float>
template <>           struct ldpc_traits<float>
{
    typedef float4 llr_ldg_t;    // Type used for LDG instructions to load LLR values
    typedef uint2  llr_sts_t;    // Type used for STS instructions to store LLR values
    typedef float  llr_src_t;    // Type of source LLR data
    typedef __half app_buf_t;    // Type of shared memory APP buffer data access
    typedef __half app_elem_t;   // Underlying APP element arithmetic type
    typedef float  word_union_t; // Type of value(s) (one or more) in word_t union
    static constexpr int values_per_word = 1;
};

////////////////////////////////////////////////////////////////////////
// ldpc_traits<__half>
template <>           struct ldpc_traits<__half>
{
    typedef uint4   llr_ldg_t;    // Type used for LDG instructions to load LLR values
    typedef uint4   llr_sts_t;    // Type used for STS instructions to store LLR values
    typedef __half  llr_src_t;    // Type of source LLR data
    typedef __half  app_buf_t;    // Type of shared memory APP buffer data access
    typedef __half  app_elem_t;   // Underlying APP element arithmetic type
    typedef __half2 word_union_t; // Type of value(s) (one or more) in word_t union
    static constexpr int values_per_word = 2;
};

////////////////////////////////////////////////////////////////////////
// ldpc_traits<__half2>
// ldpc_traits specialization for __half2, which used for kernels that
// decode two codewords at a time.
template <> struct ldpc_traits<__half2>
{
    typedef  uint2  llr_ldg_t;    // Type used for LDG instructions to load LLR values
    typedef  uint4  llr_sts_t;    // Type used for STS instructions to store LLR values
    typedef __half  llr_src_t;    // Type of source LLR data
    typedef __half2 app_buf_t;    // Type of shared memory APP buffer data access
    typedef __half  app_elem_t;   // Underlying APP element arithmetic type
    typedef __half2 word_union_t; // Type of value(s) (one or more) in word_t union
    static constexpr int values_per_word = 1;
};

////////////////////////////////////////////////////////////////////////
// ldpc_traits<__nv_fp8_e5m2>
template <>           struct ldpc_traits<__nv_fp8_e5m2>
{
    typedef uint2           llr_ldg_t;    // Type used for LDG instructions to load LLR values
    typedef uint2           llr_sts_t;    // Type used for STS instructions to store LLR values
    typedef __nv_fp8_e5m2   llr_src_t;    // Type of source LLR data
    typedef __nv_fp8_e5m2   app_buf_t;    // Type of shared memory APP buffer data access
    typedef __nv_fp8_e5m2   app_elem_t;   // Underlying APP element arithmetic type
    typedef __nv_fp8x4_e5m2 word_union_t; // Type of value(s) (one or more) in word_t union
    static constexpr int values_per_word = 4;
};

////////////////////////////////////////////////////////////////////////
// ldpc_traits<__nv_fp8_e4m3>
template <>           struct ldpc_traits<__nv_fp8_e4m3>
{
    typedef uint2           llr_ldg_t;    // Type used for LDG instructions to load LLR values
    typedef uint2           llr_sts_t;    // Type used for STS instructions to store LLR values
    typedef __nv_fp8_e4m3   llr_src_t;    // Type of source LLR data
    typedef __nv_fp8_e4m3   app_buf_t;    // Type of shared memory APP buffer data access
    typedef __nv_fp8_e4m3   app_elem_t;   // Underlying APP element arithmetic type
    typedef __nv_fp8x4_e4m3 word_union_t; // Type of value(s) (one or more) in word_t union
    static constexpr int values_per_word = 4;
};

// Returns a half2 value containing the result of conversion to __half of the
// first pair of fp8 values of the given fp8x4 argument.
// See operator float4() in cuda_fp8.hpp for the float version.
inline
__host__ __device__
__half2 fp8x4_to_half2_low(__nv_fp8x4_e4m3 fp8x4)
{
    const __nv_fp8x2_storage_t slo = static_cast<__nv_fp8x2_storage_t>(fp8x4.__x);
    return static_cast<__half2>(__nv_cvt_fp8x2_to_halfraw2(slo, __NV_E4M3));
}

// Returns a half2 value containing the result of conversion to __half of the
// second pair of fp8 values of the given fp8x4 argument.
// See operator float4() in cuda_fp8.hpp for the float version.
inline
__host__ __device__
__half2 fp8x4_to_half2_high(__nv_fp8x4_e4m3 fp8x4)
{
    const __nv_fp8x2_storage_t shi = static_cast<__nv_fp8x2_storage_t>(fp8x4.__x >> 16U);
    return static_cast<__half2>(__nv_cvt_fp8x2_to_halfraw2(shi, __NV_E4M3));
}

// Returns a half2 value containing the result of conversion to __half of the
// first pair of fp8 values of the given fp8x4 argument.
// See operator float4() in cuda_fp8.hpp for the float version.
inline
__host__ __device__
__half2 fp8x4_to_half2_low(__nv_fp8x4_e5m2 fp8x4)
{
    const __nv_fp8x2_storage_t slo = static_cast<__nv_fp8x2_storage_t>(fp8x4.__x);
    return static_cast<__half2>(__nv_cvt_fp8x2_to_halfraw2(slo, __NV_E5M2));
}

// Returns a half2 value containing the result of conversion to __half of the
// second pair of fp8 values of the given fp8x4 argument.
// See operator float4() in cuda_fp8.hpp for the float version.
inline
__host__ __device__
__half2 fp8x4_to_half2_high(__nv_fp8x4_e5m2 fp8x4)
{
    const __nv_fp8x2_storage_t shi = static_cast<__nv_fp8x2_storage_t>(fp8x4.__x >> 16U);
    return static_cast<__half2>(__nv_cvt_fp8x2_to_halfraw2(shi, __NV_E5M2));
}

template <class TElem>
__device__
void print_word_values_as(const word_t& w);

template <>
__device__ inline
void print_word_values_as<float>(const word_t& w)
{
    printf("%7.3f ", w.f32);
}
template <>
__device__ inline
void print_word_values_as<__half>(const word_t& w)
{
    float2 f = __half22float2(w.f16x2);
    printf("%7.3f %7.3f ", f.x, f.y);
}
template <>
__device__ inline
void print_word_values_as<__nv_fp8_e4m3>(const word_t& w)
{
    word_t low, high;
    low.f16x2  = fp8x4_to_half2_low(w.fp8x4_e4m3);
    print_word_values_as<__half>(low);
    high.f16x2 = fp8x4_to_half2_high(w.fp8x4_e4m3);
    print_word_values_as<__half>(high);
}
template <>
__device__ inline
void print_word_values_as<__nv_fp8_e5m2>(const word_t& w)
{
    word_t low, high;
    low.f16x2  = fp8x4_to_half2_low(w.fp8x4_e5m2);
    print_word_values_as<__half>(low);
    high.f16x2 = fp8x4_to_half2_high(w.fp8x4_e5m2);
    print_word_values_as<__half>(high);
}
template <>
__device__ inline
void print_word_values_as<uint32_t>(const word_t& w)
{
    printf("0x%08X ", w.u32);
}
template <>
__device__ inline
void print_word_values_as<uint16_t>(const word_t& w)
{
    printf(" 0x%04X  0x%04X ", w.u16x2.x, w.u16x2.y);
}

////////////////////////////////////////////////////////////////////////
// interleave_llr()
inline __device__
uint4 interleave_llr(uint2 llr0, uint2 llr1)
{
    //                7/6   5/4 | 3/2  1/0  (bytes)
    //                    y     |     x
    // input:   llr0   h3   h2  |  h1   h0
    //          llr1   h7   h6  |  h5   h4
    //                 high  low
    // output:     x   h4    h0
    //             y   h5    h1
    //             z   h6    h2
    //             w   h7    h3
    uint4 d;
    LDPC2_ASM("prmt.b32 %0, %1, %2, 0x5410;\n" : "=r"(d.x) : "r"(llr0.x), "r"(llr1.x));
    LDPC2_ASM("prmt.b32 %0, %1, %2, 0x7632;\n" : "=r"(d.y) : "r"(llr0.x), "r"(llr1.x));
    LDPC2_ASM("prmt.b32 %0, %1, %2, 0x5410;\n" : "=r"(d.z) : "r"(llr0.y), "r"(llr1.y));
    LDPC2_ASM("prmt.b32 %0, %1, %2, 0x7632;\n" : "=r"(d.w) : "r"(llr0.y), "r"(llr1.y));
    return d;
}

class unused {};

template <typename T> struct is_fp8 : std::bool_constant<false> {};
template <> struct is_fp8<__nv_fp8_e5m2> : std::bool_constant<true> {};
template <> struct is_fp8<__nv_fp8_e4m3> : std::bool_constant<true> {};

// Default LLR values are used:
// - as LLR values in structures that group multiple LLR values when
//   the number of values is not a multiple of the group size. For
//   example, when the row degree is odd, and we store 32-bit words
//   with 2 16-bit values each, the last value will have the default
//   value.
// - as LLR values when writing hard outputs where the total number
//   of values is not a multiple of the 32-bit integer word size.
//
// As such, default values should:
// - Have a zero sign bit (i.e. be greater than zero) so that they do not
//   contribute to the parity count for a row
// - Have large values (e.g. Inf when possible) so that they do not modify
//   the min-sum determination of the valid LLR values.
template <typename T>
struct default_llr_value;

template <>
struct default_llr_value<__half>
{
    __device__
    static __half value()
    {
        // Infinity representation in the fp16 type is 0x7C00
        __half_raw inf_h_raw{0x7C00};
        return __half{inf_h_raw};
    }
};

template <>
struct default_llr_value<__nv_fp8_e5m2>
{
    // https://arxiv.org/pdf/2209.05433
    // e5m2 infinity: S.11111.00 = 0x7C
    static constexpr __nv_fp8_storage_t storage_value = 0x7C;
    __device__
    static __nv_fp8_e5m2 value()
    {
        return __nv_fp8_e5m2{storage_value};
    }
};

template <>
struct default_llr_value<__nv_fp8_e4m3>
{
    // https://arxiv.org/pdf/2209.05433
    // e4m3 does not have an infinity representation.
    // Max normal value: S.1111.110 = 1.75x2^8 = 448
    // Hex encoding of max normal value: 0x7E
    static constexpr __nv_fp8_storage_t storage_value = 0x7E;
    __device__
    static __nv_fp8_e4m3 value()
    {
        return __nv_fp8_e4m3{storage_value};
    }
};

// Convert an array of fp8x4 source values to a (larger) array of fp16x2 values.
// We leave the source array size unconstrained because we may pad C2V storage
// sizes to the maximum for any row, and rely on the compiler to perform dead
// code elimination to remove references to storage that is not read.
template <class TSrc,
          int   NUM_VALUES,
          int   NUM_SRC_VALUES>
__host__ __device__
void fp8x4_to_half2_array(word_t (&dst)[div_round_up_t<NUM_VALUES, 2>::value],
                          word_t (&src)[NUM_SRC_VALUES])
{
    static_assert(NUM_SRC_VALUES >= div_round_up_t<NUM_VALUES, 4>::value);

    // Determine the word union type (__nv_fp8x4_e5m2 or __nv_fp8x4_e4m3)
    // corresponding to the source fp8 type (__nv_fp8_e5m2 or __nv_fp8_e4m3).
    using fp8_word_t = typename ldpc_traits<TSrc>::word_union_t;

    #pragma unroll
    for(int i = 0; i < div_round_up_t<NUM_VALUES, 2>::value; ++i)
    {
        if(0 == (i % 2))
        {
            dst[i].f16x2 = fp8x4_to_half2_low(word_t_as<fp8_word_t>(src[i >> 1]));
        }
        else
        {
            dst[i].f16x2 = fp8x4_to_half2_high(word_t_as<fp8_word_t>(src[i >> 1]));
        }
    }
}

// Convert an array of fp16x2 source values to a (smaller) array of fp8x4 values.
// Extra values in the destination array of words will be set to zero.
// Note that this function uses the constructors for the __nv_fp8x4_e4m3 and
// __nv_fp8x5_e5m2 structs, which saturate to the largest non-inf, non-NaN value
// representable by the fp8 type when the input __half is larger than the fp8
// type can represent.
template <class TDst,
          int   NUM_VALUES,
          int   NUM_DST_VALUES>
__host__ __device__
void half2_to_fp8x4_array(word_t (&dst)[NUM_DST_VALUES],
                          word_t (&src)[div_round_up_t<NUM_VALUES, 2>::value])
{
    static_assert(NUM_DST_VALUES >= div_round_up_t<NUM_VALUES, 4>::value);

    // Determine the word union type (__nv_fp8x4_e5m2 or __nv_fp8x4_e4m3)
    // corresponding to the source fp8 type (__nv_fp8_e5m2 or __nv_fp8_e4m3).
    using fp8_word_t = typename ldpc_traits<TDst>::word_union_t;

    #pragma unroll
    for(int i = 0; i < div_round_up_t<NUM_VALUES, 4>::value; ++i)
    {
        if((i * 2 + 1) > (div_round_up_t<NUM_VALUES, 2>::value - 1))
        {
            word_t_as<fp8_word_t>(dst[i]) = fp8_word_t(src[i * 2].f16x2,
                                                       __float2half2_rn(0.0f));
        }
        else
        {
            word_t_as<fp8_word_t>(dst[i]) = fp8_word_t(src[i * 2].f16x2,
                                                       src[i * 2 + 1].f16x2);
        }
    }
}

////////////////////////////////////////////////////////////////////////
// app_loader
// Loads APP values from shared memory, given an array of shared memory
// addresses.
template <typename T, int ROW_DEGREE, class Enable = void> struct app_loader;

template <int   ROW_DEGREE,
          class TAPP>
struct app_loader<TAPP, ROW_DEGREE, std::enable_if_t<is_fp8<TAPP>::value>>
{
    using fp8_t = TAPP;
    __device__
    static void load(word_t (&app)     [row_num_words<fp8_t, ROW_DEGREE>::value],
                     int    (&app_addr)[ROW_DEGREE],
                     int    smem_offset)
    {
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Fetch app values
        #pragma unroll
        for(int i = 0; i < round_up_t<ROW_DEGREE, 4>::value; ++i)
        {
              __nv_fp8_storage_t s;
            //-  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
            // Load channel APP from given address
            if(i >= ROW_DEGREE)
            {
                /* use a maximum positive value for values outside the row degree */
                s = default_llr_value<fp8_t>::storage_value;
            }
            else
            {
                s  = smem_address_as<__nv_fp8_storage_t>(smem_offset + app_addr[i]);
                //if(0 == threadIdx.x)
                //    printf("Loaded %f (0x%X) at offset %u\n", __half2float(__half(TAPP{s})), (unsigned int)(s), app_addr[i]);
            }
            //-  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
            // Load 8-bit value into the correct part of the APP word
            if(0 == (i%4))
            {
                app[i >> 2].u32 = s;
            }
            else if(1 == (i%4))
            {
                app[i >> 2].u32 |= (s << 8);
            }
            else if(2 == (i%4))
            {
                app[i >> 2].u32 |= (s << 16);
            }
            else
            {
                app[i >> 2].u32 |= (s << 24);
            }
        }
    }
};

template <int ROW_DEGREE>
struct app_loader<__half, ROW_DEGREE>
{
    __device__
    static void load(word_t (&app)     [row_num_words<__half, ROW_DEGREE>::value],
                     int    (&app_addr)[ROW_DEGREE],
                     int    smem_offset)
    {
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Fetch app values
        #pragma unroll
        for(int i = 0; i < round_up_t<ROW_DEGREE, 2>::value; ++i)
        {
            half_word_t happ;
            //-  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
            // Load channel APP from given address
            if(i >= ROW_DEGREE)
            {
                happ.u16 = 0x7C00; /* load inf(fp16) for the last pair high word when row degree is odd*/
            }
            else
            {
                happ  = smem_address_as<half_word_t>(smem_offset + app_addr[i]);
                //if(0 == threadIdx.x)
                //    printf("Loaded %f at offset %u\n", __half2float(happ.f16), app_addr[i]);
            }
            //-  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
            // Load half word into lo or hi part of the APP word
            if(0 == (i%2))
            {
                app[i >> 1].f16x2.x = happ.u16;
            }
            else
            {
                app[i >> 1].f16x2.y = happ.u16;
            }
        }
    }
};

template <int ROW_DEGREE>
struct app_loader<__half2, ROW_DEGREE>
{
    __device__
    static void load(word_t (&app)     [row_num_words<__half2, ROW_DEGREE>::value],
                     int    (&app_addr)[ROW_DEGREE],
                     int    smem_offset)
    {
        #pragma unroll
        for(int i = 0; i < ROW_DEGREE; ++i)
        {
            app[i] = smem_address_as<word_t>(smem_offset + app_addr[i]);
        }
    }
};

////////////////////////////////////////////////////////////////////////
// app_loader_checked
// Loads APP values from shared memory, given an array of shared memory
// addresses. Checks the given shared memory offset before loading
// values.
// For small lifting sizes that split threads among multiple codewords,
// there may be "extra" threads that are not assigned to a valid
// codeword. We used this "checked" loader variant to avoid out-of-
// bounds shared memory reads. (Efforts to have the compiler do this at
// compile time with the non-checked variant resulted in register
// spilling.)
template <typename T, int ROW_DEGREE> struct app_loader_checked;

template <int ROW_DEGREE>
struct app_loader_checked<__half, ROW_DEGREE>
{
    __device__
    static void load(word_t (&app)     [row_num_words<__half, ROW_DEGREE>::value],
                     int    (&app_addr)[ROW_DEGREE],
                     int    smem_offset)
    {
        if(smem_offset >= 0)
        {
            //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            // Fetch app values and update the min0/min1/address fields
            #pragma unroll
            for(int i = 0; i < round_up_t<ROW_DEGREE, 2>::value; ++i)
            {
                half_word_t happ;
                //-  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
                // Load channel APP from given address
                if(i >= ROW_DEGREE)
                {
                    happ.u16 = 0x7C00; /* load inf(fp16) for the last pair high word when row degree is odd*/
                }
                else
                {
                    happ  = smem_address_as<half_word_t>(smem_offset + app_addr[i]);
                }
                //-  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
                // Load half word into lo or hi part of the APP word
                if(0 == (i%2))
                {
                    app[i >> 1].f16x2.x = happ.u16;
                }
                else
                {
                    app[i >> 1].f16x2.y = happ.u16;
                }
            }
        } // if(smem_offset >= 0)
    }
};

////////////////////////////////////////////////////////////////////////
// app_writer
// Writes (non-extension) APP values to shared memory, given the APP
// values and an array of shared memory addresses.
template <typename T,
          int ROW_DEGREE,
          int UPDATE_ROW_DEGREE,
          class TEnable = void> struct app_writer;

template <int ROW_DEGREE, int UPDATE_ROW_DEGREE>
struct app_writer<__half, ROW_DEGREE, UPDATE_ROW_DEGREE>
{
    __device__
    static void write_non_ext(word_t (&app)     [row_num_words<__half, ROW_DEGREE>::value],
                              int    (&app_addr)[ROW_DEGREE],
                              int    smem_offset)
    {
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Write output values
        #pragma unroll
        for(int i = 0; i < UPDATE_ROW_DEGREE; ++i)
        {
            if(0 == (i % 2))
            {
                write_shared_word_low(app[i >> 1], smem_offset + app_addr[i]);
            }
            else
            {
                write_shared_word_high(app[i >> 1], smem_offset + app_addr[i]);
            }
        }
    }

    __device__
    static bool write_non_ext_sign_change(word_t (&app)     [row_num_words<__half, ROW_DEGREE>::value],
                                          int    (&app_addr)[ROW_DEGREE],
                                          int    smem_offset)
    {
        bool sign_change = false;
        #pragma unroll
        for(int i = 0; i < UPDATE_ROW_DEGREE; ++i)
        {
            const int         addr     = smem_offset + app_addr[i];
            const half_word_t old_app  = smem_address_as<half_word_t>(addr);
            const uint32_t    app_word = app[i >> 1].u32;
            const uint32_t    new_app  = (0 == (i & 1)) ? (app_word & 0xffffu) : (app_word >> 16);
            sign_change |= 0u != ((static_cast<uint32_t>(old_app.u16) ^ new_app) & 0x8000u);
            if(0 == (i & 1))
            {
                write_shared_word_low(app[i >> 1], addr);
            }
            else
            {
                write_shared_word_high(app[i >> 1], addr);
            }
        }
        return sign_change;
    }
};

template <int ROW_DEGREE, int UPDATE_ROW_DEGREE>
struct app_writer<__half2, ROW_DEGREE, UPDATE_ROW_DEGREE>
{
    __device__
    static void write_non_ext(word_t (&app)     [row_num_words<__half2, ROW_DEGREE>::value],
                              int    (&app_addr)[ROW_DEGREE],
                              int    smem_offset)
    {
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Write output values
        #pragma unroll
        for(int i = 0; i < UPDATE_ROW_DEGREE; ++i)
        {
            write_shared_word(app[i], smem_offset + app_addr[i]);
        }
    }

    __device__
    static bool write_non_ext_sign_change(word_t (&app)     [row_num_words<__half2, ROW_DEGREE>::value],
                                          int    (&app_addr)[ROW_DEGREE],
                                          int    smem_offset)
    {
        bool sign_change = false;
        #pragma unroll
        for(int i = 0; i < UPDATE_ROW_DEGREE; ++i)
        {
            const int    addr    = smem_offset + app_addr[i];
            const word_t old_app = smem_address_as<word_t>(addr);
            sign_change |= 0u != ((old_app.u32 ^ app[i].u32) & 0x80008000u);
            write_shared_word(app[i], addr);
        }
        return sign_change;
    }
};

// Specialization for fp8 types
template <typename T, int ROW_DEGREE, int UPDATE_ROW_DEGREE>
struct app_writer<T, ROW_DEGREE, UPDATE_ROW_DEGREE, std::enable_if_t<is_fp8<T>::value>>
{
    __device__
    static void write_non_ext(word_t (&app)     [row_num_words<T, ROW_DEGREE>::value],
                              int    (&app_addr)[ROW_DEGREE],
                              int    smem_offset)
    {
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Write output values
        #pragma unroll
        for(int i = 0; i < UPDATE_ROW_DEGREE; ++i)
        {
            // Passing an unsigned char (i.e. __nv_fp8_storage_t) and casting
            // to a 32-bit value (for use with inline assembly inside the
            // write_shared_uchar() function) can cause the compiler to emit
            // a bitwise AND with 0xFF before writing to shared memory.
            // Therefore, we use uint32_t as the type for s below instead of
            // __nv_fp8_storage_t.
            // TODO: Stop using integer shared memory offsets as required by
            // write_shared_uchar() and use proper shared memory pointers.
            //__nv_fp8_storage_t s;
            uint32_t s;
            if(0 == (i % 4))
            {
                s = app[i >> 2].u32;
            }
            else if(1 == (i % 4))
            {
                s = (app[i >> 2].u32 >> 8);
            }
            else if(2 == (i % 4))
            {
                s = (app[i >> 2].u32 >> 16);
            }
            else
            {
                s = (app[i >> 2].u32 >> 24);
            }
            write_shared_uchar(s, smem_offset + app_addr[i]);
        }
    }
};

////////////////////////////////////////////////////////////////////////
// app_writer_checked
// Writes (non-extension) APP values to shared memory, given the APP
// values and an array of shared memory addresses.
// For small lifting sizes that split threads among multiple codewords,
// there may be "extra" threads that are not assigned to a valid
// codeword. We used this "checked" writer variant to avoid out-of-
// bounds shared memory writes. (Efforts to have the compiler do this at
// compile time with the non-checked variant resulted in register spilling.)
template <typename T, int ROW_DEGREE, int UPDATE_ROW_DEGREE> struct app_writer_checked;

template <int ROW_DEGREE, int UPDATE_ROW_DEGREE>
struct app_writer_checked<__half, ROW_DEGREE, UPDATE_ROW_DEGREE>
{
    __device__
    static void write_non_ext(word_t (&app)     [row_num_words<__half, ROW_DEGREE>::value],
                              int    (&app_addr)[ROW_DEGREE],
                              int    smem_offset)
    {
        if(smem_offset >= 0)
        {
            //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            // Write output values
            #pragma unroll
            for(int i = 0; i < UPDATE_ROW_DEGREE; ++i)
            {
                if(0 == (i % 2))
                {
                    write_shared_word_low(app[i >> 1], smem_offset + app_addr[i]);
                }
                else
                {
                    write_shared_word_high(app[i >> 1], smem_offset + app_addr[i]);
                }
            }
        }
    }
};

////////////////////////////////////////////////////////////////////////
// thread_is_active
// Struct to mask threads when doing row processing
template <int Z> struct thread_is_active
{
    __device__
    static bool value()
    {
        if(0 == (Z % 32))
            return true;
        else
            return (threadIdx.x < Z);
    }
};

template <typename T>
struct codewords_per_CTA
{
    static const int value = 1;
};

template <>
struct codewords_per_CTA<__half2>
{
    static const int value = 2;
};

////////////////////////////////////////////////////////////////////////
// get_num_LLRs()
// Returns the number of valid LLRs for a given LDPC configuration.
inline
__device__ int get_num_LLRs(const LDPC_kernel_params& params)
{
    // Precomputed in LDPC_kernel_params constructor
    return params.Z_var;
}

inline
__device__ int get_num_LLRs(const cuphyLDPCDecodeDesc_t& decodeDesc)
{
    return decodeDesc.config.Z * (decodeDesc.config.num_parity_nodes + ((1 == decodeDesc.config.BG) ? 22 : 10));
}

////////////////////////////////////////////////////////////////////////
// get_num_output_bits()
// Returns the number of output bits for a given LDPC configuration.
inline
__device__ int get_num_output_bits(const LDPC_kernel_params& params)
{
    // Precomputed in LDPC_kernel_params constructor
    return params.KbZ;
}

inline
__device__ int get_num_output_bits(const cuphyLDPCDecodeDesc_t& decodeDesc)
{
    return (decodeDesc.config.Kb * decodeDesc.config.Z);
}

struct multi_codeword_config
{
    int codewords_per_cta;      // number of codewords processed by each CTA
    int cta_start_index;        // starting index for this CTA
    int cta_codeword_count;     // may be less than codewords_per_cta for last block
    int thread_codeword_index;  // [0, codewords_per_cta-1]
    int thread_sub_index;       // index of thread within its codeword
    template <class TConfig>
    __device__
    multi_codeword_config(TConfig& cfg, unsigned int decodeIndex, int total_num_codewords) :
        codewords_per_cta(blockDim.x / cfg.Z),
        cta_start_index(codewords_per_cta * decodeIndex),
        cta_codeword_count(min(codewords_per_cta, total_num_codewords - cta_start_index)),
        thread_codeword_index(threadIdx.x / cfg.Z),
        thread_sub_index(threadIdx.x % cfg.Z)
    {
        //printf("threadIdx.x = %u, codewords_per_cta = %i, cta_start_index = %i, cta_codeword_count = %i, thread_codeword_index = %i, thread_sub_index = %i\n",
        //       threadIdx.x,
        //       codewords_per_cta,
        //       cta_start_index,
        //       cta_codeword_count,
        //       thread_codeword_index,
        //       thread_sub_index);
    }
};

inline
__device__
int get_num_codewords(const cuphyLDPCDecodeDesc_t& decodeDesc)
{
    int num = 0;
    for(int i = 0; i < decodeDesc.num_tbs; ++i)
    {
        num += decodeDesc.llr_input[i].num_codewords;
    }
    return num;
}

} // namespace ldpc2

////////////////////////////////////////////////////////////////////////
// tb_token
// A transport block token (tb_token) represents the results of a search
// for the codeword that a CTA will process in the LDPC decoder.
// The cuphyLDPCDecodeDesc_t structure supports a fixed number of
// transport block descriptors. Each transport block descriptor has a
// corresponding address and number of codewords. A CTA must determine
// which codeword it will process at the start of the decoder kernel.
//
// To save register space, multiple values are encoded into a tb_token:
// Bits    Description
// ----    -----------
// 27-31   tb index within the cuphyLDPCDecodeDesc_t array
// 26      partial (indicates only 1 CW is present for "2 CW" kernels
// 0-25    codeword offset/index within the transport block
typedef uint32_t tb_token;

__host__ __device__
inline
tb_token to_token_partial(int tb, int offset, bool partial)
{
    tb_token p = partial ? (1 << 26) : 0;
    return ((tb << 27) | p | offset);
}

template <int CW_PER_CTA>
__host__ __device__
tb_token to_token(int tb, int offset, bool partial);


template <>
__host__ __device__
inline
tb_token to_token<1>(int tb, int offset, bool /*partial*/)
{
    return ((tb << 27) | offset);
}

template <>
__host__ __device__
inline
tb_token to_token<2>(int tb, int offset, bool partial)
{
    return to_token_partial(tb, offset, partial);
}

__host__ __device__
inline
int tb_from_token(tb_token t)
{
    return (t >> 27);
}

__host__ __device__
inline
int offset_from_token(tb_token t)
{
    return (t & 0x3FFFFFF);
}

__host__ __device__
inline
bool is_partial_from_token(tb_token t)
{
    return (t & 0x04000000);
}

#if 0
// Debug printing of APP shared memory data for 2 codewords per CTA
__device__ inline
void thread0_dump_app(const __half2* app, int num)
{
    __syncthreads();
    if(0 == threadIdx.x)
    {
        for(int i = 0; i < num; ++i)
        {
            printf("[%5i] %f %f\n", i, __high2float(app[i]), __low2float(app[i]));
        }
    }
    __syncthreads();
}

// Debug printing of APP shared memory data for 1 codeword per CTA
__device__ inline
void thread0_dump_app(const __half* app, int num)
{
    __syncthreads();
    if(0 == threadIdx.x)
    {
        for(int i = 0; i < num; ++i)
        {
            printf("[%5i] %f\n", i, __half2float(app[i]));
        }
    }
    __syncthreads();
}
#endif

#endif // !defined(LDPC2_CUH_INCLUDED_)
