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

#if !defined(LDPC2_LLR_LOADER_FP8_CUH_INCLUDED_)
#define LDPC2_LLR_LOADER_FP8_CUH_INCLUDED_

#include "ldpc2_llr_loader.cuh"

namespace ldpc2
{

template <typename TFP8>
__device__
uint2 apply_clamp_fp8(const uint2& src, float clamp_value)
{
    using fp8_t = TFP8;

    word_t wSrc[2];
    // Copy input words to a local array with type word_t, for use with
    // the fp8/fp16 conversion utility functions.
    wSrc[0].u32 = src.x;
    wSrc[1].u32 = src.y;

    // Convert the clamp value to fp16
    __half2 clampValue = __float2half2_rn(clamp_value);

    // Convert 8 fp8 values (in two word) to 8 fp16 values (in 4 words)
    word_t wH_in[4], wH_out[4];
    fp8x4_to_half2_array<fp8_t, 8>(wH_in, wSrc);

    // Clamp fp16 values to +/- clampValue
    wH_out[0].f16x2 = clamp_signed(wH_in[0].f16x2, clampValue);
    wH_out[1].f16x2 = clamp_signed(wH_in[1].f16x2, clampValue);
    wH_out[2].f16x2 = clamp_signed(wH_in[2].f16x2, clampValue);
    wH_out[3].f16x2 = clamp_signed(wH_in[3].f16x2, clampValue);

    // Convert fp16 values back to fp8
    uint2 result;
    word_t wOut[2];
    half2_to_fp8x4_array<fp8_t, 8>(wOut, wH_out);
    result.x = wOut[0].u32;
    result.y = wOut[1].u32;

    //if(0 == threadIdx.x)
    //{
    //    printf("clamp: %f, in: %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f, out: %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f\n",
    //    clamp_value,
    //    __high2float(wH_in[3].f16x2), __low2float(wH_in[3].f16x2), __high2float(wH_in[2].f16x2), __low2float(wH_in[2].f16x2),
    //    __high2float(wH_in[1].f16x2), __low2float(wH_in[1].f16x2), __high2float(wH_in[0].f16x2), __low2float(wH_in[0].f16x2),
    //    __high2float(wH_out[3].f16x2), __low2float(wH_out[3].f16x2), __high2float(wH_out[2].f16x2), __low2float(wH_out[2].f16x2),
    //    __high2float(wH_out[1].f16x2), __low2float(wH_out[1].f16x2), __high2float(wH_out[0].f16x2), __low2float(wH_out[0].f16x2));
    //}
    return result;
}

template <> struct llr_op_clamp<__nv_fp8_e4m3, uint2>
{
    __device__
    static uint2 apply(const uint2& src, float clamp_value)
    {
        return apply_clamp_fp8<__nv_fp8_e4m3>(src, clamp_value);
    }
};

template <> struct llr_op_clamp<__nv_fp8_e5m2, uint2>
{
    __device__
    static uint2 apply(const uint2& src, float clamp_value)
    {
        return apply_clamp_fp8<__nv_fp8_e5m2>(src, clamp_value);
    }
};

} // namespace ldpc2

#endif // !defined(LDPC2_LLR_LOADER_FP8_CUH_INCLUDED_)
