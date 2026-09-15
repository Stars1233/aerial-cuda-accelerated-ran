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

#if !defined(TYPE_CONVERT_H_INCLUDED_)
#define TYPE_CONVERT_H_INCLUDED_

#include <cuda_fp16.hpp>
#include "cuphy_internal.h"

// clang-format off

////////////////////////////////////////////////////////////////////////
// TYPE CONVERSIONS
// Note: No range checking is performed for narrowing conversions here.

////////////////////////////////////////////////////////////////////////
// Type conversions from signed char (int8)
template <typename T> T             CUDA_BOTH_INLINE type_convert(signed char s);
template <>           signed char   CUDA_BOTH_INLINE type_convert(signed char s) { return s; }
template <>           short         CUDA_BOTH_INLINE type_convert(signed char s) { return static_cast<short>(s); }
template <>           int           CUDA_BOTH_INLINE type_convert(signed char s) { return static_cast<int>(s); }
template <>           __half        CUDA_BOTH_INLINE type_convert(signed char s) { return __float2half(static_cast<float>(s)); }
template <>           float         CUDA_BOTH_INLINE type_convert(signed char s) { return static_cast<float>(s); }
template <>           double        CUDA_BOTH_INLINE type_convert(signed char s) { return static_cast<double>(s); }
template <>           __nv_fp8_e4m3 CUDA_BOTH_INLINE type_convert(signed char s) { return static_cast<__nv_fp8_e4m3>(s); }
template <>           __nv_fp8_e5m2 CUDA_BOTH_INLINE type_convert(signed char s) { return static_cast<__nv_fp8_e5m2>(s); }

////////////////////////////////////////////////////////////////////////
// Type conversions from char2 (int8 complex)
template <typename T> T               CUDA_BOTH_INLINE type_convert(char2 s);
template <>           char2           CUDA_BOTH_INLINE type_convert(char2 s) { return s; }
template <>           short2          CUDA_BOTH_INLINE type_convert(char2 s) { short2 r; r.x = static_cast<short>(s.x); r.y = static_cast<short>(s.y); return r; }
template <>           int2            CUDA_BOTH_INLINE type_convert(char2 s) { int2 r; r.x = static_cast<int>(s.x); r.y = static_cast<int>(s.y); return r; }
template <>           __half2         CUDA_BOTH_INLINE type_convert(char2 s) { return __floats2half2_rn(static_cast<float>(s.x), static_cast<float>(s.y)); }
template <>           cuComplex       CUDA_BOTH_INLINE type_convert(char2 s) { return make_cuComplex(static_cast<float>(s.x), static_cast<float>(s.y)); }
template <>           cuDoubleComplex CUDA_BOTH_INLINE type_convert(char2 s) { return make_cuDoubleComplex(static_cast<double>(s.x), static_cast<double>(s.y)); }

////////////////////////////////////////////////////////////////////////
// Type conversions from unsigned char (uint8)
template <typename T> T              CUDA_BOTH_INLINE type_convert(unsigned char u);
template <>           unsigned char  CUDA_BOTH_INLINE type_convert(unsigned char u) { return u; }
template <>           short          CUDA_BOTH_INLINE type_convert(unsigned char u) { return static_cast<short>(u); }
template <>           unsigned short CUDA_BOTH_INLINE type_convert(unsigned char u) { return static_cast<unsigned short>(u); }
template <>           int            CUDA_BOTH_INLINE type_convert(unsigned char u) { return static_cast<int>(u); }
template <>           unsigned int   CUDA_BOTH_INLINE type_convert(unsigned char u) { return static_cast<unsigned int>(u); }
template <>           __half         CUDA_BOTH_INLINE type_convert(unsigned char u) { return __float2half(static_cast<float>(u)); }
template <>           float          CUDA_BOTH_INLINE type_convert(unsigned char u) { return static_cast<float>(u); }
template <>           double         CUDA_BOTH_INLINE type_convert(unsigned char u) { return static_cast<double>(u); }
template <>           __nv_fp8_e4m3  CUDA_BOTH_INLINE type_convert(unsigned char u) { return static_cast<__nv_fp8_e4m3>(u); }
template <>           __nv_fp8_e5m2  CUDA_BOTH_INLINE type_convert(unsigned char u) { return static_cast<__nv_fp8_e5m2>(u); }

////////////////////////////////////////////////////////////////////////
// Type conversions from uchar2 (uint8 complex)
template <typename T> T               CUDA_BOTH_INLINE type_convert(uchar2 u);
template <>           uchar2          CUDA_BOTH_INLINE type_convert(uchar2 u) { return u; }
template <>           short2          CUDA_BOTH_INLINE type_convert(uchar2 u) { short2 r; r.x = static_cast<short>(u.x); r.y = static_cast<short>(u.y); return r; }
template <>           ushort2         CUDA_BOTH_INLINE type_convert(uchar2 u) { ushort2 r; r.x = static_cast<unsigned short>(u.x); r.y = static_cast<unsigned short>(u.y); return r; }
template <>           int2            CUDA_BOTH_INLINE type_convert(uchar2 u) { int2 r; r.x = static_cast<int>(u.x); r.y = static_cast<int>(u.y); return r; }
template <>           uint2           CUDA_BOTH_INLINE type_convert(uchar2 u) { uint2 r; r.x = static_cast<unsigned int>(u.x); r.y = static_cast<unsigned int>(u.y); return r; }
template <>           __half2         CUDA_BOTH_INLINE type_convert(uchar2 u) { return __floats2half2_rn(static_cast<float>(u.x), static_cast<float>(u.y)); }
template <>           cuComplex       CUDA_BOTH_INLINE type_convert(uchar2 u) { return make_cuComplex(static_cast<float>(u.x), static_cast<float>(u.y)); }
template <>           cuDoubleComplex CUDA_BOTH_INLINE type_convert(uchar2 u) { return make_cuDoubleComplex(static_cast<double>(u.x), static_cast<double>(u.y)); }

////////////////////////////////////////////////////////////////////////
// Type conversions from short (int16)
template <typename T> T             CUDA_BOTH_INLINE type_convert(short s);
template <>           short         CUDA_BOTH_INLINE type_convert(short s) { return s; }
template <>           int           CUDA_BOTH_INLINE type_convert(short s) { return static_cast<int>(s); }
template <>           __half        CUDA_BOTH_INLINE type_convert(short s) { return static_cast<__half>(s); }
template <>           float         CUDA_BOTH_INLINE type_convert(short s) { return static_cast<float>(s); }
template <>           double        CUDA_BOTH_INLINE type_convert(short s) { return static_cast<double>(s); }
template <>           __nv_fp8_e4m3 CUDA_BOTH_INLINE type_convert(short s) { return static_cast<__nv_fp8_e4m3>(s); }
template <>           __nv_fp8_e5m2 CUDA_BOTH_INLINE type_convert(short s) { return static_cast<__nv_fp8_e5m2>(s); }

////////////////////////////////////////////////////////////////////////
// Type conversions from short2 (int16 complex)
template <typename T> T               CUDA_BOTH_INLINE type_convert(short2 s);
template <>           short2          CUDA_BOTH_INLINE type_convert(short2 s) { return s; }
template <>           int2            CUDA_BOTH_INLINE type_convert(short2 s) { int2 r; r.x = static_cast<int>(s.x); r.y = static_cast<int>(s.y); return r; }
template <>           cuComplex       CUDA_BOTH_INLINE type_convert(short2 s) { return make_cuComplex(static_cast<float>(s.x), static_cast<float>(s.y)); }
template <>           cuDoubleComplex CUDA_BOTH_INLINE type_convert(short2 s) { return make_cuDoubleComplex(static_cast<double>(s.x), static_cast<double>(s.y)); }

////////////////////////////////////////////////////////////////////////
// Type conversions from unsigned short (uint16)
template <typename T> T              CUDA_BOTH_INLINE type_convert(unsigned short u);
template <>           unsigned short CUDA_BOTH_INLINE type_convert(unsigned short u) { return u; }
template <>           int            CUDA_BOTH_INLINE type_convert(unsigned short u) { return static_cast<int>(u); }
template <>           unsigned int   CUDA_BOTH_INLINE type_convert(unsigned short u) { return static_cast<unsigned int>(u); }
template <>           __half         CUDA_BOTH_INLINE type_convert(unsigned short u) { return static_cast<__half>(u); }
template <>           float          CUDA_BOTH_INLINE type_convert(unsigned short u) { return static_cast<float>(u); }
template <>           double         CUDA_BOTH_INLINE type_convert(unsigned short u) { return static_cast<double>(u); }
template <>           __nv_fp8_e4m3  CUDA_BOTH_INLINE type_convert(unsigned short u) { return static_cast<__nv_fp8_e4m3>(u); }
template <>           __nv_fp8_e5m2  CUDA_BOTH_INLINE type_convert(unsigned short u) { return static_cast<__nv_fp8_e5m2>(u); }

////////////////////////////////////////////////////////////////////////
// Type conversions from ushort2 (uint16 complex)
template <typename T> T               CUDA_BOTH_INLINE type_convert(ushort2 u);
template <>           ushort2         CUDA_BOTH_INLINE type_convert(ushort2 u) { return u; }
template <>           int2            CUDA_BOTH_INLINE type_convert(ushort2 u) { int2 r; r.x = static_cast<int>(u.x); r.y = static_cast<int>(u.y); return r; }
template <>           uint2           CUDA_BOTH_INLINE type_convert(ushort2 u) { uint2 r; r.x = static_cast<unsigned int>(u.x); r.y = static_cast<unsigned int>(u.y); return r; }
template <>           cuComplex       CUDA_BOTH_INLINE type_convert(ushort2 u) { return make_cuComplex(static_cast<float>(u.x), static_cast<float>(u.y)); }
template <>           cuDoubleComplex CUDA_BOTH_INLINE type_convert(ushort2 u) { return make_cuDoubleComplex(static_cast<double>(u.x), static_cast<double>(u.y)); }

////////////////////////////////////////////////////////////////////////
// Type conversions from int (int32)
template <typename T> T             CUDA_BOTH_INLINE type_convert(int i);
template <>           int           CUDA_BOTH_INLINE type_convert(int i) { return i; }
template <>           __half        CUDA_BOTH_INLINE type_convert(int i) { return static_cast<__half>(i); }
template <>           float         CUDA_BOTH_INLINE type_convert(int i) { return static_cast<float>(i); }
template <>           double        CUDA_BOTH_INLINE type_convert(int i) { return static_cast<double>(i); }
template <>           __nv_fp8_e4m3 CUDA_BOTH_INLINE type_convert(int i) { return static_cast<__nv_fp8_e4m3>(i); }
template <>           __nv_fp8_e5m2 CUDA_BOTH_INLINE type_convert(int i) { return static_cast<__nv_fp8_e5m2>(i); }

////////////////////////////////////////////////////////////////////////
// Type conversions from int2 (int32 complex)
template <typename T> T               CUDA_BOTH_INLINE type_convert(int2 i);
template <>           int2            CUDA_BOTH_INLINE type_convert(int2 i) { return i; }
template <>           cuComplex       CUDA_BOTH_INLINE type_convert(int2 i) { return make_cuComplex(static_cast<float>(i.x), static_cast<float>(i.y)); }
template <>           cuDoubleComplex CUDA_BOTH_INLINE type_convert(int2 i) { return make_cuDoubleComplex(static_cast<double>(i.x), static_cast<double>(i.y)); }

////////////////////////////////////////////////////////////////////////
// Type conversions from uint (uint32)
template <typename T> T             CUDA_BOTH_INLINE type_convert(unsigned int u);
template <>           uint          CUDA_BOTH_INLINE type_convert(unsigned int u) { return u; }
template <>           __half        CUDA_BOTH_INLINE type_convert(unsigned int u) { return static_cast<__half>(u); }
template <>           float         CUDA_BOTH_INLINE type_convert(unsigned int u) { return static_cast<float>(u); }
template <>           double        CUDA_BOTH_INLINE type_convert(unsigned int u) { return static_cast<double>(u); }
template <>           __nv_fp8_e4m3 CUDA_BOTH_INLINE type_convert(unsigned int u) { return static_cast<__nv_fp8_e4m3>(u); }
template <>           __nv_fp8_e5m2 CUDA_BOTH_INLINE type_convert(unsigned int u) { return static_cast<__nv_fp8_e5m2>(u); }

////////////////////////////////////////////////////////////////////////
// Type conversions from uint2 (uint32 complex)
template <typename T> T               CUDA_BOTH_INLINE type_convert(uint2 u);
template <>           uint2           CUDA_BOTH_INLINE type_convert(uint2 u) { return u; }
template <>           cuComplex       CUDA_BOTH_INLINE type_convert(uint2 u) { return make_cuComplex(static_cast<float>(u.x), static_cast<float>(u.y)); }
template <>           cuDoubleComplex CUDA_BOTH_INLINE type_convert(uint2 u) { return make_cuDoubleComplex(static_cast<double>(u.x), static_cast<double>(u.y)); }

////////////////////////////////////////////////////////////////////////
// Type conversions from __half (fp16)
template <typename T> T             CUDA_BOTH_INLINE type_convert(__half h);
template <>           __half        CUDA_BOTH_INLINE type_convert(__half h) { return h; }
template <>           float         CUDA_BOTH_INLINE type_convert(__half h) { return __half2float(h); }
template <>           double        CUDA_BOTH_INLINE type_convert(__half h) { return static_cast<double>(__half2float(h)); }
template <>           __nv_fp8_e4m3 CUDA_BOTH_INLINE type_convert(__half h) { return static_cast<__nv_fp8_e4m3>(h); }
template <>           __nv_fp8_e5m2 CUDA_BOTH_INLINE type_convert(__half h) { return static_cast<__nv_fp8_e5m2>(h); }
// Note: __half2char_rz clamps the result to SCHAR_MIN/SCHAR_MAX.
template <>           int8_t        CUDA_BOTH_INLINE type_convert(__half h) { return __half2char_rz(h); }

////////////////////////////////////////////////////////////////////////
// Type conversions from __half2 (fp16 complex)
template <typename T> T               CUDA_BOTH_INLINE type_convert(__half2 h);
template <>           __half2         CUDA_BOTH_INLINE type_convert(__half2 h) { return h; }
#if defined(__CUDACC__)
template <>           cuComplex       CUDA_BOTH_INLINE type_convert(__half2 h) { return __half22float2(h); }
template <>           cuDoubleComplex CUDA_BOTH_INLINE type_convert(__half2 h) { float2 c = __half22float2(h); return cuComplexFloatToDouble(c); }
#else
template <>           cuComplex       CUDA_BOTH_INLINE type_convert(__half2 h) { return make_cuComplex(__half2float(h.x), __half2float(h.y)); }
template <>           cuDoubleComplex CUDA_BOTH_INLINE type_convert(__half2 h) { return cuComplexFloatToDouble(type_convert<cuComplex>(h)); }
#endif // defined(__CUDACC__)

////////////////////////////////////////////////////////////////////////
// Type conversions from float (fp32)
template <typename T> T             CUDA_BOTH_INLINE type_convert(float f);
template <>           float         CUDA_BOTH_INLINE type_convert(float f) { return f; }
template <>           double        CUDA_BOTH_INLINE type_convert(float f) { return static_cast<double>(f); }
template <>           __half        CUDA_BOTH_INLINE type_convert(float f) { return __float2half(f); }
template <>           __nv_fp8_e4m3 CUDA_BOTH_INLINE type_convert(float f) { return static_cast<__nv_fp8_e4m3>(f); }
template <>           __nv_fp8_e5m2 CUDA_BOTH_INLINE type_convert(float f) { return static_cast<__nv_fp8_e5m2>(f); }
template <>           int8_t        CUDA_BOTH_INLINE type_convert(float f)
{
    // The C++ standard describes the result of conversion for out-of-bounds
    // values as undefined behavior. Since there is no infinity representation
    // for integer values, we clamp.
    // Casting a NaN value to an integer is undefined behavior, so we arbitrarily
    // choose to convert to 0 here for now.
    if(isnan(f))
    {
        return 0;
    }
    else if(f >= static_cast<float>(INT8_MAX))
    {
        return INT8_MAX;
    }
    else if(f <= static_cast<float>(INT8_MIN))
    {
        return INT8_MIN;
    }
    else
    {
        return static_cast<int8_t>(f);
    }
}

////////////////////////////////////////////////////////////////////////
// Type conversions from cuComplex (fp32 complex)
template <typename T> T               CUDA_BOTH_INLINE type_convert(cuComplex c);
template <>           __half2         CUDA_BOTH_INLINE type_convert(cuComplex c) { return __floats2half2_rn(c.x, c.y); }
template <>           cuComplex       CUDA_BOTH_INLINE type_convert(cuComplex c) { return c; }
template <>           cuDoubleComplex CUDA_BOTH_INLINE type_convert(cuComplex c) { return cuComplexFloatToDouble(c); }

////////////////////////////////////////////////////////////////////////
// Type conversions from double (fp64)
template <typename T> T             CUDA_BOTH_INLINE type_convert(double d);
template <>           __half        CUDA_BOTH_INLINE type_convert(double d) { return static_cast<__half>(d); }
template <>           double        CUDA_BOTH_INLINE type_convert(double d) { return d; }
template <>           float         CUDA_BOTH_INLINE type_convert(double d) { return static_cast<float>(d); }
template <>           __nv_fp8_e4m3 CUDA_BOTH_INLINE type_convert(double d) { return static_cast<__nv_fp8_e4m3>(d); }
template <>           __nv_fp8_e5m2 CUDA_BOTH_INLINE type_convert(double d) { return static_cast<__nv_fp8_e5m2>(d); }
template <>           int8_t        CUDA_BOTH_INLINE type_convert(double d)
{
    // The C++ standard describes the result of conversion for out-of-bounds
    // values as undefined behavior. Since there is no infinity representation
    // for integer values, we clamp.
    // Casting a NaN value to an integer is undefined behavior, so we arbitrarily
    // choose to convert to 0 here for now.
    if(isnan(d))
    {
        return 0;
    }
    else if(d >= static_cast<double>(INT8_MAX))
    {
        return INT8_MAX;
    }
    else if(d <= static_cast<double>(INT8_MIN))
    {
        return INT8_MIN;
    }
    else
    {
        return static_cast<int8_t>(d);
    }
}

////////////////////////////////////////////////////////////////////////
// Type conversions from cuDoubleComplex (fp64 complex)
template <typename T> T               CUDA_BOTH_INLINE type_convert(cuDoubleComplex d);
template <>           cuDoubleComplex CUDA_BOTH_INLINE type_convert(cuDoubleComplex d) { return d; }
template <>           cuComplex       CUDA_BOTH_INLINE type_convert(cuDoubleComplex d) { return cuComplexDoubleToFloat(d); }

////////////////////////////////////////////////////////////////////////
// Type conversions from __nv_fp8_e4m3
template <typename T> T             CUDA_BOTH_INLINE type_convert(__nv_fp8_e4m3 f);
template <>           __half        CUDA_BOTH_INLINE type_convert(__nv_fp8_e4m3 f) { return static_cast<__half>(f); }
template <>           float         CUDA_BOTH_INLINE type_convert(__nv_fp8_e4m3 f) { return static_cast<float>(f); }
template <>           double        CUDA_BOTH_INLINE type_convert(__nv_fp8_e4m3 f) { return static_cast<double>(f); }
template <>           __nv_fp8_e4m3 CUDA_BOTH_INLINE type_convert(__nv_fp8_e4m3 f) { return f; }
template <>           __nv_fp8_e5m2 CUDA_BOTH_INLINE type_convert(__nv_fp8_e4m3 f)
{
    return static_cast<__nv_fp8_e5m2>(static_cast<__half>(f));
}

////////////////////////////////////////////////////////////////////////
// Type conversions from __nv_fp8_e5m2
template <typename T> T             CUDA_BOTH_INLINE type_convert(__nv_fp8_e5m2 f);
template <>           __half        CUDA_BOTH_INLINE type_convert(__nv_fp8_e5m2 f) { return static_cast<__half>(f); }
template <>           float         CUDA_BOTH_INLINE type_convert(__nv_fp8_e5m2 f) { return static_cast<float>(f); }
template <>           double        CUDA_BOTH_INLINE type_convert(__nv_fp8_e5m2 f) { return static_cast<double>(f); }
template <>           __nv_fp8_e5m2 CUDA_BOTH_INLINE type_convert(__nv_fp8_e5m2 f) { return f; }
template <>           __nv_fp8_e4m3 CUDA_BOTH_INLINE type_convert(__nv_fp8_e5m2 f)
{
    return static_cast<__nv_fp8_e4m3>(static_cast<__half>(f));
}

// clang-format on

#endif // !defined(TYPE_CONVERT_H_INCLUDED_)
