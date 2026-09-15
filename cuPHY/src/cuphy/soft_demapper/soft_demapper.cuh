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

#if !defined(CUPHY_SOFT_DEMAPPER_CUH_INCLUDED_)
#define CUPHY_SOFT_DEMAPPER_CUH_INCLUDED_

#include "tensor_desc.hpp"
#include "cuphy_kernel_util.cuh"
#include "soft_demapper_tables.cuh"
#include "type_convert.hpp"

namespace soft_demapper
{

//**********************************************************************
// Linear mapping (y = mx + b) from symbol value (x), representing the
// in-phase or quadrature component, to a normalized texture coordinate
// in the range of (0..1).
// Texture tables are generated for symbol values from -NA to (N-2)A
// (where N is the table size and A is the QAM normalization factor)
// with a spacing of 2A.
// We want symbol value -NA to map to texture coordinate 1/(2N), which
// is the texture coordinate of the first provided sample value.
// We want (N-2)A to map to texture coordinate 1 - 1/(2N), which is
// the texture coordinate of the last provided sample value.
constexpr float symbol_to_tex_coord_slope(int N, double A)
{
    return static_cast<float>(1.0 / (2 * N * A));
}
constexpr float symbol_to_tex_coord_intercept(int N)
{
    return static_cast<float>(0.5 + 1.0 / (2 * N));
}

////////////////////////////////////////////////////////////////////////
// LLR_group
// Templated unions to represent the LLR values that come from a single
// estimate.
// T: LLR type (float or __half)
// N: Number of bits for the symbol, based on modulation/QAM
template <typename T, int N> union LLR_group;

template <> union LLR_group<float, 1>
{
    float f[1];
    __device__ void write(void* dst) { *((float*)(dst)) = f[1]; }
};
template <> union LLR_group<__half, 1>
{
    __half f16[1];
    __device__ void write(void* dst) { *((__half*)(dst)) = f16[0]; }
};
template <> union LLR_group<__nv_fp8_e4m3, 1>
{
    __nv_fp8_e4m3 fp8_e4m3[1];
    __device__ void write(void* dst) { *((__nv_fp8_e4m3*)(dst)) = fp8_e4m3[0]; }
};
template <> union LLR_group<__nv_fp8_e5m2, 1>
{
    __nv_fp8_e5m2 fp8_e5m2[1];
    __device__ void write(void* dst) { *((__nv_fp8_e5m2*)(dst)) = fp8_e5m2[0]; }
};

template <> union LLR_group<float, 2>
{
    float f[2];
    uint2 ui32_2;
    __device__ void write(void* dst) { *((uint2*)(dst)) = ui32_2; }
};
template <> union LLR_group<__half, 2>
{
    __half2      f16x2[1];
    unsigned int ui32;
    __device__ void write(void* dst) { *((unsigned int*)(dst)) = ui32; }
};
template <> union LLR_group<__nv_fp8_e4m3, 2>
{
    __nv_fp8x2_e4m3      fp8x2_e4m3[1];
    __nv_fp8x2_storage_t storage;
    __device__ void write(void* dst) { *((__nv_fp8x2_storage_t*)(dst)) = storage; }
};
template <> union LLR_group<__nv_fp8_e5m2, 2>
{
    __nv_fp8x2_e5m2      fp8x2_e5m2[1];
    __nv_fp8x2_storage_t storage;
    __device__ void write(void* dst) { *((__nv_fp8x2_storage_t*)(dst)) = storage; }
};

template <> union LLR_group<float, 4>
{
    float f[4];
    uint4 ui32_4;
    __device__ void write(void* dst) { *((uint4*)(dst)) = ui32_4; }
};
template <> union LLR_group<__half, 4>
{
    __half2 f16x2[2];
    uint2   ui32_2;
    __device__ void write(void* dst) { *((uint2*)(dst)) = ui32_2; }
};
template <> union LLR_group<__nv_fp8_e4m3, 4>
{
    __nv_fp8x2_e4m3      fp8x2_e4m3[2];
    __nv_fp8x4_e4m3      fp8x4_e4m3[1];
    __nv_fp8x4_storage_t storage;
    __device__ void write(void* dst) { *((__nv_fp8x4_storage_t*)(dst)) = storage; }
};
template <> union LLR_group<__nv_fp8_e5m2, 4>
{
    __nv_fp8x2_e5m2      fp8x2_e5m2[2];
    __nv_fp8x4_e5m2      fp8x4_e5m2[1];
    __nv_fp8x4_storage_t storage;
    __device__ void write(void* dst) { *((__nv_fp8x4_storage_t*)(dst)) = storage; }
};

template <> union LLR_group<float, 6>
{
    float        f[6];
    unsigned int ui32[6];
    __device__ void write(void* dst)
    {
        float* fdst = (float*)dst;
        #pragma unroll
        for(int i = 0; i < 6; ++i) { fdst[i] = f[i]; }
    }
};
template <> union LLR_group<__half, 6>
{
    __half2      f16x2[3];
    unsigned int ui32[3];
    __device__ void write(void* dst)
    {
        unsigned int* uidst = (unsigned int*)dst;
        #pragma unroll
        for(int i = 0; i < 3; ++i) { uidst[i] = ui32[i]; }
    }
};
template <> union LLR_group<__nv_fp8_e4m3, 6>
{
    __nv_fp8x2_e4m3      fp8x2_e4m3[3];
    __nv_fp8x2_storage_t storage[3];
    __device__ void write(void* dst)
    {
        __nv_fp8x2_storage_t* sdst = (__nv_fp8x2_storage_t*)dst;
        #pragma unroll
        for(int i = 0; i < 3; ++i) { sdst[i] = storage[i]; }
    }
};
template <> union LLR_group<__nv_fp8_e5m2, 6>
{
    __nv_fp8x2_e5m2      fp8x2_e5m2[3];
    __nv_fp8x2_storage_t storage[3];
    __device__ void write(void* dst)
    {
        __nv_fp8x2_storage_t* sdst = (__nv_fp8x2_storage_t*)dst;
        #pragma unroll
        for(int i = 0; i < 3; ++i) { sdst[i] = storage[i]; }
    }
};

template <> union LLR_group<float, 8>
{
    float        f[8];
    unsigned int ui32[8];
    float4       f4[2];
    float2       f2[4];
    __device__ void write(void* dst)
    {
        float* fdst = (float*)dst;
        #pragma unroll
        for(int i = 0; i < 8; ++i) { fdst[i] = f[i]; }
    }
    __device__ void write(void* dst, int count)
    {
        if(8 == count)
        {
            float4* f4dst = static_cast<float4*>(dst);
            f4dst[0] = f4[0];
            f4dst[1] = f4[1];
        }
        else if(6 == count)
        {
            float* fdst = static_cast<float*>(dst);
            #pragma unroll
            for(int i = 0; i < 6; ++i) { fdst[i] = f[i]; }
        }
        else if(4 == count)
        {
            float4* f4dst = static_cast<float4*>(dst);
            f4dst[0] = f4[0];
        }
        else if(2 == count)
        {
            float2* f2dst = static_cast<float2*>(dst);
            f2dst[0] = f2[0];
        }
        else if(1 == count)
        {
            float* fdst = static_cast<float*>(dst);
            fdst[0] = f[0];
        }
    }
    __device__ float& operator[](int i) { return f[i]; }
};
template <> union LLR_group<__half, 8>
{
    __half  f16[1];
    __half2 f16x2[4];
    uint4   ui32_4;
    uint2   ui32_2[2];
    __device__ void write(void* dst) { *((uint4*)(dst)) = ui32_4; }
    __device__ float operator[](int i)
    {
        return (0 == (i % 2)) ? __low2float(f16x2[i/2]) : __high2float(f16x2[i/2]) ;
    }
    __device__ void write(void* dst, int count)
    {
        if(8 == count)
        {
            uint4* ui4dst = static_cast<uint4*>(dst);
            ui4dst[0] = ui32_4;
        }
        else if(6 == count)
        {
            __half2* hdst = static_cast<__half2*>(dst);
            #pragma unroll
            for(int i = 0; i < 3; ++i) { hdst[i] = f16x2[i]; }
        }
        else if(4 == count)
        {
            uint2* ui2dst = static_cast<uint2*>(dst);
            ui2dst[0] = ui32_2[0];
        }
        else if(2 == count)
        {
            __half2* h2dst = static_cast<__half2*>(dst);
            h2dst[0] = f16x2[0];
        }
        else if(1 == count)
        {
            __half* hdst = static_cast<__half*>(dst);
            hdst[0] = f16[0];
        }
    }
};
template <> union LLR_group<__nv_fp8_e4m3, 8>
{
    __nv_fp8_e4m3        fp8_e4m3[8];
    __nv_fp8x2_e4m3      fp8x2_e4m3[4];
    __nv_fp8x2_storage_t fp8x2_storage[4];
    __nv_fp8x4_e4m3      fp8x4_e4m3[2];
    __nv_fp8x4_storage_t fp8x4_storage[2];
    uint2                ui32_2;
    __device__ void write(void* dst) { *((uint2*)(dst)) = ui32_2; }
    // Debugging float conversion
    __device__ float operator[](int i)
    {
        __half h = static_cast<__half>(fp8_e4m3[i]);
        return static_cast<float>(h);
    }
    __device__ void write(void* dst, int count)
    {
        if(8 == count)
        {
            uint2* ui2dst = static_cast<uint2*>(dst);
            ui2dst[0] = ui32_2;
        }
        else if(6 == count)
        {
            __nv_fp8x2_storage_t* sdst = static_cast<__nv_fp8x2_storage_t*>(dst);
            #pragma unroll
            for(int i = 0; i < 3; ++i) { sdst[i] = fp8x2_storage[i]; }
        }
        else if(4 == count)
        {
            __nv_fp8x4_storage_t* sdst = static_cast<__nv_fp8x4_storage_t*>(dst);
            sdst[0] = fp8x4_storage[0];
        }
        else if(2 == count)
        {
            __nv_fp8x2_storage_t* sdst = static_cast<__nv_fp8x2_storage_t*>(dst);
            sdst[0] = fp8x2_storage[0];
        }
        else if(1 == count)
        {
            __nv_fp8_e4m3* sdst = static_cast<__nv_fp8_e4m3*>(dst);
            sdst[0] = fp8_e4m3[0];
        }
    }
};
template <> union LLR_group<__nv_fp8_e5m2, 8>
{
    __nv_fp8_e5m2        fp8_e5m2[8];
    __nv_fp8x2_e5m2      fp8x2_e5m2[4];
    __nv_fp8x2_storage_t fp8x2_storage[4];
    __nv_fp8x4_e5m2      fp8x4_e5m2[2];
    __nv_fp8x4_storage_t fp8x4_storage[2];
    uint2                ui32_2;
    __device__ void write(void* dst) { *((uint2*)(dst)) = ui32_2; }
    // Debugging float conversion
    __device__ float operator[](int i)
    {
        __half h = static_cast<__half>(fp8_e5m2[i]);
        return static_cast<float>(h);
    }
    __device__ void write(void* dst, int count)
    {
        if(8 == count)
        {
            uint2* ui2dst = static_cast<uint2*>(dst);
            ui2dst[0] = ui32_2;
        }
        else if(6 == count)
        {
            __nv_fp8x2_storage_t* sdst = static_cast<__nv_fp8x2_storage_t*>(dst);
            #pragma unroll
            for(int i = 0; i < 3; ++i) { sdst[i] = fp8x2_storage[i]; }
        }
        else if(4 == count)
        {
            __nv_fp8x4_storage_t* sdst = static_cast<__nv_fp8x4_storage_t*>(dst);
            sdst[0] = fp8x4_storage[0];
        }
        else if(2 == count)
        {
            __nv_fp8x2_storage_t* sdst = static_cast<__nv_fp8x2_storage_t*>(dst);
            sdst[0] = fp8x2_storage[0];
        }
        else if(1 == count)
        {
            __nv_fp8_e5m2* sdst = static_cast<__nv_fp8_e5m2*>(dst);
            sdst[0] = fp8_e5m2[0];
        }
    }
};

////////////////////////////////////////////////////////////////////////
// QAM_traits
template <int QAM> struct QAM_traits;
template <>
struct QAM_traits<256>
{
    static constexpr int    bits     = 8;
    static constexpr int    PAM_bits = 4;
    static constexpr int    N        = 32;                               // texture size
    static constexpr double A        = 0.076696498884737;                // 1.0 / sqrt(170.0) (modulation norm)
    static constexpr float  m        = symbol_to_tex_coord_slope(N, A);  // slope in y = mx + b
    static constexpr float  b        = symbol_to_tex_coord_intercept(N); // y-intercept in y = mx + b
    static constexpr float  LEVEL    = 0.0f;                             // texture LOD
    static float symbol_to_tex_coord(float x) { return (m * x) + b; }
};
template <>
struct QAM_traits<64>
{
    static constexpr int    bits     = 6;
    static constexpr int    PAM_bits = 3;
    static constexpr int    N        = 16;                               // texture size
    static constexpr double A        = 0.154303349962092;                // 1.0 / sqrt(42.0) (modulation norm)
    static constexpr float  m        = symbol_to_tex_coord_slope(N, A);  // slope in y = mx + b
    static constexpr float  b        = symbol_to_tex_coord_intercept(N); // y-intercept in y = mx + b
    static constexpr float  LEVEL    = 1.0f;                             // texture LOD
    static float symbol_to_tex_coord(float x) { return (m * x) + b; }
};
template <>
struct QAM_traits<16>
{
    static constexpr int    bits     = 4;
    static constexpr int    PAM_bits = 2;
    static constexpr int    N        = 8;                                // texture size
    static constexpr double A        = 0.316227766016838;                // 1.0 / sqrt(10.0) (modulation norm)
    static constexpr float  m        = symbol_to_tex_coord_slope(N, A);  // slope in y = mx + b
    static constexpr float  b        = symbol_to_tex_coord_intercept(N); // y-intercept in y = mx + b
    static constexpr float  LEVEL    = 2.0f;                             // texture LOD
    static float symbol_to_tex_coord(float x) { return (m * x) + b; }
};
template <>
struct QAM_traits<4>
{
    static constexpr int    bits     = 2;
    static constexpr int    PAM_bits = 1;
    static constexpr int    N        = 4;                                // texture size
    static constexpr double A        = 0.707106781186547;                // 1.0 / sqrt(2.0) (modulation norm)
    static constexpr float  m        = symbol_to_tex_coord_slope(N, A);  // slope in y = mx + b
    static constexpr float  b        = symbol_to_tex_coord_intercept(N); // y-intercept in y = mx + b
    static constexpr float  LEVEL    = 3.0f;                             // texture LOD
    static float symbol_to_tex_coord(float x) { return (m * x) + b; }
};
template <>
struct QAM_traits<2>
{
    static constexpr int    bits     = 1;
    static constexpr int    PAM_bits = 1;
    static constexpr int    N        = 4;                                // texture size (reusing QAM_traits<4>)
    static constexpr double A        = 0.707106781186547;                // 1.0 / sqrt(2.0) (modulation norm)
    static constexpr float  m        = symbol_to_tex_coord_slope(N, A);  // slope in y = mx + b
    static constexpr float  b        = symbol_to_tex_coord_intercept(N); // y-intercept in y = mx + b
    static constexpr float  LEVEL    = 3.0f;                             // texture LOD (reusing QAM4!)
    static float symbol_to_tex_coord(float x) { return (m * x) + b; }
};

////////////////////////////////////////////////////////////////////////
// Constant memory table lookup for transformation from symbol values
// to texture coordinates.
struct mod_symbol_to_tex_coord_t
{
    float m;
    float b;
};
__constant__ mod_symbol_to_tex_coord_t sym_transform[9] =
{
    //                m                   b         num_bits
    {              0.0f,               0.0f },   // 0   (INVALID)
    {  QAM_traits<2>::m,   QAM_traits<2>::b },   // 1   BPSK
    {  QAM_traits<4>::m,   QAM_traits<4>::b },   // 2   QPSK
    {              0.0f,               0.0f },   // 3   (INVALID)
    { QAM_traits<16>::m,  QAM_traits<16>::b },   // 4   QAM16
    {              0.0f,               0.0f },   // 5   (INVALID)
    { QAM_traits<64>::m,  QAM_traits<64>::b },   // 6   QAM64
    {              0.0f,               0.0f },   // 7   (INVALID)
    {QAM_traits<256>::m, QAM_traits<256>::b }    // 8   QAM256
};

////////////////////////////////////////////////////////////////////////
// Constant memory table lookup for mipmap texture level
__constant__ float mod_mipmap_level[9] =
{
    // level                   num_bits
    0.0f,                   // 0   (INVALID)
    QAM_traits<2>::LEVEL,   // 1   BPSK
    QAM_traits<4>::LEVEL,   // 2   QPSK
    0.0f,                   // 3   (INVALID)
    QAM_traits<16>::LEVEL,  // 4   QAM16
    0.0f,                   // 5   (INVALID)
    QAM_traits<64>::LEVEL,  // 6   QAM64
    0.0f,                   // 7   (INVALID)
    QAM_traits<256>::LEVEL  // 8   QAM256
};

////////////////////////////////////////////////////////////////////////
// symbol_to_tex_coords()
// Linear transformation from a complex symbol value to a pair of
// normalized texture coordinates.
inline __device__
float2 symbol_to_tex_coords(const cuFloatComplex& sym, float m, float b)
{
    float2 f2;
    f2.x = m * sym.x + b;
    f2.y = m * sym.y + b;
    return f2;
}
inline __device__
float2 symbol_to_tex_coords(const cuFloatComplex& sym, int QAM_bits)
{
    float2 f2;
    f2.x = sym_transform[QAM_bits].m * sym.x + sym_transform[QAM_bits].b;
    f2.y = sym_transform[QAM_bits].m * sym.y + sym_transform[QAM_bits].b;
    return f2;
}
inline __device__
float2 symbol_to_tex_coords(const __half2& sym, float m, float b)
{
    // Note that texture fetches require float input, so we convert __half
    // symbol components to float before returning.
    __half2 m_m = __float2half2_rn(m);
    __half2 b_b = __float2half2_rn(b);
    __half2 u_v = __hfma2(m_m, sym, b_b);
    return __half22float2(u_v);
}
inline __device__
float2 symbol_to_tex_coords(const __half2& sym, const __half2&  m_m, const __half2& b_b)
{
    // Note that texture fetches require float input, so we convert __half
    // symbol components to float before returning.
    __half2 u_v = __hfma2(m_m, sym, b_b);
    return __half22float2(u_v);
}
inline __device__
float2 symbol_to_tex_coords(const __half2& sym, int QAM_bits)
{
    // Note that texture fetches require float input, so we convert __half
    // symbol components to float before returning.
    __half2 u_v = __hfma2(sym, sym_transform_h[QAM_bits].m, sym_transform_h[QAM_bits].b);
    return __half22float2(u_v);
}
// Special case for BPSK (QAM2)
inline __device__
float symbol_to_tex_coords(float sym_component, float m, float b)
{
    return (m * sym_component) + b;
}

template <typename T> struct demapper_llr_traits_t;

template <>
struct demapper_llr_traits_t<float>
{
    using tex_scalar_t = float;
    using tex_result_t = cuphy_i::tex_result_v4<tex_scalar_t>;
};
template <>
struct demapper_llr_traits_t<__half>
{
    using tex_scalar_t = __half;
    using tex_result_t = cuphy_i::tex_result_v4<tex_scalar_t>;
};
template <>
struct demapper_llr_traits_t<__nv_fp8_e4m3>
{
    using tex_scalar_t = __half;
    using tex_result_t = cuphy_i::tex_result_v4<tex_scalar_t>;
};
template <>
struct demapper_llr_traits_t<__nv_fp8_e5m2>
{
    using tex_scalar_t = __half;
    using tex_result_t = cuphy_i::tex_result_v4<tex_scalar_t>;
};

inline __device__
void apply_noise(cuphy_i::tex_result_v4<float>& r, float noiseInv)
{
    r.x *= noiseInv;
    r.y *= noiseInv;
    r.z *= noiseInv;
    r.w *= noiseInv;
}

inline __device__
void apply_noise(cuphy_i::tex_result_v4<__half>& r, float noiseInv)
{
    __half2 hnoise = __float2half2_rn(noiseInv);
    r.a.f16x2 = __hmul2(r.a.f16x2, hnoise);
    r.b.f16x2 = __hmul2(r.b.f16x2, hnoise);
}

inline __device__
float sum_components(const __half2& sym)
{
    return (__low2float(sym) + __high2float(sym));
}

inline __device__
float sum_components(const cuFloatComplex& sym)
{
    return (sym.x + sym.y);
}

////////////////////////////////////////////////////////////////////////
// swizzle_LLRs()
inline __device__
void swizzle_LLRs(LLR_group<float, 1>&                 LLR_grp,
                  const cuphy_i::tex_result_v4<float>& res_I,
                  const cuphy_i::tex_result_v4<float>& res_Q)
{
    LLR_grp.f[0] = res_I.x;
}
inline __device__
void swizzle_LLRs(LLR_group<float, 2>&                 LLR_grp,
                  const cuphy_i::tex_result_v4<float>& res_I,
                  const cuphy_i::tex_result_v4<float>& res_Q)
{
    LLR_grp.f[0] = res_I.x;
    LLR_grp.f[1] = res_Q.x;
}
inline __device__
void swizzle_LLRs(LLR_group<float, 4>&                 LLR_grp,
                  const cuphy_i::tex_result_v4<float>& res_I,
                  const cuphy_i::tex_result_v4<float>& res_Q)
{
    LLR_grp.f[0] = res_I.x;
    LLR_grp.f[1] = res_Q.x;
    LLR_grp.f[2] = res_I.y;
    LLR_grp.f[3] = res_Q.y;
}
inline __device__
void swizzle_LLRs(LLR_group<float, 6>&                 LLR_grp,
                  const cuphy_i::tex_result_v4<float>& res_I,
                  const cuphy_i::tex_result_v4<float>& res_Q)
{
    LLR_grp.f[0] = res_I.x;
    LLR_grp.f[1] = res_Q.x;
    LLR_grp.f[2] = res_I.y;
    LLR_grp.f[3] = res_Q.y;
    LLR_grp.f[4] = res_I.z;
    LLR_grp.f[5] = res_Q.z;
}
inline __device__
void swizzle_LLRs(LLR_group<float, 8>&                 LLR_grp,
                  const cuphy_i::tex_result_v4<float>& res_I,
                  const cuphy_i::tex_result_v4<float>& res_Q)
{
    LLR_grp.f[0] = res_I.x;
    LLR_grp.f[1] = res_Q.x;
    LLR_grp.f[2] = res_I.y;
    LLR_grp.f[3] = res_Q.y;
    LLR_grp.f[4] = res_I.z;
    LLR_grp.f[5] = res_Q.z;
    LLR_grp.f[6] = res_I.w;
    LLR_grp.f[7] = res_Q.w;
}
inline __device__
void swizzle_LLRs(LLR_group<__half, 1>&                 LLR_grp,
                  const cuphy_i::tex_result_v4<__half>& res_I,
                  const cuphy_i::tex_result_v4<__half>& res_Q)
{
    LLR_grp.f16[0] = __low2half(res_I.a.f16x2);
}
inline __device__
void swizzle_LLRs(LLR_group<__half, 2>&                 LLR_grp,
                  const cuphy_i::tex_result_v4<__half>& res_I,
                  const cuphy_i::tex_result_v4<__half>& res_Q)
{
    //   7  6  5  4     3  2  1  0
    // [ Q.hi  Q.lo ] [ I.hi  I.lo ] --> [ Q.lo I.lo ]
    LLR_grp.ui32 = uint32_permute<0x5410>(res_I.a.u32, res_Q.a.u32);
}
inline __device__
void swizzle_LLRs(LLR_group<__half, 4>&                 LLR_grp,
                  const cuphy_i::tex_result_v4<__half>& res_I,
                  const cuphy_i::tex_result_v4<__half>& res_Q)
{
    //    7  6    5  4       3  2    1  0
    // [ Q.a.hi  Q.a.lo ] [ I.a.hi  I.a.lo ] --> [ Q.a.lo I.a.lo ]
    LLR_grp.ui32_2.x = uint32_permute<0x5410>(res_I.a.u32, res_Q.a.u32);
    //    7  6    5  4       3  2    1  0
    // [ Q.a.hi  Q.a.lo ] [ I.a.hi  I.a.lo ] --> [ Q.a.hi I.a.hi ]
    LLR_grp.ui32_2.y = uint32_permute<0x7632>(res_I.a.u32, res_Q.a.u32);
}
inline __device__
void swizzle_LLRs(LLR_group<__half, 6>&                 LLR_grp,
                  const cuphy_i::tex_result_v4<__half>& res_I,
                  const cuphy_i::tex_result_v4<__half>& res_Q)
{
    //    7  6    5  4       3  2    1  0
    // [ Q.a.hi  Q.a.lo ] [ I.a.hi  I.a.lo ] --> [ Q.a.lo I.a.lo ]
    LLR_grp.ui32[0] = uint32_permute<0x5410>(res_I.a.u32, res_Q.a.u32);
    //    7  6    5  4       3  2    1  0
    // [ Q.a.hi  Q.a.lo ] [ I.a.hi  I.a.lo ] --> [ Q.a.hi I.a.hi ]
    LLR_grp.ui32[1] = uint32_permute<0x7632>(res_I.a.u32, res_Q.a.u32);
    //    7  6    5  4       3  2    1  0
    // [ Q.b.hi  Q.b.lo ] [ I.b.hi  I.b.lo ] --> [ Q.b.lo I.b.lo ]
    LLR_grp.ui32[2] = uint32_permute<0x5410>(res_I.b.u32, res_Q.b.u32);
}
inline __device__
void swizzle_LLRs(LLR_group<__half, 8>&                 LLR_grp,
                  const cuphy_i::tex_result_v4<__half>& res_I,
                  const cuphy_i::tex_result_v4<__half>& res_Q)
{
    //    7  6    5  4       3  2    1  0
    // [ Q.a.hi  Q.a.lo ] [ I.a.hi  I.a.lo ] --> [ Q.a.lo I.a.lo ]
    LLR_grp.ui32_4.x = uint32_permute<0x5410>(res_I.a.u32, res_Q.a.u32);
    //    7  6    5  4       3  2    1  0
    // [ Q.a.hi  Q.a.lo ] [ I.a.hi  I.a.lo ] --> [ Q.a.hi I.a.hi ]
    LLR_grp.ui32_4.y = uint32_permute<0x7632>(res_I.a.u32, res_Q.a.u32);
    //    7  6    5  4       3  2    1  0
    // [ Q.b.hi  Q.b.lo ] [ I.b.hi  I.b.lo ] --> [ Q.b.lo I.b.lo ]
    LLR_grp.ui32_4.z = uint32_permute<0x5410>(res_I.b.u32, res_Q.b.u32);
    //    7  6    5  4       3  2    1  0
    // [ Q.b.hi  Q.b.lo ] [ I.b.hi  I.b.lo ] --> [ Q.b.hi I.b.hi ]
    LLR_grp.ui32_4.w = uint32_permute<0x7632>(res_I.b.u32, res_Q.b.u32);
}

inline __device__
void swizzle_LLRs(LLR_group<__nv_fp8_e4m3, 1>&          LLR_grp,
                  const cuphy_i::tex_result_v4<__half>& res_I,
                  const cuphy_i::tex_result_v4<__half>& res_Q)
{
    LLR_grp.fp8_e4m3[0] = __nv_fp8_e4m3(__low2half(res_I.a.f16x2));
}
inline __device__
void swizzle_LLRs(LLR_group<__nv_fp8_e4m3, 2>&          LLR_grp,
                  const cuphy_i::tex_result_v4<__half>& res_I,
                  const cuphy_i::tex_result_v4<__half>& res_Q)
{
    //   7  6  5  4     3  2  1  0
    // [ Q.hi  Q.lo ] [ I.hi  I.lo ] --> [ Q.lo I.lo ]
    __half2 hres = __lows2half2(res_I.a.f16x2, res_Q.a.f16x2);
    LLR_grp.fp8x2_e4m3[0] = __nv_fp8x2_e4m3(hres);
}
inline __device__
void swizzle_LLRs(LLR_group<__nv_fp8_e4m3, 4>&          LLR_grp,
                  const cuphy_i::tex_result_v4<__half>& res_I,
                  const cuphy_i::tex_result_v4<__half>& res_Q)
{
    //    7  6    5  4       3  2    1  0
    // [ Q.a.hi  Q.a.lo ] [ I.a.hi  I.a.lo ] --> [ Q.a.lo I.a.lo ]
    __half2 hres_low = __lows2half2(res_I.a.f16x2, res_Q.a.f16x2);
    //    7  6    5  4       3  2    1  0
    // [ Q.a.hi  Q.a.lo ] [ I.a.hi  I.a.lo ] --> [ Q.a.hi I.a.hi ]
    __half2 hres_hi = __highs2half2(res_I.a.f16x2, res_Q.a.f16x2);
    LLR_grp.fp8x4_e4m3[0] = __nv_fp8x4_e4m3(hres_low, hres_hi);
}
inline __device__
void swizzle_LLRs(LLR_group<__nv_fp8_e4m3, 6>&          LLR_grp,
                  const cuphy_i::tex_result_v4<__half>& res_I,
                  const cuphy_i::tex_result_v4<__half>& res_Q)
{
    //    7  6    5  4       3  2    1  0
    // [ Q.a.hi  Q.a.lo ] [ I.a.hi  I.a.lo ] --> [ Q.a.lo I.a.lo ]
    __half2 hres_0 = __lows2half2(res_I.a.f16x2, res_Q.a.f16x2);
    //    7  6    5  4       3  2    1  0
    // [ Q.a.hi  Q.a.lo ] [ I.a.hi  I.a.lo ] --> [ Q.a.hi I.a.hi ]
    __half2 hres_1 = __highs2half2(res_I.a.f16x2, res_Q.a.f16x2);
    //    7  6    5  4       3  2    1  0
    // [ Q.b.hi  Q.b.lo ] [ I.b.hi  I.b.lo ] --> [ Q.b.lo I.b.lo ]
    __half2 hres_2 = __lows2half2(res_I.b.f16x2, res_Q.b.f16x2);

    LLR_grp.fp8x2_e4m3[0] = __nv_fp8x2_e4m3(hres_0);
    LLR_grp.fp8x2_e4m3[1] = __nv_fp8x2_e4m3(hres_1);
    LLR_grp.fp8x2_e4m3[2] = __nv_fp8x2_e4m3(hres_2);
}
inline __device__
void swizzle_LLRs(LLR_group<__nv_fp8_e4m3, 8>&          LLR_grp,
                  const cuphy_i::tex_result_v4<__half>& res_I,
                  const cuphy_i::tex_result_v4<__half>& res_Q)
{
    //    7  6    5  4       3  2    1  0
    // [ Q.a.hi  Q.a.lo ] [ I.a.hi  I.a.lo ] --> [ Q.a.lo I.a.lo ]
    __half2 hres_0 = __lows2half2(res_I.a.f16x2, res_Q.a.f16x2);
    //    7  6    5  4       3  2    1  0
    // [ Q.a.hi  Q.a.lo ] [ I.a.hi  I.a.lo ] --> [ Q.a.hi I.a.hi ]
    __half2 hres_1 = __highs2half2(res_I.a.f16x2, res_Q.a.f16x2);
    //    7  6    5  4       3  2    1  0
    // [ Q.b.hi  Q.b.lo ] [ I.b.hi  I.b.lo ] --> [ Q.b.lo I.b.lo ]
    __half2 hres_2 = __lows2half2(res_I.b.f16x2, res_Q.b.f16x2);
    //    7  6    5  4       3  2    1  0
    // [ Q.b.hi  Q.b.lo ] [ I.b.hi  I.b.lo ] --> [ Q.b.hi I.b.hi ]
    __half2 hres_3 = __highs2half2(res_I.b.f16x2, res_Q.b.f16x2);

    LLR_grp.fp8x4_e4m3[0] = __nv_fp8x4_e4m3(hres_0, hres_1);
    LLR_grp.fp8x4_e4m3[1] = __nv_fp8x4_e4m3(hres_2, hres_3);
}

inline __device__
void swizzle_LLRs(LLR_group<__nv_fp8_e5m2, 1>&          LLR_grp,
                  const cuphy_i::tex_result_v4<__half>& res_I,
                  const cuphy_i::tex_result_v4<__half>& res_Q)
{
    LLR_grp.fp8_e5m2[0] = __nv_fp8_e5m2(__low2half(res_I.a.f16x2));
}
inline __device__
void swizzle_LLRs(LLR_group<__nv_fp8_e5m2, 2>&          LLR_grp,
                  const cuphy_i::tex_result_v4<__half>& res_I,
                  const cuphy_i::tex_result_v4<__half>& res_Q)
{
    //   7  6  5  4     3  2  1  0
    // [ Q.hi  Q.lo ] [ I.hi  I.lo ] --> [ Q.lo I.lo ]
    __half2 hres = __lows2half2(res_I.a.f16x2, res_Q.a.f16x2);
    LLR_grp.fp8x2_e5m2[0] = __nv_fp8x2_e5m2(hres);
}
inline __device__
void swizzle_LLRs(LLR_group<__nv_fp8_e5m2, 4>&          LLR_grp,
                  const cuphy_i::tex_result_v4<__half>& res_I,
                  const cuphy_i::tex_result_v4<__half>& res_Q)
{
    //    7  6    5  4       3  2    1  0
    // [ Q.a.hi  Q.a.lo ] [ I.a.hi  I.a.lo ] --> [ Q.a.lo I.a.lo ]
    __half2 hres_low = __lows2half2(res_I.a.f16x2, res_Q.a.f16x2);
    //    7  6    5  4       3  2    1  0
    // [ Q.a.hi  Q.a.lo ] [ I.a.hi  I.a.lo ] --> [ Q.a.hi I.a.hi ]
    __half2 hres_hi = __highs2half2(res_I.a.f16x2, res_Q.a.f16x2);
    LLR_grp.fp8x4_e5m2[0] = __nv_fp8x4_e5m2(hres_low, hres_hi);
}
inline __device__
void swizzle_LLRs(LLR_group<__nv_fp8_e5m2, 6>&          LLR_grp,
                  const cuphy_i::tex_result_v4<__half>& res_I,
                  const cuphy_i::tex_result_v4<__half>& res_Q)
{
    //    7  6    5  4       3  2    1  0
    // [ Q.a.hi  Q.a.lo ] [ I.a.hi  I.a.lo ] --> [ Q.a.lo I.a.lo ]
    __half2 hres_0 = __lows2half2(res_I.a.f16x2, res_Q.a.f16x2);
    //    7  6    5  4       3  2    1  0
    // [ Q.a.hi  Q.a.lo ] [ I.a.hi  I.a.lo ] --> [ Q.a.hi I.a.hi ]
    __half2 hres_1 = __highs2half2(res_I.a.f16x2, res_Q.a.f16x2);
    //    7  6    5  4       3  2    1  0
    // [ Q.b.hi  Q.b.lo ] [ I.b.hi  I.b.lo ] --> [ Q.b.lo I.b.lo ]
    __half2 hres_2 = __lows2half2(res_I.b.f16x2, res_Q.b.f16x2);

    LLR_grp.fp8x2_e5m2[0] = __nv_fp8x2_e5m2(hres_0);
    LLR_grp.fp8x2_e5m2[1] = __nv_fp8x2_e5m2(hres_1);
    LLR_grp.fp8x2_e5m2[2] = __nv_fp8x2_e5m2(hres_2);
}
inline __device__
void swizzle_LLRs(LLR_group<__nv_fp8_e5m2, 8>&          LLR_grp,
                  const cuphy_i::tex_result_v4<__half>& res_I,
                  const cuphy_i::tex_result_v4<__half>& res_Q)
{
    //    7  6    5  4       3  2    1  0
    // [ Q.a.hi  Q.a.lo ] [ I.a.hi  I.a.lo ] --> [ Q.a.lo I.a.lo ]
    __half2 hres_0 = __lows2half2(res_I.a.f16x2, res_Q.a.f16x2);
    //    7  6    5  4       3  2    1  0
    // [ Q.a.hi  Q.a.lo ] [ I.a.hi  I.a.lo ] --> [ Q.a.hi I.a.hi ]
    __half2 hres_1 = __highs2half2(res_I.a.f16x2, res_Q.a.f16x2);
    //    7  6    5  4       3  2    1  0
    // [ Q.b.hi  Q.b.lo ] [ I.b.hi  I.b.lo ] --> [ Q.b.lo I.b.lo ]
    __half2 hres_2 = __lows2half2(res_I.b.f16x2, res_Q.b.f16x2);
    //    7  6    5  4       3  2    1  0
    // [ Q.b.hi  Q.b.lo ] [ I.b.hi  I.b.lo ] --> [ Q.b.hi I.b.hi ]
    __half2 hres_3 = __highs2half2(res_I.b.f16x2, res_Q.b.f16x2);

    LLR_grp.fp8x4_e5m2[0] = __nv_fp8x4_e5m2(hres_0, hres_1);
    LLR_grp.fp8x4_e5m2[1] = __nv_fp8x4_e5m2(hres_2, hres_3);
}

////////////////////////////////////////////////////////////////////////
// LLR_BPSK
// Specialized soft demapper for BPSK, using direct arithmetic instead
// of a texture fetch.
template <typename TSymbolScalar, typename TLLR> struct LLR_BPSK;

// complex float symbol, float LLR
template <>
struct LLR_BPSK<float, float>
{
    //typedef LLR_group<float, 1>                       llr_group_t;
    typedef typename complex_from_scalar<float>::type symbol_t;
    template <class TLLRGroup>
    __device__
    static void symbol_to_LLR_group(TLLRGroup&      grp,            // output LLRs
                                    const symbol_t& sym,            // input symbol
                                    float           PAMnoiseVarInv) // inverse of PAM noise variance
    {
        grp.f[0]= PAMnoiseVarInv * 2 * QAM_traits<2>::A * (sym.x + sym.y);
    }
};

// complex float symbol, __half LLR
template <>
struct LLR_BPSK<float, __half>
{
    //typedef LLR_group<__half, 1>                      llr_group_t;
    typedef typename complex_from_scalar<float>::type symbol_t;
    template <class TLLRGroup>
    __device__
    static void symbol_to_LLR_group(TLLRGroup&      grp,            // output LLRs
                                    const symbol_t& sym,            // input symbol
                                    float           PAMnoiseVarInv) // inverse of PAM noise variance
    {
        __half ZrA2 = __float2half(2 * QAM_traits<2>::A * (sym.x + sym.y));
        grp.f16[0]  = __hmul(__float2half(PAMnoiseVarInv), ZrA2);
    }
};

/// complex __half symbol, __half LLR
template <>
struct LLR_BPSK<__half, __half>
{
    //typedef LLR_group<__half, 1>                       llr_group_t;
    typedef typename complex_from_scalar<__half>::type symbol_t;
    template <class TLLRGroup>
    __device__
    static void symbol_to_LLR_group(TLLRGroup&      grp,            // output LLRs
                                    const symbol_t& sym,            // input symbol
                                    float           PAMnoiseVarInv) // inverse of PAM noise variance
    {
        __half Zr  = __low2half(sym) + __high2half(sym);
        __half A2  = __float2half(2 * QAM_traits<2>::A);
        grp.f16[0] = __hmul(__hmul(__float2half(PAMnoiseVarInv), A2), Zr);
    }
};

// complex __half symbol, float LLR
template <>
struct LLR_BPSK<__half, float>
{
    //typedef LLR_group<float, 1>                        llr_group_t;
    typedef typename complex_from_scalar<__half>::type symbol_t;
    template <class TLLRGroup>
    __device__
    static void symbol_to_LLR_group(TLLRGroup&      grp,            // output LLRs
                                    const symbol_t& sym,            // input symbol
                                    float           PAMnoiseVarInv) // inverse of PAM noise variance
    {
        __half Zr = __low2half(sym) + __high2half(sym);
        grp.f[0]  = 2 * QAM_traits<2>::A * PAMnoiseVarInv * __half2float(Zr);
    }
};

// complex __half symbol, __nv_fp8_e4m3 LLR
template <>
struct LLR_BPSK<__half, __nv_fp8_e4m3>
{
    //typedef LLR_group<__nv_fp8_e4m3, 1>                        llr_group_t;
    typedef typename complex_from_scalar<__half>::type symbol_t;
    template <class TLLRGroup>
    __device__
    static void symbol_to_LLR_group(TLLRGroup&      grp,            // output LLRs
                                    const symbol_t& sym,            // input symbol
                                    float           PAMnoiseVarInv) // inverse of PAM noise variance
    {
        __half Zr       = __low2half(sym) + __high2half(sym);
        __half result   = __hmul(__float2half_rn(2 * QAM_traits<2>::A * PAMnoiseVarInv), Zr);
        grp.fp8_e4m3[0] = __nv_fp8_e4m3(result);
    }
};

// complex __half symbol, __nv_fp8_e5m2 LLR
template <>
struct LLR_BPSK<__half, __nv_fp8_e5m2>
{
    //typedef LLR_group<__nv_fp8_e5m2, 1>                        llr_group_t;
    typedef typename complex_from_scalar<__half>::type symbol_t;
    template <class TLLRGroup>
    __device__
    static void symbol_to_LLR_group(TLLRGroup&      grp,            // output LLRs
                                    const symbol_t& sym,            // input symbol
                                    float           PAMnoiseVarInv) // inverse of PAM noise variance
    {
        __half Zr       = __low2half(sym) + __high2half(sym);
        __half result   = __hmul(__float2half_rn(2 * QAM_traits<2>::A * PAMnoiseVarInv), Zr);
        grp.fp8_e5m2[0] = __nv_fp8_e5m2(result);
    }
};

// complex float symbol, __nv_fp8_e4m3 LLR
template <>
struct LLR_BPSK<float, __nv_fp8_e4m3>
{
    //typedef LLR_group<__nv_fp8_e4m3, 1>             llr_group_t;
    typedef typename complex_from_scalar<float>::type symbol_t;
    template <class TLLRGroup>
    __device__
    static void symbol_to_LLR_group(TLLRGroup&      grp,            // output LLRs
                                    const symbol_t& sym,            // input symbol
                                    float           PAMnoiseVarInv) // inverse of PAM noise variance
    {
        float ZrA2      = 2 * QAM_traits<2>::A * PAMnoiseVarInv * (sym.x + sym.y);
        grp.fp8_e4m3[0] = __nv_fp8_e4m3(ZrA2);
    }
};

// complex float symbol, __nv_fp8_e5m2 LLR
template <>
struct LLR_BPSK<float, __nv_fp8_e5m2>
{
    //typedef LLR_group<__nv_fp8_e5m2, 1>             llr_group_t;
    typedef typename complex_from_scalar<float>::type symbol_t;
    template <class TLLRGroup>
    __device__
    static void symbol_to_LLR_group(TLLRGroup&      grp,            // output LLRs
                                    const symbol_t& sym,            // input symbol
                                    float           PAMnoiseVarInv) // inverse of PAM noise variance
    {
        float ZrA2      = 2 * QAM_traits<2>::A * PAMnoiseVarInv * (sym.x + sym.y);
        grp.fp8_e5m2[0] = __nv_fp8_e5m2(ZrA2);
    }
};

////////////////////////////////////////////////////////////////////////
// LLR_QPSK
// Specialized soft demapper for QPSK, using direct arithmetic instead
// of a texture fetch.
template <typename TSymbolScalar, typename TLLR> struct LLR_QPSK;

// complex float symbol, float LLR
template <>
struct LLR_QPSK<float, float>
{
    //typedef LLR_group<float, 2>                       llr_group_t;
    typedef typename complex_from_scalar<float>::type symbol_t;
    template <class TLLRGroup>
    __device__
    static void symbol_to_LLR_group(TLLRGroup&      grp,            // output LLRs
                                    const symbol_t& sym,            // input symbol
                                    float           PAMnoiseVarInv) // inverse of PAM noise variance
    {
        grp.f[0]= PAMnoiseVarInv * 2 * QAM_traits<2>::A * sym.x;
        grp.f[1]= PAMnoiseVarInv * 2 * QAM_traits<2>::A * sym.y;
    }
};

// complex float symbol, __half LLR
template <>
struct LLR_QPSK<float, __half>
{
    //typedef LLR_group<__half, 2>                      llr_group_t;
    typedef typename complex_from_scalar<float>::type symbol_t;
    template <class TLLRGroup>
    __device__
    static void symbol_to_LLR_group(TLLRGroup&      grp,            // output LLRs
                                    const symbol_t& sym,            // input symbol
                                    float           PAMnoiseVarInv) // inverse of PAM noise variance
    {
        __half2 A2       = __float2half2_rn(2 * QAM_traits<2>::A);
        __half2 noiseInv = __float2half2_rn(PAMnoiseVarInv);
        grp.f16x2[0] = __hmul2(__hmul2(noiseInv, A2), __floats2half2_rn(sym.x, sym.y));

    }
};

// complex __half symbol, __half LLR
template <>
struct LLR_QPSK<__half, __half>
{
    //typedef LLR_group<__half, 2>                       llr_group_t;
    typedef typename complex_from_scalar<__half>::type symbol_t;
    template <class TLLRGroup>
    __device__
    static void symbol_to_LLR_group(TLLRGroup&      grp,            // output LLRs
                                    const symbol_t& sym,            // input symbol
                                    float           PAMnoiseVarInv) // inverse of PAM noise variance
    {
        __half2 A2       = __float2half2_rn(2 * QAM_traits<2>::A);
        __half2 noiseInv = __float2half2_rn(PAMnoiseVarInv);
        grp.f16x2[0]     = __hmul2(__hmul2(noiseInv, A2), sym);
    }
};

// complex __half symbol, float LLR
template <>
struct LLR_QPSK<__half, float>
{
    //typedef LLR_group<float, 2>                        llr_group_t;
    typedef typename complex_from_scalar<__half>::type symbol_t;
    template <class TLLRGroup>
    __device__
    static void symbol_to_LLR_group(TLLRGroup&      grp,            // output LLRs
                                    const symbol_t& sym,            // input symbol
                                    float           PAMnoiseVarInv) // inverse of PAM noise variance
    {
        grp.f[0]  = 2 * QAM_traits<2>::A * PAMnoiseVarInv * __low2float(sym);
        grp.f[1]  = 2 * QAM_traits<2>::A * PAMnoiseVarInv * __high2float(sym);
    }
};

// complex __half symbol, __nv_fp8_e4m3 LLR
template <>
struct LLR_QPSK<__half, __nv_fp8_e4m3>
{
    //typedef LLR_group<__nv_fp8_e4m3, 2>              llr_group_t;
    typedef typename complex_from_scalar<__half>::type symbol_t;
    template <class TLLRGroup>
    __device__
    static void symbol_to_LLR_group(TLLRGroup&      grp,            // output LLRs
                                    const symbol_t& sym,            // input symbol
                                    float           PAMnoiseVarInv) // inverse of PAM noise variance
    {
        __half2 result = __hmul2(__float2half2_rn(2 * QAM_traits<2>::A * PAMnoiseVarInv),
                                 sym);
        grp.fp8x2_e4m3[0] = __nv_fp8x2_e4m3(result);
    }
};

// complex __half symbol, __nv_fp8_e5m2 LLR
template <>
struct LLR_QPSK<__half, __nv_fp8_e5m2>
{
    //typedef LLR_group<__nv_fp8_e5m2, 2>              llr_group_t;
    typedef typename complex_from_scalar<__half>::type symbol_t;
    template <class TLLRGroup>
    __device__
    static void symbol_to_LLR_group(TLLRGroup&      grp,            // output LLRs
                                    const symbol_t& sym,            // input symbol
                                    float           PAMnoiseVarInv) // inverse of PAM noise variance
    {
        __half2 result = __hmul2(__float2half2_rn(2 * QAM_traits<2>::A * PAMnoiseVarInv),
                                 sym);
        grp.fp8x2_e5m2[0] = __nv_fp8x2_e5m2(result);
    }
};

// complex float symbol, __nv_fp8_e4m3 LLR
template <>
struct LLR_QPSK<float, __nv_fp8_e4m3>
{
    //typedef LLR_group<__nv_fp8_e4m3, 2>             llr_group_t;
    typedef typename complex_from_scalar<float>::type symbol_t;
    template <class TLLRGroup>
    __device__
    static void symbol_to_LLR_group(TLLRGroup&      grp,            // output LLRs
                                    const symbol_t& sym,            // input symbol
                                    float           PAMnoiseVarInv) // inverse of PAM noise variance
    {
        float2 result = make_float2(PAMnoiseVarInv * 2 * QAM_traits<2>::A * sym.x,
                                    PAMnoiseVarInv * 2 * QAM_traits<2>::A * sym.y);
        grp.fp8x2_e4m3[0] = __nv_fp8x2_e4m3(result);
    }
};

// complex float symbol, __nv_fp8_e5m2 LLR
template <>
struct LLR_QPSK<float, __nv_fp8_e5m2>
{
    //typedef LLR_group<__nv_fp8_e5m2, 2>             llr_group_t;
    typedef typename complex_from_scalar<float>::type symbol_t;
    template <class TLLRGroup>
    __device__
    static void symbol_to_LLR_group(TLLRGroup&      grp,            // output LLRs
                                    const symbol_t& sym,            // input symbol
                                    float           PAMnoiseVarInv) // inverse of PAM noise variance
    {
        float2 result = make_float2(PAMnoiseVarInv * 2 * QAM_traits<2>::A * sym.x,
                                    PAMnoiseVarInv * 2 * QAM_traits<2>::A * sym.y);
        grp.fp8x2_e5m2[0] = __nv_fp8x2_e5m2(result);
    }
};

template <typename TSymbolScalar, typename TLLR> struct LLR_16QAM;

// complex float symbol, __half LLR
template <>
struct LLR_16QAM<float, __half>
{
    typedef typename complex_from_scalar<float>::type symbol_t;
    template <class TLLRGroup>
    __device__
    static void symbol_to_LLR_group(TLLRGroup&      grp,            // output LLRs
                                    const symbol_t& sym,            // input symbol
                                    float           PAMnoiseVarInv) // inverse of PAM noise variance
    {
        constexpr float A = 0.3162f;//1.0f/sqrt(10.0f);
        __half2 A2       = __float2half2_rn(2 * A);
        __half2 input    = __floats2half2_rn(sym.x, sym.y);
        __half2 noiseInv = __float2half2_rn(PAMnoiseVarInv);
        __half2 coeff    = __hmul2(noiseInv, A2);
        grp.f16x2[0]     = __hmul2(coeff, input);
        grp.f16x2[1]     = __hmul2(__hsub2(A2, __habs2(input)), coeff);
    }
};

template <typename TSymbolScalar, typename TLLR> struct LLR_64QAM;

// complex float symbol, __half LLR
template <>
struct LLR_64QAM<float, __half>
{
    typedef typename complex_from_scalar<float>::type symbol_t;
    template <class TLLRGroup>
    __device__
    static void symbol_to_LLR_group(TLLRGroup&      grp,            // output LLRs
                                    const symbol_t& sym,            // input symbol
                                    float           PAMnoiseVarInv) // inverse of PAM noise variance
    {
        constexpr float A = 0.1543f;//1.0f/sqrt(42.0f);
        __half2 A2       = __float2half2_rn(2 * A);
        __half2 A4       = __float2half2_rn(4 * A);
        __half2 input    = __floats2half2_rn(sym.x, sym.y);
        __half2 noiseInv = __float2half2_rn(PAMnoiseVarInv);
        __half2 coeff    = __hmul2(noiseInv, A2);
        __half2 temp     = __hsub2(A4, __habs2(input));
        grp.f16x2[0]     = __hmul2(coeff, input);
        grp.f16x2[1]     = __hmul2(temp, coeff);
        grp.f16x2[2]     = __hmul2(__hsub2(A2, __habs2(temp)), coeff);
     }
};

template <typename TSymbolScalar, typename TLLR> struct LLR_256QAM;

// complex float symbol, __half LLR
template <>
struct LLR_256QAM<float, __half>
{
    typedef typename complex_from_scalar<float>::type symbol_t;
    template <class TLLRGroup>
    __device__
    static void symbol_to_LLR_group(TLLRGroup&      grp,            // output LLRs
                                    const symbol_t& sym,            // input symbol
                                    float           PAMnoiseVarInv) // inverse of PAM noise variance
    {
        constexpr float A = 0.0766965f;//1.0f/sqrt(170.0f);
        __half2 A2       = __float2half2_rn(2 * A);
        __half2 A4       = __float2half2_rn(4 * A);
        __half2 A8       = __float2half2_rn(8 * A);
        __half2 input    = __floats2half2_rn(sym.x, sym.y);
        __half2 noiseInv = __float2half2_rn(PAMnoiseVarInv);
        __half2 coeff    = __hmul2(noiseInv, A2);
        __half2 temp     = __hsub2(A8, __habs2(input));
        __half2 temp1    = __hsub2(A4, __habs2(temp));
        grp.f16x2[0]     = __hmul2(coeff, input);
        grp.f16x2[1]     = __hmul2(temp, coeff);
        grp.f16x2[2]     = __hmul2(temp1, coeff);
        grp.f16x2[3]     = __hmul2(__hsub2(A2, __habs2(temp1)), coeff);
     }
};

////////////////////////////////////////////////////////////////////////
// soft_demapper
// Soft demapper structure for use when the demodulation (QAM) is fixed
// and known at compile time.
template <typename TSymbolScalar, typename TLLR, int QAM> struct soft_demapper
{
    typedef QAM_traits<QAM>                                   QAM_traits_t;
    //typedef LLR_group<TLLR, QAM_traits_t::bits>               llr_group_t;
    typedef typename complex_from_scalar<TSymbolScalar>::type symbol_t;
    typedef cuphy_i::tex_result_v4<TLLR>                      tex_result_t;
    template <class TLLRGroup>
    __device__
    static void symbol_to_LLR_group(TLLRGroup&          grp,            // output LLRs
                                    const symbol_t&     sym,            // input symbol
                                    float               PAMnoiseVarInv, // inverse of PAM noise variance
                                    cudaTextureObject_t texObj)         // texture map
    {
        // Texture-fetch implementation of the soft demapper function.
        // Can be used for all QAMs, but is probably less efficient for
        // BPSK and QPSK.
        tex_result_t res_I, res_Q;
        float2       t;
        t = symbol_to_tex_coords(sym, QAM_traits<QAM>::m, QAM_traits<QAM>::b);
        tex_1D_lod_ptx(res_I, texObj, t.x, QAM_traits<QAM>::LEVEL);
        tex_1D_lod_ptx(res_Q, texObj, t.y, QAM_traits<QAM>::LEVEL);
        apply_noise(res_I, PAMnoiseVarInv);
        apply_noise(res_Q, PAMnoiseVarInv);
        swizzle_LLRs(grp, res_I, res_Q);
    }
};

// Specialization for BPSK
template <typename TSymbolScalar, typename TLLR> struct soft_demapper<TSymbolScalar, TLLR, 2>
{
    typedef QAM_traits<2>                                     QAM_traits_t;
    //typedef LLR_group<TLLR, QAM_traits_t::bits>               llr_group_t;
    typedef typename complex_from_scalar<TSymbolScalar>::type symbol_t;
    typedef cuphy_i::tex_result_v4<TLLR>                      tex_result_t;
    template <class TLLRGroup>
    __device__
    static void symbol_to_LLR_group(TLLRGroup&          grp,            // output LLRs
                                    const symbol_t&     sym,            // input symbol
                                    float               PAMnoiseVarInv, // inverse of PAM noise variance
                                    cudaTextureObject_t texObj)         // texture map
    {
#if 0
        // There is probably no sense in actually doing the TEX lookup,
        // since this can be done with a couple of operations. Keep it
        // here for now, just to validate that the TEX approach is
        // correct.
        tex_result_t res_I, res_Q;
        tex_1D_lod_ptx(res_I,
                       texObj,
                       symbol_to_tex_coords(sum_components(sym),
                                            QAM_traits<QAM>::m,
                                            QAM_traits<QAM>::b),
                       QAM_traits<QAM>::LEVEL);
        apply_noise(res_I, PAMnoiseVarInv);
        swizzle_LLRs(grp, res_I, res_Q);
#else
        // Direct implementation of the soft demapper function.
        // Probably more efficient than the TEX approach for BPSK and
        // QPSK.
        LLR_BPSK<TSymbolScalar, TLLR>::symbol_to_LLR_group(grp,             // output LLRs
                                                           sym,             // input symbol
                                                           PAMnoiseVarInv); // inverse of PAM noise variance
#endif
    }
};

// Specialization for QPSK
template <typename TSymbolScalar, typename TLLR> struct soft_demapper<TSymbolScalar, TLLR, 4>
{
    typedef QAM_traits<4>                                     QAM_traits_t;
    //typedef LLR_group<TLLR, QAM_traits_t::bits>               llr_group_t;
    typedef typename complex_from_scalar<TSymbolScalar>::type symbol_t;
    typedef cuphy_i::tex_result_v4<TLLR>                      tex_result_t;
    template <class TLLRGroup>
    __device__
    static void symbol_to_LLR_group(TLLRGroup&          grp,            // output LLRs
                                    const symbol_t&     sym,            // input symbol
                                    float               PAMnoiseVarInv, // inverse of PAM noise variance
                                    cudaTextureObject_t texObj)         // texture map
    {
#if 0
        // There is probably no sense in actually doing the TEX lookup,
        // since this can be done with a couple of operations. Keep it
        // here for now, just to validate that the TEX approach is
        // correct.
        tex_result_t res_I, res_Q;
        tex_1D_lod_ptx(res_I,
                       texObj,
                       symbol_to_tex_coords(sum_components(sym),
                                            QAM_traits<QAM>::m,
                                            QAM_traits<QAM>::b),
                       QAM_traits<QAM>::LEVEL);
        apply_noise(res_I, PAMnoiseVarInv);
        swizzle_LLRs(grp, res_I, res_Q);
#else
        // Direct implementation of the soft demapper function.
        // Probably more efficient than the TEX approach for BPSK and
        // QPSK.
        LLR_QPSK<TSymbolScalar, TLLR>::symbol_to_LLR_group(grp,             // output LLRs
                                                           sym,             // input symbol
                                                           PAMnoiseVarInv); // inverse of PAM noise variance
#endif
    }
};

////////////////////////////////////////////////////////////////////////
// soft_demapper_any
// Soft demapper structure for use when the demodulation (QAM) is NOT
// known at compile time.
template <typename TSymbolScalar, typename TLLR> struct soft_demapper_any
{
    using symbol_t     = typename complex_from_scalar<TSymbolScalar>::type;
    using tex_result_t = typename demapper_llr_traits_t<TLLR>::tex_result_t;
    template <class TLLRGroup>
    __device__
    static void symbol_to_LLR_group(TLLRGroup&          grp,            // output LLRs
                                    const symbol_t&     sym,            // input symbol
                                    float               PAMnoiseVarInv, // inverse of PAM noise variance
                                    int                 nBits,          // num QAM bits
                                    cudaTextureObject_t texObj)         // texture map
    {
        // Assuming below that BPSK and QPSK are more efficient using
        // direct calculation than the texture lookup approach.
        if(1 == nBits)
        {
            LLR_BPSK<TSymbolScalar, TLLR>::symbol_to_LLR_group(grp, sym, PAMnoiseVarInv);
        }
        else if(2 == nBits)
        {
            LLR_QPSK<TSymbolScalar, TLLR>::symbol_to_LLR_group(grp, sym, PAMnoiseVarInv);
        }
        else
        {
            // QAM16, QAM64, and QAM256
            tex_result_t res_I, res_Q;
            float2       t;
            t = symbol_to_tex_coords(sym, nBits);
            tex_1D_lod_ptx(res_I, texObj, t.x, mod_mipmap_level[nBits]);
            tex_1D_lod_ptx(res_Q, texObj, t.y, mod_mipmap_level[nBits]);
            apply_noise(res_I, PAMnoiseVarInv);
            apply_noise(res_Q, PAMnoiseVarInv);
            swizzle_LLRs(grp, res_I, res_Q);
        }
    }
};

//////////////////////////////////////////////////////////////////////////
template <typename TSymbolScalar, typename TLLR> struct soft_demapper_simplified
{
    typedef typename complex_from_scalar<TSymbolScalar>::type symbol_t;
    template <class TLLRGroup>
    __device__
    static void symbol_to_LLR_group(TLLRGroup&          grp,            // output LLRs
                                    const symbol_t&     sym,            // input symbol
                                    float               PAMnoiseVarInv, // inverse of PAM noise variance
                                    int                 nBits)          // num QAM bits
    {
        if(1 == nBits)
        {
            LLR_BPSK<TSymbolScalar, TLLR>::symbol_to_LLR_group(grp, sym, PAMnoiseVarInv);
        }
        else if(2 == nBits)
        {
            LLR_QPSK<TSymbolScalar, TLLR>::symbol_to_LLR_group(grp, sym, PAMnoiseVarInv);
        }
        else if(4 == nBits)
        {
            LLR_16QAM<TSymbolScalar, TLLR>::symbol_to_LLR_group(grp, sym, PAMnoiseVarInv);
        }
        else if(6 == nBits)
        {
            LLR_64QAM<TSymbolScalar, TLLR>::symbol_to_LLR_group(grp, sym, PAMnoiseVarInv);
        }
        else if(8 == nBits)
        {
            LLR_256QAM<TSymbolScalar, TLLR>::symbol_to_LLR_group(grp, sym, PAMnoiseVarInv);
        }
    }
};

} // namespace soft_demapper

#endif // !defined(CUPHY_SOFT_DEMAPPER_CUH_INCLUDED_)
