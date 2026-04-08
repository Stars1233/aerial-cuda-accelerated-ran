/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

/**
 * Common CUDA complex number operations and utilities.
 *
 * This header provides:
 * - Complex number arithmetic operations for cuComplex and __half2
 * - Type conversion templates (cuGet, cuAbs, cuRSqrt)
 * - Tensor reference and block storage templates
 * - Common constants (N_THREADS_PER_WARP)
 *
 * All symbols are in the cuphy_cmplx namespace. Use one of:
 * - using namespace cuphy_cmplx;
 * - using cuphy_cmplx::symbol_name;
 * - Namespace alias: namespace cc = cuphy_cmplx;
 *
 * Note: The __half2 operators perform COMPLEX multiplication (not element-wise),
 * so they must be in a namespace to avoid conflicts with CUDA's built-in operators.
 */

#include "cuComplex.h"
#include "cuda_fp16.h"
#include "cuphy.h"
#include "math_utils.cuh"

namespace cuphy_cmplx {

//=============================================================================
// Constants
//=============================================================================

static constexpr uint32_t N_THREADS_PER_WARP = 32; //!< cudaDeviceProp::warpSize

//=============================================================================
// Utility Functions
//=============================================================================

/**
 * Divide and round up to nearest integer (constexpr version).
 * @param[in] val Value to divide
 * @param[in] divide_by Divisor
 * @return Ceiling of val/divide_by
 */
template <typename T>
CUDA_BOTH_INLINE constexpr T div_round_up_cexp(T val, T divide_by)
{
    return ((val + (divide_by - 1)) / divide_by);
}

//=============================================================================
// Type Conversion Templates
//=============================================================================

// clang-format off

/// Convert integer to specified type
template <typename T> CUDA_BOTH_INLINE T cuGet(int);
template<>            CUDA_BOTH_INLINE float     cuGet(int x) { return float(x); }
template<>            CUDA_BOTH_INLINE __half    cuGet(int x) { return __half(x); }
template<>            CUDA_BOTH_INLINE cuComplex cuGet(int x) { return make_cuComplex(float(x), 0.0f); }
template<>            CUDA_BOTH_INLINE __half2   cuGet(int x) { return make_half2(__half(x), 0.0f); }

/// Convert float to specified type
template <typename T> CUDA_BOTH_INLINE T cuGet(float);
template<>            CUDA_BOTH_INLINE float     cuGet(float x) { return float(x); }
template<>            CUDA_BOTH_INLINE cuComplex cuGet(float x) { return make_cuComplex(x, 0.0f); }
template<>            CUDA_BOTH_INLINE __half2   cuGet(float x) { return make_half2(x, 0.0f); }

/// Convert __half to specified type
template <typename T> CUDA_BOTH_INLINE T cuGet(__half);
template<>            CUDA_BOTH_INLINE __half  cuGet(__half x) { return x; }
template<>            CUDA_BOTH_INLINE __half2 cuGet(__half x) { return make_half2(x, 0.0f); }

/// Absolute value
template <typename T> CUDA_BOTH_INLINE T cuAbs(T);
template<>            CUDA_BOTH_INLINE float cuAbs(float x) { return fabsf(x); }

/// Reciprocal square root
template <typename T> CUDA_INLINE T cuRSqrt(T);
template<>            CUDA_INLINE float cuRSqrt(float x) { return rsqrtf(x); }
template<>            CUDA_INLINE half  cuRSqrt(half x)  { return hrsqrt(x); }

// clang-format on

//=============================================================================
// Complex Number Operations - cuComplex (float2)
//=============================================================================

/// Get real part of complex number
static CUDA_BOTH_INLINE float cuReal(cuComplex x) { return cuCrealf(x); }

/// Get imaginary part of complex number
static CUDA_BOTH_INLINE float cuImag(cuComplex x) { return cuCimagf(x); }

/// Complex conjugate
static CUDA_BOTH_INLINE cuComplex cuConj(cuComplex x) { return cuConjf(x); }

/// Complex multiply-add: (x*y) + a
static CUDA_BOTH_INLINE cuComplex cuCma(cuComplex x, cuComplex y, cuComplex a) { return cuCfmaf(x, y, a); }

/// Complex multiplication
static CUDA_BOTH_INLINE cuComplex cuCmul(cuComplex x, cuComplex y) { return cuCmulf(x, y); }

// Operators for cuComplex
static CUDA_BOTH_INLINE cuComplex operator*(cuComplex x, float y)     { return make_cuComplex(cuCrealf(x) * y, cuCimagf(x) * y); }
static CUDA_BOTH_INLINE cuComplex operator*(float x, cuComplex y)     { return make_cuComplex(cuCrealf(y) * x, cuCimagf(y) * x); }
static CUDA_BOTH_INLINE cuComplex operator+(cuComplex x, cuComplex y) { return cuCaddf(x, y); }
static CUDA_BOTH_INLINE cuComplex operator-(cuComplex x, cuComplex y) { return cuCsubf(x, y); }

static CUDA_BOTH_INLINE cuComplex& operator+=(cuComplex& x, float y)     { x = make_cuComplex(cuCrealf(x) + y, cuCimagf(x)); return x; }
static CUDA_BOTH_INLINE cuComplex& operator*=(cuComplex& x, float y)     { x = make_cuComplex(cuCrealf(x) * y, cuCimagf(x) * y); return x; }
static CUDA_BOTH_INLINE cuComplex& operator+=(cuComplex& x, cuComplex y) { x = cuCaddf(x, y); return x; }

//=============================================================================
// Complex Number Operations - __half2 (half-precision complex)
// Note: These operators perform COMPLEX multiplication, not element-wise!
//=============================================================================

/// Get real part of half-precision complex number
static CUDA_INLINE __half cuReal(__half2 x) { return x.x; }

/// Get imaginary part of half-precision complex number
static CUDA_INLINE __half cuImag(__half2 x) { return x.y; }

/// Half-precision complex conjugate (uses conj_fast from math_utils.cuh)
static CUDA_INLINE __half2 cuConj(__half2 x) { return conj_fast(x); }

/// Half-precision complex multiply-add: (x*y) + a
static CUDA_INLINE __half2 cuCma(__half2 x, __half2 y, __half2 a) { return __hcmadd(x, y, a); }

/// Half-precision complex multiplication
static CUDA_INLINE __half2 cuCmul(__half2 x, __half2 y) { return __hcmadd(x, y, __float2half2_rn(0.f)); }

// Operators for __half2 (element-wise)
static CUDA_INLINE __half2 operator*(__half2 x, __half y)  { return __hmul2(x, make_half2(y, y)); }
static CUDA_INLINE __half2 operator*(__half x, __half2 y)  { return __hmul2(y, make_half2(x, x)); }

static CUDA_INLINE __half2& operator+=(__half2& x, __half y) { x = make_half2(cuReal(x) + y, cuImag(x)); return x; }
static CUDA_INLINE __half2& operator*=(__half2& x, __half y) { x = __hmul2(x, make_half2(y, y)); return x; }

//=============================================================================
// Tensor Reference Template
//=============================================================================

/**
 * Lightweight tensor reference for GPU kernel access.
 * Provides multi-dimensional indexing using pre-computed strides.
 * @tparam TElem Element type of the tensor
 */
template <typename TElem>
struct tensor_ref
{
    TElem*         addr{};     //!< Pointer to tensor data
    const int32_t* strides{};  //!< Pointer to stride array

    CUDA_BOTH
    tensor_ref(void* pAddr, const int32_t* pStrides) :
        addr(static_cast<TElem*>(pAddr)),
        strides(pStrides)
    {
    }

    CUDA_BOTH int offset(int i0) const
    {
        return (strides[0] * i0);
    }

    CUDA_BOTH int offset(int i0, int i1) const
    {
        return (strides[0] * i0) + (strides[1] * i1);
    }

    CUDA_BOTH int offset(int i0, int i1, int i2) const
    {
        return (strides[0] * i0) + (strides[1] * i1) + (strides[2] * i2);
    }

    CUDA_BOTH int offset(int i0, int i1, int i2, int i3) const
    {
        return (strides[0] * i0) + (strides[1] * i1) + (strides[2] * i2) + (strides[3] * i3);
    }

    CUDA_BOTH int offset(int i0, int i1, int i2, int i3, int i4) const
    {
        return (strides[0] * i0) + (strides[1] * i1) + (strides[2] * i2) + (strides[3] * i3) + (strides[4] * i4);
    }

    // clang-format off
    CUDA_BOTH TElem&       operator()(int i0)                                 { return *(addr + offset(i0));                 }
    CUDA_BOTH TElem&       operator()(int i0, int i1)                         { return *(addr + offset(i0, i1));             }
    CUDA_BOTH TElem&       operator()(int i0, int i1, int i2)                 { return *(addr + offset(i0, i1, i2));         }
    CUDA_BOTH TElem&       operator()(int i0, int i1, int i2, int i3)         { return *(addr + offset(i0, i1, i2, i3));     }
    CUDA_BOTH TElem&       operator()(int i0, int i1, int i2, int i3, int i4) { return *(addr + offset(i0, i1, i2, i3, i4)); }

    CUDA_BOTH const TElem& operator()(int i0) const                                 { return *(addr + offset(i0));                 }
    CUDA_BOTH const TElem& operator()(int i0, int i1) const                         { return *(addr + offset(i0, i1));             }
    CUDA_BOTH const TElem& operator()(int i0, int i1, int i2) const                 { return *(addr + offset(i0, i1, i2));         }
    CUDA_BOTH const TElem& operator()(int i0, int i1, int i2, int i3) const         { return *(addr + offset(i0, i1, i2, i3));     }
    CUDA_BOTH const TElem& operator()(int i0, int i1, int i2, int i3, int i4) const { return *(addr + offset(i0, i1, i2, i3, i4)); }
    // clang-format on
};

//=============================================================================
// Block Storage Templates
//=============================================================================

/**
 * 1D block storage for register or shared memory.
 * @tparam T Element type
 * @tparam M Number of elements
 */
template <typename T, int M>
struct block_1D
{
    T data[M]{};
    CUDA_BOTH T& operator()(int idx) { return data[idx]; }
    CUDA_BOTH const T& operator()(int idx) const { return data[idx]; }
};

/**
 * 2D block storage (column-major).
 * @tparam T Element type
 * @tparam M Number of rows
 * @tparam N Number of columns
 */
template <typename T, int M, int N>
struct block_2D
{
    T data[M * N]{};
    CUDA_BOTH T& operator()(int m, int n) { return data[(n * M) + m]; }
    CUDA_BOTH const T& operator()(int m, int n) const { return data[(n * M) + m]; }
};

/**
 * 3D block storage (column-major).
 * @tparam T Element type
 * @tparam L First dimension
 * @tparam M Second dimension
 * @tparam N Third dimension
 */
template <typename T, int L, int M, int N>
struct block_3D
{
    T data[L * M * N]{};
    CUDA_BOTH T& operator()(int l, int m, int n) { return data[((n * M) + m) * L + l]; }
    CUDA_BOTH const T& operator()(int l, int m, int n) const { return data[((n * M) + m) * L + l]; }
};

//=============================================================================
// Block Storage Partial Specializations for Shared Memory Pointers
//=============================================================================

/**
 * 1D block storage using shared memory pointer.
 * @tparam T Element type (pointer specialization)
 * @tparam M Number of elements
 */
template <typename T, int M>
struct block_1D<T*, M>
{
    CUDA_BOTH block_1D(T* pData) :
        m_pData(pData) {}
    block_1D()                    = delete;
    block_1D(block_1D const& blk) = delete;
    CUDA_BOTH block_1D& operator=(block_1D const& block) { m_pData = block.m_pData; return *this; }
    ~block_1D()                   = default;

    CUDA_BOTH T& operator()(int idx) { return m_pData[idx]; }
    CUDA_BOTH const T& operator()(int idx) const { return m_pData[idx]; }
    static constexpr CUDA_BOTH size_t num_elem() { return M; }

private:
    T* m_pData{};
};

/**
 * 2D block storage using shared memory pointer (column-major).
 * @tparam T Element type (pointer specialization)
 * @tparam M Number of rows
 * @tparam N Number of columns
 */
template <typename T, int M, int N>
struct block_2D<T*, M, N>
{
    CUDA_BOTH block_2D(T* pData) :
        m_pData(pData) {}
    block_2D()                    = delete;
    block_2D(block_2D const& blk) = delete;
    CUDA_BOTH block_2D& operator=(block_2D const& block) { m_pData = block.m_pData; return *this; }
    ~block_2D()                   = default;

    CUDA_BOTH T& operator()(int m, int n) { return m_pData[(n * M) + m]; }
    CUDA_BOTH const T& operator()(int m, int n) const { return m_pData[(n * M) + m]; }
    static constexpr CUDA_BOTH size_t num_elem() { return M * N; }

private:
    T* m_pData{};
};

/**
 * 3D block storage using shared memory pointer (column-major).
 * @tparam T Element type (pointer specialization)
 * @tparam L First dimension
 * @tparam M Second dimension
 * @tparam N Third dimension
 */
template <typename T, int L, int M, int N>
struct block_3D<T*, L, M, N>
{
    CUDA_BOTH block_3D(T* pData) :
        m_pData(pData) {}
    block_3D()                    = delete;
    block_3D(block_3D const& blk) = delete;
    CUDA_BOTH block_3D& operator=(block_3D const& block) { m_pData = block.m_pData; return *this; }
    ~block_3D()                   = default;

    CUDA_BOTH T& operator()(int l, int m, int n) { return m_pData[((n * M) + m) * L + l]; }
    CUDA_BOTH const T& operator()(int l, int m, int n) const { return m_pData[((n * M) + m) * L + l]; }
    static constexpr CUDA_BOTH size_t num_elem() { return L * M * N; }

private:
    T* m_pData{};
};

} // namespace cuphy_cmplx
