/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "cuphy.h"

namespace 
{
  ////////////////////////////////////////////////////////////////////////
  // complex_mul()
  template <typename T>
__forceinline__ __device__ T complex_mul(const T& a, const T& b)
  {
      // Use explicit cast to avoid warnings for narrowing. We assume the
      // caller knows the expected range is "safe".
      typedef typename scalar_from_complex<T>::type scalar_t;
      return T{static_cast<scalar_t>((a.x * b.x) - (a.y * b.y)),
              static_cast<scalar_t>((a.x * b.y) + (a.y * b.x))};
  }
  template <>
  __forceinline__ __device__ __half2 complex_mul(const __half2& a, const __half2& b)
  {
    __half2 c{0.0, 0.0};
    return __hcmadd(a, b, c);
  }

// complex_addmul() //ToDo: may consider switching to cuCma()
  __device__ __inline__  float2 complex_addmul( float2 x, float2 y, float2 acc)
  {
      float real_res;
      float imag_res;

      real_res = (cuCrealf(x) *  cuCrealf(y)) + cuCrealf(acc);
      imag_res = (cuCrealf(x) *  cuCimagf(y)) + cuCimagf(acc);

      real_res = -(cuCimagf(x) * cuCimagf(y))  + real_res;
      imag_res =  (cuCimagf(x) *  cuCrealf(y)) + imag_res;

      return make_cuComplex(real_res, imag_res);
  }

  __device__ __inline__  __half2 complex_addmul( __half2 x, __half2 y, __half2 acc)
  {
      return __hcmadd(x,y,acc);
  }

  __device__ float2 complex_scalar_multiply(float2 vector_val, float scalar_val)
  {
      return make_float2(vector_val.x * scalar_val, vector_val.y * scalar_val);
  }

  __device__ __half2 complex_scalar_multiply(__half2 vector_val, __half scalar_val)
  {
      return __hmul2(vector_val, make_half2(scalar_val, scalar_val));
  }


  ////////////////////////////////////////////////////////////////////////
  // real_mul() multiplies complex value with a real value
  template <typename T1, typename T2>
__forceinline__ __device__ T1 real_mul(const T1& a, const T2& b)
  {
      // Use explicit cast to avoid warnings for narrowing. We assume the
      // caller knows the expected range is "safe".
      using scalar_t = typename scalar_from_complex<T1>::type;
      return T1{static_cast<scalar_t>(a.x * b),
              static_cast<scalar_t>(a.y * b)};
  }
  template <>
  __forceinline__ __device__ __half2 real_mul(const __half2& a, const __half& b)
  {
      return __halves2half2(a.x*b,a.y*b);
  }

  ////////////////////////////////////////////////////////////////////////

  __device__ __forceinline__  __half2 conj_fast(__half2 z)
  {
      union { __half2 h; uint32_t u; } v;
      v.h  = z;                  // bit‑cast half2 → 32‑bit register
      v.u ^= 0x80000000u;        // toggle sign bit of the *upper* 16 bits
      return v.h;                // bit‑cast back
  }

  ////////////////////////////////////////////////////////////////////////
  // complex_conjmul()
  template <typename T>
  __forceinline__ __device__ T complex_conjmul(const T& a, const T& b)
  {
      // Use explicit cast to avoid warnings for narrowing. We assume the
      // caller knows the expected range is "safe".
      typedef typename scalar_from_complex<T>::type scalar_t;
      return T{static_cast<scalar_t>((a.x * b.x) + (a.y * b.y)),
               static_cast<scalar_t>((a.y * b.x) - (a.x * b.y))};
  }
  template <>
  __forceinline__ __device__ __half2 complex_conjmul(const __half2& a, const __half2& b)
  {
      // b’ = conj(b) in one XOR
      __half2 b_conj = conj_fast(b);
      // c = a * b’
      const __half2 zero{0.f, 0.f};
      return __hcmadd(a, b_conj, zero);
  }

  ////////////////////////////////////////////////////////////////////////
  // complex_add()
  template <typename T>
  __device__ T complex_add(const T& a, const T& b)
  {
      // Use explicit cast to avoid warnings for narrowing. We assume the
      // caller knows the expected range is "safe".
      typedef typename scalar_from_complex<T>::type scalar_t;

      return T{static_cast<scalar_t>(a.x + b.x), static_cast<scalar_t>(a.y + b.y)};
  }

  template <>
  __device__ __half2 complex_add(const __half2& a, const __half2& b)
  {
      return __hadd2(a, b);
  }

  ////////////////////////////////////////////////////////////////////////
  // calculate abs(a)^2
  template <typename T>
  __forceinline__  __device__ float absSqure(const T& a)
  {
    return static_cast<float>(a.x) * static_cast<float>(a.x) + static_cast<float>(a.y) * static_cast<float>(a.y);
  }

  template <>
  __forceinline__  __device__ float absSqure(const __half2 & a)
  {
    return static_cast<float>(a.x) * static_cast<float>(a.x) + static_cast<float>(a.y) * static_cast<float>(a.y);
  }

  ////////////////////////////////////////////////////////////////////////
  //least common multiple and greatest common divisor
  template <typename T>
  constexpr __host__ __device__ uint32_t compute_gcd(T m, T n)
  {
      static_assert(std::is_unsigned_v<T>, "only unsigned values supported");
      if (n == 0) return m;
      return compute_gcd(n, m % n);
  }

  template <typename T>
  constexpr __host__ __device__ uint32_t compute_lcm(T m, T n)
  {
      static_assert(std::is_unsigned_v<T>, "only unsigned values supported");
      //return std::lcm(m, n); //nvcc shipped with CUDA 12.2 gives compiler error due to not handling constexpr input properly
      return (m * n) / compute_gcd(m, n);
  }

};
