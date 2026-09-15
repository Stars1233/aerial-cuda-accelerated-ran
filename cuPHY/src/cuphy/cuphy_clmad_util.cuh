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

#if !defined(CUPHY_CLMAD_UTIL_CUH_INCLUDED_)
#define CUPHY_CLMAD_UTIL_CUH_INCLUDED_

#include <cstdint>

#ifndef CUPHY_CLMAD_BUILD_ENABLED
#define CUPHY_CLMAD_BUILD_ENABLED 1
#endif

// CUPHY_CLMAD_AVAILABLE is a compile-time capability check: the CLMAD
// (carry-less multiply-add) paths are used automatically whenever the device
// architecture and CUDA toolkit support the `clmad` PTX instruction, otherwise
// callers fall back to the scalar/LUT implementations. Requires SM 8.0+ and the
// public compiler support introduced in CUDA 13.3. The CUPHY_ENABLE_CLMAD build
// option can disable these paths even when the compiler and architecture support them.
#if CUPHY_CLMAD_BUILD_ENABLED && defined(__CUDACC__) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800) && \
    defined(__CUDACC_VER_MAJOR__) && defined(__CUDACC_VER_MINOR__) && \
    ((__CUDACC_VER_MAJOR__ > 13) || ((__CUDACC_VER_MAJOR__ == 13) && (__CUDACC_VER_MINOR__ >= 3)))
#define CUPHY_CLMAD_AVAILABLE 1
#else
#define CUPHY_CLMAD_AVAILABLE 0
#endif

#if CUPHY_CLMAD_AVAILABLE
namespace cuphy_clmad
{

union u64_u32x2 {
    uint64_t u64;
    uint32_t u32[2];
};

__device__ __forceinline__ uint64_t clmul_lo(uint64_t a, uint64_t b)
{
    uint64_t clm;
    asm("clmad.lo.u64 %0, %1, %2, 0x0000000000000000;"
        : "=l"(clm)
        : "l"(a), "l"(b));
    return clm;
}

// Barrett-reduction constants for CLMAD-based polynomial reduction.
//
// For a polynomial p of degree T (leading x^T bit set):
//   gstar    = p mod x^T            = p with the leading x^T bit cleared
//   qplusX   = floor(x^(2T)   / p)  (degree T,  used for the S=T  postadjust)
//   qplusCRC = floor(x^(32+T) / p)  (degree 32, used for the S=32 preadjust)
// The gstar values follow directly from the generator polynomial by masking off
// the leading bit; the qplus values are outputs of GF(2)[x] long-division of
// x^k by p and cannot be expressed in closed form, so they are precomputed.
//
// CRC generator polynomials (3GPP TS 38.212 §5.1):
//   G_CRC_16   (T=16): 0x011021  = x^16 + x^12 + x^5 + 1
//   G_CRC_24_A (T=24): 0x1864CFB = x^24 + x^23 + x^18 + x^17 + x^14 + x^11
//                                  + x^10 + x^7 + x^6 + x^5 + x^4 + x^3 + x + 1
//   G_CRC_24_B (T=24): 0x1800063 = x^24 + x^23 + x^6 + x^5 + x + 1
constexpr uint32_t gcrc16astrX   = 0x00001021u;            // G_CRC_16   mod x^16
constexpr uint32_t gcrc24aastrX  = 0x00864CFBu;            // G_CRC_24_A mod x^24
constexpr uint32_t gcrc24bastrX  = 0x00800063u;            // G_CRC_24_B mod x^24

constexpr uint32_t qplusX_16     = 0x00011130u;            // floor(x^32 / G_CRC_16)
constexpr uint32_t qplusX_24_A   = 0x01F845FEu;            // floor(x^48 / G_CRC_24_A)
constexpr uint32_t qplusX_24_B   = 0x01FFFF83u;            // floor(x^48 / G_CRC_24_B)

constexpr uint64_t qplusCRC_16   = 0x0000000111303471ull;  // floor(x^48 / G_CRC_16)
constexpr uint64_t qplusCRC_24_A = 0x00000001F845FE24ull;  // floor(x^56 / G_CRC_24_A)
constexpr uint64_t qplusCRC_24_B = 0x00000001FFFF83FFull;  // floor(x^56 / G_CRC_24_B)

// Gold-31 sequence polynomial used by descrambling (mulModPoly31LUT, POLY_2):
//   POLY_2 (T=31): 0x8000000F = x^31 + x^3 + x^2 + x + 1
constexpr uint64_t qplusX_gold31 = 0x000000008000000Full;  // floor(x^62 / POLY_2)
constexpr uint32_t gastrX_gold31 = 0x0000000Fu;            // POLY_2 mod x^31 (== POLY_2_GMASK)

// Computes c * x^T mod p (low T bits), where QPLUS = floor(x^(S+T) / p)
// and GSTAR = p mod x^T (= p with leading x^T bit cleared).
// Used both to bake the x^T factor into the per-word LUT (during
// early_term_initialize) and to fold the high half of clmul products back
// into the low T bits (during compute_crc_clmad).
template <uint32_t S, uint32_t T, uint64_t QPLUS, uint32_t GSTAR>
__device__ __forceinline__ uint32_t opt_reduction(uint32_t c)
{
    u64_u32x2 step1;
    step1.u64 = clmul_lo(static_cast<uint64_t>(c), QPLUS);

    u64_u32x2 step2;
    step2.u32[0] = __funnelshift_rc(step1.u32[0], step1.u32[1], S);
    step2.u32[1] = 0;

    u64_u32x2 step3;
    constexpr uint32_t lsbMask = (1u << T) - 1u;
    step3.u64 = clmul_lo(step2.u64, static_cast<uint64_t>(GSTAR));

    return step3.u32[0] & lsbMask;
}

} // namespace cuphy_clmad
#endif

#endif // !defined(CUPHY_CLMAD_UTIL_CUH_INCLUDED_)
