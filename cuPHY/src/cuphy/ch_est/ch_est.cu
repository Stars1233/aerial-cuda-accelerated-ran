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

 #include <algorithm>
 #include <functional>
 #include <vector>
 #include <sstream>
 #include "cuComplex.h"
 #include <cooperative_groups.h>
 #include <cooperative_groups/reduce.h>
 #include "descrambling.cuh"
 #include "ch_est.hpp"
 #include "type_convert.hpp"
 #include "utils.cuh"
 #include "nvlog.hpp"
 #include "cuphy.hpp"
 #include "constants.hpp"
 #include "math_utils.cuh"
 #include "cuda_fp16.h"
 #include "mma.h"

#include "pusch_start_kernels.hpp"


// RKHS constants:
 #define RKHS_MAX_N_GNB_ANTS 4
 #define RKHS_MAX_DMRS_SC_PER_THREAD 2
 #define RKHS_MAX_SC_PER_THREAD 4
 #define RKHS_MAX_N_SYM 2
 #define RKHS_N_DMRS_GRID_TONES_PER_PRB 6
 #define RKHS_N_EIGS 3
 #define RKHS_N_ZP_EIGS 6
 #define RKHS_MAX_LAYERS 4
 #define RKHS_MAX_N_INTS 50
 #define RKHS_USE_HAMMING
//#define DO_NOT_USE_HASH_TABLE

 using namespace cooperative_groups;
 using namespace descrambling;
 namespace cg = cooperative_groups;



 namespace ch_est
 {
 // #define ENABLE_PROFILING
 // #define ENABLE_DEBUG

 // #define ENABLE_PRIME_WHILE_LOOP
 #define ENABLE_COMMON_DFTSOFDM_DESCRCODE_SUBROUTINE // if ENABLE_COMMON_DFTSOFDM_DESCRCODE_SUBROUTINE is defined, ENABLE_PRIME_WHILE_LOOP should not be defined.

 // PRB size in tones
 static constexpr uint32_t N_TONES_PER_PRB    = 12;
 static constexpr uint32_t N_TOCC   = 2;

 // Total # of symbols after tOCC and fOCC removal
 static constexpr uint32_t N_DMRS_SYMS_OCC = 4; // reserve memory for the worst-case

 template <typename TElem>
 struct tensor_ref
 {
     TElem*     pAddr;
     const int* strides;

     CUDA_BOTH
     tensor_ref(void* pAddr, const int* pStrides) :
         pAddr(static_cast<TElem*>(pAddr)),
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
     };
     CUDA_BOTH int offset(int i0, int i1, int i2, int i3) const
     {
         return (strides[0] * i0) + (strides[1] * i1) + (strides[2] * i2) + (strides[3] * i3);
     };
     CUDA_BOTH TElem* trace(int i1)
     {
        return pAddr + offset(0, i1);
     };

     // clang-format off
     CUDA_BOTH TElem&       operator()(int i0)                               { return *(pAddr + offset(i0));             }
     CUDA_BOTH TElem&       operator()(int i0, int i1)                       { return *(pAddr + offset(i0, i1));         }
     CUDA_BOTH TElem&       operator()(int i0, int i1, int i2)               { return *(pAddr + offset(i0, i1, i2));     }
     CUDA_BOTH TElem&       operator()(int i0, int i1, int i2, int i3)       { return *(pAddr + offset(i0, i1, i2, i3)); }

     CUDA_BOTH const TElem& operator()(int i0) const                         { return *(pAddr + offset(i0));             }
     CUDA_BOTH const TElem& operator()(int i0, int i1) const                 { return *(pAddr + offset(i0, i1));         }
     CUDA_BOTH const TElem& operator()(int i0, int i1, int i2) const         { return *(pAddr + offset(i0, i1, i2));     }
     CUDA_BOTH const TElem& operator()(int i0, int i1, int i2, int i3) const { return *(pAddr + offset(i0, i1, i2, i3)); }
     // clang-format on
 };

 template <typename T, int M>
 struct block_1D
 {
     T         data[M];
     CUDA_BOTH T& operator[](int idx) { return data[idx]; }
 };

 template <typename T, int M, int N>
 struct block_2D
 {
     T         data[M * N];
     CUDA_BOTH T& operator()(int m, int n) { return data[(n * M) + m]; }
 };

 template <typename T, int L, int M, int N>
 struct block_3D
 {
     T         data[L * M * N];
     CUDA_BOTH T& operator()(int l, int m, int n) { return data[((n * M) + m) * L + l]; }
 };

 // Partial specialization of block_1D to use shared memory pointers
 template <typename T, int M>
 struct block_1D<T*, M>
 {
     CUDA_BOTH block_1D(T* pData) :
         m_pData(pData){}; // static_assert(std::is_pointer<T>::value, "Must be a pointer type")
     block_1D()                    = delete;
     block_1D(block_1D const& blk) = delete;
     CUDA_BOTH block_1D& operator  =(block_1D const& block) { m_pData = block.m_pData; };
     ~block_1D()                   = default;

     CUDA_BOTH T&               operator[](int idx) { return m_pData[idx]; }
     static constexpr CUDA_BOTH size_t num_elem() { return M; }

 private:
     T* m_pData = nullptr;
 };

 // Partial specialization of block_2D to use shared memory pointers
 template <typename T, int M, int N>
 struct block_2D<T*, M, N>
 {
     CUDA_BOTH block_2D(T* pData) :
         m_pData(pData){};
     block_2D()                    = delete;
     block_2D(block_2D const& blk) = delete;
     CUDA_BOTH block_2D& operator  =(block_2D const& block) { m_pData = block.m_pData; };
     ~block_2D()                   = default;

     CUDA_BOTH T&               operator()(int m, int n) { return m_pData[(n * M) + m]; }
     static constexpr CUDA_BOTH size_t num_elem() { return M * N; }

 private:
     T* m_pData = nullptr;
 };

 // Partial specialization of block_3D to use shared memory pointers
 template <typename T, int L, int M, int N>
 struct block_3D<T*, L, M, N>
 {
     CUDA_BOTH block_3D(T* pData) :
         m_pData(pData){};
     block_3D()                    = delete;
     block_3D(block_3D const& blk) = delete;
     CUDA_BOTH block_3D& operator  =(block_3D const& block) { m_pData = block.m_pData; };
     ~block_3D()                   = default;

     CUDA_BOTH T&               operator()(int l, int m, int n) { return m_pData[((n * M) + m) * L + l]; }
     static constexpr CUDA_BOTH size_t num_elem() { return L * M * N; }

 private:
     T* m_pData = nullptr;
 };

 // clang-format off
 template <typename T> CUDA_BOTH_INLINE constexpr T     cuGet(uint32_t);
 template<>            CUDA_BOTH_INLINE constexpr float cuGet(uint32_t x) { return(float(x)); }

 template <typename T> CUDA_BOTH_INLINE T         cuGet(int);
 template<>            CUDA_BOTH_INLINE float     cuGet(int x) { return(float(x)); }
 template<>            CUDA_BOTH_INLINE cuComplex cuGet(int x) { return(make_cuComplex(float(x), 0.0f)); }

 template <typename T> CUDA_BOTH_INLINE T         cuGet(float);
 template<>            CUDA_BOTH_INLINE cuComplex cuGet(float x) { return(make_cuComplex(x, 0.0f)); }

 template <typename T> CUDA_BOTH_INLINE T         cuGet(float,float);
 template <>           CUDA_BOTH_INLINE cuComplex cuGet<cuComplex>(float x, float y) { return make_cuComplex(x,y); }

 static CUDA_BOTH_INLINE cuComplex operator+=(cuComplex &x, cuComplex y)       { x = cuCaddf(x, y); return x; };
 static CUDA_BOTH_INLINE cuComplex operator*(cuComplex x, int y)               { return(make_cuComplex(cuCrealf(x)*float(y), cuCimagf(x)*float(y))); }
 static CUDA_BOTH_INLINE cuComplex operator*(cuComplex x, float y)             { return(make_cuComplex(cuCrealf(x)*y, cuCimagf(x)*y)); }
 static CUDA_BOTH_INLINE cuComplex operator*=(cuComplex &x, const cuComplex y) { x = cuCmulf(x, y); return x; };

 // #ifdef ENABLE_DEBUG
 static CUDA_BOTH_INLINE cuComplex operator*(cuComplex x, cuComplex y) { return(cuCmulf(x, y)); }
 // #endif

 //static CUDA_BOTH_INLINE float cuReal(cuComplex x) { return(cuCrealf(x)); }
 //static CUDA_BOTH_INLINE float cuImag(cuComplex x) { return(cuCimagf(x)); }
 static CUDA_BOTH_INLINE cuComplex cuConj(cuComplex x) { return(cuConjf(x)); }

 static CUDA_BOTH_INLINE cuComplex operator+(cuComplex x, cuComplex y) { return(cuCaddf(x, y)); }
 static CUDA_BOTH_INLINE cuComplex operator-(cuComplex x, cuComplex y) { return(cuCsubf(x, y)); }
 static CUDA_BOTH_INLINE cuComplex operator*(float y, cuComplex x)     { return(make_cuComplex(cuCrealf(x)*y, cuCimagf(x)*y)); }

 #if 0
 static CUDA_BOTH_INLINE float cuReal(cuComplex x) { return(cuCrealf(x)); }
 static CUDA_BOTH_INLINE float cuImag(cuComplex x) { return(cuCimagf(x)); }
 static CUDA_BOTH_INLINE cuComplex cuAdd(cuComplex x, cuComplex y) { return(cuCaddf(x, y)); }
 static CUDA_BOTH_INLINE cuComplex cuMul(cuComplex x, cuComplex y) { return(cuCmulf(x, y)); }
 static CUDA_BOTH_INLINE cuComplex cuDiv(cuComplex x, cuComplex y) { return(cuCdivf(x, y)); }
 static CUDA_BOTH_INLINE cuComplex operator+(cuComplex x, cuComplex y) { return(cuCaddf(x, y)); }
 static CUDA_BOTH_INLINE cuComplex operator-(cuComplex x, cuComplex y) { return(cuCsubf(x, y)); }
 static CUDA_BOTH_INLINE cuComplex operator+=(cuComplex &x, cuComplex y)  { x = cuCaddf(x, y); return x; };
 static CUDA_BOTH_INLINE cuComplex operator-=(cuComplex x, cuComplex y) { return(cuCsubf(x, y)); }
 static CUDA_BOTH_INLINE cuComplex operator*(cuComplex x, float y) { return(make_cuComplex(cuCrealf(x)*y, cuCimagf(x)*y)); }
 static CUDA_BOTH_INLINE cuComplex operator/(cuComplex x, float y) { return(make_cuComplex(cuCrealf(x)/y, cuCimagf(x)/y)); }
 static CUDA_BOTH_INLINE cuComplex operator*(cuComplex x, cuComplex y) { return(cuCmulf(x, y)); }
 static CUDA_BOTH_INLINE cuComplex operator*(const cuComplex x, const cuComplex y) { return(cuCmulf(x, y)); }
 static CUDA_BOTH_INLINE cuComplex operator*=(cuComplex &x, float y) { x = make_cuComplex(cuCrealf(x)*y, cuCimagf(x)*y); return x; }
 static CUDA_BOTH_INLINE cuComplex operator*=(cuComplex &x, cuComplex y) { x = cuCmulf(x, y); return x; };
 static CUDA_BOTH_INLINE cuComplex cuFma(cuComplex x, cuComplex y, cuComplex a) { return cuCfmaf(x,y,a); }// a = (x*y) + a;

 // cuda_fp16.hpp
 //__device__ __forceinline__ __half operator*(const __half &lh, const __half &rh) { return __hmul(lh, rh); }
 // __device__ __forceinline__ __half2& operator*=(__half2 &lh, const __half2 &rh) { lh = __hmul2(lh, rh); return lh; }

 // static CUDA_BOTH_INLINE __half2 cuConj(__half2 &hc) { __half2 t; t.x = hc.x; t.y = -hc.y; return t; }
 // static CUDA_BOTH_INLINE __half2 cuGet(int x) {  __half2 t; t.x = __half(x); t.y = __float2half(0.0f); return t; }
 #endif
 // clang-format on

 CUDA_INLINE constexpr uint32_t get_inter_dmrs_grid_freq_shift(const uint32_t nDmrsGridsPerPrb)
 {
     return (2 == nDmrsGridsPerPrb) ? 1 : 2;
 }

 CUDA_INLINE constexpr uint32_t get_inter_dmrs_grid_freq_shift_idx(const uint32_t nDmrsGridsPerPrb, const uint32_t gridIdx)
 {
     return ((nDmrsGridsPerPrb - 1) - gridIdx) * get_inter_dmrs_grid_freq_shift(nDmrsGridsPerPrb);
 }

 CUDA_INLINE constexpr uint32_t get_smem_dmrs_tone_idx(const uint32_t nDmrsGridsPerPrb, const uint32_t nInterDmrsGridFreqShift, const uint32_t tIdx)
 {
     return (2 == nDmrsGridsPerPrb) ? (tIdx / nDmrsGridsPerPrb) :
                                      (nInterDmrsGridFreqShift * (tIdx / (nInterDmrsGridFreqShift * nDmrsGridsPerPrb)) +
                                       (tIdx % nInterDmrsGridFreqShift));
 }

 CUDA_INLINE constexpr uint32_t get_smem_dmrs_grid_idx(const uint32_t nDmrsGridsPerPrb, const uint32_t nInterDmrsGridFreqShift, const uint32_t tIdx)
 {
     return (2 == nDmrsGridsPerPrb) ? (tIdx % nDmrsGridsPerPrb) :
                                      (tIdx / nInterDmrsGridFreqShift) % nDmrsGridsPerPrb;
 }

static __device__ __constant__ __half2 d_twiddle32[31];
static __half2 twiddle32[31] = {{1.000000,0.000000},{1.000000,0.000000},{0.000000,1.000000},{1.000000,0.000000},{0.707031,0.707031},{0.000000,1.000000},{-0.707031,0.707031},{1.000000,0.000000},{0.923828,0.382568},{0.707031,0.707031},{0.382568,0.923828},{0.000000,1.000000},{-0.382568,0.923828},{-0.707031,0.707031},{-0.923828,0.382568},{1.000000,0.000000},{0.980957,0.195068},{0.923828,0.382568},{0.831543,0.555664},{0.707031,0.707031},{0.555664,0.831543},{0.382568,0.923828},{0.195068,0.980957},{0.000000,1.000000},{-0.195068,0.980957},{-0.382568,0.923828},{-0.555664,0.831543},{-0.707031,0.707031},{-0.831543,0.555664},{-0.923828,0.382568},{-0.980957,0.195068}};

static __device__ __constant__ uint8_t d_fourier32PermuteIdx[32];
static uint8_t fourier32PermuteIdx[32] = {0, 16, 8, 24, 4, 20, 12, 28, 2, 18, 10, 26, 6, 22, 14, 30, 1, 17, 9, 25, 5, 21, 13, 29, 3, 19, 11, 27, 7, 23, 15, 31};

static __device__ __constant__ uint8_t d_fourier8PermuteIdx[8];
static uint8_t fourier8PermuteIdx[8] = {0, 4, 2 , 6, 1, 5, 3, 7};

 static __device__ __constant__ int8_t d_phi_6[30][6];

static __device__ __constant__ int8_t d_phi_12[30][12];

static __device__ __constant__ int8_t d_phi_18[30][18];

static __device__ __constant__ int8_t d_phi_24[30][24];

#ifdef ENABLE_PRIME_WHILE_LOOP
    static __device__ __constant__ uint16_t d_primeNums[303];
    static uint16_t                         primeNums[303] = {2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97,101,103,107,109,113,127,131,137,139,149,151,157,163,167,173,179,181,191,193,197,199,211,223,227,229,233,239,241,251,257,263,269,271,277,281,283,293,307,311,313,317,331,337,347,349,353,359,367,373,379,383,389,397,401,409,419,421,431,433,439,443,449,457,461,463,467,479,487,491,499,503,509,521,523,541,547,557,563,569,571,577,587,593,599,601,607,613,617,619,631,641,643,647,653,659,661,673,677,683,691,701,709,719,727,733,739,743,751,757,761,769,773,787,797,809,811,821,823,827,829,839,853,857,859,863,877,881,883,887,907,911,919,929,937,941,947,953,967,971,977,983,991,997,1009,1013,1019,1021,1031,1033,1039,1049,1051,1061,1063,1069,1087,1091,1093,1097,1103,1109,1117,1123,1129,1151,1153,1163,1171,1181,1187,1193,1201,1213,1217,1223,1229,1231,1237,1249,1259,1277,1279,1283,1289,1291,1297,1301,1303,1307,1319,1321,1327,1361,1367,1373,1381,1399,1409,1423,1427,1429,1433,1439,1447,1451,1453,1459,1471,1481,1483,1487,1489,1493,1499,1511,1523,1531,1543,1549,1553,1559,1567,1571,1579,1583,1597,1601,1607,1609,1613,1619,1621,1627,1637,1657,1663,1667,1669,1693,1697,1699,1709,1721,1723,1733,1741,1747,1753,1759,1777,1783,1787,1789,1801,1811,1823,1831,1847,1861,1867,1871,1873,1877,1879,1889,1901,1907,1913,1931,1933,1949,1951,1973,1979,1987,1993,1997,1999};
#else
// d_primeNums[273] is a precalculated table of prime numbers per nPrb to avoid runtime search based on M_ZC implemented as a while loop when ENABLE_PRIME_WHILE_LOOP is not defined.
// M_ZC = N_DMRS_GRID_TONES_PER_PRB * nPrb;
    static __device__ __constant__ uint16_t d_primeNums[273];
    static uint16_t                         primeNums[273] = {5 ,11 ,17 ,23 ,29 ,31 ,41 ,47 ,53 ,59 ,61 ,71 ,73 ,83 ,89 ,89 ,101 ,107 ,113 ,113 ,113 ,131 ,137 ,139 ,149 ,151 ,157 ,167 ,173 ,179 ,181 ,191 ,197 ,199 ,199 ,211 ,211 ,227 ,233 ,239 ,241 ,251 ,257 ,263 ,269 ,271 ,281 ,283 ,293 ,293 ,293 ,311 ,317 ,317 ,317 ,331 ,337 ,347 ,353 ,359 ,359 ,367 ,373 ,383 ,389 ,389 ,401 ,401 ,409 ,419 ,421 ,431 ,433 ,443 ,449 ,449 ,461 ,467 ,467 ,479 ,479 ,491 ,491 ,503 ,509 ,509 ,521 ,523 ,523 ,523 ,541 ,547 ,557 ,563 ,569 ,571 ,577 ,587 ,593 ,599 ,601 ,607 ,617 ,619 ,619 ,631 ,641 ,647 ,653 ,659 ,661 ,661 ,677 ,683 ,683 ,691 ,701 ,701 ,709 ,719 ,719 ,727 ,733 ,743 ,743 ,751 ,761 ,761 ,773 ,773 ,773 ,787 ,797 ,797 ,809 ,811 ,821 ,827 ,829 ,839 ,839 ,839 ,857 ,863 ,863 ,863 ,881 ,887 ,887 ,887 ,887 ,911 ,911 ,919 ,929 ,929 ,941 ,947 ,953 ,953 ,953 ,971 ,977 ,983 ,983 ,991 ,997 ,997 ,1013 ,1019 ,1021 ,1031 ,1033 ,1039 ,1049 ,1051 ,1061 ,1063 ,1069 ,1069 ,1069 ,1091 ,1097 ,1103 ,1109 ,1109 ,1117 ,1123 ,1129 ,1129 ,1129 ,1151 ,1153 ,1163 ,1163 ,1171 ,1181 ,1187 ,1193 ,1193 ,1201 ,1201 ,1217 ,1223 ,1229 ,1231 ,1237 ,1237 ,1249 ,1259 ,1259 ,1259 ,1277 ,1283 ,1289 ,1291 ,1301 ,1307 ,1307 ,1319 ,1321 ,1327 ,1327 ,1327 ,1327 ,1327 ,1361 ,1367 ,1373 ,1373 ,1381 ,1381 ,1381 ,1399 ,1409 ,1409 ,1409 ,1427 ,1433 ,1439 ,1439 ,1451 ,1453 ,1459 ,1459 ,1471 ,1481 ,1487 ,1493 ,1499 ,1499 ,1511 ,1511 ,1523 ,1523 ,1531 ,1531 ,1543 ,1553 ,1559 ,1559 ,1571 ,1571 ,1583 ,1583 ,1583 ,1601 ,1607 ,1613 ,1619 ,1621 ,1627 ,1637};
#endif

#ifndef ENABLE_COMMON_DFTSOFDM_DESCRCODE_SUBROUTINE
static inline __device__ float2 gen_pusch_dftsofdm_descrcode(uint16_t M_ZC, uint16_t rIdx, int u, int v, uint16_t nPrb)
{
    float2 descrCode;
    if(M_ZC < 36)
    {
        if(rIdx < M_ZC)
        {
            switch(M_ZC)
            {
            case 6: {
                descrCode.x =(float)cos(M_PI * (d_phi_6[u][rIdx]) / 4.0f);
                descrCode.y= (float)sin(M_PI * (d_phi_6[u][rIdx]) / 4.0f);
                break;
            }
            case 12: {
                descrCode.x =(float)cos(M_PI * (d_phi_12[u][rIdx]) / 4.0f);
                descrCode.y= (float)sin(M_PI * (d_phi_12[u][rIdx]) / 4.0f);
                break;
            }
            case 18: {
                descrCode.x =(float)cos(M_PI * (d_phi_18[u][rIdx]) / 4.0f);
                descrCode.y= (float)sin(M_PI * (d_phi_18[u][rIdx]) / 4.0f);
                break;
            }
            case 24: {
                descrCode.x =(float)cos(M_PI * (d_phi_24[u][rIdx]) / 4.0f);
                descrCode.y= (float)sin(M_PI * (d_phi_24[u][rIdx]) / 4.0f);
                break;
            }
            case 30: {
                descrCode.x =(float)cos(M_PI * (u + 1) * (rIdx + 1) * (rIdx + 2) / 31.0f);
                descrCode.y= (float)(-sin(M_PI * (u + 1) * (rIdx + 1) * (rIdx + 2) / 31.0f));
                break;
            }
            }
        }
    }
    else
    {
    #ifdef ENABLE_PRIME_WHILE_LOOP
        int idx = 0;
        while(M_ZC > d_primeNums[idx])
        {
            idx++;
        }
        idx--;
        uint16_t d_primeNum = d_primeNums[idx];
    #else
        uint16_t d_primeNum = d_primeNums[nPrb-1];
    #endif
        float qbar = d_primeNum * (u + 1) / 31.0f;
        float q    = (int)(qbar + 0.5f) + (v * (((int)(2 * qbar) & 1) * -2 + 1));
        uint32_t m = rIdx % d_primeNum;
        descrCode.x =(float)cos(M_PI * q * m * (m + 1) / d_primeNum);
        descrCode.y= (float)(-sin(M_PI * q * m * (m + 1) / d_primeNum));
    }
    return descrCode;
}
#endif


 // Channel Estimation kernel:
 // Performs frequency domain interpolation: Uses DMRS tones in N_DMRS_PRB_IN_PER_CLUSTER PRBs and generates
 // channel estimate over N_DMRS_INTERP_PRB_OUT_PER_CLUSTER PRBs for all the layers present in N_DMRS_PRB_IN_PER_CLUSTER PRBs
 // Each thread block consumes a pilot chunk: N_DMRS_PRB_IN_PER_CLUSTER x N_DMRS_SYMS pilot and outputs H as:
 // N_DMRS_SYMS_FOCC x N_DMRS_SYMS_TOCC x N_DMRS_GRIDS_PER_PRB x N_DMRS_INTERP_PRB_OUT_PER_CLUSTER = N_LAYERS x N_DMRS_INTERP_PRB_OUT_PER_CLUSTER
 // Inputs and outputs assumed to be column major

 // Note: The shift and unshift sequences are the same for all DMRS time resources (symbols) but different for
 // DMRS frequency resources (tones). The descrambling sequence is different for each time (symbol) and
 // frequencey (tone) resource

 // Since each thread block estimates channel H for a PRB cluster of size N_DMRS_INTERP_PRB_OUT_PER_CLUSTER PRBs
 // # of thread blocks needed = gridDim = N_DATA_PRB/N_DMRS_INTERP_PRB_OUT_PER_CLUSTER
 // dimBlock: (N_DMRS_PRB_IN_PER_CLUSTER*N_TONES_PER_PRB) dimGrid: (N_DATA_PRB/N_DMRS_INTERP_PRB_OUT_PER_CLUSTER, nBSAnt)
 // Tested for: N_DATA_PRB = 64, N_DMRS_INTERP_PRB_OUT_PER_CLUSTER = 4, N_DMRS_PRB_IN_PER_CLUSTER = 8, dimBlock(96) and dimGrid(68, 16)
 // where N_DATA_PRB is the total # of PRBs bearing data i.e. total number interpolatd DMRS PRBs produced by

 // Channel estimates from only the active DMRS grids are saved. Table below contains mapping of active DMRS grid
 // index bitmask to DMRS grid write index (-1 for inactive grids). Choose data type to be smallest possible type
 // to save on table size (and hence the memory foot print)
 __constant__ int8_t DMRS_GRID_WR_IDX_TBL[][3]{{-1, -1, -1}, {0, -1, -1}, {-1, 0, -1}, {0, 1, -1}, {-1, -1, 0}, {0, -1, 1}, {-1, 0, 1}, {0, 1, 2}};

 #if 0
 template <typename TStorage,
           typename TDataRx,
           typename TCompute,
           uint32_t N_LAYERS,                          // # of layers (# of cols in H matrix)
           uint32_t N_DMRS_GRIDS_PER_PRB,              // # of DMRS grids per PRB (2 or 3)
           uint32_t N_DMRS_PRB_IN_PER_CLUSTER,         // # of PRBs bearing DMRS tones to be processed by each thread block (i.e. used in channel estimation)
           uint32_t N_DMRS_INTERP_PRB_OUT_PER_CLUSTER, // # of PRBs bearing channel estimates (interpolated tones) at output
           uint32_t N_DMRS_SYMS>                       // # of time domain DMRS symbols (1,2 or 4)
 static __global__ void
 windowedChEstKernel(puschRxChEstStatDescr_t* pStatDescr, puschRxChEstDynDescr_t* pDynDescr)
 {
     // PRB cluster being processed by this thread block
     const uint32_t PRB_CLUSTER_IDX = blockIdx.x;
     // BS antenna being processed by this thread block
     const uint32_t BS_ANT_IDX = blockIdx.y;

     if((0 != BS_ANT_IDX) || (0 != PRB_CLUSTER_IDX)) return;

     if((0 == threadIdx.x) && (0 == threadIdx.y) && (0 == threadIdx.z))
     {
         puschRxChEstDynDescr_t& dynDescr      = *(pDynDescr);
         uint16_t                slotNum            = dynDescr.slotNum;
         uint8_t                 activeDmrsGridBmsk = dynDescr.activeDmrsGridBmsk;
         tensor_ref<const uint16_t>       tDmrsScId(dynDescr.tPrmDmrsScId.pAddr             , dynDescr.tPrmDmrsScId.strides);        // (N_UE_GRPS)
         uint16_t dmrsScId = tDmrsScId(blockIdx.z);
         printf("dmrsScId %d slotNum %d activeDmrsGridBmsk 0x%08x\n", dmrsScId, slotNum, activeDmrsGridBmsk);
     }
 }
 #else
 template <typename TStorage,
           typename TDataRx,
           typename TCompute,
           uint32_t N_LAYERS,                          // # of layers (# of cols in H matrix) ** may be larger than the actual number of layers in the group
           uint32_t N_DMRS_GRIDS_PER_PRB,              // # of DMRS grids per PRB (2 or 3)
           uint32_t N_DMRS_PRB_IN_PER_CLUSTER,         // # of PRBs bearing DMRS tones to be processed by each thread block (i.e. used in channel estimation)
           uint32_t N_DMRS_INTERP_PRB_OUT_PER_CLUSTER, // # of PRBs bearing channel estimates (interpolated tones) at output
           uint32_t N_DMRS_SYMS>                       // # of consecutive DMRS symbols (1 or 2)
 static __global__ void
 windowedChEstNoDftSOfdmKernel(puschRxChEstStatDescr_t* pStatDescr, puschRxChEstDynDescr_t* pDynDescr)
 {
     //--------------------------------------------------------------------------------------------------------
     puschRxChEstDynDescr_t& dynDescr = *pDynDescr;

     // UE group processed by this thread block
     const uint32_t UE_GRP_IDX = dynDescr.hetCfgUeGrpMap[blockIdx.z];
     // PRB cluster being processed by this thread block
     const uint32_t PRB_CLUSTER_IDX = blockIdx.x;
     // BS antenna being processed by this thread block
     const uint32_t BS_ANT_IDX = blockIdx.y;

     // Early exit check
     // The grid is sized to process the max # of PRB clusters in a given heterogenous config. Exit if the PRB cluster to be
     // processed by this thread block does not exist in the UE group
     cuphyPuschRxUeGrpPrms_t& drvdUeGrpPrms = dynDescr.pDrvdUeGrpPrms[UE_GRP_IDX];
     const uint16_t nPrb   = drvdUeGrpPrms.nPrb;
     const uint16_t nRxAnt = drvdUeGrpPrms.nRxAnt;
     const uint32_t N_PRB_CLUSTERS_PER_BS_ANT = div_round_up(nPrb, static_cast<uint16_t>(N_DMRS_INTERP_PRB_OUT_PER_CLUSTER));
     if((PRB_CLUSTER_IDX >= N_PRB_CLUSTERS_PER_BS_ANT) || (BS_ANT_IDX >= nRxAnt)) return;

#ifdef ENABLE_DEBUG
     if((0 == blockIdx.x) && (0 == blockIdx.y) && (0 == threadIdx.x) && (0 == threadIdx.y) && (0 == threadIdx.z))
     printf("%s\n blockIdx.z %d UE_GRP_IDX %d blockDim (%d,%d,%d) gridDim (%d,%d,%d)\n", __PRETTY_FUNCTION__, blockIdx.z, UE_GRP_IDX, blockDim.x, blockDim.y, blockDim.z, gridDim.x, gridDim.y, gridDim.z);

     if((0 == threadIdx.x) && (0 == threadIdx.y) && (0 == threadIdx.z) && (0 == blockIdx.y) && (0 == blockIdx.z))
     printf("PRB_CLUSTER_IDX %d N_PRB_CLUSTERS_PER_BS_ANT %d chEstTimeInst %d\n", PRB_CLUSTER_IDX, N_PRB_CLUSTERS_PER_BS_ANT, dynDescr.chEstTimeInst);
#endif

     //--------------------------------------------------------------------------------------------------------
     // Setup local parameters based on descriptor
     puschRxChEstStatDescr_t& statDescr = *pStatDescr;
     const uint16_t  slotNum            = drvdUeGrpPrms.slotNum;
     const uint8_t   chEstTimeInst      = dynDescr.chEstTimeInst;
     const uint8_t   activeDmrsGridBmsk = drvdUeGrpPrms.activeDMRSGridBmsk;
     uint8_t*        OCCIdx             = drvdUeGrpPrms.OCCIdx;
     const uint16_t  nLayers            = drvdUeGrpPrms.nLayers;
     const uint8_t   dmrsMaxLen         = drvdUeGrpPrms.dmrsMaxLen;
     const uint8_t   nPrbsMod2          = nPrb & 0x1;
     const uint32_t  N_DATA_PRB         = nPrb;
     const uint8_t   scid               = drvdUeGrpPrms.scid;

     // Pointer to DMRS symbol used for channel estimation (single-symbol if maxLen = 1, double-symbol if maxLen = 2)
     const uint8_t* const pDmrsSymPos   = &drvdUeGrpPrms.dmrsSymLoc[chEstTimeInst*dmrsMaxLen];
     const uint16_t  startPrb = drvdUeGrpPrms.startPrb;
     const uint16_t  dmrsScId = drvdUeGrpPrms.dmrsScrmId;
     const uint8_t   nDmrsCdmGrpsNoData = drvdUeGrpPrms.nDmrsCdmGrpsNoData;

     //--------------------------------------------------------------------------------------------------------
     typedef typename complex_from_scalar<TCompute>::type TComplexCompute;
     typedef typename complex_from_scalar<TDataRx>::type TComplexDataRx;
     typedef typename complex_from_scalar<TStorage>::type TComplexStorage;

     // clang-format off
     tensor_ref<const TCompute>       tFreqInterpCoefs((8==N_DMRS_PRB_IN_PER_CLUSTER) ? statDescr.tPrmFreqInterpCoefs.pAddr : statDescr.tPrmFreqInterpCoefs4.pAddr,
                                                       (8==N_DMRS_PRB_IN_PER_CLUSTER) ? statDescr.tPrmFreqInterpCoefs.strides : statDescr.tPrmFreqInterpCoefs4.strides); // (N_TOTAL_DMRS_INTERP_GRID_TONES_PER_CLUSTER + N_INTER_DMRS_GRID_FREQ_SHIFT, N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER, 3), 3 filters: 1 for middle, 1 lower edge and 1 upper edge

   //  tensor_ref<const TStorage>       tFreqInterpCoefs(statDescr.tPrmFreqInterpCoefs.pAddr, statDescr.tPrmFreqInterpCoefs.strides);
 #if 1 // shift/unshift sequences same precision as data (FP16 or FP32)
     tensor_ref<const TComplexDataRx> tShiftSeq       (statDescr.tPrmShiftSeq.pAddr       , statDescr.tPrmShiftSeq.strides);        // (N_DATA_PRB*N_DMRS_GRID_TONES_PER_PRB, N_DMRS_SYMS)
     tensor_ref<const TComplexDataRx> tUnShiftSeq     (statDescr.tPrmUnShiftSeq.pAddr     , statDescr.tPrmUnShiftSeq.strides);      // (N_DATA_PRB*N_DMRS_INTERP_TONES_PER_GRID*N_DMRS_GRIDS_PER_PRB + N_INTER_DMRS_GRID_FREQ_SHIFT)
 #else // shift/unshift sequences same precision as channel estimates (typically FP32)
     tensor_ref<const TComplexStorage> tShiftSeq       (statDescr.tPrmShiftSeq.pAddr       , statDescr.tPrmShiftSeq.strides);        // (N_DATA_PRB*N_DMRS_GRID_TONES_PER_PRB, N_DMRS_SYMS)
     tensor_ref<const TComplexStorage> tUnShiftSeq     (statDescr.tPrmUnShiftSeq.pAddr     , statDescr.tPrmUnShiftSeq.strides);      // (N_DATA_PRB*N_DMRS_INTERP_TONES_PER_GRID*N_DMRS_GRIDS_PER_PRB + N_INTER_DMRS_GRID_FREQ_SHIFT)
 #endif
     tensor_ref<const TComplexDataRx> tDataRx        (drvdUeGrpPrms.tInfoDataRx.pAddr  , drvdUeGrpPrms.tInfoDataRx.strides);// (NF, ND, N_BS_ANTS)
     tensor_ref<TComplexStorage>      tHEst          (drvdUeGrpPrms.tInfoHEst.pAddr    , drvdUeGrpPrms.tInfoHEst.strides);
     tensor_ref<TComplexStorage>      tDbg           (drvdUeGrpPrms.tInfoChEstDbg.pAddr, drvdUeGrpPrms.tInfoChEstDbg.strides);
     // clang-format on

#ifdef ENABLE_DEBUG
     if((0 == threadIdx.x) && (0 == threadIdx.y) && (0 == threadIdx.z) && (0 == blockIdx.x) && (0 == blockIdx.y))
     printf("%s\n: NH_IDX %d hetCfgUeGrpIdx %d ueGrpIdx %d nPrb %d startPb %d pDmrsSymPos[0] %d pDmrsSymPos[1] %d dmrsScId %d\n", __PRETTY_FUNCTION__, dynDescr.chEstTimeInst, blockIdx.z, UE_GRP_IDX, nPrb, startPrb, pDmrsSymPos[0], pDmrsSymPos[1], dmrsScId);
#endif

     //--------------------------------------------------------------------------------------------------------
     // Dimensions and indices

     // Estimates of H in time supported
     const uint32_t NH_IDX = chEstTimeInst;

     // Channel estimation expands tones in a DMRS grid (4 or 6, given by N_DMRS_GRID_TONES_PER_PRB) into a full PRB
     constexpr uint32_t N_DMRS_INTERP_TONES_PER_GRID = N_TONES_PER_PRB;

     // # of tones per DMRS grid in a PRB
     constexpr uint32_t N_DMRS_GRID_TONES_PER_PRB = N_TONES_PER_PRB / N_DMRS_GRIDS_PER_PRB;
     // Max permissible DMRS grids within a PRB based on spec
     constexpr uint32_t N_DMRS_TYPE1_GRIDS_PER_PRB = 2;
     constexpr uint32_t N_DMRS_TYPE2_GRIDS_PER_PRB = 3;
     static_assert(((N_DMRS_TYPE1_GRIDS_PER_PRB == N_DMRS_GRIDS_PER_PRB) || (N_DMRS_TYPE2_GRIDS_PER_PRB == N_DMRS_GRIDS_PER_PRB)),
                   "DMRS grid count exceeds max value");

     // Within a PRB, successive DMRS grids are shifted by 2 tones
     constexpr uint32_t N_INTER_DMRS_GRID_FREQ_SHIFT = get_inter_dmrs_grid_freq_shift(N_DMRS_GRIDS_PER_PRB);

     // Per grid tone counts present in input and output PRB clusters. These tones counts are expected to be equal
     constexpr uint32_t N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER        = N_DMRS_GRID_TONES_PER_PRB * N_DMRS_PRB_IN_PER_CLUSTER;
     constexpr uint32_t N_TOTAL_DMRS_INTERP_GRID_TONES_PER_CLUSTER = N_DMRS_INTERP_TONES_PER_GRID * N_DMRS_INTERP_PRB_OUT_PER_CLUSTER;

     // Total # of DMRS tones consumed by this thread block (this number should equal number of threads in
     // thread block since each DMRS tone is processed by a thread)
     constexpr uint32_t N_DMRS_TONES = N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER * N_DMRS_GRIDS_PER_PRB; // blockDim.x
     // Total # of interpolated DMRS tones produced by this thread block (this number should also equal number
     // of threads in thread block)
     constexpr uint32_t N_INTERP_TONES = N_TOTAL_DMRS_INTERP_GRID_TONES_PER_CLUSTER * N_DMRS_GRIDS_PER_PRB; // blockDim.x

     static_assert((N_DMRS_PRB_IN_PER_CLUSTER * N_TONES_PER_PRB == N_DMRS_TONES),
                   "Mismatch in expected vs calcualted DMRS tone count");
     static_assert((N_DMRS_TONES == N_INTERP_TONES),
                   "Thread allocation assumes input DMRS tone count and interpolated tone count are equal, ensure sufficient threads are allocated for interpoloation etc");

     // Ensure configured symbol count does not exceed max value prescribed by spec
     static_assert((N_DMRS_SYMS <= N_MAX_DMRS_SYMS), "DMRS symbol count exceeds max value");

     // Interpolation filter indices for middle and edge PRBs
     constexpr uint32_t MIDDLE_INTERP_FILT_IDX     = 0;
     constexpr uint32_t LOWER_EDGE_INTERP_FILT_IDX = 1;
     constexpr uint32_t UPPER_EDGE_INTERP_FILT_IDX = 2;

     // DMRS descrambling
     constexpr uint32_t N_DMRS_DESCR_BITS_PER_TONE    = 2; // 1bit for I and 1 bit for Q
     constexpr uint32_t N_DMRS_DESCR_BITS_PER_CLUSTER = N_DMRS_DESCR_BITS_PER_TONE * N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER;
     // # of DMRS descrambler bits generated at one time
     constexpr uint32_t N_DMRS_DESCR_BITS_GEN = 32;
     // Round up to the next multiple of N_DMRS_DESCR_BITS_GEN plus 1 (+1 because DMRS_DESCR_PRB_CLUSTER_START_BIT_OFFSET
     // may be large enough to spill the descrambler bits to the next word)
     constexpr uint32_t N_DMRS_DESCR_WORDS =
         ((N_DMRS_DESCR_BITS_PER_CLUSTER + N_DMRS_DESCR_BITS_GEN - 1) / N_DMRS_DESCR_BITS_GEN) + 1;
     // round_up_to_next(N_DMRS_DESCR_BITS_PER_CLUSTER, N_DMRS_DESCR_BITS_GEN) + 1;

     // number of "edge" tones, not estimated but used to extract additional dmrs
     constexpr uint32_t HALF_N_EDGE_TONES = N_TONES_PER_PRB * (N_DMRS_PRB_IN_PER_CLUSTER - N_DMRS_INTERP_PRB_OUT_PER_CLUSTER) / 2;

     const uint32_t ACTIVE_DMRS_GRID_BMSK = 0x3;

     // Total number of PRB clusers to be processed (N_PRB_CLUSTERS*N_DMRS_INTERP_PRB_OUT_PER_CLUSTER = N_DATA_PRB)
     const uint32_t N_PRB_CLUSTERS = N_PRB_CLUSTERS_PER_BS_ANT;

     // Per UE group descrambling ID
     uint16_t dmrsScramId = dmrsScId;

#ifdef ENABLE_DEBUG

     if((0 == blockIdx.x) && (0 == blockIdx.y) && (0 == blockIdx.z) && (0 == threadIdx.x) && (0 == threadIdx.y) && (0 == threadIdx.z))
     {
         printf("Addr: tFreqInterpCoefs %lp tShiftSeq %lp tUnShiftSeq %lp tDataRx %lp tHEst %lp \n", tFreqInterpCoefs.pAddr, tShiftSeq.pAddr, tUnShiftSeq.pAddr, tDataRx.pAddr, tHEst.pAddr);

         printf("tFreqInterpCoefs: addr %lp strides[0] %d strides[1] %d strides[2] %d\n", static_cast<const TCompute*>(tFreqInterpCoefs.pAddr) , tFreqInterpCoefs.strides[0], tFreqInterpCoefs.strides[1], tFreqInterpCoefs.strides[2]);
         printf("tShiftSeq       : addr %lp strides[0] %d strides[1] %d strides[2] %d\n", static_cast<const TComplexDataRx*>(tShiftSeq.pAddr)  , tShiftSeq.strides[0]       , tShiftSeq.strides[1]       , tShiftSeq.strides[2]       );
         printf("tUnShiftSeq     : addr %lp strides[0] %d strides[1] %d strides[2] %d\n", static_cast<const TComplexDataRx*>(tUnShiftSeq.pAddr), tUnShiftSeq.strides[0]     , tUnShiftSeq.strides[1]     , tUnShiftSeq.strides[2]     );

         printf("startPrb       : %d \n", startPrb);
         printf("dmrsScId       : %d\n", dmrsScId);
         printf("tDataRx         : addr %lp strides[0] %d strides[1] %d strides[2] %d\n", static_cast<const TComplexDataRx*>(tDataRx.pAddr)    , tDataRx.strides[0]         , tDataRx.strides[1]         , tDataRx.strides[2]         );
         printf("tHEst           : addr %lp strides[0] %d strides[1] %d strides[2] %d\n", static_cast<TComplexStorage*>(tHEst.pAddr)     , tHEst.strides[0]                 , tHEst.strides[1]           , tHEst.strides[2]           );
         // printf("tDbg    strides[0] %d strides[1] %d strides[2] %d\n", tDbg.strides[0], tDbg.strides[1], tDbg.strides[2]);

         printf("dmrsScramId %d slotNum %d activeDmrsGridBmsk 0x%08x chEstTimeInst %d\n", dmrsScramId, slotNum, activeDmrsGridBmsk, NH_IDX);
     }

     // printf("Block(%d %d %d) Thread(%d %d %d)\n", blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z);
     // if((0 != BS_ANT_IDX) || (0 != PRB_CLUSTER_IDX)) return;
     // printf("dmrsScramId %d slotNum %d activeDmrsGridBmsk 0x%08x", dmrsScramId, slotNum, activeDmrsGridBmsk);
     // if((0 == blockIdx.x) && (0 == blockIdx.y) && (0 == blockIdx.z) && (0 == threadIdx.x) && (0 == threadIdx.y) && (0 == threadIdx.z))
     //    printf("tDataRx strides[0] %d strides[1] %d strides[2] %d\n", tDataRx.strides[0], tDataRx.strides[1], tDataRx.strides[2]);
 #if 0
     printf("InterpCoefs[%d][%d][%d] = %f ShiftSeq[%d][%d] = %f+j%f UnShiftSeq[%d] %f+j%f DataRx[%d][%d][%d]= %f+j%f\n",
            0,0,0,
            tFreqInterpCoefs(0,0,0),
            0,0,
            tShiftSeq(0,0).x,
            tShiftSeq(0,0).y,
            0,
            tUnShiftSeq(0).x,
            tUnShiftSeq(0).y,
            0,0,0,
            tDataRx(0,0,0).x,
            tDataRx(0,0,0).y);
 #endif
#endif

     const uint32_t THREAD_IDX = threadIdx.x;

     // # of PRBs for which channel must be estimated
     const uint32_t N_EDGE_PRB = (N_DMRS_PRB_IN_PER_CLUSTER - N_DMRS_INTERP_PRB_OUT_PER_CLUSTER) / 2; // Lower and Upper edge PRBs

     // Determine first PRB in the cluster being processed
     uint32_t prbClusterStartIdx = (PRB_CLUSTER_IDX * N_DMRS_INTERP_PRB_OUT_PER_CLUSTER) - N_EDGE_PRB;
     if(0 == PRB_CLUSTER_IDX) prbClusterStartIdx = 0;                                                             // Lower edge
     if((N_PRB_CLUSTERS - 1) == PRB_CLUSTER_IDX) prbClusterStartIdx = N_DATA_PRB - N_DMRS_PRB_IN_PER_CLUSTER;     // Upper edge

     uint32_t prbAbsStartIdx = prbClusterStartIdx + startPrb;
     // Absolute index of DMRS tone within the input OFDM symbol (used as index when loading tone from OFDM
     // symbol)
     const uint32_t DMRS_ABS_TONE_IDX = prbAbsStartIdx * N_TONES_PER_PRB + THREAD_IDX;

     // This index calculation intends to divvy up threads in the thread block for processing as follows:
     // the first group of N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER to process the first DMRS grid, the second group
     // to process the second DMRS grid and so on
     // Relative index of DMRS tone (within a DMRS grid) being processed by this thread
     const uint32_t DMRS_TONE_IDX        = THREAD_IDX % N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER;
     const uint32_t DMRS_INTERP_TONE_IDX = THREAD_IDX % N_TOTAL_DMRS_INTERP_GRID_TONES_PER_CLUSTER;

     // Index of DMRS grid in which the DMRS tone being processed by this thread resides
     // Note: although the grid index is calculated using total number of DMRS grid tones in the cluster, its
     // used as an index in context of both input DMRS tones and interpolated DMRS tones under the assumption:
     // N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER == N_TOTAL_DMRS_INTERP_GRID_TONES_PER_CLUSTER
     const uint32_t DMRS_GRID_IDX = THREAD_IDX / N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER;

     const uint32_t tmpGridIdx = DMRS_GRID_IDX > 1 ? 1 : DMRS_GRID_IDX;
     const uint8_t  activeTOCCBmsk     = drvdUeGrpPrms.activeTOCCBmsk[tmpGridIdx];
     const uint8_t  activeFOCCBmsk     = drvdUeGrpPrms.activeFOCCBmsk[tmpGridIdx];
     const uint32_t N_DMRS_SYMS_FOCC = activeFOCCBmsk == 3 ? 2 : 1;
     // Index which enables extraction of DMRS tones of a given DMRS grid scattered within the PRB into one
     // contiguous set for processing. Note that the read from GMEM is coalesced and write into SMEM is scattered
     // @todo: check if index calculation can be simplified
     const uint32_t SMEM_DMRS_TONE_IDX = get_smem_dmrs_tone_idx(N_DMRS_GRIDS_PER_PRB, N_INTER_DMRS_GRID_FREQ_SHIFT, THREAD_IDX);
     const uint32_t SMEM_DMRS_GRID_IDX = get_smem_dmrs_grid_idx(N_DMRS_GRIDS_PER_PRB, N_INTER_DMRS_GRID_FREQ_SHIFT, THREAD_IDX);

     // Absolute index of descrambling + shift sequence element
     const uint32_t DMRS_DESCR_SHIFT_SEQ_START_IDX = prbAbsStartIdx * N_DMRS_GRID_TONES_PER_PRB;
     // const uint32_t DMRS_DESCR_SHIFT_SEQ_ABS_IDX   = DMRS_DESCR_SHIFT_SEQ_START_IDX + DMRS_TONE_IDX;
     const uint32_t DMRS_DESCR_SHIFT_SEQ_ABS_IDX = DMRS_TONE_IDX;

     // Select one of 3 interpolation filters for middle section, lower and upper edges of the frequency band
     uint32_t filtIdx = MIDDLE_INTERP_FILT_IDX;                                        // All tones in between lower and upper edges
     if(0 == PRB_CLUSTER_IDX) filtIdx = LOWER_EDGE_INTERP_FILT_IDX;                    // Lower edge
     if((N_PRB_CLUSTERS - 1) == PRB_CLUSTER_IDX) filtIdx = UPPER_EDGE_INTERP_FILT_IDX; // Upper edge

     // Absolute index of interpolated tone produced by this thread
     const uint32_t INTERP_PRB_CLUSTER_IDX   = blockIdx.x;
     uint32_t INTERP_DMRS_ABS_TONE_IDX = INTERP_PRB_CLUSTER_IDX * N_DMRS_INTERP_PRB_OUT_PER_CLUSTER * N_TONES_PER_PRB + DMRS_INTERP_TONE_IDX;

     if((N_DMRS_PRB_IN_PER_CLUSTER == 4) && (nPrbsMod2 == 1) && (filtIdx == UPPER_EDGE_INTERP_FILT_IDX))
         INTERP_DMRS_ABS_TONE_IDX = INTERP_DMRS_ABS_TONE_IDX - N_TONES_PER_PRB;

     // Select the shift in interpolation filter coefficients and delay shift based on grid index
     // (for e.g. for 2 DMRS grids and 48 tones per grid, multiply DMRS tone vector with top 48 rows for
     // DMRS_GRID_IDX 0 and bottom 48 rows for DMRS_GRID_IDX 1 to acheieve the effect of shift)
     uint32_t gridShiftIdx = get_inter_dmrs_grid_freq_shift_idx(N_DMRS_GRIDS_PER_PRB, DMRS_GRID_IDX);

     // Section 5.2.1 in 3GPP TS 38.211
     // The fast-forward of 1600 prescribed by spec is already baked into the gold sequence generator
     constexpr uint32_t DMRS_DESCR_FF = 0; // 1600;

     // First descrambler bit index needed by this thread block
     // Note:The DMRS scrambling sequence is the same for all the DMRS grids. There are 2 sequences one for
     // scid 0 and other for scid 1 but the same sequences is reused for all DMRS grids
     const uint32_t DMRS_DESCR_PRB_CLUSTER_START_BIT =
         DMRS_DESCR_FF + (DMRS_DESCR_SHIFT_SEQ_START_IDX * N_DMRS_DESCR_BITS_PER_TONE);

     // The descrambling sequence generator outputs 32 descrambler bits at a time. Thus, compute the earliest
     // multiple of 32 bits which contains the descrambler bit of the first tone in the PRB cluster as the
     // start index
     const uint32_t DMRS_DESCR_GEN_ALIGNED_START_BIT =
         (DMRS_DESCR_PRB_CLUSTER_START_BIT / N_DMRS_DESCR_BITS_GEN) * N_DMRS_DESCR_BITS_GEN;
     // Offset to descrambler bit of the first tone in the PRB cluster
     const uint32_t DMRS_DESCR_PRB_CLUSTER_START_BIT_OFFSET =
         DMRS_DESCR_PRB_CLUSTER_START_BIT - DMRS_DESCR_GEN_ALIGNED_START_BIT;

     // DMRS descrambling bits generated correspond to subcarriers across frequency
     // e.g. 2 bits for tone0(grid 0) | 2 bits for tone1(grid 1) | 2 bits for tone 2(grid 0) | 2 bits for tone 3(grid 1) | ...
     const uint32_t DMRS_TONE_DESCR_BIT_IDX = DMRS_DESCR_PRB_CLUSTER_START_BIT_OFFSET +
                                              (DMRS_TONE_IDX * N_DMRS_DESCR_BITS_PER_TONE);
     const uint32_t DMRS_DESCR_SEQ_RD_BIT_IDX  = DMRS_TONE_DESCR_BIT_IDX % N_DMRS_DESCR_BITS_GEN;
     const uint32_t DMRS_DESCR_SEQ_RD_WORD_IDX = DMRS_TONE_DESCR_BIT_IDX / N_DMRS_DESCR_BITS_GEN;

     const uint32_t DMRS_DESCR_SEQ_WR_WORD_IDX = THREAD_IDX % N_DMRS_DESCR_WORDS;
     const uint32_t DMRS_DESCR_SEQ_WR_SYM_IDX  = THREAD_IDX / N_DMRS_DESCR_WORDS;

 #if 0
     if((0 == blockIdx.x) && (0 == blockIdx.y) && (0 == blockIdx.z))
        printf("N_DMRS_DESCR_BITS_PER_CLUSTER %d N_DMRS_DESCR_WORDS %d DMRS_DESCR_PRB_CLUSTER_START_BIT %d DMRS_DESCR_GEN_ALIGNED_START_BIT %d, "
               "DMRS_DESCR_SEQ_RD_WORD_IDX %d, DMRS_DESCR_SEQ_RD_BIT_IDX %d, DMRS_DESCR_SEQ_WR_WORD_IDX %d, DMRS_DESCR_SEQ_WR_SYM_IDX %d\n",
               N_DMRS_DESCR_BITS_PER_CLUSTER, N_DMRS_DESCR_WORDS, DMRS_DESCR_PRB_CLUSTER_START_BIT, DMRS_DESCR_GEN_ALIGNED_START_BIT,
               DMRS_DESCR_SEQ_RD_WORD_IDX, DMRS_DESCR_SEQ_RD_BIT_IDX, DMRS_DESCR_SEQ_WR_WORD_IDX, DMRS_DESCR_SEQ_WR_SYM_IDX);
 #endif

     // Data layouts:
     // Global memory read into shared memory
     // N_DMRS_TONES x N_DMRS_SYMS -> N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER x N_DMRS_SYMS x N_DMRS_GRIDS_PER_PRB

     // tOCC removal
     // N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER x N_DMRS_SYMS x N_DMRS_GRIDS_PER_PRB ->
     // N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER x N_DMRS_SYMS_TOCC x N_DMRS_GRIDS_PER_PRB

     // fOCC removal
     // N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER x N_DMRS_SYMS_TOCC x N_DMRS_GRIDS_PER_PRB ->
     // N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER x N_DMRS_SYMS_FOCC x NUM_DMRS_SYMS_TOCC x N_DMRS_GRIDS_PER_PRB

     // Interpolation
     // N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER x N_DMRS_SYMS_FOCC x NUM_DMRS_SYMS_TOCC x N_DMRS_GRIDS_PER_PRB ->
     // N_TOTAL_DMRS_INTERP_GRID_TONES_PER_CLUSTER x N_DMRS_SYMS_FOCC x NUM_DMRS_SYMS_TOCC x N_DMRS_GRIDS_PER_PRB =
     // N_TOTAL_DMRS_INTERP_GRID_TONES_PER_CLUSTER x N_LAYERS x N_DMRS_GRIDS_PER_PRB

     //--------------------------------------------------------------------------------------------------------
     // Allocate shared memory

     constexpr uint32_t N_SMEM_ELEMS1 = N_DMRS_TONES * N_DMRS_SYMS; // (N_DMRS_TONES + N_DMRS_GRIDS_PER_PRB)*N_DMRS_SYMS;
     constexpr uint32_t N_SMEM_ELEMS2 = N_INTERP_TONES * N_DMRS_SYMS_OCC;
     constexpr uint32_t N_SMEM_ELEMS  = (N_SMEM_ELEMS1 > N_SMEM_ELEMS2) ? N_SMEM_ELEMS1 : N_SMEM_ELEMS2;
     // constexpr uint32_t N_SMEM_ELEMS  = (N_SMEM_ELEMS1 + N_SMEM_ELEMS2);
     // constexpr uint32_t N_SMEM_ELEMS  = max(N_SMEM_ELEMS1, N_SMEM_ELEMS2);

     __shared__ TComplexCompute smemBlk[N_SMEM_ELEMS];
     // overlay1
     block_3D<TComplexCompute*, N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER, N_DMRS_SYMS, N_DMRS_GRIDS_PER_PRB> shPilots(&smemBlk[0]);
     // overlay2
     block_3D<TComplexCompute*, N_TOTAL_DMRS_INTERP_GRID_TONES_PER_CLUSTER, N_DMRS_SYMS_OCC, N_DMRS_GRIDS_PER_PRB> shH(&smemBlk[0]);
     // block_3D<TComplexCompute*, N_TOTAL_DMRS_INTERP_GRID_TONES_PER_CLUSTER, N_DMRS_SYMS_OCC, N_DMRS_GRIDS_PER_PRB> shH(&smemBlk[shPilots.num_elem()]);
     static_assert((shPilots.num_elem() <= N_SMEM_ELEMS) && (shH.num_elem() <= N_SMEM_ELEMS), "Insufficient shared memory");

     __shared__ uint32_t descrWords[N_DMRS_SYMS][N_DMRS_DESCR_WORDS];

     //--------------------------------------------------------------------------------------------------------
     // Read DMRS tones into shared memory (separate the tones into different DMRS grids during the write)

     // Cache shift sequence in register
     TComplexCompute shiftSeq = type_convert<TComplexCompute>(tShiftSeq(DMRS_DESCR_SHIFT_SEQ_ABS_IDX, 0));

 #pragma unroll
     for(uint32_t i = 0; i < N_DMRS_SYMS; ++i)
     {
         shPilots(SMEM_DMRS_TONE_IDX, i, SMEM_DMRS_GRID_IDX) =
             type_convert<TComplexCompute>(tDataRx(DMRS_ABS_TONE_IDX, pDmrsSymPos[i], BS_ANT_IDX));

 #ifdef ENABLE_DEBUG
         printf("Pilots[%d][%d][%d] -> shPilots[%d][%d][%d] = %f+j%f, ShiftSeq[%d][%d] = %f+j%f\n",
                DMRS_ABS_TONE_IDX,
                pDmrsSymPos[i],
                BS_ANT_IDX,
                SMEM_DMRS_TONE_IDX,
                i,
                SMEM_DMRS_GRID_IDX,
                shPilots(SMEM_DMRS_TONE_IDX, i, SMEM_DMRS_GRID_IDX).x,
                shPilots(SMEM_DMRS_TONE_IDX, i, SMEM_DMRS_GRID_IDX).y,
                DMRS_DESCR_SHIFT_SEQ_ABS_IDX,
                0,
                shiftSeq.x,
                shiftSeq.y);
 #endif
     }

     // Compute the descsrambler sequence
     const uint32_t TWO_POW_17 = bit(17);

     if(DMRS_DESCR_SEQ_WR_SYM_IDX < N_DMRS_SYMS)
     {
         uint32_t symIdx = pDmrsSymPos[DMRS_DESCR_SEQ_WR_SYM_IDX];

         // see 38.211 section 6.4.1.1.1.1
         uint32_t cInit = TWO_POW_17 * (slotNum * OFDM_SYMBOLS_PER_SLOT + symIdx + 1) * (2 * dmrsScramId + 1) + (2 * dmrsScramId) + scid;
         cInit &= ~bit(31);

         // descrWords[DMRS_DESCR_SEQ_WR_SYM_IDX][DMRS_DESCR_SEQ_WR_WORD_IDX] =
         //  __brev(gold32(cInit, (DMRS_DESCR_GEN_ALIGNED_START_BIT + DMRS_DESCR_SEQ_WR_WORD_IDX*N_DMRS_DESCR_BITS_GEN)));

         descrWords[DMRS_DESCR_SEQ_WR_SYM_IDX][DMRS_DESCR_SEQ_WR_WORD_IDX] =
             gold32(cInit, (DMRS_DESCR_GEN_ALIGNED_START_BIT + DMRS_DESCR_SEQ_WR_WORD_IDX * N_DMRS_DESCR_BITS_GEN));
 #if 0
         printf("symIdx %d, DMRS_DESCR_SEQ_WR_WORD_IDX %d, cInit 0x%08x, DMRS_DESCR_GEN_ALIGNED_START_BIT %d, descrWords[%d][%d] 0x%08x\n",
                symIdx, DMRS_DESCR_SEQ_WR_WORD_IDX, cInit,
                (DMRS_DESCR_GEN_ALIGNED_START_BIT + DMRS_DESCR_SEQ_WR_WORD_IDX*N_DMRS_DESCR_BITS_GEN),
                DMRS_DESCR_SEQ_WR_SYM_IDX, DMRS_DESCR_SEQ_WR_WORD_IDX, descrWords[DMRS_DESCR_SEQ_WR_SYM_IDX][DMRS_DESCR_SEQ_WR_WORD_IDX]);
 #endif
     }

     // To ensure coalesced reads, input DMRS tones are read preserving input order but swizzled while writing
     // to shared memory. Thus each thread may not process the same tone which it wrote to shared memory
     thread_block const& thisThrdBlk = this_thread_block();
     __syncthreads();

     //--------------------------------------------------------------------------------------------------------
     // Apply de-scrambling + delay domain centering sequence for tone index processed by this thread across all
     // DMRS symbols
     const TCompute RECIPROCAL_SQRT2 = 0.7071068f;
     const TCompute SQRT2            = 1.41421356f;

 #pragma unroll
     for(uint32_t i = 0; i < N_DMRS_SYMS; ++i)
     {
         int8_t descrIBit = (descrWords[i][DMRS_DESCR_SEQ_RD_WORD_IDX] >> DMRS_DESCR_SEQ_RD_BIT_IDX) & 0x1;
         int8_t descrQBit = (descrWords[i][DMRS_DESCR_SEQ_RD_WORD_IDX] >> (DMRS_DESCR_SEQ_RD_BIT_IDX + 1)) & 0x1;

         TComplexCompute descrCode =
             cuConj(cuGet<TComplexCompute>((1 - 2 * descrIBit) * RECIPROCAL_SQRT2, (1 - 2 * descrQBit) * RECIPROCAL_SQRT2));
         TComplexCompute descrShiftSeq = shiftSeq * descrCode;

 #ifdef ENABLE_DEBUG
         TComplexCompute descrShiftPilot = shPilots(DMRS_TONE_IDX, i, DMRS_GRID_IDX) * descrShiftSeq;
         printf("descrShiftAbsIdx: %d, shPilots[%d][%d][%d] (%f+j%f) * DescrShiftSeq[%d][%d] (%f+j%f) = %f+j%f, ShiftSeq = %f+j%f, DescrCode = %f+j%f, descrIQ (%d,%d) descrWordIdx %d descrBitIdx %d\n",
                DMRS_DESCR_SHIFT_SEQ_ABS_IDX,
                DMRS_TONE_IDX,
                i,
                DMRS_GRID_IDX,
                shPilots(DMRS_TONE_IDX, i, DMRS_GRID_IDX).x,
                shPilots(DMRS_TONE_IDX, i, DMRS_GRID_IDX).y,
                DMRS_TONE_IDX,
                i,
                descrShiftSeq.x,
                descrShiftSeq.y,
                descrShiftPilot.x,
                descrShiftPilot.y,
                shiftSeq.x,
                shiftSeq.y,
                descrCode.x,
                descrCode.y,
                descrIBit,
                descrQBit,
                DMRS_DESCR_SEQ_RD_WORD_IDX,
                DMRS_DESCR_SEQ_RD_BIT_IDX);
         if((0 == DMRS_GRID_IDX) && (((0 == prbAbsStartIdx) && (DMRS_TONE_IDX < (N_EDGE_PRB * N_DMRS_GRID_TONES_PER_PRB))) || ((0 != prbAbsStartIdx) && (prbAbsStartIdx + N_DMRS_PRB_IN_PER_CLUSTER) <= N_DATA_PRB)))
         {
 #if 0
            TComplexCompute descrShiftPilot = shPilots(DMRS_TONE_IDX, i, DMRS_GRID_IDX) * descrShiftSeq;
            printf("descrShiftAbsIdx: %d shPilots[%d][%d][%d] (%f+j%f) * DescrShiftSeq[%d][%d] (%f+j%f) = %f+j%f, ShiftSeq = %f+j%f, DescrCode = %f+j%f, descrIQ (%d,%d) descrWordIdx %d descrBitIdx %d\n",
                  DMRS_DESCR_SHIFT_SEQ_ABS_IDX,
                  DMRS_TONE_IDX,
                  i,
                  DMRS_GRID_IDX,
                  shPilots(DMRS_TONE_IDX, i, DMRS_GRID_IDX).x,
                  shPilots(DMRS_TONE_IDX, i, DMRS_GRID_IDX).y,
                  DMRS_TONE_IDX,
                  i,
                  descrShiftSeq.x,
                  descrShiftSeq.y,
                  descrShiftPilot.x,
                  descrShiftPilot.y,
                  shiftSeq.x,
                  shiftSeq.y,
                  descrCode.x,
                  descrCode.y,
                  descrIBit,
                  descrQBit,
                  DMRS_DESCR_SEQ_RD_WORD_IDX,
                  DMRS_DESCR_SEQ_RD_BIT_IDX);
 #endif

             // tDbg(DMRS_DESCR_SHIFT_SEQ_ABS_IDX, i, 0, 0) = type_convert<TComplexStorage>(shiftSeq);
             // tDbg(DMRS_DESCR_SHIFT_SEQ_ABS_IDX, i, 0, 0) = type_convert<TComplexStorage>(descrShiftSeq);
             tDbg(DMRS_DESCR_SHIFT_SEQ_ABS_IDX, i, 0, 0) = type_convert<TComplexStorage>(descrCode);
         }
 #endif // ENABLE_DEBUG

         shPilots(DMRS_TONE_IDX, i, DMRS_GRID_IDX) *= descrShiftSeq;
     }

     //--------------------------------------------------------------------------------------------------------
     // Time domain cover code removal
     constexpr TCompute AVG_SCALE = cuGet<TCompute>(1u) / cuGet<TCompute>(N_DMRS_SYMS);
     TComplexCompute    avg[N_TOCC]{};

     int32_t tOCCIdx = 0;

 #pragma unroll
     for(int32_t i = 0; i < 2; ++i)
     {
        int32_t temp = (activeTOCCBmsk >> i) & 0x1;
        if (!temp) {
            continue;
        }
 #pragma unroll
         for(int32_t j = 0; j < N_DMRS_SYMS; ++j)
         {
             // For first tOCC (i = 0) output, multiply all DMRS symbols with +1 and average
             // For second tOCC (i = 1) output, multiply even DMRS symbols with +1, odd DMRS symbols with -1 and average
             int32_t sign = (-(i & j)) | 1;
             avg[tOCCIdx] += (shPilots(DMRS_TONE_IDX, j, DMRS_GRID_IDX) * (sign * AVG_SCALE));
 #ifdef ENABLE_DEBUG
             TComplexCompute prod = (shPilots(DMRS_TONE_IDX, j, DMRS_GRID_IDX) * (sign * AVG_SCALE));
             printf("sign*AVG_SCALE %f Pilot[%d][%d][%d] = %f+j%f avg[%d] = %f+j%f, prod = %f+j%f\n",
                    sign * AVG_SCALE,
                    DMRS_TONE_IDX,
                    j,
                    DMRS_GRID_IDX,
                    shPilots(DMRS_TONE_IDX, j, DMRS_GRID_IDX).x,
                    shPilots(DMRS_TONE_IDX, j, DMRS_GRID_IDX).y,
                    i,
                    avg[i].x,
                    avg[i].y,
                    prod.x,
                    prod.y);
 #endif
         }
         tOCCIdx++;
     }

     // shPilots and shH are overlaid in shared memory and can have different sizes (based on config). For this reason
     // ensure shPilots access from all threads is completed before writing into shH
     __syncthreads();

     //--------------------------------------------------------------------------------------------------------
     // Apply frequecy domain cover code and store inplace in shared memory
     // Multiply even tones with +1 and odd tones with -1

     // Note that the loop termination count below is tOCC symbol count
 #pragma unroll
     for(int32_t i = 0; i < tOCCIdx; ++i)
     {
        int32_t fOCCIdx = 0;

 #pragma unroll
         for(int32_t j = 0; j < 2; ++j)
         {
            int32_t temp = (activeFOCCBmsk >> j) & 0x1;
            if (!temp) {
                continue;
            }
             // First fOCC output: multiply all tones by +1s
             // Second fOCC output: multiply even tones by +1s and odd tones by -1s
             int32_t sign                                                  = (-(DMRS_TONE_IDX & j)) | 1;
             shH(DMRS_TONE_IDX, (N_DMRS_SYMS_FOCC * i) + fOCCIdx, DMRS_GRID_IDX) = avg[i] * sign;

 #ifdef ENABLE_DEBUG
             printf("PilotsPostOCC[%d][%d][%d] = %f+j%f\n",
                    DMRS_TONE_IDX,
                    (N_DMRS_SYMS_FOCC * i) + j,
                    DMRS_GRID_IDX,
                    cuReal(shH(DMRS_TONE_IDX, (N_DMRS_SYMS_FOCC * i) + j, DMRS_GRID_IDX)),
                    cuImag(shH(DMRS_TONE_IDX, (N_DMRS_SYMS_FOCC * i) + j, DMRS_GRID_IDX)));
 #endif
             fOCCIdx++;
         }
     }

     // Ensure all threads complete writing results to shared memory since each thread computing an inner product
     // during interpolation stage will use results from other threads in the thread block
     __syncthreads();

     //--------------------------------------------------------------------------------------------------------
     // Interpolate (matrix-vector multiply)
     for(uint32_t i = 0; i < (N_DMRS_SYMS_FOCC*tOCCIdx); ++i)
     {
         TComplexCompute innerProd{};

         // H = W x Y: (N_INTERP_TONES x N_DMRS_TONES) x (N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER x N_DMRS_SYMS_OCC)
         // Each thread selects one row of W and computes N_DMRS_TONES length inner product to produce one interpolated
         // tone of H
         for(uint32_t j = 0; j < N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER; ++j)
         {
             TCompute interpCoef = type_convert<TCompute>(tFreqInterpCoefs(DMRS_INTERP_TONE_IDX + gridShiftIdx, j, filtIdx));
             innerProd += (shH(j, i, DMRS_GRID_IDX) * interpCoef);
         }
         // Wait for all threads to complete their inner products before updating the shared memory inplace
         // The sync is needed because shPilots and shH are overlaid.
         // Note that tOCCIdx can vary per thread, so this sync and the next need to be thisThrdBlk.sync()
         // calls rather than __syncthreads().
         thisThrdBlk.sync();

         shH(DMRS_INTERP_TONE_IDX, i, DMRS_GRID_IDX) = innerProd;

 #ifdef ENABLE_DEBUG
         printf("InterpPilots[%d][%d][%d] = %f+j%f\n", DMRS_INTERP_TONE_IDX, i, DMRS_GRID_IDX, innerProd.x, innerProd.y);
 #endif
     }

     // Wait for shared memory writes to complete from all threads
     // See above comment for prior thisThrdBlk.sync() call as to why this is thisThrdBlk.sync()
     // rather than __syncthreads().
     thisThrdBlk.sync();

     // Write channel estimates for active grids only
     const int32_t DMRS_GRID_WR_IDX = DMRS_GRID_WR_IDX_TBL[activeDmrsGridBmsk & ACTIVE_DMRS_GRID_BMSK][DMRS_GRID_IDX];
     // if(!is_set(bit(DMRS_GRID_IDX), activeDmrsGridBmsk) || (DMRS_GRID_WR_IDX < 0)) return;
     if(DMRS_GRID_WR_IDX < 0) return;

     //--------------------------------------------------------------------------------------------------------
     // Unshift the channel in delay back to its original location and write to GMEM. This is a scattered write
     // (@todo: any opportunities to make it coalesced?)
     // Output format is N_BS_ANT x (N_DMRS_SYMS_FOCC x N_DMRS_SYMS_TOCC x N_DMRS_GRIDS_PER_PRB) x N_DMRS_INTERP_PRB_OUT_PER_CLUSTER
     // where N_DMRS_SYMS_FOCC x N_DMRS_SYMS_TOCC x N_DMRS_GRIDS_PER_PRB = N_LAYERS
     //const uint32_t DMRS_GRID_OFFSET_H = DMRS_GRID_WR_IDX * N_DMRS_SYMS_OCC;

 #ifdef CH_EST_COALESCED_WRITE
     // H (N_DATA_PRB*N_TONES_PER_PRB, N_LAYERS, N_BS_ANTS, NH)
     //read the number of rx antennas
     uint32_t N_BS_ANTS  = drvdUeGrpPrms.nRxAnt;
     TComplexStorage* pHEst = tHEst.addr + ((NH_IDX * N_BS_ANTS + BS_ANT_IDX) * N_LAYERS * N_DATA_PRB * N_TONES_PER_PRB);
 #endif


    // index of interpolated tone within cluster
    uint32_t CLUSTER_INTERP_TONE_IDX = DMRS_INTERP_TONE_IDX + HALF_N_EDGE_TONES;
    if(0 == PRB_CLUSTER_IDX) CLUSTER_INTERP_TONE_IDX = CLUSTER_INTERP_TONE_IDX - HALF_N_EDGE_TONES;                        // Lower edge
    if((N_PRB_CLUSTERS - 1) == PRB_CLUSTER_IDX) CLUSTER_INTERP_TONE_IDX = CLUSTER_INTERP_TONE_IDX + HALF_N_EDGE_TONES;     // Upper edge

    // check if estimated tone dropped
    if(N_DMRS_PRB_IN_PER_CLUSTER == 4)
    {
        if((filtIdx == UPPER_EDGE_INTERP_FILT_IDX) && (nPrbsMod2 == 1) && (DMRS_INTERP_TONE_IDX < 12))
            return;
    }

 #pragma unroll
     for(uint32_t i = 0; i < N_LAYERS; ++i)
     {
         if (i < nLayers) {
            uint32_t j = OCCIdx[i] & 0x3;
            uint32_t k = (OCCIdx[i] >> 2) & 0x1;
            if (DMRS_GRID_IDX == k) {
                shH(DMRS_INTERP_TONE_IDX, j, DMRS_GRID_IDX) *=
                type_convert<TComplexCompute>(tUnShiftSeq(CLUSTER_INTERP_TONE_IDX + gridShiftIdx)); //INTERP_DMRS_ABS_TONE_IDX
                if(nDmrsCdmGrpsNoData==1)
                {
                    shH(DMRS_INTERP_TONE_IDX, j, DMRS_GRID_IDX) *= cuGet<TComplexCompute>(SQRT2, 0.0f);
                }
#ifndef CH_EST_COALESCED_WRITE
                tHEst(BS_ANT_IDX, i, INTERP_DMRS_ABS_TONE_IDX, NH_IDX) =
                     type_convert<TComplexStorage>(shH(DMRS_INTERP_TONE_IDX, j, DMRS_GRID_IDX));

                ////Test/////
                //if (BS_ANT_IDX == 0 && i == 3 && INTERP_DMRS_ABS_TONE_IDX && !NH_IDX)
                //printf("minus itHEst = %f+j%f\n", tHEst(BS_ANT_IDX, i, INTERP_DMRS_ABS_TONE_IDX, NH_IDX).x-tHEst(BS_ANT_IDX, 1, INTERP_DMRS_ABS_TONE_IDX, NH_IDX).x, tHEst(BS_ANT_IDX, i, INTERP_DMRS_ABS_TONE_IDX, NH_IDX).y-tHEst(BS_ANT_IDX, 1, INTERP_DMRS_ABS_TONE_IDX, NH_IDX).y);
                /////////////

#else //fix me for flexbile DMRS port mapping
                pHEst[(DMRS_GRID_OFFSET_H + i) * N_DATA_PRB * N_TONES_PER_PRB + INTERP_DMRS_ABS_TONE_IDX] =
                     type_convert<TComplexStorage>(shH(DMRS_INTERP_TONE_IDX, i, DMRS_GRID_IDX));
#endif
            }
#ifdef ENABLE_DEBUG
#if 0
     printf("shH[%d][%d][%d] = %f+j%f\n", DMRS_INTERP_TONE_IDX, i, DMRS_GRID_IDX,
         shH(DMRS_INTERP_TONE_IDX, i, DMRS_GRID_IDX).x, shH(DMRS_INTERP_TONE_IDX, i, DMRS_GRID_IDX).y);
#endif
#if 0
     if(((6 == UE_GRP_IDX) || (7 == UE_GRP_IDX) || (8 == UE_GRP_IDX)) && (PRB_CLUSTER_IDX < 1))
     {
        TCompute hEstReal = type_convert<TCompute>(tHEst(BS_ANT_IDX, i, INTERP_DMRS_ABS_TONE_IDX, NH_IDX).x);
        TCompute hEstImag = type_convert<TCompute>(tHEst(BS_ANT_IDX, i, INTERP_DMRS_ABS_TONE_IDX, NH_IDX).y);
        printf("ueGrpIdx %d blockIdx.z %d tH[%d][%d][%d][%d] = %f+j%f\n", UE_GRP_IDX, blockIdx.z, BS_ANT_IDX, i, INTERP_DMRS_ABS_TONE_IDX, NH_IDX, hEstReal, hEstImag);
     }

#endif
#if 0
       printf("tUnshift[%d] = %f+j%f\n", INTERP_DMRS_ABS_TONE_IDX + gridShiftIdx,tUnShiftSeq(INTERP_DMRS_ABS_TONE_IDX + gridShiftIdx).x, tUnShiftSeq(INTERP_DMRS_ABS_TONE_IDX + gridShiftIdx).y);

       printf("tH[%d][%d][%d][%d] = %f+j%f\n", BS_ANT_IDX, DMRS_GRID_OFFSET_H + i, INTERP_DMRS_ABS_TONE_IDX, NH_IDX,
         tHEst(BS_ANT_IDX, DMRS_GRID_OFFSET_H + i, INTERP_DMRS_ABS_TONE_IDX, NH_IDX).x,
         tHEst(BS_ANT_IDX, DMRS_GRID_OFFSET_H + i, INTERP_DMRS_ABS_TONE_IDX, NH_IDX).y);
#endif
#endif
        }
     }
 } //windowedChEstNoDftSOfdmKernel

template <typename TStorage,
           typename TDataRx,
           typename TCompute,
           //uint32_t N_LAYERS,                          // # of layers (# of cols in H matrix) ** may be larger than the actual number of layers in the group
           uint32_t N_DMRS_GRIDS_PER_PRB,              // # of DMRS grids per PRB (2 or 3)
           uint32_t N_DMRS_PRB_IN_PER_CLUSTER,
           uint32_t N_DMRS_INTERP_PRB_OUT_PER_CLUSTER,
           uint32_t N_DMRS_SYMS>                       // # of consecutive DMRS symbols (1 or 2)
static __device__ void
windowedChEstFilterNoDftSOfdmKernel(
    puschRxChEstStatDescr_t* pStatDescr, puschRxChEstDynDescr_t* pDynDescr,
    typename complex_from_scalar<TCompute>::type *sh_shiftSeq,
    typename complex_from_scalar<TCompute>::type *sh_unShiftSeq,
    typename complex_from_scalar<TCompute>::type *sh_ls_est)
{
     typedef typename complex_from_scalar<TCompute>::type TComplexCompute;
     typedef typename complex_from_scalar<TStorage>::type TComplexStorage;

     // We only currently support type 1 DMRS grids, which have DMRS tones in every other frequency bin.
     constexpr uint32_t N_DMRS_TYPE1_GRIDS_PER_PRB = 2;
     static_assert(N_DMRS_GRIDS_PER_PRB == N_DMRS_TYPE1_GRIDS_PER_PRB, "Kernel only supports type 1 DMRS grids");
     constexpr uint32_t N_DMRS_TONE_STRIDE = 2;

     thread_block const& block = this_thread_block();

     puschRxChEstStatDescr_t& statDescr = *pStatDescr;
     puschRxChEstDynDescr_t& dynDescr = *pDynDescr;

     // UE group processed by this thread block
     const uint32_t UE_GRP_IDX = dynDescr.hetCfgUeGrpMap[blockIdx.z];
     // PRB cluster being processed by this thread block
     const uint32_t PRB_CLUSTER_IDX = blockIdx.x;

     // Channel estimation expands tones in a DMRS grid (4 or 6, given by N_DMRS_GRID_TONES_PER_PRB) into a full PRB
     constexpr uint32_t N_DMRS_INTERP_TONES_PER_GRID = N_TONES_PER_PRB;

     // Per grid tone counts present in input and output PRB clusters. These tones counts are expected to be equal
     constexpr uint32_t N_TOTAL_DMRS_INTERP_GRID_TONES_PER_CLUSTER = N_DMRS_INTERP_TONES_PER_GRID * N_DMRS_INTERP_PRB_OUT_PER_CLUSTER;

     const uint32_t THREAD_IDX = threadIdx.x;
     const uint32_t LAYER_IDX = THREAD_IDX / N_TOTAL_DMRS_INTERP_GRID_TONES_PER_CLUSTER;
     const uint32_t THREAD_IDX_MOD_LAYER = THREAD_IDX - LAYER_IDX * (N_TOTAL_DMRS_INTERP_GRID_TONES_PER_CLUSTER);

     const uint32_t tid = threadIdx.x;
     const uint32_t nthreads = blockDim.x;

     cuphyPuschRxUeGrpPrms_t& drvdUeGrpPrms = dynDescr.pDrvdUeGrpPrms[UE_GRP_IDX];
     const uint16_t nPrb   = drvdUeGrpPrms.nPrb;
     const uint16_t nRxAnt = drvdUeGrpPrms.nRxAnt;
     uint8_t*       OCCIdx = drvdUeGrpPrms.OCCIdx;
     const uint8_t   nDmrsCdmGrpsNoData = drvdUeGrpPrms.nDmrsCdmGrpsNoData;
     // BS antenna being processed by this thread block. The y dimension of the CTA is an rx antenna
     // blocking dimension.
     const uint32_t BS_ANT_IDX = blockDim.y * blockIdx.y + threadIdx.y;
     const uint32_t N_PRB_CLUSTERS_PER_BS_ANT = div_round_up(nPrb, static_cast<uint16_t>(N_DMRS_INTERP_PRB_OUT_PER_CLUSTER));
     if((PRB_CLUSTER_IDX >= N_PRB_CLUSTERS_PER_BS_ANT) || (BS_ANT_IDX >= nRxAnt)) return;

     const uint32_t N_PRB_CLUSTERS = N_PRB_CLUSTERS_PER_BS_ANT;
     const uint16_t  nLayers            = drvdUeGrpPrms.nLayers;
     const uint8_t   chEstTimeInst      = dynDescr.chEstTimeInst;

     const uint32_t N_EDGE_PRB = (N_DMRS_PRB_IN_PER_CLUSTER - N_DMRS_INTERP_PRB_OUT_PER_CLUSTER) / 2; // Lower and Upper edge PRBs

     constexpr uint32_t N_DMRS_GRID_TONES_PER_PRB = N_TONES_PER_PRB / N_DMRS_GRIDS_PER_PRB;
     constexpr uint32_t N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER = N_DMRS_GRID_TONES_PER_PRB * N_DMRS_PRB_IN_PER_CLUSTER;

     // The filter coefficients are different depending on PRB count and PRB location (i.e.
     // edge PRBs have different filter coefficients than central PRBs)
     tensor_ref<const TCompute> tFreqInterpCoefs = [nPrb, &statDescr]() -> auto {
        if (nPrb <= 3) {
            return tensor_ref<const TCompute>{ statDescr.tPrmFreqInterpCoefsSmall.pAddr, statDescr.tPrmFreqInterpCoefsSmall.strides };
        } else if (nPrb >= 8 && nPrb % 4 == 0) {
            return tensor_ref<const TCompute>{ statDescr.tPrmFreqInterpCoefs.pAddr, statDescr.tPrmFreqInterpCoefs.strides };
        } else {
            return tensor_ref<const TCompute>{ statDescr.tPrmFreqInterpCoefs4.pAddr, statDescr.tPrmFreqInterpCoefs4.strides };
        }
     }();

     const uint32_t filtIdx = [nPrb, PRB_CLUSTER_IDX, N_PRB_CLUSTERS]() -> uint32_t {
        if (nPrb == 0) {
            return 0;
        } else if (nPrb <= 3) {
            return nPrb - 1;
        } else {
            // Interpolation filter indices for middle and edge PRBs
            constexpr uint32_t MIDDLE_INTERP_FILT_IDX     = 0;
            constexpr uint32_t LOWER_EDGE_INTERP_FILT_IDX = 1;
            constexpr uint32_t UPPER_EDGE_INTERP_FILT_IDX = 2;
            if (PRB_CLUSTER_IDX == 0) {
                return LOWER_EDGE_INTERP_FILT_IDX;
            } else if (PRB_CLUSTER_IDX == N_PRB_CLUSTERS-1) {
                return UPPER_EDGE_INTERP_FILT_IDX;
            } else {
                return MIDDLE_INTERP_FILT_IDX;
            }
        }
     }();

     // Determine first PRB in the cluster being processed
     uint32_t prbClusterStartIdx = (PRB_CLUSTER_IDX * N_DMRS_INTERP_PRB_OUT_PER_CLUSTER) - N_EDGE_PRB;
     if(0 == PRB_CLUSTER_IDX) prbClusterStartIdx = 0;                                                             // Lower edge
     if((N_PRB_CLUSTERS - 1) == PRB_CLUSTER_IDX) prbClusterStartIdx = nPrb - N_DMRS_PRB_IN_PER_CLUSTER;     // Upper edge

    // When populating the LS intermediate data structure, different values of DMRS_GRID_IDX
    // map to different layers rather than an offset in the tone subcarrier frequency.
    const uint32_t toneOffset = prbClusterStartIdx * N_TONES_PER_PRB;
    const uint32_t DMRS_LS_EST_TONE_OFFSET = toneOffset / N_DMRS_TONE_STRIDE;

    const uint32_t NH_IDX = chEstTimeInst;

    tensor_ref<const TComplexCompute> tInfoDmrsLSEst(drvdUeGrpPrms.tInfoDmrsLSEst.pAddr, drvdUeGrpPrms.tInfoDmrsLSEst.strides);
    const TComplexCompute *dmrsLSEst = tInfoDmrsLSEst.pAddr + tInfoDmrsLSEst.offset(DMRS_LS_EST_TONE_OFFSET, 0, BS_ANT_IDX, NH_IDX);
    const uint32_t toneStride = tInfoDmrsLSEst.strides[0];
    const uint32_t layerStride = tInfoDmrsLSEst.strides[1];

    const int nIter = N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER * nLayers;
    for (int i = tid; i < nIter; i += nthreads) {
        const int layer = i / N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER;
        const int tone = i % N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER;
        sh_ls_est[i] = dmrsLSEst[tone * toneStride + layer * layerStride] * sh_shiftSeq[tone];
    }

     __syncthreads();

     // All block synchronizations have completed, so threads that will not generate outputs can now exit.
     if (LAYER_IDX >= nLayers) {
        return;
     }

     const uint32_t DMRS_INTERP_TONE_IDX = THREAD_IDX_MOD_LAYER % N_TOTAL_DMRS_INTERP_GRID_TONES_PER_CLUSTER;

     constexpr uint32_t HALF_N_EDGE_TONES = N_TONES_PER_PRB * (N_DMRS_PRB_IN_PER_CLUSTER - N_DMRS_INTERP_PRB_OUT_PER_CLUSTER) / 2;
     uint32_t CLUSTER_INTERP_TONE_IDX = DMRS_INTERP_TONE_IDX + HALF_N_EDGE_TONES;
     if(0 == PRB_CLUSTER_IDX) CLUSTER_INTERP_TONE_IDX = CLUSTER_INTERP_TONE_IDX - HALF_N_EDGE_TONES;                        // Lower edge
     if((N_PRB_CLUSTERS - 1) == PRB_CLUSTER_IDX) CLUSTER_INTERP_TONE_IDX = CLUSTER_INTERP_TONE_IDX + HALF_N_EDGE_TONES;     // Upper edge

     // Absolute index of interpolated tone produced by this thread
     const uint32_t INTERP_PRB_CLUSTER_IDX   = blockIdx.x;
     uint32_t INTERP_DMRS_ABS_TONE_IDX = INTERP_PRB_CLUSTER_IDX * N_DMRS_INTERP_PRB_OUT_PER_CLUSTER * N_TONES_PER_PRB + DMRS_INTERP_TONE_IDX;

     if((N_DMRS_PRB_IN_PER_CLUSTER == 4) && (nPrb % 2 == 1) && (PRB_CLUSTER_IDX == N_PRB_CLUSTERS-1)) {
         INTERP_DMRS_ABS_TONE_IDX = INTERP_DMRS_ABS_TONE_IDX - N_TONES_PER_PRB;
     }
     const uint32_t MAX_INTERP_ABS_TONE = nPrb * N_TONES_PER_PRB - 1;

     if (INTERP_DMRS_ABS_TONE_IDX <= MAX_INTERP_ABS_TONE) {
        const TCompute scaling = (nDmrsCdmGrpsNoData == 1) ? static_cast<TCompute>(1.414213562373095f) : 1.0f;
        const uint32_t gridShiftIdx = get_inter_dmrs_grid_freq_shift_idx(N_DMRS_GRIDS_PER_PRB, (OCCIdx[LAYER_IDX] >> 2) & 0x1);

        tensor_ref<TComplexStorage> tHEst(drvdUeGrpPrms.tInfoHEst.pAddr, drvdUeGrpPrms.tInfoHEst.strides);
        TComplexStorage *HEst = tHEst.pAddr + tHEst.offset(BS_ANT_IDX, 0, INTERP_DMRS_ABS_TONE_IDX, NH_IDX);
        const uint32_t estLayerStride = tHEst.strides[1];

        const TCompute *coefs = tFreqInterpCoefs.pAddr + tFreqInterpCoefs.offset(DMRS_INTERP_TONE_IDX+gridShiftIdx, 0, filtIdx);
        const int coefToneStride = tFreqInterpCoefs.strides[1];

        TComplexCompute accum { 0.0f, 0.0f };
        for (int j = 0; j < N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER; j++) {
            const TComplexCompute prod = sh_ls_est[LAYER_IDX*N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER+j] *
                coefs[j * coefToneStride];
            accum += prod;
        }

        accum *= (sh_unShiftSeq[CLUSTER_INTERP_TONE_IDX+gridShiftIdx] * scaling);
        HEst[LAYER_IDX * estLayerStride] = type_convert<TComplexStorage>(accum);
     }
}

template <typename TCompute>
static __device__ void
chEstDelayShiftReduction(
    TCompute &sh_delay_mean,
    const puschRxChEstDynDescr_t& dynDescr,
    const cuphyPuschRxUeGrpPrms_t& drvdUeGrpPrms,
    thread_block &block,
    uint8_t numChEstTimeInst,
    float df,
    uint32_t tid)
{
    typedef typename complex_from_scalar<TCompute>::type TComplexCompute;

    if (block.thread_rank() == 0) {
        const TComplexCompute *pAccum = reinterpret_cast<TComplexCompute *>(drvdUeGrpPrms.tInfoDmrsAccum.pAddr);
        const TComplexCompute accum = pAccum[drvdUeGrpPrms.dmrsActiveAccumBuf];
        const float P_dmrs = (drvdUeGrpPrms.nLayers == 1) ? 2.0f : 4.0f;
        sh_delay_mean = -atan2f(accum.y, accum.x)/2.0f/static_cast<float>(M_PI)/P_dmrs;

        // All threads in this block will use the value sh_delay_mean from shared memory,
        // but we also save it to global memory for debugging purposes.
        if (blockIdx.x == 0 && blockIdx.y == 0) {
            tensor_ref<TCompute> tInfoDmrsDelayMean(drvdUeGrpPrms.tInfoDmrsDelayMean.pAddr, drvdUeGrpPrms.tInfoDmrsDelayMean.strides);
            tInfoDmrsDelayMean(dynDescr.chEstTimeInst) = sh_delay_mean;
        }

        sh_delay_mean /= df;
        
//        if((blockIdx.x==0)&&(blockIdx.y==0))
//        {
//            printf("UEG[%d]sh_delay_mean[%.9f]prgSize[%d]\n", blockIdx.z, sh_delay_mean*1000000.0, drvdUeGrpPrms.prgSize);
//        }
    }

    __syncthreads();
}

template <typename TStorage,
          typename TDataRx,
          typename TCompute,
          uint32_t N_DMRS_GRIDS_PER_PRB>              // # of DMRS grids per PRB (2 or 3)
static __global__ void
chEstFilterNoDftSOfdmDispatchKernel(puschRxChEstStatDescr_t* pStatDescr, puschRxChEstDynDescr_t* pDynDescr)
{
    typedef typename complex_from_scalar<TCompute>::type TComplexCompute;

    thread_block block = this_thread_block();

    const puschRxChEstDynDescr_t& dynDescr = *pDynDescr;

    // UE group processed by this thread block
    const uint32_t UE_GRP_IDX = dynDescr.hetCfgUeGrpMap[blockIdx.z];
    cuphyPuschRxUeGrpPrms_t& drvdUeGrpPrms = dynDescr.pDrvdUeGrpPrms[UE_GRP_IDX];

    constexpr uint32_t MAX_N_DMRS_PRB_IN_PER_CLUSTER = 8;
    constexpr uint32_t N_DMRS_GRID_TONES_PER_PRB = N_TONES_PER_PRB / N_DMRS_GRIDS_PER_PRB;
    constexpr uint32_t MAX_N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER = N_DMRS_GRID_TONES_PER_PRB * MAX_N_DMRS_PRB_IN_PER_CLUSTER;

    // Worst-case shared memory allocations
    constexpr uint32_t MAX_NUM_PRBS_PER_FILTER = 8;
    __shared__ TComplexCompute sh_shiftSeq[MAX_NUM_PRBS_PER_FILTER*(N_TONES_PER_PRB/N_DMRS_GRIDS_PER_PRB)];
    __shared__ TComplexCompute sh_unShiftSeq[(MAX_NUM_PRBS_PER_FILTER*N_TONES_PER_PRB)+1];
    extern __shared__ TComplexCompute sh_ls_est_full[];

    const uint32_t tid = block.thread_rank();
    const uint32_t nthreads = block.num_threads();

    const uint8_t mu = drvdUeGrpPrms.mu;
    // Subcarrier spacing df is 15 kHz * 2^mu for mu in [0, 6]
    const float df = 15.0e3f * static_cast<float>(1U << mu);

    __shared__ TCompute sh_delay_mean;
    chEstDelayShiftReduction<TCompute>(
        sh_delay_mean,
        dynDescr,
        drvdUeGrpPrms,
        block,
        drvdUeGrpPrms.dmrsAddlnPos + 1,
        df,
        tid);

    const uint16_t nPrb = drvdUeGrpPrms.nPrb;
    const uint32_t nPrbInPerCluster = [nPrb]() -> uint32_t {
        if (nPrb <= 3) {
            return nPrb;
        } else if (nPrb >= 8 && nPrb % 4 == 0) {
            return 8;
        } else {
            return 4;
        }
    }();

    // Calculate the shift/unshift coefficients, which depend on sh_delay_mean
    const float partial_arg = df * 2.0f * static_cast<float>(M_PI) * sh_delay_mean;
    for (uint32_t i = tid; i < nPrbInPerCluster*(N_TONES_PER_PRB/N_DMRS_GRIDS_PER_PRB); i += nthreads) {
        const float f_dmrs = partial_arg * (2 * i);
        sincosf(f_dmrs, &sh_shiftSeq[i].y, &sh_shiftSeq[i].x);
    }

    for (uint32_t i = tid; i < (nPrbInPerCluster*N_TONES_PER_PRB)+1; i += nthreads) {
        const float f_data = -1.0f * partial_arg * (static_cast<float>(i)-1.0f);
        sincosf(f_data, &sh_unShiftSeq[i].y, &sh_unShiftSeq[i].x);
    }

    __syncthreads();

    // Logically, the smem data is stored as [antenna][layer][tone]
    TComplexCompute *sh_ls_est = sh_ls_est_full + threadIdx.y *
        drvdUeGrpPrms.nLayers * MAX_N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER;

    const uint8_t maxLen = drvdUeGrpPrms.dmrsMaxLen;
    if (nPrb <= 3) {
        if (maxLen == 1) {
            const uint32_t MAX_LEN = 1;
            switch (nPrb) {
                case 1:
                    windowedChEstFilterNoDftSOfdmKernel<
                        TStorage, TDataRx, TCompute, N_DMRS_GRIDS_PER_PRB, 1, 1, MAX_LEN>(
                            pStatDescr, pDynDescr, sh_shiftSeq, sh_unShiftSeq, sh_ls_est);
                    break;
                case 2:
                    windowedChEstFilterNoDftSOfdmKernel<
                        TStorage, TDataRx, TCompute, N_DMRS_GRIDS_PER_PRB, 2, 2, MAX_LEN>(
                            pStatDescr, pDynDescr, sh_shiftSeq, sh_unShiftSeq, sh_ls_est);
                    break;
                case 3:
                    windowedChEstFilterNoDftSOfdmKernel<
                        TStorage, TDataRx, TCompute, N_DMRS_GRIDS_PER_PRB, 3, 3, MAX_LEN>(
                            pStatDescr, pDynDescr, sh_shiftSeq, sh_unShiftSeq, sh_ls_est);
                    break;
            }
        } else {
            const uint32_t MAX_LEN = 2;
            switch (nPrb) {
                case 1:
                    windowedChEstFilterNoDftSOfdmKernel<
                        TStorage, TDataRx, TCompute, N_DMRS_GRIDS_PER_PRB, 1, 1, MAX_LEN>(
                            pStatDescr, pDynDescr, sh_shiftSeq, sh_unShiftSeq, sh_ls_est);
                    break;
                case 2:
                    windowedChEstFilterNoDftSOfdmKernel<
                        TStorage, TDataRx, TCompute, N_DMRS_GRIDS_PER_PRB, 2, 2, MAX_LEN>(
                            pStatDescr, pDynDescr, sh_shiftSeq, sh_unShiftSeq, sh_ls_est);
                    break;
                case 3:
                    windowedChEstFilterNoDftSOfdmKernel<
                        TStorage, TDataRx, TCompute, N_DMRS_GRIDS_PER_PRB, 3, 3, MAX_LEN>(
                            pStatDescr, pDynDescr, sh_shiftSeq, sh_unShiftSeq, sh_ls_est);
                    break;
            }
        }
    } else if (nPrb >= 8 && nPrb % 4 == 0) {
        const uint32_t PRB_IN_PER_CLUSTER = 8;
        const uint32_t INTERP_PRB_OUT_PER_CLUSTER = 4;
        if (maxLen == 1) {
            const uint32_t MAX_LEN = 1;
            windowedChEstFilterNoDftSOfdmKernel<
                TStorage, TDataRx, TCompute, N_DMRS_GRIDS_PER_PRB,
                PRB_IN_PER_CLUSTER, INTERP_PRB_OUT_PER_CLUSTER, MAX_LEN>(
                    pStatDescr, pDynDescr, sh_shiftSeq, sh_unShiftSeq, sh_ls_est);
        } else {
            const uint32_t MAX_LEN = 2;
            windowedChEstFilterNoDftSOfdmKernel<
                TStorage, TDataRx, TCompute, N_DMRS_GRIDS_PER_PRB,
                PRB_IN_PER_CLUSTER, INTERP_PRB_OUT_PER_CLUSTER, MAX_LEN>(
                    pStatDescr, pDynDescr, sh_shiftSeq, sh_unShiftSeq, sh_ls_est);
        }
    } else {
        const uint32_t PRB_IN_PER_CLUSTER = 4;
        const uint32_t INTERP_PRB_OUT_PER_CLUSTER = 2;
        if (maxLen == 1) {
            const uint32_t MAX_LEN = 1;
            windowedChEstFilterNoDftSOfdmKernel<
                TStorage, TDataRx, TCompute, N_DMRS_GRIDS_PER_PRB,
                PRB_IN_PER_CLUSTER, INTERP_PRB_OUT_PER_CLUSTER, MAX_LEN>(
                    pStatDescr, pDynDescr, sh_shiftSeq, sh_unShiftSeq, sh_ls_est);
        } else {
            const uint32_t MAX_LEN = 2;
            windowedChEstFilterNoDftSOfdmKernel<
                TStorage, TDataRx, TCompute, N_DMRS_GRIDS_PER_PRB,
                PRB_IN_PER_CLUSTER, INTERP_PRB_OUT_PER_CLUSTER, MAX_LEN>(
                    pStatDescr, pDynDescr, sh_shiftSeq, sh_unShiftSeq, sh_ls_est);
        }
    }
} //chEstFilterNoDftSOfdmDispatchKernel

template <typename TStorage,
           typename TDataRx,
           typename TCompute,
           //uint32_t N_LAYERS,                          // # of layers (# of cols in H matrix) ** may be larger than the actual number of layers in the group
           uint32_t N_DMRS_GRIDS_PER_PRB,              // # of DMRS grids per PRB (2 or 3)
           uint32_t N_DMRS_PRB_IN_PER_CLUSTER,
           uint32_t N_DMRS_INTERP_PRB_OUT_PER_CLUSTER,
           uint32_t N_DMRS_SYMS>                       // # of consecutive DMRS symbols (1 or 2)
static __device__ void
windowedChEstFilterPrgNoDftSOfdmKernel(
    puschRxChEstStatDescr_t* pStatDescr, puschRxChEstDynDescr_t* pDynDescr,
    typename complex_from_scalar<TCompute>::type *sh_shiftSeq,
    typename complex_from_scalar<TCompute>::type *sh_unShiftSeq,
    typename complex_from_scalar<TCompute>::type *sh_ls_est,
    uint16_t nPrb,
    uint16_t startPrb,
    uint16_t startPrb_out)
{
     typedef typename complex_from_scalar<TCompute>::type TComplexCompute;
     typedef typename complex_from_scalar<TStorage>::type TComplexStorage;

     // We only currently support type 1 DMRS grids, which have DMRS tones in every other frequency bin.
     constexpr uint32_t N_DMRS_TYPE1_GRIDS_PER_PRB = 2;
     static_assert(N_DMRS_GRIDS_PER_PRB == N_DMRS_TYPE1_GRIDS_PER_PRB, "Kernel only supports type 1 DMRS grids");
     constexpr uint32_t N_DMRS_TONE_STRIDE = 2;

     thread_block const& block = this_thread_block();

     puschRxChEstStatDescr_t& statDescr = *pStatDescr;
     puschRxChEstDynDescr_t& dynDescr = *pDynDescr;

     // UE group processed by this thread block
     const uint32_t UE_GRP_IDX = dynDescr.hetCfgUeGrpMap[blockIdx.z];
     // PRB cluster being processed by this thread block
     const uint32_t PRB_CLUSTER_IDX = blockIdx.x;

     // Channel estimation expands tones in a DMRS grid (4 or 6, given by N_DMRS_GRID_TONES_PER_PRB) into a full PRB
     constexpr uint32_t N_DMRS_INTERP_TONES_PER_GRID = N_TONES_PER_PRB;

     // Per grid tone counts present in input and output PRB clusters. These tones counts are expected to be equal
     constexpr uint32_t N_TOTAL_DMRS_INTERP_GRID_TONES_PER_CLUSTER = N_DMRS_INTERP_TONES_PER_GRID * N_DMRS_INTERP_PRB_OUT_PER_CLUSTER;

     const uint32_t THREAD_IDX = threadIdx.x;
     const uint32_t LAYER_IDX = THREAD_IDX / N_TOTAL_DMRS_INTERP_GRID_TONES_PER_CLUSTER;
     const uint32_t THREAD_IDX_MOD_LAYER = THREAD_IDX - LAYER_IDX * (N_TOTAL_DMRS_INTERP_GRID_TONES_PER_CLUSTER);

     const uint32_t tid = threadIdx.x;
     const uint32_t nthreads = blockDim.x;

     cuphyPuschRxUeGrpPrms_t& drvdUeGrpPrms = dynDescr.pDrvdUeGrpPrms[UE_GRP_IDX];
     ///////////////////const uint16_t nPrb   = drvdUeGrpPrms.nPrb;
     const uint16_t nRxAnt = drvdUeGrpPrms.nRxAnt;
     uint8_t*       OCCIdx = drvdUeGrpPrms.OCCIdx;
     const uint8_t   nDmrsCdmGrpsNoData = drvdUeGrpPrms.nDmrsCdmGrpsNoData;
     // BS antenna being processed by this thread block. The y dimension of the CTA is an rx antenna
     // blocking dimension.
     const uint32_t BS_ANT_IDX = blockDim.y * blockIdx.y + threadIdx.y;
     ///////////////////////////////const uint32_t N_PRB_CLUSTERS_PER_BS_ANT = div_round_up(nPrb, static_cast<uint16_t>(N_DMRS_INTERP_PRB_OUT_PER_CLUSTER));
     if(BS_ANT_IDX >= nRxAnt) return;

     //////////////////////////const uint32_t N_PRB_CLUSTERS = N_PRB_CLUSTERS_PER_BS_ANT;
     const uint16_t  nLayers            = drvdUeGrpPrms.nLayers;
     const uint8_t   chEstTimeInst      = dynDescr.chEstTimeInst;

     constexpr uint32_t N_DMRS_GRID_TONES_PER_PRB = N_TONES_PER_PRB / N_DMRS_GRIDS_PER_PRB;
     constexpr uint32_t N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER = N_DMRS_GRID_TONES_PER_PRB * N_DMRS_PRB_IN_PER_CLUSTER;

     // The filter coefficients are different depending on PRB count and PRB location (i.e.
     // edge PRBs have different filter coefficients than central PRBs)
     tensor_ref<const TCompute> tFreqInterpCoefs = [nPrb, &statDescr]() -> auto {
        if (nPrb <= 3) {
            return tensor_ref<const TCompute>{ statDescr.tPrmFreqInterpCoefsSmall.pAddr, statDescr.tPrmFreqInterpCoefsSmall.strides };
        } else if (nPrb >= 8 && nPrb % 4 == 0) {
            return tensor_ref<const TCompute>{ statDescr.tPrmFreqInterpCoefs.pAddr, statDescr.tPrmFreqInterpCoefs.strides };
        } else {
            return tensor_ref<const TCompute>{ statDescr.tPrmFreqInterpCoefs4.pAddr, statDescr.tPrmFreqInterpCoefs4.strides };
        }
     }();

     const uint32_t filtIdx = [nPrb, PRB_CLUSTER_IDX]() -> uint32_t {
        if (nPrb == 0) {
            return 0;
        } else if (nPrb <= 3) {
            return nPrb - 1;
        } else {
            // Interpolation filter indices for middle and edge PRBs
            constexpr uint32_t LOWER_EDGE_INTERP_FILT_IDX = 1;
            constexpr uint32_t UPPER_EDGE_INTERP_FILT_IDX = 2;
            if (PRB_CLUSTER_IDX%2 == 0) {
                return LOWER_EDGE_INTERP_FILT_IDX;
            } else {
                return UPPER_EDGE_INTERP_FILT_IDX;
            }
        }
     }();

     // Determine first PRB in the cluster being processed
     uint32_t prbClusterStartIdx = startPrb;

    // When populating the LS intermediate data structure, different values of DMRS_GRID_IDX
    // map to different layers rather than an offset in the tone subcarrier frequency.
    const uint32_t toneOffset = prbClusterStartIdx * N_TONES_PER_PRB;
    const uint32_t DMRS_LS_EST_TONE_OFFSET = toneOffset / N_DMRS_TONE_STRIDE;

    const uint32_t NH_IDX = chEstTimeInst;

    tensor_ref<const TComplexCompute> tInfoDmrsLSEst(drvdUeGrpPrms.tInfoDmrsLSEst.pAddr, drvdUeGrpPrms.tInfoDmrsLSEst.strides);
    const TComplexCompute *dmrsLSEst = tInfoDmrsLSEst.pAddr + tInfoDmrsLSEst.offset(DMRS_LS_EST_TONE_OFFSET, 0, BS_ANT_IDX, NH_IDX);
    const uint32_t toneStride = tInfoDmrsLSEst.strides[0];
    const uint32_t layerStride = tInfoDmrsLSEst.strides[1];

    const int nIter = N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER * nLayers;
    for (int i = tid; i < nIter; i += nthreads) {
        const int layer = i / N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER;
        const int tone = i % N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER;
        sh_ls_est[i] = dmrsLSEst[tone * toneStride + layer * layerStride] * sh_shiftSeq[tone];
    }

     __syncthreads();

     // All block synchronizations have completed, so threads that will not generate outputs can now exit.
     if (LAYER_IDX >= nLayers) {
        return;
     }

     const uint32_t DMRS_INTERP_TONE_IDX = THREAD_IDX_MOD_LAYER % N_TOTAL_DMRS_INTERP_GRID_TONES_PER_CLUSTER;

     constexpr uint32_t HALF_N_EDGE_TONES = N_TONES_PER_PRB * (N_DMRS_PRB_IN_PER_CLUSTER - N_DMRS_INTERP_PRB_OUT_PER_CLUSTER) / 2;
     uint32_t CLUSTER_INTERP_TONE_IDX = DMRS_INTERP_TONE_IDX + HALF_N_EDGE_TONES;
     if(PRB_CLUSTER_IDX%2==0) 
         CLUSTER_INTERP_TONE_IDX = CLUSTER_INTERP_TONE_IDX - HALF_N_EDGE_TONES;     // Lower edge
     else
         CLUSTER_INTERP_TONE_IDX = CLUSTER_INTERP_TONE_IDX + HALF_N_EDGE_TONES;     // Upper edge 

     // Absolute index of interpolated tone produced by this thread
     uint32_t INTERP_DMRS_ABS_TONE_IDX = startPrb_out * N_TONES_PER_PRB + DMRS_INTERP_TONE_IDX;

     const uint32_t MAX_INTERP_ABS_TONE = drvdUeGrpPrms.nPrb * N_TONES_PER_PRB - 1;

     if (INTERP_DMRS_ABS_TONE_IDX <= MAX_INTERP_ABS_TONE) {
        const TCompute scaling = (nDmrsCdmGrpsNoData == 1) ? static_cast<TCompute>(1.414213562373095f) : 1.0f;
        const uint32_t gridShiftIdx = get_inter_dmrs_grid_freq_shift_idx(N_DMRS_GRIDS_PER_PRB, (OCCIdx[LAYER_IDX] >> 2) & 0x1);

        tensor_ref<TComplexStorage> tHEst(drvdUeGrpPrms.tInfoHEst.pAddr, drvdUeGrpPrms.tInfoHEst.strides);
        TComplexStorage *HEst = tHEst.pAddr + tHEst.offset(BS_ANT_IDX, 0, INTERP_DMRS_ABS_TONE_IDX, NH_IDX);
        const uint32_t estLayerStride = tHEst.strides[1];

        const TCompute *coefs = tFreqInterpCoefs.pAddr + tFreqInterpCoefs.offset(DMRS_INTERP_TONE_IDX+gridShiftIdx, 0, filtIdx);
        const int coefToneStride = tFreqInterpCoefs.strides[1];

        TComplexCompute accum { 0.0f, 0.0f };
        for (int j = 0; j < N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER; j++) {
            const TComplexCompute prod = sh_ls_est[LAYER_IDX*N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER+j] *
                coefs[j * coefToneStride];
            accum += prod;
        }
        
        accum *= (sh_unShiftSeq[CLUSTER_INTERP_TONE_IDX+gridShiftIdx] * scaling);
        HEst[LAYER_IDX * estLayerStride] = type_convert<TComplexStorage>(accum);
     }
}

template <typename TStorage,
          typename TDataRx,
          typename TCompute,
          uint32_t N_DMRS_GRIDS_PER_PRB>              // # of DMRS grids per PRB (2 or 3)
static __global__ void
chEstFilterPrgNoDftSOfdmDispatchKernel(puschRxChEstStatDescr_t* pStatDescr, puschRxChEstDynDescr_t* pDynDescr)
{
    typedef typename complex_from_scalar<TCompute>::type TComplexCompute;

    thread_block block = this_thread_block();

    const puschRxChEstDynDescr_t& dynDescr = *pDynDescr;

    // UE group processed by this thread block
    const uint32_t UE_GRP_IDX = dynDescr.hetCfgUeGrpMap[blockIdx.z];
    cuphyPuschRxUeGrpPrms_t& drvdUeGrpPrms = dynDescr.pDrvdUeGrpPrms[UE_GRP_IDX];
    
    uint32_t PRB_CLUSTER_IDX = blockIdx.x;
    uint16_t nPrb = drvdUeGrpPrms.nPrb;
    uint16_t prgSize = drvdUeGrpPrms.prgSize;
    uint16_t nLayers = drvdUeGrpPrms.nLayers;
    
    if(prgSize > nPrb)
    {
        prgSize = nPrb;
    }
    
    uint16_t nPrbClusters = 0;
    uint16_t nThreads = 0;
    uint16_t startPrb = 0;
    uint16_t startPrb_out = 0;
    if(prgSize < 4)
    {
        nPrbClusters = div_round_up(nPrb, static_cast<uint16_t>(prgSize));
    }
    else if(prgSize == 4)
    {
        uint16_t quotient = (nPrb>>2);
        uint16_t remainder = (nPrb & 0x3);
        nPrbClusters = 2 * quotient;
        if(remainder>0)
        {
            nPrbClusters += 1;  
        }
        
        if(remainder == 3)
        {
            nThreads = 3 * N_TONES_PER_PRB * nLayers;
        }
        else
        {
            nThreads = 2 * N_TONES_PER_PRB * nLayers;
        }
    }
    
    if(PRB_CLUSTER_IDX>=nPrbClusters)
    {
        return;
    }
    
    const uint32_t THREAD_IDX = threadIdx.x;
    if(PRB_CLUSTER_IDX == nPrbClusters-1)
    {
        if(prgSize <4 )
        {
            nPrb = nPrb - (nPrbClusters-1)*prgSize;
            startPrb = (nPrbClusters-1)*prgSize;
            startPrb_out = startPrb;
        }
        else if(prgSize==4)
        {
            nPrb = nPrb - (nPrbClusters-1)*2;
            startPrb = (nPrbClusters-1)*2;
            startPrb_out = startPrb;
        }
        
        nThreads = nPrb * N_TONES_PER_PRB * nLayers;
        if(THREAD_IDX>=nThreads)
            return;
    }
    else
    {
        nPrb = prgSize;
        if(prgSize<4)
        {
            nThreads = prgSize * N_TONES_PER_PRB * nLayers;
            if(THREAD_IDX>=nThreads)
                return;
            
            startPrb = PRB_CLUSTER_IDX*prgSize;
            startPrb_out = startPrb;
        }
        else if(prgSize == 4)
        {
            nThreads = 2 * N_TONES_PER_PRB * nLayers;
            if(THREAD_IDX>=nThreads)
                return;
                
            startPrb = (PRB_CLUSTER_IDX>>1)*4;
            startPrb_out = PRB_CLUSTER_IDX*2;
        }
    }
    
    constexpr uint32_t MAX_N_DMRS_PRB_IN_PER_CLUSTER = 8;
    constexpr uint32_t N_DMRS_GRID_TONES_PER_PRB = N_TONES_PER_PRB / N_DMRS_GRIDS_PER_PRB;
    constexpr uint32_t MAX_N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER = N_DMRS_GRID_TONES_PER_PRB * MAX_N_DMRS_PRB_IN_PER_CLUSTER;

    // Worst-case shared memory allocations
    constexpr uint32_t MAX_NUM_PRBS_PER_FILTER = 8;
    __shared__ TComplexCompute sh_shiftSeq[MAX_NUM_PRBS_PER_FILTER*(N_TONES_PER_PRB/N_DMRS_GRIDS_PER_PRB)];
    __shared__ TComplexCompute sh_unShiftSeq[(MAX_NUM_PRBS_PER_FILTER*N_TONES_PER_PRB)+1];
    extern __shared__ TComplexCompute sh_ls_est_full[];

    const uint32_t tid = block.thread_rank();

    const uint8_t mu = drvdUeGrpPrms.mu;
    // Subcarrier spacing df is 15 kHz * 2^mu for mu in [0, 6]
    const float df = 15.0e3f * static_cast<float>(1U << mu);

    __shared__ TCompute sh_delay_mean;
    chEstDelayShiftReduction<TCompute>(
        sh_delay_mean,
        dynDescr,
        drvdUeGrpPrms,
        block,
        drvdUeGrpPrms.dmrsAddlnPos + 1,
        df,
        tid);

    
    const uint32_t nPrbInPerCluster = [nPrb]() -> uint32_t {
        if (nPrb <= 3) {
            return nPrb;
        } else if (nPrb >= 8 && nPrb % 4 == 0) {
            return 8;
        } else {
            return 4;
        }
    }();

    // Calculate the shift/unshift coefficients, which depend on sh_delay_mean
    const float partial_arg = df * 2.0f * static_cast<float>(M_PI) * sh_delay_mean;
    for (uint32_t i = tid; i < nPrbInPerCluster*(N_TONES_PER_PRB/N_DMRS_GRIDS_PER_PRB); i += nThreads) {
        const float f_dmrs = partial_arg * (2 * i);
        sincosf(f_dmrs, &sh_shiftSeq[i].y, &sh_shiftSeq[i].x);
    }

    for (uint32_t i = tid; i < (nPrbInPerCluster*N_TONES_PER_PRB)+1; i += nThreads) {
        const float f_data = -1.0f * partial_arg * (static_cast<float>(i)-1.0f);
        sincosf(f_data, &sh_unShiftSeq[i].y, &sh_unShiftSeq[i].x);
    }

    __syncthreads();

    // Logically, the smem data is stored as [antenna][layer][tone]
    TComplexCompute *sh_ls_est = sh_ls_est_full + threadIdx.y * nLayers * MAX_N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER;

    const uint8_t maxLen = drvdUeGrpPrms.dmrsMaxLen;
    if (nPrb <= 3) {
        if (maxLen == 1) {
            const uint32_t MAX_LEN = 1;
            switch (nPrb) {
                case 1:
                    windowedChEstFilterPrgNoDftSOfdmKernel<
                        TStorage, TDataRx, TCompute, N_DMRS_GRIDS_PER_PRB, 1, 1, MAX_LEN>(
                            pStatDescr, pDynDescr, sh_shiftSeq, sh_unShiftSeq, sh_ls_est, nPrb, startPrb, startPrb_out);
                    break;
                case 2:
                    windowedChEstFilterPrgNoDftSOfdmKernel<
                        TStorage, TDataRx, TCompute, N_DMRS_GRIDS_PER_PRB, 2, 2, MAX_LEN>(
                            pStatDescr, pDynDescr, sh_shiftSeq, sh_unShiftSeq, sh_ls_est, nPrb, startPrb, startPrb_out);
                    break;
                case 3:
                    windowedChEstFilterPrgNoDftSOfdmKernel<
                        TStorage, TDataRx, TCompute, N_DMRS_GRIDS_PER_PRB, 3, 3, MAX_LEN>(
                            pStatDescr, pDynDescr, sh_shiftSeq, sh_unShiftSeq, sh_ls_est, nPrb, startPrb, startPrb_out);
                    break;
            }
        } else {
            const uint32_t MAX_LEN = 2;
            switch (nPrb) {
                case 1:
                    windowedChEstFilterPrgNoDftSOfdmKernel<
                        TStorage, TDataRx, TCompute, N_DMRS_GRIDS_PER_PRB, 1, 1, MAX_LEN>(
                            pStatDescr, pDynDescr, sh_shiftSeq, sh_unShiftSeq, sh_ls_est, nPrb, startPrb, startPrb_out);
                    break;
                case 2:
                    windowedChEstFilterPrgNoDftSOfdmKernel<
                        TStorage, TDataRx, TCompute, N_DMRS_GRIDS_PER_PRB, 2, 2, MAX_LEN>(
                            pStatDescr, pDynDescr, sh_shiftSeq, sh_unShiftSeq, sh_ls_est, nPrb, startPrb, startPrb_out);
                    break;
                case 3:
                    windowedChEstFilterPrgNoDftSOfdmKernel<
                        TStorage, TDataRx, TCompute, N_DMRS_GRIDS_PER_PRB, 3, 3, MAX_LEN>(
                            pStatDescr, pDynDescr, sh_shiftSeq, sh_unShiftSeq, sh_ls_est, nPrb, startPrb, startPrb_out);
                    break;
            }
        }
    } else {
        const uint32_t PRB_IN_PER_CLUSTER = 4;
        const uint32_t INTERP_PRB_OUT_PER_CLUSTER = 2;
        if (maxLen == 1) {
            const uint32_t MAX_LEN = 1;
            windowedChEstFilterPrgNoDftSOfdmKernel<
                TStorage, TDataRx, TCompute, N_DMRS_GRIDS_PER_PRB,
                PRB_IN_PER_CLUSTER, INTERP_PRB_OUT_PER_CLUSTER, MAX_LEN>(
                    pStatDescr, pDynDescr, sh_shiftSeq, sh_unShiftSeq, sh_ls_est, nPrb, startPrb, startPrb_out);
        } else {
            const uint32_t MAX_LEN = 2;
            windowedChEstFilterPrgNoDftSOfdmKernel<
                TStorage, TDataRx, TCompute, N_DMRS_GRIDS_PER_PRB,
                PRB_IN_PER_CLUSTER, INTERP_PRB_OUT_PER_CLUSTER, MAX_LEN>(
                    pStatDescr, pDynDescr, sh_shiftSeq, sh_unShiftSeq, sh_ls_est, nPrb, startPrb, startPrb_out);
        }
    }
} //chEstFilterPrgNoDftSOfdmDispatchKernel

template <typename TStorage,
           typename TDataRx,
           typename TCompute,
           uint32_t N_DMRS_GRIDS_PER_PRB,              // # of DMRS grids per PRB (2 or 3)
           uint32_t N_DMRS_SYMS,                       // # of consecutive DMRS symbols (1 or 2)
           uint16_t CUPHY_PUSCH_RX_CH_EST_DELAY_EST_PRB_CLUSTER_SIZE>                       
 static __global__ void
 windowedChEstPreNoDftSOfdmKernel(puschRxChEstStatDescr_t* pStatDescr, puschRxChEstDynDescr_t* pDynDescr)
 {
     puschRxChEstDynDescr_t& dynDescr = *pDynDescr;

     // UE group processed by this thread block
     const uint32_t UE_GRP_IDX = dynDescr.hetCfgUeGrpMap[blockIdx.z];
     // PRB cluster being processed by this thread block
     const uint32_t PRB_CLUSTER_IDX = blockIdx.x;
     // BS antenna being processed by this thread block
     const uint32_t BS_ANT_IDX = blockIdx.y;

     // Early exit check
     // The grid is sized to process the max # of PRB clusters in a given heterogenous config. Exit if the PRB cluster to be
     // processed by this thread block does not exist in the UE group
     cuphyPuschRxUeGrpPrms_t& drvdUeGrpPrms = dynDescr.pDrvdUeGrpPrms[UE_GRP_IDX];
     const uint16_t nPrb   = drvdUeGrpPrms.nPrb;
     const uint16_t nRxAnt = drvdUeGrpPrms.nRxAnt;
     const uint32_t N_PRB_CLUSTERS_PER_BS_ANT = div_round_up(nPrb, static_cast<uint16_t>(CUPHY_PUSCH_RX_CH_EST_DELAY_EST_PRB_CLUSTER_SIZE));
     if((PRB_CLUSTER_IDX >= N_PRB_CLUSTERS_PER_BS_ANT) || (BS_ANT_IDX >= nRxAnt)) return;

     //--------------------------------------------------------------------------------------------------------
     // Setup local parameters based on descriptor
     const uint16_t  slotNum            = drvdUeGrpPrms.slotNum;
     const uint8_t   chEstTimeInst      = dynDescr.chEstTimeInst;
     uint8_t*        OCCIdx             = drvdUeGrpPrms.OCCIdx;
     const uint16_t  nLayers            = drvdUeGrpPrms.nLayers;
     const uint8_t   dmrsMaxLen         = drvdUeGrpPrms.dmrsMaxLen;
     const uint8_t   scid               = drvdUeGrpPrms.scid;
     // We only apply DFT-s-OFDM if both the enableDftSOfdm and enableTfPrcd flags are
     // set. enableDftSOfdm is a global setting and enableTfPrcd is finer-grained.
     const bool      enableTfPrcd       = (drvdUeGrpPrms.enableTfPrcd != 0) && (drvdUeGrpPrms.enableDftSOfdm != 0);
     // Pointer to DMRS symbol used for channel estimation (single-symbol if maxLen = 1, double-symbol if maxLen = 2)
     const uint8_t* const pDmrsSymPos   = &drvdUeGrpPrms.dmrsSymLoc[chEstTimeInst*dmrsMaxLen];
     const uint16_t  startPrb = drvdUeGrpPrms.startPrb;
     const uint16_t  dmrsScId = drvdUeGrpPrms.dmrsScrmId;
     //--------------------------------------------------------------------------------------------------------
     typedef typename complex_from_scalar<TCompute>::type TComplexCompute;
     typedef typename complex_from_scalar<TDataRx>::type TComplexDataRx;
     typedef typename complex_from_scalar<TStorage>::type TComplexStorage;

     tensor_ref<const TComplexDataRx> tDataRx           (drvdUeGrpPrms.tInfoDataRx.pAddr  , drvdUeGrpPrms.tInfoDataRx.strides);// (NF, ND, N_BS_ANTS)
     tensor_ref<TComplexCompute>      tInfoDmrsLSEst    (drvdUeGrpPrms.tInfoDmrsLSEst.pAddr, drvdUeGrpPrms.tInfoDmrsLSEst.strides);

     //--------------------------------------------------------------------------------------------------------
     // Dimensions and indices

     // Estimates of H in time supported
     const uint32_t NH_IDX = chEstTimeInst;

     // We currently only support type 1 grids
     constexpr uint32_t N_DMRS_TYPE1_GRIDS_PER_PRB = 2;
     static_assert(N_DMRS_TYPE1_GRIDS_PER_PRB == N_DMRS_GRIDS_PER_PRB, "DMRS kernel currently only supports type 1 grid");
     constexpr uint32_t N_DMRS_TONE_STRIDE = 2;

     constexpr uint32_t N_DMRS_GRID_TONES_PER_PRB = N_TONES_PER_PRB / N_DMRS_GRIDS_PER_PRB;
     // Per grid tone counts present in input and output PRB clusters. These tones counts are expected to be equal
     constexpr uint32_t N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER        = N_DMRS_GRID_TONES_PER_PRB * CUPHY_PUSCH_RX_CH_EST_DELAY_EST_PRB_CLUSTER_SIZE;

     // Ensure configured symbol count does not exceed max value prescribed by spec
     static_assert((N_DMRS_SYMS <= N_MAX_DMRS_SYMS), "DMRS symbol count exceeds max value");

     // DMRS descrambling
     constexpr uint32_t N_DMRS_DESCR_BITS_PER_TONE    = 2; // 1bit for I and 1 bit for Q
     constexpr uint32_t N_DMRS_DESCR_BITS_PER_CLUSTER = N_DMRS_DESCR_BITS_PER_TONE * N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER;
     // # of DMRS descrambler bits generated at one time
     constexpr uint32_t N_DMRS_DESCR_BITS_GEN = 32;
     // Round up to the next multiple of N_DMRS_DESCR_BITS_GEN plus 1 (+1 because DMRS_DESCR_PRB_CLUSTER_START_BIT_OFFSET
     // may be large enough to spill the descrambler bits to the next word)
     constexpr uint32_t N_DMRS_DESCR_WORDS =
         ((N_DMRS_DESCR_BITS_PER_CLUSTER + N_DMRS_DESCR_BITS_GEN - 1) / N_DMRS_DESCR_BITS_GEN) + 1;
     // round_up_to_next(N_DMRS_DESCR_BITS_PER_CLUSTER, N_DMRS_DESCR_BITS_GEN) + 1;
     // Absolute index of descrambling + shift sequence element

     // Per UE group descrambling ID
     uint16_t dmrsScramId = dmrsScId;

     const uint32_t THREAD_IDX = threadIdx.x;

     const uint32_t prbClusterStartIdx = PRB_CLUSTER_IDX * CUPHY_PUSCH_RX_CH_EST_DELAY_EST_PRB_CLUSTER_SIZE + startPrb;
     if (prbClusterStartIdx >= startPrb+nPrb) {
        return;
     }

     const uint32_t DMRS_DESCR_SHIFT_SEQ_START_IDX = prbClusterStartIdx * N_DMRS_GRID_TONES_PER_PRB;

     // Index of DMRS grid in which the DMRS tone being processed by this thread resides
     // Note: although the grid index is calculated using total number of DMRS grid tones in the cluster, its
     // used as an index in context of both input DMRS tones and interpolated DMRS tones under the assumption:
     // N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER == N_TOTAL_DMRS_INTERP_GRID_TONES_PER_CLUSTER
     const uint32_t DMRS_GRID_IDX = THREAD_IDX / N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER;

     // This index calculation intends to divvy up threads in the thread block for processing as follows:
     // the first group of N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER to process the first DMRS grid, the second group
     // to process the second DMRS grid and so on
     // Relative index of DMRS tone (within a DMRS grid) being processed by this thread
     const uint32_t DMRS_TONE_IDX        = THREAD_IDX % N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER;

     // Absolute index of DMRS tone within the input OFDM symbol (used as index when loading tone from OFDM
     // symbol)
     const uint32_t DMRS_ABS_TONE_IDX = prbClusterStartIdx * N_TONES_PER_PRB + N_DMRS_TONE_STRIDE * DMRS_TONE_IDX + DMRS_GRID_IDX; // FIXME: handle type 2 grid

     const uint32_t lastPrbThisCluster = min(startPrb+nPrb-1, prbClusterStartIdx + CUPHY_PUSCH_RX_CH_EST_DELAY_EST_PRB_CLUSTER_SIZE - 1);
     const bool isValidTone = (DMRS_ABS_TONE_IDX / N_TONES_PER_PRB) <= lastPrbThisCluster;
     const bool nextToneIsValid = ((DMRS_ABS_TONE_IDX+N_DMRS_TONE_STRIDE) / N_TONES_PER_PRB) <= lastPrbThisCluster; // FIXME: handle type 2 grid
     bool isValidR1Cal = true;
     uint8_t enablePerPrgChEst = drvdUeGrpPrms.enablePerPrgChEst;
     uint16_t prgSize = drvdUeGrpPrms.prgSize;

     const uint32_t tmpGridIdx = DMRS_GRID_IDX > 1 ? 1 : DMRS_GRID_IDX;
     const uint8_t  activeTOCCBmsk     = drvdUeGrpPrms.activeTOCCBmsk[tmpGridIdx];
     const uint8_t  activeFOCCBmsk     = drvdUeGrpPrms.activeFOCCBmsk[tmpGridIdx];
     const uint32_t N_DMRS_SYMS_FOCC = activeFOCCBmsk == 3 ? 2 : 1;

     // Section 5.2.1 in 3GPP TS 38.211
     // The fast-forward of 1600 prescribed by spec is already baked into the gold sequence generator
     constexpr uint32_t DMRS_DESCR_FF = 0; // 1600;

     // First descrambler bit index needed by this thread block
     // Note:The DMRS scrambling sequence is the same for all the DMRS grids. There are 2 sequences one for
     // scid 0 and other for scid 1 but the same sequences is reused for all DMRS grids
     const uint32_t DMRS_DESCR_PRB_CLUSTER_START_BIT =
         DMRS_DESCR_FF + (DMRS_DESCR_SHIFT_SEQ_START_IDX * N_DMRS_DESCR_BITS_PER_TONE);

     // The descrambling sequence generator outputs 32 descrambler bits at a time. Thus, compute the earliest
     // multiple of 32 bits which contains the descrambler bit of the first tone in the PRB cluster as the
     // start index
     const uint32_t DMRS_DESCR_GEN_ALIGNED_START_BIT =
         (DMRS_DESCR_PRB_CLUSTER_START_BIT / N_DMRS_DESCR_BITS_GEN) * N_DMRS_DESCR_BITS_GEN;
     // Offset to descrambler bit of the first tone in the PRB cluster
     const uint32_t DMRS_DESCR_PRB_CLUSTER_START_BIT_OFFSET =
         DMRS_DESCR_PRB_CLUSTER_START_BIT - DMRS_DESCR_GEN_ALIGNED_START_BIT;

     // DMRS descrambling bits generated correspond to subcarriers across frequency
     // e.g. 2 bits for tone0(grid 0) | 2 bits for tone1(grid 1) | 2 bits for tone 2(grid 0) | 2 bits for tone 3(grid 1) | ...
     const uint32_t DMRS_TONE_DESCR_BIT_IDX = DMRS_DESCR_PRB_CLUSTER_START_BIT_OFFSET +
                                              (DMRS_TONE_IDX * N_DMRS_DESCR_BITS_PER_TONE);
     const uint32_t DMRS_DESCR_SEQ_RD_BIT_IDX  = DMRS_TONE_DESCR_BIT_IDX % N_DMRS_DESCR_BITS_GEN;
     const uint32_t DMRS_DESCR_SEQ_RD_WORD_IDX = DMRS_TONE_DESCR_BIT_IDX / N_DMRS_DESCR_BITS_GEN;

     const uint32_t DMRS_DESCR_SEQ_WR_WORD_IDX = THREAD_IDX % N_DMRS_DESCR_WORDS;
     const uint32_t DMRS_DESCR_SEQ_WR_SYM_IDX  = THREAD_IDX / N_DMRS_DESCR_WORDS;

     // Data layouts:
     // Global memory read into shared memory
     // N_DMRS_TONES x N_DMRS_SYMS -> N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER x N_DMRS_SYMS x N_DMRS_GRIDS_PER_PRB

     // tOCC removal
     // N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER x N_DMRS_SYMS x N_DMRS_GRIDS_PER_PRB ->
     // N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER x N_DMRS_SYMS_TOCC x N_DMRS_GRIDS_PER_PRB

     // fOCC removal
     // N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER x N_DMRS_SYMS_TOCC x N_DMRS_GRIDS_PER_PRB ->
     // N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER x N_DMRS_SYMS_FOCC x NUM_DMRS_SYMS_TOCC x N_DMRS_GRIDS_PER_PRB

     // Interpolation
     // N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER x N_DMRS_SYMS_FOCC x NUM_DMRS_SYMS_TOCC x N_DMRS_GRIDS_PER_PRB ->
     // N_TOTAL_DMRS_INTERP_GRID_TONES_PER_CLUSTER x N_DMRS_SYMS_FOCC x NUM_DMRS_SYMS_TOCC x N_DMRS_GRIDS_PER_PRB =
     // N_TOTAL_DMRS_INTERP_GRID_TONES_PER_CLUSTER x N_LAYERS x N_DMRS_GRIDS_PER_PRB

     //--------------------------------------------------------------------------------------------------------
     // Allocate shared memory

     constexpr uint32_t N_SMEM_ELEMS = N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER * N_DMRS_SYMS_OCC * N_DMRS_GRIDS_PER_PRB;

     __shared__ TComplexCompute smemBlk[N_SMEM_ELEMS];
     block_3D<TComplexCompute*, N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER, N_DMRS_SYMS_OCC, N_DMRS_GRIDS_PER_PRB> shH(&smemBlk[0]);

     __shared__ uint32_t descrWords[N_DMRS_SYMS][N_DMRS_DESCR_WORDS];

     if (enableTfPrcd == 0)
     {
        // Compute the descsrambler sequence
        const uint32_t TWO_POW_17 = bit(17);

        if(DMRS_DESCR_SEQ_WR_SYM_IDX < N_DMRS_SYMS)
        {
            uint32_t symIdx = pDmrsSymPos[DMRS_DESCR_SEQ_WR_SYM_IDX];

            // see 38.211 section 6.4.1.1.1.1
            uint32_t cInit = TWO_POW_17 * (slotNum * OFDM_SYMBOLS_PER_SLOT + symIdx + 1) * (2 * dmrsScramId + 1) + (2 * dmrsScramId) + scid;
            cInit &= ~bit(31);

            descrWords[DMRS_DESCR_SEQ_WR_SYM_IDX][DMRS_DESCR_SEQ_WR_WORD_IDX] =
                gold32(cInit, (DMRS_DESCR_GEN_ALIGNED_START_BIT + DMRS_DESCR_SEQ_WR_WORD_IDX * N_DMRS_DESCR_BITS_GEN));
        }
     }

     // To ensure coalesced reads, input DMRS tones are read preserving input order but swizzled while writing
     // to shared memory. Thus each thread may not process the same tone which it wrote to shared memory
     thread_block const& thisThrdBlk = this_thread_block();
     thread_block_tile<WARP_SIZE> tile = tiled_partition<WARP_SIZE>(thisThrdBlk);
     thisThrdBlk.sync();

     //--------------------------------------------------------------------------------------------------------
     // Apply de-scrambling + delay domain centering sequence for tone index processed by this thread across all
     // DMRS symbols
     const TCompute RECIPROCAL_SQRT2 = 0.7071068f;
     const TCompute SQRT2            = 1.41421356f;

     TComplexCompute pilots[N_TOCC]{};
     if (isValidTone && enableTfPrcd == 0)
     {
 #pragma unroll
        for(uint32_t i = 0; i < N_DMRS_SYMS; ++i)
        {
            int8_t descrIBit = (descrWords[i][DMRS_DESCR_SEQ_RD_WORD_IDX] >> DMRS_DESCR_SEQ_RD_BIT_IDX) & 0x1;
            int8_t descrQBit = (descrWords[i][DMRS_DESCR_SEQ_RD_WORD_IDX] >> (DMRS_DESCR_SEQ_RD_BIT_IDX + 1)) & 0x1;

            const TComplexCompute descrCode =
                cuConj(cuGet<TComplexCompute>((1 - 2 * descrIBit) * RECIPROCAL_SQRT2, (1 - 2 * descrQBit) * RECIPROCAL_SQRT2));
            const TComplexCompute data = type_convert<TComplexCompute>(tDataRx(DMRS_ABS_TONE_IDX, pDmrsSymPos[i], BS_ANT_IDX));
            pilots[i] = descrCode * data;
        }
     }
     else if (isValidTone && enableTfPrcd == 1)
     {
 #pragma unroll
        for(uint32_t i = 0; i < N_DMRS_SYMS; ++i)
        {
            uint16_t M_ZC  = N_DMRS_GRID_TONES_PER_PRB * nPrb; //different from DMRS_ABS_TONE_IDX
            int u = 0;
            int v = 0;

            if(drvdUeGrpPrms.optionalDftSOfdm)
            {
                u = (int)drvdUeGrpPrms.lowPaprGroupNumber;
                v = (int)drvdUeGrpPrms.lowPaprSequenceNumber;
            }
            else
            {
                const uint32_t puschIdentity          = drvdUeGrpPrms.puschIdentity;
                const uint8_t  groupOrSequenceHopping = drvdUeGrpPrms.groupOrSequenceHopping;
                const uint8_t  N_symb_slot            = drvdUeGrpPrms.N_symb_slot;

                int f_gh       = 0;

                if(groupOrSequenceHopping==1)
                {
                    uint32_t cInit = floor(puschIdentity/30);
                    for(int m = 0; m < 8; m++)
                    {
                        uint32_t idxSeq = 8 * (slotNum * N_symb_slot + pDmrsSymPos[i]) + m;
                        f_gh = f_gh + ((gold32(cInit, idxSeq) >> (idxSeq % 32)) & 0x1) * (1 << m);
                    }
                    f_gh = f_gh % 30;
                }
                else if(groupOrSequenceHopping==2)
                {
                    if(M_ZC > 6 * N_TONES_PER_PRB)
                    {
                        uint32_t idxSeq = slotNum * N_symb_slot + pDmrsSymPos[i];
                        v = (gold32(puschIdentity, idxSeq) >> (idxSeq % 32)) & 0x1;
                    }
                }

                u = (f_gh + puschIdentity)%30;
            }
            // prbRelClusterStartIdx is the start index without accounting for the startPrb offset
            const uint16_t prbRelClusterStartIdx = PRB_CLUSTER_IDX * CUPHY_PUSCH_RX_CH_EST_DELAY_EST_PRB_CLUSTER_SIZE;
            uint16_t rIdx = prbRelClusterStartIdx * N_DMRS_GRID_TONES_PER_PRB + DMRS_TONE_IDX;
#ifdef ENABLE_COMMON_DFTSOFDM_DESCRCODE_SUBROUTINE
            float2 descrCode = gen_pusch_dftsofdm_descrcode(M_ZC, rIdx, u, v, nPrb, d_phi_6, d_phi_12, d_phi_18, d_phi_24, d_primeNums);
#else
            float2 descrCode = gen_pusch_dftsofdm_descrcode(M_ZC, rIdx, u, v, nPrb);
#endif
            const TComplexCompute data = type_convert<TComplexCompute>(tDataRx(DMRS_ABS_TONE_IDX, pDmrsSymPos[i], BS_ANT_IDX));
            TComplexCompute descrCodeConj = cuConj(cuGet<TComplexCompute>(descrCode.x, descrCode.y));
            pilots[i] = descrCodeConj * data;
        }
     }

    int32_t tOCCIdx = 0;
    TComplexCompute    avg[N_TOCC]{};
    if (isValidTone) {
        //--------------------------------------------------------------------------------------------------------
        // Time domain cover code removal

        // For first tOCC output, multiply all DMRS symbols with +1 and average
        if (activeTOCCBmsk & 0x1) {
            if constexpr (N_DMRS_SYMS == 1) {
                avg[tOCCIdx] = pilots[0];
            } else {
                avg[tOCCIdx] = 0.5f * (pilots[0] + pilots[1]);
            }
            tOCCIdx++;
        }

        // For second tOCC output, multiply even DMRS symbols with +1, odd DMRS symbols with -1 and average
        if (activeTOCCBmsk & 0x2) {
            if constexpr (N_DMRS_SYMS == 1) {
                avg[tOCCIdx] = pilots[0];
            } else {
                avg[tOCCIdx] = 0.5f * (pilots[0] - pilots[1]);
            }
            tOCCIdx++;
        }

        //--------------------------------------------------------------------------------------------------------
        // Apply frequecy domain cover code and store inplace in shared memory
        // Multiply even tones with +1 and odd tones with -1

        // Note that the loop termination count below is tOCC symbol count
 #pragma unroll
        for(int32_t i = 0; i < tOCCIdx; ++i)
        {
            int32_t fOCCIdx = 0;

 #pragma unroll
            for(int32_t j = 0; j < 2; ++j)
            {
                int32_t temp = (activeFOCCBmsk >> j) & 0x1;
                if (!temp) {
                    continue;
                }
                // First fOCC output: multiply all tones by +1s
                // Second fOCC output: multiply even tones by +1s and odd tones by -1s
                const int32_t sign = (-(DMRS_TONE_IDX & j)) | 1;
                shH(DMRS_TONE_IDX, (N_DMRS_SYMS_FOCC * i) + fOCCIdx, DMRS_GRID_IDX) = avg[i] * sign;
                fOCCIdx++;
            }
        }
     }

     thisThrdBlk.sync();

    const uint32_t DMRS_LS_EST_TONE_IDX = (DMRS_ABS_TONE_IDX - startPrb * N_TONES_PER_PRB - DMRS_GRID_IDX) / N_DMRS_TONE_STRIDE;

    coalesced_group active = coalesced_threads();
    TComplexCompute R1_thread { 0.0f, 0.0f };

    if (nLayers > 1) {
        if (isValidTone) {
            for (uint32_t i = 0; i < nLayers; i++) {
                uint32_t k = (OCCIdx[i] >> 2) & 0x1;
                if (DMRS_GRID_IDX != k) {
                    continue;
                }
                uint32_t j = OCCIdx[i] & 0x3;
                tInfoDmrsLSEst(DMRS_LS_EST_TONE_IDX, i, BS_ANT_IDX, NH_IDX) =
                    type_convert<TComplexCompute>(shH(DMRS_TONE_IDX, j, DMRS_GRID_IDX));
            }
        }
        
        if(enablePerPrgChEst==1)
        {
            if((prgSize==1)&&(DMRS_TONE_IDX%6==4))
                isValidR1Cal = false;
                
            if((prgSize==2)&&(DMRS_TONE_IDX%12==10))
                isValidR1Cal = false;
                
            if((prgSize==3)&&(DMRS_TONE_IDX%18==16))
                isValidR1Cal = false;
                
            if((prgSize==4)&&(DMRS_TONE_IDX%24==22))
                isValidR1Cal = false;
        }
        
        const bool nextToneIsValidAfterAvg = ((DMRS_ABS_TONE_IDX+2*N_DMRS_TONE_STRIDE) / N_TONES_PER_PRB) <= lastPrbThisCluster;
        if (DMRS_TONE_IDX % 2 == 0 && nextToneIsValidAfterAvg && isValidR1Cal) {
            for (uint32_t i = 0; i < nLayers; i++) {
                uint32_t j = OCCIdx[i] & 0x3;
                uint32_t k = (OCCIdx[i] >> 2) & 0x1;
                if (DMRS_GRID_IDX != k) {
                    continue;
                }
                const TComplexCompute avg0 = 0.5f *
                    (shH(DMRS_TONE_IDX, j, DMRS_GRID_IDX) + shH(DMRS_TONE_IDX+1, j, DMRS_GRID_IDX));
                const TComplexCompute avg1 = 0.5f *
                    (shH(DMRS_TONE_IDX+N_DMRS_TONE_STRIDE, j, DMRS_GRID_IDX) + shH(DMRS_TONE_IDX+N_DMRS_TONE_STRIDE+1, j, DMRS_GRID_IDX));
                const TComplexCompute prod = cuConjf(avg0) * avg1;
                R1_thread += prod;
            }
        }
    } else {
        if (isValidTone) {
            for (uint32_t i = 0; i < nLayers; i++) {
                uint32_t j = OCCIdx[i] & 0x3;
                uint32_t k = (OCCIdx[i] >> 2) & 0x1;
                if (DMRS_GRID_IDX != k) {
                    continue;
                }
                
                if(enablePerPrgChEst==1)
                {
                    if((prgSize==1)&&(DMRS_TONE_IDX%6==5))
                        isValidR1Cal = false;
                        
                    if((prgSize==2)&&(DMRS_TONE_IDX%12==11))
                        isValidR1Cal = false;
                        
                    if((prgSize==3)&&(DMRS_TONE_IDX%18==17))
                        isValidR1Cal = false;
                        
                    if((prgSize==4)&&(DMRS_TONE_IDX%24==23))
                        isValidR1Cal = false;
                }
                
                if (nextToneIsValid && isValidR1Cal) {
                    R1_thread += cuConj(shH(DMRS_TONE_IDX, j, DMRS_GRID_IDX)) * shH(DMRS_TONE_IDX+1, j, DMRS_GRID_IDX);
                }
                tInfoDmrsLSEst(DMRS_LS_EST_TONE_IDX, i, BS_ANT_IDX, NH_IDX) =
                    type_convert<TComplexCompute>(shH(DMRS_TONE_IDX, j, DMRS_GRID_IDX));
            }
        }
    }

    constexpr uint32_t MAX_NUM_WARPS = 32;
    __shared__ TComplexCompute sh_R1[MAX_NUM_WARPS];
    TComplexCompute R1_block;
    R1_block.x = reduce(tile, R1_thread.x, plus<TCompute>());
    R1_block.y = reduce(tile, R1_thread.y, plus<TCompute>());
    if (tile.thread_rank() == 0) {
        sh_R1[tile.meta_group_rank()] = R1_block;
    }
    thisThrdBlk.sync();
    if (threadIdx.x == 0) {
        for (int i = 1; i < tile.meta_group_size(); i++) {
            sh_R1[0] += sh_R1[i];
        }
        TComplexCompute *pAccum = reinterpret_cast<TComplexCompute *>(drvdUeGrpPrms.tInfoDmrsAccum.pAddr);
        atomicAdd(&pAccum[drvdUeGrpPrms.dmrsActiveAccumBuf].x, sh_R1[0].x);
        atomicAdd(&pAccum[drvdUeGrpPrms.dmrsActiveAccumBuf].y, sh_R1[0].y);
        if (blockIdx.x == 0 && blockIdx.y == 0) {
            // Clear the inactive accumulation buffer
            pAccum[drvdUeGrpPrms.dmrsActiveAccumBuf ^ 1].x = 0.0f;
            pAccum[drvdUeGrpPrms.dmrsActiveAccumBuf ^ 1].y = 0.0f;
        }
    }
 } //windowedChEstPreNoDftSOfdmKernel

 template <typename TStorage,
           typename TDataRx,
           typename TCompute,
           uint32_t N_LAYERS,                          // # of layers (# of cols in H matrix) ** may be larger than the actual number of layers in the group
           uint32_t N_DMRS_GRIDS_PER_PRB,              // # of DMRS grids per PRB (2 or 3)
           uint32_t N_DMRS_PRB_IN_PER_CLUSTER,         // # of PRBs bearing DMRS tones to be processed by each thread block (i.e. used in channel estimation)
           uint32_t N_DMRS_INTERP_PRB_OUT_PER_CLUSTER, // # of PRBs bearing channel estimates (interpolated tones) at output
           uint32_t N_DMRS_SYMS>                       // # of consecutive DMRS symbols (1 or 2)
 static __global__ void
 windowedChEstKernel(puschRxChEstStatDescr_t* pStatDescr, puschRxChEstDynDescr_t* pDynDescr)
 {
     //--------------------------------------------------------------------------------------------------------
     puschRxChEstDynDescr_t& dynDescr = *pDynDescr;

     // UE group processed by this thread block
     const uint32_t UE_GRP_IDX = dynDescr.hetCfgUeGrpMap[blockIdx.z];
     // PRB cluster being processed by this thread block
     const uint32_t PRB_CLUSTER_IDX = blockIdx.x;
     // BS antenna being processed by this thread block
     const uint32_t BS_ANT_IDX = blockIdx.y;

     // Early exit check
     // The grid is sized to process the max # of PRB clusters in a given heterogenous config. Exit if the PRB cluster to be
     // processed by this thread block does not exist in the UE group
     cuphyPuschRxUeGrpPrms_t& drvdUeGrpPrms = dynDescr.pDrvdUeGrpPrms[UE_GRP_IDX];
     const uint16_t nPrb   = drvdUeGrpPrms.nPrb;
     const uint16_t nRxAnt = drvdUeGrpPrms.nRxAnt;
     const uint32_t N_PRB_CLUSTERS_PER_BS_ANT = div_round_up(nPrb, static_cast<uint16_t>(N_DMRS_INTERP_PRB_OUT_PER_CLUSTER));
     if((PRB_CLUSTER_IDX >= N_PRB_CLUSTERS_PER_BS_ANT) || (BS_ANT_IDX >= nRxAnt)) return;

#ifdef ENABLE_DEBUG
     if((0 == blockIdx.x) && (0 == blockIdx.y) && (0 == threadIdx.x) && (0 == threadIdx.y) && (0 == threadIdx.z))
     printf("%s\n blockIdx.z %d UE_GRP_IDX %d blockDim (%d,%d,%d) gridDim (%d,%d,%d)\n", __PRETTY_FUNCTION__, blockIdx.z, UE_GRP_IDX, blockDim.x, blockDim.y, blockDim.z, gridDim.x, gridDim.y, gridDim.z);

     if((0 == threadIdx.x) && (0 == threadIdx.y) && (0 == threadIdx.z) && (0 == blockIdx.y) && (0 == blockIdx.z))
     printf("PRB_CLUSTER_IDX %d N_PRB_CLUSTERS_PER_BS_ANT %d chEstTimeInst %d\n", PRB_CLUSTER_IDX, N_PRB_CLUSTERS_PER_BS_ANT, dynDescr.chEstTimeInst);
#endif

     //--------------------------------------------------------------------------------------------------------
     // Setup local parameters based on descriptor
     puschRxChEstStatDescr_t& statDescr    = *pStatDescr;
     const uint16_t  slotNum               = drvdUeGrpPrms.slotNum;
     const uint8_t   chEstTimeInst         = dynDescr.chEstTimeInst;
     const uint8_t   activeDmrsGridBmsk    = drvdUeGrpPrms.activeDMRSGridBmsk;
     uint8_t*        OCCIdx                = drvdUeGrpPrms.OCCIdx;
     const uint16_t  nLayers               = drvdUeGrpPrms.nLayers;
     const uint8_t   dmrsMaxLen            = drvdUeGrpPrms.dmrsMaxLen;
     const uint8_t   nPrbsMod2             = nPrb & 0x1;
     const uint32_t  N_DATA_PRB            = nPrb;
     const uint8_t   scid                  = drvdUeGrpPrms.scid;
     const uint8_t   enableTfPrcd          = drvdUeGrpPrms.enableTfPrcd;

     // Pointer to DMRS symbol used for channel estimation (single-symbol if maxLen = 1, double-symbol if maxLen = 2)
     const uint8_t* const pDmrsSymPos   = &drvdUeGrpPrms.dmrsSymLoc[chEstTimeInst*dmrsMaxLen];
     const uint16_t  startPrb = drvdUeGrpPrms.startPrb;
     const uint16_t  dmrsScId = drvdUeGrpPrms.dmrsScrmId;
     const uint8_t   nDmrsCdmGrpsNoData = drvdUeGrpPrms.nDmrsCdmGrpsNoData;

     //--------------------------------------------------------------------------------------------------------
     typedef typename complex_from_scalar<TCompute>::type TComplexCompute;
     typedef typename complex_from_scalar<TDataRx>::type TComplexDataRx;
     typedef typename complex_from_scalar<TStorage>::type TComplexStorage;

     // clang-format off
     tensor_ref<const TCompute>       tFreqInterpCoefs((8==N_DMRS_PRB_IN_PER_CLUSTER) ? statDescr.tPrmFreqInterpCoefs.pAddr : statDescr.tPrmFreqInterpCoefs4.pAddr,
                                                       (8==N_DMRS_PRB_IN_PER_CLUSTER) ? statDescr.tPrmFreqInterpCoefs.strides : statDescr.tPrmFreqInterpCoefs4.strides); // (N_TOTAL_DMRS_INTERP_GRID_TONES_PER_CLUSTER + N_INTER_DMRS_GRID_FREQ_SHIFT, N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER, 3), 3 filters: 1 for middle, 1 lower edge and 1 upper edge

   //  tensor_ref<const TStorage>       tFreqInterpCoefs(statDescr.tPrmFreqInterpCoefs.pAddr, statDescr.tPrmFreqInterpCoefs.strides);
 #if 1 // shift/unshift sequences same precision as data (FP16 or FP32)
     tensor_ref<const TComplexDataRx> tShiftSeq       (statDescr.tPrmShiftSeq.pAddr       , statDescr.tPrmShiftSeq.strides);        // (N_DATA_PRB*N_DMRS_GRID_TONES_PER_PRB, N_DMRS_SYMS)
     tensor_ref<const TComplexDataRx> tUnShiftSeq     (statDescr.tPrmUnShiftSeq.pAddr     , statDescr.tPrmUnShiftSeq.strides);      // (N_DATA_PRB*N_DMRS_INTERP_TONES_PER_GRID*N_DMRS_GRIDS_PER_PRB + N_INTER_DMRS_GRID_FREQ_SHIFT)
 #else // shift/unshift sequences same precision as channel estimates (typically FP32)
     tensor_ref<const TComplexStorage> tShiftSeq       (statDescr.tPrmShiftSeq.pAddr       , statDescr.tPrmShiftSeq.strides);        // (N_DATA_PRB*N_DMRS_GRID_TONES_PER_PRB, N_DMRS_SYMS)
     tensor_ref<const TComplexStorage> tUnShiftSeq     (statDescr.tPrmUnShiftSeq.pAddr     , statDescr.tPrmUnShiftSeq.strides);      // (N_DATA_PRB*N_DMRS_INTERP_TONES_PER_GRID*N_DMRS_GRIDS_PER_PRB + N_INTER_DMRS_GRID_FREQ_SHIFT)
 #endif
     tensor_ref<const TComplexDataRx> tDataRx        (drvdUeGrpPrms.tInfoDataRx.pAddr  , drvdUeGrpPrms.tInfoDataRx.strides);// (NF, ND, N_BS_ANTS)
     tensor_ref<TComplexStorage>      tHEst          (drvdUeGrpPrms.tInfoHEst.pAddr    , drvdUeGrpPrms.tInfoHEst.strides);
     tensor_ref<TComplexStorage>      tDbg           (drvdUeGrpPrms.tInfoChEstDbg.pAddr, drvdUeGrpPrms.tInfoChEstDbg.strides);
     // clang-format on

#ifdef ENABLE_DEBUG
     if((0 == threadIdx.x) && (0 == threadIdx.y) && (0 == threadIdx.z) && (0 == blockIdx.x) && (0 == blockIdx.y))
     printf("%s\n: NH_IDX %d hetCfgUeGrpIdx %d ueGrpIdx %d nPrb %d startPb %d pDmrsSymPos[0] %d pDmrsSymPos[1] %d dmrsScId %d\n", __PRETTY_FUNCTION__, dynDescr.chEstTimeInst, blockIdx.z, UE_GRP_IDX, nPrb, startPrb, pDmrsSymPos[0], pDmrsSymPos[1], dmrsScId);
#endif

     //--------------------------------------------------------------------------------------------------------
     // Dimensions and indices

     // Estimates of H in time supported
     const uint32_t NH_IDX = chEstTimeInst;

     // Channel estimation expands tones in a DMRS grid (4 or 6, given by N_DMRS_GRID_TONES_PER_PRB) into a full PRB
     constexpr uint32_t N_DMRS_INTERP_TONES_PER_GRID = N_TONES_PER_PRB;

     // # of tones per DMRS grid in a PRB
     constexpr uint32_t N_DMRS_GRID_TONES_PER_PRB = N_TONES_PER_PRB / N_DMRS_GRIDS_PER_PRB;
     // Max permissible DMRS grids within a PRB based on spec
     constexpr uint32_t N_DMRS_TYPE1_GRIDS_PER_PRB = 2;
     constexpr uint32_t N_DMRS_TYPE2_GRIDS_PER_PRB = 3;
     static_assert(((N_DMRS_TYPE1_GRIDS_PER_PRB == N_DMRS_GRIDS_PER_PRB) || (N_DMRS_TYPE2_GRIDS_PER_PRB == N_DMRS_GRIDS_PER_PRB)),
                   "DMRS grid count exceeds max value");

     // Within a PRB, successive DMRS grids are shifted by 2 tones
     constexpr uint32_t N_INTER_DMRS_GRID_FREQ_SHIFT = get_inter_dmrs_grid_freq_shift(N_DMRS_GRIDS_PER_PRB);

     // Per grid tone counts present in input and output PRB clusters. These tones counts are expected to be equal
     constexpr uint32_t N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER        = N_DMRS_GRID_TONES_PER_PRB * N_DMRS_PRB_IN_PER_CLUSTER;
     constexpr uint32_t N_TOTAL_DMRS_INTERP_GRID_TONES_PER_CLUSTER = N_DMRS_INTERP_TONES_PER_GRID * N_DMRS_INTERP_PRB_OUT_PER_CLUSTER;

     // Total # of DMRS tones consumed by this thread block (this number should equal number of threads in
     // thread block since each DMRS tone is processed by a thread)
     constexpr uint32_t N_DMRS_TONES = N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER * N_DMRS_GRIDS_PER_PRB; // blockDim.x
     // Total # of interpolated DMRS tones produced by this thread block (this number should also equal number
     // of threads in thread block)
     constexpr uint32_t N_INTERP_TONES = N_TOTAL_DMRS_INTERP_GRID_TONES_PER_CLUSTER * N_DMRS_GRIDS_PER_PRB; // blockDim.x

     static_assert((N_DMRS_PRB_IN_PER_CLUSTER * N_TONES_PER_PRB == N_DMRS_TONES),
                   "Mismatch in expected vs calcualted DMRS tone count");
     static_assert((N_DMRS_TONES == N_INTERP_TONES),
                   "Thread allocation assumes input DMRS tone count and interpolated tone count are equal, ensure sufficient threads are allocated for interpoloation etc");

     // Ensure configured symbol count does not exceed max value prescribed by spec
     static_assert((N_DMRS_SYMS <= N_MAX_DMRS_SYMS), "DMRS symbol count exceeds max value");

     // Interpolation filter indices for middle and edge PRBs
     constexpr uint32_t MIDDLE_INTERP_FILT_IDX     = 0;
     constexpr uint32_t LOWER_EDGE_INTERP_FILT_IDX = 1;
     constexpr uint32_t UPPER_EDGE_INTERP_FILT_IDX = 2;

     // DMRS descrambling
     constexpr uint32_t N_DMRS_DESCR_BITS_PER_TONE    = 2; // 1bit for I and 1 bit for Q
     constexpr uint32_t N_DMRS_DESCR_BITS_PER_CLUSTER = N_DMRS_DESCR_BITS_PER_TONE * N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER;
     // # of DMRS descrambler bits generated at one time
     constexpr uint32_t N_DMRS_DESCR_BITS_GEN = 32;
     // Round up to the next multiple of N_DMRS_DESCR_BITS_GEN plus 1 (+1 because DMRS_DESCR_PRB_CLUSTER_START_BIT_OFFSET
     // may be large enough to spill the descrambler bits to the next word)
     constexpr uint32_t N_DMRS_DESCR_WORDS =
         ((N_DMRS_DESCR_BITS_PER_CLUSTER + N_DMRS_DESCR_BITS_GEN - 1) / N_DMRS_DESCR_BITS_GEN) + 1;
     // round_up_to_next(N_DMRS_DESCR_BITS_PER_CLUSTER, N_DMRS_DESCR_BITS_GEN) + 1;

     // number of "edge" tones, not estimated but used to extract additional dmrs
     constexpr uint32_t HALF_N_EDGE_TONES = N_TONES_PER_PRB * (N_DMRS_PRB_IN_PER_CLUSTER - N_DMRS_INTERP_PRB_OUT_PER_CLUSTER) / 2;

     const uint32_t ACTIVE_DMRS_GRID_BMSK = 0x3;

     // Total number of PRB clusers to be processed (N_PRB_CLUSTERS*N_DMRS_INTERP_PRB_OUT_PER_CLUSTER = N_DATA_PRB)
     const uint32_t N_PRB_CLUSTERS = N_PRB_CLUSTERS_PER_BS_ANT;

     // Per UE group descrambling ID
     uint16_t dmrsScramId = dmrsScId;

#ifdef ENABLE_DEBUG

     if((0 == blockIdx.x) && (0 == blockIdx.y) && (0 == blockIdx.z) && (0 == threadIdx.x) && (0 == threadIdx.y) && (0 == threadIdx.z))
     {
         printf("Addr: tFreqInterpCoefs %lp tShiftSeq %lp tUnShiftSeq %lp tDataRx %lp tHEst %lp \n", tFreqInterpCoefs.pAddr, tShiftSeq.pAddr, tUnShiftSeq.pAddr, tDataRx.pAddr, tHEst.pAddr);

         printf("tFreqInterpCoefs: addr %lp strides[0] %d strides[1] %d strides[2] %d\n", static_cast<const TCompute*>(tFreqInterpCoefs.pAddr) , tFreqInterpCoefs.strides[0], tFreqInterpCoefs.strides[1], tFreqInterpCoefs.strides[2]);
         printf("tShiftSeq       : addr %lp strides[0] %d strides[1] %d strides[2] %d\n", static_cast<const TComplexDataRx*>(tShiftSeq.pAddr)  , tShiftSeq.strides[0]       , tShiftSeq.strides[1]       , tShiftSeq.strides[2]       );
         printf("tUnShiftSeq     : addr %lp strides[0] %d strides[1] %d strides[2] %d\n", static_cast<const TComplexDataRx*>(tUnShiftSeq.pAddr), tUnShiftSeq.strides[0]     , tUnShiftSeq.strides[1]     , tUnShiftSeq.strides[2]     );

         printf("startPrb       : %d \n", startPrb);
         printf("dmrsScId       : %d\n", dmrsScId);
         printf("tDataRx         : addr %lp strides[0] %d strides[1] %d strides[2] %d\n", static_cast<const TComplexDataRx*>(tDataRx.pAddr)    , tDataRx.strides[0]         , tDataRx.strides[1]         , tDataRx.strides[2]         );
         printf("tHEst           : addr %lp strides[0] %d strides[1] %d strides[2] %d\n", static_cast<TComplexStorage*>(tHEst.pAddr)     , tHEst.strides[0]                 , tHEst.strides[1]           , tHEst.strides[2]           );
         // printf("tDbg    strides[0] %d strides[1] %d strides[2] %d\n", tDbg.strides[0], tDbg.strides[1], tDbg.strides[2]);

         printf("dmrsScramId %d slotNum %d activeDmrsGridBmsk 0x%08x chEstTimeInst %d\n", dmrsScramId, slotNum, activeDmrsGridBmsk, NH_IDX);
     }

     // printf("Block(%d %d %d) Thread(%d %d %d)\n", blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z);
     // if((0 != BS_ANT_IDX) || (0 != PRB_CLUSTER_IDX)) return;
     // printf("dmrsScramId %d slotNum %d activeDmrsGridBmsk 0x%08x", dmrsScramId, slotNum, activeDmrsGridBmsk);
     // if((0 == blockIdx.x) && (0 == blockIdx.y) && (0 == blockIdx.z) && (0 == threadIdx.x) && (0 == threadIdx.y) && (0 == threadIdx.z))
     //    printf("tDataRx strides[0] %d strides[1] %d strides[2] %d\n", tDataRx.strides[0], tDataRx.strides[1], tDataRx.strides[2]);
 #if 0
     printf("InterpCoefs[%d][%d][%d] = %f ShiftSeq[%d][%d] = %f+j%f UnShiftSeq[%d] %f+j%f DataRx[%d][%d][%d]= %f+j%f\n",
            0,0,0,
            tFreqInterpCoefs(0,0,0),
            0,0,
            tShiftSeq(0,0).x,
            tShiftSeq(0,0).y,
            0,
            tUnShiftSeq(0).x,
            tUnShiftSeq(0).y,
            0,0,0,
            tDataRx(0,0,0).x,
            tDataRx(0,0,0).y);
 #endif
#endif

     const uint32_t THREAD_IDX = threadIdx.x;

     // # of PRBs for which channel must be estimated
     const uint32_t N_EDGE_PRB = (N_DMRS_PRB_IN_PER_CLUSTER - N_DMRS_INTERP_PRB_OUT_PER_CLUSTER) / 2; // Lower and Upper edge PRBs

     // Determine first PRB in the cluster being processed
     uint32_t prbClusterStartIdx = (PRB_CLUSTER_IDX * N_DMRS_INTERP_PRB_OUT_PER_CLUSTER) - N_EDGE_PRB;
     if(0 == PRB_CLUSTER_IDX) prbClusterStartIdx = 0;                                                             // Lower edge
     if((N_PRB_CLUSTERS - 1) == PRB_CLUSTER_IDX) prbClusterStartIdx = N_DATA_PRB - N_DMRS_PRB_IN_PER_CLUSTER;     // Upper edge

     uint32_t prbAbsStartIdx = prbClusterStartIdx + startPrb;
     // Absolute index of DMRS tone within the input OFDM symbol (used as index when loading tone from OFDM
     // symbol)
     const uint32_t DMRS_ABS_TONE_IDX = prbAbsStartIdx * N_TONES_PER_PRB + THREAD_IDX;

     // This index calculation intends to divvy up threads in the thread block for processing as follows:
     // the first group of N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER to process the first DMRS grid, the second group
     // to process the second DMRS grid and so on
     // Relative index of DMRS tone (within a DMRS grid) being processed by this thread
     const uint32_t DMRS_TONE_IDX        = THREAD_IDX % N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER;
     const uint32_t DMRS_INTERP_TONE_IDX = THREAD_IDX % N_TOTAL_DMRS_INTERP_GRID_TONES_PER_CLUSTER;

     // Index of DMRS grid in which the DMRS tone being processed by this thread resides
     // Note: although the grid index is calculated using total number of DMRS grid tones in the cluster, its
     // used as an index in context of both input DMRS tones and interpolated DMRS tones under the assumption:
     // N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER == N_TOTAL_DMRS_INTERP_GRID_TONES_PER_CLUSTER
     const uint32_t DMRS_GRID_IDX = THREAD_IDX / N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER;

     const uint32_t tmpGridIdx = DMRS_GRID_IDX > 1 ? 1 : DMRS_GRID_IDX;
     const uint8_t  activeTOCCBmsk     = drvdUeGrpPrms.activeTOCCBmsk[tmpGridIdx];
     const uint8_t  activeFOCCBmsk     = drvdUeGrpPrms.activeFOCCBmsk[tmpGridIdx];
     const uint32_t N_DMRS_SYMS_FOCC = activeFOCCBmsk == 3 ? 2 : 1;
     // Index which enables extraction of DMRS tones of a given DMRS grid scattered within the PRB into one
     // contiguous set for processing. Note that the read from GMEM is coalesced and write into SMEM is scattered
     // @todo: check if index calculation can be simplified
     const uint32_t SMEM_DMRS_TONE_IDX = get_smem_dmrs_tone_idx(N_DMRS_GRIDS_PER_PRB, N_INTER_DMRS_GRID_FREQ_SHIFT, THREAD_IDX);
     const uint32_t SMEM_DMRS_GRID_IDX = get_smem_dmrs_grid_idx(N_DMRS_GRIDS_PER_PRB, N_INTER_DMRS_GRID_FREQ_SHIFT, THREAD_IDX);

     // Absolute index of descrambling + shift sequence element
     const uint32_t DMRS_DESCR_SHIFT_SEQ_START_IDX = prbAbsStartIdx * N_DMRS_GRID_TONES_PER_PRB;
     // const uint32_t DMRS_DESCR_SHIFT_SEQ_ABS_IDX   = DMRS_DESCR_SHIFT_SEQ_START_IDX + DMRS_TONE_IDX;
     const uint32_t DMRS_DESCR_SHIFT_SEQ_ABS_IDX = DMRS_TONE_IDX;

     // Select one of 3 interpolation filters for middle section, lower and upper edges of the frequency band
     uint32_t filtIdx = MIDDLE_INTERP_FILT_IDX;                                        // All tones in between lower and upper edges
     if(0 == PRB_CLUSTER_IDX) filtIdx = LOWER_EDGE_INTERP_FILT_IDX;                    // Lower edge
     if((N_PRB_CLUSTERS - 1) == PRB_CLUSTER_IDX) filtIdx = UPPER_EDGE_INTERP_FILT_IDX; // Upper edge

     // Absolute index of interpolated tone produced by this thread
     const uint32_t INTERP_PRB_CLUSTER_IDX   = blockIdx.x;
     uint32_t INTERP_DMRS_ABS_TONE_IDX = INTERP_PRB_CLUSTER_IDX * N_DMRS_INTERP_PRB_OUT_PER_CLUSTER * N_TONES_PER_PRB + DMRS_INTERP_TONE_IDX;

     if((N_DMRS_PRB_IN_PER_CLUSTER == 4) && (nPrbsMod2 == 1) && (filtIdx == UPPER_EDGE_INTERP_FILT_IDX))
         INTERP_DMRS_ABS_TONE_IDX = INTERP_DMRS_ABS_TONE_IDX - N_TONES_PER_PRB;

     // Select the shift in interpolation filter coefficients and delay shift based on grid index
     // (for e.g. for 2 DMRS grids and 48 tones per grid, multiply DMRS tone vector with top 48 rows for
     // DMRS_GRID_IDX 0 and bottom 48 rows for DMRS_GRID_IDX 1 to acheieve the effect of shift)
     uint32_t gridShiftIdx = get_inter_dmrs_grid_freq_shift_idx(N_DMRS_GRIDS_PER_PRB, DMRS_GRID_IDX);

     // Section 5.2.1 in 3GPP TS 38.211
     // The fast-forward of 1600 prescribed by spec is already baked into the gold sequence generator
     constexpr uint32_t DMRS_DESCR_FF = 0; // 1600;

     // First descrambler bit index needed by this thread block
     // Note:The DMRS scrambling sequence is the same for all the DMRS grids. There are 2 sequences one for
     // scid 0 and other for scid 1 but the same sequences is reused for all DMRS grids
     const uint32_t DMRS_DESCR_PRB_CLUSTER_START_BIT =
         DMRS_DESCR_FF + (DMRS_DESCR_SHIFT_SEQ_START_IDX * N_DMRS_DESCR_BITS_PER_TONE);

     // The descrambling sequence generator outputs 32 descrambler bits at a time. Thus, compute the earliest
     // multiple of 32 bits which contains the descrambler bit of the first tone in the PRB cluster as the
     // start index
     const uint32_t DMRS_DESCR_GEN_ALIGNED_START_BIT =
         (DMRS_DESCR_PRB_CLUSTER_START_BIT / N_DMRS_DESCR_BITS_GEN) * N_DMRS_DESCR_BITS_GEN;
     // Offset to descrambler bit of the first tone in the PRB cluster
     const uint32_t DMRS_DESCR_PRB_CLUSTER_START_BIT_OFFSET =
         DMRS_DESCR_PRB_CLUSTER_START_BIT - DMRS_DESCR_GEN_ALIGNED_START_BIT;

     // DMRS descrambling bits generated correspond to subcarriers across frequency
     // e.g. 2 bits for tone0(grid 0) | 2 bits for tone1(grid 1) | 2 bits for tone 2(grid 0) | 2 bits for tone 3(grid 1) | ...
     const uint32_t DMRS_TONE_DESCR_BIT_IDX = DMRS_DESCR_PRB_CLUSTER_START_BIT_OFFSET +
                                              (DMRS_TONE_IDX * N_DMRS_DESCR_BITS_PER_TONE);
     const uint32_t DMRS_DESCR_SEQ_RD_BIT_IDX  = DMRS_TONE_DESCR_BIT_IDX % N_DMRS_DESCR_BITS_GEN;
     const uint32_t DMRS_DESCR_SEQ_RD_WORD_IDX = DMRS_TONE_DESCR_BIT_IDX / N_DMRS_DESCR_BITS_GEN;

     const uint32_t DMRS_DESCR_SEQ_WR_WORD_IDX = THREAD_IDX % N_DMRS_DESCR_WORDS;
     const uint32_t DMRS_DESCR_SEQ_WR_SYM_IDX  = THREAD_IDX / N_DMRS_DESCR_WORDS;

 #if 0
     if((0 == blockIdx.x) && (0 == blockIdx.y) && (0 == blockIdx.z))
        printf("N_DMRS_DESCR_BITS_PER_CLUSTER %d N_DMRS_DESCR_WORDS %d DMRS_DESCR_PRB_CLUSTER_START_BIT %d DMRS_DESCR_GEN_ALIGNED_START_BIT %d, "
               "DMRS_DESCR_SEQ_RD_WORD_IDX %d, DMRS_DESCR_SEQ_RD_BIT_IDX %d, DMRS_DESCR_SEQ_WR_WORD_IDX %d, DMRS_DESCR_SEQ_WR_SYM_IDX %d\n",
               N_DMRS_DESCR_BITS_PER_CLUSTER, N_DMRS_DESCR_WORDS, DMRS_DESCR_PRB_CLUSTER_START_BIT, DMRS_DESCR_GEN_ALIGNED_START_BIT,
               DMRS_DESCR_SEQ_RD_WORD_IDX, DMRS_DESCR_SEQ_RD_BIT_IDX, DMRS_DESCR_SEQ_WR_WORD_IDX, DMRS_DESCR_SEQ_WR_SYM_IDX);
 #endif

     // Data layouts:
     // Global memory read into shared memory
     // N_DMRS_TONES x N_DMRS_SYMS -> N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER x N_DMRS_SYMS x N_DMRS_GRIDS_PER_PRB

     // tOCC removal
     // N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER x N_DMRS_SYMS x N_DMRS_GRIDS_PER_PRB ->
     // N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER x N_DMRS_SYMS_TOCC x N_DMRS_GRIDS_PER_PRB

     // fOCC removal
     // N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER x N_DMRS_SYMS_TOCC x N_DMRS_GRIDS_PER_PRB ->
     // N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER x N_DMRS_SYMS_FOCC x NUM_DMRS_SYMS_TOCC x N_DMRS_GRIDS_PER_PRB

     // Interpolation
     // N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER x N_DMRS_SYMS_FOCC x NUM_DMRS_SYMS_TOCC x N_DMRS_GRIDS_PER_PRB ->
     // N_TOTAL_DMRS_INTERP_GRID_TONES_PER_CLUSTER x N_DMRS_SYMS_FOCC x NUM_DMRS_SYMS_TOCC x N_DMRS_GRIDS_PER_PRB =
     // N_TOTAL_DMRS_INTERP_GRID_TONES_PER_CLUSTER x N_LAYERS x N_DMRS_GRIDS_PER_PRB

     //--------------------------------------------------------------------------------------------------------
     // Allocate shared memory

     constexpr uint32_t N_SMEM_ELEMS1 = N_DMRS_TONES * N_DMRS_SYMS; // (N_DMRS_TONES + N_DMRS_GRIDS_PER_PRB)*N_DMRS_SYMS;
     constexpr uint32_t N_SMEM_ELEMS2 = N_INTERP_TONES * N_DMRS_SYMS_OCC;
     constexpr uint32_t N_SMEM_ELEMS  = (N_SMEM_ELEMS1 > N_SMEM_ELEMS2) ? N_SMEM_ELEMS1 : N_SMEM_ELEMS2;
     // constexpr uint32_t N_SMEM_ELEMS  = (N_SMEM_ELEMS1 + N_SMEM_ELEMS2);
     // constexpr uint32_t N_SMEM_ELEMS  = max(N_SMEM_ELEMS1, N_SMEM_ELEMS2);

     __shared__ TComplexCompute smemBlk[N_SMEM_ELEMS];
     // overlay1
     block_3D<TComplexCompute*, N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER, N_DMRS_SYMS, N_DMRS_GRIDS_PER_PRB> shPilots(&smemBlk[0]);
     // overlay2
     block_3D<TComplexCompute*, N_TOTAL_DMRS_INTERP_GRID_TONES_PER_CLUSTER, N_DMRS_SYMS_OCC, N_DMRS_GRIDS_PER_PRB> shH(&smemBlk[0]);
     // block_3D<TComplexCompute*, N_TOTAL_DMRS_INTERP_GRID_TONES_PER_CLUSTER, N_DMRS_SYMS_OCC, N_DMRS_GRIDS_PER_PRB> shH(&smemBlk[shPilots.num_elem()]);
     static_assert((shPilots.num_elem() <= N_SMEM_ELEMS) && (shH.num_elem() <= N_SMEM_ELEMS), "Insufficient shared memory");

     __shared__ uint32_t descrWords[N_DMRS_SYMS][N_DMRS_DESCR_WORDS];

     //--------------------------------------------------------------------------------------------------------
     // Read DMRS tones into shared memory (separate the tones into different DMRS grids during the write)

     // Cache shift sequence in register
     TComplexCompute shiftSeq = type_convert<TComplexCompute>(tShiftSeq(DMRS_DESCR_SHIFT_SEQ_ABS_IDX, 0));

 #pragma unroll
     for(uint32_t i = 0; i < N_DMRS_SYMS; ++i)
     {
         shPilots(SMEM_DMRS_TONE_IDX, i, SMEM_DMRS_GRID_IDX) =
             type_convert<TComplexCompute>(tDataRx(DMRS_ABS_TONE_IDX, pDmrsSymPos[i], BS_ANT_IDX));

#ifdef ENABLE_DEBUG
         printf("Pilots[%d][%d][%d] -> shPilots[%d][%d][%d] = %f+j%f, ShiftSeq[%d][%d] = %f+j%f\n",
                DMRS_ABS_TONE_IDX,
                pDmrsSymPos[i],
                BS_ANT_IDX,
                SMEM_DMRS_TONE_IDX,
                i,
                SMEM_DMRS_GRID_IDX,
                shPilots(SMEM_DMRS_TONE_IDX, i, SMEM_DMRS_GRID_IDX).x,
                shPilots(SMEM_DMRS_TONE_IDX, i, SMEM_DMRS_GRID_IDX).y,
                DMRS_DESCR_SHIFT_SEQ_ABS_IDX,
                0,
                shiftSeq.x,
                shiftSeq.y);
#endif
     }
     if(enableTfPrcd==0)
     {
         // Compute the descsrambler sequence
         const uint32_t TWO_POW_17 = bit(17);

         if(DMRS_DESCR_SEQ_WR_SYM_IDX < N_DMRS_SYMS)
         {
             uint32_t symIdx = pDmrsSymPos[DMRS_DESCR_SEQ_WR_SYM_IDX];

             // see 38.211 section 6.4.1.1.1.1
             uint32_t cInit = TWO_POW_17 * (slotNum * OFDM_SYMBOLS_PER_SLOT + symIdx + 1) * (2 * dmrsScramId + 1) + (2 * dmrsScramId) + scid;
             cInit &= ~bit(31);

             // descrWords[DMRS_DESCR_SEQ_WR_SYM_IDX][DMRS_DESCR_SEQ_WR_WORD_IDX] =
             //  __brev(gold32(cInit, (DMRS_DESCR_GEN_ALIGNED_START_BIT + DMRS_DESCR_SEQ_WR_WORD_IDX*N_DMRS_DESCR_BITS_GEN)));

             descrWords[DMRS_DESCR_SEQ_WR_SYM_IDX][DMRS_DESCR_SEQ_WR_WORD_IDX] =
                 gold32(cInit, (DMRS_DESCR_GEN_ALIGNED_START_BIT + DMRS_DESCR_SEQ_WR_WORD_IDX * N_DMRS_DESCR_BITS_GEN));
 #if 0
             printf("symIdx %d, DMRS_DESCR_SEQ_WR_WORD_IDX %d, cInit 0x%08x, DMRS_DESCR_GEN_ALIGNED_START_BIT %d, descrWords[%d][%d] 0x%08x\n",
                    symIdx, DMRS_DESCR_SEQ_WR_WORD_IDX, cInit,
                    (DMRS_DESCR_GEN_ALIGNED_START_BIT + DMRS_DESCR_SEQ_WR_WORD_IDX*N_DMRS_DESCR_BITS_GEN),
                    DMRS_DESCR_SEQ_WR_SYM_IDX, DMRS_DESCR_SEQ_WR_WORD_IDX, descrWords[DMRS_DESCR_SEQ_WR_SYM_IDX][DMRS_DESCR_SEQ_WR_WORD_IDX]);
 #endif
         }
     }

     // To ensure coalesced reads, input DMRS tones are read preserving input order but swizzled while writing
     // to shared memory. Thus each thread may not process the same tone which it wrote to shared memory
     thread_block const& thisThrdBlk = this_thread_block();
     __syncthreads();

     //--------------------------------------------------------------------------------------------------------
     // Apply de-scrambling + delay domain centering sequence for tone index processed by this thread across all
     // DMRS symbols
     const TCompute RECIPROCAL_SQRT2 = 0.7071068f;
     const TCompute SQRT2            = 1.41421356f;

 #pragma unroll
     for(uint32_t i = 0; i < N_DMRS_SYMS; ++i)
     {
         if(enableTfPrcd==0)
         {
             int8_t descrIBit = (descrWords[i][DMRS_DESCR_SEQ_RD_WORD_IDX] >> DMRS_DESCR_SEQ_RD_BIT_IDX) & 0x1;
             int8_t descrQBit = (descrWords[i][DMRS_DESCR_SEQ_RD_WORD_IDX] >> (DMRS_DESCR_SEQ_RD_BIT_IDX + 1)) & 0x1;

             TComplexCompute descrCode =
                 cuConj(cuGet<TComplexCompute>((1 - 2 * descrIBit) * RECIPROCAL_SQRT2, (1 - 2 * descrQBit) * RECIPROCAL_SQRT2));
             TComplexCompute descrShiftSeq = shiftSeq * descrCode;

#ifdef ENABLE_DEBUG
             TComplexCompute descrShiftPilot = shPilots(DMRS_TONE_IDX, i, DMRS_GRID_IDX) * descrShiftSeq;
             printf("descrShiftAbsIdx: %d, shPilots[%d][%d][%d] (%f+j%f) * DescrShiftSeq[%d][%d] (%f+j%f) = %f+j%f, ShiftSeq = %f+j%f, DescrCode = %f+j%f, descrIQ (%d,%d) descrWordIdx %d descrBitIdx %d\n",
                    DMRS_DESCR_SHIFT_SEQ_ABS_IDX,
                    DMRS_TONE_IDX,
                    i,
                    DMRS_GRID_IDX,
                    shPilots(DMRS_TONE_IDX, i, DMRS_GRID_IDX).x,
                    shPilots(DMRS_TONE_IDX, i, DMRS_GRID_IDX).y,
                    DMRS_TONE_IDX,
                    i,
                    descrShiftSeq.x,
                    descrShiftSeq.y,
                    descrShiftPilot.x,
                    descrShiftPilot.y,
                    shiftSeq.x,
                    shiftSeq.y,
                    descrCode.x,
                    descrCode.y,
                    descrIBit,
                    descrQBit,
                    DMRS_DESCR_SEQ_RD_WORD_IDX,
                    DMRS_DESCR_SEQ_RD_BIT_IDX);}
             if((0 == DMRS_GRID_IDX) && (((0 == prbAbsStartIdx) && (DMRS_TONE_IDX < (N_EDGE_PRB * N_DMRS_GRID_TONES_PER_PRB))) || ((0 != prbAbsStartIdx) && (prbAbsStartIdx + N_DMRS_PRB_IN_PER_CLUSTER) <= N_DATA_PRB)))
             {
 #if 0
                TComplexCompute descrShiftPilot = shPilots(DMRS_TONE_IDX, i, DMRS_GRID_IDX) * descrShiftSeq;
                printf("descrShiftAbsIdx: %d shPilots[%d][%d][%d] (%f+j%f) * DescrShiftSeq[%d][%d] (%f+j%f) = %f+j%f, ShiftSeq = %f+j%f, DescrCode = %f+j%f, descrIQ (%d,%d) descrWordIdx %d descrBitIdx %d\n",
                      DMRS_DESCR_SHIFT_SEQ_ABS_IDX,
                      DMRS_TONE_IDX,
                      i,
                      DMRS_GRID_IDX,
                      shPilots(DMRS_TONE_IDX, i, DMRS_GRID_IDX).x,
                      shPilots(DMRS_TONE_IDX, i, DMRS_GRID_IDX).y,
                      DMRS_TONE_IDX,
                      i,
                      descrShiftSeq.x,
                      descrShiftSeq.y,
                      descrShiftPilot.x,
                      descrShiftPilot.y,
                      shiftSeq.x,
                      shiftSeq.y,
                      descrCode.x,
                      descrCode.y,
                      descrIBit,
                      descrQBit,
                      DMRS_DESCR_SEQ_RD_WORD_IDX,
                      DMRS_DESCR_SEQ_RD_BIT_IDX);
#endif

                 // tDbg(DMRS_DESCR_SHIFT_SEQ_ABS_IDX, i, 0, 0) = type_convert<TComplexStorage>(shiftSeq);
                 // tDbg(DMRS_DESCR_SHIFT_SEQ_ABS_IDX, i, 0, 0) = type_convert<TComplexStorage>(descrShiftSeq);
                 tDbg(DMRS_DESCR_SHIFT_SEQ_ABS_IDX, i, 0, 0) = type_convert<TComplexStorage>(descrCode);
             }
#endif // ENABLE_DEBUG
             shPilots(DMRS_TONE_IDX, i, DMRS_GRID_IDX) *= descrShiftSeq;
         }
         else if(enableTfPrcd==1)
         {
               uint16_t M_ZC  = N_DMRS_GRID_TONES_PER_PRB * nPrb; //different from DMRS_ABS_TONE_IDX
               int u = 0;
               int v = 0;

               if(drvdUeGrpPrms.optionalDftSOfdm)
               {
                   u = (int)drvdUeGrpPrms.lowPaprGroupNumber;
                   v = (int)drvdUeGrpPrms.lowPaprSequenceNumber;
               }
               else
               {
                   const uint32_t puschIdentity          = drvdUeGrpPrms.puschIdentity;
                   const uint8_t  groupOrSequenceHopping = drvdUeGrpPrms.groupOrSequenceHopping;
                   const uint8_t  N_symb_slot            = drvdUeGrpPrms.N_symb_slot;

                   int f_gh       = 0;

                   if(groupOrSequenceHopping==1)
                   {
                       uint32_t cInit = floor(puschIdentity/30);
                       for(int m = 0; m < 8; m++)
                       {
                           uint32_t idxSeq = 8 * (slotNum * N_symb_slot + pDmrsSymPos[i]) + m;
                           f_gh = f_gh + ((gold32(cInit, idxSeq) >> (idxSeq % 32)) & 0x1) * (1 << m);
                       }
                       f_gh = f_gh % 30;
    //                   if((blockIdx.x==0)&&(blockIdx.y==0)&&(threadIdx.x==0)&&(threadIdx.y==0))
    //                   {
    //                       printf("f_gh[%d]\n", f_gh);
    //                   }
                   }
                   else if(groupOrSequenceHopping==2)
                   {
                       if(M_ZC > 6 * N_TONES_PER_PRB)
                       {
                           uint32_t idxSeq = slotNum * N_symb_slot + pDmrsSymPos[i];
                           v = (gold32(puschIdentity, idxSeq) >> (idxSeq % 32)) & 0x1;

    //                       if((blockIdx.x==0)&&(blockIdx.y==0)&&(threadIdx.x==0)&&(threadIdx.y==0))
    //                       {
    //                           printf("idxSeq[%d]v[%d]\n", idxSeq, v);
    //                       }
                       }
                   }

                   u = (f_gh + puschIdentity)%30;
               }
               uint16_t rIdx = prbClusterStartIdx * N_DMRS_GRID_TONES_PER_PRB + DMRS_TONE_IDX;
#ifdef ENABLE_COMMON_DFTSOFDM_DESCRCODE_SUBROUTINE
               float2 descrCode = gen_pusch_dftsofdm_descrcode(M_ZC, rIdx, u, v, nPrb, d_phi_6, d_phi_12, d_phi_18, d_phi_24, d_primeNums);
#else
               float2 descrCode = gen_pusch_dftsofdm_descrcode(M_ZC, rIdx, u, v, nPrb);
#endif
               TComplexCompute descrShiftSeq = shiftSeq * cuConj(cuGet<TComplexCompute>(descrCode.x, descrCode.y));
               shPilots(DMRS_TONE_IDX, i, DMRS_GRID_IDX) *= descrShiftSeq;
         }
     } // for(uint32_t i = 0; i < N_DMRS_SYMS; ++i)

     //--------------------------------------------------------------------------------------------------------
     // Time domain cover code removal
     constexpr TCompute AVG_SCALE = cuGet<TCompute>(1u) / cuGet<TCompute>(N_DMRS_SYMS);
     TComplexCompute    avg[N_TOCC]{};

     int32_t tOCCIdx = 0;

 #pragma unroll
     for(int32_t i = 0; i < 2; ++i)
     {
        int32_t temp = (activeTOCCBmsk >> i) & 0x1;
        if (!temp) {
            continue;
        }
 #pragma unroll
         for(int32_t j = 0; j < N_DMRS_SYMS; ++j)
         {
             // For first tOCC (i = 0) output, multiply all DMRS symbols with +1 and average
             // For second tOCC (i = 1) output, multiply even DMRS symbols with +1, odd DMRS symbols with -1 and average
             int32_t sign = (-(i & j)) | 1;
             avg[tOCCIdx] += (shPilots(DMRS_TONE_IDX, j, DMRS_GRID_IDX) * (sign * AVG_SCALE));
 #ifdef ENABLE_DEBUG
             TComplexCompute prod = (shPilots(DMRS_TONE_IDX, j, DMRS_GRID_IDX) * (sign * AVG_SCALE));
             printf("sign*AVG_SCALE %f Pilot[%d][%d][%d] = %f+j%f avg[%d] = %f+j%f, prod = %f+j%f\n",
                    sign * AVG_SCALE,
                    DMRS_TONE_IDX,
                    j,
                    DMRS_GRID_IDX,
                    shPilots(DMRS_TONE_IDX, j, DMRS_GRID_IDX).x,
                    shPilots(DMRS_TONE_IDX, j, DMRS_GRID_IDX).y,
                    i,
                    avg[i].x,
                    avg[i].y,
                    prod.x,
                    prod.y);
 #endif
         }
         tOCCIdx++;
     }

     // shPilots and shH are overlaid in shared memory and can have different sizes (based on config). For this reason
     // ensure shPilots access from all threads is completed before writing into shH
     __syncthreads();

     //--------------------------------------------------------------------------------------------------------
     // Apply frequecy domain cover code and store inplace in shared memory
     // Multiply even tones with +1 and odd tones with -1

     // Note that the loop termination count below is tOCC symbol count
 #pragma unroll
     for(int32_t i = 0; i < tOCCIdx; ++i)
     {
        int32_t fOCCIdx = 0;

 #pragma unroll
         for(int32_t j = 0; j < 2; ++j)
         {
            int32_t temp = (activeFOCCBmsk >> j) & 0x1;
            if (!temp) {
                continue;
            }
             // First fOCC output: multiply all tones by +1s
             // Second fOCC output: multiply even tones by +1s and odd tones by -1s
             int32_t sign                                                  = (-(DMRS_TONE_IDX & j)) | 1;
             shH(DMRS_TONE_IDX, (N_DMRS_SYMS_FOCC * i) + fOCCIdx, DMRS_GRID_IDX) = avg[i] * sign;

 #ifdef ENABLE_DEBUG
             printf("PilotsPostOCC[%d][%d][%d] = %f+j%f\n",
                    DMRS_TONE_IDX,
                    (N_DMRS_SYMS_FOCC * i) + j,
                    DMRS_GRID_IDX,
                    cuReal(shH(DMRS_TONE_IDX, (N_DMRS_SYMS_FOCC * i) + j, DMRS_GRID_IDX)),
                    cuImag(shH(DMRS_TONE_IDX, (N_DMRS_SYMS_FOCC * i) + j, DMRS_GRID_IDX)));
 #endif
             fOCCIdx++;
         }
     }

     // Ensure all threads complete writing results to shared memory since each thread computing an inner product
     // during interpolation stage will use results from other threads in the thread block
     __syncthreads();

     //--------------------------------------------------------------------------------------------------------
     // Interpolate (matrix-vector multiply)
     for(uint32_t i = 0; i < (N_DMRS_SYMS_FOCC*tOCCIdx); ++i)
     {
         TComplexCompute innerProd{};

         // H = W x Y: (N_INTERP_TONES x N_DMRS_TONES) x (N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER x N_DMRS_SYMS_OCC)
         // Each thread selects one row of W and computes N_DMRS_TONES length inner product to produce one interpolated
         // tone of H
         for(uint32_t j = 0; j < N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER; ++j)
         {
             TCompute interpCoef = type_convert<TCompute>(tFreqInterpCoefs(DMRS_INTERP_TONE_IDX + gridShiftIdx, j, filtIdx));
             innerProd += (shH(j, i, DMRS_GRID_IDX) * interpCoef);
         }
         // Wait for all threads to complete their inner products before updating the shared memory inplace
         // The sync is needed because shPilots and shH are overlaid
         // Note that tOCCIdx can vary per thread, so this sync and the next need to be thisThrdBlk.sync()
         // calls rather than __syncthreads().
         thisThrdBlk.sync();

         shH(DMRS_INTERP_TONE_IDX, i, DMRS_GRID_IDX) = innerProd;

 #ifdef ENABLE_DEBUG
         printf("InterpPilots[%d][%d][%d] = %f+j%f\n", DMRS_INTERP_TONE_IDX, i, DMRS_GRID_IDX, innerProd.x, innerProd.y);
 #endif
     }

     // Wait for shared memory writes to complete from all threads
     thisThrdBlk.sync();

     // Write channel estimates for active grids only
     const int32_t DMRS_GRID_WR_IDX = DMRS_GRID_WR_IDX_TBL[activeDmrsGridBmsk & ACTIVE_DMRS_GRID_BMSK][DMRS_GRID_IDX];
     // if(!is_set(bit(DMRS_GRID_IDX), activeDmrsGridBmsk) || (DMRS_GRID_WR_IDX < 0)) return;
     if(DMRS_GRID_WR_IDX < 0) return;

     //--------------------------------------------------------------------------------------------------------
     // Unshift the channel in delay back to its original location and write to GMEM. This is a scattered write
     // (@todo: any opportunities to make it coalesced?)
     // Output format is N_BS_ANT x (N_DMRS_SYMS_FOCC x N_DMRS_SYMS_TOCC x N_DMRS_GRIDS_PER_PRB) x N_DMRS_INTERP_PRB_OUT_PER_CLUSTER
     // where N_DMRS_SYMS_FOCC x N_DMRS_SYMS_TOCC x N_DMRS_GRIDS_PER_PRB = N_LAYERS
     //const uint32_t DMRS_GRID_OFFSET_H = DMRS_GRID_WR_IDX * N_DMRS_SYMS_OCC;

 #ifdef CH_EST_COALESCED_WRITE
     // H (N_DATA_PRB*N_TONES_PER_PRB, N_LAYERS, N_BS_ANTS, NH)
     //read the number of rx antennas
     uint32_t N_BS_ANTS  = drvdUeGrpPrms.nRxAnt;
     TComplexStorage* pHEst = tHEst.addr + ((NH_IDX * N_BS_ANTS + BS_ANT_IDX) * N_LAYERS * N_DATA_PRB * N_TONES_PER_PRB);
 #endif


    // index of interpolated tone within cluster
    uint32_t CLUSTER_INTERP_TONE_IDX = DMRS_INTERP_TONE_IDX + HALF_N_EDGE_TONES;
    if(0 == PRB_CLUSTER_IDX) CLUSTER_INTERP_TONE_IDX = CLUSTER_INTERP_TONE_IDX - HALF_N_EDGE_TONES;                        // Lower edge
    if((N_PRB_CLUSTERS - 1) == PRB_CLUSTER_IDX) CLUSTER_INTERP_TONE_IDX = CLUSTER_INTERP_TONE_IDX + HALF_N_EDGE_TONES;     // Upper edge

    // check if estimated tone dropped
    if(N_DMRS_PRB_IN_PER_CLUSTER == 4)
    {
        if((filtIdx == UPPER_EDGE_INTERP_FILT_IDX) && (nPrbsMod2 == 1) && (DMRS_INTERP_TONE_IDX < 12))
            return;
    }

 #pragma unroll
     for(uint32_t i = 0; i < N_LAYERS; ++i)
     {
         if (i < nLayers) {
            uint32_t j = OCCIdx[i] & 0x3;
            uint32_t k = (OCCIdx[i] >> 2) & 0x1;
            if (DMRS_GRID_IDX == k) {
                shH(DMRS_INTERP_TONE_IDX, j, DMRS_GRID_IDX) *=
                type_convert<TComplexCompute>(tUnShiftSeq(CLUSTER_INTERP_TONE_IDX + gridShiftIdx)); //INTERP_DMRS_ABS_TONE_IDX
                if(nDmrsCdmGrpsNoData==1)
                {
                    shH(DMRS_INTERP_TONE_IDX, j, DMRS_GRID_IDX) *= cuGet<TComplexCompute>(SQRT2, 0.0f);
                }
#ifndef CH_EST_COALESCED_WRITE
                tHEst(BS_ANT_IDX, i, INTERP_DMRS_ABS_TONE_IDX, NH_IDX) =
                     type_convert<TComplexStorage>(shH(DMRS_INTERP_TONE_IDX, j, DMRS_GRID_IDX));

                ////Test/////
                //if (BS_ANT_IDX == 0 && i == 3 && INTERP_DMRS_ABS_TONE_IDX && !NH_IDX)
                //printf("minus itHEst = %f+j%f\n", tHEst(BS_ANT_IDX, i, INTERP_DMRS_ABS_TONE_IDX, NH_IDX).x-tHEst(BS_ANT_IDX, 1, INTERP_DMRS_ABS_TONE_IDX, NH_IDX).x, tHEst(BS_ANT_IDX, i, INTERP_DMRS_ABS_TONE_IDX, NH_IDX).y-tHEst(BS_ANT_IDX, 1, INTERP_DMRS_ABS_TONE_IDX, NH_IDX).y);
                /////////////

#else //fix me for flexbile DMRS port mapping
                pHEst[(DMRS_GRID_OFFSET_H + i) * N_DATA_PRB * N_TONES_PER_PRB + INTERP_DMRS_ABS_TONE_IDX] =
                     type_convert<TComplexStorage>(shH(DMRS_INTERP_TONE_IDX, i, DMRS_GRID_IDX));
#endif
            }
#ifdef ENABLE_DEBUG
#if 0
     printf("shH[%d][%d][%d] = %f+j%f\n", DMRS_INTERP_TONE_IDX, i, DMRS_GRID_IDX,
         shH(DMRS_INTERP_TONE_IDX, i, DMRS_GRID_IDX).x, shH(DMRS_INTERP_TONE_IDX, i, DMRS_GRID_IDX).y);
#endif
#if 0
     if(((6 == UE_GRP_IDX) || (7 == UE_GRP_IDX) || (8 == UE_GRP_IDX)) && (PRB_CLUSTER_IDX < 1))
     {
        TCompute hEstReal = type_convert<TCompute>(tHEst(BS_ANT_IDX, i, INTERP_DMRS_ABS_TONE_IDX, NH_IDX).x);
        TCompute hEstImag = type_convert<TCompute>(tHEst(BS_ANT_IDX, i, INTERP_DMRS_ABS_TONE_IDX, NH_IDX).y);
        printf("ueGrpIdx %d blockIdx.z %d tH[%d][%d][%d][%d] = %f+j%f\n", UE_GRP_IDX, blockIdx.z, BS_ANT_IDX, i, INTERP_DMRS_ABS_TONE_IDX, NH_IDX, hEstReal, hEstImag);
     }

#endif
#if 0
       printf("tUnshift[%d] = %f+j%f\n", INTERP_DMRS_ABS_TONE_IDX + gridShiftIdx,tUnShiftSeq(INTERP_DMRS_ABS_TONE_IDX + gridShiftIdx).x, tUnShiftSeq(INTERP_DMRS_ABS_TONE_IDX + gridShiftIdx).y);

       printf("tH[%d][%d][%d][%d] = %f+j%f\n", BS_ANT_IDX, DMRS_GRID_OFFSET_H + i, INTERP_DMRS_ABS_TONE_IDX, NH_IDX,
         tHEst(BS_ANT_IDX, DMRS_GRID_OFFSET_H + i, INTERP_DMRS_ABS_TONE_IDX, NH_IDX).x,
         tHEst(BS_ANT_IDX, DMRS_GRID_OFFSET_H + i, INTERP_DMRS_ABS_TONE_IDX, NH_IDX).y);
#endif
#endif
        }
     }
 }
 #endif


 template <typename TStorage,
           typename TDataRx,
           typename TCompute,
           uint32_t N_LAYERS,                          // # of layers (# of cols in H matrix) ** may be larger than the actual number of layers in the group
           uint32_t N_PRBS,
           uint32_t N_DMRS_GRIDS_PER_PRB,              // # of DMRS grids per PRB (2 or 3)
           uint32_t N_DMRS_SYMS>                       // # of time domain DMRS symbols (1,2 or 4)
 static __global__ void
 smallChEstKernel(puschRxChEstStatDescr_t* pStatDescr, puschRxChEstDynDescr_t* pDynDescr)
 {
     //--------------------------------------------------------------------------------------------------------
     // Setup local parameters based on descriptor
     puschRxChEstStatDescr_t& statDescr = *pStatDescr;
     // UE group processed by this thread block
     puschRxChEstDynDescr_t& dynDescr   = *pDynDescr;
     const uint32_t  UE_GRP_IDX         = dynDescr.hetCfgUeGrpMap[blockIdx.z];

     // BS antenna being processed by this thread block
     const uint32_t BS_ANT_IDX = blockIdx.y;

     cuphyPuschRxUeGrpPrms_t& drvdUeGrpPrms = dynDescr.pDrvdUeGrpPrms[UE_GRP_IDX];
     const uint16_t nRxAnt = drvdUeGrpPrms.nRxAnt;
     if(BS_ANT_IDX >= nRxAnt) return;

     const uint16_t  slotNum            = drvdUeGrpPrms.slotNum;
     const uint8_t   chEstTimeInst      = dynDescr.chEstTimeInst;
     const uint8_t   activeDmrsGridBmsk = drvdUeGrpPrms.activeDMRSGridBmsk;
     uint8_t*        OCCIdx             = drvdUeGrpPrms.OCCIdx;
     uint16_t        nLayers            = drvdUeGrpPrms.nLayers;
     const uint8_t   dmrsMaxLen         = drvdUeGrpPrms.dmrsMaxLen;
     const uint16_t  startPrb           = drvdUeGrpPrms.startPrb;
     const uint16_t  dmrsScId           = drvdUeGrpPrms.dmrsScrmId;
     const uint8_t   scid               = drvdUeGrpPrms.scid;
     const uint8_t   nDmrsCdmGrpsNoData = drvdUeGrpPrms.nDmrsCdmGrpsNoData;
     const uint8_t   enableTfPrcd       = drvdUeGrpPrms.enableTfPrcd;


     // Pointer to DMRS symbol used for channel estimation (single-symbol if maxLen = 1, double-symbol if maxLen = 2)
     const uint8_t* const pDmrsSymPos   = &drvdUeGrpPrms.dmrsSymLoc[chEstTimeInst*dmrsMaxLen];

     //--------------------------------------------------------------------------------------------------------
     typedef typename complex_from_scalar<TCompute>::type TComplexCompute;
     typedef typename complex_from_scalar<TDataRx>::type  TComplexDataRx;
     typedef typename complex_from_scalar<TStorage>::type TComplexStorage;

     // clang-format off
     tensor_ref<const TStorage>       tFreqInterpCoefs(statDescr.tPrmFreqInterpCoefsSmall.pAddr, statDescr.tPrmFreqInterpCoefsSmall.strides); // (N_TOTAL_DMRS_INTERP_GRID_TONES_PER_CLUSTER + N_INTER_DMRS_GRID_FREQ_SHIFT, N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER, 3), 3 filters: 1 for middle, 1 lower edge and 1 upper edge
 #if 1 // shift/unshift sequences same precision as data (FP16 or FP32)
     tensor_ref<const TComplexDataRx> tShiftSeq       (statDescr.tPrmShiftSeq.pAddr       , statDescr.tPrmShiftSeq.strides);        // (N_DATA_PRB*N_DMRS_GRID_TONES_PER_PRB, N_DMRS_SYMS)
     tensor_ref<const TComplexDataRx> tUnShiftSeq     (statDescr.tPrmUnShiftSeq.pAddr     , statDescr.tPrmUnShiftSeq.strides);      // (N_DATA_PRB*N_DMRS_INTERP_TONES_PER_GRID*N_DMRS_GRIDS_PER_PRB + N_INTER_DMRS_GRID_FREQ_SHIFT)
 #else // shift/unshift sequences same precision as channel estimates (typically FP32)
     tensor_ref<const TComplexStorage> tShiftSeq       (statDescr.tPrmShiftSeq.pAddr       , statDescr.tPrmShiftSeq.strides);        // (N_DATA_PRB*N_DMRS_GRID_TONES_PER_PRB, N_DMRS_SYMS)
     tensor_ref<const TComplexStorage> tUnShiftSeq     (statDescr.tPrmUnShiftSeq.pAddr     , statDescr.tPrmUnShiftSeq.strides);      // (N_DATA_PRB*N_DMRS_INTERP_TONES_PER_GRID*N_DMRS_GRIDS_PER_PRB + N_INTER_DMRS_GRID_FREQ_SHIFT)
 #endif
     tensor_ref<const TComplexDataRx> tDataRx        (drvdUeGrpPrms.tInfoDataRx.pAddr  , drvdUeGrpPrms.tInfoDataRx.strides);// (NF, ND, N_BS_ANTS)
     tensor_ref<TComplexStorage>      tHEst          (drvdUeGrpPrms.tInfoHEst.pAddr    , drvdUeGrpPrms.tInfoHEst.strides);
     tensor_ref<TComplexStorage>      tDbg           (drvdUeGrpPrms.tInfoChEstDbg.pAddr, drvdUeGrpPrms.tInfoChEstDbg.strides);
     // clang-format on

#ifdef ENABLE_DEBUG
     if((0 == threadIdx.x) && (0 == threadIdx.y) && (0 == threadIdx.z) && (0 == blockIdx.x) && (0 == blockIdx.y))
     {
        printf("%s\n: hetCfgUeGrpIdx %d ueGrpIdx %d nPrb %d startPb %d\n", __PRETTY_FUNCTION__, blockIdx.z, UE_GRP_IDX, N_PRBS, startPrb);
     }
#endif

     //--------------------------------------------------------------------------------------------------------
     // Dimensions and indices

     const uint32_t NH_IDX = chEstTimeInst;

     // Channel estimation expands tones in a DMRS grid (4 or 6, given by N_DMRS_GRID_TONES_PER_PRB) into a full PRB
     constexpr uint32_t N_DMRS_INTERP_TONES_PER_GRID = N_TONES_PER_PRB;

     // # of tones per DMRS grid in a PRB
     constexpr uint32_t N_DMRS_GRID_TONES_PER_PRB = N_TONES_PER_PRB / N_DMRS_GRIDS_PER_PRB;
     // Max permissible DMRS grids within a PRB based on spec
     constexpr uint32_t N_DMRS_TYPE1_GRIDS_PER_PRB = 2;
     constexpr uint32_t N_DMRS_TYPE2_GRIDS_PER_PRB = 3;
     static_assert(((N_DMRS_TYPE1_GRIDS_PER_PRB == N_DMRS_GRIDS_PER_PRB) || (N_DMRS_TYPE2_GRIDS_PER_PRB == N_DMRS_GRIDS_PER_PRB)),
                   "DMRS grid count exceeds max value");

     // Within a PRB, successive DMRS grids are shifted by 2 tones
     constexpr uint32_t N_INTER_DMRS_GRID_FREQ_SHIFT = get_inter_dmrs_grid_freq_shift(N_DMRS_GRIDS_PER_PRB);

     // Per grid tone counts present in input and output PRB clusters. These tones counts are expected to be equal
     constexpr uint32_t N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER        = N_DMRS_GRID_TONES_PER_PRB * N_PRBS;
     constexpr uint32_t N_TOTAL_DMRS_INTERP_GRID_TONES_PER_CLUSTER = N_DMRS_INTERP_TONES_PER_GRID * N_PRBS;

     // Total # of DMRS tones consumed by this thread block (this number should equal number of threads in
     // thread block since each DMRS tone is processed by a thread)
     constexpr uint32_t N_DMRS_TONES = N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER * N_DMRS_GRIDS_PER_PRB; // blockDim.x
     // Total # of interpolated DMRS tones produced by this thread block (this number should also equal number
     // of threads in thread block)
     constexpr uint32_t N_INTERP_TONES = N_TOTAL_DMRS_INTERP_GRID_TONES_PER_CLUSTER * N_DMRS_GRIDS_PER_PRB; // blockDim.x

     // DMRS descrambling
     constexpr uint32_t N_DMRS_DESCR_BITS_PER_TONE    = 2; // 1bit for I and 1 bit for Q
     constexpr uint32_t N_DMRS_DESCR_BITS_PER_CLUSTER = N_DMRS_DESCR_BITS_PER_TONE * N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER;
     // # of DMRS descrambler bits generated at one time
     constexpr uint32_t N_DMRS_DESCR_BITS_GEN = 32;
     // Round up to the next multiple of N_DMRS_DESCR_BITS_GEN plus 1 (+1 because DMRS_DESCR_PRB_CLUSTER_START_BIT_OFFSET
     // may be large enough to spill the descrambler bits to the next word)
     constexpr uint32_t N_DMRS_DESCR_WORDS =
         ((N_DMRS_DESCR_BITS_PER_CLUSTER + N_DMRS_DESCR_BITS_GEN - 1) / N_DMRS_DESCR_BITS_GEN) + 1;
     // round_up_to_next(N_DMRS_DESCR_BITS_PER_CLUSTER, N_DMRS_DESCR_BITS_GEN) + 1;

     const uint32_t ACTIVE_DMRS_GRID_BMSK = 0x3;

     // Per UE group descrambling ID
     uint16_t dmrsScramId = dmrsScId;

 #ifdef ENABLE_DEBUG

     if((0 == blockIdx.x) && (0 == blockIdx.y) && (0 == blockIdx.z) && (0 == threadIdx.x) && (0 == threadIdx.y) && (0 == threadIdx.z))
     {
         printf("Addr: tFreqInterpCoefs %lp tShiftSeq %lp tUnShiftSeq %lp tDataRx %lp tHEst %lp\n", tFreqInterpCoefs.pAddr, tShiftSeq.pAddr, tUnShiftSeq.pAddr, tDataRx.pAddr, tHEst.pAddr);

         printf("tFreqInterpCoefs strides[0] %d strides[1] %d strides[2] %d\n", tFreqInterpCoefs.strides[0], tFreqInterpCoefs.strides[1], tFreqInterpCoefs.strides[2]);
         printf("tShiftSeq        strides[0] %d strides[1] %d\n", tShiftSeq.strides[0], tShiftSeq.strides[1]);
         printf("tUnShiftSeq      strides[0] %d strides[1] %d\n", tUnShiftSeq.strides[0], tUnShiftSeq.strides[1]);

         printf("tDataRx strides[0] %d strides[1] %d strides[2] %d\n", tDataRx.strides[0], tDataRx.strides[1], tDataRx.strides[2]);
         printf("tHEst   strides[0] %d strides[1] %d strides[2] %d\n", tHEst.strides[0], tHEst.strides[1], tHEst.strides[2]);
         // printf("tDbg    strides[0] %d strides[1] %d strides[2] %d\n", tDbg.strides[0], tDbg.strides[1], tDbg.strides[2]);

         printf("dmrsScramId %d slotNum %d activeDmrsGridBmsk 0x%08x\n", dmrsScramId, slotNum, activeDmrsGridBmsk);
     }

     // printf("Block(%d %d %d) Thread(%d %d %d)\n", blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z);
     // if((0 != BS_ANT_IDX) || (0 != PRB_CLUSTER_IDX)) return;
     // printf("dmrsScramId %d slotNum %d activeDmrsGridBmsk 0x%08x", dmrsScramId, slotNum, activeDmrsGridBmsk);
     // if((0 == blockIdx.x) && (0 == blockIdx.y) && (0 == blockIdx.z) && (0 == threadIdx.x) && (0 == threadIdx.y) && (0 == threadIdx.z))
     //    printf("tDataRx strides[0] %d strides[1] %d strides[2] %d\n", tDataRx.strides[0], tDataRx.strides[1], tDataRx.strides[2]);
 #if 0
     printf("InterpCoefs[%d][%d][%d] = %f ShiftSeq[%d][%d] = %f+j%f UnShiftSeq[%d] %f+j%f DataRx[%d][%d][%d]= %f+j%f\n",
            0,0,0,
            tFreqInterpCoefs(0,0,0),
            0,0,
            tShiftSeq(0,0).x,
            tShiftSeq(0,0).y,
            0,
            tUnShiftSeq(0).x,
            tUnShiftSeq(0).y,
            0,0,0,
            tDataRx(0,0,0).x,
            tDataRx(0,0,0).y);
 #endif
 #endif

     const uint32_t THREAD_IDX = threadIdx.x;



     // Determine first PRB in the cluster being processed
     uint32_t prbClusterStartIdx = 0;

     uint32_t prbAbsStartIdx = prbClusterStartIdx + startPrb;
     // Absolute index of DMRS tone within the input OFDM symbol (used as index when loading tone from OFDM
     // symbol)
     const uint32_t DMRS_ABS_TONE_IDX = prbAbsStartIdx * N_TONES_PER_PRB + THREAD_IDX;

     // This index calculation intends to divvy up threads in the thread block for processing as follows:
     // the first group of N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER to process the first DMRS grid, the second group
     // to process the second DMRS grid and so on
     // Relative index of DMRS tone (within a DMRS grid) being processed by this thread
     const uint32_t DMRS_TONE_IDX        = THREAD_IDX % N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER;
     const uint32_t DMRS_INTERP_TONE_IDX = THREAD_IDX % N_TOTAL_DMRS_INTERP_GRID_TONES_PER_CLUSTER;

     // Index of DMRS grid in which the DMRS tone being loaded by this thread resides
     const uint32_t DMRS_GRID_IDX = THREAD_IDX / N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER;

     const uint32_t tempGridIdx = DMRS_GRID_IDX > 1 ? 1 : DMRS_GRID_IDX;

     uint8_t   activeTOCCBmsk     = drvdUeGrpPrms.activeTOCCBmsk[tempGridIdx];
     uint8_t   activeFOCCBmsk     = drvdUeGrpPrms.activeFOCCBmsk[tempGridIdx];
     uint32_t  N_DMRS_SYMS_TOCC   = activeTOCCBmsk == 3 ? 2 : 1;
     uint32_t  N_DMRS_SYMS_FOCC   = activeFOCCBmsk == 3 ? 2 : 1;

     // Index of DMRS grid which thread computes
     const uint32_t DMRS_GRID_IDX_COMPUTE = THREAD_IDX / N_TOTAL_DMRS_INTERP_GRID_TONES_PER_CLUSTER;

     // Index which enables extraction of DMRS tones of a given DMRS grid scattered within the PRB into one
     // contiguous set for processing. Note that the read from GMEM is coalesced and write into SMEM is scattered
     // @todo: check if index calculation can be simplified
     const uint32_t SMEM_DMRS_TONE_IDX = get_smem_dmrs_tone_idx(N_DMRS_GRIDS_PER_PRB, N_INTER_DMRS_GRID_FREQ_SHIFT, THREAD_IDX);
     const uint32_t SMEM_DMRS_GRID_IDX = get_smem_dmrs_grid_idx(N_DMRS_GRIDS_PER_PRB, N_INTER_DMRS_GRID_FREQ_SHIFT, THREAD_IDX);

     // Absolute index of descrambling + shift sequence element
     const uint32_t DMRS_DESCR_SHIFT_SEQ_START_IDX = prbAbsStartIdx * N_DMRS_GRID_TONES_PER_PRB;
     const uint32_t DMRS_DESCR_SHIFT_SEQ_ABS_IDX = DMRS_TONE_IDX;

     // Absolute index of interpolated tone produced by this thread
     const uint32_t INTERP_DMRS_ABS_TONE_IDX =  DMRS_INTERP_TONE_IDX;

     // Select which estimation filter to used based on size of prb cluster
     constexpr uint32_t filtIdx = N_PRBS - 1;


     // Select the shift in interpolation filter coefficients and delay shift based on grid index
     // (for e.g. for 2 DMRS grids and 48 tones per grid, multiply DMRS tone vector with top 48 rows for
     // DMRS_GRID_IDX 0 and bottom 48 rows for DMRS_GRID_IDX 1 to acheieve the effect of shift)
     uint32_t gridShiftIdx = get_inter_dmrs_grid_freq_shift_idx(N_DMRS_GRIDS_PER_PRB, DMRS_GRID_IDX_COMPUTE);

     // Section 5.2.1 in 3GPP TS 38.211
     // The fast-forward of 1600 prescribed by spec is already baked into the gold sequence generator
     constexpr uint32_t DMRS_DESCR_FF = 0; // 1600;

     // First descrambler bit index needed by this thread block
     // Note:The DMRS scrambling sequence is the same for all the DMRS grids. There are 2 sequences one for
     // scid 0 and other for scid 1 but the same sequences is reused for all DMRS grids
     const uint32_t DMRS_DESCR_PRB_CLUSTER_START_BIT =
         DMRS_DESCR_FF + (DMRS_DESCR_SHIFT_SEQ_START_IDX * N_DMRS_DESCR_BITS_PER_TONE);

     // The descrambling sequence generator outputs 32 descrambler bits at a time. Thus, compute the earliest
     // multiple of 32 bits which contains the descrambler bit of the first tone in the PRB cluster as the
     // start index
     const uint32_t DMRS_DESCR_GEN_ALIGNED_START_BIT =
         (DMRS_DESCR_PRB_CLUSTER_START_BIT / N_DMRS_DESCR_BITS_GEN) * N_DMRS_DESCR_BITS_GEN;
     // Offset to descrambler bit of the first tone in the PRB cluster
     const uint32_t DMRS_DESCR_PRB_CLUSTER_START_BIT_OFFSET =
         DMRS_DESCR_PRB_CLUSTER_START_BIT - DMRS_DESCR_GEN_ALIGNED_START_BIT;

     // DMRS descrambling bits generated correspond to subcarriers across frequency
     // e.g. 2 bits for tone0(grid 0) | 2 bits for tone1(grid 1) | 2 bits for tone 2(grid 0) | 2 bits for tone 3(grid 1) | ...
     const uint32_t DMRS_TONE_DESCR_BIT_IDX = DMRS_DESCR_PRB_CLUSTER_START_BIT_OFFSET +
                                              (DMRS_TONE_IDX * N_DMRS_DESCR_BITS_PER_TONE);
     const uint32_t DMRS_DESCR_SEQ_RD_BIT_IDX  = DMRS_TONE_DESCR_BIT_IDX % N_DMRS_DESCR_BITS_GEN;
     const uint32_t DMRS_DESCR_SEQ_RD_WORD_IDX = DMRS_TONE_DESCR_BIT_IDX / N_DMRS_DESCR_BITS_GEN;

     const uint32_t DMRS_DESCR_SEQ_WR_WORD_IDX = THREAD_IDX % N_DMRS_DESCR_WORDS;
     const uint32_t DMRS_DESCR_SEQ_WR_SYM_IDX  = THREAD_IDX / N_DMRS_DESCR_WORDS;

     // determine if thread used to load dmrs:
     const bool loadFlag = (THREAD_IDX < N_PRBS*N_TONES_PER_PRB) ? true : false;

 #if 0
     if((0 == blockIdx.x) && (0 == blockIdx.y) && (0 == blockIdx.z))
        printf("N_DMRS_DESCR_BITS_PER_CLUSTER %d N_DMRS_DESCR_WORDS %d DMRS_DESCR_PRB_CLUSTER_START_BIT %d DMRS_DESCR_GEN_ALIGNED_START_BIT %d, "
               "DMRS_DESCR_SEQ_RD_WORD_IDX %d, DMRS_DESCR_SEQ_RD_BIT_IDX %d, DMRS_DESCR_SEQ_WR_WORD_IDX %d, DMRS_DESCR_SEQ_WR_SYM_IDX %d\n",
               N_DMRS_DESCR_BITS_PER_CLUSTER, N_DMRS_DESCR_WORDS, DMRS_DESCR_PRB_CLUSTER_START_BIT, DMRS_DESCR_GEN_ALIGNED_START_BIT,
               DMRS_DESCR_SEQ_RD_WORD_IDX, DMRS_DESCR_SEQ_RD_BIT_IDX, DMRS_DESCR_SEQ_WR_WORD_IDX, DMRS_DESCR_SEQ_WR_SYM_IDX);
 #endif

     // Data layouts:
     // Global memory read into shared memory
     // N_DMRS_TONES x N_DMRS_SYMS -> N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER x N_DMRS_SYMS x N_DMRS_GRIDS_PER_PRB

     // tOCC removal
     // N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER x N_DMRS_SYMS x N_DMRS_GRIDS_PER_PRB ->
     // N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER x N_DMRS_SYMS_TOCC x N_DMRS_GRIDS_PER_PRB

     // fOCC removal
     // N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER x N_DMRS_SYMS_TOCC x N_DMRS_GRIDS_PER_PRB ->
     // N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER x N_DMRS_SYMS_FOCC x NUM_DMRS_SYMS_TOCC x N_DMRS_GRIDS_PER_PRB

     // Interpolation
     // N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER x N_DMRS_SYMS_FOCC x NUM_DMRS_SYMS_TOCC x N_DMRS_GRIDS_PER_PRB ->
     // N_TOTAL_DMRS_INTERP_GRID_TONES_PER_CLUSTER x N_DMRS_SYMS_FOCC x NUM_DMRS_SYMS_TOCC x N_DMRS_GRIDS_PER_PRB =
     // N_TOTAL_DMRS_INTERP_GRID_TONES_PER_CLUSTER x N_LAYERS x N_DMRS_GRIDS_PER_PRB

     //--------------------------------------------------------------------------------------------------------
     // Allocate shared memory

     constexpr uint32_t N_SMEM_ELEMS1 = N_DMRS_TONES * N_DMRS_SYMS; // (N_DMRS_TONES + N_DMRS_GRIDS_PER_PRB)*N_DMRS_SYMS;
     constexpr uint32_t N_SMEM_ELEMS2 = N_INTERP_TONES * N_DMRS_SYMS_OCC;
     constexpr uint32_t N_SMEM_ELEMS  = (N_SMEM_ELEMS1 > N_SMEM_ELEMS2) ? N_SMEM_ELEMS1 : N_SMEM_ELEMS2;
     // constexpr uint32_t N_SMEM_ELEMS  = (N_SMEM_ELEMS1 + N_SMEM_ELEMS2);
     // constexpr uint32_t N_SMEM_ELEMS  = max(N_SMEM_ELEMS1, N_SMEM_ELEMS2);

     __shared__ TComplexCompute smemBlk[N_SMEM_ELEMS];
     // overlay1
     block_3D<TComplexCompute*, N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER, N_DMRS_SYMS, N_DMRS_GRIDS_PER_PRB> shPilots(&smemBlk[0]);
     // overlay2
     block_3D<TComplexCompute*, N_TOTAL_DMRS_INTERP_GRID_TONES_PER_CLUSTER, N_DMRS_SYMS_OCC, N_DMRS_GRIDS_PER_PRB> shH(&smemBlk[0]);
     // block_3D<TComplexCompute*, N_TOTAL_DMRS_INTERP_GRID_TONES_PER_CLUSTER, N_DMRS_SYMS_OCC, N_DMRS_GRIDS_PER_PRB> shH(&smemBlk[shPilots.num_elem()]);
     static_assert((shPilots.num_elem() <= N_SMEM_ELEMS) && (shH.num_elem() <= N_SMEM_ELEMS), "Insufficient shared memory");

     __shared__ uint32_t descrWords[N_DMRS_SYMS][N_DMRS_DESCR_WORDS];

     //--------------------------------------------------------------------------------------------------------
     // Read DMRS tones into shared memory (separate the tones into different DMRS grids during the write)

     // Cache shift sequence in register
     TComplexCompute shiftSeq = type_convert<TComplexCompute>(tShiftSeq(DMRS_DESCR_SHIFT_SEQ_ABS_IDX, 0));

     if(loadFlag)
     {

     #pragma unroll
         for(uint32_t i = 0; i < N_DMRS_SYMS; ++i)
         {
             shPilots(SMEM_DMRS_TONE_IDX, i, SMEM_DMRS_GRID_IDX) =
                 type_convert<TComplexCompute>(tDataRx(DMRS_ABS_TONE_IDX, pDmrsSymPos[i], BS_ANT_IDX));

     #ifdef ENABLE_DEBUG
             printf("Pilots[%d][%d][%d] -> shPilots[%d][%d][%d] = %f+j%f, ShiftSeq[%d][%d] = %f+j%f\n",
                 DMRS_ABS_TONE_IDX,
                 pDmrsSymPos[i],
                 BS_ANT_IDX,
                 SMEM_DMRS_TONE_IDX,
                 i,
                 SMEM_DMRS_GRID_IDX,
                 shPilots(SMEM_DMRS_TONE_IDX, i, SMEM_DMRS_GRID_IDX).x,
                 shPilots(SMEM_DMRS_TONE_IDX, i, SMEM_DMRS_GRID_IDX).y,
                 DMRS_DESCR_SHIFT_SEQ_ABS_IDX,
                 0,
                 shiftSeq.x,
                 shiftSeq.y);
     #endif
         }

         // Compute the descsrambler sequence
         const uint32_t TWO_POW_17 = bit(17);

         if(DMRS_DESCR_SEQ_WR_SYM_IDX < N_DMRS_SYMS)
         {
             uint32_t symIdx = pDmrsSymPos[DMRS_DESCR_SEQ_WR_SYM_IDX];

             // see 38.211 section 6.4.1.1.1.1
             uint32_t cInit = TWO_POW_17 * (slotNum * OFDM_SYMBOLS_PER_SLOT + symIdx + 1) * (2 * dmrsScramId + 1) + (2 * dmrsScramId) + scid;
             cInit &= ~bit(31);

             // descrWords[DMRS_DESCR_SEQ_WR_SYM_IDX][DMRS_DESCR_SEQ_WR_WORD_IDX] =
             //  __brev(gold32(cInit, (DMRS_DESCR_GEN_ALIGNED_START_BIT + DMRS_DESCR_SEQ_WR_WORD_IDX*N_DMRS_DESCR_BITS_GEN)));

             descrWords[DMRS_DESCR_SEQ_WR_SYM_IDX][DMRS_DESCR_SEQ_WR_WORD_IDX] =
                 gold32(cInit, (DMRS_DESCR_GEN_ALIGNED_START_BIT + DMRS_DESCR_SEQ_WR_WORD_IDX * N_DMRS_DESCR_BITS_GEN));
     #if 0
             printf("symIdx %d, DMRS_DESCR_SEQ_WR_WORD_IDX %d, cInit 0x%08x, DMRS_DESCR_GEN_ALIGNED_START_BIT %d, descrWords[%d][%d] 0x%08x\n",
                 symIdx, DMRS_DESCR_SEQ_WR_WORD_IDX, cInit,
                 (DMRS_DESCR_GEN_ALIGNED_START_BIT + DMRS_DESCR_SEQ_WR_WORD_IDX*N_DMRS_DESCR_BITS_GEN),
                 DMRS_DESCR_SEQ_WR_SYM_IDX, DMRS_DESCR_SEQ_WR_WORD_IDX, descrWords[DMRS_DESCR_SEQ_WR_SYM_IDX][DMRS_DESCR_SEQ_WR_WORD_IDX]);
     #endif
         }
     }

     // To ensure coalesced reads, input DMRS tones are read preserving input order but swizzled while writing
     // to shared memory. Thus each thread may not process the same tone which it wrote to shared memory
     thread_block const& thisThrdBlk = this_thread_block();
     __syncthreads();

     //--------------------------------------------------------------------------------------------------------
     // Apply de-scrambling + delay domain centering sequence for tone index processed by this thread across all
     // DMRS symbols
     const TCompute RECIPROCAL_SQRT2 = 0.7071068f;
     const TCompute SQRT2            = 1.41421356f;
     TComplexCompute    avg[N_TOCC]{};


     if(loadFlag)
     {
     #pragma unroll
         for(uint32_t i = 0; i < N_DMRS_SYMS; ++i)
         {
             if(enableTfPrcd==0)
             {
                 int8_t descrIBit = (descrWords[i][DMRS_DESCR_SEQ_RD_WORD_IDX] >> DMRS_DESCR_SEQ_RD_BIT_IDX) & 0x1;
                 int8_t descrQBit = (descrWords[i][DMRS_DESCR_SEQ_RD_WORD_IDX] >> (DMRS_DESCR_SEQ_RD_BIT_IDX + 1)) & 0x1;

                 TComplexCompute descrCode =
                     cuConj(cuGet<TComplexCompute>((1 - 2 * descrIBit) * RECIPROCAL_SQRT2, (1 - 2 * descrQBit) * RECIPROCAL_SQRT2));
                 TComplexCompute descrShiftSeq = shiftSeq * descrCode;

    #ifdef ENABLE_DEBUG
                 TComplexCompute descrShiftPilot = shPilots(DMRS_TONE_IDX, i, DMRS_GRID_IDX) * descrShiftSeq;
                 printf("descrShiftAbsIdx: %d, shPilots[%d][%d][%d] (%f+j%f) * DescrShiftSeq[%d][%d] (%f+j%f) = %f+j%f, ShiftSeq = %f+j%f, DescrCode = %f+j%f, descrIQ (%d,%d) descrWordIdx %d descrBitIdx %d\n",
                     DMRS_DESCR_SHIFT_SEQ_ABS_IDX,
                     DMRS_TONE_IDX,
                     i,
                     DMRS_GRID_IDX,
                     shPilots(DMRS_TONE_IDX, i, DMRS_GRID_IDX).x,
                     shPilots(DMRS_TONE_IDX, i, DMRS_GRID_IDX).y,
                     DMRS_TONE_IDX,
                     i,
                     descrShiftSeq.x,
                     descrShiftSeq.y,
                     descrShiftPilot.x,
                     descrShiftPilot.y,
                     shiftSeq.x,
                     shiftSeq.y,
                     descrCode.x,
                     descrCode.y,
                     descrIBit,
                     descrQBit,
                     DMRS_DESCR_SEQ_RD_WORD_IDX,
                     DMRS_DESCR_SEQ_RD_BIT_IDX);

                 if((0 == DMRS_GRID_IDX) && (((0 == prbAbsStartIdx) && (DMRS_TONE_IDX < (N_EDGE_PRB * N_DMRS_GRID_TONES_PER_PRB))) || ((0 != prbAbsStartIdx) && (prbAbsStartIdx + N_DMRS_PRB_IN_PER_CLUSTER) <= N_DATA_PRB)))
                 {
         #if 0
                 TComplexCompute descrShiftPilot = shPilots(DMRS_TONE_IDX, i, DMRS_GRID_IDX) * descrShiftSeq;
                 printf("descrShiftAbsIdx: %d shPilots[%d][%d][%d] (%f+j%f) * DescrShiftSeq[%d][%d] (%f+j%f) = %f+j%f, ShiftSeq = %f+j%f, DescrCode = %f+j%f, descrIQ (%d,%d) descrWordIdx %d descrBitIdx %d\n",
                         DMRS_DESCR_SHIFT_SEQ_ABS_IDX,
                         DMRS_TONE_IDX,
                         i,
                         DMRS_GRID_IDX,
                         shPilots(DMRS_TONE_IDX, i, DMRS_GRID_IDX).x,
                         shPilots(DMRS_TONE_IDX, i, DMRS_GRID_IDX).y,
                         DMRS_TONE_IDX,
                         i,
                         descrShiftSeq.x,
                         descrShiftSeq.y,
                         descrShiftPilot.x,
                         descrShiftPilot.y,
                         shiftSeq.x,
                         shiftSeq.y,
                         descrCode.x,
                         descrCode.y,
                         descrIBit,
                         descrQBit,
                         DMRS_DESCR_SEQ_RD_WORD_IDX,
                         DMRS_DESCR_SEQ_RD_BIT_IDX);
#endif

                     // tDbg(DMRS_DESCR_SHIFT_SEQ_ABS_IDX, i, 0, 0) = type_convert<TComplexStorage>(shiftSeq);
                     // tDbg(DMRS_DESCR_SHIFT_SEQ_ABS_IDX, i, 0, 0) = type_convert<TComplexStorage>(descrShiftSeq);
                     tDbg(DMRS_DESCR_SHIFT_SEQ_ABS_IDX, i, 0, 0) = type_convert<TComplexStorage>(descrCode);
                 }
#endif // ENABLE_DEBUG

                 shPilots(DMRS_TONE_IDX, i, DMRS_GRID_IDX) *= descrShiftSeq;
             }
             else if(enableTfPrcd==1)
             {
                 uint16_t M_ZC  = N_DMRS_GRID_TONES_PER_PRB * N_PRBS; //different from DMRS_ABS_TONE_IDX
                 int u = 0;
                 int v = 0;

                 if(drvdUeGrpPrms.optionalDftSOfdm)
                 {
                     u = (int)drvdUeGrpPrms.lowPaprGroupNumber;
                     v = (int)drvdUeGrpPrms.lowPaprSequenceNumber;
                 }
                 else
                 {
                     const uint32_t puschIdentity          = drvdUeGrpPrms.puschIdentity;
                     const uint8_t  groupOrSequenceHopping = drvdUeGrpPrms.groupOrSequenceHopping;
                     const uint8_t  N_symb_slot            = drvdUeGrpPrms.N_symb_slot;

                     int f_gh       = 0;

                     if(groupOrSequenceHopping==1)
                     {
                         uint32_t cInit = floor(puschIdentity/30);
                         for(int m = 0; m < 8; m++)
                         {
                             uint32_t idxSeq = 8 * (slotNum * N_symb_slot + pDmrsSymPos[i]) + m;
                             f_gh = f_gh + ((gold32(cInit, idxSeq) >> (idxSeq % 32)) & 0x1) * (1 << m);
                         }
                         f_gh = f_gh % 30;
      //                   if((blockIdx.x==0)&&(blockIdx.y==0)&&(threadIdx.x==0)&&(threadIdx.y==0))
      //                   {
      //                       printf("f_gh[%d]\n", f_gh);
      //                   }
                     }
                     else if(groupOrSequenceHopping==2)
                     {
                         if(M_ZC > 6 * N_TONES_PER_PRB)
                         {
                             uint32_t idxSeq = slotNum * N_symb_slot + pDmrsSymPos[i];
                             v = (gold32(puschIdentity, idxSeq) >> (idxSeq % 32)) & 0x1;

      //                       if((blockIdx.x==0)&&(blockIdx.y==0)&&(threadIdx.x==0)&&(threadIdx.y==0))
      //                       {
      //                           printf("idxSeq[%d]v[%d]\n", idxSeq, v);
      //                       }
                         }
                     }

                     u = (f_gh + puschIdentity)%30;
                 }
                 uint16_t rIdx = prbClusterStartIdx * N_DMRS_GRID_TONES_PER_PRB + DMRS_TONE_IDX;
#ifdef ENABLE_COMMON_DFTSOFDM_DESCRCODE_SUBROUTINE
                 float2 descrCode = gen_pusch_dftsofdm_descrcode(M_ZC, rIdx, u, v, N_PRBS, d_phi_6, d_phi_12, d_phi_18, d_phi_24, d_primeNums);
#else
                 float2 descrCode = gen_pusch_dftsofdm_descrcode(M_ZC, rIdx, u, v, N_PRBS);
#endif
                 TComplexCompute descrShiftSeq = shiftSeq * cuConj(cuGet<TComplexCompute>(descrCode.x, descrCode.y));
                 shPilots(DMRS_TONE_IDX, i, DMRS_GRID_IDX) *= descrShiftSeq;
             }
         }

         //--------------------------------------------------------------------------------------------------------
         // Time domain cover code removal
         constexpr TCompute AVG_SCALE = cuGet<TCompute>(1u) / cuGet<TCompute>(N_DMRS_SYMS);
         // TComplexCompute    avg[N_DMRS_SYMS_TOCC]{};

         int32_t tOCCIdx = 0;

     #pragma unroll
         for(int32_t i = 0; i < 2; ++i)
         {
            int32_t temp = (activeTOCCBmsk >> i) & 0x1;
            if (!temp) {
                continue;
            }
     #pragma unroll
             for(int32_t j = 0; j < N_DMRS_SYMS; ++j)
             {
                 // For first tOCC (i = 0) output, multiply all DMRS symbols with +1 and average
                 // For second tOCC (i = 1) output, multiply even DMRS symbols with +1, odd DMRS symbols with -1 and average
                 int32_t sign = (-(i & j)) | 1;
                 avg[tOCCIdx] += (shPilots(DMRS_TONE_IDX, j, DMRS_GRID_IDX) * (sign * AVG_SCALE));
     #ifdef ENABLE_DEBUG
                 TComplexCompute prod = (shPilots(DMRS_TONE_IDX, j, DMRS_GRID_IDX) * (sign * AVG_SCALE));
                 printf("sign*AVG_SCALE %f Pilot[%d][%d][%d] = %f+j%f avg[%d] = %f+j%f, prod = %f+j%f\n",
                     sign * AVG_SCALE,
                     DMRS_TONE_IDX,
                     j,
                     DMRS_GRID_IDX,
                     shPilots(DMRS_TONE_IDX, j, DMRS_GRID_IDX).x,
                     shPilots(DMRS_TONE_IDX, j, DMRS_GRID_IDX).y,
                     i,
                     avg[i].x,
                     avg[i].y,
                     prod.x,
                     prod.y);
     #endif
             }
             tOCCIdx++;
         }
     }

     // shPilots and shH are overlaid in shared memory and can have different sizes (based on config). For this reason
     // ensure shPilots access from all threads is completed before writing into shH
     __syncthreads();

     //--------------------------------------------------------------------------------------------------------
     // Apply frequecy domain cover code and store inplace in shared memory
     // Multiply even tones with +1 and odd tones with -1

     if(loadFlag)
     {
         // Note that the loop termination count below is tOCC symbol count
     #pragma unroll
         for(int32_t i = 0; i < N_DMRS_SYMS_TOCC; ++i)
         {
            int32_t fOCCIdx = 0;

     #pragma unroll
             for(int32_t j = 0; j < 2; ++j)
             {
                int32_t temp = (activeFOCCBmsk >> j) & 0x1;
                if (!temp) {
                    continue;
                }
                 // First fOCC output: multiply all tones by +1s
                 // Second fOCC output: multiply even tones by +1s and odd tones by -1s
                 int32_t sign                                                  = (-(DMRS_TONE_IDX & j)) | 1;
                 shH(DMRS_TONE_IDX, (N_DMRS_SYMS_FOCC * i) + fOCCIdx, DMRS_GRID_IDX) = avg[i] * sign;
     #ifdef ENABLE_DEBUG
                 printf("PilotsPostOCC[%d][%d][%d] = %f+j%f\n",
                     DMRS_TONE_IDX,
                     (N_DMRS_SYMS_FOCC * i) + j,
                     DMRS_GRID_IDX,
                     cuReal(shH(DMRS_TONE_IDX, (N_DMRS_SYMS_FOCC * i) + j, DMRS_GRID_IDX)),
                     cuImag(shH(DMRS_TONE_IDX, (N_DMRS_SYMS_FOCC * i) + j, DMRS_GRID_IDX)));
     #endif
                 fOCCIdx++;
             }
         }
     }

     // Ensure all threads complete writing results to shared memory since each thread computing an inner product
     // during interpolation stage will use results from other threads in the thread block
     __syncthreads();

     //--------------------------------------------------------------------------------------------------------
     // Interpolate (matrix-vector multiply)

     // early exit for inactive grids
     const int32_t DMRS_GRID_WR_IDX = DMRS_GRID_WR_IDX_TBL[activeDmrsGridBmsk & ACTIVE_DMRS_GRID_BMSK][DMRS_GRID_IDX_COMPUTE];
     // if(!is_set(bit(DMRS_GRID_IDX), activeDmrsGridBmsk) || (DMRS_GRID_WR_IDX < 0)) return;
     if(DMRS_GRID_WR_IDX < 0) return;

     activeTOCCBmsk     = drvdUeGrpPrms.activeTOCCBmsk[DMRS_GRID_IDX_COMPUTE];
     activeFOCCBmsk     = drvdUeGrpPrms.activeFOCCBmsk[DMRS_GRID_IDX_COMPUTE];
     N_DMRS_SYMS_TOCC   = activeTOCCBmsk == 3 ? 2 : 1;
     N_DMRS_SYMS_FOCC   = activeFOCCBmsk == 3 ? 2 : 1;

     for(uint32_t i = 0; i < (N_DMRS_SYMS_FOCC*N_DMRS_SYMS_TOCC); ++i)
     {
         TComplexCompute innerProd{};

         // H = W x Y: (N_INTERP_TONES x N_DMRS_TONES) x (N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER x N_DMRS_SYMS_OCC)
         // Each thread selects one row of W and computes N_DMRS_TONES length inner product to produce one interpolated
         // tone of H
         for(uint32_t j = 0; j < N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER; ++j)
         {
             TCompute interpCoef = type_convert<TCompute>(tFreqInterpCoefs(DMRS_INTERP_TONE_IDX + gridShiftIdx, j, filtIdx));
             innerProd += (shH(j, i, DMRS_GRID_IDX_COMPUTE) * interpCoef);
         }
         // Wait for all threads to complete their inner products before updating the shared memory inplace
         // The sync is needed because shPilots and shH are overlaid
         thisThrdBlk.sync();

         shH(DMRS_INTERP_TONE_IDX, i, DMRS_GRID_IDX_COMPUTE) = innerProd;

 #ifdef ENABLE_DEBUG
         printf("InterpPilots[%d][%d][%d] = %f+j%f\n", DMRS_INTERP_TONE_IDX, i, DMRS_GRID_IDX, innerProd.x, innerProd.y);
 #endif
     }

     // Wait for shared memory writes to complete from all threads
     thisThrdBlk.sync();

     //--------------------------------------------------------------------------------------------------------
     // Unshift the channel in delay back to its original location and write to GMEM. This is a scattered write
     // (@todo: any opportunities to make it coalesced?)
     // Output format is N_BS_ANT x (N_DMRS_SYMS_FOCC x N_DMRS_SYMS_TOCC x N_DMRS_GRIDS_PER_PRB) x N_DMRS_INTERP_PRB_OUT_PER_CLUSTER
     // where N_DMRS_SYMS_FOCC x N_DMRS_SYMS_TOCC x N_DMRS_GRIDS_PER_PRB = N_LAYERS
     //const uint32_t DMRS_GRID_OFFSET_H = DMRS_GRID_WR_IDX * N_DMRS_SYMS_OCC;

 // #ifdef CH_EST_COALESCED_WRITE
 //     // H (N_DATA_PRB*N_TONES_PER_PRB, N_LAYERS, N_BS_ANTS, NH)
 //     // read the number of rx antennas
 //     uint32_t N_BS_ANTS = drvdUeGrpPrms.nRxAnt;
 //     TComplexStorage* pHEst = tHEst.addr + ((NH_IDX * N_BS_ANTS + BS_ANT_IDX) * N_LAYERS * N_DATA_PRB * N_TONES_PER_PRB);
 // #endif


 // index of interpolated tone within cluster
 uint32_t CLUSTER_INTERP_TONE_IDX = DMRS_INTERP_TONE_IDX;

 #pragma unroll
     for(uint32_t i = 0; i < N_LAYERS; ++i)
     {
        if (i < nLayers) {
            uint32_t j = OCCIdx[i] & 0x3;
            uint32_t k = (OCCIdx[i] >> 2) & 0x1;
            if (DMRS_GRID_IDX_COMPUTE == k) {
                shH(DMRS_INTERP_TONE_IDX, j, DMRS_GRID_IDX_COMPUTE) *=
                    type_convert<TComplexCompute>(tUnShiftSeq(CLUSTER_INTERP_TONE_IDX + gridShiftIdx)); //INTERP_DMRS_ABS_TONE_IDX
                if(nDmrsCdmGrpsNoData==1)
                {
                    shH(DMRS_INTERP_TONE_IDX, j, DMRS_GRID_IDX) *= cuGet<TComplexCompute>(SQRT2, 0.0f);
                }

 #ifndef CH_EST_COALESCED_WRITE
     tHEst(BS_ANT_IDX, i, INTERP_DMRS_ABS_TONE_IDX, NH_IDX) =
             type_convert<TComplexStorage>(shH(DMRS_INTERP_TONE_IDX, j, DMRS_GRID_IDX_COMPUTE));
 #else //fix me for flexbile DMRS port mapping
         pHEst[(DMRS_GRID_OFFSET_H + i) * N_DATA_PRB * N_TONES_PER_PRB + INTERP_DMRS_ABS_TONE_IDX] =
             type_convert<TComplexStorage>(shH(DMRS_INTERP_TONE_IDX, i, DMRS_GRID_IDX_COMPUTE));
 #endif
            }
 #ifdef ENABLE_DEBUG
 #if 0
      printf("shH[%d][%d][%d] = %f+j%f\n", DMRS_INTERP_TONE_IDX, i, DMRS_GRID_IDX,
          shH(DMRS_INTERP_TONE_IDX, i, DMRS_GRID_IDX).x, shH(DMRS_INTERP_TONE_IDX, i, DMRS_GRID_IDX).y);
 #endif
 #if 0
       printf("tH[%d][%d][%d][%d] = %f+j%f\n", BS_ANT_IDX, DMRS_GRID_OFFSET_H + i, INTERP_DMRS_ABS_TONE_IDX, NH_IDX,
          tHEst(BS_ANT_IDX, DMRS_GRID_OFFSET_H + i, INTERP_DMRS_ABS_TONE_IDX, NH_IDX).x,
          tHEst(BS_ANT_IDX, DMRS_GRID_OFFSET_H + i, INTERP_DMRS_ABS_TONE_IDX, NH_IDX).y);
 #endif
 #if 0
        printf("tUnshift[%d] = %f+j%f\n", INTERP_DMRS_ABS_TONE_IDX + gridShiftIdx,tUnShiftSeq(INTERP_DMRS_ABS_TONE_IDX + gridShiftIdx).x, tUnShiftSeq(INTERP_DMRS_ABS_TONE_IDX + gridShiftIdx).y);

        printf("tH[%d][%d][%d][%d] = %f+j%f\n", BS_ANT_IDX, DMRS_GRID_OFFSET_H + i, INTERP_DMRS_ABS_TONE_IDX, NH_IDX,
          tHEst(BS_ANT_IDX, DMRS_GRID_OFFSET_H + i, INTERP_DMRS_ABS_TONE_IDX, NH_IDX).x,
          tHEst(BS_ANT_IDX, DMRS_GRID_OFFSET_H + i, INTERP_DMRS_ABS_TONE_IDX, NH_IDX).y);
 #endif
 #endif
        }
     }
 } //smallChEstKernel
 // #endif

 template <typename TStorage,
           typename TDataRx,
           typename TCompute,
           uint32_t N_LAYERS,                          // # of layers (# of cols in H matrix) ** may be larger than the actual number of layers in the group
           uint32_t N_PRBS,
           uint32_t N_DMRS_GRIDS_PER_PRB,              // # of DMRS grids per PRB (2 or 3)
           uint32_t N_DMRS_SYMS>                       // # of time domain DMRS symbols (1,2 or 4)
 static __global__ void
 smallChEstNoDftSOfdmKernel(puschRxChEstStatDescr_t* pStatDescr, puschRxChEstDynDescr_t* pDynDescr)
 {
     //--------------------------------------------------------------------------------------------------------
     // Setup local parameters based on descriptor
     puschRxChEstStatDescr_t& statDescr = *pStatDescr;
     // UE group processed by this thread block
     puschRxChEstDynDescr_t& dynDescr   = *pDynDescr;
     const uint32_t  UE_GRP_IDX         = dynDescr.hetCfgUeGrpMap[blockIdx.z];

     // BS antenna being processed by this thread block
     const uint32_t BS_ANT_IDX = blockIdx.y;

     cuphyPuschRxUeGrpPrms_t& drvdUeGrpPrms = dynDescr.pDrvdUeGrpPrms[UE_GRP_IDX];
     const uint16_t nRxAnt = drvdUeGrpPrms.nRxAnt;
     if(BS_ANT_IDX >= nRxAnt) return;

     const uint16_t  slotNum            = drvdUeGrpPrms.slotNum;
     const uint8_t   chEstTimeInst      = dynDescr.chEstTimeInst;
     const uint8_t   activeDmrsGridBmsk = drvdUeGrpPrms.activeDMRSGridBmsk;
     uint8_t*        OCCIdx             = drvdUeGrpPrms.OCCIdx;
     uint16_t        nLayers            = drvdUeGrpPrms.nLayers;
     const uint8_t   dmrsMaxLen         = drvdUeGrpPrms.dmrsMaxLen;
     const uint16_t  startPrb           = drvdUeGrpPrms.startPrb;
     const uint16_t  dmrsScId           = drvdUeGrpPrms.dmrsScrmId;
     const uint8_t   scid               = drvdUeGrpPrms.scid;
     const uint8_t   nDmrsCdmGrpsNoData = drvdUeGrpPrms.nDmrsCdmGrpsNoData;


     // Pointer to DMRS symbol used for channel estimation (single-symbol if maxLen = 1, double-symbol if maxLen = 2)
     const uint8_t* const pDmrsSymPos        = &drvdUeGrpPrms.dmrsSymLoc[chEstTimeInst*dmrsMaxLen];

     //--------------------------------------------------------------------------------------------------------
     typedef typename complex_from_scalar<TCompute>::type TComplexCompute;
     typedef typename complex_from_scalar<TDataRx>::type  TComplexDataRx;
     typedef typename complex_from_scalar<TStorage>::type TComplexStorage;

     // clang-format off
     tensor_ref<const TStorage>       tFreqInterpCoefs(statDescr.tPrmFreqInterpCoefsSmall.pAddr, statDescr.tPrmFreqInterpCoefsSmall.strides); // (N_TOTAL_DMRS_INTERP_GRID_TONES_PER_CLUSTER + N_INTER_DMRS_GRID_FREQ_SHIFT, N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER, 3), 3 filters: 1 for middle, 1 lower edge and 1 upper edge
 #if 1 // shift/unshift sequences same precision as data (FP16 or FP32)
     tensor_ref<const TComplexDataRx> tShiftSeq       (statDescr.tPrmShiftSeq.pAddr       , statDescr.tPrmShiftSeq.strides);        // (N_DATA_PRB*N_DMRS_GRID_TONES_PER_PRB, N_DMRS_SYMS)
     tensor_ref<const TComplexDataRx> tUnShiftSeq     (statDescr.tPrmUnShiftSeq.pAddr     , statDescr.tPrmUnShiftSeq.strides);      // (N_DATA_PRB*N_DMRS_INTERP_TONES_PER_GRID*N_DMRS_GRIDS_PER_PRB + N_INTER_DMRS_GRID_FREQ_SHIFT)
 #else // shift/unshift sequences same precision as channel estimates (typically FP32)
     tensor_ref<const TComplexStorage> tShiftSeq       (statDescr.tPrmShiftSeq.pAddr       , statDescr.tPrmShiftSeq.strides);        // (N_DATA_PRB*N_DMRS_GRID_TONES_PER_PRB, N_DMRS_SYMS)
     tensor_ref<const TComplexStorage> tUnShiftSeq     (statDescr.tPrmUnShiftSeq.pAddr     , statDescr.tPrmUnShiftSeq.strides);      // (N_DATA_PRB*N_DMRS_INTERP_TONES_PER_GRID*N_DMRS_GRIDS_PER_PRB + N_INTER_DMRS_GRID_FREQ_SHIFT)
 #endif
     tensor_ref<const TComplexDataRx> tDataRx        (drvdUeGrpPrms.tInfoDataRx.pAddr  , drvdUeGrpPrms.tInfoDataRx.strides);// (NF, ND, N_BS_ANTS)
     tensor_ref<TComplexStorage>      tHEst          (drvdUeGrpPrms.tInfoHEst.pAddr    , drvdUeGrpPrms.tInfoHEst.strides);
     tensor_ref<TComplexStorage>      tDbg           (drvdUeGrpPrms.tInfoChEstDbg.pAddr, drvdUeGrpPrms.tInfoChEstDbg.strides);
     // clang-format on

#ifdef ENABLE_DEBUG
     if((0 == threadIdx.x) && (0 == threadIdx.y) && (0 == threadIdx.z) && (0 == blockIdx.x) && (0 == blockIdx.y))
     {
        printf("%s\n: hetCfgUeGrpIdx %d ueGrpIdx %d nPrb %d startPb %d\n", __PRETTY_FUNCTION__, blockIdx.z, UE_GRP_IDX, N_PRBS, startPrb);
     }
#endif

     //--------------------------------------------------------------------------------------------------------
     // Dimensions and indices

     const uint32_t NH_IDX = chEstTimeInst;

     // Channel estimation expands tones in a DMRS grid (4 or 6, given by N_DMRS_GRID_TONES_PER_PRB) into a full PRB
     constexpr uint32_t N_DMRS_INTERP_TONES_PER_GRID = N_TONES_PER_PRB;

     // # of tones per DMRS grid in a PRB
     constexpr uint32_t N_DMRS_GRID_TONES_PER_PRB = N_TONES_PER_PRB / N_DMRS_GRIDS_PER_PRB;
     // Max permissible DMRS grids within a PRB based on spec
     constexpr uint32_t N_DMRS_TYPE1_GRIDS_PER_PRB = 2;
     constexpr uint32_t N_DMRS_TYPE2_GRIDS_PER_PRB = 3;
     static_assert(((N_DMRS_TYPE1_GRIDS_PER_PRB == N_DMRS_GRIDS_PER_PRB) || (N_DMRS_TYPE2_GRIDS_PER_PRB == N_DMRS_GRIDS_PER_PRB)),
                   "DMRS grid count exceeds max value");

     // Within a PRB, successive DMRS grids are shifted by 2 tones
     constexpr uint32_t N_INTER_DMRS_GRID_FREQ_SHIFT = get_inter_dmrs_grid_freq_shift(N_DMRS_GRIDS_PER_PRB);

     // Per grid tone counts present in input and output PRB clusters. These tones counts are expected to be equal
     constexpr uint32_t N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER        = N_DMRS_GRID_TONES_PER_PRB * N_PRBS;
     constexpr uint32_t N_TOTAL_DMRS_INTERP_GRID_TONES_PER_CLUSTER = N_DMRS_INTERP_TONES_PER_GRID * N_PRBS;

     // Total # of DMRS tones consumed by this thread block (this number should equal number of threads in
     // thread block since each DMRS tone is processed by a thread)
     constexpr uint32_t N_DMRS_TONES = N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER * N_DMRS_GRIDS_PER_PRB; // blockDim.x
     // Total # of interpolated DMRS tones produced by this thread block (this number should also equal number
     // of threads in thread block)
     constexpr uint32_t N_INTERP_TONES = N_TOTAL_DMRS_INTERP_GRID_TONES_PER_CLUSTER * N_DMRS_GRIDS_PER_PRB; // blockDim.x

     // DMRS descrambling
     constexpr uint32_t N_DMRS_DESCR_BITS_PER_TONE    = 2; // 1bit for I and 1 bit for Q
     constexpr uint32_t N_DMRS_DESCR_BITS_PER_CLUSTER = N_DMRS_DESCR_BITS_PER_TONE * N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER;
     // # of DMRS descrambler bits generated at one time
     constexpr uint32_t N_DMRS_DESCR_BITS_GEN = 32;
     // Round up to the next multiple of N_DMRS_DESCR_BITS_GEN plus 1 (+1 because DMRS_DESCR_PRB_CLUSTER_START_BIT_OFFSET
     // may be large enough to spill the descrambler bits to the next word)
     constexpr uint32_t N_DMRS_DESCR_WORDS =
         ((N_DMRS_DESCR_BITS_PER_CLUSTER + N_DMRS_DESCR_BITS_GEN - 1) / N_DMRS_DESCR_BITS_GEN) + 1;
     // round_up_to_next(N_DMRS_DESCR_BITS_PER_CLUSTER, N_DMRS_DESCR_BITS_GEN) + 1;

     const uint32_t ACTIVE_DMRS_GRID_BMSK = 0x3;

     // Per UE group descrambling ID
     uint16_t dmrsScramId = dmrsScId;

 #ifdef ENABLE_DEBUG

     if((0 == blockIdx.x) && (0 == blockIdx.y) && (0 == blockIdx.z) && (0 == threadIdx.x) && (0 == threadIdx.y) && (0 == threadIdx.z))
     {
         printf("Addr: tFreqInterpCoefs %lp tShiftSeq %lp tUnShiftSeq %lp tDataRx %lp tHEst %lp\n", tFreqInterpCoefs.pAddr, tShiftSeq.pAddr, tUnShiftSeq.pAddr, tDataRx.pAddr, tHEst.pAddr);

         printf("tFreqInterpCoefs strides[0] %d strides[1] %d strides[2] %d\n", tFreqInterpCoefs.strides[0], tFreqInterpCoefs.strides[1], tFreqInterpCoefs.strides[2]);
         printf("tShiftSeq        strides[0] %d strides[1] %d\n", tShiftSeq.strides[0], tShiftSeq.strides[1]);
         printf("tUnShiftSeq      strides[0] %d strides[1] %d\n", tUnShiftSeq.strides[0], tUnShiftSeq.strides[1]);

         printf("tDataRx strides[0] %d strides[1] %d strides[2] %d\n", tDataRx.strides[0], tDataRx.strides[1], tDataRx.strides[2]);
         printf("tHEst   strides[0] %d strides[1] %d strides[2] %d\n", tHEst.strides[0], tHEst.strides[1], tHEst.strides[2]);
         // printf("tDbg    strides[0] %d strides[1] %d strides[2] %d\n", tDbg.strides[0], tDbg.strides[1], tDbg.strides[2]);

         printf("dmrsScramId %d slotNum %d activeDmrsGridBmsk 0x%08x\n", dmrsScramId, slotNum, activeDmrsGridBmsk);
     }

     // printf("Block(%d %d %d) Thread(%d %d %d)\n", blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z);
     // if((0 != BS_ANT_IDX) || (0 != PRB_CLUSTER_IDX)) return;
     // printf("dmrsScramId %d slotNum %d activeDmrsGridBmsk 0x%08x", dmrsScramId, slotNum, activeDmrsGridBmsk);
     // if((0 == blockIdx.x) && (0 == blockIdx.y) && (0 == blockIdx.z) && (0 == threadIdx.x) && (0 == threadIdx.y) && (0 == threadIdx.z))
     //    printf("tDataRx strides[0] %d strides[1] %d strides[2] %d\n", tDataRx.strides[0], tDataRx.strides[1], tDataRx.strides[2]);
 #if 0
     printf("InterpCoefs[%d][%d][%d] = %f ShiftSeq[%d][%d] = %f+j%f UnShiftSeq[%d] %f+j%f DataRx[%d][%d][%d]= %f+j%f\n",
            0,0,0,
            tFreqInterpCoefs(0,0,0),
            0,0,
            tShiftSeq(0,0).x,
            tShiftSeq(0,0).y,
            0,
            tUnShiftSeq(0).x,
            tUnShiftSeq(0).y,
            0,0,0,
            tDataRx(0,0,0).x,
            tDataRx(0,0,0).y);
 #endif
 #endif

     const uint32_t THREAD_IDX = threadIdx.x;



     // Determine first PRB in the cluster being processed
     uint32_t prbClusterStartIdx = 0;

     uint32_t prbAbsStartIdx = prbClusterStartIdx + startPrb;
     // Absolute index of DMRS tone within the input OFDM symbol (used as index when loading tone from OFDM
     // symbol)
     const uint32_t DMRS_ABS_TONE_IDX = prbAbsStartIdx * N_TONES_PER_PRB + THREAD_IDX;

     // This index calculation intends to divvy up threads in the thread block for processing as follows:
     // the first group of N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER to process the first DMRS grid, the second group
     // to process the second DMRS grid and so on
     // Relative index of DMRS tone (within a DMRS grid) being processed by this thread
     const uint32_t DMRS_TONE_IDX        = THREAD_IDX % N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER;
     const uint32_t DMRS_INTERP_TONE_IDX = THREAD_IDX % N_TOTAL_DMRS_INTERP_GRID_TONES_PER_CLUSTER;

     // Index of DMRS grid in which the DMRS tone being loaded by this thread resides
     const uint32_t DMRS_GRID_IDX = THREAD_IDX / N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER;

     const uint32_t tempGridIdx = DMRS_GRID_IDX > 1 ? 1 : DMRS_GRID_IDX;

     uint8_t   activeTOCCBmsk     = drvdUeGrpPrms.activeTOCCBmsk[tempGridIdx];
     uint8_t   activeFOCCBmsk     = drvdUeGrpPrms.activeFOCCBmsk[tempGridIdx];
     uint32_t  N_DMRS_SYMS_TOCC   = activeTOCCBmsk == 3 ? 2 : 1;
     uint32_t  N_DMRS_SYMS_FOCC   = activeFOCCBmsk == 3 ? 2 : 1;

     // Index of DMRS grid which thread computes
     const uint32_t DMRS_GRID_IDX_COMPUTE = THREAD_IDX / N_TOTAL_DMRS_INTERP_GRID_TONES_PER_CLUSTER;

     // Index which enables extraction of DMRS tones of a given DMRS grid scattered within the PRB into one
     // contiguous set for processing. Note that the read from GMEM is coalesced and write into SMEM is scattered
     // @todo: check if index calculation can be simplified
     const uint32_t SMEM_DMRS_TONE_IDX = get_smem_dmrs_tone_idx(N_DMRS_GRIDS_PER_PRB, N_INTER_DMRS_GRID_FREQ_SHIFT, THREAD_IDX);
     const uint32_t SMEM_DMRS_GRID_IDX = get_smem_dmrs_grid_idx(N_DMRS_GRIDS_PER_PRB, N_INTER_DMRS_GRID_FREQ_SHIFT, THREAD_IDX);

     // Absolute index of descrambling + shift sequence element
     const uint32_t DMRS_DESCR_SHIFT_SEQ_START_IDX = prbAbsStartIdx * N_DMRS_GRID_TONES_PER_PRB;
     const uint32_t DMRS_DESCR_SHIFT_SEQ_ABS_IDX = DMRS_TONE_IDX;

     // Absolute index of interpolated tone produced by this thread
     const uint32_t INTERP_DMRS_ABS_TONE_IDX =  DMRS_INTERP_TONE_IDX;

     // Select which estimation filter to used based on size of prb cluster
     constexpr uint32_t filtIdx = N_PRBS - 1;


     // Select the shift in interpolation filter coefficients and delay shift based on grid index
     // (for e.g. for 2 DMRS grids and 48 tones per grid, multiply DMRS tone vector with top 48 rows for
     // DMRS_GRID_IDX 0 and bottom 48 rows for DMRS_GRID_IDX 1 to acheieve the effect of shift)
     uint32_t gridShiftIdx = get_inter_dmrs_grid_freq_shift_idx(N_DMRS_GRIDS_PER_PRB, DMRS_GRID_IDX_COMPUTE);

     // Section 5.2.1 in 3GPP TS 38.211
     // The fast-forward of 1600 prescribed by spec is already baked into the gold sequence generator
     constexpr uint32_t DMRS_DESCR_FF = 0; // 1600;

     // First descrambler bit index needed by this thread block
     // Note:The DMRS scrambling sequence is the same for all the DMRS grids. There are 2 sequences one for
     // scid 0 and other for scid 1 but the same sequences is reused for all DMRS grids
     const uint32_t DMRS_DESCR_PRB_CLUSTER_START_BIT =
         DMRS_DESCR_FF + (DMRS_DESCR_SHIFT_SEQ_START_IDX * N_DMRS_DESCR_BITS_PER_TONE);

     // The descrambling sequence generator outputs 32 descrambler bits at a time. Thus, compute the earliest
     // multiple of 32 bits which contains the descrambler bit of the first tone in the PRB cluster as the
     // start index
     const uint32_t DMRS_DESCR_GEN_ALIGNED_START_BIT =
         (DMRS_DESCR_PRB_CLUSTER_START_BIT / N_DMRS_DESCR_BITS_GEN) * N_DMRS_DESCR_BITS_GEN;
     // Offset to descrambler bit of the first tone in the PRB cluster
     const uint32_t DMRS_DESCR_PRB_CLUSTER_START_BIT_OFFSET =
         DMRS_DESCR_PRB_CLUSTER_START_BIT - DMRS_DESCR_GEN_ALIGNED_START_BIT;

     // DMRS descrambling bits generated correspond to subcarriers across frequency
     // e.g. 2 bits for tone0(grid 0) | 2 bits for tone1(grid 1) | 2 bits for tone 2(grid 0) | 2 bits for tone 3(grid 1) | ...
     const uint32_t DMRS_TONE_DESCR_BIT_IDX = DMRS_DESCR_PRB_CLUSTER_START_BIT_OFFSET +
                                              (DMRS_TONE_IDX * N_DMRS_DESCR_BITS_PER_TONE);
     const uint32_t DMRS_DESCR_SEQ_RD_BIT_IDX  = DMRS_TONE_DESCR_BIT_IDX % N_DMRS_DESCR_BITS_GEN;
     const uint32_t DMRS_DESCR_SEQ_RD_WORD_IDX = DMRS_TONE_DESCR_BIT_IDX / N_DMRS_DESCR_BITS_GEN;

     const uint32_t DMRS_DESCR_SEQ_WR_WORD_IDX = THREAD_IDX % N_DMRS_DESCR_WORDS;
     const uint32_t DMRS_DESCR_SEQ_WR_SYM_IDX  = THREAD_IDX / N_DMRS_DESCR_WORDS;

     // determine if thread used to load dmrs:
     const bool loadFlag = (THREAD_IDX < N_PRBS*N_TONES_PER_PRB) ? true : false;

 #if 0
     if((0 == blockIdx.x) && (0 == blockIdx.y) && (0 == blockIdx.z))
        printf("N_DMRS_DESCR_BITS_PER_CLUSTER %d N_DMRS_DESCR_WORDS %d DMRS_DESCR_PRB_CLUSTER_START_BIT %d DMRS_DESCR_GEN_ALIGNED_START_BIT %d, "
               "DMRS_DESCR_SEQ_RD_WORD_IDX %d, DMRS_DESCR_SEQ_RD_BIT_IDX %d, DMRS_DESCR_SEQ_WR_WORD_IDX %d, DMRS_DESCR_SEQ_WR_SYM_IDX %d\n",
               N_DMRS_DESCR_BITS_PER_CLUSTER, N_DMRS_DESCR_WORDS, DMRS_DESCR_PRB_CLUSTER_START_BIT, DMRS_DESCR_GEN_ALIGNED_START_BIT,
               DMRS_DESCR_SEQ_RD_WORD_IDX, DMRS_DESCR_SEQ_RD_BIT_IDX, DMRS_DESCR_SEQ_WR_WORD_IDX, DMRS_DESCR_SEQ_WR_SYM_IDX);
 #endif

     // Data layouts:
     // Global memory read into shared memory
     // N_DMRS_TONES x N_DMRS_SYMS -> N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER x N_DMRS_SYMS x N_DMRS_GRIDS_PER_PRB

     // tOCC removal
     // N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER x N_DMRS_SYMS x N_DMRS_GRIDS_PER_PRB ->
     // N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER x N_DMRS_SYMS_TOCC x N_DMRS_GRIDS_PER_PRB

     // fOCC removal
     // N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER x N_DMRS_SYMS_TOCC x N_DMRS_GRIDS_PER_PRB ->
     // N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER x N_DMRS_SYMS_FOCC x NUM_DMRS_SYMS_TOCC x N_DMRS_GRIDS_PER_PRB

     // Interpolation
     // N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER x N_DMRS_SYMS_FOCC x NUM_DMRS_SYMS_TOCC x N_DMRS_GRIDS_PER_PRB ->
     // N_TOTAL_DMRS_INTERP_GRID_TONES_PER_CLUSTER x N_DMRS_SYMS_FOCC x NUM_DMRS_SYMS_TOCC x N_DMRS_GRIDS_PER_PRB =
     // N_TOTAL_DMRS_INTERP_GRID_TONES_PER_CLUSTER x N_LAYERS x N_DMRS_GRIDS_PER_PRB

     //--------------------------------------------------------------------------------------------------------
     // Allocate shared memory

     constexpr uint32_t N_SMEM_ELEMS1 = N_DMRS_TONES * N_DMRS_SYMS; // (N_DMRS_TONES + N_DMRS_GRIDS_PER_PRB)*N_DMRS_SYMS;
     constexpr uint32_t N_SMEM_ELEMS2 = N_INTERP_TONES * N_DMRS_SYMS_OCC;
     constexpr uint32_t N_SMEM_ELEMS  = (N_SMEM_ELEMS1 > N_SMEM_ELEMS2) ? N_SMEM_ELEMS1 : N_SMEM_ELEMS2;
     // constexpr uint32_t N_SMEM_ELEMS  = (N_SMEM_ELEMS1 + N_SMEM_ELEMS2);
     // constexpr uint32_t N_SMEM_ELEMS  = max(N_SMEM_ELEMS1, N_SMEM_ELEMS2);

     __shared__ TComplexCompute smemBlk[N_SMEM_ELEMS];
     // overlay1
     block_3D<TComplexCompute*, N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER, N_DMRS_SYMS, N_DMRS_GRIDS_PER_PRB> shPilots(&smemBlk[0]);
     // overlay2
     block_3D<TComplexCompute*, N_TOTAL_DMRS_INTERP_GRID_TONES_PER_CLUSTER, N_DMRS_SYMS_OCC, N_DMRS_GRIDS_PER_PRB> shH(&smemBlk[0]);
     // block_3D<TComplexCompute*, N_TOTAL_DMRS_INTERP_GRID_TONES_PER_CLUSTER, N_DMRS_SYMS_OCC, N_DMRS_GRIDS_PER_PRB> shH(&smemBlk[shPilots.num_elem()]);
     static_assert((shPilots.num_elem() <= N_SMEM_ELEMS) && (shH.num_elem() <= N_SMEM_ELEMS), "Insufficient shared memory");

     __shared__ uint32_t descrWords[N_DMRS_SYMS][N_DMRS_DESCR_WORDS];

     //--------------------------------------------------------------------------------------------------------
     // Read DMRS tones into shared memory (separate the tones into different DMRS grids during the write)

     // Cache shift sequence in register
     TComplexCompute shiftSeq = type_convert<TComplexCompute>(tShiftSeq(DMRS_DESCR_SHIFT_SEQ_ABS_IDX, 0));

     if(loadFlag)
     {

     #pragma unroll
         for(uint32_t i = 0; i < N_DMRS_SYMS; ++i)
         {
             shPilots(SMEM_DMRS_TONE_IDX, i, SMEM_DMRS_GRID_IDX) =
                 type_convert<TComplexCompute>(tDataRx(DMRS_ABS_TONE_IDX, pDmrsSymPos[i], BS_ANT_IDX));

     #ifdef ENABLE_DEBUG
             printf("Pilots[%d][%d][%d] -> shPilots[%d][%d][%d] = %f+j%f, ShiftSeq[%d][%d] = %f+j%f\n",
                 DMRS_ABS_TONE_IDX,
                 pDmrsSymPos[i],
                 BS_ANT_IDX,
                 SMEM_DMRS_TONE_IDX,
                 i,
                 SMEM_DMRS_GRID_IDX,
                 shPilots(SMEM_DMRS_TONE_IDX, i, SMEM_DMRS_GRID_IDX).x,
                 shPilots(SMEM_DMRS_TONE_IDX, i, SMEM_DMRS_GRID_IDX).y,
                 DMRS_DESCR_SHIFT_SEQ_ABS_IDX,
                 0,
                 shiftSeq.x,
                 shiftSeq.y);
     #endif
         }

         // Compute the descsrambler sequence
         const uint32_t TWO_POW_17 = bit(17);

         if(DMRS_DESCR_SEQ_WR_SYM_IDX < N_DMRS_SYMS)
         {
             uint32_t symIdx = pDmrsSymPos[DMRS_DESCR_SEQ_WR_SYM_IDX];

             // see 38.211 section 6.4.1.1.1.1
             uint32_t cInit = TWO_POW_17 * (slotNum * OFDM_SYMBOLS_PER_SLOT + symIdx + 1) * (2 * dmrsScramId + 1) + (2 * dmrsScramId) + scid;
             cInit &= ~bit(31);

             // descrWords[DMRS_DESCR_SEQ_WR_SYM_IDX][DMRS_DESCR_SEQ_WR_WORD_IDX] =
             //  __brev(gold32(cInit, (DMRS_DESCR_GEN_ALIGNED_START_BIT + DMRS_DESCR_SEQ_WR_WORD_IDX*N_DMRS_DESCR_BITS_GEN)));

             descrWords[DMRS_DESCR_SEQ_WR_SYM_IDX][DMRS_DESCR_SEQ_WR_WORD_IDX] =
                 gold32(cInit, (DMRS_DESCR_GEN_ALIGNED_START_BIT + DMRS_DESCR_SEQ_WR_WORD_IDX * N_DMRS_DESCR_BITS_GEN));
     #if 0
             printf("symIdx %d, DMRS_DESCR_SEQ_WR_WORD_IDX %d, cInit 0x%08x, DMRS_DESCR_GEN_ALIGNED_START_BIT %d, descrWords[%d][%d] 0x%08x\n",
                 symIdx, DMRS_DESCR_SEQ_WR_WORD_IDX, cInit,
                 (DMRS_DESCR_GEN_ALIGNED_START_BIT + DMRS_DESCR_SEQ_WR_WORD_IDX*N_DMRS_DESCR_BITS_GEN),
                 DMRS_DESCR_SEQ_WR_SYM_IDX, DMRS_DESCR_SEQ_WR_WORD_IDX, descrWords[DMRS_DESCR_SEQ_WR_SYM_IDX][DMRS_DESCR_SEQ_WR_WORD_IDX]);
     #endif
         }
     }

     // To ensure coalesced reads, input DMRS tones are read preserving input order but swizzled while writing
     // to shared memory. Thus each thread may not process the same tone which it wrote to shared memory
     thread_block const& thisThrdBlk = this_thread_block();
     __syncthreads();

     //--------------------------------------------------------------------------------------------------------
     // Apply de-scrambling + delay domain centering sequence for tone index processed by this thread across all
     // DMRS symbols
     const TCompute RECIPROCAL_SQRT2 = 0.7071068f;
     const TCompute SQRT2            = 1.41421356f;
     TComplexCompute    avg[N_TOCC]{};


     if(loadFlag)
     {
     #pragma unroll
         for(uint32_t i = 0; i < N_DMRS_SYMS; ++i)
         {
             int8_t descrIBit = (descrWords[i][DMRS_DESCR_SEQ_RD_WORD_IDX] >> DMRS_DESCR_SEQ_RD_BIT_IDX) & 0x1;
             int8_t descrQBit = (descrWords[i][DMRS_DESCR_SEQ_RD_WORD_IDX] >> (DMRS_DESCR_SEQ_RD_BIT_IDX + 1)) & 0x1;

             TComplexCompute descrCode =
                 cuConj(cuGet<TComplexCompute>((1 - 2 * descrIBit) * RECIPROCAL_SQRT2, (1 - 2 * descrQBit) * RECIPROCAL_SQRT2));
             TComplexCompute descrShiftSeq = shiftSeq * descrCode;

#ifdef ENABLE_DEBUG
             TComplexCompute descrShiftPilot = shPilots(DMRS_TONE_IDX, i, DMRS_GRID_IDX) * descrShiftSeq;
             printf("descrShiftAbsIdx: %d, shPilots[%d][%d][%d] (%f+j%f) * DescrShiftSeq[%d][%d] (%f+j%f) = %f+j%f, ShiftSeq = %f+j%f, DescrCode = %f+j%f, descrIQ (%d,%d) descrWordIdx %d descrBitIdx %d\n",
                 DMRS_DESCR_SHIFT_SEQ_ABS_IDX,
                 DMRS_TONE_IDX,
                 i,
                 DMRS_GRID_IDX,
                 shPilots(DMRS_TONE_IDX, i, DMRS_GRID_IDX).x,
                 shPilots(DMRS_TONE_IDX, i, DMRS_GRID_IDX).y,
                 DMRS_TONE_IDX,
                 i,
                 descrShiftSeq.x,
                 descrShiftSeq.y,
                 descrShiftPilot.x,
                 descrShiftPilot.y,
                 shiftSeq.x,
                 shiftSeq.y,
                 descrCode.x,
                 descrCode.y,
                 descrIBit,
                 descrQBit,
                 DMRS_DESCR_SEQ_RD_WORD_IDX,
                 DMRS_DESCR_SEQ_RD_BIT_IDX);

             if((0 == DMRS_GRID_IDX) && (((0 == prbAbsStartIdx) && (DMRS_TONE_IDX < (N_EDGE_PRB * N_DMRS_GRID_TONES_PER_PRB))) || ((0 != prbAbsStartIdx) && (prbAbsStartIdx + N_DMRS_PRB_IN_PER_CLUSTER) <= N_DATA_PRB)))
             {
     #if 0
             TComplexCompute descrShiftPilot = shPilots(DMRS_TONE_IDX, i, DMRS_GRID_IDX) * descrShiftSeq;
             printf("descrShiftAbsIdx: %d shPilots[%d][%d][%d] (%f+j%f) * DescrShiftSeq[%d][%d] (%f+j%f) = %f+j%f, ShiftSeq = %f+j%f, DescrCode = %f+j%f, descrIQ (%d,%d) descrWordIdx %d descrBitIdx %d\n",
                     DMRS_DESCR_SHIFT_SEQ_ABS_IDX,
                     DMRS_TONE_IDX,
                     i,
                     DMRS_GRID_IDX,
                     shPilots(DMRS_TONE_IDX, i, DMRS_GRID_IDX).x,
                     shPilots(DMRS_TONE_IDX, i, DMRS_GRID_IDX).y,
                     DMRS_TONE_IDX,
                     i,
                     descrShiftSeq.x,
                     descrShiftSeq.y,
                     descrShiftPilot.x,
                     descrShiftPilot.y,
                     shiftSeq.x,
                     shiftSeq.y,
                     descrCode.x,
                     descrCode.y,
                     descrIBit,
                     descrQBit,
                     DMRS_DESCR_SEQ_RD_WORD_IDX,
                     DMRS_DESCR_SEQ_RD_BIT_IDX);
     #endif

                 // tDbg(DMRS_DESCR_SHIFT_SEQ_ABS_IDX, i, 0, 0) = type_convert<TComplexStorage>(shiftSeq);
                 // tDbg(DMRS_DESCR_SHIFT_SEQ_ABS_IDX, i, 0, 0) = type_convert<TComplexStorage>(descrShiftSeq);
                 tDbg(DMRS_DESCR_SHIFT_SEQ_ABS_IDX, i, 0, 0) = type_convert<TComplexStorage>(descrCode);
             }
     #endif // ENABLE_DEBUG

             shPilots(DMRS_TONE_IDX, i, DMRS_GRID_IDX) *= descrShiftSeq;
         }

         //--------------------------------------------------------------------------------------------------------
         // Time domain cover code removal
         constexpr TCompute AVG_SCALE = cuGet<TCompute>(1u) / cuGet<TCompute>(N_DMRS_SYMS);
         // TComplexCompute    avg[N_DMRS_SYMS_TOCC]{};

         int32_t tOCCIdx = 0;

     #pragma unroll
         for(int32_t i = 0; i < 2; ++i)
         {
            int32_t temp = (activeTOCCBmsk >> i) & 0x1;
            if (!temp) {
                continue;
            }
     #pragma unroll
             for(int32_t j = 0; j < N_DMRS_SYMS; ++j)
             {
                 // For first tOCC (i = 0) output, multiply all DMRS symbols with +1 and average
                 // For second tOCC (i = 1) output, multiply even DMRS symbols with +1, odd DMRS symbols with -1 and average
                 int32_t sign = (-(i & j)) | 1;
                 avg[tOCCIdx] += (shPilots(DMRS_TONE_IDX, j, DMRS_GRID_IDX) * (sign * AVG_SCALE));
     #ifdef ENABLE_DEBUG
                 TComplexCompute prod = (shPilots(DMRS_TONE_IDX, j, DMRS_GRID_IDX) * (sign * AVG_SCALE));
                 printf("sign*AVG_SCALE %f Pilot[%d][%d][%d] = %f+j%f avg[%d] = %f+j%f, prod = %f+j%f\n",
                     sign * AVG_SCALE,
                     DMRS_TONE_IDX,
                     j,
                     DMRS_GRID_IDX,
                     shPilots(DMRS_TONE_IDX, j, DMRS_GRID_IDX).x,
                     shPilots(DMRS_TONE_IDX, j, DMRS_GRID_IDX).y,
                     i,
                     avg[i].x,
                     avg[i].y,
                     prod.x,
                     prod.y);
     #endif
             }
             tOCCIdx++;
         }
     }

     // shPilots and shH are overlaid in shared memory and can have different sizes (based on config). For this reason
     // ensure shPilots access from all threads is completed before writing into shH
     __syncthreads();

     //--------------------------------------------------------------------------------------------------------
     // Apply frequecy domain cover code and store inplace in shared memory
     // Multiply even tones with +1 and odd tones with -1

     if(loadFlag)
     {
         // Note that the loop termination count below is tOCC symbol count
     #pragma unroll
         for(int32_t i = 0; i < N_DMRS_SYMS_TOCC; ++i)
         {
            int32_t fOCCIdx = 0;

     #pragma unroll
             for(int32_t j = 0; j < 2; ++j)
             {
                int32_t temp = (activeFOCCBmsk >> j) & 0x1;
                if (!temp) {
                    continue;
                }
                 // First fOCC output: multiply all tones by +1s
                 // Second fOCC output: multiply even tones by +1s and odd tones by -1s
                 int32_t sign                                                  = (-(DMRS_TONE_IDX & j)) | 1;
                 shH(DMRS_TONE_IDX, (N_DMRS_SYMS_FOCC * i) + fOCCIdx, DMRS_GRID_IDX) = avg[i] * sign;
     #ifdef ENABLE_DEBUG
                 printf("PilotsPostOCC[%d][%d][%d] = %f+j%f\n",
                     DMRS_TONE_IDX,
                     (N_DMRS_SYMS_FOCC * i) + j,
                     DMRS_GRID_IDX,
                     cuReal(shH(DMRS_TONE_IDX, (N_DMRS_SYMS_FOCC * i) + j, DMRS_GRID_IDX)),
                     cuImag(shH(DMRS_TONE_IDX, (N_DMRS_SYMS_FOCC * i) + j, DMRS_GRID_IDX)));
     #endif
                 fOCCIdx++;
             }
         }
     }

     // Ensure all threads complete writing results to shared memory since each thread computing an inner product
     // during interpolation stage will use results from other threads in the thread block
     __syncthreads();

     //--------------------------------------------------------------------------------------------------------
     // Interpolate (matrix-vector multiply)

     // early exit for inactive grids
     const int32_t DMRS_GRID_WR_IDX = DMRS_GRID_WR_IDX_TBL[activeDmrsGridBmsk & ACTIVE_DMRS_GRID_BMSK][DMRS_GRID_IDX_COMPUTE];
     // if(!is_set(bit(DMRS_GRID_IDX), activeDmrsGridBmsk) || (DMRS_GRID_WR_IDX < 0)) return;
     if(DMRS_GRID_WR_IDX < 0) return;

     activeTOCCBmsk     = drvdUeGrpPrms.activeTOCCBmsk[DMRS_GRID_IDX_COMPUTE];
     activeFOCCBmsk     = drvdUeGrpPrms.activeFOCCBmsk[DMRS_GRID_IDX_COMPUTE];
     N_DMRS_SYMS_TOCC   = activeTOCCBmsk == 3 ? 2 : 1;
     N_DMRS_SYMS_FOCC   = activeFOCCBmsk == 3 ? 2 : 1;

     for(uint32_t i = 0; i < (N_DMRS_SYMS_FOCC*N_DMRS_SYMS_TOCC); ++i)
     {
         TComplexCompute innerProd{};

         // H = W x Y: (N_INTERP_TONES x N_DMRS_TONES) x (N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER x N_DMRS_SYMS_OCC)
         // Each thread selects one row of W and computes N_DMRS_TONES length inner product to produce one interpolated
         // tone of H
         for(uint32_t j = 0; j < N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER; ++j)
         {
             TCompute interpCoef = type_convert<TCompute>(tFreqInterpCoefs(DMRS_INTERP_TONE_IDX + gridShiftIdx, j, filtIdx));
             innerProd += (shH(j, i, DMRS_GRID_IDX_COMPUTE) * interpCoef);
         }
         // Wait for all threads to complete their inner products before updating the shared memory inplace
         // The sync is needed because shPilots and shH are overlaid
         // Note that tOCCIdx can vary per thread, so this sync and the next need to be thisThrdBlk.sync()
         // calls rather than __syncthreads().
         thisThrdBlk.sync();

         shH(DMRS_INTERP_TONE_IDX, i, DMRS_GRID_IDX_COMPUTE) = innerProd;

 #ifdef ENABLE_DEBUG
         printf("InterpPilots[%d][%d][%d] = %f+j%f\n", DMRS_INTERP_TONE_IDX, i, DMRS_GRID_IDX, innerProd.x, innerProd.y);
 #endif
     }

     // Wait for shared memory writes to complete from all threads
     thisThrdBlk.sync();

     //--------------------------------------------------------------------------------------------------------
     // Unshift the channel in delay back to its original location and write to GMEM. This is a scattered write
     // (@todo: any opportunities to make it coalesced?)
     // Output format is N_BS_ANT x (N_DMRS_SYMS_FOCC x N_DMRS_SYMS_TOCC x N_DMRS_GRIDS_PER_PRB) x N_DMRS_INTERP_PRB_OUT_PER_CLUSTER
     // where N_DMRS_SYMS_FOCC x N_DMRS_SYMS_TOCC x N_DMRS_GRIDS_PER_PRB = N_LAYERS
     //const uint32_t DMRS_GRID_OFFSET_H = DMRS_GRID_WR_IDX * N_DMRS_SYMS_OCC;

 // #ifdef CH_EST_COALESCED_WRITE
 //     // H (N_DATA_PRB*N_TONES_PER_PRB, N_LAYERS, N_BS_ANTS, NH)
 //     // read the number of rx antennas
 //     uint32_t N_BS_ANTS = drvdUeGrpPrms.nRxAnt;
 //     TComplexStorage* pHEst = tHEst.addr + ((NH_IDX * N_BS_ANTS + BS_ANT_IDX) * N_LAYERS * N_DATA_PRB * N_TONES_PER_PRB);
 // #endif


 // index of interpolated tone within cluster
 uint32_t CLUSTER_INTERP_TONE_IDX = DMRS_INTERP_TONE_IDX;

 #pragma unroll
     for(uint32_t i = 0; i < N_LAYERS; ++i)
     {
        if (i < nLayers) {
            uint32_t j = OCCIdx[i] & 0x3;
            uint32_t k = (OCCIdx[i] >> 2) & 0x1;
            if (DMRS_GRID_IDX_COMPUTE == k) {
                shH(DMRS_INTERP_TONE_IDX, j, DMRS_GRID_IDX_COMPUTE) *=
                    type_convert<TComplexCompute>(tUnShiftSeq(CLUSTER_INTERP_TONE_IDX + gridShiftIdx)); //INTERP_DMRS_ABS_TONE_IDX
                if(nDmrsCdmGrpsNoData==1)
                {
                    shH(DMRS_INTERP_TONE_IDX, j, DMRS_GRID_IDX) *= cuGet<TComplexCompute>(SQRT2, 0.0f);
                }

 #ifndef CH_EST_COALESCED_WRITE
     tHEst(BS_ANT_IDX, i, INTERP_DMRS_ABS_TONE_IDX, NH_IDX) =
             type_convert<TComplexStorage>(shH(DMRS_INTERP_TONE_IDX, j, DMRS_GRID_IDX_COMPUTE));
 #else //fix me for flexbile DMRS port mapping
         pHEst[(DMRS_GRID_OFFSET_H + i) * N_DATA_PRB * N_TONES_PER_PRB + INTERP_DMRS_ABS_TONE_IDX] =
             type_convert<TComplexStorage>(shH(DMRS_INTERP_TONE_IDX, i, DMRS_GRID_IDX_COMPUTE));
 #endif
            }
 #ifdef ENABLE_DEBUG
 #if 0
      printf("shH[%d][%d][%d] = %f+j%f\n", DMRS_INTERP_TONE_IDX, i, DMRS_GRID_IDX,
          shH(DMRS_INTERP_TONE_IDX, i, DMRS_GRID_IDX).x, shH(DMRS_INTERP_TONE_IDX, i, DMRS_GRID_IDX).y);
 #endif
 #if 0
       printf("tH[%d][%d][%d][%d] = %f+j%f\n", BS_ANT_IDX, DMRS_GRID_OFFSET_H + i, INTERP_DMRS_ABS_TONE_IDX, NH_IDX,
          tHEst(BS_ANT_IDX, DMRS_GRID_OFFSET_H + i, INTERP_DMRS_ABS_TONE_IDX, NH_IDX).x,
          tHEst(BS_ANT_IDX, DMRS_GRID_OFFSET_H + i, INTERP_DMRS_ABS_TONE_IDX, NH_IDX).y);
 #endif
 #if 0
        printf("tUnshift[%d] = %f+j%f\n", INTERP_DMRS_ABS_TONE_IDX + gridShiftIdx,tUnShiftSeq(INTERP_DMRS_ABS_TONE_IDX + gridShiftIdx).x, tUnShiftSeq(INTERP_DMRS_ABS_TONE_IDX + gridShiftIdx).y);

        printf("tH[%d][%d][%d][%d] = %f+j%f\n", BS_ANT_IDX, DMRS_GRID_OFFSET_H + i, INTERP_DMRS_ABS_TONE_IDX, NH_IDX,
          tHEst(BS_ANT_IDX, DMRS_GRID_OFFSET_H + i, INTERP_DMRS_ABS_TONE_IDX, NH_IDX).x,
          tHEst(BS_ANT_IDX, DMRS_GRID_OFFSET_H + i, INTERP_DMRS_ABS_TONE_IDX, NH_IDX).y);
 #endif
 #endif
        }
     }
 } //smallChEstNoDftSOfdmKernel

template<uint16_t N, uint16_t M>
static inline __device__ void realMtrxVecMult(__half* vecOut, tensor_ref<__half>& mtrx, __half* vecIn)
{
    for(int i = 0; i < N; ++i)
    {
        vecOut[i] = mtrx(i,0) * vecIn[0];

        for(int j = 1; j < M; ++j)
        {
            vecOut[i] += mtrx(i,j) * vecIn[j];
        }
    }
}

template<uint16_t N>
static inline __device__ void cplxMtrxVecMult(__half2* vecOut, __half2 (&mtrx)[N][N], __half2* vecIn)
{
    for(int i = 0; i < N; ++i)
    {
        vecOut[i] = complex_mul(mtrx[0][i], vecIn[0]);

        for(int j = 1; j < N; ++j)
        {
            vecOut[i] += complex_mul(mtrx[j][i], vecIn[j]);
        }
    }
}


template<uint8_t fourierTranSize, uint8_t log2FourierTranSize>
static inline __device__ void warpFourierTransform(const uint32_t THREAD_IDX, __half2& sigValue, uint8_t* pFourierPermuteIdxs)
{
    __half2* pTwiddleFactors = d_twiddle32;

    // tile:
    thread_block_tile<fourierTranSize> tile = cg::tiled_partition<fourierTranSize>(cg::this_thread_block());

    // byte-reverse permute the inputs:
    uint8_t fourierBlockIdx       = THREAD_IDX % fourierTranSize;
    uint8_t idxWithinFourierBlock = 0;
    uint8_t fourierBlockSize      = 1;

    // signal index:
    uint8_t sigIdx  = THREAD_IDX % fourierTranSize;

    // Byte-reverse permute the signal:
    uint8_t inputIdx = pFourierPermuteIdxs[sigIdx];
    sigValue.x       = tile.shfl(sigValue.x, inputIdx);
    sigValue.y       = tile.shfl(sigValue.y, inputIdx);

    for(uint8_t fourierStage = 1;  fourierStage <= log2FourierTranSize; fourierStage++)
    {
        // Determine if Fourier block even or odd:
        uint8_t oddFlag = fourierBlockIdx % 2;

        // If odd Fourier block, multiply by the twiddle factor.
        // Determine butterfly input idx.
        // Determine butterfly sign:
        uint8_t butterflyInputIdx = sigIdx + fourierBlockSize;
        __half2 butterflySgn       = {static_cast<__half>(1), static_cast<__half>(1)};
        __half2 sigValueBefore     = sigValue;
        if(oddFlag)
        {
            __half2 twiddleValue = pTwiddleFactors[idxWithinFourierBlock];
            sigValue           = complex_mul(sigValue, pTwiddleFactors[idxWithinFourierBlock]);
            butterflyInputIdx -= 2*fourierBlockSize;
            butterflySgn       = {static_cast<__half>(-1), static_cast<__half>(-1)};
        }

        // load butterfly input:
        __half2 butterflyInput;
        butterflyInput.x = tile.shfl(sigValue.x, butterflyInputIdx);
        butterflyInput.y = tile.shfl(sigValue.y, butterflyInputIdx);

        // butterfly operation:
        sigValue = __hfma2(butterflySgn, sigValue, butterflyInput);

        // update twiddle buffer:
        pTwiddleFactors += fourierBlockSize;

        // update Fourier block indidices:
        idxWithinFourierBlock += fourierBlockSize * (fourierBlockIdx % 2);
        fourierBlockIdx        = fourierBlockIdx / 2;

        // update Fourier block size:
        fourierBlockSize *= 2;
    }
}



template<uint8_t log2SecondStageFourierSize>
static inline __device__ void twoStageFourierTransform(const uint32_t THREAD_IDX, __half2& sigValue, tensor_ref<__half2>& tSecondStageTwiddleFactors, uint8_t* pSecondStageFourierPerm, __half2* sh_fourierWorkspace)
{
    // Cooley-Tukey Fourier transform with two stages. First stage consists of FFTs of size 32,
    // second stage consists of FFTs of size nZpDmrs / 32 (denoted secondStageFourierSize)
    constexpr uint8_t firstStageFourierSize     = 32;
    constexpr uint8_t log2FirstStageFourierSize = 5;

    if(log2SecondStageFourierSize == 0) // if second stage size is only 1, no need to perform it.
    {
        warpFourierTransform<firstStageFourierSize, log2FirstStageFourierSize>(THREAD_IDX, sigValue, d_fourier32PermuteIdx);
        sh_fourierWorkspace[THREAD_IDX] = sigValue;
        __syncthreads();
    }else
    {
        constexpr uint8_t secondStageFourierSize = (static_cast<uint8_t>(1) << log2SecondStageFourierSize);

        // First determine thread's location within the 32-blocks:
        uint8_t idxWithinFirstStageFourierBlock = THREAD_IDX % firstStageFourierSize;
        uint8_t firstStageBlockIdx              = THREAD_IDX / firstStageFourierSize;

        // permute the signal secondStageFourierSize x firstStageFourierSize --> firstStageFourierSize x secondStageFourierSize
        sh_fourierWorkspace[THREAD_IDX] = sigValue;
        __syncthreads();
        sigValue = sh_fourierWorkspace[firstStageBlockIdx + idxWithinFirstStageFourierBlock * secondStageFourierSize];

        // Perform first stage 32-FFTs:
        warpFourierTransform<firstStageFourierSize, log2FirstStageFourierSize>(THREAD_IDX, sigValue, d_fourier32PermuteIdx);

        // Multiply by twiddle factors:
        sigValue = complex_mul(sigValue, tSecondStageTwiddleFactors(idxWithinFirstStageFourierBlock, firstStageBlockIdx));

        // permute the signal firstStageFourierSize x secondStageFourierSize --> secondStageFourierSize x firstStageFourierSize
        sh_fourierWorkspace[firstStageBlockIdx + idxWithinFirstStageFourierBlock * secondStageFourierSize] = sigValue;
        __syncthreads();
        sigValue = sh_fourierWorkspace[THREAD_IDX];

        // Perform second stage FFTs:
        warpFourierTransform<secondStageFourierSize, log2SecondStageFourierSize>(THREAD_IDX, sigValue, pSecondStageFourierPerm);

        // permute the signal secondStageFourierSize x firstStageFourierSize --> firstStageFourierSize x secondStageFourierSize
        uint8_t secondStageBlockIdx              = THREAD_IDX / secondStageFourierSize;
        uint8_t idxWithinSecondStageFourierBlock = THREAD_IDX % secondStageFourierSize;

        sh_fourierWorkspace[secondStageBlockIdx + idxWithinSecondStageFourierBlock * firstStageFourierSize] = sigValue;
        __syncthreads();
    }
}






 // template<class FFT, uint16_t nZpDmrsSc>
 template <uint16_t nZpDmrsSc, uint16_t log2numZpDmrsSc>
 __launch_bounds__(nZpDmrsSc, 1)
__global__ void puschRkhsChEstKernel(puschRxChEstStatDescr_t* pStatDescr, puschRxChEstDynDescr_t* pDynDescr)
{
    // utility constants:
    const __half RECIPROCAL_SQRT2 = 0.7071068f;

    // Number of CP intervals:
    constexpr uint16_t nCpInt    = static_cast<uint16_t>(static_cast<float>(nZpDmrsSc) * 0.1386);
    constexpr uint16_t nCorrElem = nCpInt * RKHS_N_EIGS * RKHS_N_EIGS;

    // DMRS descrambling constants
    constexpr uint32_t N_DMRS_DESCR_BITS_PER_TONE = 2; // 1bit for I and 1 bit for Q
    constexpr uint32_t N_DMRS_DESCR_BITS          = N_DMRS_DESCR_BITS_PER_TONE * nZpDmrsSc; // # of DMRS descrambler bits generated at one time
    constexpr uint32_t N_DMRS_DESCR_BITS_GEN      = 32;
    // Round up to the next multiple of N_DMRS_DESCR_BITS_GEN plus 1 (+1 because DMRS_DESCR_PRB_CLUSTER_START_BIT_OFFSET
    // may be large enough to spill the descrambler bits to the next word)
    constexpr uint32_t N_DMRS_DESCR_WORDS = ((N_DMRS_DESCR_BITS + N_DMRS_DESCR_BITS_GEN - 1) / N_DMRS_DESCR_BITS_GEN) + 1;

    // thread/block paramaters:
    const uint32_t     BLK_IDX     = blockIdx.x;
    constexpr uint32_t NUM_THREADS = nZpDmrsSc;
    const uint32_t     THREAD_IDX  = threadIdx.x;
    const uint32_t     WARP_IDX    = THREAD_IDX / 32;
    const uint32_t     LANE_IDX    = THREAD_IDX % 32;

    // warp paramaters:
    cg::thread_block thisThrdBlk   = cg::this_thread_block();
    constexpr int WARP_SIZE        = 32;
    cg::thread_block_tile<32> tile = cg::tiled_partition<WARP_SIZE>(thisThrdBlk);

    // shared memory:
    constexpr uint32_t        nBytesFourierWorkspace = nZpDmrsSc * 4;
    __shared__ __half2        sh_fourierWorkspace[nBytesFourierWorkspace];
    __shared__ uint32_t       descrWords[RKHS_MAX_N_SYM][N_DMRS_DESCR_WORDS];
    __shared__ float          sh_noiseEnergy;
    __shared__ __half2        sh_corr[nCpInt][RKHS_N_EIGS][RKHS_N_EIGS];
    __shared__ uint16_t       sh_eqIntIdxs[nCpInt];
    __shared__ __half         sh_warpMaxCoeffEnergy[32];
    __shared__ uint16_t       sh_warpMaxCoeffIdx[32];

    if(THREAD_IDX < 32)
    {
        sh_warpMaxCoeffEnergy[THREAD_IDX] = static_cast<__half>(0);
        sh_warpMaxCoeffIdx[THREAD_IDX]    = 0;
        // Only initialize sh_eqIntIdxs if THREAD_IDX is within its bounds (nCpInt)
        // and THREAD_IDX is also less than 32 (as per the outer condition)
        if (THREAD_IDX < nCpInt) {
            sh_eqIntIdxs[THREAD_IDX]      = 0;
        }

    }
    __syncthreads();

    // if((0 == blockIdx.x) && (0 == blockIdx.y) && (0 == blockIdx.z) && (0 == threadIdx.x) && (0 == threadIdx.y) && (0 == threadIdx.z))
    // {
    //     //  printf("\n tHEst(%d, %d, %d, %d).x = %f + j%f", antIdx, layerIdx, THREAD_IDX, chEstTimeInst, tHEst(antIdx, layerIdx, THREAD_IDX, chEstTimeInst).x, tHEst(antIdx, layerIdx, THREAD_IDX, chEstTimeInst).y);
    //     // printf("\n\n mpIdx = %d, nEqCoeffs = %d, N0 = %f \n\n", mpIdx, nEqCoeffs);
    //     printf("\n\n HERE! START \n\n");
    //     for(int i = 0; i < 32; ++i)
    //     {
    //         printf("\n sh_eqIntIdxs[%d] = %d", i, sh_eqIntIdxs[i]);
    //     }
    //     printf("\n\n");
    // }

    const int stide1 = 1;
    const int stide2 = RKHS_N_EIGS;
    const int stide3 = RKHS_N_EIGS * RKHS_MAX_LAYERS;
    const int stide4 = RKHS_N_EIGS * RKHS_MAX_LAYERS * RKHS_MAX_N_GNB_ANTS;

    const int eqCoeffStrides[4] = {stide1, stide2, stide3, stide4};
    tensor_ref<__half2> sh_tEqCoeff(sh_fourierWorkspace, eqCoeffStrides);
    sh_noiseEnergy = 0;

    // Ue group and compute block paramaters:
    rkhsComputeBlockPrms_t&    computeBlockPrm          = pDynDescr->rkhsCompBlockPrms[BLK_IDX];
    uint16_t                   ueGrpIdx                 = computeBlockPrm.ueGrpIdx;
    rkhsUeGrpPrms_t&           rkhsUeGrpPrm             = pDynDescr->rkhsUeGrpPrms[ueGrpIdx];
    cuphyPuschRxUeGrpPrms_t&   drvdUeGrpPrm             = pDynDescr->pDrvdUeGrpPrms[ueGrpIdx];
    computeBlocksCommonPrms_t& computeBlocksCommonPrms  = rkhsUeGrpPrm.computeBlocksCommonPrms;

    // noise estimation paramaters:
    rkhsNoiseEstMethod_t noiseEstMethod         = computeBlocksCommonPrms.noiseEstMethod;
    float                nNoiseMeasurments      = computeBlocksCommonPrms.nNoiseMeasurments;
    uint16_t             noiseRegionFirstIntIdx = computeBlocksCommonPrms.noiseRegionFirstIntIdx;
    uint16_t             nNoiseIntsPerFocc      = computeBlocksCommonPrms.nNoiseIntsPerFocc;
    uint16_t             nNoiseIntsPerGrid      = computeBlocksCommonPrms.nNoiseIntsPerGrid;

    // I/O buffers:
    uint16_t nRxAnt = drvdUeGrpPrm.nRxAnt;
    tensor_ref<__half2> tDataRx(drvdUeGrpPrm.tInfoDataRx.pAddr, drvdUeGrpPrm.tInfoDataRx.strides);// (NF, ND, N_BS_ANTS)
    tensor_ref<float2>  tHEst(drvdUeGrpPrm.tInfoHEst.pAddr, drvdUeGrpPrm.tInfoHEst.strides);

    // Output offsets:
    const uint16_t startOutputScInBlock  = computeBlockPrm.startOutputScInBlock;
    const uint16_t scOffsetIntoChEstBuff = computeBlockPrm.scOffsetIntoChEstBuff;
    const uint16_t nOutputSc             = computeBlockPrm.nOutputSc;

    // DMRS paramaters:
    uint16_t dmrsScramId        = drvdUeGrpPrm.dmrsScrmId;
    uint8_t  scid               = drvdUeGrpPrm.scid;
    uint16_t slotNum            = drvdUeGrpPrm.slotNum;
   // uint8_t  nDmrsCdmGrpsNoData = drvdUeGrpPrm.nDmrsCdmGrpsNoData;

    // if(THREAD_IDX == 0)
    // {
    //     printf("\n\n nDmrsCdmGrpsNoData = %d \n\n", nDmrsCdmGrpsNoData);
    // }

    // time allocation paramatewrs:
    const uint8_t        dmrsMaxLen    = drvdUeGrpPrm.dmrsMaxLen;
    const uint8_t chEstTimeInst  = pDynDescr->chEstTimeInst;
    const uint8_t* const pDmrsSymPos   = &drvdUeGrpPrm.dmrsSymLoc[chEstTimeInst*dmrsMaxLen];

    // frequency allocation paramaters:
    uint16_t nPrb     = computeBlocksCommonPrms.nPrb;
    uint16_t nDmrsSc  = 6 * nPrb;
    uint16_t startPrb = computeBlockPrm.startInputPrb;
    uint16_t startSc  = 12 * startPrb;

    // UE paramaters:
    uint8_t nUes                                               = drvdUeGrpPrm.nUes;
    uint8_t (&nUeLayers)[CUPHY_PUSCH_RX_MAX_N_UE_PER_UE_GROUP] = drvdUeGrpPrm.nUeLayers;

    // RKHS prb tables:
    prbRkhsDesc_t& prbRkhsDesc         = pStatDescr->prbRkhsDescs[nPrb - 1];
    uint8_t       (&gridIdxs)[8]       = rkhsUeGrpPrm.gridIdxs;
    uint8_t        zpIdx               = prbRkhsDesc.zpIdx;
    __half         sumEigValues        = prbRkhsDesc.sumEigValues;

    tensor_ref<__half>  tEigVecCob(prbRkhsDesc.tInfoEigVecCob.pAddr, prbRkhsDesc.tInfoEigVecCob.strides); // (NF, ND, N_BS_ANTS)
    tensor_ref<__half2> tCorr(prbRkhsDesc.tInfoCorr.pAddr, prbRkhsDesc.tInfoCorr.strides);
    tensor_ref<__half>  tEigVal(prbRkhsDesc.tInfoEigVal.pAddr, prbRkhsDesc.tInfoEigVal.strides);
    tensor_ref<__half>  tInterpCob(prbRkhsDesc.tInfoInterpCob.pAddr, prbRkhsDesc.tInfoInterpCob.strides);

    // RKHS zp tables:
    zpRkhsDesc_t& zpRkhsDesc = pStatDescr->zpRkhsDescs[zpIdx];
    tensor_ref<__half>  tZpDmrsScEigenVec(zpRkhsDesc.tInfoZpDmrsScEigenVec.pAddr, zpRkhsDesc.tInfoZpDmrsScEigenVec.strides);
    tensor_ref<__half>  tZpInterpVec(zpRkhsDesc.tInfoZpInterpVec.pAddr, zpRkhsDesc.tInfoZpInterpVec.strides);
    tensor_ref<__half2> tSecondStageTwiddleFactors(zpRkhsDesc.tInfoSecondStageTwiddleFactors.pAddr, zpRkhsDesc.tInfoSecondStageTwiddleFactors.strides);
    uint8_t*            pSecondStageFourierPerm = static_cast<uint8_t*>(zpRkhsDesc.tInfoSecondStageFourierPerm.pAddr);
    constexpr uint8_t   log2SecondStageFourierSize = log2numZpDmrsSc - 5;

    // compute descrambling words, see 38.211 section 6.4.1.1.1.1:
    const uint32_t DMRS_DESCR_SEQ_WR_WORD_IDX = THREAD_IDX % N_DMRS_DESCR_WORDS;
    const uint32_t DMRS_DESCR_SEQ_WR_SYM_IDX  = THREAD_IDX / N_DMRS_DESCR_WORDS;
    const uint32_t TWO_POW_17 = bit(17);

    // compute starting scrambling bit:
    const uint32_t DMRS_DESCR_PRB_CLUSTER_START_BIT = startPrb * RKHS_N_DMRS_GRID_TONES_PER_PRB * N_DMRS_DESCR_BITS_PER_TONE;
    const uint32_t DMRS_DESCR_GEN_ALIGNED_START_BIT = (DMRS_DESCR_PRB_CLUSTER_START_BIT / N_DMRS_DESCR_BITS_GEN) * N_DMRS_DESCR_BITS_GEN;
    const uint32_t DMRS_DESCR_START_BIT_OFFSET      = DMRS_DESCR_PRB_CLUSTER_START_BIT - DMRS_DESCR_GEN_ALIGNED_START_BIT;

    // projection coefficents:
    uint16_t nWarpsActiveInMp = (nCpInt * nRxAnt + 31) / 32;
    uint8_t  threadActiveInMp = (THREAD_IDX < (nCpInt * nRxAnt)) ? 1 : 0;
    uint8_t  projCoeffAntIdx  = THREAD_IDX % nRxAnt;
    uint16_t projCoeffIntIdx  = THREAD_IDX / nRxAnt;
    __half2 projCoeffs[RKHS_MAX_LAYERS][RKHS_N_EIGS];

    // load correlation coefficents into shared memory:
    for(int corrElemIdx = THREAD_IDX; corrElemIdx < nCorrElem; corrElemIdx += NUM_THREADS)
    {
        uint16_t corrIntIdx = corrElemIdx / (RKHS_N_EIGS * RKHS_N_EIGS);
        uint16_t corrMtxIdx = corrElemIdx % (RKHS_N_EIGS * RKHS_N_EIGS);
        uint16_t corrIdxOut = corrMtxIdx / RKHS_N_EIGS;
        uint16_t corrIdxIn  = corrMtxIdx % RKHS_N_EIGS;

        sh_corr[corrIntIdx][corrIdxIn][corrIdxOut] = tCorr(corrIdxIn, corrIdxOut, corrIntIdx);
    }

    // compute descrambling words:
    if(DMRS_DESCR_SEQ_WR_SYM_IDX < dmrsMaxLen && dmrsMaxLen < N_DMRS_DESCR_WORDS)
    {
        // compute cInit:
        uint32_t symIdx  =  pDmrsSymPos[DMRS_DESCR_SEQ_WR_SYM_IDX];
        uint32_t cInit   =  TWO_POW_17 * (slotNum * OFDM_SYMBOLS_PER_SLOT + symIdx + 1) * (2 * dmrsScramId + 1) + (2 * dmrsScramId) + scid;
        cInit           &=  ~bit(31);

        // compute descrambling word:
        descrWords[DMRS_DESCR_SEQ_WR_SYM_IDX][DMRS_DESCR_SEQ_WR_WORD_IDX] =
            gold32(cInit, (DMRS_DESCR_GEN_ALIGNED_START_BIT + DMRS_DESCR_SEQ_WR_WORD_IDX * N_DMRS_DESCR_BITS_GEN));
    }
    __syncthreads();


    // thread noise energy:
    __half  noiseEnergyMeasuredByThread_half = static_cast<__half>(0);
    __half  N0;

    // thread eigenvectors and descramCodes:
    __half   eigVecValuesForThread[RKHS_N_EIGS];
    __half2  descramCodeForThread[RKHS_MAX_N_SYM];

    // Hamming window:
    __half hammingWindow;

    // compute eigenvectors, hamming window, and descramCodes for this thread:
    if(THREAD_IDX < nDmrsSc)
    {
        // eigenvectors:
        realMtrxVecMult<RKHS_N_EIGS, RKHS_N_ZP_EIGS>(eigVecValuesForThread, tEigVecCob, tZpDmrsScEigenVec.trace(THREAD_IDX));

        // Hamming window:
        #ifdef RKHS_USE_HAMMING
            float hammingPhase = 2 * -3.14159265358979323846 * static_cast<float>(THREAD_IDX) / static_cast<float>(nDmrsSc - 1);
            hammingWindow      = static_cast<__half>(0.54 - 0.46 * __cosf(hammingPhase));
        #endif

        // location of frequency scrambling bits:
        const uint32_t DMRS_TONE_DESCR_BIT_IDX    = DMRS_DESCR_START_BIT_OFFSET + (THREAD_IDX * N_DMRS_DESCR_BITS_PER_TONE);
        const uint32_t DMRS_DESCR_SEQ_RD_BIT_IDX  = DMRS_TONE_DESCR_BIT_IDX % N_DMRS_DESCR_BITS_GEN;
        const uint32_t DMRS_DESCR_SEQ_RD_WORD_IDX = DMRS_TONE_DESCR_BIT_IDX / N_DMRS_DESCR_BITS_GEN;

        for(int j = 0; j < dmrsMaxLen; ++j)
        {
            // descrambling code:
            int8_t descrIBit = (descrWords[j][DMRS_DESCR_SEQ_RD_WORD_IDX] >> DMRS_DESCR_SEQ_RD_BIT_IDX) & 0x1;
            int8_t descrQBit = (descrWords[j][DMRS_DESCR_SEQ_RD_WORD_IDX] >> (DMRS_DESCR_SEQ_RD_BIT_IDX + 1)) & 0x1;

            descramCodeForThread[j] = {RECIPROCAL_SQRT2 * static_cast<__half>((1 - 2 * descrIBit)), -RECIPROCAL_SQRT2 * static_cast<__half>((1 - 2 * descrQBit))};
        }
    }
    __syncthreads();

    // compute projection coefficents and measure noise energy:
    for(int gridIdx = 0; gridIdx < 2; ++gridIdx)
    {
        // check if grid is active:
        gridPrm_t& gridPrm = rkhsUeGrpPrm.gridPrms[gridIdx];
        if(rkhsUeGrpPrm.gridBitMask >> gridIdx)
        {
            // if grid active, loop over tocc:
            for(int toccIdx = 0; toccIdx < dmrsMaxLen; toccIdx++)
            {
                // load data, remove tocc and scrambling sequences:
                for(int antIdx = 0; antIdx < nRxAnt; ++antIdx)
                {
                    __half2 antRxData = {0,0};
                    if(THREAD_IDX < nDmrsSc)
                    {
                        uint16_t scIdx     = 2*THREAD_IDX + startSc + gridIdx;
                        uint8_t  symIdx    = pDmrsSymPos[0];
                        __half2  scRx      = tDataRx(scIdx, symIdx, antIdx);

                        antRxData = complex_mul(scRx, descramCodeForThread[0]);

                        if(dmrsMaxLen > 1)
                        {
                            symIdx = pDmrsSymPos[1];
                            scRx   = tDataRx(scIdx, symIdx, antIdx);
                            if(toccIdx == 1){
                                scRx = -scRx;
                            }

                            antRxData   += complex_mul(scRx, descramCodeForThread[1]);
                            antRxData.x *= RECIPROCAL_SQRT2;
                            antRxData.y *= RECIPROCAL_SQRT2;
                        }
                    }

                    // use Hamming window to measure noise:
                    #ifdef RKHS_USE_HAMMING
                        if(noiseEstMethod > 1)
                        {
                            __half2 hammingCorr;
                            hammingCorr.x = antRxData.x * hammingWindow;
                            hammingCorr.y = antRxData.y * hammingWindow;

                            twoStageFourierTransform<log2SecondStageFourierSize>(THREAD_IDX, hammingCorr, tSecondStageTwiddleFactors, pSecondStageFourierPerm, sh_fourierWorkspace);
                            // if(THREAD_IDX < 32)
                            // {
                            //     printf("\n antIdx = %d, sh_fourierWorkspace[%d] = %f + %fj", antIdx, THREAD_IDX, static_cast<float>(sh_fourierWorkspace[THREAD_IDX].x), static_cast<float>(sh_fourierWorkspace[THREAD_IDX].y));
                            // }
                            if(THREAD_IDX < nNoiseIntsPerGrid)
                            {
                                uint8_t  noiseRegionIdx        = THREAD_IDX / nNoiseIntsPerFocc;
                                uint16_t idxWithinNoiseRegion  = THREAD_IDX - noiseRegionIdx * nNoiseIntsPerFocc;
                                uint8_t  foccIdx               = (noiseRegionIdx + 1) % 2;
                                uint16_t corrIdx               = foccIdx * nZpDmrsSc / 2 + noiseRegionFirstIntIdx + idxWithinNoiseRegion;

                                __half2  corrValue                = sh_fourierWorkspace[corrIdx];
                                noiseEnergyMeasuredByThread_half += corrValue.x * corrValue.x + corrValue.y * corrValue.y;
                                // if(THREAD_IDX < 32)
                                // {
                                //     printf("\n antIdx = %d, noiseEnergyMeasuredByThread_half[%d] = %f", antIdx, corrIdx, static_cast<float>(noiseEnergyMeasuredByThread_half));
                                // }

                            }
                        }
                        __syncthreads();
                    #endif

                    // if((THREAD_IDX < 6))
                    // {
                    //     printf("\n antIdx = %d, antRxData[%d] = %f + %fj", antIdx, THREAD_IDX, static_cast<float>(antRxData.x), static_cast<float>(antRxData.y));
                    // }



                    // correlate antRxData with eigenvectors:
                    #pragma unroll
                    for(int eigIdx = 0; eigIdx < RKHS_N_EIGS; ++eigIdx)
                    {
                        __half2 eigenVecCorr;
                        eigenVecCorr.x = antRxData.x * eigVecValuesForThread[eigIdx];
                        eigenVecCorr.y = antRxData.y * eigVecValuesForThread[eigIdx];

                        twoStageFourierTransform<log2SecondStageFourierSize>(THREAD_IDX, eigenVecCorr, tSecondStageTwiddleFactors, pSecondStageFourierPerm, sh_fourierWorkspace);

                        #ifndef RKHS_USE_HAMMING
                            if(eigIdx == 0)
                            {
                                if(THREAD_IDX < nNoiseIntsPerGrid)
                                {
                                    uint8_t  noiseRegionIdx        = THREAD_IDX / nNoiseIntsPerFocc;
                                    uint16_t idxWithinNoiseRegion  = THREAD_IDX - noiseRegionIdx * nNoiseIntsPerFocc;
                                    uint8_t  foccIdx               = (noiseRegionIdx + 1) % 2;
                                    uint16_t corrIdx               = foccIdx * nZpDmrsSc / 2 + noiseRegionFirstIntIdx + idxWithinNoiseRegion;

                                    __half2  corrValue = sh_fourierWorkspace[corrIdx];
                                    noiseEnergyMeasuredByThread_half += corrValue.x * corrValue.x + corrValue.y * corrValue.y;
                                }
                            }
                        #endif

                        //  save projection coefficents
                        if((threadActiveInMp == 1) && (projCoeffAntIdx == antIdx))
                        {
                            // loop over focc codes
                            toccPrm_t& toccPrm = gridPrm.toccPrms[toccIdx];
                            for(int foccIdx = 0; foccIdx < 2; ++foccIdx)
                            {
                                if(toccPrm.foccBitMask >> foccIdx)
                                {
                                    uint16_t corrIdx             = projCoeffIntIdx + foccIdx * nZpDmrsSc / 2;
                                    uint8_t  layerIdx            = toccPrm.foccPrms[foccIdx].layerIdx;
                                    projCoeffs[layerIdx][eigIdx] = sh_fourierWorkspace[corrIdx];
                                }
                            }
                        }
                        __syncthreads();
                    }
                } // end ant loop
            } // end tOCC loop
        }else if((noiseEstMethod == USE_EMPTY_DMRS_GRID) && (THREAD_IDX < nDmrsSc)) // use empty DMRS grid to measure noise energy
        {
            for(int antIdx = 0; antIdx < nRxAnt; ++antIdx)
            {
                uint16_t scIdx = 2 * THREAD_IDX + startSc + gridIdx;

                for(int j = 0; j < dmrsMaxLen; ++j)
                {
                    uint32_t symIdx = pDmrsSymPos[j];
                    __half2  scRx   = tDataRx(scIdx, symIdx, antIdx);

                    noiseEnergyMeasuredByThread_half += scRx.x * scRx.x + scRx.y * scRx.y;
                }
            }
        }
    } // end grid loop

    // if((0 == blockIdx.x) && (0 == blockIdx.y) && (0 == blockIdx.z) && (0 == threadIdx.x) && (0 == threadIdx.y) && (0 == threadIdx.z))
    // {
    //     // for(int i = 0; i < 3; ++i)
    //     // {
    //     printf("\n eigVecValuesForThread[%d] = %f", i, static_cast<float>(eigVecValuesForThread[i]));
    //     // }
    // }
    // if(projCoeffAntIdx == 0)
    // {
    //     printf("\n nCpInt = %d, projCoeffs[%d] = %f + %fj", nCpInt, projCoeffIntIdx, static_cast<float>(projCoeffs[0][0].x), static_cast<float>(projCoeffs[0][0].y));
    // }

    // combine thread noise measurments into a warp noise measurment:
    float noiseEnergyMeasuredByWarp_float   = static_cast<float>(noiseEnergyMeasuredByThread_half);
    for(int reduceStage = 16; reduceStage > 0; reduceStage /= 2)
    {
        noiseEnergyMeasuredByWarp_float += tile.shfl_down(noiseEnergyMeasuredByWarp_float, reduceStage);
    }

    // atmoic add warp noise measurment to global noise measurment:
    if(LANE_IDX == 0)
    {
        //printf("\n\n WARP_IDX = %d, noiseEnergyMeasuredByWarp_float = %f \n\n", WARP_IDX, noiseEnergyMeasuredByWarp_float);
        atomicAdd(&sh_noiseEnergy, noiseEnergyMeasuredByWarp_float);
    }
    __syncthreads();

    // load global noise measurment and average:
    float noiseScaling = 1;
    #ifdef RKHS_USE_HAMMING
    if(noiseEstMethod > 1)
    {
        float constexpr a = 0.3974;
        float constexpr b = 0.0032;
        noiseScaling      = a * static_cast<float>(nDmrsSc) + b;
    }
    #endif

    N0 = static_cast<__half>(sh_noiseEnergy / (noiseScaling * nNoiseMeasurments));

    // if((THREAD_IDX == 0))
    // {
    //     printf("\n\n BLK_IDX = %d, N0 = %f \n\n", BLK_IDX, static_cast<float>(N0));
    // }

//    matching pursuit:
    uint8_t ueLayerStartIdx = 0;
    for(int ueIdx = 0; ueIdx < nUes; ++ueIdx)
    {
        __half exitCriteria = static_cast<__half>(3) * N0 * static_cast<__half>(nRxAnt) * static_cast<__half>(nUeLayers[ueIdx]);
        uint8_t nEqCoeffs = 0;
        uint8_t updateFlag = 0;

        for(int mpIdx = 0; mpIdx < nCpInt; ++mpIdx)
        {
            __half   antPSD[RKHS_MAX_LAYERS];                         // power-spectral-density for each layer to this antenna
            __half   antCoeffEnergy         = static_cast<__half>(0); // antenna coefficent energy (combines layer energies)
            __half   coeffEnergy            = static_cast<__half>(0); // coefficent energy (combines antCoeffEnergy)
            __half   maxCoeffEnergyInWarp   = static_cast<__half>(0); // max coeffEnergy in warp
            uint16_t maxCoeffInWarpIdx      = 0;                      // index of max coeffEnergy in warp

            // compute energies:
            if(threadActiveInMp)
            {
                for(int ueLayerIdx = 0; ueLayerIdx < nUeLayers[ueIdx]; ++ueLayerIdx)
                {
                    antPSD[ueLayerIdx] = 0;
                    for(int eigIdx = 0; eigIdx < RKHS_N_EIGS; ++eigIdx)
                    {
                        uint8_t  layerIdx    =  ueLayerIdx + ueLayerStartIdx;
                        __half2  projCoeff   =  projCoeffs[layerIdx][eigIdx];
                        antPSD[ueLayerIdx]  +=  (tEigVal(eigIdx) / sumEigValues) * ((projCoeff.x * projCoeff.x + projCoeff.y * projCoeff.y));
                    }
                    antCoeffEnergy     += antPSD[ueLayerIdx];
                    antPSD[ueLayerIdx]  = (antPSD[ueLayerIdx] - N0) / static_cast<__half>(nZpDmrsSc);
                    if(antPSD[ueLayerIdx] < static_cast<__half>(0))
                    {
                        antPSD[ueLayerIdx] = 0;
                    }
                }
            }

            // if coefficent has already been updated set to zero
            if(updateFlag)
            {
                antCoeffEnergy = static_cast<__half>(0);
            }

            // combine antCoeffEnergy to compute coeffEnergy:
            coeffEnergy = antCoeffEnergy;
            for(int reduceStage = nRxAnt / 2; reduceStage > 0; reduceStage /= 2)
            {
                coeffEnergy += tile.shfl_down(coeffEnergy, reduceStage);
            }

            // find most energetic coefficent within the warp:
            maxCoeffEnergyInWarp = coeffEnergy;
            maxCoeffInWarpIdx    = projCoeffIntIdx;
            for(int reduceStage = nRxAnt; reduceStage < 32; reduceStage *= 2)
            {
                __half prop_maxCoeffEnergyInWarp = tile.shfl_down(maxCoeffEnergyInWarp, reduceStage);
                uint16_t prop_maxCoeffInWarpIdx  = tile.shfl_down(maxCoeffInWarpIdx, reduceStage);
                if(prop_maxCoeffEnergyInWarp > maxCoeffEnergyInWarp)
                {
                    maxCoeffEnergyInWarp = prop_maxCoeffEnergyInWarp;
                    maxCoeffInWarpIdx    = prop_maxCoeffInWarpIdx;
                }
            }

            // store max coeff energy/Idx to shared memory
            if(LANE_IDX == 0)
            {
                sh_warpMaxCoeffEnergy[WARP_IDX] = maxCoeffEnergyInWarp;
                sh_warpMaxCoeffIdx[WARP_IDX]    = maxCoeffInWarpIdx;
            }
            __syncthreads();

            // find the max coeff energy/Idx in the threadblock
            __half   maxCoeffEnergy = static_cast<__half>(0);
            uint16_t maxCoeffIdx    = 0;

            for(int warpIdx = 0; warpIdx < nWarpsActiveInMp; ++warpIdx)
            {
                if(sh_warpMaxCoeffEnergy[warpIdx] > maxCoeffEnergy)
                {
                    maxCoeffEnergy = sh_warpMaxCoeffEnergy[warpIdx];
                    maxCoeffIdx    = sh_warpMaxCoeffIdx[warpIdx];
                }
            }

            // if((0 == blockIdx.x) && (0 == blockIdx.y) && (0 == blockIdx.z) && (7 == threadIdx.x) && (0 == threadIdx.y) && (0 == threadIdx.z))
            // {
            //     if(projCoeffIntIdx > 100)
            //     {
            //         printf("\n\n HERE! maxCoeffIdx = %d, THREAD_IDX = %d \n\n", projCoeffIntIdx, THREAD_IDX);
            //     }
            // }

            // check for exit:
            if(maxCoeffEnergy < exitCriteria)
            {
                break;
            }

            // Compute eqCoeff, store in shared memory:
            if(maxCoeffIdx == projCoeffIntIdx)
            {
                updateFlag = 1;
                for(int ueLayerIdx = 0; ueLayerIdx < nUeLayers[ueIdx]; ++ueLayerIdx)
                {
                    for(int eigIdx = 0; eigIdx < RKHS_N_EIGS; ++eigIdx)
                    {
                        uint8_t layerIdx = ueLayerIdx + ueLayerStartIdx;
                        __half2 eqCoeff  = projCoeffs[layerIdx][eigIdx];

                        __half lambda = (antPSD[ueLayerIdx] * tEigVal(eigIdx)) / (antPSD[ueLayerIdx] * tEigVal(eigIdx) + N0);
                        eqCoeff.x     = lambda * eqCoeff.x;
                        eqCoeff.y     = lambda * eqCoeff.y;

                        sh_tEqCoeff(eigIdx, ueLayerIdx, projCoeffAntIdx, nEqCoeffs) = eqCoeff;
                    }
                }
                if(projCoeffAntIdx == 0)
                {
                    sh_eqIntIdxs[nEqCoeffs] = projCoeffIntIdx;
                }
            }
            __syncthreads();

            // load updated equalizer coefficent:
            if(threadActiveInMp)
            {
                __half2 newEqCoeff[RKHS_MAX_LAYERS][RKHS_N_EIGS];
                for(int ueLayerIdx = 0; ueLayerIdx < nUeLayers[ueIdx]; ++ueLayerIdx)
                {
                    for(int eigIdx = 0; eigIdx < RKHS_N_EIGS; ++eigIdx)
                    {
                        newEqCoeff[ueLayerIdx][eigIdx] = sh_tEqCoeff(eigIdx, ueLayerIdx, projCoeffAntIdx, nEqCoeffs);
                    }
                }

                // load correlation matrix:
                __half2 corr[RKHS_N_EIGS][RKHS_N_EIGS];

                uint8_t boxIdxLessTheUpdateBox = (projCoeffIntIdx < maxCoeffIdx) ? 1 : 0;
                int     distBoxIdxToUpdateBox  = abs(static_cast<int>(projCoeffIntIdx) - static_cast<int>(maxCoeffIdx));

                for(int i = 0; i < RKHS_N_EIGS; ++i)
                {
                    for(int j = 0; j < RKHS_N_EIGS; ++j)
                    {
                        __half2 corrValue;
                        corrValue.x = tCorr(j,i,distBoxIdxToUpdateBox).x;
                        corrValue.y = tCorr(j,i,distBoxIdxToUpdateBox).y;

                        if(boxIdxLessTheUpdateBox)
                        {
                            corrValue.y = -corrValue.y;
                        }
                        corr[i][j] = corrValue;
                    }
                }

                // update project coefficents:
                __half2 deltaProjCoeff[RKHS_MAX_LAYERS][RKHS_N_EIGS];
                for(int ueLayerIdx = 0; ueLayerIdx < nUeLayers[ueIdx]; ++ueLayerIdx)
                {
                    cplxMtrxVecMult(deltaProjCoeff[ueLayerIdx], corr, newEqCoeff[ueLayerIdx]);
                }

                for(int ueLayerIdx = 0; ueLayerIdx < RKHS_MAX_LAYERS; ++ueLayerIdx)
                {
                    for(int eigIdx = 0; eigIdx < RKHS_N_EIGS; ++eigIdx)
                    {
                        projCoeffs[ueLayerIdx][eigIdx] -= deltaProjCoeff[ueLayerIdx][eigIdx];
                    }
                }
            }
            // update number of equalization coefficents:
            nEqCoeffs += 1;
            // __syncthreads();

            // if((0 == blockIdx.x) && (0 == blockIdx.y) && (0 == blockIdx.z) && (0 == threadIdx.x) && (0 == threadIdx.y) && (0 == threadIdx.z))
            // {
            //     //  printf("\n tHEst(%d, %d, %d, %d).x = %f + j%f", antIdx, layerIdx, THREAD_IDX, chEstTimeInst, tHEst(antIdx, layerIdx, THREAD_IDX, chEstTimeInst).x, tHEst(antIdx, layerIdx, THREAD_IDX, chEstTimeInst).y);
            //     // printf("\n\n mpIdx = %d, nEqCoeffs = %d, N0 = %f \n\n", mpIdx, nEqCoeffs, static_cast<__half>(N0));
            //     // for(int i = 0; i < 32; ++i)
            //     // {
            //     //     printf("\n sh_eqIntIdxs[%d] = %d", i, sh_eqIntIdxs[i]);
            //     // }
            //     // printf("\n\n");
            //   //  printf("\n\n mpIdx = %d, maxCoeffEnergy = %f \n\n", mpIdx, static_cast<float>(maxCoeffEnergy));
            // }
        }

        // if((THREAD_IDX == 0))
        // {
        //     printf("\n\n BLK_IDX = %d, nEqCoeffs = %d \n\n", BLK_IDX, nEqCoeffs);
        // }

        // perform interpolation:
        __half2 H_est_sc[4][4];

        if(THREAD_IDX < nOutputSc)
        {
            // compute block subcarrier index:
            uint16_t scIdx = THREAD_IDX + startOutputScInBlock;

            // initialize thread chEst to zero:
            for(int ueLayerIdx = 0; ueLayerIdx < nUeLayers[ueIdx]; ++ueLayerIdx)
            {
                for(int antIdx = 0; antIdx < nRxAnt; ++antIdx)
                {
                    H_est_sc[ueLayerIdx][antIdx] = {static_cast<__half>(0), static_cast<__half>(0)};
                }
            }

            // load interpolation eigenvectors:
            __half interpVecValuesForThread[RKHS_N_EIGS];
            realMtrxVecMult<RKHS_N_EIGS, RKHS_N_ZP_EIGS>(interpVecValuesForThread, tInterpCob, tZpInterpVec.trace(scIdx));

            for(int eqIdx = 0; eqIdx < nEqCoeffs; ++eqIdx)
            {
                // compute wave for this subcarrier:
                int eqIntIdx = sh_eqIntIdxs[eqIdx];

                __half2  wave[2];
                for(int gridOffset = 0; gridOffset < 2; ++gridOffset)
                {
                    float2 waveFloat;
                    float phase = -3.14159265358979323846 * static_cast<float>((scIdx - gridOffset) * eqIntIdx) / static_cast<float>(nZpDmrsSc);
                    __sincosf(phase, &waveFloat.y, &waveFloat.x);
                    wave[gridOffset] = __float22half2_rn(waveFloat);
                //    wave[gridOffset] = {static_cast<__half>(1), static_cast<__half>(0)};
                }

                for(int ueLayerIdx = 0; ueLayerIdx < nUeLayers[ueIdx]; ++ueLayerIdx)
                {
                    uint8_t layerIdx = ueLayerIdx + ueLayerStartIdx;
                    uint8_t gridIdx  = gridIdxs[layerIdx];

                    for(int antIdx = 0; antIdx < nRxAnt; ++antIdx)
                    {
                        __half2 interpValue = {0,0};
                        for(int eigIdx = 0; eigIdx < RKHS_N_EIGS; ++eigIdx)
                        {
                            __half2 eqCoeff = sh_tEqCoeff(eigIdx, ueLayerIdx, antIdx, eqIdx);
                            interpValue.x += interpVecValuesForThread[eigIdx] * eqCoeff.x;
                            interpValue.y += interpVecValuesForThread[eigIdx] * eqCoeff.y;
                        }

                        __half2 modulatedInterpValue;
                        modulatedInterpValue.x = (wave[gridIdx].x * interpValue.x - wave[gridIdx].y * interpValue.y);
                        modulatedInterpValue.y = (wave[gridIdx].x * interpValue.y + wave[gridIdx].y * interpValue.x);

                        const __half RECIPROCAL_SQRT2 = 0.7071068f;
                        if(drvdUeGrpPrm.nDmrsCdmGrpsNoData == 2)
                        {
                            modulatedInterpValue.x *= RECIPROCAL_SQRT2;
                            modulatedInterpValue.y *= RECIPROCAL_SQRT2;
                        }

                        // if((0 == blockIdx.x) && (0 == blockIdx.y) && (0 == blockIdx.z) && (1 == threadIdx.x) && (0 == threadIdx.y) && (0 == threadIdx.z))
                        // {
                        //     printf("\n modulatedInterpValue = %f + j%f", static_cast<float>(modulatedInterpValue.x), static_cast<float>(modulatedInterpValue.y));
                        // }

                        // if((0 == blockIdx.x) && (0 == blockIdx.y) && (0 == blockIdx.z) && (1 == threadIdx.x) && (0 == threadIdx.y) && (0 == threadIdx.z))
                        // {
                        //     printf("\n ueLayerIdx = %d, antIdx = %d, projCoeffAntIdx = %d, scIdx = %d", ueLayerIdx, antIdx, projCoeffAntIdx, scIdx);
                        // }
                        H_est_sc[ueLayerIdx][antIdx] += modulatedInterpValue;
                    }
                }
            }

            for(int ueLayerIdx = 0; ueLayerIdx < nUeLayers[ueIdx]; ++ueLayerIdx)
            {
                uint8_t layerIdx = ueLayerIdx + ueLayerStartIdx;
                for(int antIdx = 0; antIdx < nRxAnt; ++antIdx)
                {
                    // tHEst(antIdx, layerIdx, scIdx, chEstTimeInst) = H_est_sc[ueLayerIdx][antIdx];
                    tHEst(antIdx, layerIdx, THREAD_IDX + scOffsetIntoChEstBuff, chEstTimeInst).x = static_cast<float>(H_est_sc[ueLayerIdx][antIdx].x);
                    tHEst(antIdx, layerIdx, THREAD_IDX + scOffsetIntoChEstBuff, chEstTimeInst).y = static_cast<float>(H_est_sc[ueLayerIdx][antIdx].y);


                //  //   __syncthreads();
                //     if((0 == blockIdx.x) && (0 == blockIdx.y) && (0 == blockIdx.z) && (1 == threadIdx.x) && (0 == threadIdx.y) && (0 == threadIdx.z))
                //     {
                //       //  printf("\n tHEst(%d, %d, %d, %d).x = %f + j%f", antIdx, layerIdx, THREAD_IDX, chEstTimeInst, tHEst(antIdx, layerIdx, THREAD_IDX, chEstTimeInst).x, tHEst(antIdx, layerIdx, THREAD_IDX, chEstTimeInst).y);
                //       printf("\n H_est_sc[%d][%d][%d] = %f + j%f", ueLayerIdx, antIdx, THREAD_IDX, static_cast<float>(H_est_sc[ueLayerIdx][antIdx].x), static_cast<float>(H_est_sc[ueLayerIdx][antIdx].y));
                //     }

                }
            }
        }

    }
}



 template <uint32_t N_DMRS_GRIDS_PER_PRB,              // # of DMRS grids per PRB (2 or 3)
           uint32_t N_DMRS_PRB_IN_PER_CLUSTER,         // # of PRBs bearing DMRS tones to be processed by each thread block (i.e. used in channel estimation)
           uint32_t N_DMRS_INTERP_PRB_OUT_PER_CLUSTER> // # of PRBs bearing interpolated tones at output
 void
 puschRxChEstKernelBuilder::computeKernelLaunchGeo(uint16_t nTotalDataPrb,
                                      uint16_t nUeGrps,
                                      uint32_t nRxAnt,
                                      dim3&    gridDim,
                                      dim3&    blockDim)
 {
     constexpr uint32_t N_TOTAL_INPUT_TONES = N_DMRS_PRB_IN_PER_CLUSTER * N_TONES_PER_PRB;
     constexpr uint32_t N_TOTAL_OUTPUT_TONES = N_DMRS_INTERP_PRB_OUT_PER_CLUSTER * N_DMRS_GRIDS_PER_PRB * N_TONES_PER_PRB;
     static_assert((N_TOTAL_INPUT_TONES == N_TOTAL_OUTPUT_TONES),
                   "Thread allocation assumes input DMRS tone count and interpolated tone count are equal, ensure sufficient threads are allocated for interpoloation etc");

     const uint32_t N_THREAD_BLKS_PER_BS_ANT = div_round_up(nTotalDataPrb, static_cast<uint16_t>(N_DMRS_INTERP_PRB_OUT_PER_CLUSTER));
     gridDim  = dim3(N_THREAD_BLKS_PER_BS_ANT, nRxAnt, nUeGrps);
     blockDim = dim3(N_TOTAL_OUTPUT_TONES);

#ifdef ENABLE_DEBUG
     NVLOGI_FMT(NVLOG_PUSCH, "{}: blockDim ({},{},{}), gridDim ({},{},{})", __FUNCTION__, blockDim.x, blockDim.y, blockDim.z, gridDim.x, gridDim.y, gridDim.z);
#endif
 }

 template <typename TStorage,
           typename TDataRx,
           typename TCompute,
           uint32_t N_LAYERS,                          // # of layers (# of cols in H matrix)
           uint32_t N_DMRS_GRIDS_PER_PRB,              // # of DMRS grids per PRB (2 or 3)
           uint32_t N_DMRS_PRB_IN_PER_CLUSTER,         // # of PRBs bearing DMRS tones to be processed by each thread block (i.e. used in channel estimation)
           uint32_t N_DMRS_INTERP_PRB_OUT_PER_CLUSTER, // # of PRBs bearing channel estimates (interpolated tones) at output
           uint32_t N_DMRS_SYMS>                       // # of time domain DMRS symbols (1,2 or 4)
 void
 puschRxChEstKernelBuilder::multiStageChEst(uint8_t            enablePerPrgChEstPerUeg,
                               uint16_t                        nTotalDataPrb,
                               uint16_t                        nUeGrps,
                               uint32_t                        nRxAnt,
                               uint8_t                         enableDftSOfdm,
                               uint8_t                         enablePerPrgChEst, 
                               cuphyPuschRxChEstLaunchCfg_t&   launchCfg)
 {
     if((enablePerPrgChEst==1)&&(enablePerPrgChEstPerUeg==1))
     {
         void* kernelFunc = reinterpret_cast<void*>(windowedChEstPreNoDftSOfdmKernel<TStorage,
                                                        TDataRx,
                                                        TCompute,
                                                        N_DMRS_GRIDS_PER_PRB,
                                                        N_DMRS_SYMS,
                                                        CUPHY_PUSCH_RX_CH_EST_PRG_DELAY_EST_PRB_CLUSTER_SIZE>);
    
         {MemtraceDisableScope md; CUDA_CHECK(cudaGetFuncBySymbol(&launchCfg.kernelNodeParamsDriver.func, kernelFunc));}
    
         void* kernelFunc2 = reinterpret_cast<void*>(chEstFilterPrgNoDftSOfdmDispatchKernel<TStorage,
                                                        TDataRx,
                                                        TCompute,
                                                        N_DMRS_GRIDS_PER_PRB>);
    
         {MemtraceDisableScope md; CUDA_CHECK(cudaGetFuncBySymbol(&launchCfg.kernelNodeParamsDriverSecond.func, kernelFunc2));}
     }
     else
     {
         void* kernelFunc = reinterpret_cast<void*>(windowedChEstPreNoDftSOfdmKernel<TStorage,
                                                        TDataRx,
                                                        TCompute,
                                                        N_DMRS_GRIDS_PER_PRB,
                                                        N_DMRS_SYMS,
                                                        CUPHY_PUSCH_RX_CH_EST_NON_PRG_DELAY_EST_PRB_CLUSTER_SIZE>);
    
         {MemtraceDisableScope md; CUDA_CHECK(cudaGetFuncBySymbol(&launchCfg.kernelNodeParamsDriver.func, kernelFunc));}
    
         void* kernelFunc2 = reinterpret_cast<void*>(chEstFilterNoDftSOfdmDispatchKernel<TStorage,
                                                        TDataRx,
                                                        TCompute,
                                                        N_DMRS_GRIDS_PER_PRB>);
    
         {MemtraceDisableScope md; CUDA_CHECK(cudaGetFuncBySymbol(&launchCfg.kernelNodeParamsDriverSecond.func, kernelFunc2));}
     }

     // We will not know the grid dimensions for the launch configuration until we know the number
     // of PRBs for all included UE groups. Importantly, the largest grid.x dimension for the second
     // kernel does not necessarily depend upon the highest PRB count, so just passing the max
     // PRB count to this function is not sufficient. The grid dimensions are thus computed and
     // set higher in the call stack when we can check nPrb for all UE groups.

     CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriver = launchCfg.kernelNodeParamsDriver;
     kernelNodeParamsDriver.extra          = nullptr;
     kernelNodeParamsDriver.sharedMemBytes = 0;
     CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriverSecond = launchCfg.kernelNodeParamsDriverSecond;
     kernelNodeParamsDriverSecond.extra          = nullptr;
     kernelNodeParamsDriverSecond.sharedMemBytes = 0;
 }

 template <typename TStorage,
           typename TDataRx,
           typename TCompute,
           uint32_t N_LAYERS,                          // # of layers (# of cols in H matrix)
           uint32_t N_DMRS_GRIDS_PER_PRB,              // # of DMRS grids per PRB (2 or 3)
           uint32_t N_DMRS_PRB_IN_PER_CLUSTER,         // # of PRBs bearing DMRS tones to be processed by each thread block (i.e. used in channel estimation)
           uint32_t N_DMRS_INTERP_PRB_OUT_PER_CLUSTER, // # of PRBs bearing channel estimates (interpolated tones) at output
           uint32_t N_DMRS_SYMS>                       // # of time domain DMRS symbols (1,2 or 4)
 void
 puschRxChEstKernelBuilder::lsChEst(uint16_t           nTotalDataPrb,
                       uint16_t                        nUeGrps,
                       uint32_t                        nRxAnt,
                       uint8_t                         enableDftSOfdm,
                       cuphyPuschRxChEstLaunchCfg_t&   launchCfg)
 {
     void* kernelFunc = reinterpret_cast<void*>(windowedChEstPreNoDftSOfdmKernel<TStorage,
                                                    TDataRx,
                                                    TCompute,
                                                    N_DMRS_GRIDS_PER_PRB,
                                                    N_DMRS_SYMS,
                                                    CUPHY_PUSCH_RX_CH_EST_NON_PRG_DELAY_EST_PRB_CLUSTER_SIZE>);

     {MemtraceDisableScope md; CUDA_CHECK(cudaGetFuncBySymbol(&launchCfg.kernelNodeParamsDriver.func, kernelFunc));}

     CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriver = launchCfg.kernelNodeParamsDriver;
     kernelNodeParamsDriver.extra = nullptr;
     kernelNodeParamsDriver.sharedMemBytes = 0;

     // The second kernel not used.
     launchCfg.kernelNodeParamsDriverSecond.func = nullptr;
 }


 template <typename TStorage,
           typename TDataRx,
           typename TCompute,
           uint32_t N_LAYERS,                          // # of layers (# of cols in H matrix)
           uint32_t N_DMRS_GRIDS_PER_PRB,              // # of DMRS grids per PRB (2 or 3)
           uint32_t N_DMRS_PRB_IN_PER_CLUSTER,         // # of PRBs bearing DMRS tones to be processed by each thread block (i.e. used in channel estimation)
           uint32_t N_DMRS_INTERP_PRB_OUT_PER_CLUSTER, // # of PRBs bearing channel estimates (interpolated tones) at output
           uint32_t N_DMRS_SYMS>                       // # of time domain DMRS symbols (1,2 or 4)
 void
 puschRxChEstKernelBuilder::windowedChEst(uint16_t           nTotalDataPrb,
                             uint16_t                        nUeGrps,
                             uint32_t                        nRxAnt,
                             uint8_t                         enableDftSOfdm,
                             cuphyPuschRxChEstLaunchCfg_t&   launchCfg)
 {
     if(enableDftSOfdm==1)
     {
         void* kernelFunc = reinterpret_cast<void*>(windowedChEstKernel<TStorage,
                                                                    TDataRx,
                                                                    TCompute,
                                                                    N_LAYERS,
                                                                    N_DMRS_GRIDS_PER_PRB,
                                                                    N_DMRS_PRB_IN_PER_CLUSTER,
                                                                    N_DMRS_INTERP_PRB_OUT_PER_CLUSTER,
                                                                    N_DMRS_SYMS>);

         {MemtraceDisableScope md; CUDA_CHECK(cudaGetFuncBySymbol(&launchCfg.kernelNodeParamsDriver.func, kernelFunc));}
     }
     else if(enableDftSOfdm==0)
     {
         void* kernelFunc = reinterpret_cast<void*>(windowedChEstNoDftSOfdmKernel<TStorage,
                                                                    TDataRx,
                                                                    TCompute,
                                                                    N_LAYERS,
                                                                    N_DMRS_GRIDS_PER_PRB,
                                                                    N_DMRS_PRB_IN_PER_CLUSTER,
                                                                    N_DMRS_INTERP_PRB_OUT_PER_CLUSTER,
                                                                    N_DMRS_SYMS>);

         {MemtraceDisableScope md; CUDA_CHECK(cudaGetFuncBySymbol(&launchCfg.kernelNodeParamsDriver.func, kernelFunc));}

     }
     // else
     // {
     //     assert(false);
     // }

     dim3 blockDim, gridDim;
     computeKernelLaunchGeo<N_DMRS_GRIDS_PER_PRB,
                            N_DMRS_PRB_IN_PER_CLUSTER,
                            N_DMRS_INTERP_PRB_OUT_PER_CLUSTER>(nTotalDataPrb, nUeGrps, nRxAnt, gridDim, blockDim);

     CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriver = launchCfg.kernelNodeParamsDriver;
     kernelNodeParamsDriver.blockDimX = blockDim.x;
     kernelNodeParamsDriver.blockDimY = blockDim.y;
     kernelNodeParamsDriver.blockDimZ = blockDim.z;

     kernelNodeParamsDriver.gridDimX = gridDim.x;
     kernelNodeParamsDriver.gridDimY = gridDim.y;
     kernelNodeParamsDriver.gridDimZ = gridDim.z;

     kernelNodeParamsDriver.extra          = nullptr;
     kernelNodeParamsDriver.sharedMemBytes = 0;

     // The legacy algorithm does not launch a second kernel
     launchCfg.kernelNodeParamsDriverSecond.func = nullptr;
 }


 template <typename TStorage,
           typename TDataRx,
           typename TCompute,
           uint32_t N_LAYERS,                          // # of layers (# of cols in H matrix)
           uint32_t N_PRBS,                            // 1, 2 or 3
           uint32_t N_DMRS_GRIDS_PER_PRB,              // # of DMRS grids per PRB (2 or 3)
           uint32_t N_DMRS_SYMS>                       // # of time domain DMRS symbols (1,2 or 4)
 void
 puschRxChEstKernelBuilder::smallChEst(uint16_t                        nUeGrps,
                          uint32_t                        nRxAnt,
                          uint8_t                         enableDftSOfdm,
                          cuphyPuschRxChEstLaunchCfg_t&   launchCfg)
 {

     if(enableDftSOfdm == 1)
     {
         void* kernelFunc = reinterpret_cast<void*>(smallChEstKernel<TStorage,
                                                                    TDataRx,
                                                                    TCompute,
                                                                    N_LAYERS,
                                                                    N_PRBS,
                                                                    N_DMRS_GRIDS_PER_PRB,
                                                                    N_DMRS_SYMS>);

         {MemtraceDisableScope md; CUDA_CHECK(cudaGetFuncBySymbol(&launchCfg.kernelNodeParamsDriver.func, kernelFunc));}
     }
     else if(enableDftSOfdm==0)
     {
         void* kernelFunc = reinterpret_cast<void*>(smallChEstNoDftSOfdmKernel<TStorage,
                                                                               TDataRx,
                                                                               TCompute,
                                                                               N_LAYERS,
                                                                               N_PRBS,
                                                                               N_DMRS_GRIDS_PER_PRB,
                                                                               N_DMRS_SYMS>);

         {MemtraceDisableScope md; CUDA_CHECK(cudaGetFuncBySymbol(&launchCfg.kernelNodeParamsDriver.func, kernelFunc));}
     }

     // compute launch geometry:
     uint32_t  N_THREADS = 2 * N_PRBS * N_TONES_PER_PRB;

     dim3 gridDims(1, nRxAnt, nUeGrps);
     dim3 blockDims(N_THREADS);

     CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriver = launchCfg.kernelNodeParamsDriver;
     kernelNodeParamsDriver.blockDimX = blockDims.x;
     kernelNodeParamsDriver.blockDimY = blockDims.y;
     kernelNodeParamsDriver.blockDimZ = blockDims.z;

     kernelNodeParamsDriver.gridDimX = gridDims.x;
     kernelNodeParamsDriver.gridDimY = gridDims.y;
     kernelNodeParamsDriver.gridDimZ = gridDims.z;

     kernelNodeParamsDriver.extra          = nullptr;
     kernelNodeParamsDriver.sharedMemBytes = 0;
 }

 template <typename TStorage,
           typename TDataRx,
           typename TCompute,
           uint32_t N_LAYERS,
           uint32_t N_DMRS_GRIDS_PER_PRB,
           uint32_t N_DMRS_SYMS>
 void puschRxChEstKernelBuilder::kernelSelectL0(uint8_t            enablePerPrgChEstPerUeg,
                                   uint16_t                        nTotalDataPrb,
                                   uint16_t                        nUeGrps,
                                   uint32_t                        nRxAnt,
                                   uint8_t                         enableDftSOfdm,
                                   uint8_t                         chEstAlgo,
                                   uint8_t                         enablePerPrgChEst,
                                   cuphyPuschRxChEstLaunchCfg_t&   launchCfg)
 {
     bool noKernelFound = false;

     if((0 == (nTotalDataPrb % 4)) && (nTotalDataPrb > 7)) // (nTotalDataPrb >= 8)
     {
         constexpr uint32_t N_DMRS_PRB_IN_PER_CLUSTER         = 8; // # of DMRS PRBs processed by a thread block
         constexpr uint32_t N_DMRS_INTERP_PRB_OUT_PER_CLUSTER = 4; // # of DMRS interpolated PRBs produced by a thread block

         if (chEstAlgo==PUSCH_CH_EST_ALGO_TYPE_LEGACY_MMSE) {
            windowedChEst<TStorage,
                            TDataRx,
                            TCompute,
                            N_LAYERS,
                            N_DMRS_GRIDS_PER_PRB,
                            N_DMRS_PRB_IN_PER_CLUSTER,
                            N_DMRS_INTERP_PRB_OUT_PER_CLUSTER,
                            N_DMRS_SYMS>(nTotalDataPrb, nUeGrps, nRxAnt, enableDftSOfdm, launchCfg);
         } else if(chEstAlgo==PUSCH_CH_EST_ALGO_TYPE_MULTISTAGE_MMSE_WITH_DELAY_EST) {
            multiStageChEst<TStorage,
                            TDataRx,
                            TCompute,
                            N_LAYERS,
                            N_DMRS_GRIDS_PER_PRB,
                            N_DMRS_PRB_IN_PER_CLUSTER,
                            N_DMRS_INTERP_PRB_OUT_PER_CLUSTER,
                            N_DMRS_SYMS>(enablePerPrgChEstPerUeg, nTotalDataPrb, nUeGrps, nRxAnt, enableDftSOfdm, enablePerPrgChEst, launchCfg);
         } else if(chEstAlgo==PUSCH_CH_EST_ALGO_TYPE_LS_ONLY) {
            lsChEst<TStorage,
                    TDataRx,
                    TCompute,
                    N_LAYERS,
                    N_DMRS_GRIDS_PER_PRB,
                    N_DMRS_PRB_IN_PER_CLUSTER,
                    N_DMRS_INTERP_PRB_OUT_PER_CLUSTER,
                    N_DMRS_SYMS>(nTotalDataPrb, nUeGrps, nRxAnt, enableDftSOfdm, launchCfg);
         }
     }
     else if(((0 != (nTotalDataPrb % 4)) || (nTotalDataPrb == 4)) && (nTotalDataPrb > 3))  // (nTotalDataPrb >= 4)
     {
         constexpr uint32_t N_DMRS_PRB_IN_PER_CLUSTER         = 4; // # of DMRS PRBs processed by a thread block
         constexpr uint32_t N_DMRS_INTERP_PRB_OUT_PER_CLUSTER = 2; // # of DMRS interpolated PRBs produced by a thread block
         if (chEstAlgo==PUSCH_CH_EST_ALGO_TYPE_LEGACY_MMSE) {
            windowedChEst<TStorage,
                            TDataRx,
                            TCompute,
                            N_LAYERS,
                            N_DMRS_GRIDS_PER_PRB,
                            N_DMRS_PRB_IN_PER_CLUSTER,
                            N_DMRS_INTERP_PRB_OUT_PER_CLUSTER,
                            N_DMRS_SYMS>(nTotalDataPrb, nUeGrps, nRxAnt, enableDftSOfdm, launchCfg);
         } else if(chEstAlgo==PUSCH_CH_EST_ALGO_TYPE_MULTISTAGE_MMSE_WITH_DELAY_EST) {
            multiStageChEst<TStorage,
                            TDataRx,
                            TCompute,
                            N_LAYERS,
                            N_DMRS_GRIDS_PER_PRB,
                            N_DMRS_PRB_IN_PER_CLUSTER,
                            N_DMRS_INTERP_PRB_OUT_PER_CLUSTER,
                            N_DMRS_SYMS>(enablePerPrgChEstPerUeg, nTotalDataPrb, nUeGrps, nRxAnt, enableDftSOfdm, enablePerPrgChEst, launchCfg);
         } else if(chEstAlgo==PUSCH_CH_EST_ALGO_TYPE_LS_ONLY) {
            lsChEst<TStorage,
                    TDataRx,
                    TCompute,
                    N_LAYERS,
                    N_DMRS_GRIDS_PER_PRB,
                    N_DMRS_PRB_IN_PER_CLUSTER,
                    N_DMRS_INTERP_PRB_OUT_PER_CLUSTER,
                    N_DMRS_SYMS>(nTotalDataPrb, nUeGrps, nRxAnt, enableDftSOfdm, launchCfg);
         }
     }
     else if(nTotalDataPrb < 4) // (nTotalDataPrb < 4)
     {
         switch(nTotalDataPrb)
         {
             case 3:
             {
                 constexpr uint32_t N_PRBS = 3;
                 if (chEstAlgo==PUSCH_CH_EST_ALGO_TYPE_LEGACY_MMSE) {
                     smallChEst<TStorage,
                                 TDataRx,
                                 TCompute,
                                 N_LAYERS,
                                 N_PRBS,
                                 N_DMRS_GRIDS_PER_PRB,
                                 N_DMRS_SYMS>(nUeGrps, nRxAnt, enableDftSOfdm, launchCfg);
                 } else if(chEstAlgo==PUSCH_CH_EST_ALGO_TYPE_MULTISTAGE_MMSE_WITH_DELAY_EST) {
                    multiStageChEst<TStorage,
                                    TDataRx,
                                    TCompute,
                                    N_LAYERS,
                                    N_DMRS_GRIDS_PER_PRB,
                                    N_PRBS,
                                    N_PRBS,
                                    N_DMRS_SYMS>(enablePerPrgChEstPerUeg, nTotalDataPrb, nUeGrps, nRxAnt, enableDftSOfdm, enablePerPrgChEst, launchCfg);
                 } else if(chEstAlgo==PUSCH_CH_EST_ALGO_TYPE_LS_ONLY) {
                     lsChEst<TStorage,
                             TDataRx,
                             TCompute,
                             N_LAYERS,
                             N_DMRS_GRIDS_PER_PRB,
                             N_PRBS,
                             N_PRBS,
                             N_DMRS_SYMS>(nTotalDataPrb, nUeGrps, nRxAnt, enableDftSOfdm, launchCfg);
                 }
                 break;
             }

             case 2:
             {
                 constexpr uint32_t N_PRBS = 2;
                 if (chEstAlgo==PUSCH_CH_EST_ALGO_TYPE_LEGACY_MMSE) {
                    smallChEst<TStorage,
                                TDataRx,
                                TCompute,
                                N_LAYERS,
                                N_PRBS,
                                N_DMRS_GRIDS_PER_PRB,
                                N_DMRS_SYMS>(nUeGrps, nRxAnt, enableDftSOfdm, launchCfg);
                 } else if(chEstAlgo==PUSCH_CH_EST_ALGO_TYPE_MULTISTAGE_MMSE_WITH_DELAY_EST) {
                    multiStageChEst<TStorage,
                                    TDataRx,
                                    TCompute,
                                    N_LAYERS,
                                    N_DMRS_GRIDS_PER_PRB,
                                    N_PRBS,
                                    N_PRBS,
                                    N_DMRS_SYMS>(enablePerPrgChEstPerUeg, nTotalDataPrb, nUeGrps, nRxAnt, enableDftSOfdm, enablePerPrgChEst, launchCfg);
                 } else if(chEstAlgo==PUSCH_CH_EST_ALGO_TYPE_LS_ONLY) {
                     lsChEst<TStorage,
                             TDataRx,
                             TCompute,
                             N_LAYERS,
                             N_DMRS_GRIDS_PER_PRB,
                             N_PRBS,
                             N_PRBS,
                             N_DMRS_SYMS>(nTotalDataPrb, nUeGrps, nRxAnt, enableDftSOfdm, launchCfg);
                 }
                 break;
             }

             case 1:
             {
                 constexpr uint32_t N_PRBS = 1;
                 if (chEstAlgo==PUSCH_CH_EST_ALGO_TYPE_LEGACY_MMSE) {
                    smallChEst<TStorage,
                                TDataRx,
                                TCompute,
                                N_LAYERS,
                                N_PRBS,
                                N_DMRS_GRIDS_PER_PRB,
                                N_DMRS_SYMS>(nUeGrps, nRxAnt, enableDftSOfdm, launchCfg);
                 } else if(chEstAlgo==PUSCH_CH_EST_ALGO_TYPE_MULTISTAGE_MMSE_WITH_DELAY_EST) {
                    multiStageChEst<TStorage,
                                    TDataRx,
                                    TCompute,
                                    N_LAYERS,
                                    N_DMRS_GRIDS_PER_PRB,
                                    N_PRBS,
                                    N_PRBS,
                                    N_DMRS_SYMS>(enablePerPrgChEstPerUeg, nTotalDataPrb, nUeGrps, nRxAnt, enableDftSOfdm, enablePerPrgChEst, launchCfg);
                 } else if(chEstAlgo==PUSCH_CH_EST_ALGO_TYPE_LS_ONLY) {
                     lsChEst<TStorage,
                             TDataRx,
                             TCompute,
                             N_LAYERS,
                             N_DMRS_GRIDS_PER_PRB,
                             N_PRBS,
                             N_PRBS,
                             N_DMRS_SYMS>(nTotalDataPrb, nUeGrps, nRxAnt, enableDftSOfdm, launchCfg);
                 }
                 break;
             }
         }
     }
     else
     {
         noKernelFound = true;
     }
     if(noKernelFound)
     {
         NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "{}: No kernel available nTotalDataPrb {}", __FUNCTION__, nTotalDataPrb);
     }
 }

 void puschRxChEstKernelBuilder::rkhsKernelSelectL1(uint16_t nTotalDataPrb, cuphyPuschRxChEstLaunchCfg_t& launchCfg)
 {
    uint16_t nDmrsSc                      = nTotalDataPrb * 6;
    uint16_t kernelSelect_log2numZpDmrsSc = 9;

    for(int i = 3; i < 10; ++i)
    {
        if(nDmrsSc <= (1 << i))
        {
            kernelSelect_log2numZpDmrsSc = i;
            break;
        }
    }

    if(nTotalDataPrb < 4)
    {
        kernelSelect_log2numZpDmrsSc += 2;
    }else
    {
        kernelSelect_log2numZpDmrsSc += 1;
    }

  //  printf("\n\n kernelSelect_log2numZpDmrsSc = %d, nDmrsSc = %d \n\n", kernelSelect_log2numZpDmrsSc, nDmrsSc);

    if(kernelSelect_log2numZpDmrsSc == 5)
    {
        constexpr uint16_t log2numZpDmrsSc = 5;
        constexpr uint16_t nZpDmrsSc       = 32;

        void* kernelFunc = reinterpret_cast<void*>(puschRkhsChEstKernel<nZpDmrsSc, log2numZpDmrsSc>);
        {MemtraceDisableScope md; CUDA_CHECK(cudaGetFuncBySymbol(&launchCfg.kernelNodeParamsDriver.func, kernelFunc));}

    }else if(kernelSelect_log2numZpDmrsSc == 6)
    {
        constexpr uint16_t log2numZpDmrsSc = 6;
        constexpr uint16_t nZpDmrsSc       = 64;

        void* kernelFunc = reinterpret_cast<void*>(puschRkhsChEstKernel<nZpDmrsSc, log2numZpDmrsSc>);
        {MemtraceDisableScope md; CUDA_CHECK(cudaGetFuncBySymbol(&launchCfg.kernelNodeParamsDriver.func, kernelFunc));}

   }else if(kernelSelect_log2numZpDmrsSc == 7)
   {
        constexpr uint16_t log2numZpDmrsSc = 7;
        constexpr uint16_t nZpDmrsSc       = 128;

        void* kernelFunc = reinterpret_cast<void*>(puschRkhsChEstKernel<nZpDmrsSc, log2numZpDmrsSc>);
        {MemtraceDisableScope md; CUDA_CHECK(cudaGetFuncBySymbol(&launchCfg.kernelNodeParamsDriver.func, kernelFunc));}

   }else if(kernelSelect_log2numZpDmrsSc == 8)
   {
        constexpr uint16_t log2numZpDmrsSc = 8;
        constexpr uint16_t nZpDmrsSc       = 256;

        void* kernelFunc = reinterpret_cast<void*>(puschRkhsChEstKernel<nZpDmrsSc, log2numZpDmrsSc>);
        {MemtraceDisableScope md; CUDA_CHECK(cudaGetFuncBySymbol(&launchCfg.kernelNodeParamsDriver.func, kernelFunc));}

   }else if(kernelSelect_log2numZpDmrsSc == 9)
   {
        constexpr uint16_t log2numZpDmrsSc = 9;
        constexpr uint16_t nZpDmrsSc       = 512;

        void* kernelFunc = reinterpret_cast<void*>(puschRkhsChEstKernel<nZpDmrsSc, log2numZpDmrsSc>);
        {MemtraceDisableScope md; CUDA_CHECK(cudaGetFuncBySymbol(&launchCfg.kernelNodeParamsDriver.func, kernelFunc));}

   }else if(kernelSelect_log2numZpDmrsSc == 10)
   {
        constexpr uint16_t log2numZpDmrsSc = 10;
        constexpr uint16_t nZpDmrsSc       = 1024;

        void* kernelFunc = reinterpret_cast<void*>(puschRkhsChEstKernel<nZpDmrsSc, log2numZpDmrsSc>);
        {MemtraceDisableScope md; CUDA_CHECK(cudaGetFuncBySymbol(&launchCfg.kernelNodeParamsDriver.func, kernelFunc));}
   }


    // constexpr uint16_t nZpDmrs = 32;
    // // using FFT = decltype( cufftdx::Size<nZpDmrs>() + cufftdx::Type<fft_type::c2c>()
    // //             + cufftdx::Direction<fft_direction::inverse>()
    // //             + cufftdx::Precision<__half>() + SM<800>() + cufftdx::Block() +  cufftdx::ElementsPerThread<2>() + FFTsPerBlock<2>());

    // void* kernelFunc = reinterpret_cast<void*>(puschRkhsChEstKernel<nZpDmrs>);
    // CUDA_CHECK(cudaGetFuncBySymbol(&launchCfg.kernelNodeParamsDriver.func, kernelFunc));
 }


 template <typename TStorage, typename TDataRx, typename TCompute>
 void puschRxChEstKernelBuilder::kernelSelectL1(uint16_t           nBSAnts,
                                   uint8_t                         nLayers,
                                   uint8_t                         nDmrsSyms,
                                   uint8_t                         nDmrsGridsPerPrb,
                                   uint8_t                         enablePerPrgChEstPerUeg,
                                   uint16_t                        nTotalDataPrb,
                                   uint8_t                         Nh,
                                   uint16_t                        nUeGrps,
                                   uint8_t                         enableDftSOfdm,
                                   uint8_t                         chEstAlgo,
                                   uint8_t                         enablePerPrgChEst,
                                   cuphyPuschRxChEstLaunchCfg_t&   launchCfg)
 {
     bool noKernelFound = false;

     // Check below ensures the parameters match the dimensions assumed in the kernel. Among others it ensures
     // that nTotalDataPrb is divisible by N_DMRS_INTERP_PRB_OUT_PER_CLUSTER and divisible by N_DMRS_PRB_IN_PER_CLUSTER

     if((nBSAnts >= nLayers) && (nLayers <= 8) && (2 == nDmrsGridsPerPrb) && (2 == nDmrsSyms) && (1 == Nh))
     {
         constexpr uint32_t N_DMRS_GRIDS_PER_PRB = 2; // 2 grids => 6 grid tones per PRB
         constexpr uint32_t N_DMRS_SYMS          = 2; // # of DMRS symbols

         switch(nLayers)
         {
             case 8:
             case 7:
             case 6:
             case 5:
             {
                constexpr uint32_t N_LAYERS = 8; // # of layers (# of cols in H matrix)
                kernelSelectL0<TStorage,
                               TDataRx,
                               TCompute,
                               N_LAYERS,
                               N_DMRS_GRIDS_PER_PRB,
                               N_DMRS_SYMS>(enablePerPrgChEstPerUeg, nTotalDataPrb, nUeGrps, nBSAnts, enableDftSOfdm, chEstAlgo, enablePerPrgChEst, launchCfg);
                break;
             }

             case 4:
             case 3:
             case 2:
             {
                 constexpr uint32_t N_LAYERS = 4; // # of layers (# of cols in H matrix)
                 kernelSelectL0<TStorage,
                                TDataRx,
                                TCompute,
                                N_LAYERS,
                                N_DMRS_GRIDS_PER_PRB,
                                N_DMRS_SYMS>(enablePerPrgChEstPerUeg, nTotalDataPrb, nUeGrps, nBSAnts, enableDftSOfdm, chEstAlgo, enablePerPrgChEst, launchCfg);
                 break;
             } // nLayers = 2,4

             case 1:
             {
                 constexpr uint32_t N_LAYERS = 1; // # of layers (# of cols in H matrix)
                 kernelSelectL0<TStorage,
                                TDataRx,
                                TCompute,
                                N_LAYERS,
                                N_DMRS_GRIDS_PER_PRB,
                                N_DMRS_SYMS>(enablePerPrgChEstPerUeg, nTotalDataPrb, nUeGrps, nBSAnts, enableDftSOfdm, chEstAlgo, enablePerPrgChEst, launchCfg);
                 break;
             } // nLayers = 1

             default: noKernelFound = true; break;
         } // nLayers
     }
     else if((nBSAnts >= nLayers) && (nLayers <= 4) && (2 == nDmrsGridsPerPrb) && (1 == nDmrsSyms) && (1 == Nh))
     {
         constexpr uint32_t N_DMRS_GRIDS_PER_PRB = 2; // 2 grids => 6 grid tones per PRB
         constexpr uint32_t N_DMRS_SYMS          = 1; // # of DMRS symbols

         switch(nLayers)
         {
             case 4:
             case 3:
             case 2:
             {
                 constexpr uint32_t N_LAYERS = 4; // # of layers (# of cols in H matrix)
                 kernelSelectL0<TStorage,
                                TDataRx,
                                TCompute,
                                N_LAYERS,
                                N_DMRS_GRIDS_PER_PRB,
                                N_DMRS_SYMS>(enablePerPrgChEstPerUeg, nTotalDataPrb, nUeGrps, nBSAnts, enableDftSOfdm, chEstAlgo, enablePerPrgChEst, launchCfg);
                 break;
             } // nLayers = 2,4

             case 1:
             {
                 constexpr uint32_t N_LAYERS = 1; // # of layers (# of cols in H matrix)
                 kernelSelectL0<TStorage,
                                TDataRx,
                                TCompute,
                                N_LAYERS,
                                N_DMRS_GRIDS_PER_PRB,
                                N_DMRS_SYMS>(enablePerPrgChEstPerUeg, nTotalDataPrb, nUeGrps, nBSAnts, enableDftSOfdm, chEstAlgo, enablePerPrgChEst, launchCfg);
                 break;
             } // nLayers = 1

             default: noKernelFound = true; break;
         } // nLayers
     }
     else
     {
         noKernelFound = true;
     }

     if(noKernelFound)
     {
         NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "{}: No kernel available (L1 stage) to launch with requested configuration: nBSAnts {} nLayers {} nDmrsGridsPerPrb {} nDmrsSyms {} Nh {} nTotalDataPrb {}", __FUNCTION__, nBSAnts, nLayers, nDmrsGridsPerPrb, nDmrsSyms, Nh, nTotalDataPrb);
     }
 }

 template <typename TCompute>
 void puschRxChEstKernelBuilder::kernelSelectL2(uint16_t           nBSAnts,
                                   uint8_t                         nLayers,
                                   uint8_t                         nDmrsSyms,
                                   uint8_t                         nDmrsGridsPerPrb,
                                   uint8_t                         enablePerPrgChEstPerUeg,
                                   uint16_t                        nTotalDataPrb,
                                   uint8_t                         Nh,
                                   uint16_t                        nUeGrps,
                                   uint8_t                         enableDftSOfdm,
                                   uint8_t                         chEstAlgo,
                                   uint8_t                         enablePerPrgChEst,
                                   cuphyDataType_t                 dataRxType,
                                   cuphyDataType_t                 hEstType,
                                   cuphyPuschRxChEstLaunchCfg_t&   launchCfg)
 {
    if((CUPHY_C_32F == hEstType) && (CUPHY_C_16F == dataRxType) && (chEstAlgo == PUSCH_CH_EST_ALGO_TYPE_RKHS) && (nBSAnts <= 4))
    {
        rkhsKernelSelectL1(nTotalDataPrb, launchCfg);
        return;
    }

     if(CUPHY_C_32F == hEstType)
     {
         using TStorage = scalar_from_complex<data_type_traits<CUPHY_C_32F>::type>::type;
         if(CUPHY_C_32F == dataRxType)
         {
             using TDataRx = scalar_from_complex<data_type_traits<CUPHY_C_32F>::type>::type;
             kernelSelectL1<TStorage, TDataRx, TCompute>(nBSAnts,
                                                         nLayers,
                                                         nDmrsSyms,
                                                         nDmrsGridsPerPrb,
                                                         enablePerPrgChEstPerUeg,
                                                         nTotalDataPrb,
                                                         Nh,
                                                         nUeGrps,
                                                         enableDftSOfdm,
                                                         chEstAlgo,
                                                         enablePerPrgChEst,
                                                         launchCfg);
         }
         else if(CUPHY_C_16F == dataRxType)
         {
             using TDataRx = scalar_from_complex<data_type_traits<CUPHY_C_16F>::type>::type;
             kernelSelectL1<TStorage, TDataRx, TCompute>(nBSAnts,
                                                         nLayers,
                                                         nDmrsSyms,
                                                         nDmrsGridsPerPrb,
                                                         enablePerPrgChEstPerUeg,
                                                         nTotalDataPrb,
                                                         Nh,
                                                         nUeGrps,
                                                         enableDftSOfdm,
                                                         chEstAlgo,
                                                         enablePerPrgChEst,
                                                         launchCfg);
         }
         else
         {
             NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "{}: No kernel available to launch with requested date type", __FUNCTION__);
         }
     }
     else if((CUPHY_C_16F == hEstType) && (CUPHY_C_16F == dataRxType))
     {
         using TStorage = scalar_from_complex<data_type_traits<CUPHY_C_16F>::type>::type;
         using TDataRx  = scalar_from_complex<data_type_traits<CUPHY_C_16F>::type>::type;
         kernelSelectL1<TStorage, TDataRx, TCompute>(nBSAnts,
                                                     nLayers,
                                                     nDmrsSyms,
                                                     nDmrsGridsPerPrb,
                                                     enablePerPrgChEstPerUeg,
                                                     nTotalDataPrb,
                                                     Nh,
                                                     nUeGrps,
                                                     enableDftSOfdm,
                                                     chEstAlgo,
                                                     enablePerPrgChEst,
                                                     launchCfg);
     }
     else
     {
         NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "{}: No kernel available to launch with requested date type", __FUNCTION__);
     }
 }

 template<typename STATS_OUT>
 void populateStatDescrCpu(STATS_OUT &statDescrCpuOut, cuphyTensorPrm_t const* const tensorIn) {
     const auto copyEntries = [](int *pDst, const int *pSrc, const std::size_t nEntries) {
        std::ignore = std::copy_n(pSrc, nEntries, pDst);
     };
     statDescrCpuOut.pAddr = tensorIn->pAddr;
     // Downcast from empty baseclass to tensor_desc
     const auto& pDesc = static_cast<tensor_desc&>(*tensorIn->desc);
     const tensor_layout_any &layout = pDesc.layout();
     copyEntries(statDescrCpuOut.strides, layout.strides.begin(), layout.rank());
 }

 /**
  * copy ppStatDescrsCpu to ppStatDescrsGpu.
  * Loop over [0..3]
  * for each index [i], loop over m_kernelArgsArr[4][16] I.e. iterate over 16 kernel-args.
  *  - static
  *  - dynamic.
  *  we cache the static descriptor passed in parameter (and is passed after fully populated).
  *  So now kernelArgs.pStatDescr points to this part of the CPU allocated (pinned_alloc) big contigous memory.
  *  Copy from ppStatDescrsCpu[i] to ppStatDescrsGpu[i]
  */
 void puschRxChEstKernelBuilder::init(gsl_lite::span<uint8_t*> ppStatDescrsCpu,
                                      gsl_lite::span<uint8_t*> ppStatDescrsGpu,
                                      const bool enableCpuToGpuDescrAsyncCpy,
                                      cudaStream_t strm){

     for(int32_t chEstTimeInstIdx = 0; chEstTimeInstIdx < CUPHY_PUSCH_RX_MAX_N_TIME_CH_EST; ++chEstTimeInstIdx) {

         for (auto &kernelArgs: m_kernelArgsArr[chEstTimeInstIdx]) {
             kernelArgs.pStatDescr = reinterpret_cast<puschRxChEstStatDescr_t *>(ppStatDescrsGpu[chEstTimeInstIdx]);
         }

         if (enableCpuToGpuDescrAsyncCpy) {
             //Unchecked return value
             CUDA_CHECK(cudaMemcpyAsync(ppStatDescrsGpu[chEstTimeInstIdx], ppStatDescrsCpu[chEstTimeInstIdx],
                                        sizeof(puschRxChEstStatDescr_t), cudaMemcpyHostToDevice, strm));
         }
     }
 }

 /**
  * 7 tensors:
  * - tFreqInterpCoefs tShiftSeq tUnShiftSeq - related to kernel using block of 8 PRBs to estimate the middle 4 PRBs.
  *   It required that the PRB allocation be greater than 8.
  * - Rest of tensor - added flexibility to all PRBs sizes, and reduced compute time by only using up to four
  *   PRBs for ChEst
  *
  * Origin of these tensors:
  * - in PuschRx constructor, cuphyChEstSettings is being instantiated passing it cuphyPuschStatPrms_t
  *   from cuphy_api.h.
  *   It has these 7 cuphyTensorPrm_t, from cuphy.h, which has {descr, addr}.
  *   The descriptor itself, is cuphyTensorDescriptor, empty base class of class tensor_desc (tensor_desc.hpp).
  *   Descriptor has {type, layout(strides, rank), bytes}
  *
  * - Instantiating PuschRx using C-API cuphyCreatePuschRx passing the cuphyPuschStatPrms_t
  *   phypusch_aggr.cpp in cuPHY-CP, createPhyObj(). It has static_params, that are being populated.
  *   tvStatPrms() is populating these 7 objects/tensors, each is cuphyTensorPrm_t type.
  *
  * ppStatDescrsCpu, ppStatDescrsGpu - are chunks from a big contiguous memory. [0..3] in this specific case.
  *  It lives as member in PuschRx.
  * These are part of kernelDescrs (we have 2 - static, dynamic descriptors).
  *  Static params do not change, Dynamic can change every slot.
  *  There is a CPU (pinned_alloc) and GPU (device_alloc) tensors for CPU and GPU respectively.
  *
  * Goal is to take ppStatDescrsCpu[N] and for each index, copy to the ppStatDescrsGpu[] slot.
  * - From each ppStatDescrsCpu[i], we initialize puschRxChEstStatDescr_t and populate it:
  *   - It contains 7 members for each tensor. {addr, strides}
  *   - populateStatDescrCpu - copy from the incoming parameter to the puschRxChEstStatDescr_t.X
  *     where X is one of the 7 tensors. layour.strides are copied to  puschRxChEstStatDescr_t.X.strides.
  *
  *  At the end of the loop we copy ppStatDescrsCpu to ppStatDescrsGpu in the init() function of the builder
  *
  *  The address of the tensor is represented in few locations:
  *  - cuphyTensorPrm_t
  *  - Then it is populated into another struct, puschRxChEstStatDescr_t {7 + other members}, the tensors parts have
  *    addr, dims[] that are copied from layout, coming from the params tensors.
  *    puschRxChEstStatDescr_t will be eventually copied to GPU.
  */
 void puschRxChEst::init(IKernelBuilder*                  pKernelBuilder,
                         const bool                       enableCpuToGpuDescrAsyncCpy,
                         gsl_lite::span<uint8_t*>              ppStatDescrsCpu,
                         gsl_lite::span<uint8_t*>              ppStatDescrsGpu,
                         cudaStream_t                     strm)
 {
     if (!pKernelBuilder) {
         NVLOGF_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "Invalid nullptr pKernelBuilder");
     }

     cuphyTensorPrm_t const*           pFreqInterpCoefs = &m_chEstSettings.WFreq.tPrm;
     cuphyTensorPrm_t const*           pFreqInterpCoefs4 = &m_chEstSettings.WFreq4.tPrm;
     cuphyTensorPrm_t const*           pFreqInterpCoefsSmall = &m_chEstSettings.WFreqSmall.tPrm;
     cuphyTensorPrm_t const*           pShiftSeq = &m_chEstSettings.ShiftSeq.tPrm;
     cuphyTensorPrm_t const*           pShiftSeq4 = &m_chEstSettings.ShiftSeq4.tPrm;
     cuphyTensorPrm_t const*           pUnShiftSeq = &m_chEstSettings.UnShiftSeq.tPrm;
     cuphyTensorPrm_t const*           pUnShiftSeq4 = &m_chEstSettings.UnShiftSeq4.tPrm;
     cuphyPuschRkhsPrms_t const*       pPuschRkhsPrms = m_chEstSettings.pPuschRkhsPrms;

     if(!pFreqInterpCoefs || !pShiftSeq || !pUnShiftSeq)
     {
         // only const_cast to void* works here (CU file?)
         NVLOGF_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "Invalid coef/seq pFreqInterpCoefs {:p} pShiftSeq {:p} pUnShiftSeq {:p}",
                    static_cast<void*>(const_cast<cuphyTensorPrm_t*>(pFreqInterpCoefs)),
                    static_cast<void*>(const_cast<cuphyTensorPrm_t*>(pShiftSeq)),
                    static_cast<void*>(const_cast<cuphyTensorPrm_t*>(pUnShiftSeq)));
     }

     for(auto* cpuDesc : ppStatDescrsCpu)
     {
        auto& statDescrCpu = *reinterpret_cast<puschRxChEstStatDescr_t*>(cpuDesc);
        statDescrCpu.pSymbolRxStatus = m_chEstSettings.pSymbolRxStatus;

        populateStatDescrCpu(statDescrCpu.tPrmFreqInterpCoefs, pFreqInterpCoefs);
        populateStatDescrCpu(statDescrCpu.tPrmFreqInterpCoefs4, pFreqInterpCoefs4);
        populateStatDescrCpu(statDescrCpu.tPrmFreqInterpCoefsSmall, pFreqInterpCoefsSmall);
        populateStatDescrCpu(statDescrCpu.tPrmShiftSeq, pShiftSeq);
        populateStatDescrCpu(statDescrCpu.tPrmShiftSeq4, pShiftSeq4);
        populateStatDescrCpu(statDescrCpu.tPrmUnShiftSeq, pUnShiftSeq);
        populateStatDescrCpu(statDescrCpu.tPrmUnShiftSeq4, pUnShiftSeq4);

        if(m_chEstSettings.chEstAlgo == PUSCH_CH_EST_ALGO_TYPE_RKHS)
        {
            for(int i = 0; i < pPuschRkhsPrms->nPrbSizes; ++i)
            {
                statDescrCpu.prbRkhsDescs[i].zpIdx               = pPuschRkhsPrms->pPerPrbRkhsPrms[i].zpIdx;
                statDescrCpu.prbRkhsDescs[i].sumEigValues        = pPuschRkhsPrms->pPerPrbRkhsPrms[i].sumEigValues;

                copyTensorPrm2Info(pPuschRkhsPrms->pPerPrbRkhsPrms[i].corr_half_nZpDmrsSc , statDescrCpu.prbRkhsDescs[i].tInfoCorr_half_nZpDmrsSc);
                copyTensorPrm2Info(pPuschRkhsPrms->pPerPrbRkhsPrms[i].corr                , statDescrCpu.prbRkhsDescs[i].tInfoCorr);
                copyTensorPrm2Info(pPuschRkhsPrms->pPerPrbRkhsPrms[i].eigVal              , statDescrCpu.prbRkhsDescs[i].tInfoEigVal);
                copyTensorPrm2Info(pPuschRkhsPrms->pPerPrbRkhsPrms[i].interpCob           , statDescrCpu.prbRkhsDescs[i].tInfoInterpCob);
                copyTensorPrm2Info(pPuschRkhsPrms->pPerPrbRkhsPrms[i].eigVecCob           , statDescrCpu.prbRkhsDescs[i].tInfoEigVecCob);
            }

            for(int i = 0; i < pPuschRkhsPrms->nZpSizes; ++i)
            {
                copyTensorPrm2Info(pPuschRkhsPrms->pPerZpRkhsPrms[i].zpDmrsScEigenVec          , statDescrCpu.zpRkhsDescs[i].tInfoZpDmrsScEigenVec);
                copyTensorPrm2Info(pPuschRkhsPrms->pPerZpRkhsPrms[i].zpInterpVec               , statDescrCpu.zpRkhsDescs[i].tInfoZpInterpVec);
                copyTensorPrm2Info(pPuschRkhsPrms->pPerZpRkhsPrms[i].secondStageTwiddleFactors , statDescrCpu.zpRkhsDescs[i].tInfoSecondStageTwiddleFactors);
                copyTensorPrm2Info(pPuschRkhsPrms->pPerZpRkhsPrms[i].secondStageFourierPerm    , statDescrCpu.zpRkhsDescs[i].tInfoSecondStageFourierPerm);
            }
        }
     }
     pKernelBuilder->init(ppStatDescrsCpu, ppStatDescrsGpu, enableCpuToGpuDescrAsyncCpy, strm);
 }

 void puschRxChEst::getDescrInfo(size_t& statDescrSizeBytes, size_t& statDescrAlignBytes, size_t& dynDescrSizeBytes, size_t& dynDescrAlignBytes)
 {
     statDescrSizeBytes  = sizeof(puschRxChEstStatDescr_t);
     statDescrAlignBytes = alignof(puschRxChEstStatDescr_t);

     dynDescrSizeBytes  = sizeof(puschRxChEstDynDescrVec_t);
     dynDescrAlignBytes = alignof(puschRxChEstDynDescrVec_t);
 }

 template <typename TCompute>
 cuphyStatus_t puschRxChEstKernelBuilder::batch(uint32_t                chEstTimeInstIdx,
                                   gsl_lite::span<cuphyPuschRxUeGrpPrms_t>   pDrvdUeGrpPrms,
                                   uint16_t                   nUeGrps,
                                   uint8_t                    enableDftSOfdm,
                                   uint8_t                    chEstAlgo,
                                   uint8_t                    enablePerPrgChEst,
                                   uint32_t&                  nHetCfgs,
                                   puschRxChEstDynDescrVec_t& dynDescrVecCpu)
 {
     // Initialize the batch config data structure
     puschRxChEstHetCfgArr_t& hetCfgs = m_hetCfgsArr[chEstTimeInstIdx];
     hetCfgs.fill({nullptr, 0, 0});

#ifdef DO_NOT_USE_HASH_TABLE
     // Helper to find kernel function
     auto findKernelFunc = [](puschRxChEstHetCfgArr_t const& hetCfgs, CUfunction func, int32_t& hetCfgIdx)
     {
         for(hetCfgIdx = 0; hetCfgIdx < hetCfgs.size(); ++hetCfgIdx)
         {
             // Check if kernel function is found
             if(func == hetCfgs[hetCfgIdx].func) break;

             // Check if no more kernel functions exist
             if(nullptr == hetCfgs[hetCfgIdx].func)
             {
                 hetCfgIdx = -1;
                 break;
             }
         }
         // Exhausted all heterogenous configs possible
         if(hetCfgs.size() == hetCfgIdx) hetCfgIdx = -1;
     };
#else
     m_ChEstHashTable.clear();
#endif

#ifdef ENABLE_DEBUG
     NVLOGI_FMT(NVLOG_PUSCH, "{}: # of UE groups {}", __FUNCTION__, nUeGrps);
#endif

     nHetCfgs = 0;
     for(int32_t ueGrpIdx = 0; ueGrpIdx < nUeGrps; ++ueGrpIdx)
     {
         cuphyPuschRxUeGrpPrms_t const& drvdUeGrpPrms = pDrvdUeGrpPrms[ueGrpIdx];

         // Skip UE group if there aren't enough DMRS additional positions
         // # of time domain channel estimates is equal to the number of DMRS additional positions + 1
         if(chEstTimeInstIdx > drvdUeGrpPrms.dmrsAddlnPos)  continue;

         uint16_t nPrb        = drvdUeGrpPrms.nPrb;
         uint16_t nRxAnt      = drvdUeGrpPrms.nRxAnt;
         auto  symLocBmsk     = drvdUeGrpPrms.dmrsSymLocBmsk;
         int32_t nMinDmrsSyms = std::min(static_cast<int32_t>(drvdUeGrpPrms.nDmrsSyms), __builtin_popcount(symLocBmsk));

#ifdef DO_NOT_USE_HASH_TABLE
         // @todo: extend kernelSelectL2 to support a mode which only determines the kernel function
         cuphyPuschRxChEstLaunchCfg_t launchCfg;
         kernelSelectL2<TCompute>(drvdUeGrpPrms.nRxAnt,
                                  drvdUeGrpPrms.nLayers,
                                  drvdUeGrpPrms.dmrsMaxLen,
                                  drvdUeGrpPrms.nDmrsGridsPerPrb,
                                  drvdUeGrpPrms.enablePerPrgChEst,
                                  nPrb,
                                  1,
                                  nUeGrps,
                                  enableDftSOfdm,
                                  chEstAlgo,
                                  enablePerPrgChEst,
                                  drvdUeGrpPrms.tInfoDataRx.elemType,
                                  drvdUeGrpPrms.tInfoHEst.elemType,
                                  launchCfg);

         // Check if the heterognous configuration already exists
         int32_t hetCfgIdx = 0;
         findKernelFunc(hetCfgs, launchCfg.kernelNodeParamsDriver.func, hetCfgIdx);

         // If a heterogenous configuration already exists then increment the # of UE groups for that config
         if(-1 != hetCfgIdx)
         {
             puschRxChEstHetCfg_t& hetCfg = hetCfgs[hetCfgIdx];
             if(hetCfg.nUeGrps >= MAX_N_USER_GROUPS_SUPPORTED)
             {
                 NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "{}: Exceeded limit on supported UE groups", __FUNCTION__);
                 return CUPHY_STATUS_INTERNAL_ERROR;
             }

             if(nPrb   > hetCfg.nMaxPrb)   hetCfg.nMaxPrb   = nPrb;
             if(nRxAnt > hetCfg.nMaxRxAnt) hetCfg.nMaxRxAnt = nRxAnt;

             dynDescrVecCpu[hetCfgIdx].hetCfgUeGrpMap[hetCfg.nUeGrps] = ueGrpIdx;
             hetCfg.nUeGrps++;

#ifdef ENABLE_DEBUG
             NVLOGI_FMT(NVLOG_PUSCH, "{}: UE group {} -> HetCfg {} funcPtr {:p} (nHetCfgs {} nUeGrps {} nPrb {} nMaxPrb {} nRxAnt {} nLayers {} dmrsAddlnPos {} dmrsMaxLen {} nDmrsGridsPerPrb {})", __FUNCTION__, ueGrpIdx, newHetCfgIdx, static_cast<void*>(hetCfg.func), nHetCfgs, hetCfg.nUeGrps, nPrb, hetCfg.nMaxPrb, drvdUeGrpPrms.nRxAnt, drvdUeGrpPrms.nLayers, drvdUeGrpPrms.dmrsAddlnPos, drvdUeGrpPrms.dmrsMaxLen, drvdUeGrpPrms.nDmrsGridsPerPrb);
#endif
         }
         // New heterogenous configuration found
         else
         {
             const uint32_t nMaxtHetCfgs = (chEstAlgo == PUSCH_CH_EST_ALGO_TYPE_MULTISTAGE_MMSE_WITH_DELAY_EST) ?
                                               CUPHY_PUSCH_RX_CH_EST_MULTISTAGE_MMSE_N_MAX_HET_CFGS : CUPHY_PUSCH_RX_CH_EST_LEGACY_MMSE_N_MAX_HET_CFGS;
             if(nHetCfgs >= nMaxtHetCfgs)
             {
                 NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "{}: Exceeded limit on supported heterogneous configurations", __FUNCTION__);
                 return CUPHY_STATUS_INTERNAL_ERROR;
             }

             int32_t newHetCfgIdx = nHetCfgs++;
             puschRxChEstHetCfg_t& hetCfg = hetCfgs[newHetCfgIdx];
             hetCfg.func = launchCfg.kernelNodeParamsDriver.func;
             hetCfg.nMaxPrb = nPrb;
             hetCfg.nMaxRxAnt = nRxAnt;

             dynDescrVecCpu[newHetCfgIdx].hetCfgUeGrpMap[hetCfg.nUeGrps] = ueGrpIdx;
             hetCfg.nUeGrps++;

#ifdef ENABLE_DEBUG
             NVLOGI_FMT(NVLOG_PUSCH, "{}: UE group {} -> HetCfg {} funcPtr {:p} (nHetCfgs {} nUeGrps {} nPrb {} nMaxPrb {} nRxAnt {} nLayers {} dmrsAddlnPos {} dmrsMaxLen {} nDmrsGridsPerPrb {})", __FUNCTION__, ueGrpIdx, newHetCfgIdx, static_cast<void*>(hetCfg.func), nHetCfgs, hetCfg.nUeGrps, nPrb, hetCfg.nMaxPrb, drvdUeGrpPrms.nRxAnt, drvdUeGrpPrms.nLayers, drvdUeGrpPrms.dmrsAddlnPos, drvdUeGrpPrms.dmrsMaxLen, drvdUeGrpPrms.nDmrsGridsPerPrb);
#endif
         }
#else // using hash table
         // FixMe: it is assumed all UE groups use the same element type for tInfoHEst and tInfoDataRx,
         // hence hash table is build only based on nRxAnt, nLayers, dmrsMaxLen, nDmrsGridsPerPrb and nPrb
         // also note nRxAnt is not strictly needed
         bool newHetCfgFound = false;
         cuphyPuschRxChEstLaunchCfg_t launchCfg;
         auto hashKey = std::make_tuple(drvdUeGrpPrms.nRxAnt, drvdUeGrpPrms.nLayers, drvdUeGrpPrms.dmrsMaxLen, drvdUeGrpPrms.nDmrsGridsPerPrb, drvdUeGrpPrms.enablePerPrgChEst, nPrb);
         auto hashItr = m_ChEstHashTable.find(hashKey);
         if (hashItr == m_ChEstHashTable.end() )
         {
             // key not found in the existing table
             // @todo: extend kernelSelectL2 to support a mode which only determines the kernel function
             kernelSelectL2<TCompute>(drvdUeGrpPrms.nRxAnt,
                                      drvdUeGrpPrms.nLayers,
                                      drvdUeGrpPrms.dmrsMaxLen,
                                      drvdUeGrpPrms.nDmrsGridsPerPrb,
                                      drvdUeGrpPrms.enablePerPrgChEst,
                                      nPrb,
                                      1,
                                      nUeGrps,
                                      enableDftSOfdm,
                                      chEstAlgo,
                                      enablePerPrgChEst,
                                      drvdUeGrpPrms.tInfoDataRx.elemType,
                                      drvdUeGrpPrms.tInfoHEst.elemType,
                                      launchCfg);
             newHetCfgFound = true;
             {
                 //FixMe a custom allocator could be used to allow preallocation to avoid dyn mem alloc
                 MemtraceDisableScope md;
                 // check to ensure the function pointer is indeed not available in the hash table
                 for(auto it = m_ChEstHashTable.begin(); it != m_ChEstHashTable.end(); it++)
                 {
                     if(launchCfg.kernelNodeParamsDriver.func == it->second.func)
                     {
                         newHetCfgFound            = false;      //despite a new key combination, this het config has been already registered
                         hashItr                   = it;         //update hashItr to refer to the existing
                         m_ChEstHashTable[hashKey] = it->second; //add the new key combination to the table
                         break;
                     }
                 }
             }
         }

         if (newHetCfgFound)
         {
             const uint32_t nMaxtHetCfgs = (chEstAlgo == PUSCH_CH_EST_ALGO_TYPE_MULTISTAGE_MMSE_WITH_DELAY_EST) ?
                                               CUPHY_PUSCH_RX_CH_EST_MULTISTAGE_MMSE_N_MAX_HET_CFGS : CUPHY_PUSCH_RX_CH_EST_LEGACY_MMSE_N_MAX_HET_CFGS;
             if(nHetCfgs >= nMaxtHetCfgs)
             {
                 NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "{}: Exceeded limit on supported heterogneous configurations", __FUNCTION__);
                 return CUPHY_STATUS_INTERNAL_ERROR;
             }

             int32_t newHetCfgIdx = nHetCfgs++;
             puschRxChEstHetCfg_t& hetCfg = hetCfgs[newHetCfgIdx];
             hetCfg.func = launchCfg.kernelNodeParamsDriver.func;
             hetCfg.nMaxPrb = nPrb;
             hetCfg.nMaxRxAnt = nRxAnt;

             dynDescrVecCpu[newHetCfgIdx].hetCfgUeGrpMap[hetCfg.nUeGrps] = ueGrpIdx;
             hetCfg.nUeGrps++;

             // update the hash table
             {
                 //FixMe a custom allocator could be used to allow preallocation to avoid dyn mem alloc
                 MemtraceDisableScope md;
                 m_ChEstHashTable[hashKey] = chEstHashVal(hetCfg.func, newHetCfgIdx);
             }
         }
         else
         {
             // If a heterogenous configuration already exists then increment the # of UE groups for that config
             int32_t hetCfgIdx = hashItr->second.hetCfgIdx;
             puschRxChEstHetCfg_t& hetCfg = hetCfgs[hetCfgIdx];
             if(hetCfg.nUeGrps >= MAX_N_USER_GROUPS_SUPPORTED)
             {
                 NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "{}: Exceeded limit on supported UE groups", __FUNCTION__);
                 return CUPHY_STATUS_INTERNAL_ERROR;
             }

             if(nPrb   > hetCfg.nMaxPrb)   hetCfg.nMaxPrb   = nPrb;
             if(nRxAnt > hetCfg.nMaxRxAnt) hetCfg.nMaxRxAnt = nRxAnt;

             dynDescrVecCpu[hetCfgIdx].hetCfgUeGrpMap[hetCfg.nUeGrps] = ueGrpIdx;
             hetCfg.nUeGrps++;
         }

#endif //DO_NOT_USE_HASH_TABLE
     }
     return CUPHY_STATUS_SUCCESS;
 }

 /**
  * Invoked from PuschRx setupComponents. Passing:
  * - Dynamic params type pf CPU, GPU KernelDescriptors. These are the ppDynDescrsCpu, ppDynDescrsGpu
  * - pDrvdUeGrpPrmsCpu, pDrvdUeGrpPrmsGpu - pointing to a big stuct type (many attributes) named
  *   cuphyPuschRxUeGrpPrms_t. Both are members of PuschRx class type.
  * Both are pointing to
  *  m_drvdUeGrpPrmsCpu = (cuphyPuschRxUeGrpPrms_t*)(dynCpuDescrStartAddrs[PUSCH_FRONT_END_PARAMS]);
  *  m_drvdUeGrpPrmsGpu = (cuphyPuschRxUeGrpPrms_t*)(dynGpuDescrStartAddrs[PUSCH_FRONT_END_PARAMS]);
  *  coming from m_kernelDynDescr.getCpuStartAddrs and m_kernelDynDescr.getGpuStartAddrs.
  */
 cuphyStatus_t
 puschRxChEstKernelBuilder::build(gsl_lite::span<cuphyPuschRxUeGrpPrms_t>         pDrvdUeGrpPrmsCpu,
                                  gsl_lite::span<cuphyPuschRxUeGrpPrms_t>         pDrvdUeGrpPrmsGpu,
                                  uint16_t                                   nUeGrps,
                                  uint8_t                                    maxDmrsMaxLen,
                                  uint8_t                                    enableDftSOfdm,
                                  uint8_t                                    chEstAlgo,
                                  uint8_t                                    enableMassiveMIMO,
                                  uint8_t                                    enablePerPrgChEst,
                                  uint8_t*                                   pPreEarlyHarqWaitKernelStatusGpu,
                                  uint8_t*                                   pPostEarlyHarqWaitKernelStatusGpu,
                                  const uint16_t                             waitTimeOutPreEarlyHarqUs,
                                  const uint16_t                             waitTimeOutPostEarlyHarqUs,
                                  bool                                       enableCpuToGpuDescrAsyncCpy,
                                  gsl_lite::span<uint8_t*>                        ppDynDescrsCpu,
                                  gsl_lite::span<uint8_t*>                        ppDynDescrsGpu,
                                  pusch::IStartKernels*                      pStartKernels,
                                  gsl_lite::span<cuphyPuschRxChEstLaunchCfgs_t>   launchCfgs,
                                  uint8_t                               enableEarlyHarqProc,
                                  uint8_t                               enableFrontLoadedDmrsProc,
                                  uint8_t                               enableDeviceGraphLaunch,
                                  CUgraphExec*                          pSubSlotDeviceGraphExec,
                                  CUgraphExec*                          pFullSlotDeviceGraphExec,
                                  cuphyPuschRxWaitLaunchCfg_t*          pWaitKernelLaunchCfgsPreSubSlot,
                                  cuphyPuschRxWaitLaunchCfg_t*          pWaitKernelLaunchCfgsPostSubSlot,
                                  cuphyPuschRxDglLaunchCfg_t*           pDglKernelLaunchCfgsPreSubSlot,
                                  cuphyPuschRxDglLaunchCfg_t*           pDglKernelLaunchCfgsPostSubSlot,
                                  cudaStream_t                          strm)

 {
     if (!launchCfgs.data() ||
         launchCfgs.empty()) {
         NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "Unexpected null/empty kernel launch configs");
         return CUPHY_STATUS_INVALID_ARGUMENT;
     }

     if (pDrvdUeGrpPrmsCpu.empty() ||
         !pDrvdUeGrpPrmsCpu.data() ||
         pDrvdUeGrpPrmsGpu.empty() ||
         !pDrvdUeGrpPrmsGpu.data()) {
         NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "Unexpected null/empty UeGrpPrms CPU/GPU");
         return CUPHY_STATUS_INVALID_ARGUMENT;
     }

     if (ppDynDescrsCpu.empty() ||
         !ppDynDescrsCpu.data() ||
         ppDynDescrsGpu.empty() ||
         !ppDynDescrsGpu.data()) {
         NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "Unexpected null/empty Dynamic Descriptor CPU/GPU");
         return CUPHY_STATUS_INVALID_ARGUMENT;
     }

     if((chEstAlgo != PUSCH_CH_EST_ALGO_TYPE_LEGACY_MMSE) &&
       (chEstAlgo != PUSCH_CH_EST_ALGO_TYPE_MULTISTAGE_MMSE_WITH_DELAY_EST) &&
       (chEstAlgo != PUSCH_CH_EST_ALGO_TYPE_RKHS) &&
       (chEstAlgo != PUSCH_CH_EST_ALGO_TYPE_LS_ONLY)) {
        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "Unexpected channel estimate algorithem value={}", chEstAlgo);
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    using TCompute = float;
    // Right now, we have restriction - we cannot have ppDynDescrsCpu.size() > CUPHY_PUSCH_RX_MAX_N_TIME_CH_EST
    if (ppDynDescrsCpu.size() > CUPHY_PUSCH_RX_MAX_N_TIME_CH_EST) {
        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT,
                   "Unexpected size of ppDynDescrsCpu {} cannot be > CUPHY_PUSCH_RX_MAX_N_TIME_CH_EST (={})",
                   ppDynDescrsCpu.size(), CUPHY_PUSCH_RX_MAX_N_TIME_CH_EST);
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    for(int32_t chEstTimeInstIdx = 0; chEstTimeInstIdx < ppDynDescrsCpu.size(); ++chEstTimeInstIdx)
    {
        if(!ppDynDescrsCpu[chEstTimeInstIdx] || !ppDynDescrsGpu[chEstTimeInstIdx]) {
            NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT,
                       "chEstTimeInstIdx={}, ppDynDescrsCpu or ppDynDescrsGpu is unexpectedly null", chEstTimeInstIdx);
            return CUPHY_STATUS_INVALID_ARGUMENT;
        }

        puschRxChEstDynDescrVec_t& dynDescrVecCpu = *(reinterpret_cast<ch_est::puschRxChEstDynDescrVec_t*>(ppDynDescrsCpu[chEstTimeInstIdx]));
        cuphyPuschRxChEstLaunchCfgs_t& lCfgs = launchCfgs[chEstTimeInstIdx];
        cuphyStatus_t status = batch<TCompute>(chEstTimeInstIdx,
                                     pDrvdUeGrpPrmsCpu,
                                     nUeGrps,
                                     enableDftSOfdm,
                                     chEstAlgo,
                                     enablePerPrgChEst,
                                     lCfgs.nCfgs,
                                     dynDescrVecCpu);

        if(CUPHY_STATUS_SUCCESS != status)
        {
            return status;
        }

        uint8_t nMaxSubSlotPuschSym = 0;
        if(chEstTimeInstIdx == 0)
        {
            if(enableEarlyHarqProc)
            {
                nMaxSubSlotPuschSym = EARLY_HARQ_SYM_IDX_UB + ((enableMassiveMIMO && (maxDmrsMaxLen==2)) ? 1 : 0);
            }
            else if(enableFrontLoadedDmrsProc)
            {
                for(uint32_t ueGrpsIdx = 0; ueGrpsIdx < nUeGrps; ++ueGrpsIdx)
                {
                    uint8_t frontLoadedDmrsUpperBound = pDrvdUeGrpPrmsCpu[ueGrpsIdx].frontLoadedDmrsUpperBound;
                    if(frontLoadedDmrsUpperBound > nMaxSubSlotPuschSym)
                    {
                        nMaxSubSlotPuschSym = frontLoadedDmrsUpperBound;
                    }
                }
            }
        }

        uint8_t nMaxFullSlotPuschSym = 0;
        if((chEstTimeInstIdx == 0) && ((enableEarlyHarqProc) || (enableFrontLoadedDmrsProc)))
        {
            for(uint32_t ueGrpsIdx = 0; ueGrpsIdx < nUeGrps; ++ueGrpsIdx)
            {
                uint8_t nPuschSym = pDrvdUeGrpPrmsCpu[ueGrpsIdx].nPuschSym;
                if(nPuschSym > nMaxFullSlotPuschSym)
                {
                    nMaxFullSlotPuschSym = nPuschSym;
                }
            }
            nMaxFullSlotPuschSym = min(nMaxFullSlotPuschSym, OFDM_SYMBOLS_PER_SLOT);
        }

        puschRxChEstDynDescr_t* pDynDescrVecGpu = reinterpret_cast<puschRxChEstDynDescr_t*>(ppDynDescrsGpu[chEstTimeInstIdx]);
        for(uint32_t hetCfgIdx = 0; hetCfgIdx < lCfgs.nCfgs; ++hetCfgIdx)
        {
            // Skip rest of the setup if there are no UE groups corresponding to the channel estimation time instance and hetCfg
            if(0 == m_hetCfgsArr[chEstTimeInstIdx][hetCfgIdx].nUeGrps) continue;

            // Setup descriptor in CPU memory
            puschRxChEstDynDescr_t& dynDescr   = dynDescrVecCpu[hetCfgIdx];
            puschRxChEstHetCfg_t const& hetCfg = m_hetCfgsArr[chEstTimeInstIdx][hetCfgIdx];
            dynDescr.chEstTimeInst             = chEstTimeInstIdx;
            dynDescr.pDrvdUeGrpPrms            = pDrvdUeGrpPrmsGpu.data();
            dynDescr.mPuschStartTimeNs         = 0; // always set to 0, this will be over-written in preEarlyHarqWaitKernel()
            uint16_t nZpDmrsSc;
            uint16_t nTotComputeBlocks = 0;

            // rkhs descriptor
            if(chEstAlgo == PUSCH_CH_EST_ALGO_TYPE_RKHS)
            {
                CUDA_CHECK(cudaMemcpyToSymbolAsync(d_twiddle32, twiddle32, sizeof(twiddle32), 0, cudaMemcpyHostToDevice, strm));
                CUDA_CHECK(cudaMemcpyToSymbolAsync(d_fourier32PermuteIdx, fourier32PermuteIdx, sizeof(fourier32PermuteIdx), 0, cudaMemcpyHostToDevice, strm));
                CUDA_CHECK(cudaMemcpyToSymbolAsync(d_fourier8PermuteIdx, fourier8PermuteIdx, sizeof(fourier8PermuteIdx), 0, cudaMemcpyHostToDevice, strm));

                for(int i = 0; i < m_hetCfgsArr[chEstTimeInstIdx][hetCfgIdx].nUeGrps; ++i)
                {
                    rkhsUeGrpPrms_t&         rkhsUeGrpPrm  = dynDescr.rkhsUeGrpPrms[i];
                    uint16_t                 ueGrpIdx      = dynDescr.hetCfgUeGrpMap[i];
                    cuphyPuschRxUeGrpPrms_t& drvdUeGrpPrms = pDrvdUeGrpPrmsCpu[ueGrpIdx];
                 //   rkhsUeGrpPrm.nPrb = 64;

                    // intialize grid/tocc/focc bitmasks to zero:
                    rkhsUeGrpPrm.gridBitMask = 0;
                    for(int gridIdx = 0; gridIdx < 2; ++gridIdx)
                    {
                        gridPrm_t& gridPrm  = rkhsUeGrpPrm.gridPrms[gridIdx];
                        gridPrm.toccBitMask = 0;

                        for(int toccIdx = 0; toccIdx < 2; ++toccIdx)
                        {
                            toccPrm_t& toccPrm  = gridPrm.toccPrms[toccIdx];
                            toccPrm.foccBitMask = 0;
                        }
                    }

                    // populate grid/tocc/focc paramaters:
                    for(int layerIdx = 0; layerIdx < drvdUeGrpPrms.nLayers; layerIdx++)
                    {
                        uint8_t portIdx = drvdUeGrpPrms.dmrsPortIdxs[layerIdx];
                        uint8_t foccIdx = portIdx & 1;
                        uint8_t gridIdx = (portIdx >> 1) & 1;
                        uint8_t toccIdx = (portIdx >> 2) & 1;

                        rkhsUeGrpPrm.gridBitMask        = rkhsUeGrpPrm.gridBitMask | (gridIdx + 1);
                        rkhsUeGrpPrm.gridIdxs[layerIdx] = gridIdx;

                        gridPrm_t& gridPrm  = rkhsUeGrpPrm.gridPrms[gridIdx];
                        gridPrm.toccBitMask = gridPrm.toccBitMask | (toccIdx + 1);

                        toccPrm_t& toccPrm  = gridPrm.toccPrms[toccIdx];
                        toccPrm.foccBitMask = toccPrm.foccBitMask | (foccIdx + 1);

                        foccPrm_t& foccPrm = toccPrm.foccPrms[foccIdx];
                        foccPrm.layerIdx   = layerIdx;
                    }

                    // compute the number of zpDmrsSc:
                    uint16_t nDmrsSc        = drvdUeGrpPrms.nPrb * 6;
                    uint16_t log2numZpDmrsSc = 9;

                    for(int i = 3; i < 10; ++i)
                    {
                        if(nDmrsSc <= (1 << i))
                        {
                            log2numZpDmrsSc = i;
                            break;
                        }
                    }
                    if(drvdUeGrpPrms.nPrb < 4)
                    {
                        log2numZpDmrsSc += 2;
                    }else
                    {
                        log2numZpDmrsSc += 1;
                    }
                    nZpDmrsSc = (1 << log2numZpDmrsSc);

                    // compute number of intervals covering the cylic-prefix:
                    uint16_t nCpInt = static_cast<uint16_t>(static_cast<float>(nZpDmrsSc) * 0.1386);

                    // number of prbs per compute block:
                    uint16_t nPrbsPerComputeBlock = drvdUeGrpPrms.nPrb;
                    if(nPrbsPerComputeBlock > 64)
                    {
                        nPrbsPerComputeBlock = 64;
                    }

                    // common compute block paramaters:
                    computeBlocksCommonPrms_t& computeBlocksCommonPrms = rkhsUeGrpPrm.computeBlocksCommonPrms;
                    computeBlocksCommonPrms.nPrb = nPrbsPerComputeBlock;

                    // zero padding index:
                    computeBlocksCommonPrms.zpIdx = log2numZpDmrsSc - 5;

                    // determine method for noise estimation:
                    if((rkhsUeGrpPrm.gridBitMask == 1) && (drvdUeGrpPrms.nDmrsCdmGrpsNoData == 2)) //if empty DMRS grid avaliable, use it to measure noise
                    {
                        computeBlocksCommonPrms.noiseEstMethod    = USE_EMPTY_DMRS_GRID;
                        computeBlocksCommonPrms.nNoiseMeasurments = static_cast<float>(nPrbsPerComputeBlock * 6 * drvdUeGrpPrms.nRxAnt);
                        computeBlocksCommonPrms.nNoiseIntsPerFocc = 0;
                        computeBlocksCommonPrms.nNoiseIntsPerGrid = 0;

                    }else if(drvdUeGrpPrms.nLayers == 1) // else, if empty fOCC present, use it to measure noise
                    {
                        computeBlocksCommonPrms.noiseEstMethod  = USE_EMPTY_FOCC;
                        uint16_t rolloff                       = max(nZpDmrsSc / 4, 10);

                        computeBlocksCommonPrms.nNoiseIntsPerFocc      = nZpDmrsSc / 2 - rolloff;
                        computeBlocksCommonPrms.nNoiseIntsPerGrid      = computeBlocksCommonPrms.nNoiseIntsPerFocc;
                        computeBlocksCommonPrms.nNoiseMeasurments      = computeBlocksCommonPrms.nNoiseIntsPerFocc * drvdUeGrpPrms.nRxAnt;
                        computeBlocksCommonPrms.noiseRegionFirstIntIdx = 0;
                    }
                    else // if no empty fOCC or grid, use quite regions of fOCC to measure noise
                    {
                        computeBlocksCommonPrms.noiseEstMethod = USE_QUITE_FOCC_REGIONS;
                        uint16_t rolloff                      = max((nZpDmrsSc / 2 - nCpInt) / 4, 3);

                        computeBlocksCommonPrms.nNoiseIntsPerFocc      = nZpDmrsSc / 2 - nCpInt - 2 * rolloff;
                        computeBlocksCommonPrms.nNoiseIntsPerGrid      = 2 * computeBlocksCommonPrms.nNoiseIntsPerFocc;
                        computeBlocksCommonPrms.noiseRegionFirstIntIdx = rolloff + nCpInt;
                        computeBlocksCommonPrms.nNoiseMeasurments      = computeBlocksCommonPrms.nNoiseIntsPerFocc * drvdUeGrpPrms.nRxAnt * drvdUeGrpPrms.nLayers;
                    }

                    // compute block paramaters:
                    uint16_t nComputeBlocks = (drvdUeGrpPrms.nPrb + nPrbsPerComputeBlock - 1) / nPrbsPerComputeBlock;

                    for(int compBlockIdx = 0; compBlockIdx < nComputeBlocks; ++compBlockIdx)
                    {
                        rkhsComputeBlockPrms_t& computeBlockPrm = dynDescr.rkhsCompBlockPrms[nTotComputeBlocks];

                        computeBlockPrm.ueGrpIdx              = ueGrpIdx;
                        computeBlockPrm.startInputPrb         = drvdUeGrpPrms.startPrb + compBlockIdx * nPrbsPerComputeBlock;
                        computeBlockPrm.startOutputScInBlock  = 0;
                        computeBlockPrm.scOffsetIntoChEstBuff = 12 * nPrbsPerComputeBlock * compBlockIdx;
                        computeBlockPrm.nOutputSc             = 12 * nPrbsPerComputeBlock;

                        nTotComputeBlocks += 1;
                    }

                    if(nComputeBlocks > 1)
                    {
                        uint16_t nEdgePrbs = (drvdUeGrpPrms.startPrb + drvdUeGrpPrms.nPrb) - (dynDescr.rkhsCompBlockPrms[nTotComputeBlocks - 2].startInputPrb + nPrbsPerComputeBlock);
                        rkhsComputeBlockPrms_t& computeBlockPrm = dynDescr.rkhsCompBlockPrms[nTotComputeBlocks - 1];

                        computeBlockPrm.startInputPrb         = drvdUeGrpPrms.startPrb + drvdUeGrpPrms.nPrb - nPrbsPerComputeBlock;
                        computeBlockPrm.startOutputScInBlock  = 12 * (nPrbsPerComputeBlock - nEdgePrbs);
                        computeBlockPrm.scOffsetIntoChEstBuff = 12 * (drvdUeGrpPrms.nPrb - nEdgePrbs);
                        computeBlockPrm.nOutputSc             = 12 * nEdgePrbs;
                    }
                }
            }

            dynDescr.pPreSubSlotWaitKernelStatusGpu    = pPreEarlyHarqWaitKernelStatusGpu;
            dynDescr.pPostSubSlotWaitKernelStatusGpu   = pPostEarlyHarqWaitKernelStatusGpu;
            dynDescr.waitTimeOutPreEarlyHarqUs = waitTimeOutPreEarlyHarqUs;
            dynDescr.waitTimeOutPostEarlyHarqUs= waitTimeOutPostEarlyHarqUs;

            dynDescr.nSymPreSubSlotWaitKernel  = nMaxSubSlotPuschSym;
            dynDescr.nSymPostSubSlotWaitKernel = nMaxFullSlotPuschSym;

 #ifdef ENABLE_DEBUG
            NVLOGI_FMT(NVLOG_PUSCH, "{}: startPrb {} dmrsScId {}", __FUNCTION__, pDrvdUeGrpPrmsCpu[hetCfgIdx].startPrb, pDrvdUeGrpPrmsCpu[hetCfgIdx].dmrsScrmId);
 #endif // ENABLE_DEBUG

            puschRxChEstKernelArgs_t& kernelArgs = m_kernelArgsArr[chEstTimeInstIdx][hetCfgIdx];
            kernelArgs.pDynDescr = &pDynDescrVecGpu[hetCfgIdx];

            // Optional descriptor copy to GPU memory
            if(enableCpuToGpuDescrAsyncCpy)
            {
                CUDA_CHECK(cudaMemcpyAsync(&pDynDescrVecGpu[hetCfgIdx], &dynDescr, sizeof(puschRxChEstDynDescr_t), cudaMemcpyHostToDevice, strm));
            }

            // Select kernel
            cuphyPuschRxChEstLaunchCfg_t& launchCfg = lCfgs.cfgs[hetCfgIdx];

            // TODO: Optimize function to determine kernel selection and launch geometry separately
            // TODO: for supporting per UE group layer and antenna count. Also per UE group DMRS config (nDmrsGridsPerPrb)
            int32_t ueGrpIdx = dynDescr.hetCfgUeGrpMap[0];
            cuphyPuschRxUeGrpPrms_t const& drvdUeGrpPrms = pDrvdUeGrpPrmsCpu[ueGrpIdx];
            kernelSelectL2<TCompute>(hetCfg.nMaxRxAnt,
                           drvdUeGrpPrms.nLayers,
                           drvdUeGrpPrms.dmrsMaxLen,
                           drvdUeGrpPrms.nDmrsGridsPerPrb,
                           drvdUeGrpPrms.enablePerPrgChEst,
                           hetCfg.nMaxPrb,
                           1,
                           hetCfg.nUeGrps,
                           enableDftSOfdm,
                           chEstAlgo,
                           enablePerPrgChEst,
                           drvdUeGrpPrms.tInfoDataRx.elemType,
                           drvdUeGrpPrms.tInfoHEst.elemType,
                           launchCfg);

            if(hetCfg.func != launchCfg.kernelNodeParamsDriver.func)
            {
               NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "{}: HetCfg {} (nUeGrps {} nMaxPrb {} nMaxRxAnt {} nLayers {} dmrsAddlnPos {} dmrsMaxLen {} nDmrsGridsPerPrb {})", __FUNCTION__, hetCfgIdx, hetCfg.nUeGrps, hetCfg.nMaxPrb, hetCfg.nMaxRxAnt, drvdUeGrpPrms.nLayers, drvdUeGrpPrms.dmrsAddlnPos, drvdUeGrpPrms.dmrsMaxLen, drvdUeGrpPrms.nDmrsGridsPerPrb);
               return CUPHY_STATUS_INTERNAL_ERROR;
            }

            // For the updated ChEst algorithm, we handle all of the PRB counts in a single dispatch kernel,
            // but that means that the logic for tracking the highest PRB count over all UE groups to
            // determine the launch configuration does not work. For example, if we have groups with 30
            // and 40 PRBs, then the group with 30 PRBs needs 15 clusters whereas the group with 40 only
            // needs 10 because the former handles two PRBs per cluster and the latter handles 4. Thus,
            // we loop over all UE groups to determine the maximum number of clusters of threads needed
            // across all groups and update the launch configuration accordingly.
            if(chEstAlgo == PUSCH_CH_EST_ALGO_TYPE_MULTISTAGE_MMSE_WITH_DELAY_EST || chEstAlgo==PUSCH_CH_EST_ALGO_TYPE_LS_ONLY)
            {
                uint16_t nMaxPrbClusters = 0, nMaxThreads = 0, nMaxRxAnt = 0, nMaxPrb = 0, nMaxLayers = 0;
                for (uint32_t k = 0; k < hetCfg.nUeGrps; k++) {
                    cuphyPuschRxUeGrpPrms_t & drvdUeGrpPrms = pDrvdUeGrpPrmsCpu[dynDescr.hetCfgUeGrpMap[k]];
                    const uint16_t nPrb = drvdUeGrpPrms.nPrb;
                    const uint16_t nLayers = drvdUeGrpPrms.nLayers;
                    if((enablePerPrgChEst==1)&&(drvdUeGrpPrms.enablePerPrgChEst==1))
                    {
                        uint16_t prgSize = drvdUeGrpPrms.prgSize;
                        
                        if((prgSize!=1)&&(prgSize!=2)&&(prgSize!=3)&&(prgSize!=4))
                        {
                            NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "Invalid prgSize {}", prgSize);
                            return CUPHY_STATUS_INVALID_ARGUMENT;
                        }
                        
                        if(prgSize > nPrb)
                        {
                            prgSize = nPrb;
                        }
                        
                        uint16_t nPrbClusters = 0;
                        uint16_t nThreads = 0;
                        if(prgSize < 4)
                        {
                            nPrbClusters = div_round_up(nPrb, static_cast<uint16_t>(prgSize));
                            nThreads = prgSize * N_TONES_PER_PRB * nLayers;
                        }
                        else if(prgSize == 4)
                        {
                            uint16_t quotient = (nPrb>>2);
                            uint16_t remainder = (nPrb & 0x3);
                            nPrbClusters = 2 * quotient;
                            if(remainder>0)
                            {
                                nPrbClusters += 1;  
                            }
                            
                            if(remainder == 3)
                            {
                                nThreads = 3 * N_TONES_PER_PRB * nLayers;
                            }
                            else
                            {
                                nThreads = 2 * N_TONES_PER_PRB * nLayers;
                            }
                        }
                        
                        if (nPrbClusters > nMaxPrbClusters) {
                            nMaxPrbClusters = nPrbClusters;
                        }
                        if (nThreads > nMaxThreads) {
                            nMaxThreads = nThreads;
                        }
                        
                        //printf("nPrbClusters=%d, nThreads=%d\n", nPrbClusters, nThreads);
                    }
                    else
                    {
                        uint16_t nPrbOutPerCluster;
                        if (nPrb > 7 && (nPrb % 4 == 0)) {
                            nPrbOutPerCluster = 4;
                        } else if (nPrb > 3) {
                            nPrbOutPerCluster = 2;
                        } else {
                            nPrbOutPerCluster = nPrb;
                        }
                        const uint16_t nPrbClusters = div_round_up(nPrb, static_cast<uint16_t>(nPrbOutPerCluster));
                        const uint16_t nThreads = nPrbOutPerCluster * N_TONES_PER_PRB * nLayers;
                        if (nPrbClusters > nMaxPrbClusters) {
                            nMaxPrbClusters = nPrbClusters;
                        }
                        if (nThreads > nMaxThreads) {
                            nMaxThreads = nThreads;
                        }
                    }
                    
                    if (nPrb > nMaxPrb) {
                        nMaxPrb = nPrb;
                    }
                    if (nLayers > nMaxLayers) {
                        nMaxLayers = nLayers;
                    }
                    if (drvdUeGrpPrms.nRxAnt > nMaxRxAnt) {
                        nMaxRxAnt = drvdUeGrpPrms.nRxAnt;
                    }
                }

                uint16_t CUPHY_PUSCH_RX_CH_EST_DELAY_EST_PRB_CLUSTER_SIZE = ((enablePerPrgChEst==1)&&(drvdUeGrpPrms.enablePerPrgChEst==1)) ? CUPHY_PUSCH_RX_CH_EST_PRG_DELAY_EST_PRB_CLUSTER_SIZE : CUPHY_PUSCH_RX_CH_EST_NON_PRG_DELAY_EST_PRB_CLUSTER_SIZE;
                CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriver = launchCfg.kernelNodeParamsDriver;
                kernelNodeParamsDriver.gridDimX = div_round_up(static_cast<int>(nMaxPrb), static_cast<int>(CUPHY_PUSCH_RX_CH_EST_DELAY_EST_PRB_CLUSTER_SIZE));
                kernelNodeParamsDriver.gridDimY = nMaxRxAnt;
                kernelNodeParamsDriver.gridDimZ = hetCfg.nUeGrps;
                // The PRB cluster size is fixed for the first kernel, so we always use the
                // same number of threads, at least for DMRS type 1 grids, which is all that
                // we currently support.
                kernelNodeParamsDriver.blockDimX = N_TONES_PER_PRB *
                    CUPHY_PUSCH_RX_CH_EST_DELAY_EST_PRB_CLUSTER_SIZE;
                kernelNodeParamsDriver.blockDimY = 1;
                kernelNodeParamsDriver.blockDimZ = 1;

                if(chEstAlgo == PUSCH_CH_EST_ALGO_TYPE_MULTISTAGE_MMSE_WITH_DELAY_EST) {

                    CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriverSecond = launchCfg.kernelNodeParamsDriverSecond;
                    kernelNodeParamsDriverSecond.gridDimX = nMaxPrbClusters;
                    kernelNodeParamsDriverSecond.gridDimZ = hetCfg.nUeGrps;
                    const uint32_t gridDimXZ = kernelNodeParamsDriverSecond.gridDimX * kernelNodeParamsDriverSecond.gridDimZ;
                    // For the second kernel, we want to batch by antennas in the case of large
                    // grids and small CTAs. This is particularly important on H100 where high execution
                    // times have been noted for large grid, small CTA cases of this kernel. The
                    // definition of a "large" grid is currently somewhat arbitrarily set to 256,
                    // so we batch if we will still have at least 256 blocks left after batching.
                    const uint32_t LARGE_GRID_SIZE = 256;
                    const float targetBlockSz = (static_cast<float>(nMaxRxAnt) * gridDimXZ) / LARGE_GRID_SIZE;
                    // We have only tested batching by up to 4 antennas, so for now we only support 4, 2, and 1
                    // as block sizes. We also handle layers in the x block dimension, so high layer counts yield
                    // larger CTAs. We only batch antennas for relatively small CTAs.
                    const uint32_t N_MAX_RX_ANT_PER_BLOCK = 4;
                    uint16_t nRxAntPerBlock = 1;
                    if (targetBlockSz >= N_MAX_RX_ANT_PER_BLOCK && nMaxThreads < 128) {
                        nRxAntPerBlock = N_MAX_RX_ANT_PER_BLOCK;
                    } else if (targetBlockSz >= N_MAX_RX_ANT_PER_BLOCK/2 && nMaxThreads < 256) {
                        nRxAntPerBlock = N_MAX_RX_ANT_PER_BLOCK/2;
                    }
                    kernelNodeParamsDriverSecond.gridDimY = div_round_up(nMaxRxAnt, nRxAntPerBlock);
                    kernelNodeParamsDriverSecond.blockDimX = nMaxThreads;
                    kernelNodeParamsDriverSecond.blockDimY = nRxAntPerBlock;
                    kernelNodeParamsDriverSecond.blockDimZ = 1;

                    constexpr uint32_t MAX_N_DMRS_PRB_IN_PER_CLUSTER = 8;
                    // DRMS grid type 1 has 6 tones per PRB and type 2 has 4 tones per PRB. We only currently
                    // support type 1 grids, but assume the max tones for smem allocation purposes.
                    constexpr uint32_t N_DMRS_MAX_GRID_TONES_PER_PRB = CUPHY_N_TONES_PER_PRB / 2;
                    constexpr uint32_t MAX_N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER = N_DMRS_MAX_GRID_TONES_PER_PRB * MAX_N_DMRS_PRB_IN_PER_CLUSTER;
                    using TComplexCompute = complex_from_scalar<TCompute>::type;
                    kernelNodeParamsDriverSecond.sharedMemBytes = sizeof(TComplexCompute) *
                        nRxAntPerBlock * nMaxLayers * MAX_N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER;
                }
            }
            else if(chEstAlgo == PUSCH_CH_EST_ALGO_TYPE_RKHS)
            {
                CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriver = launchCfg.kernelNodeParamsDriver;

                kernelNodeParamsDriver.gridDimX  = nTotComputeBlocks;
                kernelNodeParamsDriver.gridDimY  = 1;
                kernelNodeParamsDriver.gridDimZ  = 1;

                kernelNodeParamsDriver.blockDimX = nZpDmrsSc;
                kernelNodeParamsDriver.blockDimY = 1;
                kernelNodeParamsDriver.blockDimZ = 1;

                kernelNodeParamsDriver.extra = nullptr;
                kernelNodeParamsDriver.sharedMemBytes = 0;
            }


            launchCfg.kernelArgs[0] = &kernelArgs.pStatDescr;
            launchCfg.kernelArgs[1] = &kernelArgs.pDynDescr;

            launchCfg.kernelNodeParamsDriver.kernelParams = &(launchCfg.kernelArgs[0]);
            launchCfg.kernelNodeParamsDriverSecond.kernelParams = &(launchCfg.kernelArgs[0]);

            // pLaunchCfgsEHQ needs to be configured only once, hence running for hetCfgIdx == 0
            if((chEstTimeInstIdx == 0) && (hetCfgIdx == 0))
            {
                // setup launch configs for wait kernel used in sub-slot processing
                if((enableEarlyHarqProc) || (enableFrontLoadedDmrsProc))
                {
                    //configure wait and DGL kernels for sub-slot
                    pStartKernels->setWaitKernelParams(pWaitKernelLaunchCfgsPreSubSlot, CUPHY_PUSCH_SUB_SLOT_PATH, &kernelArgs.pStatDescr, &kernelArgs.pDynDescr);
                    pStartKernels->setDeviceGraphLaunchKernelParams(pDglKernelLaunchCfgsPreSubSlot, enableDeviceGraphLaunch, CUPHY_PUSCH_SUB_SLOT_PATH, &kernelArgs.pDynDescr, pSubSlotDeviceGraphExec);
                    //configure wait and DGL kernels for full-slot (the wait kernel is used only if sub-slot processing is enabled)
                    pStartKernels->setWaitKernelParams(pWaitKernelLaunchCfgsPostSubSlot, CUPHY_PUSCH_FULL_SLOT_PATH, &kernelArgs.pStatDescr, &kernelArgs.pDynDescr);
                    pStartKernels->setDeviceGraphLaunchKernelParams(pDglKernelLaunchCfgsPostSubSlot, enableDeviceGraphLaunch, CUPHY_PUSCH_FULL_SLOT_PATH, &kernelArgs.pDynDescr, pFullSlotDeviceGraphExec);
               }
            }
        }
    }
    if(enableDftSOfdm==1)
    {
        for(uint32_t idx = 0; idx < nUeGrps; idx++)
        {
            if(pDrvdUeGrpPrmsCpu[idx].enableTfPrcd==1)
            {
                CUDA_CHECK(cudaMemcpyToSymbolAsync(d_phi_6, nr_constants::phi_6, sizeof(d_phi_6), 0, cudaMemcpyHostToDevice, strm));
                CUDA_CHECK(cudaMemcpyToSymbolAsync(d_phi_12, nr_constants::phi_12, sizeof(d_phi_12), 0, cudaMemcpyHostToDevice, strm));
                CUDA_CHECK(cudaMemcpyToSymbolAsync(d_phi_18, nr_constants::phi_18, sizeof(d_phi_18), 0, cudaMemcpyHostToDevice, strm));
                CUDA_CHECK(cudaMemcpyToSymbolAsync(d_phi_24, nr_constants::phi_24, sizeof(d_phi_24), 0, cudaMemcpyHostToDevice, strm));
                CUDA_CHECK(cudaMemcpyToSymbolAsync(d_primeNums, primeNums, sizeof(primeNums), 0, cudaMemcpyHostToDevice, strm));

                break;
            }
        }
    }

    if(chEstAlgo == PUSCH_CH_EST_ALGO_TYPE_MULTISTAGE_MMSE_WITH_DELAY_EST || chEstAlgo == PUSCH_CH_EST_ALGO_TYPE_LS_ONLY)
    {
        for(int32_t ueGrpIdx = 0; ueGrpIdx < nUeGrps; ++ueGrpIdx)
        {
            // swap the active accumulation buffer
            cuphyPuschRxUeGrpPrms_t & drvdUeGrpPrms = pDrvdUeGrpPrmsCpu[ueGrpIdx];
            drvdUeGrpPrms.dmrsActiveAccumBuf ^= 1;
        }
    }

    return CUPHY_STATUS_SUCCESS;
 }

puschRxChEst::puschRxChEst(const cuphyChEstSettings& chEstSettings,
                           const bool earlyHarqModeEnabled) :
    m_chEstSettings(chEstSettings),
    m_chestGraphMgr(chEstSettings.nMaxChEstHetCfgs,
                    earlyHarqModeEnabled,
                    chEstSettings.chEstAlgo),
    m_chestStream(m_chestGraphMgr.getLaunchCfgs(),
                  earlyHarqModeEnabled,
                  chEstSettings.chEstAlgo) {}

void puschRxChEst::setEarlyHarqModeEnabled(const bool earlyHarqModeEnabled) {
    m_chestGraphMgr.setEarlyHarqModeEnabled(earlyHarqModeEnabled);
    m_chestStream.setEarlyHarqModeEnabled(earlyHarqModeEnabled);
}

IChestGraphNodes& puschRxChEst::chestGraph() { return m_chestGraphMgr.asChest(); }
IChestStream& puschRxChEst::chestStream() { return m_chestStream; }
IChestSubSlotNodes& puschRxChEst::earlyHarqGraph() { return m_chestGraphMgr.asEarlyHarq(); }
IChestSubSlotNodes& puschRxChEst::frontDmrsGraph() { return m_chestGraphMgr.asFrontDmrs(); }
pusch::IStartKernels& puschRxChEst::startKernels() { return m_startKernels; }

 cuphyStatus_t
 puschRxChEst::setup(IKernelBuilder*                       pKernelBuilder,
                     gsl_lite::span<cuphyPuschRxUeGrpPrms_t>    pDrvdUeGrpPrmsCpu,
                     gsl_lite::span<cuphyPuschRxUeGrpPrms_t>    pDrvdUeGrpPrmsGpu,
                     uint16_t                              nUeGrps,
                     uint8_t                               maxDmrsMaxLen,
                     uint8_t*                              pPreEarlyHarqWaitKernelStatusGpu,
                     uint8_t*                              pPostEarlyHarqWaitKernelStatusGpu,
                     const uint16_t                        waitTimeOutPreEarlyHarqUs,
                     const uint16_t                        waitTimeOutPostEarlyHarqUs,
                     bool                                  enableCpuToGpuDescrAsyncCpy,
                     gsl_lite::span<uint8_t*>                   ppDynDescrsCpu,
                     gsl_lite::span<uint8_t*>                   ppDynDescrsGpu,
                     uint8_t                               enableEarlyHarqProc,
                     uint8_t                               enableFrontLoadedDmrsProc,
                     uint8_t                               enableDeviceGraphLaunch,
                     CUgraphExec*                          pSubSlotDeviceGraphExec,
                     CUgraphExec*                          pFullSlotDeviceGraphExec,
                     cuphyPuschRxWaitLaunchCfg_t*          pWaitKernelLaunchCfgsPreSubSlot,
                     cuphyPuschRxWaitLaunchCfg_t*          pWaitKernelLaunchCfgsPostSubSlot,
                     cuphyPuschRxDglLaunchCfg_t*           pDglKernelLaunchCfgsPreSubSlot,
                     cuphyPuschRxDglLaunchCfg_t*           pDglKernelLaunchCfgsPostSubSlot,
                     cudaStream_t                          strm)

 {
    if (!pKernelBuilder) {
        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "Invalid nullptr pKernelBuilder");
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    const auto ret = pKernelBuilder->build(pDrvdUeGrpPrmsCpu,
                                           pDrvdUeGrpPrmsGpu,
                                           nUeGrps,
                                           maxDmrsMaxLen,
                                           m_chEstSettings.enableDftSOfdm,
                                           m_chEstSettings.chEstAlgo,
                                           m_chEstSettings.enableMassiveMIMO,
                                           m_chEstSettings.enablePerPrgChEst,
                                           pPreEarlyHarqWaitKernelStatusGpu,
                                           pPostEarlyHarqWaitKernelStatusGpu,
                                           waitTimeOutPreEarlyHarqUs,
                                           waitTimeOutPostEarlyHarqUs,
                                           enableCpuToGpuDescrAsyncCpy,
                                           ppDynDescrsCpu,
                                           ppDynDescrsGpu,
                                           &m_startKernels,
                                           m_chestGraphMgr.getLaunchCfgs(),
                                           enableEarlyHarqProc,
                                           enableFrontLoadedDmrsProc,
                                           enableDeviceGraphLaunch,
                                           pSubSlotDeviceGraphExec,
                                           pFullSlotDeviceGraphExec,
                                           pWaitKernelLaunchCfgsPreSubSlot,
                                           pWaitKernelLaunchCfgsPostSubSlot,
                                           pDglKernelLaunchCfgsPreSubSlot,
                                           pDglKernelLaunchCfgsPostSubSlot,
                                           strm);
    return ret;
 }

} // namespace ch_est
