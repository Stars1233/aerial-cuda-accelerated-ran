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
#include <cooperative_groups.h>
#include "bfc.hpp"
#include "cuphy_complex_ops.cuh"
#include "cuphy.hpp"
#include "type_convert.hpp"
#include "bfw_blockFP.cuh"
#include <vector>

using namespace cooperative_groups;

namespace bfw_coefComp
{
// #define ENABLE_DEBUG

/*VCAST_DONT_INSTRUMENT_START*/

// Import types from cuphy_cmplx (explicit declarations to avoid ambiguity)
using cuphy_cmplx::tensor_ref;
using cuphy_cmplx::block_1D;
using cuphy_cmplx::block_2D;
using cuphy_cmplx::block_3D;

// Import functions and operators from cuphy_cmplx
using namespace cuphy_cmplx;

template <typename TElem, int NDim>
struct tensor_ref_v0
{
    TElem* addr;
    int    dim[NDim];
    int    strides[NDim];
    size_t n_elem = 1;
    tensor_ref_v0(tensor_pair& tp) :
        addr(static_cast<TElem*>(tp.second)),
        n_elem(1)
    {
        const tensor_layout_any& layout = tp.first.get().layout();
#pragma unroll
        for(int i = 0; i < NDim; ++i)
        {
            dim[i]     = (layout.rank() > i) ? layout.dimensions[i] : 1;
            strides[i] = (layout.rank() > i) ? layout.strides[i] : 0;
            n_elem *= dim[i];
        }
    }
    tensor_ref_v0(const_tensor_pair& tp) :
        addr(static_cast<TElem*>(tp.second)),
        n_elem(1)
    {
        const tensor_layout_any& layout = tp.first.get().layout();
#pragma unroll
        for(int i = 0; i < NDim; ++i)
        {
            dim[i]     = (layout.rank() > i) ? layout.dimensions[i] : 1;
            strides[i] = (layout.rank() > i) ? layout.strides[i] : 0;
            n_elem *= dim[i];
        }
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
    CUDA_BOTH TElem& operator()(int i0) { return *(addr + offset(i0)); }
    CUDA_BOTH const TElem& operator()(int i0) const { return *(addr + offset(i0)); }
    CUDA_BOTH TElem& operator()(int i0, int i1) { return *(addr + offset(i0, i1)); }
    CUDA_BOTH const TElem& operator()(int i0, int i1) const { return *(addr + offset(i0, i1)); }
    CUDA_BOTH TElem& operator()(int i0, int i1, int i2) { return *(addr + offset(i0, i1, i2)); }
    CUDA_BOTH const TElem& operator()(int i0, int i1, int i2) const { return *(addr + offset(i0, i1, i2)); }
    CUDA_BOTH TElem& operator()(int i0, int i1, int i2, int i3) { return *(addr + offset(i0, i1, i2, i3)); }
    CUDA_BOTH const TElem& operator()(int i0, int i1, int i2, int i3) const { return *(addr + offset(i0, i1, i2, i3)); }

    CUDA_BOTH size_t num_elem() { return n_elem; };
};

// Note: tensor_ref, block_1D, block_2D, block_3D, cuGet, cuAbs, cuRSqrt, div_round_up
// are now provided by cuphy_complex_ops.cuh

template <typename TStorageIn,
          typename TCompute,
          uint32_t N_ROWS_MAT,
          uint32_t N_COLS_MAT>
__device__ __forceinline__ void cmplxMatLoadColMjr(thread_block const&                                                                      thisThrdBlk,
                                                   block_2D<const typename complex_from_scalar<TStorageIn>::type*, N_ROWS_MAT, N_COLS_MAT>& srcMat,
                                                   block_2D<typename complex_from_scalar<TCompute>::type*, N_ROWS_MAT + 1, N_COLS_MAT>&     dstMat)
{
    typedef typename complex_from_scalar<TStorageIn>::type TComplexStorageIn;
    typedef typename complex_from_scalar<TCompute>::type   TComplexCompute;

    const uint32_t     N_THRDS                 = thisThrdBlk.size();
    const uint32_t     THRD_IDX                = thisThrdBlk.thread_rank();
    constexpr uint32_t N_MAT_ELEMS_TO_RD       = N_ROWS_MAT * N_COLS_MAT;
    const uint32_t     N_MAT_ELEMS_RD_PER_ITER = (N_MAT_ELEMS_TO_RD > N_THRDS) ? N_THRDS : N_MAT_ELEMS_TO_RD;
    const uint32_t     N_ITER_TO_RD_MAT        = div_round_up(N_MAT_ELEMS_TO_RD, N_MAT_ELEMS_RD_PER_ITER);

    for(uint32_t i = 0; i < N_ITER_TO_RD_MAT; ++i)
    {
        uint32_t matElemIdx = ((i * N_MAT_ELEMS_RD_PER_ITER) + THRD_IDX);
        uint32_t iRow       = matElemIdx % N_ROWS_MAT;
        uint32_t iCol       = matElemIdx / N_ROWS_MAT;
        // Not all threads would participate in the last iteration
        if(matElemIdx < N_MAT_ELEMS_TO_RD)
        {
            dstMat(iRow, iCol) =
                type_convert<TComplexCompute>(srcMat(iRow, iCol));

#ifdef ENABLE_DEBUG
            printf("Mat[%d][%d] = %f+j%f\n", iRow, iCol, dstMat(iRow, iCol).x, dstMat(iRow, iCol).y);
#endif
        }
    }
}
/*VCAST_DONT_INSTRUMENT_END*/

template <typename TStorageIn,
          typename TCompute,
          uint32_t N_ROWS_MAT,
          uint32_t N_COLS_MAT,
	  uint32_t NLE >
__device__ __forceinline__ void cmplxMatLoadRowMjr(thread_block const&                                                                  thisThrdBlk,
                                                   uint32_t                                                                             iPrb,
                                                   tensor_ref_v0<const typename complex_from_scalar<TStorageIn>::type, 3>                  tMatSrc, // (N_BS_ANTS, N_PRB, N_LAYERS)
                                                   block_2D<typename complex_from_scalar<TCompute>::type*, N_ROWS_MAT + NLE, N_COLS_MAT>& matDst)
{
    typedef typename complex_from_scalar<TStorageIn>::type TComplexStorageIn;
    typedef typename complex_from_scalar<TCompute>::type   TComplexCompute;

    const uint32_t     N_THRDS                 = thisThrdBlk.size();
    const uint32_t     THRD_IDX                = thisThrdBlk.thread_rank();
    constexpr uint32_t N_MAT_ELEMS_TO_RD       = N_ROWS_MAT * N_COLS_MAT;
    const uint32_t     N_MAT_ELEMS_RD_PER_ITER = (N_MAT_ELEMS_TO_RD > N_THRDS) ? N_THRDS : N_MAT_ELEMS_TO_RD;
    const uint32_t     N_ITER_TO_RD_MAT        = div_round_up(N_MAT_ELEMS_TO_RD, N_MAT_ELEMS_RD_PER_ITER);

    for(uint32_t i = 0; i < N_ITER_TO_RD_MAT; ++i)
    {
        uint32_t matElemIdx = ((i * N_MAT_ELEMS_RD_PER_ITER) + THRD_IDX);
        uint32_t iCol       = matElemIdx % N_COLS_MAT;
        uint32_t iRow       = matElemIdx / N_COLS_MAT;
        // Not all threads may participate in the last iteration
        if(matElemIdx < N_MAT_ELEMS_TO_RD)
        {
#if 1 // Higher SRS_CH_EST store efficiency (but affects BFC load efficiency): (N_SRS_CH_EST_IN_FREQ, N_BS_ANTS, N_LAYERS)
            matDst(iRow, iCol) =
                type_convert<TComplexCompute>(tMatSrc(iPrb, iCol, iRow));
#else // BFW friendly load format (but negatively affects SRS_CH_EST store efficiency): (N_BS_ANTS, N_SRS_CH_EST_IN_FREQ, N_LAYERS)
            matDst(iRow, iCol) =
                type_convert<TComplexCompute>(tMatSrc(iCol, iPrb, iRow));
#endif

#ifdef ENABLE_DEBUG
            printf("Mat[%d][%d][%d] = %f+j%f\n", iPrb, iRow, iCol, matDst(iRow, iCol).x, matDst(iRow, iCol).y);
#endif
        }
    }
}

template <typename TStorageOut,
          typename TCompute,
          uint32_t N_ROWS_MAT,
          uint32_t N_COLS_MAT>
__device__ __forceinline__ void cmplxMatStore_v0(thread_block const&                                                                  thisThrdBlk,
                                                 uint32_t                                                                             iPrb,
                                                 TCompute                                                                             scale,
                                                 block_2D<typename complex_from_scalar<TCompute>::type*, N_ROWS_MAT + 1, N_COLS_MAT>& matSrc,
                                                 tensor_ref_v0<typename complex_from_scalar<TStorageOut>::type, 3>                       tMatDst) // (N_BS_ANTS, N_LAYERS, N_PRB)
{
    typedef typename complex_from_scalar<TStorageOut>::type TComplexStorageOut;
    typedef typename complex_from_scalar<TCompute>::type    TComplexCompute;

    const uint32_t     N_THRDS                 = thisThrdBlk.size();
    const uint32_t     THRD_IDX                = thisThrdBlk.thread_rank();
    constexpr uint32_t N_MAT_ELEMS_TO_WR       = N_ROWS_MAT * N_COLS_MAT;
    const uint32_t     N_MAT_ELEMS_WR_PER_ITER = (N_MAT_ELEMS_TO_WR > N_THRDS) ? N_THRDS : N_MAT_ELEMS_TO_WR;
    const uint32_t     N_ITER_TO_WR_MAT        = div_round_up(N_MAT_ELEMS_TO_WR, N_MAT_ELEMS_WR_PER_ITER);

    for(uint32_t i = 0; i < N_ITER_TO_WR_MAT; ++i)
    {
        uint32_t matElemIdx = ((i * N_MAT_ELEMS_WR_PER_ITER) + THRD_IDX);
        uint32_t iRow       = matElemIdx % N_ROWS_MAT;
        uint32_t iCol       = matElemIdx / N_ROWS_MAT;
        // Not all threads would participate in the last iteration
        if(matElemIdx < N_MAT_ELEMS_TO_WR)
        {
            tMatDst(iRow, iCol, iPrb) =
                type_convert<TComplexStorageOut>(matSrc(iRow, iCol) * scale);

#ifdef ENABLE_DEBUG
            printf("Mat[%d][%d][%d] = %f+j%f\n", iPrb, iRow, iCol, tMatDst(iRow, iCol, iPrb).x, tMatDst(iRow, iCol, iPrb).y);
#endif
        }
    }
}

/*VCAST_DONT_INSTRUMENT_START*/
template <typename TStorageOut,
          typename TCompute,
          uint32_t N_ROWS_MAT,
          uint32_t N_COLS_MAT>
__device__ __forceinline__ void cmplxMatStore(thread_block const&                                                                  thisThrdBlk,
                                              uint32_t                                                                             iPrb,
                                              TCompute                                                                             scale,
                                              block_2D<typename complex_from_scalar<TCompute>::type*, N_ROWS_MAT + 1, N_COLS_MAT>& matSrc,
                                              tensor_ref<typename complex_from_scalar<TStorageOut>::type>                          tMatDst) // (N_BS_ANTS, N_LAYERS, N_PRB)
{
    typedef typename complex_from_scalar<TStorageOut>::type TComplexStorageOut;
    typedef typename complex_from_scalar<TCompute>::type    TComplexCompute;

    const uint32_t     N_THRDS                 = thisThrdBlk.size();
    const uint32_t     THRD_IDX                = thisThrdBlk.thread_rank();
    constexpr uint32_t N_MAT_ELEMS_TO_WR       = N_ROWS_MAT * N_COLS_MAT;
    const uint32_t     N_MAT_ELEMS_WR_PER_ITER = (N_MAT_ELEMS_TO_WR > N_THRDS) ? N_THRDS : N_MAT_ELEMS_TO_WR;
    const uint32_t     N_ITER_TO_WR_MAT        = div_round_up(N_MAT_ELEMS_TO_WR, N_MAT_ELEMS_WR_PER_ITER);

    for(uint32_t i = 0; i < N_ITER_TO_WR_MAT; ++i)
    {
        uint32_t matElemIdx = ((i * N_MAT_ELEMS_WR_PER_ITER) + THRD_IDX);
        uint32_t iRow       = matElemIdx % N_ROWS_MAT;
        uint32_t iCol       = matElemIdx / N_ROWS_MAT;
        // Not all threads would participate in the last iteration
        if(matElemIdx < N_MAT_ELEMS_TO_WR)
        {
            tMatDst(iRow, iCol, iPrb) =
                type_convert<TComplexStorageOut>(matSrc(iRow, iCol) * scale);

#ifdef ENABLE_DEBUG
            printf("Mat[%d][%d][%d] = %f+j%f\n", iPrb, iRow, iCol, tMatDst(iRow, iCol, iPrb).x, tMatDst(iRow, iCol, iPrb).y);
#endif
        }
    }
}
/*VCAST_DONT_INSTRUMENT_END*/

/*VCAST_DONT_INSTRUMENT_START*/
template <uint32_t THRD_GRP_SIZE>
__device__ __forceinline__
    __half2
    thrdGrpAllReduceSum(thread_block_tile<THRD_GRP_SIZE> const& thisThrdGrp, __half2 const& val)
{
    uint32_t thrdGrpSize = thisThrdGrp.size();
    __half2  sum         = val;
    for(int32_t i = thrdGrpSize / 2; i > 0; i /= 2)
    {
        sum.x += __float2half(thisThrdGrp.shfl_xor(sum.x, i));
        sum.y += __float2half(thisThrdGrp.shfl_xor(sum.y, i));
    }
    thisThrdGrp.sync();
    return sum;
}
/*VCAST_DONT_INSTRUMENT_END*/

template <uint32_t THRD_GRP_SIZE>
__device__ __forceinline__
    cuComplex
    thrdGrpAllReduceSum(thread_block_tile<THRD_GRP_SIZE> const& thisThrdGrp, cuComplex const& val)
{
    uint32_t  thrdGrpSize = thisThrdGrp.size();
    cuComplex sum         = val;
    for(int32_t i = thrdGrpSize / 2; i > 0; i /= 2)
    {
        sum.x += thisThrdGrp.shfl_xor(cuReal(sum), i);
        sum.y += thisThrdGrp.shfl_xor(cuImag(sum), i);
    }
    thisThrdGrp.sync();
    return sum;
}

/*VCAST_DONT_INSTRUMENT_START*/
template <uint32_t THRD_GRP_SIZE>
__device__ __forceinline__ __half
thrdGrpAllReduceSum(thread_block_tile<THRD_GRP_SIZE> const& thisThrdGrp, __half const& val)
{
    uint32_t thrdGrpSize = thisThrdGrp.size();
    __half   sum         = val;
    for(int32_t i = thrdGrpSize / 2; i > 0; i /= 2)
    {
        sum += __float2half(thisThrdGrp.shfl_xor(sum, i));
    }
    thisThrdGrp.sync();
    return sum;
}
/*VCAST_DONT_INSTRUMENT_END*/

template <uint32_t THRD_GRP_SIZE>
__device__ __forceinline__ float
thrdGrpAllReduceSum(thread_block_tile<THRD_GRP_SIZE> const& thisThrdGrp, float const& val)
{
    uint32_t thrdGrpSize = thisThrdGrp.size();
    float    sum         = val;
    for(int32_t i = thrdGrpSize / 2; i > 0; i /= 2)
    {
        sum += thisThrdGrp.shfl_xor(sum, i);
    }
    thisThrdGrp.sync();
    return sum;
}

// Inplace LU factorization of Matrix A - Iterative version (submatrix updates done one row at a time)
// Iterative version maybe used if the thread block size is around the length of a row (i.e. N_COLS_MAT) of
// the augmented matrix
template <typename TCompute,
          uint32_t N_ROWS_MAT,
          uint32_t N_COLS_MAT,
	  uint32_t NLE >
__device__ __forceinline__ void luFactorizeIter(thread_block const&                                                                  thisThrdBlk,
                                                block_2D<typename complex_from_scalar<TCompute>::type*, N_ROWS_MAT + NLE, N_COLS_MAT>& matA)
{
    typedef typename complex_from_scalar<TCompute>::type TComplexCompute;

    constexpr uint32_t N_OUTER_ITER = (N_ROWS_MAT > 1) ? (N_ROWS_MAT - 1) : 1;    
    const uint32_t THRD_ABS_IDX = thisThrdBlk.thread_rank();

    // Iterate row by row of A applying Gaussian elimination. In each iteration Gaussian elimination
    // annihilates all elements of a column below main diagonal of G. In iteration k annihilate elements
    // G(k+1:n, k ). At the end of all iterations G is transformed to U
    // While transforming G to U, applying Gaussian elimination to other columns of A i.e. matrices I and M
    // produces matrices Linv and F respetively which can then be used to compute Ree and C via back
    // substitution
#pragma unroll
    for(uint32_t k = 0; k < N_OUTER_ITER; ++k)
    {
        // Gaussian elimination on submatrix A(k,k), since we know that A(k+1:n,k) will be annihilated we directly
        // proceed to applying Gaussian elimination on submatrix A(k+1:n,k+1:n)

        // Complex multiplication by inverse of real number is cheaper instead of complex division
        // @todo: add a safety check on Akk to avoid divide by zero
        TCompute minus_one_over_Akk = cuGet<TCompute>(-1) / cuReal(matA(k, k));

#ifdef ENABLE_DEBUG
        printf("Iteration: %d, A[%d][%d] = %f+j%f, inv = %f---------------\n", k, k, k, matA(k, k).x, matA(k, k).y, minus_one_over_Akk);
#endif

#pragma unroll
        for(uint32_t i = k + 1; i < N_ROWS_MAT; ++i)
        {
            // Perform Gaussian elimination:
            // linear combination of row k and row i starting from column element k+1:N_COLS_A
            if((THRD_ABS_IDX > k) && (THRD_ABS_IDX < N_COLS_MAT))
            {
	        // Compute multipliers needed for Gaussian elimination.
#ifdef ENABLE_DEBUG
                printf("Before storing multiplier: A[%d][%d] = %f+j%f\n", i, k, matA(i, k).x, matA(i, k).y);
#endif
                // All participating threads compute multiplier Aik
                TComplexCompute Aik = matA(i, k) * minus_one_over_Akk;

#ifdef ENABLE_DEBUG
                printf("After storing multiplier: A[%d][%d] = %f+j%f\n", i, k, Aik.x, Aik.y);
#endif
                matA(i, THRD_ABS_IDX) = cuCma(Aik, matA(k, THRD_ABS_IDX), matA(i, THRD_ABS_IDX));

#ifdef ENABLE_DEBUG
                printf("A[%d][%d] = %f+j%f\n", i, THRD_ABS_IDX, matA(i, THRD_ABS_IDX).x, matA(i, THRD_ABS_IDX).y);
#endif
            }

        }

        // Wait for the entire submatrix update
        __syncthreads();
    }
}

// Inplace LU factorization of Matrix A - Parallel version (entire sub-matrix update done one in parallel)
// - Parallel version maybe used (over iterative version) if the thread block size is much larger than the
// length of a row (i.e. N_COLS_MAT) of the augmented matrix resulting in fewer inactive threads during LU factorization.
// - This parallel version (luFactorizeParallel_v1) maximizes the number of active threads during the
// sub-matrix update by recomputing indices for each outerloop iteration and also reduces the number of inner
// loop iterations needed to update the submatrix. The index recomputation cost is paid once per outer loop
// iteration.
template <typename TCompute,
          uint32_t N_ROWS_MAT,
          uint32_t N_COLS_MAT,
	  uint32_t NLE >
__device__ __forceinline__ void luFactorizeParallel_v1(thread_block const&                                                                  thisThrdBlk,
                                                       block_2D<typename complex_from_scalar<TCompute>::type*, N_ROWS_MAT + NLE, N_COLS_MAT>& matA)
{
    typedef typename complex_from_scalar<TCompute>::type TComplexCompute;

    constexpr uint32_t N_OUTER_ITER = (N_ROWS_MAT > 1) ? (N_ROWS_MAT - 1) : 1;

    const uint32_t N_THRDS      = thisThrdBlk.size();
    const uint32_t THRD_ABS_IDX = thisThrdBlk.thread_rank();

    // Iterate row by row of A applying Gaussian elimination. In each iteration Gaussian elimination
    // annihilates all elements of a column below main diagonal of G. In iteration k annihilate elements
    // G(k+1:n, k ). At the end of all iterations G is transformed to U
    // While transforming G to U, applying Gaussian elimination to other columns of A i.e. matrices I and M
    // produces matrices Linv and F respetively which can then be used to compute Ree and C via back
    // substitution
    // #pragma unroll
    for(uint32_t k = 0; k < N_OUTER_ITER; ++k)
    {
        // Gaussian elimination on submatrix A(k,k), since we know that A(k+1:n,k) will be annihilated we directly
        // proceed to applying Gaussian elimination on submatrix A(k+1:n,k+1:n)

        // Complex multiplication by inverse of real number is cheaper instead of complex division
        // @todo: add a safety check on Akk to avoid divide by zero
        TCompute minus_one_over_Akk = cuGet<TCompute>(-1) / cuReal(matA(k, k));

#ifdef ENABLE_DEBUG
        printf("Iteration: %d, A[%d][%d] = %f+j%f, inv = %f---------------\n", k, k, k, matA(k, k).x, matA(k, k).y, minus_one_over_Akk);
#endif

        // The entire sub-matrix can be updated in parallel (i.e. in addition to columns, the rows are also updated in parallel)
        // to extent permitted by parallelism (i.e. thread count) available in the thread block
        uint32_t subMatStartRowOffset = (k + 1);
        uint32_t subMatStartColOffset = (k + 1);
        uint32_t nRowsSubMat          = N_ROWS_MAT - subMatStartRowOffset;
        uint32_t nColsSubMat          = N_COLS_MAT - subMatStartColOffset;
        uint32_t subMatColIdx         = THRD_ABS_IDX % nColsSubMat;
        uint32_t matColIdx            = subMatStartColOffset + subMatColIdx; // process columns > k, note: matrix is in column major layout

        // Ensure whole rows are updated at a time
        // Assumes N_THRDS_X >= nColsSubMat
        uint32_t nRowsSubMatPerIter = N_THRDS / nColsSubMat;
        bool     thrdEnable         = (THRD_ABS_IDX < (nRowsSubMatPerIter * nColsSubMat)); // Disable threads which don't update full rows
        uint32_t nIterToProcSubMat  = div_round_up(nRowsSubMat, nRowsSubMatPerIter);
        for(uint32_t i = 0; i < nIterToProcSubMat; ++i)
        {
            uint32_t subMatRowIdx = (i * nRowsSubMatPerIter) + (THRD_ABS_IDX / nColsSubMat);
            uint32_t matRowIdx    = subMatStartRowOffset + subMatRowIdx; // process rows > k

            TComplexCompute Aik = cuGet<TComplexCompute>(0);
            if(thrdEnable && (matRowIdx < N_ROWS_MAT))
            {
                // Compute multipliers needed for Gaussian elimination. For storage compactness the multipliers
                // (non-zero elements of Gauss vector/column of L) are stored in the annihilated zero location of
                // columns of U

#ifdef ENABLE_DEBUG
                printf("Before storing multiplier: A[%d][%d] = %f+j%f\n", matRowIdx, k, matA(matRowIdx, k).x, matA(matRowIdx, k).y);
#endif
                // All threads compute multiplier Aik into a register and use it but only one thread stores it back
                Aik = matA(matRowIdx, k) * minus_one_over_Akk;

#ifdef ENABLE_DEBUG
                printf("After storing multiplier: A[%d][%d] = %f+j%f\n", matRowIdx, k, Aik.x, Aik.y);
#endif
                // Perform Gaussian elimination:
                // linear combination of row k and row i starting from column element k+1:N_COLS_A
                // if((THRD_ABS_IDX > k) && (THRD_ABS_IDX < N_COLS_MAT))
                if(matColIdx < N_COLS_MAT)
                {
                    matA(matRowIdx, matColIdx) = cuCma(Aik, matA(k, matColIdx), matA(matRowIdx, matColIdx));

#ifdef ENABLE_DEBUG
                    printf("A[%d][%d] = %f+j%f\n", matRowIdx, matColIdx, matA(matRowIdx, matColIdx).x, matA(matRowIdx, matColIdx).y);
#endif
                }
            }

            /*
                 These are the updates to the lower triangular, which via Gaussian elimination, are being zeroed.
                 This is not needed.  The updates to matA would be zero if there was sufficient precision.
                 As they are essentially zero, they are not used anywhere else in the computation.  (Or could be replaced by zeros if somehow needed.)
                 Removing the update removes the need for the frequent block synchronization which is the main performance improvement.

                 Keeping code here, but commented out, since this may be useful in the future for validation.

            // Ensure all threads (which may extend across multiple thrdGrps for nColsA > 32) have read and use shA(i,k) before writing into it
            thisThrdBlk.sync();
            if(thrdEnable && (matRowIdx < N_ROWS_MAT) && (subMatStartColOffset == matColIdx))
            {
                matA(matRowIdx, k) = Aik;
            }

            */
            
        }

        // Wait for the entire submatrix update
        __syncthreads();
    }
}

// luFactorizeParallel_v2 not used in the current implementation
/*VCAST_DONT_INSTRUMENT_START*/
// Inplace LU factorization of Matrix - Parallel version (entire sub-matrix update done one in parallel)
// - Parallel version maybe used (over iterative version) if the thread block size is much larger than the
// length of a row (i.e. N_COLS_MAT) of the augmented matrix resulting in fewer inactive threads during LU factorization.
// - This parallel version computes indices used in sub-matrix update once before the outerloop eliminating
// per outer loop index recomputation cost. Consequently (unlike luFactorizeParallel_v1) the smaller sub-matrix
// updates do not utilize all the availalbe threads (inactive thread count increases with smaller sub-matrices)
// while the number of inner loop iterations does not decrease with decrease in sub-matrix dimension.
template <typename TCompute,
          uint32_t N_ROWS_MAT,
          uint32_t N_COLS_MAT>
__device__ __forceinline__ void luFactorizeParallel_v2(thread_block const&                                                                  thisThrdBlk,
                                                       block_2D<typename complex_from_scalar<TCompute>::type*, N_ROWS_MAT + 1, N_COLS_MAT>& matA)
{
    typedef typename complex_from_scalar<TCompute>::type TComplexCompute;

    const uint32_t N_THRDS      = thisThrdBlk.size();
    const uint32_t THRD_ABS_IDX = thisThrdBlk.thread_rank();

    // Ensure whole rows are updated at a time
    // Assumes N_THRDS_X >= nColsSubMat
    uint32_t nRowsMatPerIter = N_THRDS / N_COLS_MAT;
    bool     thrdEnableMain  = (THRD_ABS_IDX < (nRowsMatPerIter * N_COLS_MAT)); // Disable threads which don't update full rows
    uint32_t nIterToProcMat  = div_round_up(N_ROWS_MAT, nRowsMatPerIter);
    uint32_t matColIdx       = THRD_ABS_IDX % N_COLS_MAT;
    uint32_t matRowOffset    = THRD_ABS_IDX / N_COLS_MAT;

    // Iterate row by row of A applying Gaussian elimination. In each iteration Gaussian elimination
    // annihilates all elements of a column below main diagonal of G. In iteration k annihilate elements
    // G(k+1:n, k ). At the end of all iterations G is transformed to U
    // While transforming G to U, applying Gaussian elimination to other columns of A i.e. matrices I and M
    // produces matrices Linv and F respetively which can then be used to compute Ree and C via back
    // substitution
    // #pragma unroll
    for(uint32_t k = 0; k < N_ROWS_MAT - 1; ++k)
    {
        // Gaussian elimination on submatrix A(k,k), since we know that A(k+1:n,k) will be annihilated we directly
        // proceed to applying Gaussian elimination on submatrix A(k+1:n,k+1:n)

        // Complex multiplication by inverse of real number is cheaper instead of complex division
        // @todo: add a safety check on Akk to avoid divide by zero
        TCompute minus_one_over_Akk = cuGet<TCompute>(-1) / cuReal(matA(k, k));

#ifdef ENABLE_DEBUG
        printf("Iteration: %d, A[%d][%d] = %f+j%f, inv = %f---------------\n", k, k, k, matA(k, k).x, matA(k, k).y, minus_one_over_Akk);
#endif

        // The entire sub-matrix can be updated in parallel (i.e. in addition to columns, the rows are also updated in parallel)
        // to extent permitted by parallelism (i.e. thread count) available in the thread block
        for(uint32_t i = 0; i < nIterToProcMat; ++i)
        {
            uint32_t matRowIdx = (i * nRowsMatPerIter) + matRowOffset;
            // process rows > k and process columns > k
            bool thrdEnable = thrdEnableMain && ((matRowIdx > k) && (matRowIdx < N_ROWS_MAT) &&
                                                 (matColIdx > k));

            TComplexCompute Aik = cuGet<TComplexCompute>(0);
            if(thrdEnable)
            {
                // Compute multipliers needed for Gaussian elimination. For storage compactness the multipliers
                // (non-zero elements of Gauss vector/column of L) are stored in the annihilated zero location of
                // columns of U

#ifdef ENABLE_DEBUG
                printf("Before storing multiplier: A[%d][%d] = %f+j%f\n", matRowIdx, k, matA(matRowIdx, k).x, matA(matRowIdx, k).y);
#endif
                // All threads compute multiplier Aik into a register and use it but only one thread stores it back
                Aik = matA(matRowIdx, k) * minus_one_over_Akk;

#ifdef ENABLE_DEBUG
                printf("After storing multiplier: A[%d][%d] = %f+j%f\n", matRowIdx, k, Aik.x, Aik.y);
#endif
                // Perform Gaussian elimination:
                // linear combination of row k and row i starting from column element k+1:N_COLS_A
                matA(matRowIdx, matColIdx) = cuCma(Aik, matA(k, matColIdx), matA(matRowIdx, matColIdx));

#ifdef ENABLE_DEBUG
                printf("A[%d][%d] = %f+j%f\n", matRowIdx, matColIdx, matA(matRowIdx, matColIdx).x, matA(matRowIdx, matColIdx).y);
#endif
            }

            // Ensure all threads (which may extend across multiple thrdGrps for nColsA > 32) have read and use shA(i,k) before writing into it
            thisThrdBlk.sync();
            if(thrdEnable && ((k + 1) == matColIdx))
            {
                matA(matRowIdx, k) = Aik;
            }
        }

        // Wait for the entire submatrix update
        thisThrdBlk.sync();
    }
}
/*VCAST_DONT_INSTRUMENT_END*/

// BeamFormingCancellation (BFC) coefficient computation kernel
// {N_LAYERS, N_BS_ANTS} = {16,64}
// Inputs and outputs assumed to be column major
// dimBlock: (N_THREADS_PER_WARP, N_LAYERS)
// dimGrid : (Nprb)
template <typename TStorageIn,
          typename TStorageOut,
          typename TCompute,
          uint32_t N_BS_ANTS, // # of BS antenna (# of rows in H matrix)
          uint32_t N_LAYERS,  // # of layers (# of cols in H matrix)
          uint32_t N_THRD_GRPS_PER_THRD_BLK,
          uint32_t N_THRDS_PER_GRP>
__global__ void
bfc_mmse_coef_comp_kernel_v0(tensor_ref_v0<const typename complex_from_scalar<TStorageIn>::type, 3> tH,      // (N_BS_ANTS, N_PRB, N_LAYERS)
                             tensor_ref_v0<const TStorageIn, 2>                                     tLambda, // (N_LAYERS, N_PRB)
                             tensor_ref_v0<typename complex_from_scalar<TStorageOut>::type, 3>      tCoef,   // (N_BS_ANTS, N_LAYERS, N_PRB)
                             tensor_ref_v0<typename complex_from_scalar<TStorageOut>::type, 4>      tDbg)
{
    // H is channel matrix
    // G is the enhanced Gram matrix
    // A is the augmented matrix, A = [ G | I | H ]

    // The shared memory data is stored column-major where the number of rows is N_LAYERS.  The column stride
    // had been N_LAYERS+1, likely to minimize shared memory bank conflicts.  However, for the case of
    // N_LAYERS == 15, the shared memory bank conflict was not avoided at all.
    // To remedy this a constant NLE is created, which is 1 when N_LAYERS is even.  This is then
    // used to pad the columns of shared memory matrix data, such that all strides are odd, and shared memory
    // bank conflicts are always avoided.
    // NLE = short for N_Layers is Even.
    constexpr uint32_t NLE = (N_LAYERS%2) ? 0 : 1;

    //--------------------------------------------------------------------------------------------------------
    // Dimensions

    // H  : Channel matrix
    constexpr uint32_t N_ROWS_H = N_LAYERS;
    constexpr uint32_t N_COLS_H = N_BS_ANTS;

    // R  : Diagonal matrix (lambda) with per layer regularization coefficients
    constexpr uint32_t N_ROWS_R = N_LAYERS;
    // constexpr uint32_t N_COLS_R = N_LAYERS;

    // G  : Enhanced Gram matrix, G = H*H' + R
    constexpr uint32_t N_ROWS_G = N_LAYERS;
    constexpr uint32_t N_COLS_G = N_LAYERS;

    // I  : Identity matrix
    constexpr uint32_t N_ROWS_I = N_LAYERS;
    constexpr uint32_t N_COLS_I = N_LAYERS;

    // Linv: inverse lower trianuglar matrix in LU factorization
    constexpr uint32_t N_ROWS_LINV = N_LAYERS;

    // U  : Upper triangular matrix
    // constexpr uint32_t N_ROWS_U = N_ROWS_G;
    constexpr uint32_t N_COLS_U = N_COLS_G;

    // C  : MMSE coefficient matrix, C = H'*Ginv = H'*inv(H*H' + D)
    constexpr uint32_t N_ROWS_C = N_COLS_H;
    constexpr uint32_t N_COLS_C = N_COLS_G;

    // A  : Augmented result matrix, A = [ G | I | H ] -> [ U | Linv | F ]
    constexpr uint32_t N_ROWS_A = N_ROWS_G;
    constexpr uint32_t N_COLS_A = N_COLS_G + N_COLS_I + N_COLS_H;

    static_assert((N_THRDS_PER_GRP <= N_THREADS_PER_WARP), "Using co-operative groups");
    static_assert((0 == N_BS_ANTS % N_THRDS_PER_GRP) && (N_BS_ANTS >= N_THRDS_PER_GRP), "Expect BS antenna to be a multiple of thread group size");
    static_assert((0 == N_THRDS_PER_GRP % N_LAYERS) && (N_THRDS_PER_GRP >= N_LAYERS), "Expect thread group size to be a multiple of layer count"); 

    thread_block const& thisThrdBlk = this_thread_block();

    // Co-operative thread groups used in computation of inner products
    thread_block_tile<N_THRDS_PER_GRP> const& thrdGrp = tiled_partition<N_THRDS_PER_GRP>(thisThrdBlk);

    // G is Hermitian symmetric i.e. only the upper or lower diagonal elements need to be computed
    constexpr uint32_t N_TRI_ELEMS_G = N_ROWS_G * (N_ROWS_G + 1) / 2;

    // Iterations to compute one element of G. Each thread group computes the inner product needed to produce
    // one element of G
    constexpr uint32_t N_INNER_ITER_TO_COMP_G_ELEM = div_round_up_cexp(N_COLS_H, N_THRDS_PER_GRP);

    // Each thread group computes one element of G per outer loop iteration
    constexpr uint32_t N_OUTER_ITER_TO_COMP_G = div_round_up_cexp(N_TRI_ELEMS_G, N_THRD_GRPS_PER_THRD_BLK);

    // Each thrdGrp computes either part of or whole column of C
    constexpr uint32_t N_MAX_INNER_ITER_TO_COMP_C = N_ROWS_LINV;
    constexpr uint32_t N_THRD_GRPS_PER_C_COL_COMP = N_ROWS_C / N_THRDS_PER_GRP;
    constexpr uint32_t N_COLS_C_COMP_PER_THRD_BLK = N_THRD_GRPS_PER_THRD_BLK / N_THRD_GRPS_PER_C_COL_COMP;
    constexpr uint32_t N_OUTER_ITER_TO_COMP_C     = (N_COLS_C >= N_COLS_C_COMP_PER_THRD_BLK) ? (N_COLS_C / N_COLS_C_COMP_PER_THRD_BLK) : 1;

    
    // Number of iterations to compute Frobeius norm
    constexpr uint32_t N_INNER_ITER_TO_COMP_FNORM = N_ROWS_C / N_THRDS_PER_GRP;
    constexpr uint32_t N_OUTER_ITER_TO_COMP_FNORM = div_round_up_cexp(N_COLS_C, N_THRD_GRPS_PER_THRD_BLK);

    //--------------------------------------------------------------------------------------------------------
    // Compute indices used for element access
    const uint32_t THRD_IDX     = threadIdx.x; // thrdGrp.thread_rank()
    const uint32_t THRD_GRP_IDX = threadIdx.y;
    const uint32_t THRD_ABS_IDX = (threadIdx.y * blockDim.x) + threadIdx.x;

    const uint32_t PRB_IDX = blockIdx.x;

    const uint32_t ROW_IDX_R = THRD_ABS_IDX % N_ROWS_R;

    const uint32_t ROW_IDX_I = THRD_ABS_IDX % N_ROWS_I;
    const uint32_t COL_IDX_I = THRD_ABS_IDX / N_ROWS_I;

    // const uint32_t ROW_IDX_G = THRD_ABS_IDX % N_ROWS_G;
    // const uint32_t COL_IDX_G = THRD_ABS_IDX / N_ROWS_G; // COL_IDX_G needs a bounds check (since N_THRDS_X > # of G elements)

    // const uint32_t ROW_IDX_C = THRD_ABS_IDX % N_ROWS_C;
    // const uint32_t COL_IDX_C = THRD_ABS_IDX / N_ROWS_C;

    //--------------------------------------------------------------------------------------------------------
    // Shared memory allocation
    // H[N_TONES_PER_ITER*N_INST]

    // Shared memory contents as processing progresses:
    // A = [ G | I | H ] -> [ U | Linv | F ]

    constexpr uint32_t N_SMEM_R_ELEMS = N_ROWS_R;
    constexpr uint32_t N_SMEM_A_ELEMS = (N_ROWS_A + 1) * N_COLS_A; // (N_ROWS_A + 1) for SMEM padding to avoid bank conflicts
    constexpr uint32_t N_SMEM_C_ELEMS = (N_ROWS_C + 1) * N_COLS_C; // (N_ROWS_C + 1) for SMEM padding to avoid bank conflicts

    typedef typename complex_from_scalar<TCompute>::type    TComplexCompute;
    typedef typename complex_from_scalar<TStorageIn>::type  TComplexStorageIn;
    typedef typename complex_from_scalar<TStorageOut>::type TComplexStorageOut;

    __shared__ TCompute smemBlkR[N_SMEM_R_ELEMS];
    __shared__ TComplexCompute smemBlkA[N_SMEM_A_ELEMS];
    __shared__ TComplexCompute smemBlkC[N_SMEM_C_ELEMS];
    __shared__ TCompute shCFrobeniusNorm;

    constexpr uint32_t            SMEM_START_OFFSET_R = 0;
    block_1D<TCompute*, N_ROWS_R> shR(&smemBlkR[SMEM_START_OFFSET_R]);

    constexpr uint32_t                                 SMEM_START_OFFSET_A = 0;
    block_2D<TComplexCompute*, N_ROWS_A + NLE, N_COLS_A> shA(&smemBlkA[SMEM_START_OFFSET_A]);

    // SMEM overlay: A with [ G | I | H ]
    const uint32_t                                     SMEM_START_OFFSET_G = SMEM_START_OFFSET_A;
    block_2D<TComplexCompute*, N_ROWS_G + NLE, N_COLS_G> shG(&smemBlkA[SMEM_START_OFFSET_G]);

    const uint32_t                                     SMEM_START_OFFSET_I = SMEM_START_OFFSET_G + shG.num_elem();
    block_2D<TComplexCompute*, N_ROWS_I + NLE, N_COLS_I> shI(&smemBlkA[SMEM_START_OFFSET_I]);

    const uint32_t                                     SMEM_START_OFFSET_H = SMEM_START_OFFSET_I + shI.num_elem();
    block_2D<TComplexCompute*, N_ROWS_H + NLE, N_COLS_H> shH(&smemBlkA[SMEM_START_OFFSET_H]);

    const uint32_t                                     SMEM_START_OFFSET_C = 0;
    block_2D<TComplexCompute*, N_ROWS_C + 1, N_COLS_C> shC(&smemBlkC[SMEM_START_OFFSET_C]);

    // SMEM overlay:
    // After LU - U replaces G, Linv replaces I and F replaces H
    auto& shU    = shG;
    auto& shLinv = shI;
    auto& shF    = shH;

    // Dinv overlays with R
    auto& shDinv = shR;

    //--------------------------------------------------------------------------------------------------------
    // Stage1: Load inputs

#ifdef ENABLE_DEBUG
    if(0 != blockIdx.x) return;
#endif

    cmplxMatLoadRowMjr<TStorageIn, TCompute, N_ROWS_H, N_COLS_H, NLE>(thisThrdBlk, PRB_IDX, tH, shH);

    if(THRD_ABS_IDX < N_ROWS_R)
    {
        shR(ROW_IDX_R) = type_convert<TCompute>(tLambda(ROW_IDX_R, PRB_IDX));
    }

    // Wait for loads to complete. Thread(s) processing an entry of H may not be the same ones loading it
    __syncthreads();

#ifdef ENABLE_DEBUG
    // H
    for(uint32_t i = 0; i < N_ROWS_H; ++i)
    {
        if(THRD_ABS_IDX < N_COLS_H)
            tDbg(i, THRD_ABS_IDX, PRB_IDX) = type_convert<TComplexStorageOut>(shH(i, THRD_ABS_IDX));
    }
#endif

    //---------------------------------------------------------------------------------------------------
    // Stage1: Compute the enhanced Gram matrix: G = (H*H' + R),  G - N_LAYERS x N_LAYERS
    uint32_t matGRowEndMrkr = N_COLS_G;
    uint32_t iGRow          = 0;
    uint32_t iGCol          = 0;
    uint32_t iGIdx          = 0;
    for(uint32_t i = 0; i < N_OUTER_ITER_TO_COMP_G; ++i)
    {
        // linear index, each thread group computes one element of G per outer loop iteration
        iGIdx = (i * N_THRD_GRPS_PER_THRD_BLK) + THRD_GRP_IDX;

        if(iGIdx >= N_TRI_ELEMS_G) break;

        // Since G is Hermitian symmetric, its sufficient if only the upper (or lower) triangular elements of
        // H*H' are computed
        // Convert linear index to row and column indices of the upper triangular elements of matrix G
        while((iGIdx + iGRow) >= matGRowEndMrkr)
        {
            matGRowEndMrkr += (N_COLS_G - iGRow);
            ++iGRow;
        }
        iGCol = N_COLS_G - (matGRowEndMrkr - iGIdx) + iGRow;

        // Compute G(iGRow,iGCol) via N_BS_ANTS x N_BS_ANTS inner product
        TComplexCompute G = cuGet<TComplexCompute>(0);
        for(uint32_t j = 0; j < N_INNER_ITER_TO_COMP_G_ELEM; ++j)
        {
            uint32_t        iElem = (j * N_THRDS_PER_GRP) + THRD_IDX;
            TComplexCompute prod  = cuCmul(shH(iGRow, iElem), cuConj(shH(iGCol, iElem)));
            G += thrdGrpAllReduceSum<N_THRDS_PER_GRP>(thrdGrp, prod);
        }

        if(0 == THRD_IDX)
        {
            if(iGRow != iGCol)
            {
                shG(iGCol, iGRow) = cuConj(G);
            }
            else
            {
                G.x += shR(iGRow);
            }
            shG(iGRow, iGCol) = G;

            // printf("G[%d][%d] = %f+j%f, linIdx %d, threadIdx (%d,%d), blockIdx.x %d, matGRowEndMrkr %d\n", iGRow, iGCol, cuReal(G), cuImag(G), iGIdx, threadIdx.x, threadIdx.y, blockIdx.x, matGRowEndMrkr);
        }
    }

    if(COL_IDX_I < N_COLS_I)
    {
        shI(ROW_IDX_I, COL_IDX_I) =
            (ROW_IDX_I != COL_IDX_I) ? cuGet<TComplexCompute>(0) : cuGet<TComplexCompute>(1);
    }

    // Wait for G matrix compute and I matrix init to complete
    __syncthreads();

#ifdef ENABLE_DEBUG
    // A0
    for(uint32_t i = 0; i < N_ROWS_A; ++i)
    {
        if(THRD_ABS_IDX < N_COLS_A)
            tDbg(i, THRD_ABS_IDX, PRB_IDX) = type_convert<TComplexStorageOut>(shA(i, THRD_ABS_IDX));
    }
#endif

    //---------------------------------------------------------------------------------------------------
    // Stage2: Perform joint LU factorization
    // A = [ G | I | H ] -> [ U | Linv | F ]
    // where U = L\G, Linv = L\I, F = L\H

    // bfc_mmse_coef_comp_kernel_v0: 
    // For Large layer count (e.g. 8, 16) thread block size >> # of columns of augmented matrix
    // (i.e. (N_THRDS_PER_GRP * N_LAYERS) >> (2*N_LAYERS + N_BS_ANTS)). Thus use parallel version of the
    // factorization algorithm to cut down iteration count and increase active threads during sub-matrix
    // updates
    // For small layer counts (e.g. 2, 4) thread block size >= # of columns of augmented matrix. Use iterative
    // version since the iteration count = N_ROWS_A = N_LAYERS is expected to be small and thread block size is 
    // not large relative to N_COLS_A

    ((2 != N_LAYERS) && (4 != N_LAYERS)) ? luFactorizeParallel_v1<TCompute, N_ROWS_A, N_COLS_A, NLE>(thisThrdBlk, shA) : 
      luFactorizeIter<TCompute, N_ROWS_A, N_COLS_A, NLE>(thisThrdBlk, shA);

#ifdef ENABLE_DEBUG
    // A1
    for(uint32_t i = 0; i < N_ROWS_A; ++i)
    {
        if(THRD_ABS_IDX < N_COLS_A)
            tDbg(i, THRD_ABS_IDX, PRB_IDX) = type_convert<TComplexStorageOut>(shA(i, THRD_ABS_IDX));
    }
#endif

    //---------------------------------------------------------------------------------------------------
    // Stage3: Multiply C = F'*(inv(D)*inv(L)), where D = I*(diag(U)), G - N_BS_ANTS x N_LAYERS

    // Compute inv(D)
    if(THRD_ABS_IDX < N_COLS_U)
    {
        shDinv(THRD_ABS_IDX) = cuGet<TCompute>(1) / cuReal(shU(THRD_ABS_IDX, THRD_ABS_IDX));
    }

    // Initialize matrix C Frobenius norm. Use a thread which was not used in above
    if(N_COLS_U == THRD_ABS_IDX)
    {
        shCFrobeniusNorm = cuGet<TCompute>(0);
    }

    __syncthreads();

    // Each column of C maybe computed by one or more thread groups, C_COL_COMP_THRD_GRP_IDX is common index
    // of threads computing a column of C
    const uint32_t C_COL_COMP_THRD_GRP_IDX = (THRD_GRP_IDX / N_THRD_GRPS_PER_C_COL_COMP);

    // Due to the nature of lower triangular multiply, some inner products are longer than others (fewer
    // columns of F' to be combined when multpilying with later columns of Linv than initial columns).
    // To balance workload on each thread group, the outerloop iterations are divvied up to multiply as many
    // initial columns of Linv as its later columns
    constexpr uint32_t N_HALF_OUTER_ITER_TO_COMP_C = (N_OUTER_ITER_TO_COMP_C >= 2) ? (N_OUTER_ITER_TO_COMP_C / 2) : 1;

    // Offset to start column to be computed by this thread group
    const uint32_t C_COL_COMP_OFFSET = C_COL_COMP_THRD_GRP_IDX * N_HALF_OUTER_ITER_TO_COMP_C;
    const uint32_t C_ROW_COMP_OFFSET = (THRD_GRP_IDX % N_THRD_GRPS_PER_C_COL_COMP) * N_THRDS_PER_GRP;
    const uint32_t C_ROW_IDX         = C_ROW_COMP_OFFSET + THRD_IDX;

    // #pragma unroll
    for(uint32_t i = 0; i < N_OUTER_ITER_TO_COMP_C; ++i)
    {
        // Column index
        int32_t iCCol = C_COL_COMP_OFFSET + (i % N_HALF_OUTER_ITER_TO_COMP_C);
        if(iCCol >= N_COLS_C) continue;  // this condition holds when there are excess threads than needed i.e. N_COLS_C_COMP_PER_THRD_BLK > N_COLS_C (e.g. 1 layer case)

        // Process initial columns of Linv for first half iterations and later columns for the last half iterations
        if(i >= N_HALF_OUTER_ITER_TO_COMP_C) iCCol = N_COLS_C - iCCol - 1;

        // Due to the nature of lower triangular multiply, number of accumulations needed depends on the
        // column of Linv being multiplied
        TComplexCompute C = cuGet<TComplexCompute>(0);
        for(uint32_t iElem = iCCol; iElem < N_MAX_INNER_ITER_TO_COMP_C; ++iElem)
        {
            // Multiply inv(D)*inv(L))
            TComplexCompute DinvLinv = shDinv(iElem) * shLinv(iElem, iCCol);

            // Multiply F'*(inv(D)*inv(L))
            C = cuCma(cuConj(shF(iElem, C_ROW_IDX)), DinvLinv, C);
        }
        // TCompute absCSqr = cuReal(cuConj(C) * C);
        // atomicAdd(&shCFrobeniusNorm, absCSqr);

        shC(C_ROW_IDX, iCCol) = C;
        // printf("C[%d][%d] = %f+j%f, frobNorm %f, threadIdx (%d,%d), blockIdx.x %d i %d C_COL_COMP_OFFSET %d N_HALF_OUTER_ITER_TO_COMP_C %d\n", C_ROW_IDX, iCCol, cuReal(C), cuImag(C), shCFrobeniusNorm, threadIdx.x, threadIdx.y, blockIdx.x, i, C_COL_COMP_OFFSET, N_HALF_OUTER_ITER_TO_COMP_C);
    }

    // Wait for C matrix compute to complete
    __syncthreads();

#ifdef ENABLE_DEBUG
    // Coefs pre-norm
    for(uint32_t i = 0; i < N_ROWS_C; ++i)
    {
        if(THRD_ABS_IDX < N_COLS_C)
            tDbg(i, THRD_ABS_IDX, PRB_IDX) = type_convert<TComplexStorageOut>(shC(i, THRD_ABS_IDX));
    }
#endif
    for(uint32_t i = 0; i < N_OUTER_ITER_TO_COMP_FNORM; ++i)
    {
        // Each thread group computes coefficient magnitude for one column of C per outer loop iteration
        uint32_t iColC = (i * N_THRD_GRPS_PER_THRD_BLK) + THRD_GRP_IDX;

        TCompute absCSqrSum = 0;
        for(uint32_t j = 0; j < N_INNER_ITER_TO_COMP_FNORM; ++j)
        {
            uint32_t iRowC = (j * N_THRDS_PER_GRP) + THRD_IDX;

            TCompute absCSqr = cuGet<TCompute>(0);
            if((iRowC < N_ROWS_C) && (iColC < N_COLS_C))
            {
                TComplexCompute C = shC(iRowC, iColC);
                absCSqr = cuReal(C) * cuReal(C) + cuImag(C) * cuImag(C);
            }
            absCSqrSum += thrdGrpAllReduceSum<N_THRDS_PER_GRP>(thrdGrp, absCSqr);
        }

        if(0 == THRD_IDX)
        {
            atomicAdd(&shCFrobeniusNorm, absCSqrSum);
        }
    }

    __syncthreads();

    // Frobenius norm compute
    if(0 == THRD_ABS_IDX)
    {
#ifdef ENABLE_DEBUG
        printf("FrobNorm[%d] before = %f\n", PRB_IDX, shCFrobeniusNorm);
#endif
        shCFrobeniusNorm = cuRSqrt<TCompute>(shCFrobeniusNorm);
#ifdef ENABLE_DEBUG
        printf("FrobNorm[%d] after = %f\n", PRB_IDX, shCFrobeniusNorm);
#endif
    }

    // Wait for Frobenius norm to be computed
    __syncthreads();

    //--------------------------------------------------------------------------------------------------------
    // Stage4: Write the result BFC coefficients C into device memory
    cmplxMatStore_v0<TStorageOut, TCompute, N_ROWS_C, N_COLS_C>(thisThrdBlk, PRB_IDX, shCFrobeniusNorm, shC, tCoef);

#ifdef ENABLE_DEBUG
    // C
    for(uint32_t i = 0; i < N_ROWS_C; ++i)
    {
        if(THRD_ABS_IDX < N_COLS_C)
            printf("C[%d][%d][%d] = %f+j%f\n", PRB_IDX, i, COL_IDX_C, shC(i, COL_IDX_C).x, shC(i, COL_IDX_C).y);
    }
#endif
}

template <typename TStorageIn,
          typename TStorageOut,
          typename TCompute,
          uint32_t N_BS_ANTS, // # of BS antenna (# of rows in H matrix)
          uint32_t N_LAYERS>  // # of layers (# of cols in H matrix)
void
bfc_mmse_coef_comp_kernel_launch(uint32_t           Nprb,
                                 const_tensor_pair& tH,
                                 const_tensor_pair& tLambda,
                                 tensor_pair&       tCoef,
                                 tensor_pair&       tDbg,
                                 cudaStream_t       strm)
{
    constexpr uint32_t N_THRDS_PER_GRP            = N_THREADS_PER_WARP;
    constexpr uint32_t N_THRD_GRPS_PER_THRD_BLK_1 = N_LAYERS/2; // Large layer count (e.g. 8,16)
    constexpr uint32_t N_THRD_GRPS_PER_THRD_BLK_2 = div_round_up_cexp(N_BS_ANTS+2*N_LAYERS, N_THRDS_PER_GRP); // Small layer count (e.g. 2,4)

    constexpr uint32_t N_THRD_GRPS_PER_THRD_BLK = (N_THRD_GRPS_PER_THRD_BLK_1 > N_THRD_GRPS_PER_THRD_BLK_2) ? N_THRD_GRPS_PER_THRD_BLK_1 : N_THRD_GRPS_PER_THRD_BLK_2;

    dim3 gridDim(Nprb);
    dim3 blockDim(N_THRDS_PER_GRP, N_THRD_GRPS_PER_THRD_BLK);

    typedef typename complex_from_scalar<TCompute>::type    TComplexCompute;
    typedef typename complex_from_scalar<TStorageIn>::type  TComplexStorageIn;
    typedef typename complex_from_scalar<TStorageOut>::type TComplexStorageOut;

    tensor_ref_v0<const TComplexStorageIn, 3> H(tH);
    tensor_ref_v0<const TStorageIn, 2>        Lambda(tLambda);
    tensor_ref_v0<TComplexStorageOut, 3>      C(tCoef);
    tensor_ref_v0<TComplexStorageOut, 4>      Dbg(tDbg);

    // For V100, max permitted shared memory capacity is 96KB

#if 0
    constexpr int32_t  N_ITER = N_THRD_BLK_TONES / N_TONES_PER_ITER;
    constexpr uint32_t N_INST = (1 == N_ITER) ? 1 : 2; // double buffering for pipelining
    constexpr uint32_t N_SMEM_ELEMS =
        (((N_BS_ANTS + 1) * N_LAYERS * N_INST) +
         ((N_LAYERS + 1) * (N_LAYERS + N_LAYERS + N_BS_ANTS))) *
            N_TONES_PER_ITER;

    int nShmemBytes    = N_SMEM_ELEMS * sizeof(TComplexCompute);
    int nMaxShmemBytes = nShmemBytes;
    cudaFuncSetAttribute(bfc_mmse_coef_comp_kernel_v0<TStorageIn, TStorageOut, TCompute, N_THRD_BLK_TONES, N_TONES_PER_ITER, N_BS_ANTS, N_LAYERS, NH>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         nMaxShmemBytes);
#else

    int nShmemBytes = 0;
#endif
    bfc_mmse_coef_comp_kernel_v0<TStorageIn, TStorageOut, TCompute, N_BS_ANTS, N_LAYERS, N_THRD_GRPS_PER_THRD_BLK, N_THRDS_PER_GRP>
        <<<gridDim, blockDim, nShmemBytes, strm>>>(H,
                                                   Lambda,
                                                   C,
                                                   Dbg);
}

template <typename TStorageIn, typename TStorageOut, typename TCompute>
void bfc_coef_comp_kernel_launch(uint32_t           nBSAnts,
                                 uint32_t           nLayers,
                                 uint32_t           Nprb,
                                 const_tensor_pair& tH,
                                 const_tensor_pair& tLambda,
                                 tensor_pair&       tCoef,
                                 tensor_pair&       tDbg,
                                 cudaStream_t       strm)
{
    if(64 == nBSAnts)
    {
        constexpr uint32_t N_BS_ANTS = 64; // # of BS antenna (# of rows in H matrix)
        switch(nLayers)
        {
        // nLayers == 16
        case 16:
        {
            constexpr uint32_t N_LAYERS = 16; // # of layers (# of cols in H matrix)
            bfc_mmse_coef_comp_kernel_launch<TStorageIn, TStorageOut, TCompute, N_BS_ANTS, N_LAYERS>(Nprb,
                                                                                                     tH,
                                                                                                     tLambda,
                                                                                                     tCoef,
                                                                                                     tDbg,
                                                                                                     strm);
            break;
        }
        // nLayers == 8
        case 8:
        {
            constexpr uint32_t N_LAYERS = 8; // # of layers (# of cols in H matrix)
            bfc_mmse_coef_comp_kernel_launch<TStorageIn, TStorageOut, TCompute, N_BS_ANTS, N_LAYERS>(Nprb,
                                                                                                     tH,
                                                                                                     tLambda,
                                                                                                     tCoef,
                                                                                                     tDbg,
                                                                                                     strm);
            break;
        }        
        // nLayers == 4
        case 4:
        {
            constexpr uint32_t N_LAYERS = 4; // # of layers (# of cols in H matrix)
            bfc_mmse_coef_comp_kernel_launch<TStorageIn, TStorageOut, TCompute, N_BS_ANTS, N_LAYERS>(Nprb,
                                                                                                     tH,
                                                                                                     tLambda,
                                                                                                     tCoef,
                                                                                                     tDbg,
                                                                                                     strm);
            break;
        }
        // nLayers == 2
        case 2:
        {
            constexpr uint32_t N_LAYERS = 2; // # of layers (# of cols in H matrix)
            bfc_mmse_coef_comp_kernel_launch<TStorageIn, TStorageOut, TCompute, N_BS_ANTS, N_LAYERS>(Nprb,
                                                                                                     tH,
                                                                                                     tLambda,
                                                                                                     tCoef,
                                                                                                     tDbg,
                                                                                                     strm);
            break;
        }        
        default:
        {
            NVLOGE_FMT(NVLOG_BFW, AERIAL_CUPHY_EVENT, "{}: No kernel available to launch with requested configuration: nBSAnts {} nLayers {}", 
                       __FUNCTION__, nBSAnts, nLayers);
            break;
        }
        }
    }
    // nBSAnts = 32
    else if(32 == nBSAnts)
    {
        constexpr uint32_t N_BS_ANTS = 32; // # of BS antenna (# of rows in H matrix)
        switch(nLayers)
        {
        // nLayers == 8
        case 8:
        {
            constexpr uint32_t N_LAYERS = 8; // # of layers (# of cols in H matrix)
            bfc_mmse_coef_comp_kernel_launch<TStorageIn, TStorageOut, TCompute, N_BS_ANTS, N_LAYERS>(Nprb,
                                                                                                     tH,
                                                                                                     tLambda,
                                                                                                     tCoef,
                                                                                                     tDbg,
                                                                                                     strm);
            break;
        }
        // nLayers == 4
        case 4:
        {
            constexpr uint32_t N_LAYERS = 4; // # of layers (# of cols in H matrix)
            bfc_mmse_coef_comp_kernel_launch<TStorageIn, TStorageOut, TCompute, N_BS_ANTS, N_LAYERS>(Nprb,
                                                                                                     tH,
                                                                                                     tLambda,
                                                                                                     tCoef,
                                                                                                     tDbg,
                                                                                                     strm);
            break;
        }
        // nLayers == 2
        case 2:
        {
            constexpr uint32_t N_LAYERS = 2; // # of layers (# of cols in H matrix)
            bfc_mmse_coef_comp_kernel_launch<TStorageIn, TStorageOut, TCompute, N_BS_ANTS, N_LAYERS>(Nprb,
                                                                                                     tH,
                                                                                                     tLambda,
                                                                                                     tCoef,
                                                                                                     tDbg,
                                                                                                     strm);
            break;
        }
        // nLayers == 1
        case 1:
        {
            constexpr uint32_t N_LAYERS = 1; // # of layers (# of cols in H matrix)
            bfc_mmse_coef_comp_kernel_launch<TStorageIn, TStorageOut, TCompute, N_BS_ANTS, N_LAYERS>(Nprb,
                                                                                                     tH,
                                                                                                     tLambda,
                                                                                                     tCoef,
                                                                                                     tDbg,
                                                                                                     strm);
            break;
        }        
        default:
        {
            NVLOGE_FMT(NVLOG_BFW, AERIAL_CUPHY_EVENT, "{}: No kernel available to launch with requested configuration: nBSAnts {} nLayers {}", 
            __FUNCTION__, nBSAnts, nLayers);
            break;
        }
        }
    }
    else
    {
        NVLOGE_FMT(NVLOG_BFW, AERIAL_CUPHY_EVENT, "{}: No kernel available to launch with requested configuration: nBSAnts {} nLayers {}", 
            __FUNCTION__, nBSAnts, nLayers);
    }
}

void bfcCoefCompute(uint32_t           nBSAnts,
                    uint32_t           nLayers,
                    uint32_t           Nprb,
                    const_tensor_pair& tH,
                    const_tensor_pair& tLambda,
                    tensor_pair&       tCoef,
                    tensor_pair&       tDbg,
                    cudaStream_t       strm)
{
#ifdef ENABLE_DEBUG
    NVLOGD_FMT(NVLOG_BFW, AERIAL_CUPHY_EVENT, "{}() begin", __FUNCTION__);
#endif
    using TCompute = float;
    if(CUPHY_C_32F == tH.first.get().type())
    {
        using TStorageIn = scalar_from_complex<data_type_traits<CUPHY_C_32F>::type>::type;
        if(CUPHY_C_32F == tCoef.first.get().type())
        {
            using TStorageOut = scalar_from_complex<data_type_traits<CUPHY_C_32F>::type>::type;
            bfc_coef_comp_kernel_launch<TStorageIn, TStorageOut, TCompute>(nBSAnts,
                                                                           nLayers,
                                                                           Nprb,
                                                                           tH,
                                                                           tLambda,
                                                                           tCoef,
                                                                           tDbg,
                                                                           strm);
        }
        else if(CUPHY_C_16F == tCoef.first.get().type())
        {
            using TStorageOut = scalar_from_complex<data_type_traits<CUPHY_C_16F>::type>::type;
            bfc_coef_comp_kernel_launch<TStorageIn, TStorageOut, TCompute>(nBSAnts,
                                                                           nLayers,
                                                                           Nprb,
                                                                           tH,
                                                                           tLambda,
                                                                           tCoef,
                                                                           tDbg,
                                                                           strm);
        }
        else
        {
            NVLOGE_FMT(NVLOG_BFW, AERIAL_CUPHY_EVENT, "{}: No kernel available to launch with requested data type {}/{}", 
                       __FUNCTION__, +tH.first.get().type(), +tCoef.first.get().type());
        }
    }
    else if((CUPHY_C_16F == tH.first.get().type()) && (CUPHY_C_16F == tCoef.first.get().type()))
    {
        using TStorageIn  = scalar_from_complex<data_type_traits<CUPHY_C_16F>::type>::type;
        using TStorageOut = scalar_from_complex<data_type_traits<CUPHY_C_16F>::type>::type;
        bfc_coef_comp_kernel_launch<TStorageIn, TStorageOut, TCompute>(nBSAnts,
                                                                       nLayers,
                                                                       Nprb,
                                                                       tH,
                                                                       tLambda,
                                                                       tCoef,
                                                                       tDbg,
                                                                       strm);
    }
    else
    {
        NVLOGE_FMT(NVLOG_BFW, AERIAL_CUPHY_EVENT, "{}: No kernel available to launch with requested data type {}/{}", 
            __FUNCTION__, +tH.first.get().type(), +tCoef.first.get().type());
    }
#ifdef ENABLE_DEBUG
        NVLOGD_FMT(NVLOG_BFW, AERIAL_CUPHY_EVENT, "{}() end", __FUNCTION__);
#endif
}

/* New Beamforming API -------------------------------------------------------------------------------------------------------------- */

template <typename TStorageIn,
          typename TCompute,
          uint32_t N_ROWS_H_MAT,
          uint32_t N_COLS_H_MAT,
	  uint32_t NLE >
__device__ __forceinline__ void srsChEstLoadRowMjr(thread_block const&                  thisThrdBlk,
                                                   uint32_t                             iPrbGrp,
                                                   bfwCoefCompKernelBfLayerPrm_t const* pBfLayerPrm,
                                                   block_2D<typename complex_from_scalar<TCompute>::type*, N_ROWS_H_MAT+NLE, N_COLS_H_MAT>& shSrsChEst)
{
    typedef typename complex_from_scalar<TStorageIn>::type TComplexStorageIn;
    typedef typename complex_from_scalar<TCompute>::type   TComplexCompute;

    const uint32_t     N_THRDS                 = thisThrdBlk.size();
    const uint32_t     THRD_IDX                = thisThrdBlk.thread_rank();
    constexpr uint32_t N_MAT_ELEMS_TO_RD       = N_ROWS_H_MAT * N_COLS_H_MAT;
    constexpr uint16_t MAX_PRB_GRP_IDX         = CUPHY_BFW_N_MAX_PRB_GRPS - 1; // Largest PRB Group Index
    const uint32_t     N_MAT_ELEMS_RD_PER_ITER = (N_MAT_ELEMS_TO_RD > N_THRDS) ? N_THRDS : N_MAT_ELEMS_TO_RD;
    const uint32_t     N_ITER_TO_RD_MAT        = div_round_up(N_MAT_ELEMS_TO_RD, N_MAT_ELEMS_RD_PER_ITER);

    for(uint32_t i = 0; i < N_ITER_TO_RD_MAT; ++i)
    {
        uint32_t matElemIdx = ((i * N_MAT_ELEMS_RD_PER_ITER) + THRD_IDX);
        uint32_t iCol       = matElemIdx % N_COLS_H_MAT;
        uint32_t iRow       = matElemIdx / N_COLS_H_MAT;
        // Not all threads may participate in the last iteration
        if(matElemIdx < N_MAT_ELEMS_TO_RD)
        {
            bfwCoefCompKernelBfLayerPrm_t const& bfLayerPrm = pBfLayerPrm[iRow];
            uint32_t iSrcPrbGrp = bfLayerPrm.startPrbGrpOffset + bfLayerPrm.prbGrpStride * iPrbGrp;
            uint8_t iSrcRow     = bfLayerPrm.ueLayerIdx;

            // Gather SRS channel estimates from different layers of potentially different UEs
            tensor_ref<const TComplexStorageIn> tSrsChEst(bfLayerPrm.tInfoSrsChEst.pAddr, bfLayerPrm.tInfoSrsChEst.strides); // (N_PRB_GRP, N_GNB_ANTS, N_LAYERS)
            
            // Clamp srsPrgIdx to valid range and compute iSrcPrbGrp
            const uint16_t srsPrgIdx     = iSrcPrbGrp + bfLayerPrm.chEstInfoStartPrbGrp;
            const uint16_t startValidPrg = bfLayerPrm.startValidPrg;
            const uint16_t endValidPrg   = startValidPrg + bfLayerPrm.nValidPrg - 1;
            iSrcPrbGrp = max(startValidPrg, min(srsPrgIdx, endValidPrg)) - bfLayerPrm.chEstInfoStartPrbGrp;

            // The last PRB group will not be populated with group size of 2 since SRS will not occupy 
            // the final odd PRB so we copy the previous channel estimate
            // Use __ldg() for read-only cached access (texture cache path) - better for scattered reads
            const TComplexStorageIn* srcPtr = &tSrsChEst(iSrcPrbGrp-(iSrcPrbGrp==MAX_PRB_GRP_IDX), iCol, iSrcRow);
            shSrsChEst(iRow, iCol) = type_convert<TComplexCompute>(__ldg(srcPtr));

#ifdef ENABLE_DEBUG
            printf("Mat[%d][%d][%d] = %f+j%f\n", iPrbGrp, iRow, iCol, shSrsChEst(iRow, iCol).x, shSrsChEst(iRow, iCol).y);
#endif
        }
    }
}

// BeamFormingWeight (BFW) coefficient computation kernel
// {N_LAYERS, N_BS_ANTS} = {16,64}
// Inputs and outputs assumed to be column major
// dimBlock: (N_THREADS_PER_WARP, N_LAYERS)
// dimGrid : (Nprb)
template <typename TStorageIn,
          typename TStorageOut,
          typename TCompute,
          uint32_t N_BS_ANTS, // # of BS antenna (# of rows in H matrix)
          uint32_t N_LAYERS,  // # of layers (# of cols in H matrix)
          uint32_t N_THRD_GRPS_PER_THRD_BLK,
          uint32_t N_THRDS_PER_GRP>
__global__ void //__launch_bounds__(N_THRDS_PER_GRP * N_THRD_GRPS_PER_THRD_BLK, 2048 / (N_THRDS_PER_GRP * N_THRD_GRPS_PER_THRD_BLK))
bfwMmseCoefCompKernel_v1(bfwCoefCompStatDescr_t* pStatDescr, bfwCoefCompDynDescr_t* pDynDescr)
{
    //--------------------------------------------------------------------------------------------------------
    // Setup local parameters based on descriptor
    bfwCoefCompStatDescr_t& statDescr = *(pStatDescr);

    // Early exit check
    // The grid is sized to process the max # of PRBs in a given heterogenous config. Exit if the PRB to be
    // processed by this thread block does not exist in the UE group
    // PRB index processed by this thread
    const uint32_t PRB_GRP_IDX = blockIdx.x;
    const uint32_t UE_GRP_IDX  = statDescr.pHetCfgUeGrpMap[pDynDescr->hetCfgIdx][blockIdx.y];

    bfwCoefCompKernelUeGrpPrm_t& ueGrpPrms = statDescr.pKernelUeGrpPrms[UE_GRP_IDX];
    bfwCoefCompKernelBfLayerPrm_t* pBfLayerPrms = ueGrpPrms.pBfLayerPrmGpu;
    
    uint8_t bfwPowerNormAlg_selector = statDescr.bfwPowerNormAlg_selector;
    //printf("bfwPowerNormAlg_selector=%d\n", bfwPowerNormAlg_selector);

    const uint16_t nPrbGrp = ueGrpPrms.nPrbGrp;
    if(PRB_GRP_IDX >= nPrbGrp) return;

    // The shared memory data is stored column-major where the number of rows is N_LAYERS.  The column stride
    // had been N_LAYERS+1, likely to minimize shared memory bank conflicts.  However, for the case of
    // N_LAYERS == 15, the shared memory bank conflict was not avoided at all.
    // To remedy this a constant NLE is created, which is 1 when N_LAYERS is even.  This is then
    // used to pad the columns of shared memory matrix data, such that all strides are odd, and shared memory
    // bank conflicts are always avoided.
    // NLE = short for N_Layers is Even.
    constexpr uint32_t NLE = (N_LAYERS%2) ? 0 : 1;

    //--------------------------------------------------------------------------------------------------------
    typedef typename complex_from_scalar<TCompute>::type    TComplexCompute;
    typedef typename complex_from_scalar<TStorageIn>::type  TComplexStorageIn;
    typedef typename complex_from_scalar<TStorageOut>::type TComplexStorageOut;

    TCompute lambda = statDescr.lambda;

    // H is channel matrix
    // G is the enhanced Gram matrix
    // A is the augmented matrix, A = [ G | I | H ]

    //--------------------------------------------------------------------------------------------------------
    // Dimensions

    // H  : Channel matrix
    constexpr uint32_t N_ROWS_H = N_LAYERS;
    constexpr uint32_t N_COLS_H = N_BS_ANTS;

    // R  : Diagonal matrix (lambda) with per layer regularization coefficients
    constexpr uint32_t N_ROWS_R = N_LAYERS;
    // constexpr uint32_t N_COLS_R = N_LAYERS;

    // G  : Enhanced Gram matrix, G = H*H' + R
    constexpr uint32_t N_ROWS_G = N_LAYERS;
    constexpr uint32_t N_COLS_G = N_LAYERS;

    // I  : Identity matrix
    constexpr uint32_t N_ROWS_I = N_LAYERS;
    constexpr uint32_t N_COLS_I = N_LAYERS;

    // Linv: inverse lower trianuglar matrix in LU factorization
    constexpr uint32_t N_ROWS_LINV = N_LAYERS;

    // U  : Upper triangular matrix
    constexpr uint32_t N_COLS_U = N_COLS_G;

    // C  : MMSE coefficient matrix, C = H'*Ginv = H'*inv(H*H' + D)
    constexpr uint32_t N_ROWS_C = N_COLS_H;
    constexpr uint32_t N_COLS_C = N_COLS_G;

    // A  : Augmented result matrix, A = [ G | I | H ] -> [ U | Linv | F ]
    constexpr uint32_t N_ROWS_A = N_ROWS_G;
    constexpr uint32_t N_COLS_A = N_COLS_G + N_COLS_I + N_COLS_H;

    static_assert((N_THRDS_PER_GRP <= N_THREADS_PER_WARP), "Using co-operative groups");
    static_assert((0 == N_BS_ANTS % N_THRDS_PER_GRP) && (N_BS_ANTS >= N_THRDS_PER_GRP), "Expect BS antenna to be a multiple of thread group size");
    static_assert(N_THRDS_PER_GRP >= N_LAYERS, "number of threads per group must be more than number of layers");

    thread_block const& thisThrdBlk = this_thread_block();

    // Co-operative thread groups used in computation of inner products
    thread_block_tile<N_THRDS_PER_GRP> const& thrdGrp = tiled_partition<N_THRDS_PER_GRP>(thisThrdBlk);

    // G is Hermitian symmetric i.e. only the upper or lower diagonal elements need to be computed
    constexpr uint32_t N_TRI_ELEMS_G = N_ROWS_G * (N_ROWS_G + 1) / 2;

    // Iterations to compute one element of G. Each thread group computes the inner product needed to produce
    // one element of G
    constexpr uint32_t N_INNER_ITER_TO_COMP_G_ELEM = div_round_up_cexp(N_COLS_H, N_THRDS_PER_GRP);

    // Each thread group computes one element of G per outer loop iteration
    constexpr uint32_t N_OUTER_ITER_TO_COMP_G = div_round_up_cexp(N_TRI_ELEMS_G, N_THRD_GRPS_PER_THRD_BLK);

    //--------------------------------------------------------------------------------------------------------
    // Compute indices used for element access
    const uint32_t THRD_IDX     = threadIdx.x; // thrdGrp.thread_rank()
    const uint32_t THRD_GRP_IDX = threadIdx.y;
    const uint32_t THRD_ABS_IDX = (threadIdx.y * blockDim.x) + threadIdx.x;

    const uint32_t PRB_IDX = blockIdx.x;

    const uint32_t ROW_IDX_R = THRD_ABS_IDX % N_ROWS_R;

    const uint32_t ROW_IDX_I = THRD_ABS_IDX % N_ROWS_I;
    const uint32_t COL_IDX_I = THRD_ABS_IDX / N_ROWS_I;

    //--------------------------------------------------------------------------------------------------------
    // Shared memory allocation (all dynamic to avoid per-instantiation static memory bloat)
    // H[N_TONES_PER_ITER*N_INST]

    // Shared memory contents as processing progresses:
    // A = [ G | I | H ] -> [ U | Linv | F ]

    constexpr uint32_t N_SMEM_R_ELEMS = N_ROWS_R;
    constexpr uint32_t N_SMEM_A_ELEMS = (N_ROWS_A + 1) * N_COLS_A; // (N_ROWS_A + 1) for SMEM padding to avoid bank conflicts
    constexpr uint32_t N_SMEM_C_ELEMS = (N_ROWS_C + 1) * N_COLS_C; // (N_ROWS_C + 1) for SMEM padding to avoid bank conflicts

    typedef typename complex_from_scalar<TCompute>::type    TComplexCompute;
    typedef typename complex_from_scalar<TStorageIn>::type  TComplexStorageIn;
    typedef typename complex_from_scalar<TStorageOut>::type TComplexStorageOut;

    // Dynamic shared memory layout:
    // [smemBlkA: N_SMEM_A_ELEMS * TComplexCompute]
    // [smemBlkC: N_SMEM_C_ELEMS * TComplexCompute]
    // [smemBlkR: N_SMEM_R_ELEMS * TCompute]
    // [layerScalingFactors: N_SMEM_R_ELEMS * TCompute]
    // [antEnergies: N_ROWS_C * TCompute]
    // [shCFrobeniusNorm: 1 * TCompute]
    extern __shared__ char sharedMemory[];

    // Compute offsets with proper alignment (TComplexCompute is 8 bytes for float2)
    constexpr size_t OFFSET_A = 0;
    constexpr size_t OFFSET_C = OFFSET_A + N_SMEM_A_ELEMS * sizeof(TComplexCompute);
    constexpr size_t OFFSET_R = OFFSET_C + N_SMEM_C_ELEMS * sizeof(TComplexCompute);
    constexpr size_t OFFSET_LAYER_SCALING = OFFSET_R + N_SMEM_R_ELEMS * sizeof(TCompute);
    constexpr size_t OFFSET_ANT_ENERGIES = OFFSET_LAYER_SCALING + N_SMEM_R_ELEMS * sizeof(TCompute);
    constexpr size_t OFFSET_FROBENIUS = OFFSET_ANT_ENERGIES + N_ROWS_C * sizeof(TCompute);

    TComplexCompute* smemBlkA = reinterpret_cast<TComplexCompute*>(sharedMemory + OFFSET_A);
    TComplexCompute* smemBlkC = reinterpret_cast<TComplexCompute*>(sharedMemory + OFFSET_C);
    TCompute* smemBlkR = reinterpret_cast<TCompute*>(sharedMemory + OFFSET_R);
    TCompute* layerScalingFactors = reinterpret_cast<TCompute*>(sharedMemory + OFFSET_LAYER_SCALING);
    TCompute* antEnergies = reinterpret_cast<TCompute*>(sharedMemory + OFFSET_ANT_ENERGIES);
    TCompute& shCFrobeniusNorm = *reinterpret_cast<TCompute*>(sharedMemory + OFFSET_FROBENIUS);

    constexpr uint32_t            SMEM_START_OFFSET_R = 0;
    block_1D<TCompute*, N_ROWS_R> shR(&smemBlkR[SMEM_START_OFFSET_R]);

    constexpr uint32_t                                 SMEM_START_OFFSET_A = 0;
    block_2D<TComplexCompute*, N_ROWS_A + NLE, N_COLS_A> shA(&smemBlkA[SMEM_START_OFFSET_A]);

    // SMEM overlay: A with [ G | I | H ]
    const uint32_t                                     SMEM_START_OFFSET_G = SMEM_START_OFFSET_A;
    block_2D<TComplexCompute*, N_ROWS_G + NLE, N_COLS_G> shG(&smemBlkA[SMEM_START_OFFSET_G]);

    const uint32_t                                     SMEM_START_OFFSET_I = SMEM_START_OFFSET_G + shG.num_elem();
    block_2D<TComplexCompute*, N_ROWS_I + NLE, N_COLS_I> shI(&smemBlkA[SMEM_START_OFFSET_I]);

    const uint32_t                                     SMEM_START_OFFSET_H = SMEM_START_OFFSET_I + shI.num_elem();
    block_2D<TComplexCompute*, N_ROWS_H + NLE, N_COLS_H> shH(&smemBlkA[SMEM_START_OFFSET_H]);

    const uint32_t                                     SMEM_START_OFFSET_C = 0;
    block_2D<TComplexCompute*, N_ROWS_C + 1, N_COLS_C> shC(&smemBlkC[SMEM_START_OFFSET_C]);

    // SMEM overlay:
    // After LU - U replaces G, Linv replaces I and F replaces H
    auto& shU    = shG;
    auto& shLinv = shI;
    auto& shF    = shH;

    // Dinv overlays with R
    auto& shDinv = shR;

    //--------------------------------------------------------------------------------------------------------
    // Stage1: Load inputs

#ifdef ENABLE_DEBUG
    if(0 != blockIdx.x) return;
#endif

    srsChEstLoadRowMjr<TStorageIn, TCompute, N_ROWS_H, N_COLS_H, NLE>(thisThrdBlk, PRB_IDX, pBfLayerPrms, shH);

    if(THRD_ABS_IDX < N_ROWS_R)
    {
        shR(ROW_IDX_R) = lambda;
        layerScalingFactors[ROW_IDX_R] = cuGet<TCompute>(0);
    }
    
    if(THRD_ABS_IDX < N_ROWS_C)
    {
        antEnergies[THRD_ABS_IDX] = cuGet<TCompute>(0);
    }

    // Wait for loads to complete. Thread(s) processing an entry of H may not be the same ones loading it
    thisThrdBlk.sync();

#ifdef ENABLE_DEBUG
    // H
    for(uint32_t i = 0; i < N_ROWS_H; ++i)
    {
        if(THRD_ABS_IDX < N_COLS_H)
            tDbg(i, THRD_ABS_IDX, PRB_IDX) = type_convert<TComplexStorageOut>(shH(i, THRD_ABS_IDX));
    }
#endif

    //---------------------------------------------------------------------------------------------------
    // Stage1: Compute the enhanced Gram matrix: G = (H*H' + R),  G - N_LAYERS x N_LAYERS

    if ( N_LAYERS < 4 ) {

        // For the case of N_LAYERS < 4, where the max number of independent elements of G is 6, it makes sense to use
        // warp-reductions to compute the elements of G.  These involve synchronizations at every step of the
        // reduction, but there are not enough elements of G to parallelize over otherwise.
        
        // Note that the parallel reduction naturally preserves the accuracy of the sum relatively well.
        
        uint32_t matGRowEndMrkr = N_COLS_G;
        uint32_t iGRow          = 0;
        uint32_t iGCol          = 0;
        uint32_t iGIdx          = 0;
        
        for(uint32_t i = 0; i < N_OUTER_ITER_TO_COMP_G; ++i)
        {
            // linear index, each thread group computes one element of G per outer loop iteration
            iGIdx = (i * N_THRD_GRPS_PER_THRD_BLK) + THRD_GRP_IDX;
            
            if(iGIdx >= N_TRI_ELEMS_G) break;
            
            // Since G is Hermitian, its sufficient if only the upper (or lower) triangular elements of G are computed.
            // Convert linear index to row and column indices of the upper triangular elements of matrix G
            while((iGIdx + iGRow) >= matGRowEndMrkr)
            {
             	matGRowEndMrkr += (N_COLS_G - iGRow);
                ++iGRow;
            }
            iGCol = N_COLS_G - (matGRowEndMrkr - iGIdx) + iGRow;

            // Compute G(iGRow,iGCol) via N_BS_ANTS x N_BS_ANTS inner product
            TComplexCompute G = cuGet<TComplexCompute>(0);
            for(uint32_t j = 0; j < N_INNER_ITER_TO_COMP_G_ELEM; ++j)
            {
             	uint32_t        iElem = (j * N_THRDS_PER_GRP) + THRD_IDX;
                TComplexCompute prod  = cuCmul(shH(iGRow, iElem), cuConj(shH(iGCol, iElem)));
                G += thrdGrpAllReduceSum<N_THRDS_PER_GRP>(thrdGrp, prod);
            }

            if(0 == THRD_IDX)
            {
                if(iGRow != iGCol)
                {
                    shG(iGCol, iGRow) = cuConj(G);
                }
                else
                {
                    G.x += shR(iGRow);
                }
                shG(iGRow, iGCol) = G;
#ifdef ENABLE_DEBUG                
                printf("G[%d][%d] = %f+j%f, linIdx %d, threadIdx (%d,%d), blockIdx.x %d, matGRowEndMrkr %d\n", iGRow, iGCol, cuReal(G), cuImag(G), iGIdx, threadIdx.x, threadIdx.y, blockIdx.x, matGRowEndMrkr);
#endif                
            }
        }
        
    }
    else {

        // Single-pass Gram matrix computation: each thread computes one full element of G.
        // For small N_BS_ANTS (e.g., <= 64), the inner product over K=N_BS_ANTS provides
        // sufficient work per thread for latency hiding, eliminating the need for batching,
        // atomicAdds, and extra barriers.

        TComplexCompute g;
        
        // Lower triangular element count (including diagonal)
        constexpr uint32_t HALF_G_ELEMS = (N_LAYERS*N_LAYERS + N_LAYERS)/2;

        // Each thread maps to one element of G; only first HALF_G_ELEMS threads participate
        const uint32_t i = THRD_ABS_IDX;
        const bool participates = (i < HALF_G_ELEMS);

        // Compute (irow, icol) from linear index i
        // For lower triangular: i = irow*(irow+1)/2 + icol, solve for irow, icol
        uint32_t irow = 0, icol = 0;
        if (participates) {
            irow = (uint32_t)((sqrtf(1.0f + 8.0f*i) - 1.0f) * 0.5f);
            icol = i - (irow*irow + irow)/2;
        }

        // Single-pass: each thread computes full k=0..N_BS_ANTS inner product
        g = cuGet<TComplexCompute>(0);
        if (participates) {
            for (int k = 0; k < N_BS_ANTS; ++k) {
                g = cuCma(shH(irow,k), cuConj(shH(icol,k)), g);
            }
            
            // Direct store - no atomics needed since each element has exactly one writer
            // For diagonal, imaginary part is mathematically zero (Hermitian property)
            shG(irow, icol) = g;
        }

        // Sync before upper triangular mirror and diagonal update
        thisThrdBlk.sync();

        // Mirror to upper triangular and add regularization to diagonal
        if (participates) {
            if (irow != icol) {
                shG(icol, irow) = cuConj(g);
            } else {
                g.x += shR(irow);
                shG(irow, icol) = g;
            }
        }

    }

    if(COL_IDX_I < N_COLS_I)
    {
        shI(ROW_IDX_I, COL_IDX_I) =
            (ROW_IDX_I != COL_IDX_I) ? cuGet<TComplexCompute>(0) : cuGet<TComplexCompute>(1);
    }

    // Wait for G matrix compute and I matrix init to complete
    thisThrdBlk.sync();

#ifdef ENABLE_DEBUG
    // A0
    for(uint32_t i = 0; i < N_ROWS_A; ++i)
    {
        if(THRD_ABS_IDX < N_COLS_A)
            tDbg(i, THRD_ABS_IDX, PRB_IDX) = type_convert<TComplexStorageOut>(shA(i, THRD_ABS_IDX));
    }
#endif

    //---------------------------------------------------------------------------------------------------
    // Stage2: Perform joint LU factorization
    // A = [ G | I | H ] -> [ U | Linv | F ]
    // where U = L\G, Linv = L\I, F = L\H

    // bfwMmseCoefCompKernel_v1: 
    // For Large layer count (e.g. 8, 16) thread block size >> # of columns of augmented matrix
    // (i.e. (N_THRDS_PER_GRP * N_LAYERS) >> (2*N_LAYERS + N_BS_ANTS)). Thus use parallel version of the
    // factorization algorithm to cut down iteration count and increase active threads during sub-matrix
    // updates
    // For small layer counts (e.g. 2, 4) thread block size >= # of columns of augmented matrix. Use iterative
    // version since the iteration count = N_ROWS_A = N_LAYERS is expected to be small and thread block size is 
    // not large relative to N_COLS_A


    /*

      Experimentation showed the luFactorizeParallel to never outperform luFactorizeIter.
      This had a big impact on performance.

    ((2 != N_LAYERS) && (4 != N_LAYERS)) ? luFactorizeParallel_v1<TCompute, N_ROWS_A, N_COLS_A>(thisThrdBlk, shA) : 
                                           luFactorizeIter<TCompute, N_ROWS_A, N_COLS_A>(thisThrdBlk, shA);
    */
    
    luFactorizeIter<TCompute, N_ROWS_A, N_COLS_A, NLE>(thisThrdBlk, shA);

#ifdef ENABLE_DEBUG
    // A1
    for(uint32_t i = 0; i < N_ROWS_A; ++i)
    {
        if(THRD_ABS_IDX < N_COLS_A)
            tDbg(i, THRD_ABS_IDX, PRB_IDX) = type_convert<TComplexStorageOut>(shA(i, THRD_ABS_IDX));
    }
#endif

    //---------------------------------------------------------------------------------------------------
    // Stage3: Multiply C = F'*(inv(D)*inv(L)), where D = I*(diag(U)), G - N_BS_ANTS x N_LAYERS


    // Compute inv(D)
    // Investigated applying shDiv to shLinv here vs. in the inv(D)*inv(L) loop.  Not exactly clear why, but his is faster.
    if(THRD_ABS_IDX < N_COLS_U)
    {
        shDinv(THRD_ABS_IDX) = cuGet<TCompute>(1) / cuReal(shU(THRD_ABS_IDX, THRD_ABS_IDX));
    }

    // Initialize matrix C Frobenius norm. Use a thread which was not used in above
    if(N_COLS_U == THRD_ABS_IDX)
    {
        shCFrobeniusNorm = cuGet<TCompute>(0);
    }
    
    thisThrdBlk.sync();

    TCompute absCSqr = cuGet<TCompute>(0);
      
    // the previous implementation of inv(D)*inv(L) unnecessarily repeated computations when gridDim.x was odd.
    // so this removes all of the inner and outer loops and replaces them with a block-stride loop.
    // This might be faster due to better resource usage and lower loop overhead.
    
    for ( uint32_t idx=THRD_ABS_IDX; idx<N_ROWS_C*N_COLS_C; idx+=blockDim.x*blockDim.y ) {

      // Compute the (row,col) for this index.
      uint32_t icol = idx/N_ROWS_C;
      uint32_t irow = idx - icol*N_ROWS_C;                              // avoid the use of %modulus
      
      TComplexCompute C = cuGet <TComplexCompute>(0);
      
      for ( uint32_t iElem=icol; iElem< N_ROWS_LINV; ++iElem ) 
      {
	        // Multiply inv(D)*inv(L)
	        TComplexCompute DinvLinv = shDinv(iElem) * shLinv(iElem,icol);
	        // Multiply F'*(inv(D)*inv(L))
          C = cuCma( cuConj(shF(iElem,irow)), DinvLinv, C );
      }
      
      // Store the computed element of C
      shC( irow, icol ) = C;
      
      TCompute absCSqVal = cuReal(C) * cuReal(C) + cuImag(C) * cuImag(C);
      
      if(bfwPowerNormAlg_selector==0)
      {
          // Collect inputs for the FrobeniusNorm in place - while we have C.
          // Sum here so we only need to do the thrdGrpAllReduceSum once.
          absCSqr += absCSqVal;
      }
      else if(bfwPowerNormAlg_selector==1)
      {
          // All threads in a warp work on the same column (icol) per iteration.
          // Using warp-level reduction to sum values, then single atomicAdd per warp.
          TCompute warpSum = reduce(thrdGrp, absCSqVal, plus<TCompute>());
          if (thrdGrp.thread_rank() == 0)
          {
              atomicAdd(&layerScalingFactors[icol], warpSum);
          }
      }
    }
    
    if(bfwPowerNormAlg_selector==0)
    {
        // Group (warp) - reduceSum absCSqr in place
        absCSqr = thrdGrpAllReduceSum<N_THRDS_PER_GRP>(thrdGrp,absCSqr);
        
        if ( 0 == THRD_IDX ) {
          // collect the inputs to the FrobeniusNorm from each group
          atomicAdd( &shCFrobeniusNorm, absCSqr );
        }
        
        // Wait for all the Frobenius terms to be collected
        thisThrdBlk.sync();
        
        // Compute the reciprocal of the Frobenius norm
        if(0 == THRD_ABS_IDX  ) {
          shCFrobeniusNorm = cuRSqrt<TCompute>( shCFrobeniusNorm );
        }
        
        // Wait for the reciprocal of the Frobenius norm to be computed
        thisThrdBlk.sync();
    
#ifdef ENABLE_DEBUG
        // Coefs pre-norm
        for(uint32_t i = 0; i < N_ROWS_C; ++i)
        {
            if(THRD_ABS_IDX < N_COLS_C)
                tDbg(i, THRD_ABS_IDX, PRB_IDX) = type_convert<TComplexStorageOut>(shC(i, THRD_ABS_IDX));
        }
#endif
    }
    else if(bfwPowerNormAlg_selector==1)
    {
        thisThrdBlk.sync();
        
        if(THRD_ABS_IDX < N_ROWS_R)
        {
            layerScalingFactors[ROW_IDX_R] = rsqrtf(layerScalingFactors[ROW_IDX_R]);
        }
        
        thisThrdBlk.sync();


        // Compute per-antenna energy ==================================================================================
        //
        // Goal: reduce atomicAdd traffic when accumulating antEnergies[irow].
        //
        // Original behavior:
        //   For each visited (irow, icol), compute |shC(irow,icol)|^2 and atomicAdd into antEnergies[irow].
        //
        // Optimization idea:
        //   Each thread processes multiple (irow,icol) elements across the loop. Instead of doing an atomicAdd
        //   every iteration, accumulate per-row energy in registers and perform only 1-2 atomicAdds per thread
        //   at the end.
        //
        // Why 1-2 rows per thread is safe here:
        //   The loop advances idx by STRIDE = blockDim.x * blockDim.y. With the cuPHY BFW configuration
        //   (warp-sized groups), STRIDE is a multiple of 32. For N_ROWS_C (N_BS_ANTS) in {32, 64} this implies:
        //     ROW_SHIFT = STRIDE % N_ROWS_C is either:
        //       - 0  (thread always maps to the same row), or
        //       - N_ROWS_C/2 (thread alternates between two rows separated by N_ROWS_C/2).
        //
        // IMPORTANT: If we change the launch geometry or allow other N_ROWS_C values, this assumption may break
        //            and the two-accumulator scheme would be incorrect. The static_asserts below protect this.
        //
        // Examples:
        //   N_BS_ANTS=64, N_LAYERS=8  -> N_THRD_GRPS=4 (even) -> ROW_SHIFT=0  (single-row path)
        //   N_BS_ANTS=64, N_LAYERS=3  -> N_THRD_GRPS=3 (odd)  -> ROW_SHIFT=32 (two-row path)
        //   N_BS_ANTS=32, N_LAYERS=*  -> ROW_SHIFT=0 always    (single-row path)
        // ----------------------------------------------------------------------

        constexpr uint32_t STRIDE = N_THRDS_PER_GRP * N_THRD_GRPS_PER_THRD_BLK;
        static_assert(N_THRDS_PER_GRP == 32, "Energy reduction assumes warp-sized thread groups (32).");
        static_assert(N_ROWS_C == 32 || N_ROWS_C == 64, "Energy reduction assumes N_ROWS_C (N_BS_ANTS) is 32 or 64.");

        constexpr uint32_t ROW_SHIFT = STRIDE % N_ROWS_C;
        // Valid cases: 0 (single-row) or N_ROWS_C/2 (two-row alternation)
        static_assert(ROW_SHIFT == 0 || (2 * ROW_SHIFT == N_ROWS_C),
                      "Energy reduction assumes each thread touches <=2 rows; update logic if this changes.");

        const uint32_t row0 = THRD_ABS_IDX % N_ROWS_C;

        TCompute energy0 = cuGet<TCompute>(0);
        TCompute energy1 = cuGet<TCompute>(0);

        for (uint32_t idx = THRD_ABS_IDX; idx < N_ROWS_C * N_COLS_C; idx += STRIDE)
        {
            const uint32_t icol = idx / N_ROWS_C;
            const uint32_t irow = idx - icol * N_ROWS_C;

            // Apply scaling (same as original)
            shC(irow, icol) = shC(irow, icol) * type_convert<TCompute>(layerScalingFactors[icol]);

            // |C|^2 = Re^2 + Im^2  (same as original)
            const auto cre = cuReal(shC(irow, icol));
            const auto cim = cuImag(shC(irow, icol));
            const TCompute absCSqVal = cre * cre + cim * cim;

            if constexpr (ROW_SHIFT == 0)
            {
                // Thread always contributes to the same antenna row.
                energy0 += absCSqVal;
            }
            else
            {
                // Thread alternates between row0 and row1 = row0 + N_ROWS_C/2 (mod N_ROWS_C).
                if (irow == row0) energy0 += absCSqVal;
                else energy1 += absCSqVal;
            }
        }

        // Commit once per touched row (1-2 atomics per thread)
        atomicAdd(&antEnergies[row0], energy0);

        if constexpr (ROW_SHIFT != 0)
        {
            const uint32_t row1 = (row0 + ROW_SHIFT) % N_ROWS_C; // i.e., row0 + N_ROWS_C/2 for N_ROWS_C=64
            atomicAdd(&antEnergies[row1], energy1);
        }
        
        thisThrdBlk.sync();
        //==============================================================================================================

        for (uint32_t s = N_ROWS_C/2; s > 0; s/=2) 
        {
            if (THRD_ABS_IDX < s) 
            {
                antEnergies[THRD_ABS_IDX] = fmaxf(antEnergies[THRD_ABS_IDX], antEnergies[THRD_ABS_IDX + s]);
            }
            thisThrdBlk.sync();
        }
          
        if (THRD_ABS_IDX == 0) 
        {
            shCFrobeniusNorm = type_convert<TCompute>(rsqrtf(antEnergies[0]));
        }
        
        thisThrdBlk.sync();        
    }
    
    //--------------------------------------------------------------------------------------------------------
    // Stage4: Write the result BFC coefficients C into device memory

    float beta           = statDescr.beta; // Scale factor for converting FP16 to integer
    uint8_t compressBits = statDescr.compressBitwidth;

    int32_t compbytes = (compressBits == 16) ? N_BS_ANTS * 4 : 2 * N_BS_ANTS / 8 * compressBits + 1 + 2*(ueGrpPrms.beamIdOffset>=0);
    if(compressBits == 32)
    {
        compbytes = N_BS_ANTS * sizeof(TCompute);
        beta = 1.0f;
    }
    uint32_t output_index = PRB_GRP_IDX * compbytes; // (ANTENNAS, PRB_GRP, LAYERS)
    uint8_t* output = ueGrpPrms.pBfwCompCoef;
    const int16_t beamIdOffset = ueGrpPrms.beamIdOffset < 0 ? ueGrpPrms.beamIdOffset : ueGrpPrms.beamIdOffset + ueGrpPrms.startPrb/CUPHY_BFW_MIN_PRB_GRP_SIZE;
    constexpr uint32_t THRD_DIVS = N_BS_ANTS > 32 ? N_BS_ANTS : 32;
    constexpr uint32_t COMP_THRDS = THRD_DIVS*(N_THRDS_PER_GRP*N_THRD_GRPS_PER_THRD_BLK/THRD_DIVS);
    constexpr uint32_t MAX_PRBG_PER_LAYER = CUPHY_BFW_N_MAX_PRB_GRPS;
    bfw_scale_compress_blockFP<TCompute, N_BS_ANTS + 1, N_BS_ANTS, N_LAYERS, MAX_PRBG_PER_LAYER, COMP_THRDS>( 
        &smemBlkC[0],               // Shared memory input pointer for the antennas
        output + output_index,      // Output pointer for the first antenna
        beta*shCFrobeniusNorm,      // Scaling factor
        compressBits,               // Number of compressed bits, if 16=uncompressed, 32=FP pass-through
        beamIdOffset,               // Starting offset for dynamic beam IDs
        THRD_ABS_IDX,               // 1D thread rank
        nPrbGrp);                   // Stride between 2 layers (number of PRB groups)
        

#ifdef ENABLE_DEBUG
    // C
    for(uint32_t i = 0; i < N_ROWS_C; ++i)
    {
        if(THRD_ABS_IDX < N_COLS_C)
            printf("C[%d][%d][%d] = %f+j%f\n", PRB_IDX, i, COL_IDX_C, shC(i, COL_IDX_C).x, shC(i, COL_IDX_C).y);
    }
#endif
}




template <uint32_t N_BS_ANTS, // # of BS antenna (# of rows in H matrix)
          uint32_t N_LAYERS,  // # of layers (# of cols in H matrix)
          uint32_t N_THRD_GRPS_PER_THRD_BLK,
          uint32_t N_THRDS_PER_GRP>
void bfwCoefComp::bfwMmseCoefCompKernelLaunchGeo(uint16_t nMaxPrbGrp,
                                                 uint16_t nUeGrps,
                                                 dim3&    gridDim,
                                                 dim3&    blockDim)
{
    gridDim  = dim3(nMaxPrbGrp, nUeGrps);
    blockDim = dim3(N_THRDS_PER_GRP, N_THRD_GRPS_PER_THRD_BLK);

#ifdef ENABLE_DEBUG
    printf("blockDim (%d,%d,%d), gridDim (%d,%d,%d)\n", blockDim.x, blockDim.y, blockDim.z, gridDim.x, gridDim.y, gridDim.z);
#endif
}

template <typename TStorageIn,
          typename TStorageOut,
          typename TCompute,
          uint32_t N_BS_ANTS, // # of BS antenna (# of rows in H matrix)
          uint32_t N_LAYERS>  // # of layers (# of cols in H matrix)
void bfwCoefComp::bfwMmseCoefComp(bool                         getKernelFuncOnly,
                                  uint16_t                     nMaxPrbGrp,
                                  uint16_t                     nUeGrps,
                                  cuphyBfwCoefCompLaunchCfg_t& launchCfg)
{
    // ========================================================================
    // CTA Size Selection
    // ========================================================================
    // Block dimensions: (N_THRDS_PER_GRP, N_THRD_GRPS_PER_THRD_BLK)
    //                   (    warp_size  ,       num_warps        )
    //
    // The number of warps affects different kernel stages:
    //   - LU factor:    Needs enough threads for matrix operations
    //   - MMSE coeffs:  Block-stride loop over N_BS_ANTS × N_LAYERS elements
    //   - Normalization: Per-layer and per-antenna operations
    // ========================================================================

    constexpr uint32_t N_THRDS_PER_GRP = N_THREADS_PER_WARP;  // 32 (warp size)

    // Strategy 1: Scale with layer count (good for large N_LAYERS: 8, 16, 32)
    // Rationale: More layers need more parallel work for MMSE coefficient computation
    constexpr uint32_t WARPS_FOR_LARGE_LAYERS = N_LAYERS / 2;

    // Strategy 2: Scale with matrix size (good for small N_LAYERS: 1, 2, 4)
    // Rationale: Need enough threads to cover H matrix load and C matrix computation
    constexpr uint32_t MIN_THREADS_NEEDED = N_BS_ANTS + 2 * N_LAYERS;
    constexpr uint32_t WARPS_FOR_SMALL_LAYERS = div_round_up_cexp(MIN_THREADS_NEEDED, N_THRDS_PER_GRP);
    
    // Take the larger of the two strategies
    constexpr uint32_t WARPS_CALCULATED = (WARPS_FOR_LARGE_LAYERS > WARPS_FOR_SMALL_LAYERS)
                                        ? WARPS_FOR_LARGE_LAYERS
                                        : WARPS_FOR_SMALL_LAYERS;

    // Special case: When warp size (32) is not divisible by N_LAYERS,
    // force N_LAYERS warps to ensure correct thread-to-layer mapping in stage 3
    // (e.g., layerScalingFactors initialization and per-layer normalization)
    constexpr bool WARP_ALIGNS_WITH_LAYERS = (N_THRDS_PER_GRP % N_LAYERS == 0);

    // Final selection (with special case for N_LAYERS=32)
    constexpr uint32_t N_THRD_GRPS_PER_THRD_BLK = (N_LAYERS == 32) ? 32
                                                : (WARP_ALIGNS_WITH_LAYERS ? WARPS_CALCULATED : N_LAYERS);

    void* kernelFunc = reinterpret_cast<void*>(bfwMmseCoefCompKernel_v1<TStorageIn,
                                                                        TStorageOut,
                                                                        TCompute,
                                                                        N_BS_ANTS,
                                                                        N_LAYERS,
                                                                        N_THRD_GRPS_PER_THRD_BLK,
                                                                        N_THRDS_PER_GRP>);

    CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriver = launchCfg.kernelNodeParamsDriver;
    {MemtraceDisableScope md; CUDA_CHECK(cudaGetFuncBySymbol(&kernelNodeParamsDriver.func, kernelFunc));}

    if(! getKernelFuncOnly)
    {
        dim3 blockDim, gridDim;
        bfwMmseCoefCompKernelLaunchGeo<N_BS_ANTS, N_LAYERS, N_THRD_GRPS_PER_THRD_BLK, N_THRDS_PER_GRP>(nMaxPrbGrp, nUeGrps, gridDim, blockDim);
    
        // Dynamic shared memory size calculation:
        // smemBlkA: (N_LAYERS+1) * (2*N_LAYERS + N_BS_ANTS) complex floats
        // smemBlkC: (N_BS_ANTS+1) * N_LAYERS complex floats
        // smemBlkR: N_LAYERS floats
        // layerScalingFactors: N_LAYERS floats
        // antEnergies: N_BS_ANTS floats
        // shCFrobeniusNorm: 1 float
        constexpr int nSmemA = (N_LAYERS + 1) * (2 * N_LAYERS + N_BS_ANTS) * sizeof(TCompute) * 2;  // complex = 2 floats
        constexpr int nSmemC = (N_BS_ANTS + 1) * N_LAYERS * sizeof(TCompute) * 2;
        constexpr int nSmemR = N_LAYERS * sizeof(TCompute);
        constexpr int nSmemLayerScaling = N_LAYERS * sizeof(TCompute);
        constexpr int nSmemAntEnergies = N_BS_ANTS * sizeof(TCompute);
        constexpr int nSmemFrobenius = sizeof(TCompute);

        int nShmemBytes = nSmemA + nSmemC + nSmemR + nSmemLayerScaling + nSmemAntEnergies + nSmemFrobenius;

        /*cudaFuncSetAttribute(bfwMmseCoefCompKernel_v1<TStorageIn, TStorageOut, TCompute, N_BS_ANTS, N_LAYERS, N_THRD_GRPS_PER_THRD_BLK, N_THRDS_PER_GRP>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             nShmemBytes);*/
        CU_CHECK(cuFuncSetAttribute(kernelNodeParamsDriver.func, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, nShmemBytes));
    
        kernelNodeParamsDriver.blockDimX = blockDim.x;
        kernelNodeParamsDriver.blockDimY = blockDim.y;
        kernelNodeParamsDriver.blockDimZ = blockDim.z;
        
        kernelNodeParamsDriver.gridDimX = gridDim.x;
        kernelNodeParamsDriver.gridDimY = gridDim.y;
        kernelNodeParamsDriver.gridDimZ = gridDim.z;
    
        kernelNodeParamsDriver.extra          = nullptr;
        kernelNodeParamsDriver.sharedMemBytes = nShmemBytes;    
    }
}

template <typename TStorageIn, typename TStorageOut, typename TCompute>
void bfwCoefComp::bfwCoefCompKernelSelL0(bool                         getKernelFuncOnly,
                                         uint16_t                     nMaxPrbGrp,
                                         uint16_t                     nUeGrps,
                                         uint16_t                     nRxAnts,
                                         uint8_t                      nLayers,
                                         cuphyBfwCoefCompLaunchCfg_t& launchCfg)
{
    if(64 == nRxAnts)
    {
        constexpr uint32_t N_BS_ANTS = 64; // # of BS antenna (# of rows in H matrix)
        switch(nLayers)
        {
            // nLayers == 32
            case 32:
            {
                constexpr uint32_t N_LAYERS = 32; // # of layers (# of cols in H matrix)
                bfwMmseCoefComp<TStorageIn, TStorageOut, TCompute, N_BS_ANTS, N_LAYERS>(getKernelFuncOnly, nMaxPrbGrp, nUeGrps, launchCfg);
                break;
            }
            // nLayers == 31
            case 31:
            {
                constexpr uint32_t N_LAYERS = 31; // # of layers (# of cols in H matrix)
                bfwMmseCoefComp<TStorageIn, TStorageOut, TCompute, N_BS_ANTS, N_LAYERS>(getKernelFuncOnly, nMaxPrbGrp, nUeGrps, launchCfg);
                break;
            }
            // nLayers == 30
            case 30:
            {
                constexpr uint32_t N_LAYERS = 30; // # of layers (# of cols in H matrix)
                bfwMmseCoefComp<TStorageIn, TStorageOut, TCompute, N_BS_ANTS, N_LAYERS>(getKernelFuncOnly, nMaxPrbGrp, nUeGrps, launchCfg);
                break;
            }
            // nLayers == 29
            case 29:
            {
                constexpr uint32_t N_LAYERS = 29; // # of layers (# of cols in H matrix)
                bfwMmseCoefComp<TStorageIn, TStorageOut, TCompute, N_BS_ANTS, N_LAYERS>(getKernelFuncOnly, nMaxPrbGrp, nUeGrps, launchCfg);
                break;
            }
            // nLayers == 28
            case 28:
            {
                constexpr uint32_t N_LAYERS = 28; // # of layers (# of cols in H matrix)
                bfwMmseCoefComp<TStorageIn, TStorageOut, TCompute, N_BS_ANTS, N_LAYERS>(getKernelFuncOnly, nMaxPrbGrp, nUeGrps, launchCfg);
                break;
            }
            // nLayers == 27
            case 27:
            {
                constexpr uint32_t N_LAYERS = 27; // # of layers (# of cols in H matrix)
                bfwMmseCoefComp<TStorageIn, TStorageOut, TCompute, N_BS_ANTS, N_LAYERS>(getKernelFuncOnly, nMaxPrbGrp, nUeGrps, launchCfg);
                break;
            }
            // nLayers == 26
            case 26:
            {
                constexpr uint32_t N_LAYERS = 26; // # of layers (# of cols in H matrix)
                bfwMmseCoefComp<TStorageIn, TStorageOut, TCompute, N_BS_ANTS, N_LAYERS>(getKernelFuncOnly, nMaxPrbGrp, nUeGrps, launchCfg);
                break;
            }
            // nLayers == 25
            case 25:
            {
                constexpr uint32_t N_LAYERS = 25; // # of layers (# of cols in H matrix)
                bfwMmseCoefComp<TStorageIn, TStorageOut, TCompute, N_BS_ANTS, N_LAYERS>(getKernelFuncOnly, nMaxPrbGrp, nUeGrps, launchCfg);
                break;
            }
            // nLayers == 24
            case 24:
            {
                constexpr uint32_t N_LAYERS = 24; // # of layers (# of cols in H matrix)
                bfwMmseCoefComp<TStorageIn, TStorageOut, TCompute, N_BS_ANTS, N_LAYERS>(getKernelFuncOnly, nMaxPrbGrp, nUeGrps, launchCfg);
                break;
            }
            // nLayers == 23
            case 23:
            {
                constexpr uint32_t N_LAYERS = 23; // # of layers (# of cols in H matrix)
                bfwMmseCoefComp<TStorageIn, TStorageOut, TCompute, N_BS_ANTS, N_LAYERS>(getKernelFuncOnly, nMaxPrbGrp, nUeGrps, launchCfg);
                break;
            }
            // nLayers == 22
            case 22:
            {
                constexpr uint32_t N_LAYERS = 22; // # of layers (# of cols in H matrix)
                bfwMmseCoefComp<TStorageIn, TStorageOut, TCompute, N_BS_ANTS, N_LAYERS>(getKernelFuncOnly, nMaxPrbGrp, nUeGrps, launchCfg);
                break;
            }
            // nLayers == 21
            case 21:
            {
                constexpr uint32_t N_LAYERS = 21; // # of layers (# of cols in H matrix)
                bfwMmseCoefComp<TStorageIn, TStorageOut, TCompute, N_BS_ANTS, N_LAYERS>(getKernelFuncOnly, nMaxPrbGrp, nUeGrps, launchCfg);
                break;
            }
            // nLayers == 20
            case 20:
            {
                constexpr uint32_t N_LAYERS = 20; // # of layers (# of cols in H matrix)
                bfwMmseCoefComp<TStorageIn, TStorageOut, TCompute, N_BS_ANTS, N_LAYERS>(getKernelFuncOnly, nMaxPrbGrp, nUeGrps, launchCfg);
                break;
            }
            // nLayers == 19
            case 19:
            {
                constexpr uint32_t N_LAYERS = 19; // # of layers (# of cols in H matrix)
                bfwMmseCoefComp<TStorageIn, TStorageOut, TCompute, N_BS_ANTS, N_LAYERS>(getKernelFuncOnly, nMaxPrbGrp, nUeGrps, launchCfg);
                break;
            }
            // nLayers == 18
            case 18:
            {
                constexpr uint32_t N_LAYERS = 18; // # of layers (# of cols in H matrix)
                bfwMmseCoefComp<TStorageIn, TStorageOut, TCompute, N_BS_ANTS, N_LAYERS>(getKernelFuncOnly, nMaxPrbGrp, nUeGrps, launchCfg);
                break;
            }
            // nLayers == 17
            case 17:
            {
                constexpr uint32_t N_LAYERS = 17; // # of layers (# of cols in H matrix)
                bfwMmseCoefComp<TStorageIn, TStorageOut, TCompute, N_BS_ANTS, N_LAYERS>(getKernelFuncOnly, nMaxPrbGrp, nUeGrps, launchCfg);
                break;
            }
            // nLayers == 16
            case 16:
            {
                constexpr uint32_t N_LAYERS = 16; // # of layers (# of cols in H matrix)
                bfwMmseCoefComp<TStorageIn, TStorageOut, TCompute, N_BS_ANTS, N_LAYERS>(getKernelFuncOnly, nMaxPrbGrp, nUeGrps, launchCfg);
                break;
            }
            // nLayers == 15
            case 15:
            {
                constexpr uint32_t N_LAYERS = 15; // # of layers (# of cols in H matrix)
                bfwMmseCoefComp<TStorageIn, TStorageOut, TCompute, N_BS_ANTS, N_LAYERS>(getKernelFuncOnly, nMaxPrbGrp, nUeGrps, launchCfg);
                break;
            }
            // nLayers == 14
            case 14:
            {
                constexpr uint32_t N_LAYERS = 14; // # of layers (# of cols in H matrix)
                bfwMmseCoefComp<TStorageIn, TStorageOut, TCompute, N_BS_ANTS, N_LAYERS>(getKernelFuncOnly, nMaxPrbGrp, nUeGrps, launchCfg);
                break;
            }
            // nLayers == 13
            case 13:
            {
                constexpr uint32_t N_LAYERS = 13; // # of layers (# of cols in H matrix)
                bfwMmseCoefComp<TStorageIn, TStorageOut, TCompute, N_BS_ANTS, N_LAYERS>(getKernelFuncOnly, nMaxPrbGrp, nUeGrps, launchCfg);
                break;
            }
            // nLayers == 12
            case 12:
            {
                constexpr uint32_t N_LAYERS = 12; // # of layers (# of cols in H matrix)
                bfwMmseCoefComp<TStorageIn, TStorageOut, TCompute, N_BS_ANTS, N_LAYERS>(getKernelFuncOnly, nMaxPrbGrp, nUeGrps, launchCfg);
                break;
            }
            // nLayers == 11
            case 11:
            {
                constexpr uint32_t N_LAYERS = 11; // # of layers (# of cols in H matrix)
                bfwMmseCoefComp<TStorageIn, TStorageOut, TCompute, N_BS_ANTS, N_LAYERS>(getKernelFuncOnly, nMaxPrbGrp, nUeGrps, launchCfg);
                break;
            }
            // nLayers == 10
            case 10:
            {
                constexpr uint32_t N_LAYERS = 10; // # of layers (# of cols in H matrix)
                bfwMmseCoefComp<TStorageIn, TStorageOut, TCompute, N_BS_ANTS, N_LAYERS>(getKernelFuncOnly, nMaxPrbGrp, nUeGrps, launchCfg);
                break;
            }
            // nLayers == 9
            case 9:
            {
                constexpr uint32_t N_LAYERS = 9; // # of layers (# of cols in H matrix)
                bfwMmseCoefComp<TStorageIn, TStorageOut, TCompute, N_BS_ANTS, N_LAYERS>(getKernelFuncOnly, nMaxPrbGrp, nUeGrps, launchCfg);
                break;
            }
            // nLayers == 8
            case 8:
            {
                constexpr uint32_t N_LAYERS = 8; // # of layers (# of cols in H matrix)
                bfwMmseCoefComp<TStorageIn, TStorageOut, TCompute, N_BS_ANTS, N_LAYERS>(getKernelFuncOnly, nMaxPrbGrp, nUeGrps, launchCfg);
                break;
            }        
            // nLayers == 7
            case 7:
            {
                constexpr uint32_t N_LAYERS = 7; // # of layers (# of cols in H matrix)
                bfwMmseCoefComp<TStorageIn, TStorageOut, TCompute, N_BS_ANTS, N_LAYERS>(getKernelFuncOnly, nMaxPrbGrp, nUeGrps, launchCfg);
                break;
            }        
            // nLayers == 6
            case 6:
            {
                constexpr uint32_t N_LAYERS = 6; // # of layers (# of cols in H matrix)
                bfwMmseCoefComp<TStorageIn, TStorageOut, TCompute, N_BS_ANTS, N_LAYERS>(getKernelFuncOnly, nMaxPrbGrp, nUeGrps, launchCfg);
                break;
            }        
            // nLayers == 5
            case 5:
            {
                constexpr uint32_t N_LAYERS = 5; // # of layers (# of cols in H matrix)
                bfwMmseCoefComp<TStorageIn, TStorageOut, TCompute, N_BS_ANTS, N_LAYERS>(getKernelFuncOnly, nMaxPrbGrp, nUeGrps, launchCfg);
                break;
            }        
            // nLayers == 4
            case 4:
            {
                constexpr uint32_t N_LAYERS = 4; // # of layers (# of cols in H matrix)
                bfwMmseCoefComp<TStorageIn, TStorageOut, TCompute, N_BS_ANTS, N_LAYERS>(getKernelFuncOnly, nMaxPrbGrp, nUeGrps, launchCfg);
                break;
            }
            // nLayers == 3
            case 3:
            {
                constexpr uint32_t N_LAYERS = 3; // # of layers (# of cols in H matrix)
                bfwMmseCoefComp<TStorageIn, TStorageOut, TCompute, N_BS_ANTS, N_LAYERS>(getKernelFuncOnly, nMaxPrbGrp, nUeGrps, launchCfg);
                break;
            }
            // nLayers == 2
            case 2:
            {
                constexpr uint32_t N_LAYERS = 2; // # of layers (# of cols in H matrix)
                bfwMmseCoefComp<TStorageIn, TStorageOut, TCompute, N_BS_ANTS, N_LAYERS>(getKernelFuncOnly, nMaxPrbGrp, nUeGrps, launchCfg);
                break;
            }        
            // nLayers == 1
            case 1:
            {
                constexpr uint32_t N_LAYERS = 1; // # of layers (# of cols in H matrix)
                bfwMmseCoefComp<TStorageIn, TStorageOut, TCompute, N_BS_ANTS, N_LAYERS>(getKernelFuncOnly, nMaxPrbGrp, nUeGrps, launchCfg);
                break;
            }        
            default:
            {
                NVLOGE_FMT(NVLOG_BFW, AERIAL_CUPHY_EVENT, "{}: No kernel available to launch with requested configuration: nRxAnts {} nLayers {}", 
                           __FUNCTION__, nRxAnts, nLayers);
                break;
            }
        }
    }
    // nBSAnts = 32
    else if(32 == nRxAnts)
    {
        constexpr uint32_t N_BS_ANTS = 32; // # of BS antenna (# of rows in H matrix)
        switch(nLayers)
        {
            // nLayers == 8
            case 8:
            {
                constexpr uint32_t N_LAYERS = 8; // # of layers (# of cols in H matrix)
                bfwMmseCoefComp<TStorageIn, TStorageOut, TCompute, N_BS_ANTS, N_LAYERS>(getKernelFuncOnly, nMaxPrbGrp, nUeGrps, launchCfg);
                break;
            }
            // nLayers == 7
            case 7:
            {
                constexpr uint32_t N_LAYERS = 7; // # of layers (# of cols in H matrix)
                bfwMmseCoefComp<TStorageIn, TStorageOut, TCompute, N_BS_ANTS, N_LAYERS>(getKernelFuncOnly, nMaxPrbGrp, nUeGrps, launchCfg);
                break;
            }
            // nLayers == 6
            case 6:
            {
                constexpr uint32_t N_LAYERS = 6; // # of layers (# of cols in H matrix)
                bfwMmseCoefComp<TStorageIn, TStorageOut, TCompute, N_BS_ANTS, N_LAYERS>(getKernelFuncOnly, nMaxPrbGrp, nUeGrps, launchCfg);
                break;
            }
            // nLayers == 5
            case 5:
            {
                constexpr uint32_t N_LAYERS = 5; // # of layers (# of cols in H matrix)
                bfwMmseCoefComp<TStorageIn, TStorageOut, TCompute, N_BS_ANTS, N_LAYERS>(getKernelFuncOnly, nMaxPrbGrp, nUeGrps, launchCfg);
                break;
            }
            // nLayers == 4
            case 4:
            {
                constexpr uint32_t N_LAYERS = 4; // # of layers (# of cols in H matrix)
                bfwMmseCoefComp<TStorageIn, TStorageOut, TCompute, N_BS_ANTS, N_LAYERS>(getKernelFuncOnly, nMaxPrbGrp, nUeGrps, launchCfg);
                break;
            }
            // nLayers == 3
            case 3:
            {
                constexpr uint32_t N_LAYERS = 3; // # of layers (# of cols in H matrix)
                bfwMmseCoefComp<TStorageIn, TStorageOut, TCompute, N_BS_ANTS, N_LAYERS>(getKernelFuncOnly, nMaxPrbGrp, nUeGrps, launchCfg);
                break;
            }
            // nLayers == 2
            case 2:
            {
                constexpr uint32_t N_LAYERS = 2; // # of layers (# of cols in H matrix)
                bfwMmseCoefComp<TStorageIn, TStorageOut, TCompute, N_BS_ANTS, N_LAYERS>(getKernelFuncOnly, nMaxPrbGrp, nUeGrps, launchCfg);
                break;
            }
            // nLayers == 1
            case 1:
            {
                constexpr uint32_t N_LAYERS = 1; // # of layers (# of cols in H matrix)
                bfwMmseCoefComp<TStorageIn, TStorageOut, TCompute, N_BS_ANTS, N_LAYERS>(getKernelFuncOnly, nMaxPrbGrp, nUeGrps, launchCfg);
                break;
            }        
            default:
            {
                NVLOGE_FMT(NVLOG_BFW, AERIAL_CUPHY_EVENT, "{}: No kernel available to launch with requested configuration: nRxAnts {} nLayers {}", 
                           __FUNCTION__, nRxAnts, nLayers);
                break;
            }
        }
    }
    else
    {
        NVLOGE_FMT(NVLOG_BFW, AERIAL_CUPHY_EVENT, "{}: No kernel available to launch with requested configuration: nRxAnts {} nLayers {}", 
                   __FUNCTION__, nRxAnts, nLayers);
    }
}

void bfwCoefComp::bfwCoefCompKernelSelL1(bool                         getKernelFuncOnly,
                                         uint16_t                     nMaxPrbGrp,
                                         uint16_t                     nUeGrps,
                                         uint16_t                     nRxAnts,
                                         uint8_t                      nLayers,
                                         cuphyDataType_t              srsChEstType,
                                         cuphyDataType_t              lambdaType,
                                         cuphyBfwCoefCompLaunchCfg_t& launchCfg)
{
#ifdef ENABLE_DEBUG    
    NVLOGD_FMT(NVLOG_BFW, "{}:{} Begin",__FUNCTION__, __LINE__);
#endif
    using TCompute = float;
    if((CUPHY_C_16F == srsChEstType) && (CUPHY_R_32F == lambdaType))
    {
        using TStorageIn = scalar_from_complex<data_type_traits<CUPHY_C_16F>::type>::type;
            using TStorageOut = scalar_from_complex<data_type_traits<CUPHY_C_16F>::type>::type;
            bfwCoefCompKernelSelL0<TStorageIn, TStorageOut, TCompute>(getKernelFuncOnly,
                                                                      nMaxPrbGrp,
                                                                      nUeGrps,
                                                                      nRxAnts,
                                                                      nLayers,
                                                                      launchCfg);
    }
    // lambdaType is always CUPHY_R_32F at compile time, so the following condition is not possible
    /*VCAST_DONT_INSTRUMENT_START*/
    else if((CUPHY_C_16F == srsChEstType) && (CUPHY_R_16F == lambdaType))
    {
        using TStorageIn  = scalar_from_complex<data_type_traits<CUPHY_C_16F>::type>::type;
        using TStorageOut = scalar_from_complex<data_type_traits<CUPHY_C_16F>::type>::type;
        bfwCoefCompKernelSelL0<TStorageIn, TStorageOut, TCompute>(getKernelFuncOnly,
                                                                  nMaxPrbGrp,
                                                                  nUeGrps,
                                                                  nRxAnts,
                                                                  nLayers,
                                                                  launchCfg);    
    }
    /*VCAST_DONT_INSTRUMENT_END*/
    else
    {
        NVLOGE_FMT(NVLOG_BFW, AERIAL_CUPHY_EVENT, "{}:{} No kernel available to launch with requested data type srsChEstType ({}) lambdaType ({})", 
                   __FUNCTION__, __LINE__, +srsChEstType, +lambdaType);
    }
#ifdef ENABLE_DEBUG
    NVLOGD_FMT(NVLOG_BFW, "{}:{} done",__FUNCTION__, __LINE__);
#endif
}

void bfwCoefComp::getDescrInfo(uint16_t nMaxUeGrps,
                               uint16_t nMaxTotalLayers,
                               size_t&  statDescrSizeBytes,
                               size_t&  statDescrAlignBytes,
                               size_t&  dynDescrSizeBytes,
                               size_t&  dynDescrAlignBytes,
                               size_t&  hetCfgUeGrpMapSizeBytes,
                               size_t&  hetCfgUeGrpMapAlignBytes,
                               size_t&  ueGrpPrmsSizeBytes,
                               size_t&  ueGrpPrmsAlignBytes,
                               size_t&  bfLayerPrmsSizeBytes,
                               size_t&  bfLayerPrmsAlignBytes)
{
    // Calculate sizes for various descriptor types
    statDescrSizeBytes  = sizeof(bfwCoefCompStatDescr_t);
    statDescrAlignBytes = alignof(bfwCoefCompStatDescr_t);

    dynDescrSizeBytes  = sizeof(bfwCoefCompDynDescrArr_t);
    dynDescrAlignBytes = alignof(bfwCoefCompDynDescrArr_t);

    hetCfgUeGrpMapSizeBytes = sizeof(decltype(m_pHetCfgUeGrpMapCpu)) * CUPHY_BFW_COEF_COMP_N_MAX_HET_CFGS * nMaxUeGrps;
    hetCfgUeGrpMapAlignBytes = alignof(decltype(m_pHetCfgUeGrpMapCpu));

    // Per UE group parameter descriptors
    ueGrpPrmsSizeBytes  = sizeof(bfwCoefCompKernelUeGrpPrm_t) * nMaxUeGrps;
    ueGrpPrmsAlignBytes = alignof(bfwCoefCompKernelUeGrpPrm_t);

    // Per layer parameter descriptors
    bfLayerPrmsSizeBytes  = sizeof(bfwCoefCompKernelBfLayerPrm_t) * nMaxTotalLayers;
    bfLayerPrmsAlignBytes = alignof(bfwCoefCompKernelBfLayerPrm_t);

    // dynDescrSizeBytes  = hetCfgUeGrpMapSizeBytes + perLayerDynDescrSizeBytes + perUeGrpDynDescrSizeBytes + dynDescrSizeBytes;    
    // dynDescrAlignBytes = std::max({hetCfgUeGrpMapAlignBytes, perLayerDynDescrAlignBytes, perUeGrpDynDescrAlignBytes, dynDescrAlignBytes});
}

cuphyStatus_t bfwCoefComp::init(bool         enableCpuToGpuDescrAsyncCpy,
                                uint8_t      compressBitwidth,
                                float        beta,
                                float        lambda,
                                uint8_t      bfwPowerNormAlg_selector,
                                void*        pStatDescrCpu,
                                void*        pStatDescrGpu,
                                void*        pDynDescrsCpu,
                                void*        pDynDescrsGpu,
                                void*        pHetCfgUeGrpMapCpu,
                                void*        pHetCfgUeGrpMapGpu,
                                void*        pUeGrpPrmsCpu,
                                void*        pUeGrpPrmsGpu,
                                void*        pBfLayerPrmsCpu,
                                void*        pBfLayerPrmsGpu,
                                cudaStream_t strm)
{
    if(!pStatDescrCpu || !pStatDescrGpu || !pDynDescrsCpu || !pDynDescrsGpu || !pHetCfgUeGrpMapCpu || !pHetCfgUeGrpMapGpu ||
       !pUeGrpPrmsCpu || !pUeGrpPrmsGpu || !pBfLayerPrmsCpu || !pBfLayerPrmsGpu) 
       return CUPHY_STATUS_INVALID_ARGUMENT;
              
    m_pHetCfgUeGrpMapCpu = (static_cast<decltype(m_pHetCfgUeGrpMapCpu)>(pHetCfgUeGrpMapCpu)); 
    m_pHetCfgUeGrpMapGpu = (static_cast<decltype(m_pHetCfgUeGrpMapGpu)>(pHetCfgUeGrpMapGpu)); 

    // Note: could use std::span as std::span kernelUeGrpPrmSpanCpu{pUeGrpDynDescrsCpu, m_nMaxTotalLayers}; for bounds-safe access
    m_pKernelUeGrpPrmCpu = (static_cast<bfwCoefCompKernelUeGrpPrm_t*>(pUeGrpPrmsCpu));
    m_pKernelUeGrpPrmGpu = (static_cast<bfwCoefCompKernelUeGrpPrm_t*>(pUeGrpPrmsGpu));

    m_pKernelBfLayerPrmCpu = (static_cast<bfwCoefCompKernelBfLayerPrm_t*>(pBfLayerPrmsCpu));
    m_pKernelBfLayerPrmGpu = (static_cast<bfwCoefCompKernelBfLayerPrm_t*>(pBfLayerPrmsGpu));

    // Setup static descriptor
    m_pStatDescrCpu = static_cast<bfwCoefCompStatDescr_t*>(pStatDescrCpu);
    m_pStatDescrGpu = static_cast<bfwCoefCompStatDescr_t*>(pStatDescrGpu);
    bfwCoefCompStatDescr_t& statDescrCpu = *m_pStatDescrCpu;
    statDescrCpu.compressBitwidth      = (0==compressBitwidth) ? 32 : compressBitwidth;
    statDescrCpu.beta                  = beta;
    statDescrCpu.lambda                = lambda;
    statDescrCpu.pKernelUeGrpPrms      = m_pKernelUeGrpPrmGpu;
    statDescrCpu.bfwPowerNormAlg_selector = bfwPowerNormAlg_selector;
    // statDescrCpu.pKernelBfLayerPrms = m_pKernelBfLayerPrmGpu;

    for(uint32_t hetCfgIdx = 0; hetCfgIdx < CUPHY_BFW_COEF_COMP_N_MAX_HET_CFGS; ++hetCfgIdx)
    {
        m_pHetCfgUeGrpMapArr[hetCfgIdx] = &m_pHetCfgUeGrpMapCpu[hetCfgIdx*m_nMaxUeGrps];

        // Setup pointers to heterogenous config to UE group map statically (the map values however change dynamically)
        statDescrCpu.pHetCfgUeGrpMap[hetCfgIdx] = &m_pHetCfgUeGrpMapGpu[hetCfgIdx*m_nMaxUeGrps];

        m_coefCompKernelArgsArr[hetCfgIdx].pStatDescr = m_pStatDescrGpu;
    }

    if(enableCpuToGpuDescrAsyncCpy)
    {
        CUDA_CHECK(cudaMemcpyAsync(m_pStatDescrGpu, m_pStatDescrCpu, sizeof(bfwCoefCompStatDescr_t), cudaMemcpyHostToDevice, strm));
    }

    // Save pointers to dynamic descriptors
    m_pDynDescrCpu = (static_cast<bfwCoefCompDynDescr_t*>(pDynDescrsCpu));
    m_pDynDescrGpu = (static_cast<bfwCoefCompDynDescr_t*>(pDynDescrsGpu));

    return CUPHY_STATUS_SUCCESS;
}

void bfwCoefComp::setupUeGrpDynDescr(cuphyBfwUeGrpPrm_t const&      ueGrpPrm,
                                     bfwCoefCompKernelUeGrpPrm_t&   kernelUeGrpPrm,
                                     bfwCoefCompKernelBfLayerPrm_t* pKernelLayerPrmCpu,
                                     bfwCoefCompKernelBfLayerPrm_t* pKernelLayerPrmGpu,
                                     cuphySrsChEstBuffInfo_t*       pChEstInfo,
                                     uint8_t**                      pBfwCompCoef)
{   
    // Read UE group level parameters from API into descriptor
    kernelUeGrpPrm.nPrbGrp   = ueGrpPrm.nPrbGrp;
    kernelUeGrpPrm.startPrb  = ueGrpPrm.startPrb;
    kernelUeGrpPrm.nRxAnt    = ueGrpPrm.nRxAnt;
    kernelUeGrpPrm.nBfLayers = ueGrpPrm.nBfLayers;
    kernelUeGrpPrm.pBfwCompCoef = pBfwCompCoef[ueGrpPrm.coefBufIdx];
    kernelUeGrpPrm.beamIdOffset = ueGrpPrm.beamIdOffset;

    // Setup per layer parameter in per UE group descriptor (bfwCoefCompKernelUeGrpPrm_t) to point to per layer CPU/GPU descriptors (bfwCoefCompKernelBfLayerPrm_t)
    kernelUeGrpPrm.pBfLayerPrmCpu = pKernelLayerPrmCpu;
    kernelUeGrpPrm.pBfLayerPrmGpu = pKernelLayerPrmGpu;

    // Beamforming layer level parameters from API into descriptor
    for(uint32_t layerIdx = 0; layerIdx < ueGrpPrm.nBfLayers; ++layerIdx)
    {
        // Copy per layer parameters into per layer CPU descriptor (bfwCoefCompKernelBfLayerPrm_t) which will then be copied to the GPU 
        // counterpart as part of bulk copy
        cuphyBfwLayerPrm_t const& bfLayerPrm            = ueGrpPrm.pBfLayerPrm[layerIdx];
        bfwCoefCompKernelBfLayerPrm_t& kernelBfLayerPrm = kernelUeGrpPrm.pBfLayerPrmCpu[layerIdx];
        kernelBfLayerPrm.ueLayerIdx                     = bfLayerPrm.ueLayerIndex;

        // Determine frequency (start PRB group) offset
        cuphySrsChEstBuffInfo_t const& chEstInfo = pChEstInfo[bfLayerPrm.chEstInfoBufIdx];

        #ifdef CUPHY_ENABLE_FLEXIBLE_BFW_PRB_GRPS
            uint16_t srsStartPrb = chEstInfo.startPrbGrp * chEstInfo.srsPrbGrpSize;
            uint16_t bfwStartPrb = ueGrpPrm.startPrb;  
            if(srsStartPrb > bfwStartPrb)
            {
                throw std::runtime_error(std::string("bfwCoefComp::setupUeGrpDynDescr: SRS ChEst startPrb (" + std::to_string(srsStartPrb) + ") beyond BFW startPrb (" + std::to_string(bfwStartPrb) + ")"));
            }
        #else
            if(chEstInfo.startPrbGrp > ueGrpPrm.startPrbGrp)
            {
                throw std::runtime_error(std::string("bfwCoefComp::setupUeGrpDynDescr: SRS ChEst startPrb (" + std::to_string(chEstInfo.startPrbGrp) + ") beyond BFW startPrb (" + std::to_string(ueGrpPrm.startPrbGrp) + ")"));
            }
        #endif



        #ifdef CUPHY_ENABLE_FLEXIBLE_BFW_PRB_GRPS
            uint16_t bfwToSrs_ueGrpSizeRatio   = ueGrpPrm.bfwPrbGrpSize / chEstInfo.srsPrbGrpSize;
            uint16_t srsUeGrpOffset            = bfwToSrs_ueGrpSizeRatio / 2;
            uint16_t ueGrpAllocStartSrsPrbGrp  = ueGrpPrm.startPrb / chEstInfo.srsPrbGrpSize;
            kernelBfLayerPrm.startPrbGrpOffset = ueGrpAllocStartSrsPrbGrp + srsUeGrpOffset - chEstInfo.startPrbGrp;
            kernelBfLayerPrm.prbGrpStride      = bfwToSrs_ueGrpSizeRatio;
            kernelBfLayerPrm.chEstInfoStartPrbGrp = chEstInfo.startPrbGrp;
            kernelBfLayerPrm.startValidPrg     = chEstInfo.startValidPrg;
            kernelBfLayerPrm.nValidPrg         = chEstInfo.nValidPrg;
        #else
            kernelBfLayerPrm.prbGrpStride      = 1;
            kernelBfLayerPrm.startPrbGrpOffset = ueGrpPrm.startPrbGrp - chEstInfo.startPrbGrp;
            kernelBfLayerPrm.chEstInfoStartPrbGrp = chEstInfo.startPrbGrp;
            kernelBfLayerPrm.startValidPrg     = chEstInfo.startValidPrg;
            kernelBfLayerPrm.nValidPrg         = chEstInfo.nValidPrg;
        #endif  



        copyTensorPrm2Info(chEstInfo.tChEstBuffer, kernelBfLayerPrm.tInfoSrsChEst);
        // printf("layerIdx %d chEstInfoBufIdx %d ueLayerIdx %d elemType %d\n", layerIdx, bfLayerPrm.chEstInfoBufIdx, kernelBfLayerPrm.ueLayerIdx, kernelBfLayerPrm.tInfoSrsChEst.elemType);
    }
}

// Sweep through the UE groups, batch them into heterogenous configurations and setup kernel descriptors
void bfwCoefComp::setupAndBatchCoefComp(uint16_t                  nUeGrps,
                                        cuphyBfwUeGrpPrm_t const* pUeGrpPrms,
                                        uint32_t&                 nHetCfgs,
                                        cuphySrsChEstBuffInfo_t*  pChEstInfo,
                                        uint8_t**                 pBfwCompCoef)
{
    //--------------------------------------------------------------------------------------------------------
    // Helper to find kernel function
    auto findKernelFunc = [](bfwCoefCompHetCfgArr_t const& hetCfgs, CUfunction func, int32_t& hetCfgIdx) {
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

    //--------------------------------------------------------------------------------------------------------
    // Initialize the batch config data structure
    bfwCoefCompHetCfgArr_t& hetCfgs = m_coefCompHetCfgsArr;
    std::fill(hetCfgs.begin(), hetCfgs.end(), bfwCoefCompHetCfg_t{nullptr, 0, 0});

#ifdef ENABLE_DEBUG
    NVLOGD_FMT(NVLOG_BFW, "{}: # of UE groups {}",__FUNCTION__,nUeGrps);
#endif

    //--------------------------------------------------------------------------------------------------------
    // UE group sweep
    // Index into global descriptor pool of beamforming layers (across all UE groups)
    uint16_t globalBfLayerOffset = 0;
    nHetCfgs = 0;
    for(int32_t ueGrpIdx = 0; ueGrpIdx < nUeGrps; ++ueGrpIdx)
    {
        //----------------------------------------------------------------------------------------------------
        // Absorb input API and setup kernel descriptor per UE group
        cuphyBfwUeGrpPrm_t const& ueGrpPrm = pUeGrpPrms[ueGrpIdx];
        bfwCoefCompKernelUeGrpPrm_t& kernelUeGrpPrm = m_pKernelUeGrpPrmCpu[ueGrpIdx];

        if((globalBfLayerOffset + ueGrpPrm.nBfLayers) >= m_nMaxTotalLayers) throw std::runtime_error(std::string("bfwCoefComp::setupAndBatchCoefComp: Exceeded limit (" + std::to_string(m_nMaxTotalLayers) + ") on total number of layers"));

        setupUeGrpDynDescr(ueGrpPrm, 
                           kernelUeGrpPrm,
                           &m_pKernelBfLayerPrmCpu[globalBfLayerOffset], 
                           &m_pKernelBfLayerPrmGpu[globalBfLayerOffset],
                           pChEstInfo,
                           pBfwCompCoef);
        globalBfLayerOffset += ueGrpPrm.nBfLayers;

        //----------------------------------------------------------------------------------------------------
        // Batch UE group into heterogenous configurations
        cuphyBfwCoefCompLaunchCfg_t launchCfg;
        bool getKernelFuncOnly = true;
        bfwCoefCompKernelSelL1(getKernelFuncOnly,
                               0,
                               0,
                               kernelUeGrpPrm.nRxAnt,
                               kernelUeGrpPrm.nBfLayers,
                               kernelUeGrpPrm.pBfLayerPrmCpu[0].tInfoSrsChEst.elemType,
                               type_to_cuphy_type<decltype(m_pStatDescrCpu->lambda)>::value,
                               launchCfg);

        // Check if the heterogenous configuration already exists
        int32_t hetCfgIdx = 0;
        findKernelFunc(hetCfgs, launchCfg.kernelNodeParamsDriver.func, hetCfgIdx);

        uint16_t nPrbGrp = ueGrpPrm.nPrbGrp;
        // If a heterogenous configuration already exists then increment the # of UE groups for that config
        if(-1 != hetCfgIdx)
        {
            bfwCoefCompHetCfg_t& hetCfg = hetCfgs[hetCfgIdx];
            if(hetCfg.nUeGrps >= m_nMaxUeGrps)
            {
                throw std::runtime_error(std::string("bfwCoefComp::batchCoefComp: Exceeded limit (" + std::to_string(m_nMaxUeGrps) + ") on supported UE groups"));
            }

            if(nPrbGrp > hetCfg.nMaxPrbGrp) hetCfg.nMaxPrbGrp = nPrbGrp;

            m_pHetCfgUeGrpMapArr[hetCfgIdx][hetCfg.nUeGrps] = ueGrpIdx;
            hetCfg.nUeGrps++;

#ifdef ENABLE_DEBUG
            printf("bfwCoefComp::batchCoefComp: UE group %02d -> HetCfg %d (nUeGrps %02d nPrbGrp %03d nMaxPrbGrp %03d nRxAnt %02d nLayers %02d)\n", ueGrpIdx, hetCfgIdx, hetCfg.nUeGrps, nPrbGrp, hetCfg.nMaxPrbGrp, ueGrpPrm.nRxAnt, ueGrpPrm.nBfLayers);
#endif
        }
        // New heterogenous configuration found
        else
        {
            if(nHetCfgs >= CUPHY_BFW_COEF_COMP_N_MAX_HET_CFGS)
            {
                throw std::runtime_error("bfwCoefComp::batchCoefComp: Exceeded limit (" + std::to_string(CUPHY_BFW_COEF_COMP_N_MAX_HET_CFGS) + ") on supported heterogneous configurations");
            }

            int32_t newHetCfgIdx        = nHetCfgs++;
            bfwCoefCompHetCfg_t& hetCfg = hetCfgs[newHetCfgIdx];
            hetCfg.func                 = launchCfg.kernelNodeParamsDriver.func;
            hetCfg.nMaxPrbGrp           = nPrbGrp;

            m_pHetCfgUeGrpMapArr[newHetCfgIdx][hetCfg.nUeGrps] = ueGrpIdx;
            hetCfg.nUeGrps++;

#ifdef ENABLE_DEBUG
            printf("bfwCoefComp::setupCoefComp: UE group %02d -> HetCfg %d (nUeGrps %02d nPrbGrp %03d nMaxPrbGrp %03d nRxAnt %02d nLayers %02d)\n", ueGrpIdx, newHetCfgIdx, hetCfg.nUeGrps, nPrbGrp, hetCfg.nMaxPrbGrp, ueGrpPrm.nRxAnt, ueGrpPrm.nBfLayers);
#endif
        }
    }
}

cuphyStatus_t bfwCoefComp::setupCoefComp(uint16_t                      nUeGrps,
                                         cuphyBfwUeGrpPrm_t const*     pUeGrpPrms,
                                         bool                          enableCpuToGpuDescrAsyncCpy,
                                         cuphySrsChEstBuffInfo_t*      pChEstInfo,
                                         uint8_t**                     pBfwCompCoef,
                                         cuphyBfwCoefCompLaunchCfgs_t* pLaunchCfgs,
                                         cudaStream_t                  strm)
{
    if(!pUeGrpPrms || !pChEstInfo || !pBfwCompCoef || !pLaunchCfgs) return CUPHY_STATUS_INVALID_ARGUMENT;

    cuphyBfwCoefCompLaunchCfgs_t& launchCfgs = *pLaunchCfgs;
    setupAndBatchCoefComp(nUeGrps,
                          pUeGrpPrms,
                          launchCfgs.nCfgs,
                          pChEstInfo,
                          pBfwCompCoef);
    
    for(uint32_t hetCfgIdx = 0; hetCfgIdx < launchCfgs.nCfgs; ++hetCfgIdx)
    {
        // Skip rest of the setup if there are no UE groups corresponding to the channel equalizer instance and hetCfg
        if(0 == m_coefCompHetCfgsArr[hetCfgIdx].nUeGrps) continue;
        
        bfwCoefCompDynDescr_t& dynDescr = m_pDynDescrCpu[hetCfgIdx];
        dynDescr.hetCfgIdx = hetCfgIdx;

        bfwCoefCompHetCfg_t const& hetCfg   = m_coefCompHetCfgsArr[hetCfgIdx];
        bfwCoefCompKernelArgs_t& kernelArgs = m_coefCompKernelArgsArr[hetCfgIdx];

        // Select kernel
        cuphyBfwCoefCompLaunchCfg_t& launchCfg = launchCfgs.cfgs[hetCfgIdx];

        // All UE groups within the a heterogenous config have the same gNB antenna and layer config
        int32_t ueGrpIdx = m_pHetCfgUeGrpMapArr[hetCfgIdx][0];
        bfwCoefCompKernelUeGrpPrm_t& kernelUeGrpPrm = m_pKernelUeGrpPrmCpu[ueGrpIdx];
        bool getKernelFuncOnly = false;
        bfwCoefCompKernelSelL1(getKernelFuncOnly,
                               hetCfg.nMaxPrbGrp,
                               hetCfg.nUeGrps,
                               kernelUeGrpPrm.nRxAnt,
                               kernelUeGrpPrm.nBfLayers,
                               kernelUeGrpPrm.pBfLayerPrmCpu[0].tInfoSrsChEst.elemType,
                               type_to_cuphy_type<decltype(m_pStatDescrCpu->lambda)>::value,
                               launchCfg);

        if(hetCfg.func != launchCfg.kernelNodeParamsDriver.func)
        {
           throw std::runtime_error("bfwCoefComp::setupCoefComp: Kernel function mismatch");
        }                                   

        kernelArgs.pDynDescr    = &m_pDynDescrGpu[hetCfgIdx];
        launchCfg.kernelArgs[0] = &kernelArgs.pStatDescr;
        launchCfg.kernelArgs[1] = &kernelArgs.pDynDescr;
        launchCfg.kernelNodeParamsDriver.kernelParams = &(launchCfg.kernelArgs[0]);
    }

    // Optional descriptor copy to GPU memory
    if(enableCpuToGpuDescrAsyncCpy)
    {
        m_batchedMemcpyHelperH2D.reset(); // resets indices so updateMemcpy starts populating the buffers from the beginning
        m_batchedMemcpyHelperH2D.updateMemcpy(m_pDynDescrGpu        , m_pDynDescrCpu        , sizeof(bfwCoefCompDynDescrArr_t)                                                      , cudaMemcpyHostToDevice, strm);
        m_batchedMemcpyHelperH2D.updateMemcpy(m_pHetCfgUeGrpMapGpu  , m_pHetCfgUeGrpMapCpu  , sizeof(decltype(m_pHetCfgUeGrpMapCpu))*CUPHY_BFW_COEF_COMP_N_MAX_HET_CFGS*m_nMaxUeGrps, cudaMemcpyHostToDevice, strm);
        m_batchedMemcpyHelperH2D.updateMemcpy(m_pKernelUeGrpPrmGpu  , m_pKernelUeGrpPrmCpu  , sizeof(bfwCoefCompKernelUeGrpPrm_t)*m_nMaxUeGrps                                      , cudaMemcpyHostToDevice, strm);
        m_batchedMemcpyHelperH2D.updateMemcpy(m_pKernelBfLayerPrmGpu, m_pKernelBfLayerPrmCpu, sizeof(bfwCoefCompKernelBfLayerPrm_t)*m_nMaxTotalLayers                               , cudaMemcpyHostToDevice, strm);
        cuphyStatus_t status = m_batchedMemcpyHelperH2D.launchBatchedMemcpy(strm);
        if (status != CUPHY_STATUS_SUCCESS)
        {
            NVLOGE_FMT(NVLOG_BFW, AERIAL_CUPHY_EVENT, "Launching batched memcpy for BFW returned an error");
            return status;
        }
    }
    return CUPHY_STATUS_SUCCESS;    
}

} // namespace bfw_coefComp
