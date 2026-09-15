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

//#define CUPHY_DEBUG 1
#include "convert_tensor.cuh"
#include "type_convert.hpp"
#include "cuphy_kernel_util.cuh"

////////////////////////////////////////////////////////////////////////
// convert_kernel()
template <typename Tdst, typename Tsrc>
__global__ void convert_kernel(tensor_layout_any layoutDst,
                               Tdst*             dst,
                               tensor_layout_any layoutSrc,
                               const Tsrc*       src)
{
    static_assert(CUPHY_DIM_MAX == 5, "convert_kernel only defined for 5D tensors");
    for(int i4 = 0; i4 < layoutSrc.dimensions[4]; ++i4)
    {    
        for(int i3 = 0; i3 < layoutSrc.dimensions[3]; ++i3)
        {
            for(grid_stride_index<2> it2; it2 < layoutSrc.dimensions[2]; it2.next())
            {
                for(grid_stride_index<1> it1; it1 < layoutSrc.dimensions[1]; it1.next())
                {
                    for(grid_stride_index<0> it0; it0 < layoutSrc.dimensions[0]; it0.next())
                    {
                        int    n[5]    = {static_cast<int>(it0.value),
                                        static_cast<int>(it1.value),
                                        static_cast<int>(it2.value),
                                        i3,
                                        i4};
                        size_t out_idx = layoutDst.offset(n);
                        size_t in_idx  = layoutSrc.offset(n);
                        // printf("(i0, i1, i2, i3) = (%u, %u, %u %u), src offset = %lu, dst
                        // offset = %lu\n",
                        //       it0.value,
                        //       it1.value,
                        //       it2.value,
                        //       i3,
                        //       in_idx,
                        //       out_idx);
                        dst[out_idx] = type_convert<Tdst>(src[in_idx]);
                    } // it0
                }     // it1
            }         // it2
        }             // it3
    }                 // it4
}

////////////////////////////////////////////////////////////////////////
// launch_convert()
template <cuphyDataType_t Tdst, cuphyDataType_t Tsrc>
cuphyStatus_t launch_convert(const tensor_layout_any& kLayoutDst,
                             void*                    dstAddr,
                             const tensor_layout_any& kLayoutSrc,
                             const void*              srcAddr,
                             const dim3               gridDim,
                             const dim3               blockDim,
                             cudaStream_t             strm)
{
    typedef typename data_type_traits<Tdst>::type dst_type_t;
    typedef typename data_type_traits<Tsrc>::type src_type_t;
    convert_kernel<dst_type_t, src_type_t><<<gridDim, blockDim, 0, strm>>>(kLayoutDst,
                                                                           static_cast<dst_type_t*>(dstAddr),
                                                                           kLayoutSrc,
                                                                           static_cast<const src_type_t*>(srcAddr));
    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// convert_bits_to_bits_kernel()
__global__ void
__launch_bounds__(1024)
convert_bits_to_bits_kernel(tensor_layout_any layoutDst,
                            uint32_t*         dst,
                            uint32_t          columnEndMask,
                            tensor_layout_any layoutSrc,
                            const uint32_t*   src)
{
    static_assert(CUPHY_DIM_MAX == 5, "convert_kernel only defined for 5D tensors");
    for(int i4 = 0; i4 < layoutSrc.dimensions[4]; ++i4)
    {    
        for(int i3 = 0; i3 < layoutSrc.dimensions[3]; ++i3)
        {
            for(grid_stride_index<2> it2; it2 < layoutSrc.dimensions[2]; it2.next())
            {
                for(grid_stride_index<1> it1; it1 < layoutSrc.dimensions[1]; it1.next())
                {
                    for(grid_stride_index<0> it0; it0 < layoutSrc.dimensions[0]; it0.next())
                    {
                        int    n[5]    = {static_cast<int>(it0.value),
                                        static_cast<int>(it1.value),
                                        static_cast<int>(it2.value),
                                        i3,
                                        i4};
                        size_t out_idx = layoutDst.offset(n);
                        size_t in_idx  = layoutSrc.offset(n);
                        //printf("(i0, i1, i2, i3) = (%u, %u, %u %u), src offset = %lu, dst offset = %lu\n",
                        //       it0.value,
                        //       it1.value,
                        //       it2.value,
                        //       i3,
                        //       in_idx,
                        //       out_idx);
                        if(it0.value < (layoutSrc.dimensions[0] - 1))
                        {
                            dst[out_idx] = src[in_idx];
                        }
                        else
                        {
                            // For the last element in the column, we may
                            // want to mask out bits from the source tensor.
                            dst[out_idx] = src[in_idx] & columnEndMask;
                            //printf("it0 = %u, max = %u, mask = %u, src = 0x%X, dst = 0x%X\n",
                            //       it0.value,
                            //       layoutSrc.dimensions[0],
                            //       columnEndMask,
                            //       src[in_idx],
                            //       dst[out_idx]);
                        }

                    } // it0
                }     // it1
            }         // it2
        }             // it3
    }                 // it4
}

////////////////////////////////////////////////////////////////////////
// launch_convert_bits_to_bits()
cuphyStatus_t launch_convert_bits_to_bits(const tensor_layout_any& kLayoutDst,
                                          void*                    dstAddr,
                                          const tensor_layout_any& kLayoutSrc,
                                          const void*              srcAddr,
                                          const dim3               gridDim,
                                          const dim3               blockDim,
                                          cudaStream_t             strm)
{
    // Create modified tensor layouts that describe the uint32_t
    // word tensor.
    tensor_layout_any kDstModified  = word_layout_from_bit_layout(kLayoutDst);
    tensor_layout_any kSrcModified  = word_layout_from_bit_layout(kLayoutSrc);
    // Get a bit mask for the last element of each "column" of bits.
    // (When copying a tensor of type CUPHY_BIT, we need to make sure
    // that we don't copy the entire word unless the size is a multiple
    // of 32.)
    const uint32_t    COL_MASK      = bit_column_end_mask(kLayoutDst.dimensions[0]);
    //printf("NUM_ROWS = %i, COL_MASK = 0x%X\n", NUM_ROWS, COL_MASK);
    convert_bits_to_bits_kernel<<<gridDim, blockDim, 0, strm>>>(kDstModified,
                                                                static_cast<uint32_t*>(dstAddr),
                                                                COL_MASK,
                                                                kSrcModified,
                                                                static_cast<const uint32_t*>(srcAddr));
    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// convert_from_bits_kernel()
template <typename Tdst>
__global__ void convert_from_bits_kernel(tensor_layout_any layoutDst,
                                         Tdst*             dst,
                                         tensor_layout_any layoutSrcWords,
                                         const uint32_t*   src)
{
    static_assert(CUPHY_DIM_MAX == 5, "convert_from_bits_kernel only defined for 5D tensors");
    for(int i4 = 0; i4 < layoutDst.dimensions[4]; ++i4)
    {    
        for(int i3 = 0; i3 < layoutDst.dimensions[3]; ++i3)
        {
            for(grid_stride_index<2> it2; it2 < layoutDst.dimensions[2]; it2.next())
            {
                for(grid_stride_index<1> it1; it1 < layoutDst.dimensions[1]; it1.next())
                {
                    for(grid_stride_index<0> it0; it0 < layoutDst.dimensions[0]; it0.next())
                    {
                        int    nout[5]    = {static_cast<int>(it0.value),
                                            static_cast<int>(it1.value),
                                            static_cast<int>(it2.value),
                                            i3,
                                            i4};
                        // Convert the index associated with the output to an equivalent
                        // index assuming a uint32_t word based input. We will retrieve
                        // the entire 32-bit value and extract the individual bit.
                        int    inputBit   = static_cast<int>(it0.value) % 32;
                        int    nin[5]     = {static_cast<int>(it0.value) / 32,
                                            static_cast<int>(it1.value),
                                            static_cast<int>(it2.value),
                                            i3,
                                            i4};
                        size_t out_idx    = layoutDst.offset(nout);
                        size_t in_idx     = layoutSrcWords.offset(nin);
                        // printf("(i0, i1, i2, i3) = (%u, %u, %u %u), src offset = %lu, dst offset = %lu\n",
                        //        it0.value,
                        //        it1.value,
                        //        it2.value,
                        //        i3,
                        //        in_idx,
                        //        out_idx);
                        dst[out_idx] = static_cast<Tdst>((0 == (src[in_idx] & (1 << inputBit))) ? 0 : 1);
                    } // it0
                }     // it1
            }         // it2
        }             // it3
    }                 // it4
}

////////////////////////////////////////////////////////////////////////
// launch_convert_from_bits()
template <cuphyDataType_t Tdst>
cuphyStatus_t launch_convert_from_bits(const tensor_layout_any& kLayoutDst,
                                       void*                    dstAddr,
                                       const tensor_layout_any& kLayoutSrc,
                                       const void*              srcAddr,
                                       const dim3               gridDim,
                                       const dim3               blockDim,
                                       cudaStream_t             strm)
{
    typedef typename data_type_traits<Tdst>::type dst_type_t;
    // Create a modified tensor layout that describes the uint32_t
    // input word tensor.
    tensor_layout_any kSrcModified = word_layout_from_bit_layout(kLayoutSrc);
    convert_from_bits_kernel<dst_type_t><<<gridDim, blockDim, 0, strm>>>(kLayoutDst,
                                                                         static_cast<dst_type_t*>(dstAddr),
                                                                         kSrcModified,
                                                                         static_cast<const uint32_t*>(srcAddr));
    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// converts_to_zero_bit()
template <typename TSrc>
__device__ __forceinline__ bool converts_to_zero_bit(TSrc value)
{
    return value == 0;
}

template <>
__device__ __forceinline__ bool converts_to_zero_bit(__half value)
{
    return (static_cast<__half_raw>(value).x & 0x7FFFU) == 0;
}

template <>
__device__ __forceinline__ bool converts_to_zero_bit(__nv_fp8_e4m3 value)
{
    return (value.__x & 0x7FU) == 0;
}

template <>
__device__ __forceinline__ bool converts_to_zero_bit(__nv_fp8_e5m2 value)
{
    return (value.__x & 0x7FU) == 0;
}

////////////////////////////////////////////////////////////////////////
// convert_to_bits_kernel()
template <typename TSrc>
__global__ void convert_to_bits_kernel(tensor_layout_any layoutDstWords,
                                       uint32_t*         dst,
                                       tensor_layout_any layoutSrc,
                                       const TSrc*       src)
{
    static_assert(CUPHY_DIM_MAX == 5, "convert_to_bits_kernel only defined for 5D tensors");
    for(int i4 = 0; i4 < layoutDstWords.dimensions[4]; ++i4)
    {    
        for(int i3 = 0; i3 < layoutDstWords.dimensions[3]; ++i3)
        {
            for(grid_stride_index<2> it2; it2 < layoutDstWords.dimensions[2]; it2.next())
            {
                for(grid_stride_index<1> it1; it1 < layoutDstWords.dimensions[1]; it1.next())
                {
                    for(grid_stride_index<0> it0; it0 < layoutDstWords.dimensions[0]; it0.next())
                    {
                        int    nout[5]    = {static_cast<int>(it0.value),
                                            static_cast<int>(it1.value),
                                            static_cast<int>(it2.value),
                                            i3,
                                            i4};
                        uint32_t out_value = 0;
                        // Retrieve 32 input values and convert them to the bits of a
                        // single output word. Use a hard "equals 0" condition for
                        // setting the output bit.
                        for(int i = 0; i < 32; ++i)
                        {
                            int src_idx = (static_cast<int>(it0.value) * 32) + i;
                            // Word tensors are rounded up to 32 bits. Avoid reading
                            // from the source tensor for padded bits.
                            if(src_idx < layoutSrc.dimensions[0])
                            {
                                int      nin[5]    = {src_idx,
                                                    static_cast<int>(it1.value),
                                                    static_cast<int>(it2.value),
                                                    i3,
                                                    i4};
                                size_t   in_idx  = layoutSrc.offset(nin);
                                uint32_t src_bit = converts_to_zero_bit(src[in_idx]) ? 0 : 1;
                                out_value |= (src_bit << i);
                            }
                        }
                        size_t out_idx = layoutDstWords.offset(nout);
                        dst[out_idx]   = out_value;
                    } // it0
                }     // it1
            }         // it2
        }             // it3
    }                 // it4
}

////////////////////////////////////////////////////////////////////////
// launch_convert_to_bits()
template <cuphyDataType_t TSrc>
cuphyStatus_t launch_convert_to_bits(const tensor_layout_any& kLayoutDst,
                                     void*                    dstAddr,
                                     const tensor_layout_any& kLayoutSrc,
                                     const void*              srcAddr,
                                     const dim3               gridDim,
                                     const dim3               blockDim,
                                     cudaStream_t             strm)
{
    typedef typename data_type_traits<TSrc>::type src_type_t;
    // Create a modified tensor layout that describes the uint32_t
    // output word tensor.
    tensor_layout_any kDstModified = word_layout_from_bit_layout(kLayoutDst);
    convert_to_bits_kernel<src_type_t><<<gridDim, blockDim, 0, strm>>>(kDstModified,
                                                                       static_cast<uint32_t*>(dstAddr),
                                                                       kLayoutSrc,
                                                                       static_cast<const src_type_t*>(srcAddr));
    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// convert_tensor_layout()
cuphyStatus_t convert_tensor_layout(const tensor_desc& dstTensorDesc,
                                    void*              dstAddr,
                                    const tensor_desc& srcTensorDesc,
                                    const void*        srcAddr,
                                    cudaStream_t       strm)
{
    // printf("convert_tensor_layout(): src[0] = %i, src[1] = %i, src[2] = %i\n",
    //       srcTensorDesc.layout().dimensions[0],
    //       srcTensorDesc.layout().dimensions[1],
    //       srcTensorDesc.layout().dimensions[2]);
    cuphyStatus_t s = CUPHY_STATUS_INVALID_CONVERSION;
    //------------------------------------------------------------------
    // Create kernel structures representing the layout
    const tensor_layout_any& kLayoutSrc = srcTensorDesc.layout();
    const tensor_layout_any& kLayoutDst = dstTensorDesc.layout();
    cuphyDataType_t          dstType    = dstTensorDesc.type();
    cuphyDataType_t          srcType    = srcTensorDesc.type();
    // TODO: Make the grid size larger, and verify correctness. Swapping
    // dimensions might also provide performance benefits.
    dim3 gridDim(1);
    dim3 blockDim(32, 32);
    // clang-format off
    // Switch on destination and source types
    switch(dstType)
    {
    //------------------------------------------------------------------
    case CUPHY_BIT:
        switch(srcType)
        {
        // Floating-point types must retain their type here so positive and
        // negative zero both compare equal to zero.
        case CUPHY_R_8I:      // fall through...
        case CUPHY_R_8U:  s = launch_convert_to_bits<CUPHY_R_8U> (kLayoutDst, dstAddr, kLayoutSrc, srcAddr, gridDim, blockDim, strm); break;
        case CUPHY_R_16I: // fall through...
        case CUPHY_R_16U: s = launch_convert_to_bits<CUPHY_R_16U>(kLayoutDst, dstAddr, kLayoutSrc, srcAddr, gridDim, blockDim, strm); break;
        case CUPHY_R_32I: // fall through...
        case CUPHY_R_32U: s = launch_convert_to_bits<CUPHY_R_32U>(kLayoutDst, dstAddr, kLayoutSrc, srcAddr, gridDim, blockDim, strm); break;
        case CUPHY_R_16F:     s = launch_convert_to_bits<CUPHY_R_16F>    (kLayoutDst, dstAddr, kLayoutSrc, srcAddr, gridDim, blockDim, strm); break;
        case CUPHY_R_32F:     s = launch_convert_to_bits<CUPHY_R_32F>    (kLayoutDst, dstAddr, kLayoutSrc, srcAddr, gridDim, blockDim, strm); break;
        case CUPHY_R_64F: s = launch_convert_to_bits<CUPHY_R_64F>(kLayoutDst, dstAddr, kLayoutSrc, srcAddr, gridDim, blockDim, strm); break;
        case CUPHY_R_8F_E4M3: s = launch_convert_to_bits<CUPHY_R_8F_E4M3>(kLayoutDst, dstAddr, kLayoutSrc, srcAddr, gridDim, blockDim, strm); break;
        case CUPHY_R_8F_E5M2: s = launch_convert_to_bits<CUPHY_R_8F_E5M2>(kLayoutDst, dstAddr, kLayoutSrc, srcAddr, gridDim, blockDim, strm); break;
        case CUPHY_BIT:   s = launch_convert_bits_to_bits        (kLayoutDst, dstAddr, kLayoutSrc, srcAddr, gridDim, blockDim, strm); break;
        default:                                                                                                                      break;
        }
        break;
    //------------------------------------------------------------------
    case CUPHY_R_8I:
        switch(srcType)
        {
        case CUPHY_BIT:   s = launch_convert_from_bits<CUPHY_R_8I>   (kLayoutDst, dstAddr, kLayoutSrc, srcAddr, gridDim, blockDim, strm); break;
        case CUPHY_R_8I:  s = launch_convert<CUPHY_R_8I, CUPHY_R_8I> (kLayoutDst, dstAddr, kLayoutSrc, srcAddr, gridDim, blockDim, strm); break;
        case CUPHY_R_16F: s = launch_convert<CUPHY_R_8I, CUPHY_R_16F>(kLayoutDst, dstAddr, kLayoutSrc, srcAddr, gridDim, blockDim, strm); break;
        case CUPHY_R_32F: s = launch_convert<CUPHY_R_8I, CUPHY_R_32F>(kLayoutDst, dstAddr, kLayoutSrc, srcAddr, gridDim, blockDim, strm); break;
        case CUPHY_R_64F: s = launch_convert<CUPHY_R_8I, CUPHY_R_64F>(kLayoutDst, dstAddr, kLayoutSrc, srcAddr, gridDim, blockDim, strm); break;
        default:                                                                                                                          break;
        }
        break;
    //------------------------------------------------------------------
    case CUPHY_C_8I:
        switch(srcType)
        {
        case CUPHY_C_8I: s = launch_convert<CUPHY_C_8I, CUPHY_C_8I>(kLayoutDst, dstAddr, kLayoutSrc, srcAddr, gridDim, blockDim, strm); break;
        default:                                                                                                                        break;

        }
        break;
    //------------------------------------------------------------------
    case CUPHY_R_8U:
        switch(srcType)
        {
        case CUPHY_BIT:  s = launch_convert_from_bits<CUPHY_R_8U>  (kLayoutDst, dstAddr, kLayoutSrc, srcAddr, gridDim, blockDim, strm); break;
        case CUPHY_R_8U: s = launch_convert<CUPHY_R_8U, CUPHY_R_8U>(kLayoutDst, dstAddr, kLayoutSrc, srcAddr, gridDim, blockDim, strm); break;
        default:                                                                                                                        break;
        }
        break;
    //------------------------------------------------------------------
    case CUPHY_C_8U:
        switch(srcType)
        {
        case CUPHY_C_8U: s = launch_convert<CUPHY_C_8U, CUPHY_C_8U>(kLayoutDst, dstAddr, kLayoutSrc, srcAddr, gridDim, blockDim, strm); break;
        default:                                                                                                                        break;
        }
        break;
    //------------------------------------------------------------------
    case CUPHY_R_16I:
        switch(srcType)
        {
        case CUPHY_BIT:   s = launch_convert_from_bits<CUPHY_R_16I>   (kLayoutDst, dstAddr, kLayoutSrc, srcAddr, gridDim, blockDim, strm); break;
        case CUPHY_R_8I:  s = launch_convert<CUPHY_R_16I, CUPHY_R_8I> (kLayoutDst, dstAddr, kLayoutSrc, srcAddr, gridDim, blockDim, strm); break;
        case CUPHY_R_16I: s = launch_convert<CUPHY_R_16I, CUPHY_R_16I>(kLayoutDst, dstAddr, kLayoutSrc, srcAddr, gridDim, blockDim, strm); break;
        default:                                                                                                                           break;
        }
        break;
    //------------------------------------------------------------------
    case CUPHY_C_16I:
        switch(srcType)
        {
        case CUPHY_C_8I:  s = launch_convert<CUPHY_C_16I, CUPHY_C_8I> (kLayoutDst, dstAddr, kLayoutSrc, srcAddr, gridDim, blockDim, strm); break;
        case CUPHY_C_16I: s = launch_convert<CUPHY_C_16I, CUPHY_C_16I>(kLayoutDst, dstAddr, kLayoutSrc, srcAddr, gridDim, blockDim, strm); break;
        default:                                                                                                                           break;
        }
        break;
    //------------------------------------------------------------------
    case CUPHY_R_16U:
        switch(srcType)
        {
        case CUPHY_BIT:   s = launch_convert_from_bits<CUPHY_R_16U>   (kLayoutDst, dstAddr, kLayoutSrc, srcAddr, gridDim, blockDim, strm); break;
        case CUPHY_R_8U:  s = launch_convert<CUPHY_R_16U, CUPHY_R_8U> (kLayoutDst, dstAddr, kLayoutSrc, srcAddr, gridDim, blockDim, strm); break;
        case CUPHY_R_16U: s = launch_convert<CUPHY_R_16U, CUPHY_R_16U>(kLayoutDst, dstAddr, kLayoutSrc, srcAddr, gridDim, blockDim, strm); break;
        default:                                                                                                                           break;
        }
        break;
    //------------------------------------------------------------------
    case CUPHY_C_16U:
        switch(srcType)
        {
        case CUPHY_C_8U:  s = launch_convert<CUPHY_C_16U, CUPHY_C_8U> (kLayoutDst, dstAddr, kLayoutSrc, srcAddr, gridDim, blockDim, strm); break;
        case CUPHY_C_16U: s = launch_convert<CUPHY_C_16U, CUPHY_C_16U>(kLayoutDst, dstAddr, kLayoutSrc, srcAddr, gridDim, blockDim, strm); break;
        default:                                                                                                                           break;
        }
        break;
    //------------------------------------------------------------------
    case CUPHY_R_32I:
        switch(srcType)
        {
        case CUPHY_BIT:   s = launch_convert_from_bits<CUPHY_R_32I>   (kLayoutDst, dstAddr, kLayoutSrc, srcAddr, gridDim, blockDim, strm); break;
        case CUPHY_R_8I:  s = launch_convert<CUPHY_R_32I, CUPHY_R_8I> (kLayoutDst, dstAddr, kLayoutSrc, srcAddr, gridDim, blockDim, strm); break;
        case CUPHY_R_16I: s = launch_convert<CUPHY_R_32I, CUPHY_R_16I>(kLayoutDst, dstAddr, kLayoutSrc, srcAddr, gridDim, blockDim, strm); break;
        case CUPHY_R_32I: s = launch_convert<CUPHY_R_32I, CUPHY_R_32I>(kLayoutDst, dstAddr, kLayoutSrc, srcAddr, gridDim, blockDim, strm); break;
        default:                                                                                                                           break;
        }
        break;
    //------------------------------------------------------------------
    case CUPHY_C_32I:
        switch(srcType)
        {
        case CUPHY_C_8I:  s = launch_convert<CUPHY_C_32I, CUPHY_C_8I> (kLayoutDst, dstAddr, kLayoutSrc, srcAddr, gridDim, blockDim, strm); break;
        case CUPHY_C_16I: s = launch_convert<CUPHY_C_32I, CUPHY_C_16I>(kLayoutDst, dstAddr, kLayoutSrc, srcAddr, gridDim, blockDim, strm); break;
        case CUPHY_C_32I: s = launch_convert<CUPHY_C_32I, CUPHY_C_32I>(kLayoutDst, dstAddr, kLayoutSrc, srcAddr, gridDim, blockDim, strm); break;
        default:                                                                                                                           break;
        }
        break;
    //------------------------------------------------------------------
    case CUPHY_R_32U:
        switch(srcType)
        {
        case CUPHY_BIT:   s = launch_convert_from_bits<CUPHY_R_32U>   (kLayoutDst, dstAddr, kLayoutSrc, srcAddr, gridDim, blockDim, strm); break;
        case CUPHY_R_8U:  s = launch_convert<CUPHY_R_32U, CUPHY_R_8U> (kLayoutDst, dstAddr, kLayoutSrc, srcAddr, gridDim, blockDim, strm); break;
        case CUPHY_R_16U: s = launch_convert<CUPHY_R_32U, CUPHY_R_16U>(kLayoutDst, dstAddr, kLayoutSrc, srcAddr, gridDim, blockDim, strm); break;
        case CUPHY_R_32U: s = launch_convert<CUPHY_R_32U, CUPHY_R_32U>(kLayoutDst, dstAddr, kLayoutSrc, srcAddr, gridDim, blockDim, strm); break;
        default:                                                                                                                           break;
        }
        break;
    //------------------------------------------------------------------
    case CUPHY_C_32U:
        switch(srcType)
        {
        case CUPHY_C_8U:  s = launch_convert<CUPHY_C_32U, CUPHY_C_8U> (kLayoutDst, dstAddr, kLayoutSrc, srcAddr, gridDim, blockDim, strm); break;
        case CUPHY_C_16U: s = launch_convert<CUPHY_C_32U, CUPHY_C_16U>(kLayoutDst, dstAddr, kLayoutSrc, srcAddr, gridDim, blockDim, strm); break;
        case CUPHY_C_32U: s = launch_convert<CUPHY_C_32U, CUPHY_C_32U>(kLayoutDst, dstAddr, kLayoutSrc, srcAddr, gridDim, blockDim, strm); break;
        default:                                                                                                                           break;
        }
        break;
    //------------------------------------------------------------------
    case CUPHY_R_16F:
        switch(srcType)
        {
        case CUPHY_BIT:       s = launch_convert_from_bits<CUPHY_R_16F>       (kLayoutDst, dstAddr, kLayoutSrc, srcAddr, gridDim, blockDim, strm); break;
        case CUPHY_R_16F:     s = launch_convert<CUPHY_R_16F, CUPHY_R_16F>    (kLayoutDst, dstAddr, kLayoutSrc, srcAddr, gridDim, blockDim, strm); break;
        case CUPHY_R_32F:     s = launch_convert<CUPHY_R_16F, CUPHY_R_32F>    (kLayoutDst, dstAddr, kLayoutSrc, srcAddr, gridDim, blockDim, strm); break;
        case CUPHY_R_64F:     s = launch_convert<CUPHY_R_16F, CUPHY_R_64F>    (kLayoutDst, dstAddr, kLayoutSrc, srcAddr, gridDim, blockDim, strm); break;
        case CUPHY_R_8F_E4M3: s = launch_convert<CUPHY_R_16F, CUPHY_R_8F_E4M3>(kLayoutDst, dstAddr, kLayoutSrc, srcAddr, gridDim, blockDim, strm); break;
        case CUPHY_R_8F_E5M2: s = launch_convert<CUPHY_R_16F, CUPHY_R_8F_E5M2>(kLayoutDst, dstAddr, kLayoutSrc, srcAddr, gridDim, blockDim, strm); break;
        case CUPHY_R_8I:      s = launch_convert<CUPHY_R_16F, CUPHY_R_8I>     (kLayoutDst, dstAddr, kLayoutSrc, srcAddr, gridDim, blockDim, strm); break;
        default:                                                                                                                                   break;
        }
        break;
    //------------------------------------------------------------------
    case CUPHY_C_16F:
        switch(srcType)
        {
        case CUPHY_C_16F: s = launch_convert<CUPHY_C_16F, CUPHY_C_16F>(kLayoutDst, dstAddr, kLayoutSrc, srcAddr, gridDim, blockDim, strm); break;
        case CUPHY_C_32F: s = launch_convert<CUPHY_C_16F, CUPHY_C_32F>(kLayoutDst, dstAddr, kLayoutSrc, srcAddr, gridDim, blockDim, strm); break;
        default:                                                                                                                           break;
        }
        break;
    //------------------------------------------------------------------
    case CUPHY_R_32F:
        switch(srcType)
        {
        case CUPHY_BIT:       s = launch_convert_from_bits<CUPHY_R_32F>       (kLayoutDst, dstAddr, kLayoutSrc, srcAddr, gridDim, blockDim, strm); break;
        case CUPHY_R_16F:     s = launch_convert<CUPHY_R_32F, CUPHY_R_16F>    (kLayoutDst, dstAddr, kLayoutSrc, srcAddr, gridDim, blockDim, strm); break;
        case CUPHY_R_32F:     s = launch_convert<CUPHY_R_32F, CUPHY_R_32F>    (kLayoutDst, dstAddr, kLayoutSrc, srcAddr, gridDim, blockDim, strm); break;
        case CUPHY_R_64F:     s = launch_convert<CUPHY_R_32F, CUPHY_R_64F>    (kLayoutDst, dstAddr, kLayoutSrc, srcAddr, gridDim, blockDim, strm); break;
        case CUPHY_R_8F_E4M3: s = launch_convert<CUPHY_R_32F, CUPHY_R_8F_E4M3>(kLayoutDst, dstAddr, kLayoutSrc, srcAddr, gridDim, blockDim, strm); break;
        case CUPHY_R_8F_E5M2: s = launch_convert<CUPHY_R_32F, CUPHY_R_8F_E5M2>(kLayoutDst, dstAddr, kLayoutSrc, srcAddr, gridDim, blockDim, strm); break;
        case CUPHY_R_8I:      s = launch_convert<CUPHY_R_32F, CUPHY_R_8I>     (kLayoutDst, dstAddr, kLayoutSrc, srcAddr, gridDim, blockDim, strm); break;
        default:                                                                                                                                   break;
        }
        break;
    //------------------------------------------------------------------
    case CUPHY_C_32F:
        switch(srcType)
        {
        case CUPHY_C_16F: s = launch_convert<CUPHY_C_32F, CUPHY_C_16F>(kLayoutDst, dstAddr, kLayoutSrc, srcAddr, gridDim, blockDim, strm); break;
        case CUPHY_C_32F: s = launch_convert<CUPHY_C_32F, CUPHY_C_32F>(kLayoutDst, dstAddr, kLayoutSrc, srcAddr, gridDim, blockDim, strm); break;
        case CUPHY_C_64F: s = launch_convert<CUPHY_C_32F, CUPHY_C_64F>(kLayoutDst, dstAddr, kLayoutSrc, srcAddr, gridDim, blockDim, strm); break;
        default:                                                                                                                           break;

        }
        break;
    //------------------------------------------------------------------
    case CUPHY_R_64F:
        switch(srcType)
        {
        case CUPHY_BIT:       s = launch_convert_from_bits<CUPHY_R_64F>       (kLayoutDst, dstAddr, kLayoutSrc, srcAddr, gridDim, blockDim, strm); break;
        case CUPHY_R_16F:     s = launch_convert<CUPHY_R_64F, CUPHY_R_16F>    (kLayoutDst, dstAddr, kLayoutSrc, srcAddr, gridDim, blockDim, strm); break;
        case CUPHY_R_32F:     s = launch_convert<CUPHY_R_64F, CUPHY_R_32F>    (kLayoutDst, dstAddr, kLayoutSrc, srcAddr, gridDim, blockDim, strm); break;
        case CUPHY_R_64F:     s = launch_convert<CUPHY_R_64F, CUPHY_R_64F>    (kLayoutDst, dstAddr, kLayoutSrc, srcAddr, gridDim, blockDim, strm); break;
        case CUPHY_R_8F_E4M3: s = launch_convert<CUPHY_R_64F, CUPHY_R_8F_E4M3>(kLayoutDst, dstAddr, kLayoutSrc, srcAddr, gridDim, blockDim, strm); break;
        case CUPHY_R_8F_E5M2: s = launch_convert<CUPHY_R_64F, CUPHY_R_8F_E5M2>(kLayoutDst, dstAddr, kLayoutSrc, srcAddr, gridDim, blockDim, strm); break;
        case CUPHY_R_8I:      s = launch_convert<CUPHY_R_64F, CUPHY_R_8I>     (kLayoutDst, dstAddr, kLayoutSrc, srcAddr, gridDim, blockDim, strm); break;
        default:                                                                                                                                   break;
        }
        break;
    //------------------------------------------------------------------
    case CUPHY_C_64F:
        switch(srcType)
        {
        case CUPHY_C_16F: s = launch_convert<CUPHY_C_64F, CUPHY_C_16F>(kLayoutDst, dstAddr, kLayoutSrc, srcAddr, gridDim, blockDim, strm); break;
        case CUPHY_C_32F: s = launch_convert<CUPHY_C_64F, CUPHY_C_32F>(kLayoutDst, dstAddr, kLayoutSrc, srcAddr, gridDim, blockDim, strm); break;
        case CUPHY_C_64F: s = launch_convert<CUPHY_C_64F, CUPHY_C_64F>(kLayoutDst, dstAddr, kLayoutSrc, srcAddr, gridDim, blockDim, strm); break;
        default:                                                                                                                           break;
        }
        break;
    //------------------------------------------------------------------
    case CUPHY_R_8F_E4M3:
        switch(srcType)
        {
        case CUPHY_BIT:       s = launch_convert_from_bits<CUPHY_R_8F_E4M3>       (kLayoutDst, dstAddr, kLayoutSrc, srcAddr, gridDim, blockDim, strm); break;
        case CUPHY_R_16F:     s = launch_convert<CUPHY_R_8F_E4M3, CUPHY_R_16F>    (kLayoutDst, dstAddr, kLayoutSrc, srcAddr, gridDim, blockDim, strm); break;
        case CUPHY_R_32F:     s = launch_convert<CUPHY_R_8F_E4M3, CUPHY_R_32F>    (kLayoutDst, dstAddr, kLayoutSrc, srcAddr, gridDim, blockDim, strm); break;
        case CUPHY_R_64F:     s = launch_convert<CUPHY_R_8F_E4M3, CUPHY_R_64F>    (kLayoutDst, dstAddr, kLayoutSrc, srcAddr, gridDim, blockDim, strm); break;
        case CUPHY_R_8F_E4M3: s = launch_convert<CUPHY_R_8F_E4M3, CUPHY_R_8F_E4M3>(kLayoutDst, dstAddr, kLayoutSrc, srcAddr, gridDim, blockDim, strm); break;
        case CUPHY_R_8F_E5M2: s = launch_convert<CUPHY_R_8F_E4M3, CUPHY_R_8F_E5M2>(kLayoutDst, dstAddr, kLayoutSrc, srcAddr, gridDim, blockDim, strm); break;
        default:                                                                                                                                       break;
        }
        break;
    //------------------------------------------------------------------
    case CUPHY_R_8F_E5M2:
        switch(srcType)
        {
        case CUPHY_BIT:       s = launch_convert_from_bits<CUPHY_R_8F_E5M2>       (kLayoutDst, dstAddr, kLayoutSrc, srcAddr, gridDim, blockDim, strm); break;
        case CUPHY_R_16F:     s = launch_convert<CUPHY_R_8F_E5M2, CUPHY_R_16F>    (kLayoutDst, dstAddr, kLayoutSrc, srcAddr, gridDim, blockDim, strm); break;
        case CUPHY_R_32F:     s = launch_convert<CUPHY_R_8F_E5M2, CUPHY_R_32F>    (kLayoutDst, dstAddr, kLayoutSrc, srcAddr, gridDim, blockDim, strm); break;
        case CUPHY_R_64F:     s = launch_convert<CUPHY_R_8F_E5M2, CUPHY_R_64F>    (kLayoutDst, dstAddr, kLayoutSrc, srcAddr, gridDim, blockDim, strm); break;
        case CUPHY_R_8F_E4M3: s = launch_convert<CUPHY_R_8F_E5M2, CUPHY_R_8F_E4M3>(kLayoutDst, dstAddr, kLayoutSrc, srcAddr, gridDim, blockDim, strm); break;
        case CUPHY_R_8F_E5M2: s = launch_convert<CUPHY_R_8F_E5M2, CUPHY_R_8F_E5M2>(kLayoutDst, dstAddr, kLayoutSrc, srcAddr, gridDim, blockDim, strm); break;
        default:                                                                                                                                       break;
        }
        break;
    //------------------------------------------------------------------
    default:
        // Unknown type
        break;
    }
    // clang-format on
    if(CUPHY_STATUS_SUCCESS == s)
    {
        // If a kernel was launched, check for errors.
        cudaError_t e = cudaGetLastError();
        DEBUG_PRINTF("CUDA STATUS (%s:%i): %s\n", __FILE__, __LINE__, cudaGetErrorString(e));
        /*VCAST_DONT_INSTRUMENT_START*/
        if(e != cudaSuccess) s = CUPHY_STATUS_INTERNAL_ERROR;
        /*VCAST_DONT_INSTRUMENT_END*/
    }
    return s;
}
