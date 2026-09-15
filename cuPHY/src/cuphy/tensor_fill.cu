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

#include "tensor_fill.hpp"
#include "variant.hpp"
#include "cuphy_kernel_util.cuh"

namespace
{
////////////////////////////////////////////////////////////////////////
// tensor_fill_kernel()
// Kernel populate a tensor with a single value
template <typename T>
__global__ void tensor_fill_kernel(tensor_layout_any layoutDst,
                                   T*                dst,
                                   T                 value)
{
    static_assert(CUPHY_DIM_MAX == 5, "tensor_fill_kernel only defined for 5D tensors");
    //------------------------------------------------------------------
    //------------------------------------------------------------------
    // Loop over destination tensor (up to 5-D)
    for(int i4 = 0; i4 < layoutDst.dimensions[4]; ++i4)
    {
        for (int i3 = 0; i3 < layoutDst.dimensions[3]; ++i3)
        {
            for(grid_stride_index<2> it2; it2 < layoutDst.dimensions[2]; it2.next())
            {
                for(grid_stride_index<1> it1; it1 < layoutDst.dimensions[1]; it1.next())
                {
                    for(grid_stride_index<0> it0; it0 < layoutDst.dimensions[0]; it0.next())
                    {
                        int    n[5]    = {static_cast<int>(it0.value),
                                        static_cast<int>(it1.value),
                                        static_cast<int>(it2.value),
                                        i3,
                                        i4};
                        size_t out_idx = layoutDst.offset(n);
                        //printf("(i0, i1, i2, i3) = (%u, %u, %u %u), offset = %lu\n",
                        //       it0.value,
                        //       it1.value,
                        //       it2.value,
                        //       i3,
                        //       out_idx);
                        dst[out_idx] = value;
                    } // it0
                }     // it1
            }         // it2
        }             // it3
    }                 // it4
}

////////////////////////////////////////////////////////////////////////
// launch_fill()
template <cuphyDataType_t TType>
cuphyStatus_t launch_fill(const tensor_layout_any& tLayout,
                          void*                    dst,
                          const cuphyVariant_t&    value,
                          cudaStream_t             strm)
{
    dim3 gridDim(1);
    dim3 blockDim(1024);

    typedef typename data_type_traits<TType>::type value_t;
 
    tensor_fill_kernel<value_t><<<gridDim, blockDim, 0, strm>>>(tLayout,                             // tensor dims
                                                                static_cast<value_t*>(dst),          // output
                                                                cuphy_i::variant_as<TType>(value));  // value
    cudaError_t e = cudaGetLastError();
    DEBUG_PRINTF("CUDA STATUS (%s:%i): %s\n", __FILE__, __LINE__, cudaGetErrorString(e));
    return (e == cudaSuccess) ? CUPHY_STATUS_SUCCESS : CUPHY_STATUS_INTERNAL_ERROR;
}

} // namespace

namespace cuphy_i
{

////////////////////////////////////////////////////////////////////////
// tensor_fill()
cuphyStatus_t tensor_fill(const tensor_desc&    tdesc,
                          void*                 addr,
                          const cuphyVariant_t& var,
                          cudaStream_t          strm)
{
    //------------------------------------------------------------------
    // Create kernel structures representing the layout
    const tensor_layout_any& tLayout      = tdesc.layout();
    cuphyDataType_t          tType        = tdesc.type();
    //------------------------------------------------------------------
    // Copy fill values and convert (if necessary/possible)
    cuphyVariant_t           value = var;
    
    cuphyStatus_t            s = cuphy_i::convert_variant(value, tType);
    if(CUPHY_STATUS_SUCCESS != s) { return s; }
    //------------------------------------------------------------------
    s  = CUPHY_STATUS_UNSUPPORTED_TYPE;
    switch(tType)
    {
    case CUPHY_R_8U:      s = launch_fill<CUPHY_R_8U>     (tLayout, addr, value, strm); break;
    case CUPHY_R_8I:      s = launch_fill<CUPHY_R_8I>     (tLayout, addr, value, strm); break;
    case CUPHY_R_16U:     s = launch_fill<CUPHY_R_16U>    (tLayout, addr, value, strm); break;
    case CUPHY_R_16I:     s = launch_fill<CUPHY_R_16I>    (tLayout, addr, value, strm); break;
    case CUPHY_R_16F:     s = launch_fill<CUPHY_R_16F>    (tLayout, addr, value, strm); break;
    case CUPHY_R_32F:     s = launch_fill<CUPHY_R_32F>    (tLayout, addr, value, strm); break;
    case CUPHY_R_32U:     s = launch_fill<CUPHY_R_32U>    (tLayout, addr, value, strm); break;
    case CUPHY_R_32I:     s = launch_fill<CUPHY_R_32I>    (tLayout, addr, value, strm); break;
    case CUPHY_C_16F:     s = launch_fill<CUPHY_C_16F>    (tLayout, addr, value, strm); break;
    case CUPHY_C_32F:     s = launch_fill<CUPHY_C_32F>    (tLayout, addr, value, strm); break;
    case CUPHY_R_8F_E4M3: s = launch_fill<CUPHY_R_8F_E4M3>(tLayout, addr, value, strm); break;
    case CUPHY_R_8F_E5M2: s = launch_fill<CUPHY_R_8F_E5M2>(tLayout, addr, value, strm); break;
    default:                                                                            break;
    } // switch 

    return s;
}

} // namespace cuphy_i
