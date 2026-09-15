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

#include "cuphy.h"
#include "convert_tensor.cuh"
#include "tensor_desc.hpp"
#include "cuphy_context.hpp"


////////////////////////////////////////////////////////////////////////
// cuphyGetErrorString()
const char* cuphyGetErrorString(cuphyStatus_t status)
{ // clang-format off
    switch (status)
    {
    case CUPHY_STATUS_SUCCESS:               return "Success";
    case CUPHY_STATUS_INTERNAL_ERROR:        return "Internal error";
    case CUPHY_STATUS_NOT_SUPPORTED:         return "An operation was requested that is not currently supported";
    case CUPHY_STATUS_INVALID_ARGUMENT:      return "An invalid argument was provided";
    case CUPHY_STATUS_ARCH_MISMATCH:         return "Requested computation not supported on current architecture";
    case CUPHY_STATUS_ALLOC_FAILED:          return "Memory allocation failed";
    case CUPHY_STATUS_SIZE_MISMATCH:         return "Operand size mismatch";
    case CUPHY_STATUS_MEMCPY_ERROR:          return "Error performing memory copy";
    case CUPHY_STATUS_INVALID_CONVERSION:    return "Invalid data conversion requested";
    case CUPHY_STATUS_UNSUPPORTED_TYPE:      return "Operation requested on unsupported type";
    case CUPHY_STATUS_UNSUPPORTED_LAYOUT:    return "Operation requested on unsupported tensor layout";
    case CUPHY_STATUS_UNSUPPORTED_RANK:      return "Operation requested on unsupported rank";
    case CUPHY_STATUS_UNSUPPORTED_CONFIG:    return "Operation requested using an unsupported configuration";
    case CUPHY_STATUS_UNSUPPORTED_ALIGNMENT: return "One or more API arguments don't have the required alignment";
    case CUPHY_STATUS_VALUE_OUT_OF_RANGE:    return "Data conversion could not occur because an input value was out of range";
    case CUPHY_STATUS_REF_MISMATCH:          return "Mismatch found when comparing to TV";
    default:                                 return "Unknown status value";
    }
} // clang-format on

////////////////////////////////////////////////////////////////////////
// cuphyGetErrorName()
const char* cuphyGetErrorName(cuphyStatus_t status)
{ // clang-format off
    switch (status)
    {
    case CUPHY_STATUS_SUCCESS:               return "CUPHY_STATUS_SUCCESS";
    case CUPHY_STATUS_INTERNAL_ERROR:        return "CUPHY_STATUS_INTERNAL_ERROR";
    case CUPHY_STATUS_NOT_SUPPORTED:         return "CUPHY_STATUS_NOT_SUPPORTED";
    case CUPHY_STATUS_INVALID_ARGUMENT:      return "CUPHY_STATUS_INVALID_ARGUMENT";
    case CUPHY_STATUS_ARCH_MISMATCH:         return "CUPHY_STATUS_ARCH_MISMATCH";
    case CUPHY_STATUS_ALLOC_FAILED:          return "CUPHY_STATUS_ALLOC_FAILED";
    case CUPHY_STATUS_SIZE_MISMATCH:         return "CUPHY_STATUS_SIZE_MISMATCH";
    case CUPHY_STATUS_MEMCPY_ERROR:          return "CUPHY_STATUS_MEMCPY_ERROR";
    case CUPHY_STATUS_INVALID_CONVERSION:    return "CUPHY_STATUS_INVALID_CONVERSION";
    case CUPHY_STATUS_UNSUPPORTED_TYPE:      return "CUPHY_STATUS_UNSUPPORTED_TYPE";
    case CUPHY_STATUS_UNSUPPORTED_LAYOUT:    return "CUPHY_STATUS_UNSUPPORTED_LAYOUT";
    case CUPHY_STATUS_UNSUPPORTED_RANK:      return "CUPHY_STATUS_UNSUPPORTED_RANK";
    case CUPHY_STATUS_UNSUPPORTED_CONFIG:    return "CUPHY_STATUS_UNSUPPORTED_CONFIG";
    case CUPHY_STATUS_UNSUPPORTED_ALIGNMENT: return "CUPHY_STATUS_UNSUPPORTED_ALIGNMENT";
    case CUPHY_STATUS_VALUE_OUT_OF_RANGE:    return "CUPHY_STATUS_VALUE_OUT_OF_RANGE";
    default:                                 return "CUPHY_UNKNOWN_STATUS";
    }
} // clang-format on

////////////////////////////////////////////////////////////////////////
// cuphyGetDataTypeString()
const char* CUPHYWINAPI cuphyGetDataTypeString(cuphyDataType_t t)
{ // clang-format off
    switch(t)
    {
    case CUPHY_VOID:      return "CUPHY_VOID";
    case CUPHY_BIT:       return "CUPHY_BIT";
    case CUPHY_R_16F:     return "CUPHY_R_16F";
    case CUPHY_C_16F:     return "CUPHY_C_16F";
    case CUPHY_R_32F:     return "CUPHY_R_32F";
    case CUPHY_C_32F:     return "CUPHY_C_32F";
    case CUPHY_R_8I:      return "CUPHY_R_8I";
    case CUPHY_C_8I:      return "CUPHY_C_8I";
    case CUPHY_R_8U:      return "CUPHY_R_8U";
    case CUPHY_C_8U:      return "CUPHY_C_8U";
    case CUPHY_R_16I:     return "CUPHY_R_16I";
    case CUPHY_C_16I:     return "CUPHY_C_16I";
    case CUPHY_R_16U:     return "CUPHY_R_16U";
    case CUPHY_C_16U:     return "CUPHY_C_16U";
    case CUPHY_R_32I:     return "CUPHY_R_32I";
    case CUPHY_C_32I:     return "CUPHY_C_32I";
    case CUPHY_R_32U:     return "CUPHY_R_32U";
    case CUPHY_C_32U:     return "CUPHY_C_32U";
    case CUPHY_R_64F:     return "CUPHY_R_64F";
    case CUPHY_C_64F:     return "CUPHY_C_64F";
    case CUPHY_R_8F_E4M3: return "CUPHY_R_8F_E4M3";
    case CUPHY_R_8F_E5M2: return "CUPHY_R_8F_E5M2";
    default:              return "UNKNOWN_TYPE";
    }
} // clang-format on

////////////////////////////////////////////////////////////////////////
// cuphyCreateContext()
cuphyStatus_t CUPHYWINAPI cuphyCreateContext(cuphyContext_t* pcontext,
                                             unsigned int    flags)
{
    if(!pcontext)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    *pcontext = nullptr;
    try
    {
        cuphy_i::context* c = new cuphy_i::context;
        *pcontext           = static_cast<cuphyContext*>(c);
    }
    catch(std::bad_alloc& eba)
    {
        return CUPHY_STATUS_ALLOC_FAILED;
    }
    catch(...)
    {
        return CUPHY_STATUS_INTERNAL_ERROR;
    }
    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphyDestroyContext()
cuphyStatus_t CUPHYWINAPI cuphyDestroyContext(cuphyContext_t ctx)
{
    if(!ctx)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    cuphy_i::context* c = static_cast<cuphy_i::context*>(ctx);
    delete c;
    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphyCreateTensorDescriptor()
cuphyStatus_t cuphyCreateTensorDescriptor(cuphyTensorDescriptor_t* tensorDesc)
{
    //------------------------------------------------------------------
    // Validate arguments
    if(nullptr == tensorDesc)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    //------------------------------------------------------------------
    // Allocate the descriptor structure
    tensor_desc* tdesc = new(std::nothrow) tensor_desc;
    if(nullptr == tdesc)
    {
        return CUPHY_STATUS_ALLOC_FAILED;
    }
    //------------------------------------------------------------------
    // Populate the return address
    *tensorDesc = tdesc;
    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphyDestroyTensorDescriptor()
cuphyStatus_t CUPHYWINAPI cuphyDestroyTensorDescriptor(cuphyTensorDescriptor_t tensorDesc)
{
    //------------------------------------------------------------------
    // Validate arguments
    if(nullptr == tensorDesc)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    //------------------------------------------------------------------
    // Free the structure previously allocated by cuphyCreateTensorDescriptor()
    tensor_desc* tdesc = static_cast<tensor_desc*>(tensorDesc);
    delete tdesc;
    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphyGetTensorDescriptor()
cuphyStatus_t CUPHYWINAPI cuphyGetTensorDescriptor(cuphyTensorDescriptor_t tensorDesc,
                                                   int                     numDimsRequested,
                                                   cuphyDataType_t*        dataType,
                                                   int*                    numDims,
                                                   int                     dimensions[],
                                                   int                     strides[])
{
    //------------------------------------------------------------------
    // Validate arguments
    if((nullptr == tensorDesc) ||
       ((numDimsRequested > 0) && (nullptr == dimensions)))
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    //------------------------------------------------------------------
    const tensor_desc* tdesc = static_cast<const tensor_desc*>(tensorDesc);
    if(dataType)
    {
        *dataType = tdesc->type();
    }
    if(numDims)
    {
        *numDims = tdesc->layout().rank();
    }
    if((numDimsRequested > 0) && dimensions)
    {
        std::copy(tdesc->layout().dimensions.begin(),
                  tdesc->layout().dimensions.begin() + numDimsRequested,
                  dimensions);
    }
    if((numDimsRequested > 0) && strides)
    {
        std::copy(tdesc->layout().strides.begin(),
                  tdesc->layout().strides.begin() + numDimsRequested,
                  strides);
    }
    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphySetTensorDescriptor()
cuphyStatus_t CUPHYWINAPI cuphySetTensorDescriptor(cuphyTensorDescriptor_t tensorDesc,
                                                   cuphyDataType_t         type,
                                                   int                     numDim,
                                                   const int               dim[],
                                                   const int               str[],
                                                   unsigned int            flags)
{
    //-----------------------------------------------------------------
    // Validate arguments. Validation of dimension/stride values will
    // occur in the call below.
    // Tensor descriptor must be non-NULL.
    // Dimensions array must be non-NULL.
    // Data type must not be void
    if(!tensorDesc || !dim || (type == CUPHY_VOID))
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    //-----------------------------------------------------------------
    // If the user passed TIGHT, we use nullptr as an argument to the
    // internal function.
    const int* strArg = is_set(flags, CUPHY_TENSOR_ALIGN_TIGHT) ? nullptr : str;
    //-----------------------------------------------------------------
    // Adjust the strides array using any optional flags. Adjusting for
    // alignment only makes sense when the number of dimensions is
    // greater than 1.
    std::array<int, CUPHY_DIM_MAX> userStrides;

    if(is_set(flags, CUPHY_TENSOR_ALIGN_COALESCE) &&
       (!is_set(flags, CUPHY_TENSOR_ALIGN_TIGHT)) &&
       (numDim > 1))
    {
        // Use the given array of strides as indices into the dimension
        // vector to determine the actual strides.
        userStrides[0] = 1;
        for(int i = 1; i < numDim; ++i)
        {
            userStrides[i] = dim[i - 1] * userStrides[i - 1];
        }
        // Adjust the alignment if necessary
        if(is_set(flags, CUPHY_TENSOR_ALIGN_COALESCE))
        {
            const int COALESCE_BYTES   = 128;
            const int num_elem_aligned = round_up_to_next(dim[0],
                                                          get_element_multiple_for_alignment(COALESCE_BYTES, type));
            userStrides[1]             = num_elem_aligned;
            for(int i = 2; i < numDim; ++i)
            {
                userStrides[i] = dim[i - 1] * userStrides[i - 1];
            }
        }
        // Use the populated array to set the tensor descriptor below
        strArg = userStrides.data();
    }
    //-----------------------------------------------------------------
    // Modify the tensor descriptor using the given arguments
    tensor_desc& tdesc = static_cast<tensor_desc&>(*tensorDesc);
    return tdesc.set(type, numDim, dim, strArg) ? CUPHY_STATUS_SUCCESS : CUPHY_STATUS_INVALID_ARGUMENT;
}

////////////////////////////////////////////////////////////////////////
// cuphyGetTensorSizeInBytes()
cuphyStatus_t CUPHYWINAPI cuphyGetTensorSizeInBytes(cuphyTensorDescriptor_t tensorDesc,
                                                    size_t*                 psz)
{
    //-----------------------------------------------------------------
    // Validate arguments
    if(!tensorDesc || !psz)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    tensor_desc& tdesc = static_cast<tensor_desc&>(*tensorDesc);
    *psz               = tdesc.get_size_in_bytes();
    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphyConvertTensor()
cuphyStatus_t CUPHYWINAPI cuphyConvertTensor(cuphyTensorDescriptor_t tensorDescDst,
                                             void*                   dstAddr,
                                             cuphyTensorDescriptor_t tensorDescSrc,
                                             const void*             srcAddr,
                                             cudaStream_t            strm)
{
    //------------------------------------------------------------------
    // Validate arguments
    if(!tensorDescDst || !tensorDescSrc || !dstAddr || !srcAddr)
    {
        NVLOGE_FMT(NVLOG_TAG_BASE_CUPHY, AERIAL_CUPHY_EVENT, "{} Invalid input or output data", __FUNCTION__);
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    tensor_desc&       tdDst = static_cast<tensor_desc&>(*tensorDescDst);
    const tensor_desc& tdSrc = static_cast<const tensor_desc&>(*tensorDescSrc);
    // Types don't need to match, but they can't be VOID
    if((tdDst.type() == CUPHY_VOID) || tdSrc.type() == CUPHY_VOID)
    {
        NVLOGE_FMT(NVLOG_TAG_BASE_CUPHY, AERIAL_CUPHY_EVENT, "{} Cannot convert void data type", __FUNCTION__);
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    const tensor_layout_any& layoutDst = tdDst.layout();
    const tensor_layout_any& layoutSrc = tdSrc.layout();
    if(!layoutDst.has_same_size(layoutSrc))
    {
        NVLOGE_FMT(NVLOG_TAG_BASE_CUPHY, AERIAL_CUPHY_EVENT, "{} Input and output are different sizes", __FUNCTION__);
        return CUPHY_STATUS_SIZE_MISMATCH;
    }
    //------------------------------------------------------------------
    // Handle "memcpy" case (same type and strides)
    // We also exclude CUPHY_BIT from memcpy cases, since we may need
    // to mask off "extra" bits from the source tensor.
    if((tdDst.type() == tdSrc.type()) &&
       (tdDst.type() != CUPHY_BIT) &&
       layoutDst.has_same_strides(layoutSrc))
    {
        // Assuming availability of cudaMemcpyDefault (unified virtual
        // addressing), unifiedAddressing property in cudaDeviceProperties
        cudaError_t e = cudaMemcpyAsync(dstAddr,
                                        srcAddr,
                                        tdDst.get_size_in_bytes(),
                                        cudaMemcpyDefault,
                                        strm);
        if(cudaSuccess != e)
        {
            NVLOGE_FMT(NVLOG_TAG_BASE_CUPHY, AERIAL_CUPHY_EVENT, "{} CUPHY_STATUS_MEMCPY_ERROR: {}", __FUNCTION__, cudaGetErrorString(e));
            return CUPHY_STATUS_MEMCPY_ERROR;
        }
        return CUPHY_STATUS_SUCCESS;
    }
    //------------------------------------------------------------------
    // Handle more complex cases here (different types and/or different
    // layouts).
    // printf("tdDstType %s tdSrcType %s dstAddr %p srcAddr %p\n", cuphyGetDataTypeString(tdDst.type()), cuphyGetDataTypeString(tdSrc.type()), dstAddr, srcAddr);
    return convert_tensor_layout(tdDst, dstAddr, tdSrc, srcAddr, strm);
}
