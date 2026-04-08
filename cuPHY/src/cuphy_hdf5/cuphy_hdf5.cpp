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

#include "cuphy_hdf5.h"
#include "memfoot_global.h"
#include <limits>
#include <array>
#include <algorithm>

namespace
{

struct tensor_descriptor_info
{
    cuphyDataType_t                dataType;
    int                            numDims;
    std::array<int, CUPHY_DIM_MAX> dimensions; // only 1st numDims elements are valid
    std::array<int, CUPHY_DIM_MAX> strides;    // only 1st numDims elements are valid
};

////////////////////////////////////////////////////////////////////////
// generate_native_HDF5_complex_type()
// Note: caller should call H5Tclose() on the returned type
hid_t generate_native_HDF5_complex_type(hid_t elementType)
{
    size_t elementSizeBytes = H5Tget_size(elementType);
    if(elementSizeBytes <= 0)
    {
        return -1;
    }
    hid_t  cType = H5Tcreate(H5T_COMPOUND, 2 * elementSizeBytes);
    if(cType < 0)
    {
        return cType;
    }
    if((H5Tinsert(cType, "re", 0, elementType) < 0) ||
       (H5Tinsert(cType, "im", elementSizeBytes, elementType) < 0))
    {
        H5Tclose(cType);
        cType = -1;
    }
    return cType;
}

////////////////////////////////////////////////////////////////////////
// generate_native_HDF5_fp16_type()
// Note: caller should call H5Tclose() on the returned type
hid_t generate_native_HDF5_fp16_type()
{
    //------------------------------------------------------------------
    // Copy an existing floating point type as a starting point.
    hid_t cType = H5Tcopy(H5T_NATIVE_FLOAT);
    if(cType < 0)
    {
        return cType;
    }
    //------------------------------------------------------------------
    // https://en.wikipedia.org/wiki/Half-precision_floating-point_format
    // sign_pos = 15
    // exp_pos  = 10
    // exp_size = 5
    // mantissa_pos = 0
    // mantissa_size = 10
    // Order is important: we should not set the size before adjusting
    // the fields.
    if((H5Tset_fields(cType, 15, 10, 5, 0, 10) < 0) ||
       (H5Tset_precision(cType, 16)            < 0) ||
       (H5Tset_ebias(cType, 15)                < 0) ||
       (H5Tset_size(cType, 2)                  < 0))
    {
        H5Tclose(cType);
        cType = -1;
    }
    return cType;
}

// clang-format off
////////////////////////////////////////////////////////////////////////
// native_HDF5_type_from_cuphy_type()
// Note: caller should call H5Tclose() on the returned type
hid_t native_HDF5_type_from_cuphy_type(cuphyDataType_t t)
{
    switch(t)
    {
    case CUPHY_R_32F:    return H5Tcopy(H5T_NATIVE_FLOAT);
    case CUPHY_R_64F:    return H5Tcopy(H5T_NATIVE_DOUBLE);
    case CUPHY_R_8I:     return H5Tcopy(H5T_NATIVE_INT8);
    case CUPHY_R_8U:     return H5Tcopy(H5T_NATIVE_UINT8);
    case CUPHY_R_16I:    return H5Tcopy(H5T_NATIVE_INT16);
    case CUPHY_R_16U:    return H5Tcopy(H5T_NATIVE_UINT16);
    case CUPHY_R_32I:    return H5Tcopy(H5T_NATIVE_INT32);
    case CUPHY_R_32U:    return H5Tcopy(H5T_NATIVE_UINT32);
        
    case CUPHY_C_8I:     return generate_native_HDF5_complex_type(H5T_NATIVE_INT8);
    case CUPHY_C_8U:     return generate_native_HDF5_complex_type(H5T_NATIVE_UINT8);
    case CUPHY_C_16I:    return generate_native_HDF5_complex_type(H5T_NATIVE_INT16);
    case CUPHY_C_16U:    return generate_native_HDF5_complex_type(H5T_NATIVE_UINT16);
    case CUPHY_C_32I:    return generate_native_HDF5_complex_type(H5T_NATIVE_INT32);
    case CUPHY_C_32U:    return generate_native_HDF5_complex_type(H5T_NATIVE_UINT32);
    case CUPHY_C_32F:    return generate_native_HDF5_complex_type(H5T_NATIVE_FLOAT);
    case CUPHY_C_64F:    return generate_native_HDF5_complex_type(H5T_NATIVE_DOUBLE);

    case CUPHY_R_16F:    return generate_native_HDF5_fp16_type();
    case CUPHY_C_16F:
        {
            hid_t fp16Type        = generate_native_HDF5_fp16_type();
            hid_t fp16ComplexType = generate_native_HDF5_complex_type(fp16Type);
            H5Tclose(fp16Type);
            return fp16ComplexType;
        }
        
    case CUPHY_VOID:
    case CUPHY_BIT:
    default:
        // No valid native HDF5 representation, so we return an invalid hid_t
        return -1;
    }
}
// clang-format on

////////////////////////////////////////////////////////////////////////
// get_storage_type
// Returns a cuPHY type that can be used as the destination type, for
// file storage. This is useful for cases where we want to perform
// implicit conversion. Example: To store a fp16 value into a file, we
// will implicitly convert it to fp32 first.
// This type should be used as the destination type for the "convert
// tensor" operations.
cuphyDataType_t get_storage_type(cuphyDataType_t t)
{
    switch(t)
    {
    case CUPHY_R_32F:
    case CUPHY_R_64F:
    case CUPHY_R_8I:
    case CUPHY_R_8U:
    case CUPHY_R_16I:
    case CUPHY_R_16U:
    case CUPHY_R_32I:
    case CUPHY_R_32U:
    case CUPHY_C_8I:
    case CUPHY_C_8U:
    case CUPHY_C_16I:
    case CUPHY_C_16U:
    case CUPHY_C_32I:
    case CUPHY_C_32U:
    case CUPHY_C_32F:
    case CUPHY_C_64F:
    case CUPHY_R_16F:
    case CUPHY_C_16F:
        // No conversion necessary for these types
        return t;
    case CUPHY_BIT:
        // Store bits by default as unsigned 8-bit integers
        return CUPHY_R_8U;
    case CUPHY_VOID:
    default:
        // No currently supported HDF5 representation or implementation,
        // so we return CUPHY_VOID
        return CUPHY_VOID;
    }
}

////////////////////////////////////////////////////////////////////////
// cuphy_type_from_HDF5_datatype()
// Maps the HDF5 type to a cuPhyDataType_t. If no mapping is possible,
// the return type will by CUPHY_VOID.
cuphyDataType_t cuphy_type_from_HDF5_datatype(hid_t h5Datatype)
{
    cuphyDataType_t cuphyType = CUPHY_VOID;
    size_t          typeSize  = H5Tget_size(h5Datatype);
    H5T_class_t     H5class   = H5Tget_class(h5Datatype);
    switch(H5class)
    {
    case H5T_INTEGER:
        // clang-format off
        {
            H5T_sign_t sgn      = H5Tget_sign(h5Datatype);
            bool       isSigned = (H5T_SGN_2 == sgn);
            switch(typeSize)
            {
            case 1:
                cuphyType = isSigned ? CUPHY_R_8I  : CUPHY_R_8U;
                break;
            case 2:
                cuphyType = isSigned ? CUPHY_R_16I : CUPHY_R_16U;
                break;
            case 4:
                cuphyType = isSigned ? CUPHY_R_32I : CUPHY_R_32U;
            default:
                break;
            }
        }
        // clang-format on
        break;
    case H5T_FLOAT:
        if(sizeof(float) == typeSize)
        {
            cuphyType = CUPHY_R_32F;
        }
        else if(sizeof(double) == typeSize)
        {
            cuphyType = CUPHY_R_64F;
        }
        else if(sizeof(__half_raw) == typeSize)
        {
            cuphyType = CUPHY_R_16F;
        }
        break;
    case H5T_COMPOUND: // Complex data
        // Verify that the compound structure has two fields, with names
        // "re" and "im".
        {
            int numMembers = H5Tget_nmembers(h5Datatype);
            if((2 == numMembers) &&
               (H5Tget_member_index(h5Datatype, "re") == 0) &&
               (H5Tget_member_index(h5Datatype, "im") == 1))
            {
                H5T_class_t reClass = H5Tget_member_class(h5Datatype, 0);
                H5T_class_t imClass = H5Tget_member_class(h5Datatype, 1);
                // Types must be the same, and must be either H5T_INTEGER or H5T_FLOAT
                if((reClass == imClass) &&
                   ((H5T_INTEGER == reClass) || (H5T_FLOAT == reClass)))
                {
                    hid_t  reType = H5Tget_member_type(h5Datatype, 0);
                    hid_t  imType = H5Tget_member_type(h5Datatype, 1);
                    size_t reSize = H5Tget_size(reType);
                    size_t imSize = H5Tget_size(imType);
                    if(reSize == imSize)
                    {
                        if(H5T_FLOAT == reClass)
                        {
                            if(sizeof(float) == reSize)
                            {
                                cuphyType = CUPHY_C_32F;
                            }
                            else if(sizeof(double) == reSize)
                            {
                                cuphyType = CUPHY_C_64F;
                            }
                            else if(sizeof(__half_raw) == reSize)
                            {
                                cuphyType = CUPHY_C_16F;
                            }
                        }
                        else
                        {
                            H5T_sign_t reSign = H5Tget_sign(reType);
                            H5T_sign_t imSign = H5Tget_sign(imType);
                            if(reSign == imSign)
                            {
                                bool isSigned = (H5T_SGN_2 == reSign);
                                switch(reSize)
                                {
                                case 1:
                                    cuphyType = isSigned ? CUPHY_C_8I : CUPHY_C_8U;
                                    break;
                                case 2:
                                    cuphyType = isSigned ? CUPHY_C_16I : CUPHY_C_16U;
                                    break;
                                case 4:
                                    cuphyType = isSigned ? CUPHY_C_32I : CUPHY_C_32U;
                                default:
                                    break;
                                }
                            }
                        }
                    }
                    H5Tclose(imType);
                    H5Tclose(reType);
                } // if 2 members have the same class, which is INTEGER or FLOAT
            }     // if 2 members are named "re" and "im"
        }
        break;
    default:
        // Class is not one of INTEGER, FLOAT, or COMPOUND
        break;
    }

    return cuphyType;
}

////////////////////////////////////////////////////////////////////////
// cuphy_type_from_HDF5_dataset()
// Maps the HDF5 type to a cuPhyDataType_t. If no mapping is possible,
// the return type will by CUPHY_VOID.
cuphyDataType_t cuphy_type_from_HDF5_dataset(hid_t h5Dataset)
{
    hid_t h5Datatype = H5Dget_type(h5Dataset);
    if(h5Datatype < 0)
    {
        return CUPHY_VOID;
    }
    else
    {
        cuphyDataType_t cuphyType  = cuphy_type_from_HDF5_datatype(h5Datatype);
        H5Tclose(h5Datatype);
        return cuphyType;
    }
}


////////////////////////////////////////////////////////////////////////
// get_HDF5_dataset_info()
// Populates the given tensor_descriptor_info structure with information
// obtained from the HDF5 dataset found in a file.
// Note that the order of cuphy tensors dimensions is opposite that of
// HDF5, so we reverse the order.
// Returns one of the following values:
// CUPHYHDF5_STATUS_SUCCESS
// CUPHYHDF5_STATUS_DATATYPE_ERROR
//     The data type of the HDF5 dataset is not supported by cuPHY
// CUPHYHDF5_STATUS_DATASPACE_ERROR
//     HDF5 library error querying the dataspace structure
// CUPHYHDF5_STATUS_UNSUPPORTED_RANK
//     The rank of the dataset is larger than supported by cuPHY
// CUPHYHDF5_STATUS_DIMENSION_TOO_LARGE
//     The size of the dimension is larger than supported by cuPHY tensors
cuphyHDF5Status_t get_HDF5_dataset_info(tensor_descriptor_info& tdi,
                                        hid_t                   h5Dataset)
{
    tdi.dimensions.fill(0);
    tdi.strides.fill(0);
    //------------------------------------------------------------------
    // Get the cuphyDataType_t that corresponds to the HDF5 Datatype
    tdi.dataType = cuphy_type_from_HDF5_dataset(h5Dataset);
    if(CUPHY_VOID == tdi.dataType)
    {
        return CUPHYHDF5_STATUS_DATATYPE_ERROR;
    }
    //------------------------------------------------------------------
    // Get the HDF5 Dataspace to determine the bounds of the tensor
    hid_t h5Dataspace = H5Dget_space(h5Dataset);
    if(h5Dataspace < 0)
    {
        return CUPHYHDF5_STATUS_DATASPACE_ERROR;
    }
    cuphyHDF5Status_t status = CUPHYHDF5_STATUS_SUCCESS;
    tdi.numDims              = H5Sget_simple_extent_ndims(h5Dataspace);
    if((tdi.numDims <= 0) || (tdi.numDims > CUPHY_DIM_MAX))
    {
        status = CUPHYHDF5_STATUS_UNSUPPORTED_RANK;
    }
    else
    {
        std::array<hsize_t, CUPHY_DIM_MAX> dims;
        if(H5Sget_simple_extent_dims(h5Dataspace, dims.data(), nullptr) < 0)
        {
            status = CUPHYHDF5_STATUS_DATASPACE_ERROR;
        }
        else
        {
            // Check the size of the dimensions to make sure they
            // are not too large to be represented by an integer
            for(int i = 0; i < tdi.numDims; ++i)
            {
                if(dims[i] > std::numeric_limits<int>::max())
                {
                    status = CUPHYHDF5_STATUS_DIMENSION_TOO_LARGE;
                    break;
                }
                tdi.dimensions[i] = static_cast<int>(dims[i]);
            }
            // Reverse the order of the dimensions  to reflect the
            // different conventions between cuphy and HDF5
            std::reverse(tdi.dimensions.begin(), tdi.dimensions.begin() + tdi.numDims);

            // Calculate the strides, assuming tightly packed.
            tdi.strides[0] = 1;
            for(size_t i = 1; i < tdi.numDims; ++i)
            {
                tdi.strides[i] = tdi.strides[i - 1] * tdi.dimensions[i - 1];
            }
        }
    }
    H5Sclose(h5Dataspace);
    return status;
}

} // namespace


// clang-format off
////////////////////////////////////////////////////////////////////////
// cuphyHDF5GetErrorString()
const char* cuphyHDF5GetErrorString(cuphyHDF5Status_t status)
{
    switch(status)
    {
    case CUPHYHDF5_STATUS_SUCCESS:                return "The API call returned with no errors.";
    case CUPHYHDF5_STATUS_INVALID_ARGUMENT:       return "One or more of the arguments provided to the function was invalid.";
    case CUPHYHDF5_STATUS_INVALID_DATASET:        return "The HDF5 dataset argument provided was invalid.";
    case CUPHYHDF5_STATUS_DATATYPE_ERROR:         return "The HDF5 datatype is not supported by the cuPHY library.";
    case CUPHYHDF5_STATUS_DATASPACE_ERROR:        return "The HDF5 library returned an error creating or querying the dataspace.";
    case CUPHYHDF5_STATUS_UNSUPPORTED_RANK:       return "The HDF5 dataspace rank is not supported by cuPHY.";
    case CUPHYHDF5_STATUS_DIMENSION_TOO_LARGE:    return "One or more HDF5 dataspace dimensions are larger than cuPHY supports.";
    case CUPHYHDF5_STATUS_INVALID_TENSOR_DESC:    return "An invalid tensor descriptor was provided.";
    case CUPHYHDF5_STATUS_INADEQUATE_BUFFER_SIZE: return "The provided buffer size was inadequate.";
    case CUPHYHDF5_STATUS_TENSOR_MISMATCH:        return "Tensor descriptor arguments do not match in rank and/or dimension(s).";
    case CUPHYHDF5_STATUS_UNKNOWN_ERROR:          return "Unknown or unexpected internal error.";
    case CUPHYHDF5_STATUS_ALLOC_FAILED:           return "Memory allocation failed.";
    case CUPHYHDF5_STATUS_TENSOR_DESC_FAILURE:    return "Creating or setting the cuPHY tensor descriptor failed.";
    case CUPHYHDF5_STATUS_READ_ERROR:             return "An HDF5 read error occurred.";
    case CUPHYHDF5_STATUS_CONVERT_ERROR:          return "A conversion error occurred, or an unsupported conversion was requested.";
    case CUPHYHDF5_STATUS_WRITE_ERROR:            return "An HDF5 write error occurred.";
    case CUPHYHDF5_STATUS_DATASET_ERROR:          return "An HDF5 dataset creation/query error occurred.";
    case CUPHYHDF5_STATUS_INVALID_NAME:           return "No such scalar or structure field with the given name exists.";
    case CUPHYHDF5_STATUS_INCORRECT_OBJ_TYPE:     return "The HDF5 object provided is not of the correct/expected type.";
    case CUPHYHDF5_STATUS_OBJ_CREATE_FAILURE:     return "HDF5 object creation failure.";
    case CUPHYHDF5_STATUS_VALUE_OUT_OF_RANGE:     return "Data conversion could not occur because an input value was out of range.";
    default:                                      return "Unknown status.";
    }
}
// clang-format on

// clang-format off
////////////////////////////////////////////////////////////////////////
// cuphyHDF5GetErrorName()
const char* cuphyHDF5GetErrorName(cuphyHDF5Status_t status)
{
    switch(status)
    {
    case CUPHYHDF5_STATUS_SUCCESS:                return "CUPHYHDF5_STATUS_SUCCESS";
    case CUPHYHDF5_STATUS_INVALID_ARGUMENT:       return "CUPHYHDF5_STATUS_INVALID_ARGUMENT";
    case CUPHYHDF5_STATUS_INVALID_DATASET:        return "CUPHYHDF5_STATUS_INVALID_DATASET";
    case CUPHYHDF5_STATUS_DATATYPE_ERROR:         return "CUPHYHDF5_STATUS_DATATYPE_ERROR";
    case CUPHYHDF5_STATUS_DATASPACE_ERROR:        return "CUPHYHDF5_STATUS_DATASPACE_ERROR";
    case CUPHYHDF5_STATUS_UNSUPPORTED_RANK:       return "CUPHYHDF5_STATUS_UNSUPPORTED_RANK";
    case CUPHYHDF5_STATUS_DIMENSION_TOO_LARGE:    return "CUPHYHDF5_STATUS_DIMENSION_TOO_LARGE";
    case CUPHYHDF5_STATUS_INVALID_TENSOR_DESC:    return "CUPHYHDF5_STATUS_INVALID_TENSOR_DESC";
    case CUPHYHDF5_STATUS_INADEQUATE_BUFFER_SIZE: return "CUPHYHDF5_STATUS_INADEQUATE_BUFFER_SIZE";
    case CUPHYHDF5_STATUS_TENSOR_MISMATCH:        return "CUPHYHDF5_STATUS_TENSOR_MISMATCH";
    case CUPHYHDF5_STATUS_UNKNOWN_ERROR:          return "CUPHYHDF5_STATUS_UNKNOWN_ERROR";
    case CUPHYHDF5_STATUS_ALLOC_FAILED:           return "CUPHYHDF5_STATUS_ALLOC_FAILED";
    case CUPHYHDF5_STATUS_TENSOR_DESC_FAILURE:    return "CUPHYHDF5_STATUS_TENSOR_DESC_FAILURE";
    case CUPHYHDF5_STATUS_READ_ERROR:             return "CUPHYHDF5_STATUS_READ_ERROR";
    case CUPHYHDF5_STATUS_CONVERT_ERROR:          return "CUPHYHDF5_STATUS_CONVERT_ERROR";
    case CUPHYHDF5_STATUS_WRITE_ERROR:            return "CUPHYHDF5_STATUS_WRITE_ERROR";
    case CUPHYHDF5_STATUS_DATASET_ERROR:          return "CUPHYHDF5_STATUS_DATASET_ERROR";
    case CUPHYHDF5_STATUS_INVALID_NAME:           return "CUPHYHDF5_STATUS_INVALID_NAME";
    case CUPHYHDF5_STATUS_INCORRECT_OBJ_TYPE:     return "CUPHYHDF5_STATUS_INCORRECT_OBJ_TYPE";
    case CUPHYHDF5_STATUS_OBJ_CREATE_FAILURE:     return "CUPHYHDF5_STATUS_OBJ_CREATE_FAILURE";
    case CUPHYHDF5_STATUS_VALUE_OUT_OF_RANGE:     return "CUPHYHDF5_STATUS_VALUE_OUT_OF_RANGE";
    default:                                      return "CUPHYHDF5_UNKNOWN_STATUS";
    }
}
// clang-format on

////////////////////////////////////////////////////////////////////////
// cuphyHDF5GetDatasetInfo()
cuphyHDF5Status_t cuphyHDF5GetDatasetInfo(hid_t            h5Dataset,
                                          int              dimBufferSize,
                                          cuphyDataType_t* dataType,
                                          int*             numDims,
                                          int              outputDimensions[])
{
    //------------------------------------------------------------------
    // Validate inputs
    if(h5Dataset < 0) return CUPHYHDF5_STATUS_INVALID_DATASET;
    if((dimBufferSize > 0) && (nullptr == outputDimensions))
    {
        return CUPHYHDF5_STATUS_INVALID_ARGUMENT;
    }
    //------------------------------------------------------------------
    // Retrieve tensor info from the data set
    tensor_descriptor_info tdi = {}; // Zero-initialize
    cuphyHDF5Status_t      s   = get_HDF5_dataset_info(tdi, h5Dataset);
    //------------------------------------------------------------------
    // Populate return info
    if(dataType)
    {
        *dataType = tdi.dataType;
    }
    if(numDims)
    {
        *numDims = tdi.numDims;
    }
    if(outputDimensions)
    {
        for(int i = 0; i < dimBufferSize; ++i)
        {
            outputDimensions[i] = tdi.dimensions[i];
        }
        // If the caller-provided buffer is too small, provide an
        // error (but only if there was no other error).
        if((s == CUPHYHDF5_STATUS_SUCCESS) &&
           (dimBufferSize < tdi.numDims))
        {
            s = CUPHYHDF5_STATUS_INADEQUATE_BUFFER_SIZE;
        }
    }
    //------------------------------------------------------------------
    return s;
}

////////////////////////////////////////////////////////////////////////
// cuphyHDF5ReadDataset()
cuphyHDF5Status_t cuphyHDF5ReadDataset(const cuphyTensorDescriptor_t tensorDesc,
                                       void*                         addr,
                                       hid_t                         h5Dataset,
                                       cudaStream_t                  strm)
{
    //------------------------------------------------------------------
    // Validate arguments
    if((nullptr == tensorDesc) ||
       (nullptr == addr) ||
       (h5Dataset < 0))
    {
        return CUPHYHDF5_STATUS_INVALID_ARGUMENT;
    }
    // clang-format off
    // Retrieve properties of the caller-provided destination tensor
    tensor_descriptor_info tdi = {}; // Zero-initialize all members
    cuphyStatus_t          s   = cuphyGetTensorDescriptor(tensorDesc,                              // tensorDesc
                                                          static_cast<int>(tdi.dimensions.size()), // numDimsRequested
                                                          &tdi.dataType,                           // dataType
                                                          &tdi.numDims,                            // numDims
                                                          tdi.dimensions.data(),                   // dimensions[]
                                                          tdi.strides.data());                     // strides[]
    // clang-format on
    if(CUPHY_STATUS_SUCCESS != s)
    {
        return CUPHYHDF5_STATUS_INVALID_TENSOR_DESC;
    }
    //------------------------------------------------------------------
    // Get tensor info from the HDF5 dataset (located in the file)
    tensor_descriptor_info tdiHDF5File = {}; // Zero-initialize
    cuphyHDF5Status_t      sHDF5       = get_HDF5_dataset_info(tdiHDF5File,
                                                    h5Dataset);
    if(CUPHYHDF5_STATUS_SUCCESS != sHDF5)
    {
        return sHDF5;
    }
    //------------------------------------------------------------------
    // Compare the tensor descriptor dimensions to the HDF5 file
    // dimensions. (The strides don't need to match - the caller may
    // adjust the strides based on application requirements.) The number
    // of dimensions and the individual dimensions must match.
    if((tdi.numDims != tdiHDF5File.numDims) ||
       !std::equal(tdi.dimensions.begin(),
                   tdi.dimensions.begin() + tdi.numDims,
                   tdiHDF5File.dimensions.begin()))
    {
        return CUPHYHDF5_STATUS_TENSOR_MISMATCH;
    }
    //------------------------------------------------------------------
    // Create a tensor descriptor to represent the in-memory data
    // retrieved from the HDF5 file.
    cuphyTensorDescriptor_t memTensorDesc;
    if(CUPHY_STATUS_SUCCESS != cuphyCreateTensorDescriptor(&memTensorDesc))
    {
        return CUPHYHDF5_STATUS_TENSOR_DESC_FAILURE;
    }
    if(CUPHY_STATUS_SUCCESS != cuphySetTensorDescriptor(memTensorDesc,                 // tensor desc
                                                        tdiHDF5File.dataType,          // data type
                                                        tdiHDF5File.numDims,           // rank
                                                        tdiHDF5File.dimensions.data(), // dimensions
                                                        tdiHDF5File.strides.data(),    // strides
                                                        CUPHY_TENSOR_ALIGN_TIGHT))     // flags
    {
        cuphyDestroyTensorDescriptor(memTensorDesc);
        return CUPHYHDF5_STATUS_TENSOR_DESC_FAILURE;
    }
    //------------------------------------------------------------------
    // Allocate a pinned host buffer for the destination of the HDF5
    // read operation.
    size_t hdf5Size = 0;
    if(CUPHY_STATUS_SUCCESS != cuphyGetTensorSizeInBytes(memTensorDesc, &hdf5Size))
    {
        cuphyDestroyTensorDescriptor(memTensorDesc);
        return CUPHYHDF5_STATUS_TENSOR_DESC_FAILURE;
    }
    //printf("size in bytes: %lu\n", hdf5Size);
    void* hostBuffer = nullptr;
    if(cudaSuccess != cudaHostAlloc(&hostBuffer, hdf5Size, cudaHostAllocMapped | cudaHostAllocWriteCombined))
    {
        cuphyDestroyTensorDescriptor(memTensorDesc);
        return CUPHYHDF5_STATUS_ALLOC_FAILED;
    }
    //------------------------------------------------------------------
    // Determine the in-memory HDF5 datatype for reading, based on the
    // datatype found in the file.
    hid_t hdf5MemType = native_HDF5_type_from_cuphy_type(tdiHDF5File.dataType);
    if(hdf5MemType < 0)
    {
        cudaFreeHost(hostBuffer);
        cuphyDestroyTensorDescriptor(memTensorDesc);
        return CUPHYHDF5_STATUS_DATATYPE_ERROR;
    }
    //------------------------------------------------------------------
    // Synchronize the CUDA stream to make sure that the input tensor
    // can be read
    cudaStreamSynchronize(strm);
    //------------------------------------------------------------------
    // Invoke the HDF5 library read call to read data into the pinned
    // host buffer
    herr_t h5Status = H5Dread(h5Dataset, hdf5MemType, H5S_ALL, H5S_ALL, H5P_DEFAULT, hostBuffer);
    if(h5Status < 0)
    {
        H5Tclose(hdf5MemType);
        cudaFreeHost(hostBuffer);
        cuphyDestroyTensorDescriptor(memTensorDesc);
        return CUPHYHDF5_STATUS_READ_ERROR;
    }
    //------------------------------------------------------------------
    // Use the cuphyConvertTensor() function to perform a "copy," where
    // in this case the source is the host buffer with the H5Dread()
    // results.
    sHDF5 = CUPHYHDF5_STATUS_SUCCESS;
    if(CUPHY_STATUS_SUCCESS != cuphyConvertTensor(tensorDesc,    // dst tensor
                                                  addr,          // dst address
                                                  memTensorDesc, // src tensor
                                                  hostBuffer,    // src address
                                                  strm))         // CUDA stream
    {
        sHDF5 = CUPHYHDF5_STATUS_CONVERT_ERROR;
    }
    //------------------------------------------------------------------
    H5Tclose(hdf5MemType);
    cudaFreeHost(hostBuffer);
    cuphyDestroyTensorDescriptor(memTensorDesc);
    return sHDF5;
}

////////////////////////////////////////////////////////////////////////
// cuphyHDF5WriteDatasetFromCPU()
cuphyHDF5Status_t CUPHYWINAPI cuphyHDF5WriteDatasetFromCPU(hid_t h5LocationID, const char*  name, const cuphyDataType_t type, const int32_t size, void* pData)
{
    if((nullptr == pData) ||
       (0 == size) ||
       (nullptr == name) ||
       (0 == strlen(name)) ||
       (h5LocationID < 0))
    {
        return CUPHYHDF5_STATUS_INVALID_ARGUMENT;
    }
    cuphyDataType_t storageType = get_storage_type(type);
    hid_t hdf5MemType = native_HDF5_type_from_cuphy_type(storageType);
    if(hdf5MemType < 0)
    {
        return CUPHYHDF5_STATUS_DATATYPE_ERROR;
    }

    cuphyHDF5Status_t sHDF5 = CUPHYHDF5_STATUS_SUCCESS;
    std::array<hsize_t, CUPHY_DIM_MAX> h5Dims;
    h5Dims[0] = size;
    hid_t h5Dataspace = H5Screate_simple(1, h5Dims.data(), nullptr);
    if(h5Dataspace < 0)
    {
        sHDF5 = CUPHYHDF5_STATUS_DATASPACE_ERROR;
    }
    else
    {
        hid_t h5Dataset = H5Dcreate2(h5LocationID, // loc_id
                                     name,         // name
                                     hdf5MemType,  // datatype_id
                                     h5Dataspace,  // dataspace_id
                                     H5P_DEFAULT,  // link creation prop list
                                     H5P_DEFAULT,  // dataset creation prop list
                                     H5P_DEFAULT); // dataset access prop list
        if(h5Dataset < 0)
        {
            sHDF5 = CUPHYHDF5_STATUS_DATASET_ERROR;
        }
        else
        {
            herr_t h5Status = H5Dwrite(h5Dataset,
                                       hdf5MemType,
                                       H5S_ALL,
                                       H5S_ALL,
                                       H5P_DEFAULT,
                                       pData); 
            if(h5Status < 0)
            {
                sHDF5 = CUPHYHDF5_STATUS_WRITE_ERROR;
            }
            H5Dclose(h5Dataset);
        }
        H5Sclose(h5Dataspace);
    }
    H5Tclose(hdf5MemType);
    return sHDF5;
                                                    
}

////////////////////////////////////////////////////////////////////////
// cuphyHDF5WriteDataset()
cuphyHDF5Status_t CUPHYWINAPI cuphyHDF5WriteDataset(hid_t                         h5LocationID,
                                                    const char*                   name,
                                                    const cuphyTensorDescriptor_t srcTensorDesc,
                                                    const void*                   srcAddr,
                                                    cudaStream_t                  strm)
{
    //------------------------------------------------------------------
    // Validate arguments
    if((nullptr == srcTensorDesc) ||
       (nullptr == srcAddr) ||
       (nullptr == name) ||
       (0 == strlen(name)) ||
       (h5LocationID < 0))
    {
        return CUPHYHDF5_STATUS_INVALID_ARGUMENT;
    }
    // clang-format off
    // Retrieve properties of the caller-provided source tensor
    tensor_descriptor_info tdiSrc = {}; // Zero-initialize all members
    cuphyStatus_t          s      = cuphyGetTensorDescriptor(srcTensorDesc,                              // tensorDesc
                                                             static_cast<int>(tdiSrc.dimensions.size()), // numDimsRequested
                                                             &tdiSrc.dataType,                           // dataType
                                                             &tdiSrc.numDims,                            // numDims
                                                             tdiSrc.dimensions.data(),                   // dimensions[]
                                                             tdiSrc.strides.data());                     // strides[]
    // clang-format on
    if(CUPHY_STATUS_SUCCESS != s)
    {
        return CUPHYHDF5_STATUS_INVALID_TENSOR_DESC;
    }
    //------------------------------------------------------------------
    // Check for known implicit conversions
    cuphyDataType_t storageType = get_storage_type(tdiSrc.dataType);
    if(CUPHY_VOID == storageType)
    {
        return CUPHYHDF5_STATUS_DATATYPE_ERROR;
    }
    //------------------------------------------------------------------
    // Create a tensor descriptor to represent the in-memory data
    // that will be provided to the HDF5 library for storage.
    cuphyTensorDescriptor_t memTensorDesc;
    if(CUPHY_STATUS_SUCCESS != cuphyCreateTensorDescriptor(&memTensorDesc))
    {
        return CUPHYHDF5_STATUS_TENSOR_DESC_FAILURE;
    }
    if(CUPHY_STATUS_SUCCESS != cuphySetTensorDescriptor(memTensorDesc,             // tensor desc
                                                        storageType,               // data type
                                                        tdiSrc.numDims,            // rank
                                                        tdiSrc.dimensions.data(),  // dimensions
                                                        nullptr,                   // strides
                                                        CUPHY_TENSOR_ALIGN_TIGHT)) // flags
    {
        cuphyDestroyTensorDescriptor(memTensorDesc);
        return CUPHYHDF5_STATUS_TENSOR_DESC_FAILURE;
    }
    //------------------------------------------------------------------
    // Allocate a pinned host buffer for the destination of the read
    // from the source tensor
    size_t hdf5Size = 0;
    if(CUPHY_STATUS_SUCCESS != cuphyGetTensorSizeInBytes(memTensorDesc, &hdf5Size))
    {
        cuphyDestroyTensorDescriptor(memTensorDesc);
        return CUPHYHDF5_STATUS_TENSOR_DESC_FAILURE;
    }
    //printf("size in bytes: %lu\n", hdf5Size);
    void* hostBuffer = nullptr;
    if(cudaSuccess != cudaHostAlloc(&hostBuffer, hdf5Size, cudaHostAllocMapped | cudaHostAllocWriteCombined))
    {
        cuphyDestroyTensorDescriptor(memTensorDesc);
        return CUPHYHDF5_STATUS_ALLOC_FAILED;
    }
    //------------------------------------------------------------------
    // Determine the HDF5 datatype that corresponds to the source tensor
    hid_t hdf5MemType = native_HDF5_type_from_cuphy_type(storageType);
    if(hdf5MemType < 0)
    {
        cudaFreeHost(hostBuffer);
        cuphyDestroyTensorDescriptor(memTensorDesc);
        return CUPHYHDF5_STATUS_DATATYPE_ERROR;
    }
    //------------------------------------------------------------------
    cuphyHDF5Status_t sHDF5 = CUPHYHDF5_STATUS_SUCCESS;
    if(CUPHY_STATUS_SUCCESS != cuphyConvertTensor(memTensorDesc, // dst tensor
                                                  hostBuffer,    // dst address
                                                  srcTensorDesc, // src tensor
                                                  srcAddr,       // src address
                                                  strm))         // CUDA stream
    {
        sHDF5 = CUPHYHDF5_STATUS_CONVERT_ERROR;
    }
    else
    {
        std::array<hsize_t, CUPHY_DIM_MAX> h5Dims;
        // HDF5 order is slowest-changing first, so reverse indices
        for(size_t i = 0; i < tdiSrc.numDims; ++i)
        {
            h5Dims[tdiSrc.numDims - i - 1] = tdiSrc.dimensions[i];
        }
        hid_t h5Dataspace = H5Screate_simple(tdiSrc.numDims, h5Dims.data(), nullptr);
        if(h5Dataspace < 0)
        {
            sHDF5 = CUPHYHDF5_STATUS_DATASPACE_ERROR;
        }
        else
        {
            hid_t h5Dataset = H5Dcreate2(h5LocationID, // loc_id
                                         name,         // name
                                         hdf5MemType,  // datatype_id
                                         h5Dataspace,  // dataspace_id
                                         H5P_DEFAULT,  // link creation prop list
                                         H5P_DEFAULT,  // dataset creation prop list
                                         H5P_DEFAULT); // dataset access prop list
            if(h5Dataset < 0)
            {
                sHDF5 = CUPHYHDF5_STATUS_DATASET_ERROR;
            }
            else
            {
                // Synchronize on the stream used for conversion to ensure
                // that the result can be read by the host.
                cudaStreamSynchronize(strm);
                
                herr_t h5Status = H5Dwrite(h5Dataset,
                                           hdf5MemType,
                                           H5S_ALL,
                                           H5S_ALL,
                                           H5P_DEFAULT,
                                           hostBuffer);
                if(h5Status < 0)
                {
                    sHDF5 = CUPHYHDF5_STATUS_WRITE_ERROR;
                }
                H5Dclose(h5Dataset);
            }
            H5Sclose(h5Dataspace);
        }
    }
    //------------------------------------------------------------------
    H5Tclose(hdf5MemType);
    cudaFreeHost(hostBuffer);
    cuphyDestroyTensorDescriptor(memTensorDesc);
    return sHDF5;
}

////////////////////////////////////////////////////////////////////////
// cuphyHDF5Struct
// Empty struct for forward-declared type from cuphy_hdf5.h header
struct cuphyHDF5Struct
{
};

////////////////////////////////////////////////////////////////////////
// cuphy_HDF5_struct_element
// Internal class to represent an element of an HDF5 dataset that will
// be accessed as a "struct." This amounts to a dataspace with a single
// element corresponding to a dataset with the HDF5 compound type.
class cuphy_HDF5_struct_element : public cuphyHDF5Struct
{
public:
    ~cuphy_HDF5_struct_element()
    {
        if(H5Iis_valid(h5_dataset_) > 0)
        {
            // Seems like we can either close or dec ref. We will
            // use dec ref to be symmetric with the inc ref upon
            // construction.
            //H5Dclose(h5_dataset_);
            H5Idec_ref(h5_dataset_);
        }
        if(H5Iis_valid(h5_dataspace_) > 0)
        {
            H5Sclose(h5_dataspace_);
        }
    }
    cuphy_HDF5_struct_element(const cuphy_HDF5_struct_element&)            = delete;
    cuphy_HDF5_struct_element& operator=(const cuphy_HDF5_struct_element&) = delete;
    //------------------------------------------------------------------
    // create()
    static cuphyHDF5Status_t create(hid_t                       dset,
                                    size_t                      numDim,
                                    const hsize_t*              coord,
                                    cuphy_HDF5_struct_element** p)
    {
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Initialize output value, assuming failure
        *p = nullptr;
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Check for a valid object identifier
        // (See "proper" way to test for truth in H5public.h)
        if(H5Iis_valid(dset) <= 0)
        {
            return CUPHYHDF5_STATUS_INVALID_DATASET;
        }
        // Make sure the input object is a dataset
        if(H5I_DATASET != H5Iget_type(dset))
        {
            return CUPHYHDF5_STATUS_INCORRECT_OBJ_TYPE;
        }
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Check for the dataset type
        hid_t dtype = H5Dget_type(dset);
        if(dtype >= 0)
        {
            //-  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
            // Check the datatype class - fail if it isn't a compound type
            H5T_class_t dclass = H5Tget_class(dtype);
            H5Tclose(dtype); // No longer used...
            dtype = -1;
            if(dclass != H5T_COMPOUND)
            {
                return CUPHYHDF5_STATUS_INVALID_DATASET;
            }
            //-  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
            // Get the dataspace to determine the dimensions
            hid_t dspace = H5Dget_space(dset);
            if(dspace < 0)
            {
                return CUPHYHDF5_STATUS_DATASPACE_ERROR;
            }
            //-  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
            // Check the rank and the number of elements
            int rank = H5Sget_simple_extent_ndims(dspace);
            if((0 == numDim) && (nullptr == coord))
            {
                // Only valid if dataspace has rank 1
                if(rank != 1)
                {
                    H5Sclose(dspace);
                    return CUPHYHDF5_STATUS_DIMENSION_TOO_LARGE;
                }
                // Getting a struct with 0 == numDim and nullptr == coord is
                // only valid if the dataspace has 1 element.
                hsize_t dim0;
                if(H5Sget_simple_extent_dims(dspace, &dim0, nullptr) < 0)
                {
                    H5Sclose(dspace);
                    return CUPHYHDF5_STATUS_DATASPACE_ERROR;
                }
                if(dim0 != 1)
                {
                    H5Sclose(dspace);
                    return CUPHYHDF5_STATUS_DIMENSION_TOO_LARGE;
                }
            }
            else
            {
                // Validate coordinates
                if((rank > CUPHY_DIM_MAX) || (numDim != rank))
                {
                    H5Sclose(dspace);
                    return CUPHYHDF5_STATUS_DIMENSION_TOO_LARGE;
                }
                // Get the dataspace dimensions
                hsize_t dims[CUPHY_DIM_MAX] = {};
                if(H5Sget_simple_extent_dims(dspace, dims, nullptr) < 0)
                {
                    H5Sclose(dspace);
                    return CUPHYHDF5_STATUS_DATASPACE_ERROR;
                }
                // Compare provided indices with the dataset dimensions
                for(int i = 0; i < numDim; ++i)
                {
                    if(coord[i] >= dims[i])
                    {
                        H5Sclose(dspace);
                        return CUPHYHDF5_STATUS_VALUE_OUT_OF_RANGE;
                    }
                }
                // Select the specific element requested in the stored
                // dataspace
                if(H5Sselect_elements(dspace, H5S_SELECT_SET, 1, coord) < 0)
                {
                    H5Sclose(dspace);
                    return CUPHYHDF5_STATUS_DATASPACE_ERROR;
                }
            }
            // Create the struct element instance with the dataset and
            // dataspace
            *p = new (std::nothrow) cuphy_HDF5_struct_element(dset, dspace);
            if(!*p)
            {
                H5Sclose(dspace);
                return CUPHYHDF5_STATUS_ALLOC_FAILED;
            }
            return CUPHYHDF5_STATUS_SUCCESS;
        }
        else
        {
            // H5Dget_type() failure
            return CUPHYHDF5_STATUS_INVALID_DATASET;
        }
    }
    //------------------------------------------------------------------
    // get_field()
    cuphyHDF5Status_t get_field(cuphyVariant_t& res,
                                const char*     name,
                                cuphyDataType_t valueAs)
    {
        res.type = CUPHY_VOID;
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Retrieve the datatype for the dataset
        hid_t dsetType = H5Dget_type(h5_dataset_);
        if(dsetType < 0)
        {
            return CUPHYHDF5_STATUS_INVALID_DATASET;
        }
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Locate the member index from the name
        int idx = H5Tget_member_index(dsetType, name);
        if(idx < 0)
        {
            printf("Invalid member name %s\n", name);
            H5Tclose(dsetType);
            return CUPHYHDF5_STATUS_INVALID_NAME;
        }
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Get the HDF5 stored type of the field
        hid_t srcFieldType = H5Tget_member_type(dsetType, static_cast<unsigned>(idx));
        H5Tclose(dsetType);
        if(srcFieldType < 0)
        {
            return CUPHYHDF5_STATUS_INVALID_DATASET;
        }
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Get the corresponding cuPHY type
        cuphyDataType_t srcType = cuphy_type_from_HDF5_datatype(srcFieldType);
        if(CUPHY_VOID == srcType)
        {
            H5Tclose(srcFieldType);
            return CUPHYHDF5_STATUS_DATATYPE_ERROR;
        }
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Get the corresponding native type for the field
        hid_t nativeFieldType = H5Tget_native_type(srcFieldType, H5T_DIR_ASCEND);
        H5Tclose(srcFieldType);
        if(nativeFieldType < 0)
        {
            return CUPHYHDF5_STATUS_DATATYPE_ERROR;
        }
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Create an in-memory compound type, with a single field having
        // a name that matches the callers requested field.
        hid_t memCompoundType = H5Tcreate(H5T_COMPOUND, H5Tget_size(nativeFieldType));
        if(memCompoundType < 0)
        {
            H5Tclose(nativeFieldType);
            return CUPHYHDF5_STATUS_OBJ_CREATE_FAILURE;
        }
        if(H5Tinsert(memCompoundType, name, 0, nativeFieldType) < 0)
        {
            H5Tclose(nativeFieldType);
            H5Tclose(memCompoundType);
            return CUPHYHDF5_STATUS_OBJ_CREATE_FAILURE;
        }
        H5Tclose(nativeFieldType);
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Perform an HDF5 read into the variant storage area using
        // the in-memory compound type.
        const hsize_t one      = 1;
        hid_t         memSpace = H5Screate_simple(1, &one, nullptr);
        if(memSpace < 0)
        {
            H5Tclose(memCompoundType);
            return CUPHYHDF5_STATUS_OBJ_CREATE_FAILURE;
        }
        herr_t readStatus = H5Dread(h5_dataset_,     // dataset
                                    memCompoundType, // memory type
                                    memSpace,        // memory dataspace
                                    h5_dataspace_,   // file dataspace
                                    H5P_DEFAULT,     // xfer prop list
                                    &(res.value.r8i));
        H5Sclose(memSpace);
        H5Tclose(memCompoundType);
        if(readStatus < 0)
        {
            return CUPHYHDF5_STATUS_READ_ERROR;
        }
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Store the original type in the user-provided variant
        res.type = srcType;
        if((CUPHY_VOID != valueAs) && (srcType != valueAs))
        {
            cuphyStatus_t s = cuphyConvertVariant(&res, valueAs);

            return (CUPHY_STATUS_SUCCESS == s)    ?
                   CUPHYHDF5_STATUS_SUCCESS       :
                   CUPHYHDF5_STATUS_CONVERT_ERROR;
        }
        else
        {
            return CUPHYHDF5_STATUS_SUCCESS;
        }
    }
private:
    cuphy_HDF5_struct_element(hid_t dset, hid_t dspace) :
        h5_dataset_(dset),
        h5_dataspace_(dspace)
    {
        // Increment the reference count of the dataset
        H5Iinc_ref(h5_dataset_);
    }
    // Data
    hid_t h5_dataset_;
    hid_t h5_dataspace_;
};

////////////////////////////////////////////////////////////////////////
// cuphyHDF5GetStruct()
cuphyHDF5Status_t cuphyHDF5GetStruct(hid_t              h5Dataset,
                                     size_t             numDim,
                                     const hsize_t*     coord,
                                     cuphyHDF5Struct_t* s)
{
    //------------------------------------------------------------------
    // Validate arguments
    if (nullptr == s)
    {
        return CUPHYHDF5_STATUS_INVALID_ARGUMENT;
    }
    if((numDim > 0) && (nullptr == coord))
    {
        return CUPHYHDF5_STATUS_DIMENSION_TOO_LARGE;
    }
    *s = nullptr;
    // (Further validation provided by cuphy_HDF5_struct_element::create())
    cuphy_HDF5_struct_element* ssd    = nullptr;
    cuphyHDF5Status_t          status = cuphy_HDF5_struct_element::create(h5Dataset,
                                                                          numDim,
                                                                          coord,
                                                                          &ssd);
    if(CUPHYHDF5_STATUS_SUCCESS == status)
    {
        *s = static_cast<cuphyHDF5Struct_t>(ssd);
    }
    return status;
}

////////////////////////////////////////////////////////////////////////
// cuphyHDF5GetStructScalar()
cuphyHDF5Status_t cuphyHDF5GetStructScalar(cuphyVariant_t*         res,
                                           const cuphyHDF5Struct_t s,
                                           const char*             name,
                                           cuphyDataType_t         valueAs)
{

    //------------------------------------------------------------------
    // Validate arguments
    if(!s    ||
       !name ||
       !res)
    {
        return CUPHYHDF5_STATUS_INVALID_ARGUMENT;
    }
    //------------------------------------------------------------------
    // Cast the user handle to the internal datatype and retrieve the
    // field value
    cuphy_HDF5_struct_element& sds = static_cast<cuphy_HDF5_struct_element&>(*s);
    return sds.get_field(*res, name, valueAs);
}

////////////////////////////////////////////////////////////////////////
// cuphyHDF5ReleaseStruct()
cuphyHDF5Status_t cuphyHDF5ReleaseStruct(cuphyHDF5Struct_t s)
{
    //------------------------------------------------------------------
    // Validate arguments
    if(nullptr == s)
    {
        return CUPHYHDF5_STATUS_INVALID_ARGUMENT;
    }
    //------------------------------------------------------------------
    // Free the structure previously allocated by cuphyHDF5GetStruct()
    cuphy_HDF5_struct_element* sds = static_cast<cuphy_HDF5_struct_element*>(s);
    delete sds;
    
    return CUPHYHDF5_STATUS_SUCCESS;
}
