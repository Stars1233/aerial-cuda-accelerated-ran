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

#include "chanModelsCommon.h"

// copied function generate_native_HDF5_fp16_type() from aerial_sdk/cuPHY/src/cuphy_hdf5/cuphy_hdf5.cpp, which creates __half to be used in saving to hdf5
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

/**
 * @brief helper function to write a dataset to hdf5 file
 * 
 * @param fileId hdf5 file id
 * @param datasetName dataset name
 * @param dataType h5 compound data type, depending on __half2 or cuComplex
 * @param dataspaceId data space id
 * @param dataGpu pointer to GPU data, data can be 1D array
 * @param dims dimensions of data
 * @param rank rank of dimension
 */
template <typename T>
void writeHdf5DatasetFromGpu(hid_t & fileId, const char * datasetName, hid_t & dataType, void * dataGpu, hsize_t * dims, uint8_t & rank)
{
    hid_t dataspaceId = H5Screate_simple(rank, dims, nullptr);
    // copy temporary data from GPU to CPU
    size_t size = 1;
    for(int i = 0; i < rank; i++)
    {
        size *= dims[i];
    }
    T * dataCpu = new T[size];
    cudaMemcpy(dataCpu, dataGpu, size * sizeof(T), cudaMemcpyDeviceToHost);
    // write cpu data to hdf5 file
    hid_t datasetId = H5Dcreate2(fileId, datasetName, dataType, dataspaceId, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    herr_t status = H5Dwrite(datasetId, dataType, H5S_ALL, H5S_ALL, H5P_DEFAULT, dataCpu);

    if (status < 0) 
    {
        fprintf(stderr, "Failed to write %s to HDF5 dataset!\n", datasetName);
        H5Dclose(datasetId);
        H5Sclose(dataspaceId);
        H5Fclose(fileId);
        exit(1);
    }
    H5Dclose(datasetId);
    H5Sclose(dataspaceId);
    delete[] dataCpu;
}


template void writeHdf5DatasetFromGpu<uint16_t>(hid_t & fileId, const char * datasetName, hid_t & dataType, void * dataGpu, hsize_t * dims, uint8_t & rank);
template void writeHdf5DatasetFromGpu<uint32_t>(hid_t & fileId, const char * datasetName, hid_t & dataType, void * dataGpu, hsize_t * dims, uint8_t & rank);
template void writeHdf5DatasetFromGpu<float>(hid_t & fileId, const char * datasetName, hid_t & dataType, void * dataGpu, hsize_t * dims, uint8_t & rank);
template void writeHdf5DatasetFromGpu<__half>(hid_t & fileId, const char * datasetName, hid_t & dataType, void * dataGpu, hsize_t * dims, uint8_t & rank);
template void writeHdf5DatasetFromGpu<cuComplex>(hid_t & fileId, const char * datasetName, hid_t & dataType, void * dataGpu, hsize_t * dims, uint8_t & rank);
template void writeHdf5DatasetFromGpu<__half2>(hid_t & fileId, const char * datasetName, hid_t & dataType, void * dataGpu, hsize_t * dims, uint8_t & rank);
