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

#if !defined(GPU3GPPCHAN_COMMON_INCLUDED_)
#define GPU3GPPCHAN_COMMON_INCLUDED_

#include <cstdint>
#include <stdio.h>
#include <iostream>
#include <stdexcept>
#include <string>
#include "hdf5.h"
#include "cuda.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "cuComplex.h"
#include <curand.h>

// copied function generate_native_HDF5_fp16_type() from aerial_sdk/cuPHY/src/cuphy_hdf5/cuphy_hdf5.cpp, which creates __half to be used in saving to hdf5
////////////////////////////////////////////////////////////////////////
// generate_native_HDF5_fp16_type()
// Note: caller should call H5Tclose() on the returned type
hid_t generate_native_HDF5_fp16_type();

/**
 * @brief helper function to write a dataset to hdf5 file
 * 
 * @tparam T data type of the data, used to copy from GPU to CPU
 * @param fileId hdf5 file id
 * @param datasetName dataset name
 * @param dataType h5 compound data type, has to match with T
 * @param dataspaceId data sapce id
 * @param dataGpu pointer to GPU data, data can be 1D array
 * @param dims dimentions of data
 * @param rank rand of dimension
 */
template <typename T>
void writeHdf5DatasetFromGpu(hid_t & fileId, const char * datasetName, hid_t & dataType, void * dataGpu, hsize_t * dims, uint8_t & rank);

#define CHECK_CURESULT(status) \
    do { \
        CUresult cuStatus = (status); \
        if (cuStatus != CUDA_SUCCESS) { \
            const char* cuErrorString = nullptr; \
            (void)cuGetErrorString(cuStatus, &cuErrorString); \
            throw std::runtime_error( \
                std::string("CUDA driver error: ") + \
                (cuErrorString != nullptr ? cuErrorString : "unknown CUDA error") + \
                " at line " + std::to_string(__LINE__) + " in file " + __FILE__); \
        } \
    } while (0)

// Destructors must not throw. Cleanup failures are reported without terminating
// the embedding process or masking an exception already in flight.
#define CHECK_CURESULT_NOEXCEPT(status) \
    do { \
        CUresult cuStatus = (status); \
        if (cuStatus != CUDA_SUCCESS) { \
            const char* cuErrorString = nullptr; \
            (void)cuGetErrorString(cuStatus, &cuErrorString); \
            fprintf(stderr, "CUDA cleanup error: %s at line %d in file %s\n", \
                    cuErrorString != nullptr ? cuErrorString : "unknown CUDA error", \
                    __LINE__, __FILE__); \
        } \
    } while (0)

#define CHECK_CUDAERROR(status) \
    do { \
        cudaError_t cudaStatus = (status); \
        if (cudaStatus != cudaSuccess) { \
            throw std::runtime_error( \
                std::string("CUDA runtime error: ") + cudaGetErrorString(cudaStatus) + \
                " at line " + std::to_string(__LINE__) + " in file " + __FILE__); \
        } \
    } while (0)

#define CHECK_CURANDERROR(status) \
    do { \
        curandStatus_t curandStatus = (status); \
        if (curandStatus != CURAND_STATUS_SUCCESS) { \
            throw std::runtime_error( \
                std::string("cuRAND error: status ") + \
                std::to_string(static_cast<int>(curandStatus)) + \
                " at line " + std::to_string(__LINE__) + " in file " + __FILE__); \
        } \
    } while (0)

// Define the macro for assert with a custom message
#define ASSERT(condition, message) \
    do { \
        if (!(condition)) { \
            fprintf(stderr, "Assertion failed: (%s), %s, file: %s, line: %d\n", \
                    #condition, message, __FILE__, __LINE__); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

static uint32_t getCudaDeviceArch() {
    int device;
    CHECK_CUDAERROR(cudaGetDevice(&device));

    int major = 0;
    int minor = 0;
    CHECK_CUDAERROR(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
    CHECK_CUDAERROR(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device));

    return static_cast<uint32_t>(major) * 100 + static_cast<uint32_t>(minor) * 10;
}

#endif // !defined(GPU3GPPCHAN_COMMON_INCLUDED_)
