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
#include "cuphy_internal.h"

// Implementation of polarDecoder interface exposed as an opaque data type to abstract out implementation
// details (polarDecoder  C++ class). polarDecoder is implemented as a C++ class which inherits
// from this interface structure defiend as an empty shell (opaque type is a struct since the interface is C
// compatible). Pointer to the opaque type is also exposed in the interface as a handle to the underlying
// implementation.
struct cuphyPolarDecoder
{};

struct polarDecoderDynDescr
{
    __half**           cwTreeLLRsAddrs;
    cuphyPolarCwPrm_t* pCwPrmsGpu;
    uint32_t**         polCbEstAddrs;
    bool**             listPolScratchAddrs;
    uint8_t*           pPolCrcErrorFlags;
    uint16_t           nPolCws;            //!< number of codewords in this launch
    uint16_t           scSharedSliceBytes; //!< per-warp dynamic shared-memory slice of the SC decoder
};
typedef struct polarDecoderDynDescr polarDecoderDynDescr_t;

// uciPolarDecoder kernel arguments (supplied via descriptors)
typedef struct
{
    polarDecoderDynDescr_t* pDynDescr;
} polarDecoderKernelArgs_t;


class polarDecoder : public cuphyPolarDecoder {
public:
    // setup object state and dynamic component descriptor in prepration towards execution
    void setup(uint16_t                      nPolCws,                     // number of polar codewords
               __half**                      pCwTreeLLRsAddrs,            // pointer to codeword tree LLR addresses
               cuphyPolarCwPrm_t*            pCwPrmsGpu,                  // pointer to codeword parameters in GPU
               cuphyPolarCwPrm_t*            pCwPrmsCpu,                  // pointer to codeword parameters in CPU
               uint32_t**                    pPolCbEstAddrs,              // pointer to estimated codeblock addresses
               bool**                        pListPolScratchAddrs,        // pointer to scratch buffer used in list polar decoder
               uint8_t                       nPolLists,                   // list size for polar decoder
               uint8_t*                      pPolCrcErrorFlags,           // pointer to buffer storing CRC error flags
               bool                          enableCpuToGpuDescrAsyncCpy, // option to copy descriptors from CPU to GPU
               polarDecoderDynDescr_t*       pCpuDynDesc,                 // pointer to descriptor in cpu
               void*                         pGpuDynDesc,                 // pointer to descriptor in gpu
               cuphyPolarDecoderLaunchCfg_t* pLaunchCfg,                  // pointer to launch configuration
               cudaStream_t                  strm);                       // stream to perform copy

    template <int LIST_SZ>
    void kernelSelect(uint16_t                      nPolCws,
                      const cuphyPolarCwPrm_t*      pPolUciSegPrmsCpu,
                      cuphyPolarDecoderLaunchCfg_t* pLaunchCfg);

    static void getDescrInfo(size_t& dynDescrSizeBytes, size_t& dynDescrAlignBytes);

    polarDecoderKernelArgs_t m_kernelArgs;
};