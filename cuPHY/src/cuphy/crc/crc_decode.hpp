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

#include <functional>
#include "cuphy.h"

struct puschRxCrcDecodeDescr
{
    uint32_t*          pOutputCBCRCs;
    uint8_t*           pOutputTBs;
    const uint32_t*    pInputCodeBlocks;
    uint32_t*          pOutputTBCRCs;
    const PerTbParams* pTbPrmsArray;
    bool               reverseBytes;
    uint16_t           schUserIdxs[MAX_N_TBS_PER_CELL_GROUP_SUPPORTED];
};
typedef struct puschRxCrcDecodeDescr puschRxCrcDecodeDescr_t;

class puschRxCrcDecode : public cuphyPuschRxCrcDecode {
public:
    puschRxCrcDecode()                                   = default;
    ~puschRxCrcDecode()                                  = default;
    puschRxCrcDecode(puschRxCrcDecode const&)            = delete;
    puschRxCrcDecode& operator=(puschRxCrcDecode const&) = delete;

    static void getDescrInfo(size_t& descrSizeBytes, size_t& descrAlignBytes);

    void init(int reverseBytes);

    cuphyStatus_t setup(uint16_t                          nSchUes,
                        uint16_t*                         pSchUserIdxsCpu,
                        uint32_t*                         pOutputCBCRCs,
                        uint8_t*                          pOutputTBs,
                        const uint32_t*                   pInputCodeBlocks,
                        uint32_t*                         pOutputTBCRCs,
                        const PerTbParams*                pTbPrmsCpu,
                        const PerTbParams*                pTbPrmsGpu,
                        void*                             pCpuDesc,
                        void*                             pGpuDesc,
                        uint8_t                           enableCpuToGpuDescrAsyncCpy,
                        cuphyPuschRxCrcDecodeLaunchCfg_t* pCbCrcLaunchCfg,
                        cuphyPuschRxCrcDecodeLaunchCfg_t* pTbCrcLaunchCfg,
                        cudaStream_t                      strm);

private:
    // class state modifed by setup saved in data member
    bool       m_reverseBytes;  // option to reverse order of bytes in each word before computing the CRC
    CUfunction m_cbCrcKernelFunc;
    CUfunction m_tbCrcKernelFunc;
};
