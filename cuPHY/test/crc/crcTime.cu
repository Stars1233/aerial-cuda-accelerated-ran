/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <fstream>
#include <iostream>
#include <string>
#include "crc.hpp"
#include "cuphy_internal.h"
#include "cuphy.h"

using namespace crc;
using namespace cuphy_i;

// utility function for unit test
template <typename baseType>
unsigned long equalCount(baseType* a, baseType* b, unsigned long nElements, const std::string& label = "")
{
    unsigned long popCount = 0;
    for(unsigned long i = 0; i < nElements; i++)
    {
        popCount += a[i] != b[i];
        if(a[i] != b[i])
        {
            std::cout << label << "NOT EQUAL (" << std::dec << i << ") a: " << std::hex << a[i]
                      << " b: " << std::hex << b[i] << "\n";
        }
    }
    return popCount == 0;
}

template <typename baseType>
void linearToCoalesced(baseType*     coalescedData,
                       baseType*     linearData,
                       unsigned long nElements,
                       unsigned long elementSize,
                       unsigned long stride)
{
    for(unsigned long i = 0; i < nElements; i++)
    {
        for(unsigned long j = 0; j < elementSize; j++)
            coalescedData[j * stride + i] = linearData[i * elementSize + j];
    }
}

int main(int argc, char** argv)
{
    if(argc < 5)
    {
        printf("Usage: ./crcTime [NTransportBlocks] [NCodeBlocksPerTransportBlock] [CodeBlockSize] [NumberOfRunsForTiming] (0 means skip timing)\n");
        return -1;
    }

    uint32_t nTBs   = std::stoi(argv[1]);
    uint32_t nCBs   = std::stoi(argv[2]);
    uint32_t CBSize = std::stoi(argv[3]);
    uint32_t NRUNS  = std::stoi(argv[4]);

    if(CBSize < 3 + 1)
    {
        printf(
            "CodeBlockSize must be at least two times the CRC size in bytes (e.g., "
            "6 for "
            "CRC24). The last Code Block contains the CRC of the transport block "
            "and its own CRC.\n");
        return -1;
    }

    if(CBSize > 1056)
    {
        printf("CodeBlockSize must be at most 1056 bytes\n");
        return -1;
    }

    bool timeIt = NRUNS != 0;

    uint32_t* nCodeBlocks            = new uint32_t[nTBs];
    uint32_t  crcPolyByteSize        = 3; // 24 bits
    uint32_t* firstCodeBlockIdxArray = new uint32_t[nTBs];
    uint32_t* codeBlockByteSizes     = new uint32_t[nTBs]; // 56;
    uint32_t* codeBlockWordSizes     = new uint32_t[nTBs]; // 56;
    uint32_t* codeBlockDataByteSizes = new uint32_t[nTBs];
    // 1053;
    uint32_t* CBPaddingByteSizes      = new uint32_t[nTBs]; // pad to 32-bit boundary
    uint32_t* totalCodeBlockByteSizes = new uint32_t[nTBs];
    uint32_t  totalByteSize           = 0;
    uint32_t  totalNCodeBlocks        = 0;
    uint32_t* tbPaddedByteSizes       = new uint32_t[nTBs];
    uint32_t  totalTBPaddedByteSize   = 0;
    uint32_t  ratio                   = sizeof(uint32_t) / sizeof(uint8_t);
    // Same CRC value for each code block, code blocks are all the same
    // linear input layout : cb1|crc1, cb2|crc2, ...

    for(uint32_t i = 0; i < nTBs; i++)
    {
        codeBlockByteSizes[i] = CBSize;
        nCodeBlocks[i]        = nCBs;
        totalNCodeBlocks += nCodeBlocks[i];
        codeBlockDataByteSizes[i]  = codeBlockByteSizes[i] - crcPolyByteSize;
        CBPaddingByteSizes[i]      = (MAX_BYTES_PER_CODE_BLOCK - (codeBlockByteSizes[i] % MAX_BYTES_PER_CODE_BLOCK)) % MAX_BYTES_PER_CODE_BLOCK;
        totalCodeBlockByteSizes[i] = codeBlockByteSizes[i] + CBPaddingByteSizes[i];
        codeBlockWordSizes[i]      = totalCodeBlockByteSizes[i] / ratio;
        totalByteSize += totalCodeBlockByteSizes[i] * nCodeBlocks[i];
        tbPaddedByteSizes[i] =
            codeBlockDataByteSizes[i] * nCodeBlocks[i] +
            (4 - ((nCodeBlocks[i] * codeBlockDataByteSizes[i]) % 4)) % 4;
        totalTBPaddedByteSize += tbPaddedByteSizes[i];
    }

    PerTbParams* tbPrmsArray           = new PerTbParams[nTBs];
    uint8_t*     linearInput           = new uint8_t[totalByteSize];
    uint32_t*    goldenCRCs            = new uint32_t[totalNCodeBlocks];
    uint8_t*     goldenTransportBlocks = new uint8_t[totalTBPaddedByteSize];
    uint32_t*    transportBlocks       = new uint32_t[totalTBPaddedByteSize / ratio];
    uint32_t*    codeBlocks            = (uint32_t*)linearInput;
    uint32_t*    crcs                  = new uint32_t[totalNCodeBlocks];
    uint32_t*    tbCRCs                = new uint32_t[nTBs];
    memset(goldenTransportBlocks, 0, totalTBPaddedByteSize);
    memset(firstCodeBlockIdxArray, 0, nTBs);
    uint32_t tbBytes      = 0;
    uint32_t totalCBBytes = 0;
    for(uint32_t t = 0; t < nTBs; t++)
    {
        // Build transport block
        uint32_t cbBytes = 0;
        uint32_t ncbs    = 0;
        for(uint32_t i = 0; i < nCodeBlocks[t] - 1; i++)
        {
            memset(goldenTransportBlocks + tbBytes + cbBytes,
                   i + 1,
                   codeBlockDataByteSizes[t]);
            cbBytes += codeBlockDataByteSizes[t];
        }
        // last code block contains TB CRC in the last 3 bytes
        uint8_t* lptr = goldenTransportBlocks + tbBytes + cbBytes;
        memset(lptr,
               ncbs + nCodeBlocks[t],
               codeBlockDataByteSizes[t] - crcPolyByteSize);
        uint32_t golden_tbCRC = computeCRC<uint32_t, 24>(goldenTransportBlocks + tbBytes,
                                                         codeBlockDataByteSizes[t] * nCodeBlocks[t] - crcPolyByteSize,
                                                         G_CRC_24_A,
                                                         0,
                                                         1);
        for(uint32_t j = 0; j < crcPolyByteSize; j++)
            goldenTransportBlocks[tbBytes +
                                  nCodeBlocks[t] * codeBlockDataByteSizes[t] -
                                  crcPolyByteSize + j] =
                (golden_tbCRC >> (crcPolyByteSize - 1 - j) * 8) & 0xFF;

        // compute CB crcs
        for(uint32_t i = 0; i < nCodeBlocks[t]; i++)
        {
            uint8_t* cbPtr  = linearInput + i * totalCodeBlockByteSizes[t] + totalCBBytes;
            uint8_t* crcPtr = (cbPtr + codeBlockDataByteSizes[t]);
            memcpy(cbPtr,
                   goldenTransportBlocks + i * codeBlockDataByteSizes[t] + tbBytes,
                   codeBlockDataByteSizes[t]);
            uint32_t crc = computeCRC<uint32_t, 24>((uint8_t*)cbPtr,
                                                    codeBlockDataByteSizes[t],
                                                    G_CRC_24_B,
                                                    0,
                                                    1);
            for(uint32_t j = 0; j < crcPolyByteSize; j++)
                crcPtr[j] = (crc >> (crcPolyByteSize - 1 - j) * 8) & 0xFF;
            goldenCRCs[t * nCodeBlocks[t] + i] = 0;
            memset(cbPtr + codeBlockByteSizes[t], 0, CBPaddingByteSizes[t]);
        }
        tbBytes += tbPaddedByteSizes[t];
        ncbs += nCodeBlocks[t];
        totalCBBytes += nCodeBlocks[t] * totalCodeBlockByteSizes[t];
        tbPrmsArray[t].num_CBs             = nCodeBlocks[t];
        tbPrmsArray[t].K                   = codeBlockByteSizes[t] * 8;
        tbPrmsArray[t].F                   = 0;
        tbPrmsArray[t].firstCodeBlockIndex = 0;
    }
#if 0
  std::cout << "CBs:\n";
  for (int i = 0; i < totalByteSize; i++)
    std::cout << std::hex << (unsigned short)linearInput[i] << ",";
  std::cout << "\n";

  std::cout << "TB:\n";
  for (int i = 0; i < totalTBPaddedByteSize; i++)
    std::cout << std::hex << (unsigned short)goldenTransportBlocks[i] << ",";
  std::cout << "\n";
#endif

    unsigned long npassed = 1;

    //input
    unique_device_ptr<uint32_t> d_codeBlocks = make_unique_device<uint32_t>(totalByteSize / sizeof(uint32_t));

    unique_device_ptr<PerTbParams> d_tbPrmsArray = make_unique_device<PerTbParams>(nTBs);
    //output

    unique_device_ptr<uint32_t> d_CBCRCs = make_unique_device<uint32_t>(nTBs * nCodeBlocks[0]);
    unique_device_ptr<uint32_t> d_TBCRCs = make_unique_device<uint32_t>(nTBs);
    unique_device_ptr<uint8_t>  d_TBs    = make_unique_device<uint8_t>(tbBytes);

    cudaMemcpy(d_codeBlocks.get(), codeBlocks, totalByteSize, cudaMemcpyHostToDevice);

    cudaMemcpy(d_tbPrmsArray.get(), tbPrmsArray, sizeof(PerTbParams) * nTBs, cudaMemcpyHostToDevice);

    cuphyStatus_t status = cuphyCRCDecode(
        d_CBCRCs.get(),
        d_TBCRCs.get(),
        d_TBs.get(),
        d_codeBlocks.get(),
        d_tbPrmsArray.get(),
        nTBs,
        nCBs,
        tbPaddedByteSizes[0],
        0,
        timeIt,
        NRUNS,
        false,
        0);

    cudaMemcpy(crcs, d_CBCRCs.get(), sizeof(uint32_t) * nTBs * nCBs, cudaMemcpyDeviceToHost);

    cudaMemcpy(tbCRCs, d_TBCRCs.get(), sizeof(uint32_t) * nTBs, cudaMemcpyDeviceToHost);

    cudaMemcpy(transportBlocks, d_TBs.get(), totalTBPaddedByteSize, cudaMemcpyDeviceToHost);

    if(status != CUPHY_STATUS_SUCCESS)
        printf("CRC: CUPHY ERROR\n");
    else
    {
        uint32_t*     gt      = (uint32_t*)goldenTransportBlocks;
        npassed = equalCount(crcs, goldenCRCs, totalNCodeBlocks, "CB CRC ");
        npassed &= equalCount(transportBlocks, gt, totalTBPaddedByteSize / ratio, "TB DATA ");

        for(uint32_t i = 0; i < nTBs; i++)
        {
            npassed &= (tbCRCs[i] == 0);
            if(tbCRCs[i] != 0)
                printf("TB[%u] CRC not equal to 0: %x\n", i, tbCRCs[i]);
        }
    }

    delete[] tbPrmsArray;
    delete[] crcs;
    delete[] tbCRCs;
    delete[] goldenCRCs;
    delete[] goldenTransportBlocks;
    delete[] linearInput;
    delete[] transportBlocks;
    delete[] nCodeBlocks;
    delete[] codeBlockByteSizes;
    delete[] codeBlockWordSizes;
    delete[] codeBlockDataByteSizes;
    delete[] CBPaddingByteSizes; // pad to 32-bit boundary
    delete[] totalCodeBlockByteSizes;
    delete[] tbPaddedByteSizes;

    return !npassed;
}
