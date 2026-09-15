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

#include <gtest/gtest.h>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include "crc.hpp"
#include "cuphy.h"
#include "cuphy.hpp"
#include "cuphy_internal.h"
#include "crc_decode.hpp"
#include "crc_encode.hpp"

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

class PuschCrcDecodeHandle {
public:
    explicit PuschCrcDecodeHandle(int reverseBytes)
        : m_status(cuphyCreatePuschRxCrcDecode(&m_handle, reverseBytes))
    {
    }

    ~PuschCrcDecodeHandle()
    {
        if(m_handle != nullptr)
        {
            cuphyDestroyPuschRxCrcDecode(m_handle);
        }
    }

    PuschCrcDecodeHandle(PuschCrcDecodeHandle const&)            = delete;
    PuschCrcDecodeHandle& operator=(PuschCrcDecodeHandle const&) = delete;

    cuphyStatus_t status() const
    {
        return m_status;
    }

    bool valid() const
    {
        return m_status == CUPHY_STATUS_SUCCESS && m_handle != nullptr;
    }

    cuphyPuschRxCrcDecodeHndl_t get() const
    {
        return m_handle;
    }

private:
    cuphyPuschRxCrcDecodeHndl_t m_handle = nullptr;
    cuphyStatus_t               m_status = CUPHY_STATUS_SUCCESS;
};

cuphyStatus_t runPuschCrcDecode(uint32_t*    h_cbCRCs,
                                uint32_t*    h_tbCRCs,
                                uint8_t*     h_transportBlocks,
                                const uint8_t* h_codeBlocks,
                                const PerTbParams* h_tbPrmsArray,
                                uint32_t     nTBs,
                                uint32_t     totalByteSize,
                                uint32_t     totalNCodeBlocks,
                                uint32_t     totalTBPaddedByteSize,
                                int          reverseBytes)
{
    try
    {
        PuschCrcDecodeHandle puschDecoder(reverseBytes);
        if(!puschDecoder.valid())
        {
            return puschDecoder.status();
        }

        size_t puschDescSize = 0;
        size_t puschDescAlign = 0;
        cuphyStatus_t status = cuphyPuschRxCrcDecodeGetDescrInfo(&puschDescSize, &puschDescAlign);
        if(status != CUPHY_STATUS_SUCCESS)
        {
            return status;
        }

        cuphy::unique_pinned_ptr<uint8_t> h_puschDesc = cuphy::make_unique_pinned<uint8_t>(puschDescSize);
        cuphy::unique_device_ptr<uint8_t>     d_puschDesc       = cuphy::make_unique_device<uint8_t>(puschDescSize);
        cuphy::unique_device_ptr<uint8_t>     d_inputCodeBlocks = cuphy::make_unique_device<uint8_t>(totalByteSize);
        cuphy::unique_device_ptr<PerTbParams> d_puschTbParams   = cuphy::make_unique_device<PerTbParams>(nTBs);
        cuphy::unique_device_ptr<uint32_t>    d_decodeCBCRCs    = cuphy::make_unique_device<uint32_t>(totalNCodeBlocks);
        cuphy::unique_device_ptr<uint32_t>    d_decodeTBCRCs    = cuphy::make_unique_device<uint32_t>(nTBs);
        cuphy::unique_device_ptr<uint8_t>     d_decodedTBs      = cuphy::make_unique_device<uint8_t>(totalTBPaddedByteSize);

        CUDA_CHECK(cudaMemcpy(d_inputCodeBlocks.get(), h_codeBlocks, totalByteSize, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_puschTbParams.get(), h_tbPrmsArray, sizeof(PerTbParams) * nTBs, cudaMemcpyHostToDevice));

        std::vector<uint16_t> schUserIdxs(nTBs);
        for(uint32_t i = 0; i < nTBs; ++i)
        {
            schUserIdxs[i] = static_cast<uint16_t>(i);
        }

        cuphyPuschRxCrcDecodeLaunchCfg_t cbCrcLaunchCfg = {};
        cuphyPuschRxCrcDecodeLaunchCfg_t tbCrcLaunchCfg = {};
        status = cuphySetupPuschRxCrcDecode(puschDecoder.get(),
                                            nTBs,
                                            schUserIdxs.data(),
                                            d_decodeCBCRCs.get(),
                                            d_decodedTBs.get(),
                                            reinterpret_cast<uint32_t*>(d_inputCodeBlocks.get()),
                                            d_decodeTBCRCs.get(),
                                            h_tbPrmsArray,
                                            d_puschTbParams.get(),
                                            h_puschDesc.get(),
                                            d_puschDesc.get(),
                                            1,
                                            &cbCrcLaunchCfg,
                                            &tbCrcLaunchCfg,
                                            0);
        if(status != CUPHY_STATUS_SUCCESS)
        {
            return status;
        }

        CUresult status_k1 = launch_kernel(cbCrcLaunchCfg.kernelNodeParamsDriver, 0);
        CUresult status_k2 = launch_kernel(tbCrcLaunchCfg.kernelNodeParamsDriver, 0);
        if((status_k1 != CUDA_SUCCESS) || (status_k2 != CUDA_SUCCESS))
        {
            return CUPHY_STATUS_INTERNAL_ERROR;
        }
        CUDA_CHECK(cudaStreamSynchronize(0));

        CUDA_CHECK(cudaMemcpy(h_cbCRCs, d_decodeCBCRCs.get(), sizeof(uint32_t) * totalNCodeBlocks, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_tbCRCs, d_decodeTBCRCs.get(), sizeof(uint32_t) * nTBs, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_transportBlocks, d_decodedTBs.get(), totalTBPaddedByteSize, cudaMemcpyDeviceToHost));

        return CUPHY_STATUS_SUCCESS;
    }
    catch(const std::exception& e)
    {
        return CUPHY_STATUS_INTERNAL_ERROR;
    }
}

int CRC_GPU_UPLINK_PUSCH_TEST(bool timeIt)
{
    (void)timeIt;
    const uint32_t nTBs = MAX_N_TBS_SUPPORTED;
    std::vector<uint32_t> nCodeBlocks(nTBs);
    std::vector<uint32_t> codeBlockByteSizes(nTBs);
    std::vector<uint32_t> codeBlockDataByteSizes(nTBs);
    std::vector<uint32_t> CBPaddingByteSizes(nTBs);
    std::vector<uint32_t> crcByteSizes(nTBs);
    std::vector<uint32_t> totalCodeBlockByteSizes(nTBs);
    std::vector<uint32_t> tbPaddedByteSizes(nTBs);
    uint32_t totalByteSize         = 0;
    uint32_t totalNCodeBlocks      = 0;
    uint32_t totalTBPaddedByteSize = 0;
    // Same CRC value for each code block, code blocks are all the same
    // linear input layout : cb1|crc1, cb2|crc2, ...

    for(int i = 0; i < nTBs; i++)
    {
        if(i == 1)
        {
            nCodeBlocks[i]        = 1;
            crcByteSizes[i]       = 2;
            codeBlockByteSizes[i] = 333;
        }
        else if(i == 2)
        {
            nCodeBlocks[i]        = 1;
            crcByteSizes[i]       = 3;
            codeBlockByteSizes[i] = 945;
        }

        else
        {
            codeBlockByteSizes[i] = 1007;
            nCodeBlocks[i]        = 6;
            crcByteSizes[i]       = 3;
        }
        totalNCodeBlocks += nCodeBlocks[i];
        codeBlockDataByteSizes[i]  = codeBlockByteSizes[i] - crcByteSizes[i];
        CBPaddingByteSizes[i]      = (MAX_BYTES_PER_CODE_BLOCK - (codeBlockByteSizes[i] % MAX_BYTES_PER_CODE_BLOCK)) % MAX_BYTES_PER_CODE_BLOCK;
        totalCodeBlockByteSizes[i] = codeBlockByteSizes[i] + CBPaddingByteSizes[i];
        totalByteSize += totalCodeBlockByteSizes[i] * nCodeBlocks[i];
        tbPaddedByteSizes[i] = (codeBlockDataByteSizes[i] + (nCodeBlocks[i] == 1 ? crcByteSizes[i] : 0)) * nCodeBlocks[i] +
                               (4 - (nCodeBlocks[i] * (codeBlockDataByteSizes[i] + (nCodeBlocks[i] == 1 ? crcByteSizes[i] : 0)) % 4)) % 4;
        totalTBPaddedByteSize += tbPaddedByteSizes[i];
    }

    std::vector<PerTbParams> tbPrmsArray(nTBs);
    std::vector<uint8_t>     linearInput(totalByteSize, 0);
    std::vector<uint32_t>    goldenCRCs(totalNCodeBlocks, 0);
    std::vector<uint8_t>     goldenTransportBlocks(totalTBPaddedByteSize, 0);
    std::vector<uint8_t>     transportBlocks(totalTBPaddedByteSize, 0);
    uint8_t*                 codeBlocks = linearInput.data();
    std::vector<uint32_t>    crcs(totalNCodeBlocks, 0);
    std::vector<uint32_t>    tbCRCs(nTBs, 0);
    memset(tbPrmsArray.data(), 0, sizeof(PerTbParams) * tbPrmsArray.size());
    uint32_t tbBytes      = 0;
    uint32_t totalCBBytes = 0;
    uint32_t totalCBs     = 0;
    for(int t = 0; t < nTBs; t++)
    {
        // Build transport block
        uint32_t cbBytes = 0;

        for(int i = 0; i < nCodeBlocks[t]; i++)
        {
            memset(goldenTransportBlocks.data() + tbBytes + cbBytes,
                   rand(),
                   codeBlockDataByteSizes[t]);
            cbBytes += codeBlockDataByteSizes[t];
        }

        // last code block contains TB CRC in the last 3 bytes
        if(nCodeBlocks[t] > 1)
        {
            uint32_t golden_tbCRC = computeCRC<uint32_t, 24>(goldenTransportBlocks.data() + tbBytes,
                                                             codeBlockDataByteSizes[t] * nCodeBlocks[t] - crcByteSizes[t],
                                                             G_CRC_24_A,
                                                             0,
                                                             1);
            for(int j = 0; j < crcByteSizes[t]; j++)
                goldenTransportBlocks[tbBytes + nCodeBlocks[t] * codeBlockDataByteSizes[t] - crcByteSizes[t] + j] =
                    (golden_tbCRC >> (crcByteSizes[t] - 1 - j) * 8) & 0xFF;
        }
        // compute CB crcs
        for(int i = 0; i < nCodeBlocks[t]; i++)
        {
            uint8_t* cbPtr  = linearInput.data() + i * totalCodeBlockByteSizes[t] + totalCBBytes;
            uint8_t* crcPtr = (cbPtr + codeBlockDataByteSizes[t]);
            memcpy(cbPtr,
                   goldenTransportBlocks.data() + i * codeBlockDataByteSizes[t] + tbBytes,
                   codeBlockDataByteSizes[t]);
            uint32_t crc;
            if(nCodeBlocks[t] == 1)
            {
                if(codeBlockDataByteSizes[t] <= MAX_SMALL_A_BYTES)
                {
                    crc = computeCRC<uint32_t, 16>((uint8_t*)cbPtr,
                                                   codeBlockDataByteSizes[t],
                                                   G_CRC_16,
                                                   0,
                                                   1);
                }
                else
                    crc = computeCRC<uint32_t, 24>((uint8_t*)cbPtr,
                                                   codeBlockDataByteSizes[t],
                                                   G_CRC_24_A,
                                                   0,
                                                   1);
                for(int j = 0; j < crcByteSizes[t]; j++)
                    goldenTransportBlocks[tbBytes + nCodeBlocks[t] * codeBlockDataByteSizes[t] /*- crcByteSizes[t]*/ + j] =
                        (crc >> (crcByteSizes[t] - 1 - j) * 8) & 0xFF;
            }

            else

                crc = computeCRC<uint32_t, 24>((uint8_t*)cbPtr,
                                               codeBlockDataByteSizes[t],
                                               G_CRC_24_B,
                                               0,
                                               1);

            for(int j = 0; j < crcByteSizes[t]; j++)
                crcPtr[j] = (crc >> (crcByteSizes[t] - 1 - j) * 8) & 0xFF;
            goldenCRCs[totalCBs] = 0;
            totalCBs++;
            memset(cbPtr + codeBlockByteSizes[t], 0, CBPaddingByteSizes[t]);
        }
        tbPrmsArray[t].num_CBs             = nCodeBlocks[t];
        tbPrmsArray[t].K                   = codeBlockByteSizes[t] * 8;
        tbPrmsArray[t].F                   = 0;
        tbPrmsArray[t].firstCodeBlockIndex = 0;
        tbPrmsArray[t].nDataBytes          = tbPaddedByteSizes[t];
        tbBytes += tbPaddedByteSizes[t];
        totalCBBytes += nCodeBlocks[t] * totalCodeBlockByteSizes[t];
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

    cuphyStatus_t status = runPuschCrcDecode(crcs.data(),
                                             tbCRCs.data(),
                                             transportBlocks.data(),
                                             codeBlocks,
                                             tbPrmsArray.data(),
                                             nTBs,
                                             totalByteSize,
                                             totalNCodeBlocks,
                                             totalTBPaddedByteSize,
                                             0);

    int passed = 0;

    if(status != CUPHY_STATUS_SUCCESS)
        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT,  "CRC: CUPHY ERROR");
    else
    {
        passed = equalCount(crcs.data(), goldenCRCs.data(), totalNCodeBlocks, "CB CRC ");
        passed &= equalCount(transportBlocks.data(), goldenTransportBlocks.data(), totalTBPaddedByteSize, "TB DATA");

        for(int i = 0; i < nTBs; i++)
        {
            passed &= (tbCRCs[i] == 0);
            if(tbCRCs[i] != 0)
                NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT,  "TB[{}] CRC not equal to 0: {}", i, tbCRCs[i]);
        }
    }

    return passed;
}

int CRC_GPU_DOWNLINK_PDSCH(bool timeIt, int numCBs = 2) // FIXME numCBs has no effect!
{
    const uint32_t nTBs = 2;
    std::vector<uint32_t> nCodeBlocks(nTBs);
    std::vector<uint32_t> crcByteSizes(nTBs);
    std::vector<uint32_t> codeBlockByteSizes(nTBs);
    std::vector<uint32_t> codeBlockDataByteSizes(nTBs);
    std::vector<uint32_t> cbPaddingByteSizes(nTBs);
    std::vector<uint32_t> fillerByteSizes(nTBs);
    std::vector<uint32_t> tensorStrideByteSizes(nTBs);
    std::vector<uint32_t> totalCodeBlockByteSizes(nTBs);
    std::vector<uint32_t> tbPaddedByteSizes(nTBs);
    uint32_t totalNCodeBlocks    = 0;
    uint32_t totalInTBByteSize   = 0;
    uint32_t totalOutTBByteSizes = 0;
    uint32_t ratio               = sizeof(uint32_t) / sizeof(uint8_t);

    for(int i = 0; i < nTBs; i++)
    {
        codeBlockByteSizes[i]    = 1056;
        crcByteSizes[i]          = 3;
        nCodeBlocks[i]           = 2;
        fillerByteSizes[i]       = 0;
        tensorStrideByteSizes[i] = 0;

        codeBlockDataByteSizes[i] = codeBlockByteSizes[i] - crcByteSizes[i] - fillerByteSizes[i];
        totalNCodeBlocks += nCodeBlocks[i];
        tbPaddedByteSizes[i] = codeBlockDataByteSizes[i] * nCodeBlocks[i];
        tbPaddedByteSizes[i] += (4 - (tbPaddedByteSizes[i] % 4)) % 4;
        totalInTBByteSize += tbPaddedByteSizes[i];
        totalCodeBlockByteSizes[i] = codeBlockByteSizes[i] + tensorStrideByteSizes[i];
        cbPaddingByteSizes[i]      = (4 - totalCodeBlockByteSizes[i] % 4) % 4;
        totalCodeBlockByteSizes[i] += cbPaddingByteSizes[i];
        totalOutTBByteSizes += totalCodeBlockByteSizes[i] * nCodeBlocks[i]; // should be a multiple of 4
    }

    std::vector<PdschPerTbParams> tbPrmsArray(nTBs);
    std::vector<uint8_t>  goldenTransportBlocks(totalInTBByteSize, 0);
    std::vector<uint32_t> crcs(totalNCodeBlocks, 0);
    std::vector<uint32_t> tbCRCs(nTBs, 0);
    std::vector<uint32_t> goldenTBCRCs(nTBs, 0);
    std::vector<uint32_t> goldenCRCs(totalNCodeBlocks, 0);
    std::vector<uint8_t>  codeBlocks(totalOutTBByteSizes, 0);
    memset(tbPrmsArray.data(), 0, sizeof(PdschPerTbParams) * tbPrmsArray.size());

    uint32_t tbOutBytes = 0;
    uint32_t tbBytes    = 0;
    uint8_t* outputs    = codeBlocks.data();
    for(int t = 0; t < nTBs; t++)
    {
        uint32_t cbBytes    = 0;
        uint32_t cbOutBytes = 0;
        for(int i = 0; i < nCodeBlocks[t]; i++)
        {
            // Set the input
            memset(goldenTransportBlocks.data() + tbBytes + cbBytes,
                   rand(),
                   codeBlockDataByteSizes[t]);
            memcpy(outputs + tbOutBytes + cbOutBytes,
                   goldenTransportBlocks.data() + tbBytes + cbBytes,
                   codeBlockDataByteSizes[t]);
            cbBytes += codeBlockDataByteSizes[t];
            cbOutBytes += totalCodeBlockByteSizes[t];
        }
        tbPrmsArray[t].num_CBs             = nCodeBlocks[t];
        tbPrmsArray[t].K                   = codeBlockByteSizes[t] * 8;
        tbPrmsArray[t].F                   = 0;
        tbPrmsArray[t].firstCodeBlockIndex = 0;
        // Add tbSize, tbStartOffset and paddingBytes fields
        tbPrmsArray[t].tbSize        = codeBlockDataByteSizes[t] * nCodeBlocks[t] - 3;
        tbPrmsArray[t].tbStartOffset = (t == 0) ? 0 : (tbPrmsArray[t - 1].tbStartOffset + tbPrmsArray[t - 1].tbSize);
        tbPrmsArray[t].tbStartAddr   = nullptr; // not used in CRC encode
        //tbPrmsArray[t].paddingBytes  = tbPaddedByteSizes[t] - tbPrmsArray[t].tbSize;
        tbPrmsArray[t].cumulativeTbSizePadding =  tbPaddedByteSizes[t] + ((t == 0) ? 0 : tbPrmsArray[t-1].cumulativeTbSizePadding);
        tbPrmsArray[t].testModel               =  0; // Assume no cell is in testing mode

        // The last CB in one TB is 3 bytes shorter than other CBs
        memset(goldenTransportBlocks.data() + tbBytes +
                   nCodeBlocks[t] * codeBlockDataByteSizes[t] - 3,
               0,
               3);
        memset(outputs + tbOutBytes + (nCodeBlocks[t] - 1) * totalCodeBlockByteSizes[t] +
                   codeBlockDataByteSizes[t] - 3,
               0,
               3);
        tbBytes += tbPaddedByteSizes[t];
        tbOutBytes += totalCodeBlockByteSizes[t] * nCodeBlocks[t];
    }

    //input
    cuphy::unique_device_ptr<uint32_t>    d_transportBlocks = cuphy::make_unique_device<uint32_t>(totalInTBByteSize / ratio);
    cuphy::unique_device_ptr<PdschPerTbParams> d_tbPrmsArray     = cuphy::make_unique_device<PdschPerTbParams>(nTBs);

    //output

    cuphy::unique_device_ptr<uint32_t> d_CBCRCs     = cuphy::make_unique_device<uint32_t>(nTBs * nCodeBlocks[0]);
    cuphy::unique_device_ptr<uint32_t> d_TBCRCs     = cuphy::make_unique_device<uint32_t>(nTBs);
    cuphy::unique_device_ptr<uint8_t>  d_codeBlocks = cuphy::make_unique_device<uint8_t>(totalOutTBByteSizes);

    cudaMemcpy(d_transportBlocks.get(), reinterpret_cast<uint32_t*>(goldenTransportBlocks.data()), totalInTBByteSize, cudaMemcpyHostToDevice);

    //cudaMemcpy(d_codeBlocks.get(), (uint8_t*)codeBlocks, totalOutTBByteSizes, cudaMemcpyHostToDevice);

    CUDA_CHECK(cudaMemcpy(d_tbPrmsArray.get(), tbPrmsArray.data(), sizeof(PdschPerTbParams) * nTBs, cudaMemcpyHostToDevice));

    // Allocate launch config struct.
    std::unique_ptr<cuphyCrcEncodeLaunchConfig> crc_hndl = std::make_unique<cuphyCrcEncodeLaunchConfig>();

    // Allocate descriptors and setup rate matching component
    uint8_t       desc_async_copy = 1; // Copy descriptor to the GPU during setup. And set TB-CRCs to 0.
    size_t        desc_size = 0, alloc_size = 0;
    cuphyStatus_t status = cuphyCrcEncodeGetDescrInfo(&desc_size, &alloc_size);
    if(status != CUPHY_STATUS_SUCCESS)
    {
        printf("cuphyCrcEncodeGetDescrInfo error %d\n", status);
    }
    cuphy::unique_device_ptr<uint8_t> d_crc_encode_desc = cuphy::make_unique_device<uint8_t>(desc_size);
    cuphy::unique_pinned_ptr<uint8_t> h_crc_encode_desc = cuphy::make_unique_pinned<uint8_t>(desc_size);

    cudaStream_t cuda_strm = 0;

    status = cuphySetupCrcEncode(crc_hndl.get(),
                                 d_CBCRCs.get(),
                                 d_TBCRCs.get(),
                                 d_transportBlocks.get(),
                                 d_codeBlocks.get(),
                                 d_tbPrmsArray.get(),
                                 nTBs,
                                 nCodeBlocks[0],
                                 tbPaddedByteSizes[0],
                                 0,
                                 false,
                                 h_crc_encode_desc.get(),
                                 d_crc_encode_desc.get(),
                                 desc_async_copy,
                                 cuda_strm);

    if(status != CUPHY_STATUS_SUCCESS)
    {
        throw std::runtime_error("Invalid argument(s) for cuphySetupCrcEncode");
    }

    // CRC has 2 kernels right now
    CUresult status_k1 = launch_kernel(crc_hndl.get()->m_kernelNodeParams[0], cuda_strm);
    CUresult status_k2 = launch_kernel(crc_hndl.get()->m_kernelNodeParams[1], cuda_strm);
    if((status_k1 != CUDA_SUCCESS) ||
       (status_k2 != CUDA_SUCCESS))
    {
        throw std::runtime_error("CRC Encode error(s)");
    }

    CUDA_CHECK(cudaStreamSynchronize(cuda_strm));

    CUDA_CHECK(cudaMemcpy(crcs.data(), d_CBCRCs.get(), totalNCodeBlocks * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMemcpy(tbCRCs.data(), d_TBCRCs.get(), nTBs * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMemcpy(codeBlocks.data(), d_codeBlocks.get(), totalOutTBByteSizes, cudaMemcpyDeviceToHost));

    tbOutBytes = 0;
    tbBytes    = 0;

    int passed = 1;

    for(int t = 0; t < nTBs; t++)
    {
        // compute TB crcs
        uint32_t tbcrc  = computeCRC<uint32_t, 24>(goldenTransportBlocks.data() + tbBytes,
                                                  codeBlockDataByteSizes[t] * nCodeBlocks[t] - 3,
                                                  G_CRC_24_A,
                                                  0,
                                                  1);
        goldenTBCRCs[t] = tbcrc;

        // Compare standalone per-TB CRC w/ per-TB CRC inserted in code blocks buffer
        // Compute pointer to per-TB CRC in the CB buffer. Subtract 6 because of 3B for per-TB CRC
        // and 3 for per-CB CRC.
        uint8_t* tmp           = codeBlocks.data() + nCodeBlocks[t] * totalCodeBlockByteSizes[t] + tbOutBytes - 6;
        uint32_t gpu_perTB_crc = 0;
        for(int byte_id = 0; byte_id < 3; byte_id++)
        {
            gpu_perTB_crc |= ((*(tmp + byte_id)) << (byte_id * 8)); // Assume CRC written least significant byte of 24bits first (lower addr)
        }

        if(tbcrc != gpu_perTB_crc)
        {
            NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT,  "TB {}: Standalone per-TB CRC {} mismatch w/ CRC inserted in CB buffer {}", t, tbcrc, gpu_perTB_crc);
            passed = 0;
        }

        // compute CB crcs
        for(int i = 0; i < nCodeBlocks[t]; i++)
        {
            uint8_t* cbPtr = codeBlocks.data() + i * totalCodeBlockByteSizes[t] + tbOutBytes;
            // TODO It'd be better not to use the buffer (pointed by cbPtr) w/ the GPU generated CRCs to compute the CPU CRCs.
            uint32_t crc = computeCRC<uint32_t, 24>((uint8_t*)cbPtr,
                                                    codeBlockDataByteSizes[t],
                                                    G_CRC_24_B,
                                                    0,
                                                    1);

            goldenCRCs[t * nCodeBlocks[t] + i] = crc;

            // Also compare the per-CB CRCs CRC written in the CB buffer w/ the standalone per-CB CRC.
            uint8_t* tmp           = cbPtr + codeBlockDataByteSizes[t];
            uint32_t gpu_perCB_crc = 0;
            for(int byte_id = 0; byte_id < 3; byte_id++)
            {
                gpu_perCB_crc |= ((*(tmp + byte_id)) << (byte_id * 8)); // Assume CRC written least significant byte of 24bits first (lower addr)
            }
            if(crc != gpu_perCB_crc)
            {
                NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT,  "TB {}, CB {}: Standalone per-CB CRC %x mismatch w/ CRC inserted in CB buffer {}", t, i, crc, gpu_perCB_crc);
                passed = 0;
            }
        }
        tbBytes += tbPaddedByteSizes[t];
        tbOutBytes += nCodeBlocks[t] * totalCodeBlockByteSizes[t];
    }

    if(status != CUPHY_STATUS_SUCCESS)
        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT,  "CRC: CUPHY ERROR");
    else
    {
        // uint32_t* gt = (uint32_t*)goldenTransportBlocks;
        passed &= equalCount(crcs.data(), goldenCRCs.data(), totalNCodeBlocks, "CB CRC ");
        // passed &= equalCount(transportBlocks, gt, totalTBPaddedByteSize / ratio, "TB DATA");

        for(int i = 0; i < nTBs; i++)
        {
            passed &= (tbCRCs[i] == goldenTBCRCs[i]);
            if(tbCRCs[i] != goldenTBCRCs[i])
                NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT,  "TB[{}] CRC: {} not equal to goldenTBCRCs: {}", i, tbCRCs[i], goldenTBCRCs[i]);
        }
    }

    return passed;
}

int CRC_PUSCH_DECODE_LOOPBACK_TEST(bool corruptEncodedCodeBlock)
{
    const uint32_t nTBs                  = 2;
    const uint32_t nCodeBlocksPerTb      = 2;
    const uint32_t totalNCodeBlocks      = nTBs * nCodeBlocksPerTb;
    const uint32_t codeBlockByteSize     = MAX_BYTES_PER_CODE_BLOCK;
    const uint32_t codeBlockDataByteSize = codeBlockByteSize - LARGE_L_BYTES;
    const uint32_t tbPayloadByteSize     = codeBlockDataByteSize * nCodeBlocksPerTb - LARGE_L_BYTES;
    const uint32_t decodedTbByteSize     = codeBlockDataByteSize * nCodeBlocksPerTb;
    const uint32_t decodedTbPaddedBytes  = decodedTbByteSize + ((sizeof(uint32_t) - (decodedTbByteSize % sizeof(uint32_t))) % sizeof(uint32_t));
    const uint32_t totalTbPaddedBytes    = nTBs * decodedTbPaddedBytes;
    const uint32_t totalCodeBlockBytes   = totalNCodeBlocks * codeBlockByteSize;

    std::vector<uint8_t>  inputCodeBlocks(totalCodeBlockBytes, 0);
    std::vector<uint8_t>  expectedTransportBlocks(totalTbPaddedBytes, 0);
    std::vector<uint8_t>  decodedTransportBlocks(totalTbPaddedBytes, 0);
    std::vector<uint32_t> decodedCBCRCs(totalNCodeBlocks, 0);
    std::vector<uint32_t> decodedTBCRCs(nTBs, 0);
    std::vector<PerTbParams> puschTbParams(nTBs);

    for(uint32_t t = 0; t < nTBs; ++t)
    {
        const uint32_t tbOffset = t * decodedTbPaddedBytes;
        for(uint32_t byteIdx = 0; byteIdx < tbPayloadByteSize; ++byteIdx)
        {
            expectedTransportBlocks[tbOffset + byteIdx] = static_cast<uint8_t>((17 * byteIdx + 31 * t + 7) & 0xff);
        }

        uint32_t tbCRC = computeCRC<uint32_t, 24>(expectedTransportBlocks.data() + tbOffset,
                                                  tbPayloadByteSize,
                                                  G_CRC_24_A,
                                                  0,
                                                  1);
        for(uint32_t byteIdx = 0; byteIdx < LARGE_L_BYTES; ++byteIdx)
        {
            expectedTransportBlocks[tbOffset + tbPayloadByteSize + byteIdx] =
                (tbCRC >> ((LARGE_L_BYTES - 1 - byteIdx) * 8)) & 0xff;
        }

        const uint32_t tbCodeBlockOffset = t * nCodeBlocksPerTb * codeBlockByteSize;
        for(uint32_t cbIdx = 0; cbIdx < nCodeBlocksPerTb; ++cbIdx)
        {
            uint8_t* codeBlock = inputCodeBlocks.data() + tbCodeBlockOffset + cbIdx * codeBlockByteSize;
            memcpy(codeBlock,
                   expectedTransportBlocks.data() + tbOffset + cbIdx * codeBlockDataByteSize,
                   codeBlockDataByteSize);

            uint32_t cbCRC = computeCRC<uint32_t, 24>(codeBlock,
                                                      codeBlockDataByteSize,
                                                      G_CRC_24_B,
                                                      0,
                                                      1);
            for(uint32_t byteIdx = 0; byteIdx < LARGE_L_BYTES; ++byteIdx)
            {
                codeBlock[codeBlockDataByteSize + byteIdx] =
                    (cbCRC >> ((LARGE_L_BYTES - 1 - byteIdx) * 8)) & 0xff;
            }
        }

        puschTbParams[t].num_CBs             = nCodeBlocksPerTb;
        puschTbParams[t].K                   = codeBlockByteSize * 8;
        puschTbParams[t].F                   = 0;
        puschTbParams[t].firstCodeBlockIndex = 0;
        puschTbParams[t].nDataBytes          = decodedTbByteSize;
    }

    if(corruptEncodedCodeBlock)
    {
        inputCodeBlocks[0] ^= 0x1;
    }

    cuphyStatus_t status = runPuschCrcDecode(decodedCBCRCs.data(),
                                             decodedTBCRCs.data(),
                                             decodedTransportBlocks.data(),
                                             inputCodeBlocks.data(),
                                             puschTbParams.data(),
                                             nTBs,
                                             totalCodeBlockBytes,
                                             totalNCodeBlocks,
                                             totalTbPaddedBytes,
                                             0);
    if(status != CUPHY_STATUS_SUCCESS)
    {
        return 0;
    }

    const bool cbCrcFailure = std::any_of(decodedCBCRCs.begin(), decodedCBCRCs.end(), [](uint32_t crc) { return crc != 0; });
    const bool tbCrcFailure = std::any_of(decodedTBCRCs.begin(), decodedTBCRCs.end(), [](uint32_t crc) { return crc != 0; });

    if(corruptEncodedCodeBlock)
    {
        return (cbCrcFailure && tbCrcFailure) ? 1 : 0;
    }

    if(cbCrcFailure || tbCrcFailure)
    {
        return 0;
    }

    return equalCount(decodedTransportBlocks.data(),
                      expectedTransportBlocks.data(),
                      totalTbPaddedBytes,
                      "PUSCH LOOPBACK TB DATA ") ? 1 : 0;
}

int CRC_SINGLE_CB_GPU_UPLINK_PUSCH_TEST(bool timeIt)
{
    (void)timeIt;
    const uint32_t nTBs        = MAX_N_TBS_SUPPORTED;
    const uint32_t crcByteSize = 3; // 24 bits
    std::vector<uint32_t> nCodeBlocks(nTBs);
    std::vector<uint32_t> codeBlockByteSizes(nTBs);
    std::vector<uint32_t> codeBlockDataByteSizes(nTBs);
    std::vector<uint32_t> CBPaddingByteSizes(nTBs);
    std::vector<uint32_t> totalCodeBlockByteSizes(nTBs);
    std::vector<uint32_t> tbPaddedByteSizes(nTBs);
    uint32_t totalByteSize         = 0;
    uint32_t totalNCodeBlocks      = 0;
    uint32_t totalTBPaddedByteSize = 0;
    // Same CRC value for each code block, code blocks are all the same
    // linear input layout : cb1|crc1, cb2|crc2, ...

    for(int i = 0; i < nTBs; i++)
    {
        if(i == 2)
        {
            codeBlockByteSizes[i] = 1056;
            nCodeBlocks[i]        = 1;
        }
        else
        {
            codeBlockByteSizes[i] = 1000;
            nCodeBlocks[i]        = 1;
        }
        totalNCodeBlocks += nCodeBlocks[i];
        codeBlockDataByteSizes[i]  = codeBlockByteSizes[i] - crcByteSize;
        CBPaddingByteSizes[i]      = (MAX_BYTES_PER_CODE_BLOCK - (codeBlockByteSizes[i] % MAX_BYTES_PER_CODE_BLOCK)) % MAX_BYTES_PER_CODE_BLOCK;
        totalCodeBlockByteSizes[i] = codeBlockByteSizes[i] + CBPaddingByteSizes[i];
        totalByteSize += totalCodeBlockByteSizes[i] * nCodeBlocks[i];
        tbPaddedByteSizes[i] = (codeBlockDataByteSizes[i] + (nCodeBlocks[i] == 1 ? crcByteSize : 0)) * nCodeBlocks[i] +
                               (4 - (nCodeBlocks[i] * (codeBlockDataByteSizes[i] + (nCodeBlocks[i] == 1 ? crcByteSize : 0)) % 4)) % 4;
        totalTBPaddedByteSize += tbPaddedByteSizes[i];
    }

    std::vector<PerTbParams> tbPrmsArray(nTBs);
    std::vector<uint8_t>     linearInput(totalByteSize, 0);
    std::vector<uint32_t>    goldenCRCs(totalNCodeBlocks, 0);
    std::vector<uint8_t>     goldenTransportBlocks(totalTBPaddedByteSize, 0);
    std::vector<uint8_t>     transportBlocks(totalTBPaddedByteSize, 0);
    uint8_t*                 codeBlocks = linearInput.data();
    std::vector<uint32_t>    crcs(totalNCodeBlocks, 0);
    std::vector<uint32_t>    tbCRCs(nTBs, 0);
    memset(tbPrmsArray.data(), 0, sizeof(PerTbParams) * tbPrmsArray.size());
    uint32_t tbBytes      = 0;
    uint32_t totalCBBytes = 0;
    for(int t = 0; t < nTBs; t++)
    {
        // Build transport block
        uint32_t cbBytes = 0;
        for(int i = 0; i < nCodeBlocks[t]; i++)
        {
            memset(goldenTransportBlocks.data() + tbBytes + cbBytes,
                   rand(),
                   codeBlockDataByteSizes[t]);
            cbBytes += codeBlockDataByteSizes[t];
        }

        // just compute CB crcs using TB polynomial, as TBs contain only one CB
        for(int i = 0; i < nCodeBlocks[t]; i++)
        {
            uint8_t* cbPtr  = linearInput.data() + i * totalCodeBlockByteSizes[t] + totalCBBytes;
            uint8_t* crcPtr = (cbPtr + codeBlockDataByteSizes[t]);
            memcpy(cbPtr,
                   goldenTransportBlocks.data() + i * codeBlockDataByteSizes[t] + tbBytes,
                   codeBlockDataByteSizes[t]);
            uint32_t crc = computeCRC<uint32_t, 24>((uint8_t*)cbPtr,
                                                    codeBlockDataByteSizes[t],
                                                    G_CRC_24_A,
                                                    0,
                                                    1);
            for(int j = 0; j < crcByteSize; j++)
                crcPtr[j] = (crc >> (crcByteSize - 1 - j) * 8) & 0xFF;
            for(int j = 0; j < crcByteSize; j++)
                goldenTransportBlocks[tbBytes + nCodeBlocks[t] * codeBlockDataByteSizes[t] /*- crcByteSizes[t]*/ + j] =
                    (crc >> (crcByteSize - 1 - j) * 8) & 0xFF;

            goldenCRCs[t * nCodeBlocks[t] + i] = 0;
            memset(cbPtr + codeBlockByteSizes[t], 0, CBPaddingByteSizes[t]);
        }
        tbPrmsArray[t].num_CBs             = nCodeBlocks[t];
        tbPrmsArray[t].K                   = codeBlockByteSizes[t] * 8;
        tbPrmsArray[t].F                   = 0;
        tbPrmsArray[t].firstCodeBlockIndex = 0;
        tbPrmsArray[t].nDataBytes          = tbPaddedByteSizes[t];
        tbBytes += tbPaddedByteSizes[t];
        totalCBBytes += nCodeBlocks[t] * totalCodeBlockByteSizes[t];
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
    cuphyStatus_t status = runPuschCrcDecode(crcs.data(),
                                             tbCRCs.data(),
                                             transportBlocks.data(),
                                             codeBlocks,
                                             tbPrmsArray.data(),
                                             nTBs,
                                             totalByteSize,
                                             totalNCodeBlocks,
                                             totalTBPaddedByteSize,
                                             0);

    int passed = 0;
    if(status != CUPHY_STATUS_SUCCESS)
        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT,  "CRC: CUPHY ERROR");
    else
    {
        passed = equalCount(crcs.data(), goldenCRCs.data(), totalNCodeBlocks, "CB CRC ");
        passed &= equalCount(transportBlocks.data(), goldenTransportBlocks.data(), totalTBPaddedByteSize, "TB DATA ");

        for(int i = 0; i < nTBs; i++)
        {
            passed &= (tbCRCs[i] == 0);
            if(tbCRCs[i] != 0)
                NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT,  "TB[{}] CRC not equal to 0: {}", i, tbCRCs[i]);
        }
    }

    return passed;
}

int CRC_SINGLE_SMALL_CB_GPU_UPLINK_PUSCH_TEST(bool timeIt)
{
    (void)timeIt;
    const uint32_t nTBs        = MAX_N_TBS_SUPPORTED;
    const uint32_t crcByteSize = 2;
    std::vector<uint32_t> nCodeBlocks(nTBs);
    std::vector<uint32_t> codeBlockByteSizes(nTBs);
    std::vector<uint32_t> codeBlockDataByteSizes(nTBs);
    std::vector<uint32_t> CBPaddingByteSizes(nTBs);
    std::vector<uint32_t> totalCodeBlockByteSizes(nTBs);
    std::vector<uint32_t> tbPaddedByteSizes(nTBs);
    uint32_t totalByteSize         = 0;
    uint32_t totalNCodeBlocks      = 0;
    uint32_t totalTBPaddedByteSize = 0;
    // Same CRC value for each code block, code blocks are all the same
    // linear input layout : cb1|crc1, cb2|crc2, ...

    for(int i = 0; i < nTBs; i++)
    {
        if(i == 2)
        {
            codeBlockByteSizes[i] = 333;
            nCodeBlocks[i]        = 1;
        }
        else
        {
            codeBlockByteSizes[i] = 476;
            nCodeBlocks[i]        = 1;
        }
        totalNCodeBlocks += nCodeBlocks[i];
        codeBlockDataByteSizes[i]  = codeBlockByteSizes[i] - crcByteSize;
        CBPaddingByteSizes[i]      = (MAX_BYTES_PER_CODE_BLOCK - (codeBlockByteSizes[i] % MAX_BYTES_PER_CODE_BLOCK)) % MAX_BYTES_PER_CODE_BLOCK;
        totalCodeBlockByteSizes[i] = codeBlockByteSizes[i] + CBPaddingByteSizes[i];
        totalByteSize += totalCodeBlockByteSizes[i] * nCodeBlocks[i];
        tbPaddedByteSizes[i] = (codeBlockDataByteSizes[i] + (nCodeBlocks[i] == 1 ? crcByteSize : 0)) * nCodeBlocks[i] +
                               (4 - (nCodeBlocks[i] * (codeBlockDataByteSizes[i] + (nCodeBlocks[i] == 1 ? crcByteSize : 0)) % 4)) % 4;
        totalTBPaddedByteSize += tbPaddedByteSizes[i];
    }

    std::vector<PerTbParams> tbPrmsArray(nTBs);
    std::vector<uint8_t>     linearInput(totalByteSize, 0);
    std::vector<uint32_t>    goldenCRCs(totalNCodeBlocks, 0);
    std::vector<uint8_t>     goldenTransportBlocks(totalTBPaddedByteSize, 0);
    std::vector<uint8_t>     transportBlocks(totalTBPaddedByteSize, 0);
    uint8_t*                 codeBlocks = linearInput.data();
    std::vector<uint32_t>    crcs(totalNCodeBlocks, 0);
    std::vector<uint32_t>    tbCRCs(nTBs, 0);
    memset(tbPrmsArray.data(), 0, sizeof(PerTbParams) * tbPrmsArray.size());

    uint32_t tbBytes      = 0;
    uint32_t totalCBBytes = 0;
    for(int t = 0; t < nTBs; t++)
    {
        // Build transport block
        uint32_t cbBytes = 0;
        for(int i = 0; i < nCodeBlocks[t]; i++)
        {
            memset(goldenTransportBlocks.data() + tbBytes + cbBytes,
                   rand(),
                   codeBlockDataByteSizes[t]);
            cbBytes += codeBlockDataByteSizes[t];
        }

        // just compute CB crcs using TB polynomial, as TBs contain only one CB
        for(int i = 0; i < nCodeBlocks[t]; i++)
        {
            uint8_t* cbPtr  = linearInput.data() + i * totalCodeBlockByteSizes[t] + totalCBBytes;
            uint8_t* crcPtr = (cbPtr + codeBlockDataByteSizes[t]);
            memcpy(cbPtr,
                   goldenTransportBlocks.data() + i * codeBlockDataByteSizes[t] + tbBytes,
                   codeBlockDataByteSizes[t]);
            uint32_t crc = computeCRC<uint32_t, 16>((uint8_t*)cbPtr,
                                                    codeBlockDataByteSizes[t],
                                                    G_CRC_16,
                                                    0,
                                                    1);
            for(int j = 0; j < crcByteSize; j++)
                crcPtr[j] = (crc >> (crcByteSize - 1 - j) * 8) & 0xFF;
            for(int j = 0; j < crcByteSize; j++)
                goldenTransportBlocks[tbBytes + nCodeBlocks[t] * codeBlockDataByteSizes[t] /*- crcByteSizes[t]*/ + j] =
                    (crc >> (crcByteSize - 1 - j) * 8) & 0xFF;
            goldenCRCs[t * nCodeBlocks[t] + i] = 0;
            memset(cbPtr + codeBlockByteSizes[t], 0, CBPaddingByteSizes[t]);
        }

        tbPrmsArray[t].num_CBs             = nCodeBlocks[t];
        tbPrmsArray[t].K                   = codeBlockByteSizes[t] * 8;
        tbPrmsArray[t].F                   = 0;
        tbPrmsArray[t].firstCodeBlockIndex = 0;
        tbPrmsArray[t].nDataBytes          = tbPaddedByteSizes[t];
        tbBytes += tbPaddedByteSizes[t];
        totalCBBytes += nCodeBlocks[t] * totalCodeBlockByteSizes[t];
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
    cuphyStatus_t status = runPuschCrcDecode(crcs.data(),
                                             tbCRCs.data(),
                                             transportBlocks.data(),
                                             codeBlocks,
                                             tbPrmsArray.data(),
                                             nTBs,
                                             totalByteSize,
                                             totalNCodeBlocks,
                                             totalTBPaddedByteSize,
                                             0);

    int passed = 0;
    if(status != CUPHY_STATUS_SUCCESS)
        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT,  "CRC: CUPHY ERROR");
    else
    {
        passed = equalCount(crcs.data(), goldenCRCs.data(), totalNCodeBlocks, "CB CRC ");
        passed &= equalCount(transportBlocks.data(), goldenTransportBlocks.data(), totalTBPaddedByteSize, "TB DATA ");

        for(int i = 0; i < nTBs; i++)
        {
            passed &= (tbCRCs[i] == 0);
            if(tbCRCs[i] != 0)
                NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT,  "TB[{}] CRC not equal to 0: {}", i, tbCRCs[i]);
        }
    }

    return passed;
}

int TestPuschRxCrcDecodeSetup()
{
    PuschCrcDecodeHandle decoder(1);
    if(!decoder.valid())
    {
        return 0;
    }

    try {
        const uint16_t nSchUes = 2;
        uint16_t       schUserIdxs[2] = {1, 0};

        cuphy::unique_device_ptr<uint32_t> d_outputCBCRCs = cuphy::make_unique_device<uint32_t>(nSchUes * 3);
        cuphy::unique_device_ptr<uint8_t>  d_outputTBs = cuphy::make_unique_device<uint8_t>(nSchUes * 2048);
        cuphy::unique_device_ptr<uint32_t> d_inputCodeBlocks = cuphy::make_unique_device<uint32_t>(nSchUes * 3 * 512);
        cuphy::unique_device_ptr<uint32_t> d_outputTBCRCs = cuphy::make_unique_device<uint32_t>(nSchUes);

        PerTbParams tbParams[2] = {};
        tbParams[0].num_CBs = 2;
        tbParams[0].K = 1000 * 8;
        tbParams[0].F = 0;
        tbParams[0].nDataBytes = 1000;
        tbParams[0].firstCodeBlockIndex = 0;

        tbParams[1].num_CBs = 3;
        tbParams[1].K = 1500 * 8;
        tbParams[1].F = 0;
        tbParams[1].nDataBytes = 1500;
        tbParams[1].firstCodeBlockIndex = 2;

        cuphy::unique_device_ptr<PerTbParams> d_tbParams = cuphy::make_unique_device<PerTbParams>(nSchUes);
        CUDA_CHECK(cudaMemcpy(d_tbParams.get(), tbParams, sizeof(PerTbParams) * nSchUes, cudaMemcpyHostToDevice));

        size_t descrSize = 0, descrAlign = 0;
        cuphyStatus_t status = cuphyPuschRxCrcDecodeGetDescrInfo(&descrSize, &descrAlign);
        if(status != CUPHY_STATUS_SUCCESS)
        {
            return 0;
        }

        cuphy::unique_device_ptr<uint8_t> d_desc = cuphy::make_unique_device<uint8_t>(descrSize);
        cuphy::unique_pinned_ptr<uint8_t> h_desc = cuphy::make_unique_pinned<uint8_t>(descrSize);

        cuphyPuschRxCrcDecodeLaunchCfg_t cbCrcLaunchCfg = {};
        cuphyPuschRxCrcDecodeLaunchCfg_t tbCrcLaunchCfg = {};

        status = cuphySetupPuschRxCrcDecode(
            decoder.get(),
            nSchUes,
            schUserIdxs,
            d_outputCBCRCs.get(),
            d_outputTBs.get(),
            d_inputCodeBlocks.get(),
            d_outputTBCRCs.get(),
            tbParams,
            d_tbParams.get(),
            h_desc.get(),
            d_desc.get(),
            1,
            &cbCrcLaunchCfg,
            &tbCrcLaunchCfg,
            0);

        const puschRxCrcDecodeDescr_t* desc = reinterpret_cast<const puschRxCrcDecodeDescr_t*>(h_desc.get());
        const bool passed = status == CUPHY_STATUS_SUCCESS &&
                            desc->pOutputCBCRCs == d_outputCBCRCs.get() &&
                            desc->pOutputTBs == d_outputTBs.get() &&
                            desc->pInputCodeBlocks == d_inputCodeBlocks.get() &&
                            desc->pOutputTBCRCs == d_outputTBCRCs.get() &&
                            desc->pTbPrmsArray == d_tbParams.get() &&
                            desc->reverseBytes &&
                            desc->schUserIdxs[0] == schUserIdxs[0] &&
                            desc->schUserIdxs[1] == schUserIdxs[1] &&
                            cbCrcLaunchCfg.kernelNodeParamsDriver.gridDimX == 3 &&
                            cbCrcLaunchCfg.kernelNodeParamsDriver.gridDimY == nSchUes &&
                            tbCrcLaunchCfg.kernelNodeParamsDriver.gridDimY == nSchUes;

        return passed ? 1 : 0;
    }
    catch (const std::exception& e) {
        return 0;
    }
}

int TestPrepareCRCEncodeSetup()
{
    // Test parameters
    uint32_t nTBs = 2;
    uint32_t maxNCBsPerTB = 6;
    uint32_t maxTbSizeBytes = 1024; // 1KB per TB

    // Get descriptor size
    size_t descrSize, descrAlign;
    cuphyStatus_t status = cuphyPrepareCrcEncodeGetDescrInfo(&descrSize, &descrAlign);
    if(status != CUPHY_STATUS_SUCCESS)
    {
        return 0;
    }

    // Allocate descriptors
    cuphy::unique_device_ptr<uint8_t> d_desc = cuphy::make_unique_device<uint8_t>(descrSize);
    cuphy::unique_pinned_ptr<uint8_t> h_desc = cuphy::make_unique_pinned<uint8_t>(descrSize);

    // Create device memory for input/output
    cuphy::unique_device_ptr<uint32_t> d_inputTBs = cuphy::make_unique_device<uint32_t>(nTBs * maxTbSizeBytes / sizeof(uint32_t));
    cuphy::unique_device_ptr<uint32_t> d_inputTBsTM = cuphy::make_unique_device<uint32_t>(nTBs * maxTbSizeBytes / sizeof(uint32_t));

    // Create TB parameters
    PdschPerTbParams tbParams[2];
    tbParams[0].num_CBs = 2;
    tbParams[0].K = 1000 * 8; // 1000 bytes * 8 bits
    tbParams[0].F = 0;
    tbParams[0].firstCodeBlockIndex = 0;
    tbParams[0].tbSize = 1000;
    tbParams[0].tbStartOffset = 0;
    tbParams[0].tbStartAddr = nullptr;
    tbParams[0].cumulativeTbSizePadding = 1024;
    tbParams[0].testModel = 0;

    tbParams[1].num_CBs = 3;
    tbParams[1].K = 1500 * 8; // 1500 bytes * 8 bits
    tbParams[1].F = 0;
    tbParams[1].firstCodeBlockIndex = 2;
    tbParams[1].tbSize = 1500;
    tbParams[1].tbStartOffset = 1024;
    tbParams[1].tbStartAddr = nullptr;
    tbParams[1].cumulativeTbSizePadding = 2048;
    tbParams[1].testModel = 0;

    cuphy::unique_device_ptr<PdschPerTbParams> d_tbParams = cuphy::make_unique_device<PdschPerTbParams>(nTBs);
    cudaMemcpy(d_tbParams.get(), tbParams, sizeof(PdschPerTbParams) * nTBs, cudaMemcpyHostToDevice);

    // Create launch config
    cuphyPrepareCrcEncodeLaunchConfig prepareCrcEncodeLaunchCfg = {};

    // Test setup function
    try {
        status = cuphySetupPrepareCRCEncode(
            &prepareCrcEncodeLaunchCfg,
            nullptr,
            d_inputTBs.get(),
            d_inputTBsTM.get(),
            d_tbParams.get(),
            nTBs,
            maxNCBsPerTB,
            maxTbSizeBytes,
            h_desc.get(),
            d_desc.get(),
            1, // enable_desc_async_copy
            0  // stream
        );
        const prepareCrcEncodeDescr_t* desc = reinterpret_cast<const prepareCrcEncodeDescr_t*>(h_desc.get());
        const bool passed = status == CUPHY_STATUS_SUCCESS &&
                            desc->d_inputOrigTBs == nullptr &&
                            desc->d_inputTBs == d_inputTBs.get() &&
                            desc->d_inputTBsTM == d_inputTBsTM.get() &&
                            desc->d_tbPrmsArray == d_tbParams.get() &&
                            prepareCrcEncodeLaunchCfg.m_kernelNodeParams.gridDimY == nTBs;
        return passed ? 1 : 0;
    }
    catch (const std::exception& e) {
        return 0;
    }
}

int TestCrcEncodeSetupMatchesPdschTxUsage()
{
    const uint32_t nTBs = 2;
    const uint32_t maxNCBsPerTB = 3;
    const uint32_t maxTbSizeBytes = 2048;

    size_t descrSize = 0, descrAlign = 0;
    cuphyStatus_t status = cuphyCrcEncodeGetDescrInfo(&descrSize, &descrAlign);
    if(status != CUPHY_STATUS_SUCCESS)
    {
        return 0;
    }

    try {
        cuphy::unique_device_ptr<uint8_t>  d_desc = cuphy::make_unique_device<uint8_t>(descrSize);
        cuphy::unique_device_ptr<uint32_t> d_tbCRCs = cuphy::make_unique_device<uint32_t>(nTBs);
        cuphy::unique_device_ptr<uint32_t> d_inputTBs = cuphy::make_unique_device<uint32_t>((nTBs * maxTbSizeBytes) / sizeof(uint32_t));
        cuphy::unique_device_ptr<uint8_t>  d_codeBlocks = cuphy::make_unique_device<uint8_t>(nTBs * maxTbSizeBytes);

        PdschPerTbParams tbParams[2] = {};
        tbParams[0].num_CBs = 2;
        tbParams[0].K = 1056 * 8;
        tbParams[0].F = 0;
        tbParams[0].tbSize = 2103;
        tbParams[0].cumulativeTbSizePadding = maxTbSizeBytes;

        tbParams[1].num_CBs = 3;
        tbParams[1].K = 1056 * 8;
        tbParams[1].F = 0;
        tbParams[1].tbSize = 3156;
        tbParams[1].cumulativeTbSizePadding = 2 * maxTbSizeBytes;

        cuphy::unique_device_ptr<PdschPerTbParams> d_tbParams = cuphy::make_unique_device<PdschPerTbParams>(nTBs);
        CUDA_CHECK(cudaMemcpy(d_tbParams.get(), tbParams, sizeof(PdschPerTbParams) * nTBs, cudaMemcpyHostToDevice));
        cuphy::unique_pinned_ptr<uint8_t> h_desc = cuphy::make_unique_pinned<uint8_t>(descrSize);

        cuphyCrcEncodeLaunchConfig crcEncodeLaunchCfg = {};
        status = cuphySetupCrcEncode(
            &crcEncodeLaunchCfg,
            nullptr,
            d_tbCRCs.get(),
            d_inputTBs.get(),
            d_codeBlocks.get(),
            d_tbParams.get(),
            nTBs,
            maxNCBsPerTB,
            maxTbSizeBytes,
            1,
            1,
            h_desc.get(),
            d_desc.get(),
            1,
            0);

        const crcEncodeDescr_t* desc = reinterpret_cast<const crcEncodeDescr_t*>(h_desc.get());
        const bool passed = status == CUPHY_STATUS_SUCCESS &&
                            desc->d_cbCRCs == nullptr &&
                            desc->d_tbCRCs == d_tbCRCs.get() &&
                            desc->d_inputTransportBlocks == d_inputTBs.get() &&
                            desc->d_codeBlocks == d_codeBlocks.get() &&
                            desc->d_tbPrmsArray == d_tbParams.get() &&
                            desc->reverseBytes &&
                            crcEncodeLaunchCfg.m_kernelNodeParams[0].gridDimY == nTBs &&
                            crcEncodeLaunchCfg.m_kernelNodeParams[1].gridDimX == maxNCBsPerTB &&
                            crcEncodeLaunchCfg.m_kernelNodeParams[1].gridDimY == nTBs;

        return passed ? 1 : 0;
    }
    catch (const std::exception& e) {
        return 0;
    }
}

// Helper function to create minimal TB parameters
PerTbParams createMinimalTbParams(uint32_t num_CBs = 1) {
    PerTbParams params;
    memset(&params, 0, sizeof(params));
    params.num_CBs = num_CBs;
    params.K = 8;
    params.F = 0;
    params.firstCodeBlockIndex = 0;
    params.nDataBytes = 1;
    return params;
}

// Helper function to test production PUSCH CRC setup boundary conditions
bool testBoundaryCondition(
    uint16_t nSchUes,
    uint32_t numCBs,
    uint32_t nDataBytes,
    const char* errorMessage) {

    try {
        PuschCrcDecodeHandle decoder(0);
        if(!decoder.valid())
        {
            return false;
        }

        size_t descrSize = 0, descrAlign = 0;
        cuphyStatus_t status = cuphyPuschRxCrcDecodeGetDescrInfo(&descrSize, &descrAlign);
        if(status != CUPHY_STATUS_SUCCESS)
        {
            return false;
        }

        const uint32_t tbParamCount = (nSchUes > MAX_N_TBS_PER_CELL_GROUP_SUPPORTED) ? 1 : nSchUes;
        std::vector<uint16_t> schUserIdxs(std::max<uint32_t>(nSchUes, 1));
        std::vector<PerTbParams> tbParams(std::max<uint32_t>(tbParamCount, 1));
        for(uint32_t i = 0; i < schUserIdxs.size(); ++i)
        {
            schUserIdxs[i] = static_cast<uint16_t>(i < tbParams.size() ? i : 0);
        }
        for(uint32_t i = 0; i < tbParams.size(); ++i)
        {
            tbParams[i] = createMinimalTbParams(numCBs);
            tbParams[i].nDataBytes = nDataBytes;
        }

        cuphy::unique_device_ptr<uint32_t> d_CBCRCs = cuphy::make_unique_device<uint32_t>(1);
        cuphy::unique_device_ptr<uint32_t> d_TBCRCs = cuphy::make_unique_device<uint32_t>(1);
        cuphy::unique_device_ptr<uint8_t> d_TBs = cuphy::make_unique_device<uint8_t>(1);
        cuphy::unique_device_ptr<uint32_t> d_codeBlocks = cuphy::make_unique_device<uint32_t>(1);
        cuphy::unique_device_ptr<PerTbParams> d_tbParams = cuphy::make_unique_device<PerTbParams>(tbParams.size());
        cuphy::unique_device_ptr<uint8_t> d_desc = cuphy::make_unique_device<uint8_t>(descrSize);
        cuphy::unique_pinned_ptr<uint8_t> h_desc = cuphy::make_unique_pinned<uint8_t>(descrSize);
        CUDA_CHECK(cudaMemcpy(d_tbParams.get(), tbParams.data(), sizeof(PerTbParams) * tbParams.size(), cudaMemcpyHostToDevice));

        cuphyPuschRxCrcDecodeLaunchCfg_t cbCrcLaunchCfg = {};
        cuphyPuschRxCrcDecodeLaunchCfg_t tbCrcLaunchCfg = {};
        status = cuphySetupPuschRxCrcDecode(decoder.get(),
                                            nSchUes,
                                            schUserIdxs.data(),
                                            d_CBCRCs.get(),
                                            d_TBs.get(),
                                            d_codeBlocks.get(),
                                            d_TBCRCs.get(),
                                            tbParams.data(),
                                            d_tbParams.get(),
                                            h_desc.get(),
                                            d_desc.get(),
                                            1,
                                            &cbCrcLaunchCfg,
                                            &tbCrcLaunchCfg,
                                            0);

        if (status != CUPHY_STATUS_NOT_SUPPORTED) {
            printf("%s\n", errorMessage);
            return false;
        }
    }
    catch (const std::exception& e) {
        return false;
    }

    return true;
}

// Helper function to test setup boundary condition for cuphySetupCrcEncode
bool testSetupCrcEncodeBoundaryCondition(
    uint32_t nTBs,
    uint32_t maxNCBsPerTB,
    uint32_t maxTbSizeBytes,
    uint32_t* d_tbCRCs,
    uint32_t* d_inputTransportBlocks,
    uint8_t* d_codeBlocks,
    PdschPerTbParams* d_tbPrmsArray,
    uint8_t* cpu_desc,
    uint8_t* gpu_desc,
    const char* errorMessage) {

    // Get descriptor size
    size_t descrSize, descrAlign;
    cuphyStatus_t status = cuphyCrcEncodeGetDescrInfo(&descrSize, &descrAlign);
    if(status != CUPHY_STATUS_SUCCESS)
    {
        return false;
    }

    // Create device memory for input/output
    cuphy::unique_device_ptr<uint32_t> d_CBCRCs = cuphy::make_unique_device<uint32_t>(nTBs * maxNCBsPerTB);

    // Create launch config
    std::unique_ptr<cuphyCrcEncodeLaunchConfig> crc_hndl = std::make_unique<cuphyCrcEncodeLaunchConfig>();

    try {
        status = cuphySetupCrcEncode(
            crc_hndl.get(),
            d_CBCRCs.get(),
            d_tbCRCs,
            d_inputTransportBlocks,
            d_codeBlocks,
            d_tbPrmsArray,
            nTBs,
            maxNCBsPerTB,
            maxTbSizeBytes,
            0,
            false,
            cpu_desc,
            gpu_desc,
            1, // enable_desc_async_copy
            0  // stream
        );

        if (d_tbCRCs == nullptr || d_inputTransportBlocks == nullptr || d_codeBlocks == nullptr ||
            d_tbPrmsArray == nullptr || cpu_desc == nullptr || gpu_desc == nullptr) {
            // For null pointer checks, expect INVALID_ARGUMENT
            if (status != CUPHY_STATUS_INVALID_ARGUMENT) {
                printf("%s\n", errorMessage);
                return false;
            }
        } else {
            // For boundary condition checks, expect NOT_SUPPORTED
            if (status != CUPHY_STATUS_NOT_SUPPORTED) {
                printf("%s\n", errorMessage);
                return false;
            }
        }
        return true;
    }
    catch (const std::exception& e) {
        return false;
    }
}

TEST(CRC_GPU_UPLINK_PUSCH, 24B_24A)
{
    EXPECT_EQ(CRC_GPU_UPLINK_PUSCH_TEST(false), 1);
}

TEST(CRC_SINGLE_CB_GPU_UPLINK_PUSCH, 24A)
{
    EXPECT_EQ(CRC_SINGLE_CB_GPU_UPLINK_PUSCH_TEST(false), 1);
}

TEST(CRC_SINGLE_SMALL_CB_GPU_UPLINK_PUSCH, 16)
{
    EXPECT_EQ(CRC_SINGLE_SMALL_CB_GPU_UPLINK_PUSCH_TEST(false), 1);
}

//int cb_num_array[20] = {51, 44, 26, 18, 10, 383, 330, 195, 135, 75, 501, 429, 251, 174, 91, 752, 644, 377, 261, 137};
TEST(CRC_GPU_DOWNLINK_PDSCH, 24B_24A)
{
    /*for (int i = 0; i < 20; i++){
        EXPECT_EQ(CRC_GPU_DOWNLINK_PDSCH(false, cb_num_array[i]), 1); // FIXME the 2nd argument is not used.
    }*/
    EXPECT_EQ(CRC_GPU_DOWNLINK_PDSCH(false), 1);
}

TEST(CRCTest, PuschDecodeLoopbackPasses)
{
    EXPECT_EQ(CRC_PUSCH_DECODE_LOOPBACK_TEST(false), 1) << "PUSCH CRC decode loopback failed";
}

TEST(CRCTest, PuschDecodeLoopbackDetectsCorruption)
{
    EXPECT_EQ(CRC_PUSCH_DECODE_LOOPBACK_TEST(true), 1) << "PUSCH CRC decode did not detect corrupted encoded CRC input";
}

TEST(CRCTest, PuschSetupBoundaryCondition)
{
    int result = 0;
    // Test each boundary condition separately
    // 1. Test nTBs > MAX_N_TBS_PER_CELL_GROUP_SUPPORTED
    result = testBoundaryCondition(
        MAX_N_TBS_PER_CELL_GROUP_SUPPORTED + 1,  // Exceed max TBs
        2,  // Small value for numCBs
        1024,  // Small value for maxTbSizeBytes
        "nTBs > MAX_N_TBS_PER_CELL_GROUP_SUPPORTED check failed"
    ) ? 1 : 0;
    EXPECT_EQ(result, 1) << "nTBs boundary condition test failed";

    // 2. Test num_CBs > MAX_N_CBS_PER_TB_SUPPORTED
    result = testBoundaryCondition(
        2,  // Small value for nTBs
        MAX_N_CBS_PER_TB_SUPPORTED + 1,  // Exceed max CBs per TB
        1024,  // Small value for maxTbSizeBytes
        "num_CBs > MAX_N_CBS_PER_TB_SUPPORTED check failed"
    ) ? 1 : 0;
    EXPECT_EQ(result, 1) << "num_CBs boundary condition test failed";

    // 3. Test maxTBByteSize > MAX_BYTES_PER_TRANSPORT_BLOCK
    result = testBoundaryCondition(
        2,  // Small value for nTBs
        2,  // Small value for numCBs
        MAX_BYTES_PER_TRANSPORT_BLOCK + 1,  // Exceed max TB size
        "maxTBByteSize > MAX_BYTES_PER_TRANSPORT_BLOCK check failed"
    ) ? 1 : 0;
    EXPECT_EQ(result, 1) << "maxTBByteSize boundary condition test failed";
}

TEST(PuschRxCrcDecodeTest, SetupFunction)
{
    EXPECT_EQ(TestPuschRxCrcDecodeSetup(), 1) << "Setup function test failed";
}

TEST(CRCTest, PrepareCRCEncodeSetup)
{
    EXPECT_EQ(TestPrepareCRCEncodeSetup(), 1) << "Prepare CRC encode setup test failed";
}

TEST(CRCTest, CrcEncodeSetupMatchesPdschTxUsage)
{
    EXPECT_EQ(TestCrcEncodeSetupMatchesPdschTxUsage(), 1) << "CRC encode setup test failed";
}

TEST(CRCTest, SetupCrcEncodeBoundaryConditions)
{
    bool result = 0;

    // Test null pointer check for d_tbCRCs
    result = testSetupCrcEncodeBoundaryCondition(
        2,  // nTBs
        6,  // maxNCBsPerTB
        1024,  // maxTbSizeBytes
        nullptr,  // d_tbCRCs is nullptr
        nullptr,  // d_inputTransportBlocks
        nullptr,  // d_codeBlocks
        nullptr,  // d_tbPrmsArray
        nullptr,  // cpu_desc
        nullptr,  // gpu_desc
        "d_tbCRCs nullptr check failed in cuphySetupCrcEncode"
    );
    EXPECT_TRUE(result) << "d_tbCRCs nullptr check test failed for cuphySetupCrcEncode";

    // Test null pointer check for d_inputTransportBlocks
    result = testSetupCrcEncodeBoundaryCondition(
        2,  // nTBs
        6,  // maxNCBsPerTB
        1024,  // maxTbSizeBytes
        (uint32_t*)0x1,  // d_tbCRCs is non-nullptr
        nullptr,  // d_inputTransportBlocks is nullptr
        nullptr,  // d_codeBlocks
        nullptr,  // d_tbPrmsArray
        nullptr,  // cpu_desc
        nullptr,  // gpu_desc
        "d_inputTransportBlocks nullptr check failed in cuphySetupCrcEncode"
    );
    EXPECT_TRUE(result) << "d_inputTransportBlocks nullptr check test failed for cuphySetupCrcEncode";

    // Test null pointer check for d_codeBlocks
    result = testSetupCrcEncodeBoundaryCondition(
        2,  // nTBs
        6,  // maxNCBsPerTB
        1024,  // maxTbSizeBytes
        (uint32_t*)0x1,  // d_tbCRCs is non-nullptr
        (uint32_t*)0x1,  // d_inputTransportBlocks is non-nullptr
        nullptr,  // d_codeBlocks is nullptr
        nullptr,  // d_tbPrmsArray
        nullptr,  // cpu_desc
        nullptr,  // gpu_desc
        "d_codeBlocks nullptr check failed in cuphySetupCrcEncode"
    );
    EXPECT_TRUE(result) << "d_codeBlocks nullptr check test failed for cuphySetupCrcEncode";

    // Test null pointer check for d_tbPrmsArray
    result = testSetupCrcEncodeBoundaryCondition(
        2,  // nTBs
        6,  // maxNCBsPerTB
        1024,  // maxTbSizeBytes
        (uint32_t*)0x1,  // d_tbCRCs is non-nullptr
        (uint32_t*)0x1,  // d_inputTransportBlocks is non-nullptr
        (uint8_t*)0x1,   // d_codeBlocks is non-nullptr
        nullptr,  // d_tbPrmsArray is nullptr
        nullptr,  // cpu_desc
        nullptr,  // gpu_desc
        "d_tbPrmsArray nullptr check failed in cuphySetupCrcEncode"
    );
    EXPECT_TRUE(result) << "d_tbPrmsArray nullptr check test failed for cuphySetupCrcEncode";

    // Test null pointer check for cpu_desc
    result = testSetupCrcEncodeBoundaryCondition(
        2,  // nTBs
        6,  // maxNCBsPerTB
        1024,  // maxTbSizeBytes
        (uint32_t*)0x1,  // d_tbCRCs is non-nullptr
        (uint32_t*)0x1,  // d_inputTransportBlocks is non-nullptr
        (uint8_t*)0x1,   // d_codeBlocks is non-nullptr
        (PdschPerTbParams*)0x1,  // d_tbPrmsArray is non-nullptr
        nullptr,  // cpu_desc is nullptr
        nullptr,  // gpu_desc
        "cpu_desc nullptr check failed in cuphySetupCrcEncode"
    );
    EXPECT_TRUE(result) << "cpu_desc nullptr check test failed for cuphySetupCrcEncode";

    // Test null pointer check for gpu_desc
    result = testSetupCrcEncodeBoundaryCondition(
        2,  // nTBs
        6,  // maxNCBsPerTB
        1024,  // maxTbSizeBytes
        (uint32_t*)0x1,  // d_tbCRCs is non-nullptr
        (uint32_t*)0x1,  // d_inputTransportBlocks is non-nullptr
        (uint8_t*)0x1,   // d_codeBlocks is non-nullptr
        (PdschPerTbParams*)0x1,  // d_tbPrmsArray is non-nullptr
        (uint8_t*)0x1,   // cpu_desc is non-nullptr
        nullptr,  // gpu_desc is nullptr
        "gpu_desc nullptr check failed in cuphySetupCrcEncode"
    );
    EXPECT_TRUE(result) << "gpu_desc nullptr check test failed for cuphySetupCrcEncode";

    // Test boundary condition for nTBs > PDSCH_MAX_UES_PER_CELL_GROUP
    result = testSetupCrcEncodeBoundaryCondition(
        PDSCH_MAX_UES_PER_CELL_GROUP + 1,  // nTBs exceeds maximum
        6,  // maxNCBsPerTB
        1024,  // maxTbSizeBytes
        (uint32_t*)0x1,  // d_tbCRCs is non-nullptr
        (uint32_t*)0x1,  // d_inputTransportBlocks is non-nullptr
        (uint8_t*)0x1,   // d_codeBlocks is non-nullptr
        (PdschPerTbParams*)0x1,  // d_tbPrmsArray is non-nullptr
        (uint8_t*)0x1,   // cpu_desc is non-nullptr
        (uint8_t*)0x1,   // gpu_desc is non-nullptr
        "nTBs > PDSCH_MAX_UES_PER_CELL_GROUP check failed in cuphySetupCrcEncode"
    );
    EXPECT_TRUE(result) << "nTBs boundary condition test failed for cuphySetupCrcEncode";

    // Test boundary condition for maxNCBsPerTB > MAX_N_CBS_PER_TB_SUPPORTED
    result = testSetupCrcEncodeBoundaryCondition(
        2,  // nTBs
        MAX_N_CBS_PER_TB_SUPPORTED + 1,  // maxNCBsPerTB exceeds maximum
        1024,  // maxTbSizeBytes
        (uint32_t*)0x1,  // d_tbCRCs is non-nullptr
        (uint32_t*)0x1,  // d_inputTransportBlocks is non-nullptr
        (uint8_t*)0x1,   // d_codeBlocks is non-nullptr
        (PdschPerTbParams*)0x1,  // d_tbPrmsArray is non-nullptr
        (uint8_t*)0x1,   // cpu_desc is non-nullptr
        (uint8_t*)0x1,   // gpu_desc is non-nullptr
        "maxNCBsPerTB > MAX_N_CBS_PER_TB_SUPPORTED check failed in cuphySetupCrcEncode"
    );
    EXPECT_TRUE(result) << "maxNCBsPerTB boundary condition test failed for cuphySetupCrcEncode";

    // Test boundary condition for maxTBByteSize > MAX_BYTES_PER_TRANSPORT_BLOCK
    result = testSetupCrcEncodeBoundaryCondition(
        2,  // nTBs
        6,  // maxNCBsPerTB
        MAX_BYTES_PER_TRANSPORT_BLOCK + 1,  // maxTbSizeBytes exceeds maximum
        (uint32_t*)0x1,  // d_tbCRCs is non-nullptr
        (uint32_t*)0x1,  // d_inputTransportBlocks is non-nullptr
        (uint8_t*)0x1,   // d_codeBlocks is non-nullptr
        (PdschPerTbParams*)0x1,  // d_tbPrmsArray is non-nullptr
        (uint8_t*)0x1,   // cpu_desc is non-nullptr
        (uint8_t*)0x1,   // gpu_desc is non-nullptr
        "maxTBByteSize > MAX_BYTES_PER_TRANSPORT_BLOCK check failed in cuphySetupCrcEncode"
    );
    EXPECT_TRUE(result) << "maxTBByteSize boundary condition test failed for cuphySetupCrcEncode";
}

int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);

    int result = RUN_ALL_TESTS();
    return result;
}
