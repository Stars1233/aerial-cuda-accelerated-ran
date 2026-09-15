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
#include "ldpc.hpp"
#include "rate_matching.hpp"
#include "crc_decode.hpp"
#include "crc_encode.hpp"
#include "dl_rate_matching.hpp"


////////////////////////////////////////////////////////////////////////
// cuphyErrorCorrectionLDPCDecode()
cuphyStatus_t CUPHYWINAPI cuphyErrorCorrectionLDPCDecode(cuphyLDPCDecoder_t                 decoder,
                                                         cuphyTensorDescriptor_t            tensorDescDst,
                                                         void*                              dstAddr,
                                                         cuphyTensorDescriptor_t            tensorDescLLR,
                                                         const void*                        LLRAddr,
                                                         cuphyTensorDescriptor_t            tensorDescSoftOutputs,
                                                         void*                              softOutputsAddr,
                                                         const cuphyLDPCDecodeConfigDesc_t* config,
                                                         cudaStream_t                       strm)
{
    std::array<int, 4> BG2_Kb = {6, 8, 9, 10};
    if(!decoder ||
       !tensorDescDst ||
       !dstAddr ||
       !tensorDescLLR ||
       !LLRAddr ||
       !config ||
       (config->max_iterations < 0) ||
       (config->BG < 1) ||
       (config->BG > 2) ||
       ((1 == config->BG) ? (config->Kb != 22) :
                            (BG2_Kb.end() == std::find(BG2_Kb.begin(), BG2_Kb.end(), config->Kb))) ||
       (config->Z < 2) ||
       (config->Z > 384) ||
       (config->num_parity_nodes < 4) ||
       ((1 == config->BG) ? (config->num_parity_nodes > 46) : (config->num_parity_nodes > 42)) ||
       ((tensorDescSoftOutputs == nullptr) != (softOutputsAddr == nullptr)))
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    // clang-format off
    ldpc::decoder&    d = static_cast<ldpc::decoder&>(*decoder);
    tensor_pair       tDst(static_cast<const tensor_desc&>(*tensorDescDst), dstAddr);
    const_tensor_pair tLLR(static_cast<const tensor_desc&>(*tensorDescLLR), LLRAddr);
    // clang-format on
    //------------------------------------------------------------------
    // Check for LLR type mismatch
    if(config->llr_type != tLLR.first.get().type())
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    //------------------------------------------------------------------
    cuphy_optional<tensor_pair> softOutputs;
    if(tensorDescSoftOutputs)
    {
        const tensor_desc& tdesc = static_cast<const tensor_desc&>(*tensorDescSoftOutputs) ;
        // Check that the soft output type matches the input type, which
        // is currently required by kernel implementations.
        if(tdesc.type() != tLLR.first.get().type())
        {
            return CUPHY_STATUS_INVALID_ARGUMENT;
        }
        // Soft outputs address must be 4-byte aligned
        std::uintptr_t i = reinterpret_cast<uintptr_t>(softOutputsAddr);
        if(0 != (i % 4))
        {
            return CUPHY_STATUS_INVALID_ARGUMENT;
        }
        // Stride (in elements) must be a multiple of 2 for uint32_t stores
        if((tdesc.layout().rank() > 1) &&
           (tdesc.layout().dimensions[1] > 1) &&
           (0 != (tdesc.layout().strides[1] % 2)))
        {
            return CUPHY_STATUS_INVALID_ARGUMENT;
        }
        softOutputs = tensor_pair(tdesc, softOutputsAddr);
    }
    return d.decode(tDst,
                    tLLR,
                    softOutputs,
                    *config,
                    strm);
}

////////////////////////////////////////////////////////////////////////
// cuphyErrorCorrectionLDPCTransportBlockDecode()
cuphyStatus_t CUPHYWINAPI cuphyErrorCorrectionLDPCTransportBlockDecode(cuphyLDPCDecoder_t           decoder,
                                                                       const cuphyLDPCDecodeDesc_t* decodeDesc,
                                                                       cudaStream_t                 strm)
{
    if(!decoder ||
       !decodeDesc)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    ldpc::decoder& d = static_cast<ldpc::decoder&>(*decoder);
    return d.decode_tb(*decodeDesc, strm);
}

////////////////////////////////////////////////////////////////////////
// cuphyErrorCorrectionLDPCDecodeGetWorkspaceSize()
cuphyStatus_t CUPHYWINAPI cuphyErrorCorrectionLDPCDecodeGetWorkspaceSize(cuphyLDPCDecoder_t                 decoder,
                                                                         const cuphyLDPCDecodeConfigDesc_t* config,
                                                                         int                                numCodeWords,
                                                                         size_t*                            sizeInBytes)
{
    static const std::array<int, 2> BG_valid = {1, 2};
    static const std::array<int, 5> Kb_valid = {22, 10, 9, 8, 6};
    // clang-format off
    static const std::array<int, 51> Z_valid =
    {
        2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,  14,
       15,  16,  18,  20,  22,  24,  26,  28,  30,  32,  36,  40,  44,
       48,  52,  56,  60,  64,  72,  80,  88,  96, 104, 112, 120, 128,
      144, 160, 176, 192, 208, 224, 240, 256, 288, 320, 352, 384
    };
    // clang-format on
    if(!decoder ||
       !config ||
       (std::find(BG_valid.begin(), BG_valid.end(), config->BG) == BG_valid.end()) ||
       (std::find(Kb_valid.begin(), Kb_valid.end(), config->Kb) == Kb_valid.end()) ||
       (std::find(Z_valid.begin(), Z_valid.end(), config->Z) == Z_valid.end()) ||
       (numCodeWords <= 0) ||
       !sizeInBytes ||
       (config->num_parity_nodes < 4) ||
       ((1 == config->BG) ? (config->num_parity_nodes > 46) : (config->num_parity_nodes > 42)))
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    ldpc::decoder&          d             = static_cast<ldpc::decoder&>(*decoder);
    std::pair<bool, size_t> workspaceSize = d.workspace_size(*config, numCodeWords);
    if(workspaceSize.first)
    {
        *sizeInBytes = workspaceSize.second;
        return CUPHY_STATUS_SUCCESS;
    }
    else
    {
        return CUPHY_STATUS_UNSUPPORTED_CONFIG;
    }
}

////////////////////////////////////////////////////////////////////////
// cuphyCreateLDPCDecoder()
cuphyStatus_t CUPHYWINAPI cuphyCreateLDPCDecoder(cuphyContext_t      context,
                                                 cuphyLDPCDecoder_t* pdecoder,
                                                 unsigned int        flags)
{
    if(!pdecoder || !context)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    *pdecoder             = nullptr;
    cuphy_i::context& ctx = static_cast<cuphy_i::context&>(*context);
    try
    {
        ldpc::decoder* d = new ldpc::decoder(ctx);
        *pdecoder        = static_cast<cuphyLDPCDecoder_t>(d);
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
// cuphyDestroyLDPCDecoder()
cuphyStatus_t CUPHYWINAPI cuphyDestroyLDPCDecoder(cuphyLDPCDecoder_t decoder)
{
    if(!decoder)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    ldpc::decoder* d = static_cast<ldpc::decoder*>(decoder);
    delete d;
    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphyErrorCorrectionLDPCDecodeSetNormalization()
cuphyStatus_t CUPHYWINAPI cuphyErrorCorrectionLDPCDecodeSetNormalization(cuphyLDPCDecoder_t           decoder,
                                                                         cuphyLDPCDecodeConfigDesc_t* decodeDesc)
{
    if(!decoder || !decodeDesc)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    ldpc::decoder& d = static_cast<ldpc::decoder&>(*decoder);
    return d.set_normalization(*decodeDesc);
}

////////////////////////////////////////////////////////////////////////
// cuphyErrorCorrectionLDPCDecodeGetLaunchDescriptor()
cuphyStatus_t CUPHYWINAPI cuphyErrorCorrectionLDPCDecodeGetLaunchDescriptor(cuphyLDPCDecoder_t             decoder,
                                                                            cuphyLDPCDecodeLaunchConfig_t* launchConfig)
{
    if(!decoder || !launchConfig)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    ldpc::decoder& d = static_cast<ldpc::decoder&>(*decoder);
    return d.get_launch_config(*launchConfig);
}


////////////////////////////////////////////////////////////////////////
// cuphyErrorCorrectionLDPCGetAlgoRequirements()
cuphyStatus_t CUPHYWINAPI cuphyErrorCorrectionLDPCGetAlgoRequirements(cuphyLDPCDecoder_t decoder,
                                                                      int                algo,
                                                                      uint32_t*          requirements)
{
    if(!decoder || !requirements)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    const ldpc::decoder& d = static_cast<const ldpc::decoder&>(*decoder);
    *requirements          = d.algo_requirements(algo);
    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphyPuschRxRateMatchGetDescrInfo()

cuphyStatus_t CUPHYWINAPI cuphyPuschRxRateMatchGetDescrInfo(size_t* pDescrSizeBytes, size_t* pDescrAlignBytes)
{
    if(!pDescrSizeBytes || !pDescrAlignBytes)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    puschRxRateMatch::getDescrInfo(*pDescrSizeBytes, *pDescrAlignBytes);
    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphyCreatePuschRxRateMatch()

cuphyStatus_t CUPHYWINAPI cuphyCreatePuschRxRateMatch(cuphyPuschRxRateMatchHndl_t* pPuschRxRateMatchHndl,
                                                      int                          FPconfig, // 0: FP32 in, FP32 out; 1: FP16 in, FP32 out; 2: FP32 in, FP16 out; 3: FP16 in, FP16 out; other values: don't run
                                                      int                          descramblingOn)                    // enable/disable descrambling
{
    if(!pPuschRxRateMatchHndl)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    *pPuschRxRateMatchHndl = nullptr;
    try
    {
        puschRxRateMatch* pRateMatch = new puschRxRateMatch;
        *pPuschRxRateMatchHndl       = static_cast<cuphyPuschRxRateMatchHndl_t>(pRateMatch);

        pRateMatch->init(FPconfig, descramblingOn);
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
// cuphySetupPuschRxRateMatch()

cuphyStatus_t CUPHYWINAPI cuphySetupPuschRxRateMatch(cuphyPuschRxRateMatchHndl_t       puschRxRateMatchHndl,
                                                     uint16_t                          nSchUes,
                                                     uint16_t*                         pSchUserIdxsCpu,
                                                     const PerTbParams*                pTbPrmsCpu,
                                                     const PerTbParams*                pTbPrmsGpu,
                                                     cuphyTensorPrm_t*                 pTPrmRmIn,
                                                     cuphyTensorPrm_t*                 pTPrmCdm1RmIn,
                                                     void**                            ppRmOut,
                                                     void*                             pCpuDesc,
                                                     void*                             pGpuDesc,
                                                     uint8_t                           enableCpuToGpuDescrAsyncCpy,
                                                     cuphyPuschRxRateMatchLaunchCfg_t* pLaunchCfg,
                                                     cudaStream_t                      strm)
{
    if(!puschRxRateMatchHndl || !pTbPrmsCpu || !pTbPrmsGpu || !pTPrmRmIn || !pTPrmCdm1RmIn || !ppRmOut || !pCpuDesc || !pGpuDesc || !pLaunchCfg || !pSchUserIdxsCpu)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    puschRxRateMatch* pRateMatch = static_cast<puschRxRateMatch*>(puschRxRateMatchHndl);
    pRateMatch->setup(nSchUes,  pSchUserIdxsCpu, pTbPrmsCpu, pTbPrmsGpu, pTPrmRmIn, pTPrmCdm1RmIn, ppRmOut, pCpuDesc, pGpuDesc, enableCpuToGpuDescrAsyncCpy, pLaunchCfg, strm);
    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphyDestroyPuschRxRateMatch()

cuphyStatus_t CUPHYWINAPI cuphyDestroyPuschRxRateMatch(cuphyPuschRxRateMatchHndl_t puschRxRateMatchHndl)
{
    if(!puschRxRateMatchHndl)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    puschRxRateMatch* pRateMatch = static_cast<puschRxRateMatch*>(puschRxRateMatchHndl);
    delete pRateMatch;
    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphyPuschRxCrcDecodeGetDescrInfo()

cuphyStatus_t CUPHYWINAPI cuphyPuschRxCrcDecodeGetDescrInfo(size_t* pDescrSizeBytes, size_t* pDescrAlignBytes)
{
    if(!pDescrSizeBytes || !pDescrAlignBytes)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    puschRxCrcDecode::getDescrInfo(*pDescrSizeBytes, *pDescrAlignBytes);
    return CUPHY_STATUS_SUCCESS;
}

// ////////////////////////////////////////////////////////////////////////
// // cuphyCreatePuschRxCrcDecode()

cuphyStatus_t CUPHYWINAPI cuphyCreatePuschRxCrcDecode(cuphyPuschRxCrcDecodeHndl_t* puschRxCrcDecodeHndl,
                                                      int                          reverseBytes)
{
    if(!puschRxCrcDecodeHndl)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    *puschRxCrcDecodeHndl = nullptr;
    try
    {
        puschRxCrcDecode* pCrcDecode = new puschRxCrcDecode;
        *puschRxCrcDecodeHndl        = static_cast<cuphyPuschRxCrcDecodeHndl_t>(pCrcDecode);

        pCrcDecode->init(reverseBytes);
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

// ////////////////////////////////////////////////////////////////////////
// // cuphySetupPuschRxCrcDecode()

cuphyStatus_t CUPHYWINAPI cuphySetupPuschRxCrcDecode(cuphyPuschRxCrcDecodeHndl_t       puschRxCrcDecodeHndl,
                                                     uint16_t                          nSchUes,
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
                                                     cudaStream_t                      strm)
{
    if(!puschRxCrcDecodeHndl || !pOutputCBCRCs || !pOutputTBs || !pInputCodeBlocks || !pOutputTBCRCs ||
       !pTbPrmsCpu || !pTbPrmsGpu || !pCpuDesc || !pGpuDesc || !pCbCrcLaunchCfg ||
       !pTbCrcLaunchCfg || !pSchUserIdxsCpu) [[unlikely]]
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    puschRxCrcDecode* pCrcDecode = static_cast<puschRxCrcDecode*>(puschRxCrcDecodeHndl);
    return pCrcDecode->setup(nSchUes, pSchUserIdxsCpu, pOutputCBCRCs, pOutputTBs, pInputCodeBlocks, pOutputTBCRCs, pTbPrmsCpu, pTbPrmsGpu, pCpuDesc, pGpuDesc, enableCpuToGpuDescrAsyncCpy, pCbCrcLaunchCfg, pTbCrcLaunchCfg, strm);
}

////////////////////////////////////////////////////////////////////////
// cuphyDestroyPuschRxCrcDecode()

cuphyStatus_t CUPHYWINAPI cuphyDestroyPuschRxCrcDecode(cuphyPuschRxCrcDecodeHndl_t puschRxCrcDecodeHndl)
{
    if(!puschRxCrcDecodeHndl)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    puschRxCrcDecode* pCrcDecode = static_cast<puschRxCrcDecode*>(puschRxCrcDecodeHndl);
    delete pCrcDecode;
    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphyCrcEncodeGetDescrInfo()

cuphyStatus_t CUPHYWINAPI cuphyCrcEncodeGetDescrInfo(size_t* pDescrSizeBytes, size_t* pDescrAlignBytes)
{
    if(!pDescrSizeBytes || !pDescrAlignBytes)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    *pDescrSizeBytes  = sizeof(crcEncodeDescr_t);
    *pDescrAlignBytes = alignof(crcEncodeDescr_t);
    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphyPrepareCrcEncodeGetDescrInfo()

cuphyStatus_t CUPHYWINAPI cuphyPrepareCrcEncodeGetDescrInfo(size_t* pDescrSizeBytes, size_t* pDescrAlignBytes)
{
    if(!pDescrSizeBytes || !pDescrAlignBytes)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    *pDescrSizeBytes  = sizeof(prepareCrcEncodeDescr_t);
    *pDescrAlignBytes = alignof(prepareCrcEncodeDescr_t);
    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphyLDPCEncodeGetDescrInfo()

cuphyStatus_t CUPHYWINAPI cuphyLDPCEncodeGetDescrInfo(size_t* pDescrSizeBytes, size_t* pDescrAlignBytes, uint16_t maxUEs, size_t* pWorkspaceBytes)
{
    if(!pDescrSizeBytes || !pDescrAlignBytes || !pWorkspaceBytes)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    *pDescrSizeBytes  = sizeof(ldpcEncodeDescr_t_array);
    *pDescrAlignBytes = alignof(ldpcEncodeDescr_t_array);
    *pWorkspaceBytes  = 2 * maxUEs * sizeof(LDPC_output_t); // 2x because it includes output and input
    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphyDlRateMatchingGetDescrInfo()

cuphyStatus_t CUPHYWINAPI cuphyDlRateMatchingGetDescrInfo(size_t* pDescrSizeBytes, size_t* pDescrAlignBytes)
{
    if(!pDescrSizeBytes || !pDescrAlignBytes)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    *pDescrSizeBytes  = sizeof(dlRateMatchingDescr_t);
    *pDescrAlignBytes = alignof(dlRateMatchingDescr_t);
    return CUPHY_STATUS_SUCCESS;
}
