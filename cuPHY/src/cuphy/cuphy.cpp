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
#include "cuphy.hpp"
#include "cuphy_api.h" // to be removed
#include <algorithm>
#include <new>
#include "bfc.hpp"
#include "ch_est.hpp"
#include "pusch_noise_intf_est.hpp"
#include "cfo_ta_est.hpp"
#include "channel_eq.hpp"
#include "channel_est.hpp"
#include "pusch_rssi.hpp"
#include "rate_matching.hpp"
#include "crc_encode.hpp"
#include "dl_rate_matching.hpp"
#include "ldpc.hpp"
#include "modulation_mapper.hpp"
#include "pdsch_dmrs.hpp"
#include "polar_encoder.hpp"
#include "rm_decoder.hpp"
#include "simplex_decoder.hpp"
#include "pucch_F0_receiver.hpp"
#include "pucch_F1_receiver.hpp"
#include "pucch_F2_front_end.hpp"
#include "pucch_F3_front_end.hpp"
#include "pucch_F3_csi2Ctrl.hpp"
#include "pucch_F3_segLLRs.hpp"
#include "pucch_F234_uci_seg.hpp"
#include "soft_demapper.hpp"
#include "rng.hpp"
#include "variant.hpp"
#include "tensor_fill.hpp"
#include "tensor_tile.hpp"
#include "tensor_elementwise.hpp"
#include "tensor_reduction.hpp"
#include "comp_cwTreeTypes.hpp"
#include "polar_seg_deRm_deItl.hpp"
#include "uciOnPusch_segLLRs1.hpp"
#include "uciOnPusch_segLLRs2.hpp"
#include "uciOnPusch_segLLRs0.hpp"
#include "uciOnPusch_csi2Ctrl.hpp"
#include "polar_decoder.hpp"
#include "srs_chEst.hpp"
#include "trt_engine.hpp"
#include "empty_kernels.hpp"

#include <vector>


////////////////////////////////////////////////////////////////////////
// cuphyBfcCoefCompute()
cuphyStatus_t CUPHYWINAPI cuphyBfcCoefCompute(unsigned int            nBSAnts,
                                              unsigned int            nLayers,
                                              unsigned int            Nprb,
                                              cuphyTensorDescriptor_t tDescH,
                                              const void*             HAddr,
                                              cuphyTensorDescriptor_t tDescLambda,
                                              const void*             lambdaAddr,
                                              cuphyTensorDescriptor_t tDescCoef,
                                              void*                   coefAddr,
                                              cuphyTensorDescriptor_t tDescDbg,
                                              void*                   dbgAddr,
                                              cudaStream_t            strm)
{
    //------------------------------------------------------------------
    // Validate inputs
    if(!tDescH ||
       !HAddr ||
       !tDescLambda ||
       !lambdaAddr ||
       !tDescCoef ||
       !coefAddr ||
       !tDescDbg ||
       !dbgAddr)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    //------------------------------------------------------------------
    // clang-format off
    const_tensor_pair tPairH       (static_cast<const tensor_desc&>(*tDescH)     ,  HAddr);
    const_tensor_pair tPairLambda  (static_cast<const tensor_desc&>(*tDescLambda),  lambdaAddr);
    tensor_pair       tPairCoef    (static_cast<const tensor_desc&>(*tDescCoef)  ,  coefAddr);
    tensor_pair       tPairDbg     (static_cast<const tensor_desc&>(*tDescDbg)   ,  dbgAddr);
    // clang-format on

    bfw_coefComp::bfcCoefCompute(static_cast<uint32_t>(nBSAnts),
                        static_cast<uint32_t>(nLayers),
                        static_cast<uint32_t>(Nprb),
                        tPairH,
                        tPairLambda,
                        tPairCoef,
                        tPairDbg,
                        strm);

    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphyCreateBfwCoefComp()
cuphyStatus_t CUPHYWINAPI cuphyCreateBfwCoefComp(cuphyBfwCoefCompHndl_t* pBfwCoefCompHndl,
                                                 uint8_t                 enableCpuToGpuDescrAsyncCpy,
                                                 uint8_t                 compressBitwidth,
                                                 uint16_t                nMaxUeGrps,
                                                 uint16_t                nMaxTotalLayers,
                                                 float                   beta,
                                                 float                   lambda,
                                                 uint8_t                 bfwPowerNormAlg_selector,
                                                 uint8_t                 enableBatchedMemcpy,
                                                 void*                   pStatDescrCpu,
                                                 void*                   pStatDescrGpu,
                                                 void*                   pDynDescrsCpu,
                                                 void*                   pDynDescrsGpu,
                                                 void*                   pHetCfgUeGrpMapCpu,
                                                 void*                   pHetCfgUeGrpMapGpu,
                                                 void*                   pUeGrpPrmsCpu,
                                                 void*                   pUeGrpPrmsGpu,
                                                 void*                   pBfLayerPrmsCpu,
                                                 void*                   pBfLayerPrmsGpu,
                                                 cudaStream_t            strm)
{
    if(!pBfwCoefCompHndl || !pStatDescrCpu || !pStatDescrGpu || !pDynDescrsCpu || !pDynDescrsGpu || !pHetCfgUeGrpMapCpu || !pHetCfgUeGrpMapGpu ||
       !pUeGrpPrmsCpu || !pUeGrpPrmsGpu || !pBfLayerPrmsCpu || !pBfLayerPrmsGpu)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    *pBfwCoefCompHndl = nullptr;
    try
    {
        bfw_coefComp::bfwCoefComp* pBfwCoefComp = new bfw_coefComp::bfwCoefComp(nMaxUeGrps, nMaxTotalLayers, enableBatchedMemcpy);
        *pBfwCoefCompHndl                       = static_cast<cuphyBfwCoefCompHndl_t>(pBfwCoefComp);

        pBfwCoefComp->init((0 != enableCpuToGpuDescrAsyncCpy) ? true : false,
                           compressBitwidth,
                           beta,
                           lambda,
                           bfwPowerNormAlg_selector,
                           pStatDescrCpu,
                           pStatDescrGpu,
                           pDynDescrsCpu,
                           pDynDescrsGpu,
                           pHetCfgUeGrpMapCpu,
                           pHetCfgUeGrpMapGpu,
                           pUeGrpPrmsCpu,
                           pUeGrpPrmsGpu,
                           pBfLayerPrmsCpu,
                           pBfLayerPrmsGpu,
                           strm);
    }
    catch(std::bad_alloc& eba)
    {
        return CUPHY_STATUS_ALLOC_FAILED;
    }
    catch(std::exception& e)
    {
        NVLOGE_FMT(NVLOG_BFW, AERIAL_CUPHY_EVENT, "{} EXCEPTION: {}", __FUNCTION__, e.what());
        return CUPHY_STATUS_INTERNAL_ERROR;
    }
    catch(...)
    {
        NVLOGE_FMT(NVLOG_BFW, AERIAL_CUPHY_EVENT, "{} UNKNOWN EXCEPTION", __FUNCTION__);
        return CUPHY_STATUS_INTERNAL_ERROR;
    }
    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphyDestroyBfwCoefComp()
cuphyStatus_t CUPHYWINAPI cuphyDestroyBfwCoefComp(cuphyBfwCoefCompHndl_t bfwCoefCompHndl)
{
    if(!bfwCoefCompHndl)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    bfw_coefComp::bfwCoefComp* pBfwCoefComp = static_cast<bfw_coefComp::bfwCoefComp*>(bfwCoefCompHndl);
    delete pBfwCoefComp;
    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphyGetDescrInfoBfwCoefComp()
cuphyStatus_t CUPHYWINAPI cuphyGetDescrInfoBfwCoefComp(uint16_t               nMaxUeGrps,
                                                       uint16_t               nMaxTotalLayers,
                                                       size_t*                pStatDescrSizeBytes,
                                                       size_t*                pStatDescrAlignBytes,
                                                       size_t*                pDynDescrSizeBytes,
                                                       size_t*                pDynDescrAlignBytes,
                                                       size_t*                pHetCfgUeGrpMapSizeBytes,
                                                       size_t*                pHetCfgUeGrpMapAlignBytes,
                                                       size_t*                pUeGrpPrmsSizeBytes,
                                                       size_t*                pUeGrpPrmsAlignBytes,
                                                       size_t*                pBfLayerPrmsSizeBytes,
                                                       size_t*                pBfLayerPrmsAlignBytes)
{
    if(!pStatDescrSizeBytes || !pStatDescrAlignBytes || !pDynDescrSizeBytes || !pDynDescrAlignBytes ||
       !pHetCfgUeGrpMapSizeBytes || !pHetCfgUeGrpMapAlignBytes || !pUeGrpPrmsSizeBytes || !pUeGrpPrmsAlignBytes ||
       !pBfLayerPrmsSizeBytes || !pBfLayerPrmsAlignBytes)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    bfw_coefComp::bfwCoefComp::getDescrInfo(nMaxUeGrps,
                                            nMaxTotalLayers,
                                            *pStatDescrSizeBytes,
                                            *pStatDescrAlignBytes,
                                            *pDynDescrSizeBytes,
                                            *pDynDescrAlignBytes,
                                            *pHetCfgUeGrpMapSizeBytes,
                                            *pHetCfgUeGrpMapAlignBytes,
                                            *pUeGrpPrmsSizeBytes,
                                            *pUeGrpPrmsAlignBytes,
                                            *pBfLayerPrmsSizeBytes,
                                            *pBfLayerPrmsAlignBytes);
    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphySetupBfwCoefComp()
cuphyStatus_t CUPHYWINAPI cuphySetupBfwCoefComp(cuphyBfwCoefCompHndl_t        bfwCoefCompHndl,
                                                uint16_t                      nUeGrps,
                                                cuphyBfwUeGrpPrm_t const*     pUeGrpPrms,
                                                uint8_t                       enableCpuToGpuDescrAsyncCpy,
                                                cuphySrsChEstBuffInfo_t*      pChEstInfo,
                                                uint8_t**                     pBfwCompCoef,
                                                cuphyBfwCoefCompLaunchCfgs_t* pLaunchCfgs,
                                                cudaStream_t                  strm)
{
    if(!bfwCoefCompHndl || !pUeGrpPrms || !pChEstInfo || !pBfwCompCoef || !pLaunchCfgs)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    bfw_coefComp::bfwCoefComp* pBfwCoefComp = static_cast<bfw_coefComp::bfwCoefComp*>(bfwCoefCompHndl);

    return pBfwCoefComp->setupCoefComp(nUeGrps,
                                       pUeGrpPrms,
                                       (0 != enableCpuToGpuDescrAsyncCpy) ? true : false,
                                       pChEstInfo,
                                       pBfwCompCoef,
                                       pLaunchCfgs,
                                       strm);
}

////////////////////////////////////////////////////////////////////////
// cuphyChannelEst1DTimeFrequency()
cuphyStatus_t CUPHYWINAPI cuphyChannelEst1DTimeFrequency(cuphyTensorDescriptor_t tensorDescDst,
                                                         void*                   dstAddr,
                                                         cuphyTensorDescriptor_t tensorDescSymbols,
                                                         const void*             symbolsAddr,
                                                         cuphyTensorDescriptor_t tensorDescFreqFilters,
                                                         const void*             freqFiltersAddr,
                                                         cuphyTensorDescriptor_t tensorDescTimeFilters,
                                                         const void*             timeFiltersAddr,
                                                         cuphyTensorDescriptor_t tensorDescFreqIndices,
                                                         const void*             freqIndicesAddr,
                                                         cuphyTensorDescriptor_t tensorDescTimeIndices,
                                                         const void*             timeIndicesAddr,
                                                         cudaStream_t            strm)
{
    //------------------------------------------------------------------
    // Validate inputs
    if(!tensorDescDst ||
       !dstAddr ||
       !tensorDescSymbols ||
       !symbolsAddr ||
       !tensorDescFreqFilters ||
       !freqFiltersAddr ||
       !tensorDescTimeFilters ||
       !timeFiltersAddr ||
       !tensorDescFreqIndices ||
       !freqIndicesAddr ||
       !tensorDescTimeIndices ||
       !timeIndicesAddr)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    //------------------------------------------------------------------
    // clang-format off
    tensor_pair       tDst        (static_cast<const tensor_desc&>(*tensorDescDst),         dstAddr);
    const_tensor_pair tSymbols    (static_cast<const tensor_desc&>(*tensorDescSymbols),     symbolsAddr);
    const_tensor_pair tFreqFilters(static_cast<const tensor_desc&>(*tensorDescFreqFilters), freqFiltersAddr);
    const_tensor_pair tTimeFilters(static_cast<const tensor_desc&>(*tensorDescTimeFilters), timeFiltersAddr);
    const_tensor_pair tFreqIndices(static_cast<const tensor_desc&>(*tensorDescFreqIndices), freqIndicesAddr);
    const_tensor_pair tTimeIndices(static_cast<const tensor_desc&>(*tensorDescTimeIndices), timeIndicesAddr);
    // clang-format on
    channel_est::mmse_1D_time_frequency(tDst,
                                        tSymbols,
                                        tFreqFilters,
                                        tTimeFilters,
                                        tFreqIndices,
                                        tTimeIndices,
                                        strm);

    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphyPuschRxChEstGetDescrSizes()
cuphyStatus_t CUPHYWINAPI cuphyPuschRxChEstGetDescrInfo(size_t* pStatDescrSizeBytes, size_t* pStatDescrAlignBytes, size_t* pDynDescrSizeBytes, size_t* pDynDescrAlignBytes)
{
    if(!pStatDescrSizeBytes || !pStatDescrAlignBytes || !pDynDescrSizeBytes || !pDynDescrAlignBytes)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    ch_est::puschRxChEst::getDescrInfo(*pStatDescrSizeBytes, *pStatDescrAlignBytes, *pDynDescrSizeBytes, *pDynDescrAlignBytes);
    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphyCreateTrtEngine()
cuphyStatus_t CUPHYWINAPI cuphyCreateTrtEngine(cuphyTrtEngineHndl_t*     pTrtEngineHndl,
                                               const char*               modelFile,
                                               const uint32_t            maxBatchSize,
                                               cuphyTrtTensorPrms_t*     inputTensorPrms,
                                               uint8_t                   numInputs,
                                               cuphyTrtTensorPrms_t*     outputTensorPrms,
                                               uint8_t                   numOutputs,
                                               cudaStream_t              cuStream)
{
    if(!pTrtEngineHndl || !modelFile || !inputTensorPrms || !outputTensorPrms) {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    *pTrtEngineHndl = nullptr;

    std::vector<cuphyTrtTensorPrms_t> inputTensorPrmVec = std::vector<cuphyTrtTensorPrms_t>(inputTensorPrms, inputTensorPrms + numInputs);
    std::vector<cuphyTrtTensorPrms_t> outputTensorPrmVec = std::vector<cuphyTrtTensorPrms_t>(outputTensorPrms, outputTensorPrms + numOutputs);

    constexpr auto prePostEnqueuePolicy = nullptr;
    auto *pTrtEngine = new trt_engine::trtEngine(maxBatchSize,
                                                 inputTensorPrmVec,
                                                 outputTensorPrmVec,
                                                 prePostEnqueuePolicy,
                                                 std::make_unique<trt_engine::PrePostEnqueueTensorConversion>());
    *pTrtEngineHndl = pTrtEngine;

    cuphyStatus_t initStatus = pTrtEngine->init(modelFile);
    if(initStatus != CUPHY_STATUS_SUCCESS) {
        return initStatus;
    }

    // Run a warmup pass to get rid of the startup latency for the first pass.
    cuphyStatus_t warmupStatus = pTrtEngine->warmup(cuStream);

    return warmupStatus;
}


////////////////////////////////////////////////////////////////////////
// cuphyDestroyTrtEngine()
cuphyStatus_t CUPHYWINAPI cuphyDestroyTrtEngine(cuphyTrtEngineHndl_t trtEngineHndl)
{
    if(!trtEngineHndl) {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    trt_engine::trtEngine* pTrtEngine = static_cast<trt_engine::trtEngine*>(trtEngineHndl);
    delete pTrtEngine;
    return CUPHY_STATUS_SUCCESS;
}


////////////////////////////////////////////////////////////////////////
// cuphySetupTrtEngine()
cuphyStatus_t CUPHYWINAPI cuphySetupTrtEngine(cuphyTrtEngineHndl_t        trtEngineHndl,
                                              void**                      ppInputDeviceBuf,
                                              uint8_t                     numInputs,
                                              void**                      ppOutputDeviceBuf,
                                              uint8_t                     numOutputs,
                                              uint32_t                    batchSize)
{
    if(!trtEngineHndl || !ppInputDeviceBuf || !ppOutputDeviceBuf) {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    trt_engine::trtEngine* pTrtEngine = static_cast<trt_engine::trtEngine*>(trtEngineHndl);

    std::vector<void*> inputDeviceBuf = std::vector<void*>(ppInputDeviceBuf, ppInputDeviceBuf + numInputs);
    std::vector<void*> outputDeviceBuf = std::vector<void*>(ppOutputDeviceBuf, ppOutputDeviceBuf + numOutputs);

    cuphyStatus_t status = pTrtEngine->setup(inputDeviceBuf, outputDeviceBuf, batchSize);
    return status;
}


////////////////////////////////////////////////////////////////////////
// cuphyRunTrtEngine()
cuphyStatus_t CUPHYWINAPI cuphyRunTrtEngine(cuphyTrtEngineHndl_t       trtEngineHndl,
                                            cudaStream_t               strm)
{
    if(!trtEngineHndl) {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    trt_engine::trtEngine* pTrtEngine = static_cast<trt_engine::trtEngine*>(trtEngineHndl);
    cuphyStatus_t status = pTrtEngine->run(strm);
    return status;
}

////////////////////////////////////////////////////////////////////////
// cuphyPuschRxNoiseIntfEstGetDescrSizes()
cuphyStatus_t CUPHYWINAPI cuphyPuschRxNoiseIntfEstGetDescrInfo(size_t* pDynDescrSizeBytes, size_t* pDynDescrAlignBytes)
{
    if(!pDynDescrSizeBytes || !pDynDescrAlignBytes)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    pusch_noise_intf_est::puschRxNoiseIntfEst::getDescrInfo(*pDynDescrSizeBytes, *pDynDescrAlignBytes);
    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphyCreatePuschRxNoiseIntfEst()
cuphyStatus_t CUPHYWINAPI cuphyCreatePuschRxNoiseIntfEst(cuphyPuschRxNoiseIntfEstHndl_t* pPuschRxNoiseIntfEstHndl)
{
    if(!pPuschRxNoiseIntfEstHndl)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    *pPuschRxNoiseIntfEstHndl = nullptr;
    try
    {
        pusch_noise_intf_est::puschRxNoiseIntfEst* pNoiseIntfEst = new pusch_noise_intf_est::puschRxNoiseIntfEst;
        *pPuschRxNoiseIntfEstHndl                                = static_cast<cuphyPuschRxNoiseIntfEstHndl_t>(pNoiseIntfEst);
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
// cuphyDestroyPuschRxNoiseIntfEst()
cuphyStatus_t CUPHYWINAPI cuphyDestroyPuschRxNoiseIntfEst(cuphyPuschRxNoiseIntfEstHndl_t puschRxNoiseIntfEstHndl)
{
    if(!puschRxNoiseIntfEstHndl)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    pusch_noise_intf_est::puschRxNoiseIntfEst* pNoiseIntfEst = static_cast<pusch_noise_intf_est::puschRxNoiseIntfEst*>(puschRxNoiseIntfEstHndl);
    delete pNoiseIntfEst;
    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphySetupPuschRxNoiseIntfEst()
cuphyStatus_t CUPHYWINAPI cuphySetupPuschRxNoiseIntfEst(cuphyPuschRxNoiseIntfEstHndl_t puschRxNoiseIntfEstHndl,
                                                        cuphyPuschRxUeGrpPrms_t*       pDrvdUeGrpPrmsCpu,
                                                        cuphyPuschRxUeGrpPrms_t*       pDrvdUeGrpPrmsGpu,
                                                        uint16_t                       nUeGrps,
                                                        uint16_t                       nMaxPrb,
                                                        uint8_t                        enableDftSOfdm,
                                                        uint8_t                        dmrsSymbolIdx,
                                                        uint8_t                        enableCpuToGpuDescrAsyncCpy,
                                                        void*                          pDynDescrsCpu,
                                                        void*                          pDynDescrsGpu,
                                                        cuphyPuschRxNoiseIntfEstLaunchCfgs_t* pLaunchCfgs,
                                                        cudaStream_t                   strm,
                                                        uint8_t                        subSlotStageIdx)
{
    pusch_noise_intf_est::puschRxNoiseIntfEst* pNoiseIntfEst = static_cast<pusch_noise_intf_est::puschRxNoiseIntfEst*>(puschRxNoiseIntfEstHndl);
    return pNoiseIntfEst->setup(pDrvdUeGrpPrmsCpu,
                                pDrvdUeGrpPrmsGpu,
                                nUeGrps,
                                nMaxPrb,
                                enableDftSOfdm,
                                dmrsSymbolIdx,
                                (0 != enableCpuToGpuDescrAsyncCpy) ? true : false,
                                pDynDescrsCpu,
                                pDynDescrsGpu,
                                &pLaunchCfgs[subSlotStageIdx * CUPHY_PUSCH_RX_NOISE_INTF_EST_N_MAX_HET_CFGS],
                                strm,
                                subSlotStageIdx);
}

////////////////////////////////////////////////////////////////////////
// cuphyPuschRxCfoTaEstGetDescrInfo()
cuphyStatus_t CUPHYWINAPI cuphyPuschRxCfoTaEstGetDescrInfo(size_t* pStatDescrSizeBytes, size_t* pStatDescrAlignBytes, size_t* pDynDescrSizeBytes, size_t* pDynDescrAlignBytes)
{
    if(!pStatDescrSizeBytes || !pStatDescrAlignBytes || !pDynDescrSizeBytes || !pDynDescrAlignBytes)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    cfo_ta_est::puschRxCfoTaEst::getDescrInfo(*pStatDescrSizeBytes, *pStatDescrAlignBytes, *pDynDescrSizeBytes, *pDynDescrAlignBytes);
    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphyCreatePuschRxCfoTaEst()
cuphyStatus_t CUPHYWINAPI cuphyCreatePuschRxCfoTaEst(cuphyPuschRxCfoTaEstHndl_t* pPuschRxCfoTaEstHndl,
                                                     uint8_t                     enableCpuToGpuDescrAsyncCpy,
                                                     void*                       pStatDescrCpu,
                                                     void*                       pStatDescrGpu,
                                                     cudaStream_t                strm)
{
    if(!pPuschRxCfoTaEstHndl || !pStatDescrCpu || !pStatDescrGpu)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    *pPuschRxCfoTaEstHndl = nullptr;
    try
    {
        cfo_ta_est::puschRxCfoTaEst* pCfoTaEst               = new cfo_ta_est::puschRxCfoTaEst;
        *pPuschRxCfoTaEstHndl                                = static_cast<cuphyPuschRxCfoTaEstHndl_t>(pCfoTaEst);
        cfo_ta_est::puschRxCfoTaEstStatDescr_t& statDescrCpu = *(static_cast<cfo_ta_est::puschRxCfoTaEstStatDescr_t*>(pStatDescrCpu));

        //------------------------------------------------------------------
        pCfoTaEst->init((0 != enableCpuToGpuDescrAsyncCpy) ? true : false,
                        statDescrCpu,
                        pStatDescrGpu,
                        strm);
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
// cuphyDestroyPuschRxCfoTaEst()
cuphyStatus_t CUPHYWINAPI cuphyDestroyPuschRxCfoTaEst(cuphyPuschRxCfoTaEstHndl_t puschRxCfoTaEstHndl)
{
    if(!puschRxCfoTaEstHndl)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    cfo_ta_est::puschRxCfoTaEst* pCfoTaEst = static_cast<cfo_ta_est::puschRxCfoTaEst*>(puschRxCfoTaEstHndl);
    delete pCfoTaEst;
    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphySetupPuschRxCfoTaEst()
cuphyStatus_t CUPHYWINAPI cuphySetupPuschRxCfoTaEst(cuphyPuschRxCfoTaEstHndl_t        puschRxCfoTaEstHndl,
                                                    cuphyPuschRxUeGrpPrms_t*          pDrvdUeGrpPrmsCpu,
                                                    cuphyPuschRxUeGrpPrms_t*          pDrvdUeGrpPrmsGpu,
                                                    float**                           pFoCompensationBuffers,
                                                    uint16_t                          nUeGrps,
                                                    uint32_t                          nMaxPrb,
                                                    cuphyTensorPrm_t*                 pDbg,
                                                    uint8_t                           enableCpuToGpuDescrAsyncCpy,
                                                    void*                             pDynDescrsCpu,
                                                    void*                             pDynDescrsGpu,
                                                    cuphyPuschRxCfoTaEstLaunchCfgs_t* pLaunchCfgs,
                                                    cudaStream_t                      strm)
{
    if(!puschRxCfoTaEstHndl || !pDynDescrsCpu || !pDynDescrsGpu || !pFoCompensationBuffers || !pLaunchCfgs || (pLaunchCfgs->nCfgs > CUPHY_PUSCH_RX_CFO_EST_N_MAX_HET_CFGS))
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    cfo_ta_est::puschRxCfoTaEst*              pPuschRxCfoTaEst = static_cast<cfo_ta_est::puschRxCfoTaEst*>(puschRxCfoTaEstHndl);
    cfo_ta_est::puschRxCfoTaEstDynDescrVec_t& dynDescrVecCpu   = *(static_cast<cfo_ta_est::puschRxCfoTaEstDynDescrVec_t*>(pDynDescrsCpu));
    return pPuschRxCfoTaEst->setup(pDrvdUeGrpPrmsCpu,
                                   pDrvdUeGrpPrmsGpu,
                                   pFoCompensationBuffers,
                                   nUeGrps,
                                   nMaxPrb,
                                   (0 != enableCpuToGpuDescrAsyncCpy) ? true : false,
                                   dynDescrVecCpu,
                                   pDynDescrsGpu,
                                   pLaunchCfgs,
                                   strm);
}

////////////////////////////////////////////////////////////////////////
// cuphyPuschRxChEqGetDescrInfo()
cuphyStatus_t CUPHYWINAPI cuphyPuschRxChEqGetDescrInfo(size_t* pStatDescrSizeBytes,
                                                       size_t* pStatDescrAlignBytes,
                                                       size_t* pIdftStatDescrSizeBytes,
                                                       size_t* pIdftStatDescrAlignBytes,
                                                       size_t* pCoefCompDynDescrSizeBytes,
                                                       size_t* pCoefCompDynDescrAlignBytes,
                                                       size_t* pSoftDemapDynDescrSizeBytes,
                                                       size_t* pSoftDemapDynDescrAlignBytes)
{
    if(!pStatDescrSizeBytes         || !pStatDescrAlignBytes         ||
       !pIdftStatDescrSizeBytes     || !pIdftStatDescrAlignBytes     ||
       !pCoefCompDynDescrSizeBytes  || !pCoefCompDynDescrAlignBytes  ||
       !pSoftDemapDynDescrSizeBytes || !pSoftDemapDynDescrAlignBytes)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    channel_eq::puschRxChEq::getDescrInfo(*pStatDescrSizeBytes,
                                          *pStatDescrAlignBytes,
                                          *pIdftStatDescrSizeBytes,
                                          *pIdftStatDescrAlignBytes,
                                          *pCoefCompDynDescrSizeBytes,
                                          *pCoefCompDynDescrAlignBytes,
                                          *pSoftDemapDynDescrSizeBytes,
                                          *pSoftDemapDynDescrAlignBytes);
    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphyCreatePuschRxChEq()
cuphyStatus_t CUPHYWINAPI cuphyCreatePuschRxChEq(cuphyContext_t          ctx,
                                                 cuphyPuschRxChEqHndl_t* pPuschRxChEqHndl,
                                                 cuphyTensorInfo2_t&     tInfoDftBluesteinWorkspaceTime,
                                                 cuphyTensorInfo2_t&     tInfoDftBluesteinWorkspaceFreq,
                                                 uint                    cudaDeviceArch,
                                                 uint8_t                 enableDftSOfdm,
                                                 uint8_t                 enableDebugEqOutput,
                                                 uint8_t                 enableCpuToGpuDescrAsyncCpy,
                                                 void**                  ppStatDescrCpu,
                                                 void**                  ppStatDescrGpu,
                                                 void**                  ppIdftStatDescrCpu,
                                                 void**                  ppIdftStatDescrGpu,
                                                 cudaStream_t            strm)
{
    if(!ctx || !pPuschRxChEqHndl || !ppStatDescrCpu || !ppStatDescrGpu || !ppIdftStatDescrCpu || !ppIdftStatDescrGpu)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    *pPuschRxChEqHndl = nullptr;
    try
    {
        channel_eq::puschRxChEq* pChEq = new channel_eq::puschRxChEq;
        *pPuschRxChEqHndl              = static_cast<cuphyPuschRxChEqHndl_t>(pChEq);

        //------------------------------------------------------------------
        cuphyStatus_t status = pChEq->init(ctx,
                                           tInfoDftBluesteinWorkspaceTime,
                                           tInfoDftBluesteinWorkspaceFreq,
                                           cudaDeviceArch,
                                           enableDftSOfdm,
                                           enableDebugEqOutput,
                                           (0 != enableCpuToGpuDescrAsyncCpy) ? true : false,
                                           ppStatDescrCpu,
                                           ppStatDescrGpu,
                                           ppIdftStatDescrCpu,
                                           ppIdftStatDescrGpu,
                                           strm);
        return status;
    }
    catch(std::bad_alloc& eba)
    {
        return CUPHY_STATUS_ALLOC_FAILED;
    }
    catch(...)
    {
        return CUPHY_STATUS_INTERNAL_ERROR;
    }
}

////////////////////////////////////////////////////////////////////////
// cuphyDestroyPuschRxChEq()
cuphyStatus_t CUPHYWINAPI cuphyDestroyPuschRxChEq(cuphyPuschRxChEqHndl_t puschRxChEqHndl)
{
    if(!puschRxChEqHndl)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    channel_eq::puschRxChEq* pChEq = static_cast<channel_eq::puschRxChEq*>(puschRxChEqHndl);
    delete pChEq;
    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphySetupPuschRxChEqCoefCompute()
cuphyStatus_t CUPHYWINAPI cuphySetupPuschRxChEqCoefCompute(cuphyPuschRxChEqHndl_t        puschRxChEqHndl,
                                                           cuphyPuschRxUeGrpPrms_t*      pDrvdUeGrpPrmsCpu,
                                                           cuphyPuschRxUeGrpPrms_t*      pDrvdUeGrpPrmsGpu,
                                                           uint16_t                      nUeGrps,
                                                           uint16_t                      nMaxPrb,
                                                           uint8_t                       enableCfoCorrection,
                                                           uint8_t                       enablePuschTdi,
                                                           uint8_t                       enableCpuToGpuDescrAsyncCpy,
                                                           void**                        ppDynDescrsCpu,
                                                           void**                        ppDynDescrsGpu,
                                                           cuphyPuschRxChEqLaunchCfgs_t* pLaunchCfgs,

                                                           cudaStream_t strm)
{
    if(!puschRxChEqHndl || !ppDynDescrsCpu || !ppDynDescrsGpu || !pLaunchCfgs)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    channel_eq::puschRxChEq* pChEq = static_cast<channel_eq::puschRxChEq*>(puschRxChEqHndl);

    return pChEq->setupCoefCompute(pDrvdUeGrpPrmsCpu,
                                   pDrvdUeGrpPrmsGpu,
                                   nUeGrps,
                                   nMaxPrb,
                                   enableCfoCorrection,
                                   enablePuschTdi,
                                   (0 != enableCpuToGpuDescrAsyncCpy) ? true : false,
                                   ppDynDescrsCpu,
                                   ppDynDescrsGpu,
                                   pLaunchCfgs,
                                   strm);
}

////////////////////////////////////////////////////////////////////////
// cuphySetupPuschRxChEqSoftDemap()
cuphyStatus_t CUPHYWINAPI cuphySetupPuschRxChEqSoftDemap(cuphyPuschRxChEqHndl_t        puschRxChEqHndl,
                                                         cuphyPuschRxUeGrpPrms_t*      pDrvdUeGrpPrmsCpu,
                                                         cuphyPuschRxUeGrpPrms_t*      pDrvdUeGrpPrmsGpu,
                                                         uint16_t                      nUeGrps,
                                                         uint16_t                      nMaxPrb,
                                                         uint8_t                       enableCfoCorrection,
                                                         uint8_t                       enablePuschTdi,
                                                         uint16_t                      symbolBitmask,
                                                         uint8_t                       enableCpuToGpuDescrAsyncCpy,
                                                         void*                         pDynDescrsCpu,
                                                         void*                         pDynDescrsGpu,
                                                         cuphyPuschRxChEqLaunchCfgs_t* pLaunchCfgs,
                                                         cudaStream_t                  strm)
{
    if(!puschRxChEqHndl || !pDynDescrsCpu || !pDynDescrsGpu || !pLaunchCfgs)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    channel_eq::puschRxChEq*                       pChEq          = static_cast<channel_eq::puschRxChEq*>(puschRxChEqHndl);
    channel_eq::puschRxChEqSoftDemapDynDescrVec_t& dynDescrVecCpu = *(static_cast<channel_eq::puschRxChEqSoftDemapDynDescrVec_t*>(pDynDescrsCpu));

    return pChEq->setupSoftDemap(pDrvdUeGrpPrmsCpu,
                                 pDrvdUeGrpPrmsGpu,
                                 nUeGrps,
                                 nMaxPrb,
                                 enableCfoCorrection,
                                 enablePuschTdi,
                                 symbolBitmask,
                                 (0 != enableCpuToGpuDescrAsyncCpy) ? true : false,
                                 dynDescrVecCpu,
                                 pDynDescrsGpu,
                                 pLaunchCfgs,
                                 strm);
}

////////////////////////////////////////////////////////////////////////
// cuphySetupPuschRxChEqSoftDemapIdft()
cuphyStatus_t CUPHYWINAPI cuphySetupPuschRxChEqSoftDemapIdft(cuphyPuschRxChEqHndl_t        puschRxChEqHndl,
                                                             cuphyPuschRxUeGrpPrms_t*      pDrvdUeGrpPrmsCpu,
                                                             cuphyPuschRxUeGrpPrms_t*      pDrvdUeGrpPrmsGpu,
                                                             uint16_t                      nUeGrps,
                                                             uint16_t                      nMaxPrb,
                                                             uint                          cudaDeviceArch,
                                                             uint8_t                       enableCfoCorrection,
                                                             uint8_t                       enablePuschTdi,
                                                             uint16_t                      symbolBitmask,
                                                             uint8_t                       enableCpuToGpuDescrAsyncCpy,
                                                             void*                         pDynDescrsCpu,
                                                             void*                         pDynDescrsGpu,
                                                             cuphyPuschRxChEqLaunchCfgs_t* pLaunchCfgs,
                                                             cudaStream_t                  strm)
{
    if(!puschRxChEqHndl || !pDynDescrsCpu || !pDynDescrsGpu || !pLaunchCfgs)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    channel_eq::puschRxChEq*                       pChEq          = static_cast<channel_eq::puschRxChEq*>(puschRxChEqHndl);
    channel_eq::puschRxChEqSoftDemapDynDescrVec_t& dynDescrVecCpu = *(static_cast<channel_eq::puschRxChEqSoftDemapDynDescrVec_t*>(pDynDescrsCpu));

    return pChEq->setupSoftDemapIdft(pDrvdUeGrpPrmsCpu,
                                     pDrvdUeGrpPrmsGpu,
                                     nUeGrps,
                                     nMaxPrb,
                                     cudaDeviceArch,
                                     enableCfoCorrection,
                                     enablePuschTdi,
                                     symbolBitmask,
                                     (0 != enableCpuToGpuDescrAsyncCpy) ? true : false,
                                     dynDescrVecCpu,
                                     pDynDescrsGpu,
                                     pLaunchCfgs,
                                     strm);
}

////////////////////////////////////////////////////////////////////////
// cuphySetupPuschRxChEqSoftDemapAfterDft()
cuphyStatus_t CUPHYWINAPI cuphySetupPuschRxChEqSoftDemapAfterDft(cuphyPuschRxChEqHndl_t        puschRxChEqHndl,
                                                                 cuphyPuschRxUeGrpPrms_t*      pDrvdUeGrpPrmsCpu,
                                                                 cuphyPuschRxUeGrpPrms_t*      pDrvdUeGrpPrmsGpu,
                                                                 uint16_t                      nUeGrps,
                                                                 uint16_t                      nMaxPrb,
                                                                 uint8_t                       enableCfoCorrection,
                                                                 uint8_t                       enablePuschTdi,
                                                                 uint16_t                      symbolBitmask,
                                                                 uint8_t                       enableCpuToGpuDescrAsyncCpy,
                                                                 void*                         pDynDescrsCpu,
                                                                 void*                         pDynDescrsGpu,
                                                                 cuphyPuschRxChEqLaunchCfgs_t* pLaunchCfgs,
                                                                 cudaStream_t                  strm)
{
    if(!puschRxChEqHndl || !pDynDescrsCpu || !pDynDescrsGpu || !pLaunchCfgs)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    channel_eq::puschRxChEq*                       pChEq          = static_cast<channel_eq::puschRxChEq*>(puschRxChEqHndl);
    channel_eq::puschRxChEqSoftDemapDynDescrVec_t& dynDescrVecCpu = *(static_cast<channel_eq::puschRxChEqSoftDemapDynDescrVec_t*>(pDynDescrsCpu));

    return pChEq->setupSoftDemapAfterDft(pDrvdUeGrpPrmsCpu,
                                         pDrvdUeGrpPrmsGpu,
                                         nUeGrps,
                                         nMaxPrb,
                                         enableCfoCorrection,
                                         enablePuschTdi,
                                         symbolBitmask,
                                         (0 != enableCpuToGpuDescrAsyncCpy) ? true : false,
                                         dynDescrVecCpu,
                                         pDynDescrsGpu,
                                         pLaunchCfgs,
                                         strm);
}

////////////////////////////////////////////////////////////////////////
// cuphyCreatePuschRxRssi()
cuphyStatus_t CUPHYWINAPI cuphyCreatePuschRxRssi(cuphyPuschRxRssiHndl_t* pPuschRxRssiHndl)
{
    if(!pPuschRxRssiHndl)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    *pPuschRxRssiHndl = nullptr;
    try
    {
        puschRx_rssi::puschRxRssi* pPuschRxRssi = new puschRx_rssi::puschRxRssi;
        *pPuschRxRssiHndl                       = static_cast<cuphyPuschRxRssiHndl_t>(pPuschRxRssi);
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
// cuphyDestroyPuschRxRssi()
cuphyStatus_t CUPHYWINAPI cuphyDestroyPuschRxRssi(cuphyPuschRxRssiHndl_t puschRxRssiHndl)
{
    if(!puschRxRssiHndl)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    puschRx_rssi::puschRxRssi* pPuschRxRssi = static_cast<puschRx_rssi::puschRxRssi*>(puschRxRssiHndl);
    delete pPuschRxRssi;
    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphySetupPuschRxRssi()
cuphyStatus_t cuphySetupPuschRxRssi(cuphyPuschRxRssiHndl_t        puschRxRssiHndl,
                                    cuphyPuschRxUeGrpPrms_t*      pDrvdUeGrpPrmsCpu,
                                    cuphyPuschRxUeGrpPrms_t*      pDrvdUeGrpPrmsGpu,
                                    uint16_t                      nUeGrps,
                                    uint32_t                      nMaxPrb,
                                    uint8_t                       dmrsSymbolIdx,
                                    uint8_t                       enableCpuToGpuDescrAsyncCpy,
                                    void*                         pDynDescrsCpu,
                                    void*                         pDynDescrsGpu,
                                    cuphyPuschRxRssiLaunchCfgs_t* pLaunchCfgs,
                                    cudaStream_t                  strm)
{
    if(!puschRxRssiHndl || !pDynDescrsCpu || !pDynDescrsGpu || !pLaunchCfgs)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    puschRx_rssi::puschRxRssi*              pPuschRssi     = static_cast<puschRx_rssi::puschRxRssi*>(puschRxRssiHndl);
    puschRx_rssi::puschRxRssiDynDescrVec_t& dynDescrVecCpu = *(static_cast<puschRx_rssi::puschRxRssiDynDescrVec_t*>(pDynDescrsCpu));

    return pPuschRssi->setupRssiMeas(pDrvdUeGrpPrmsCpu,
                                     pDrvdUeGrpPrmsGpu,
                                     nUeGrps,
                                     nMaxPrb,
                                     dmrsSymbolIdx,
                                     (0 != enableCpuToGpuDescrAsyncCpy) ? true : false,
                                     dynDescrVecCpu,
                                     pDynDescrsGpu,
                                     pLaunchCfgs,
                                     strm);
}

////////////////////////////////////////////////////////////////////////
// cuphySetupPuschRxRsrp()
cuphyStatus_t cuphySetupPuschRxRsrp(cuphyPuschRxRssiHndl_t        puschRxRssiHndl,
                                    cuphyPuschRxUeGrpPrms_t*      pDrvdUeGrpPrmsCpu,
                                    cuphyPuschRxUeGrpPrms_t*      pDrvdUeGrpPrmsGpu,
                                    uint16_t                      nUeGrps,
                                    uint32_t                      nMaxPrb,
                                    uint8_t                       dmrsSymbolIdx,
                                    uint8_t                       enableCpuToGpuDescrAsyncCpy,
                                    void*                         pDynDescrsCpu,
                                    void*                         pDynDescrsGpu,
                                    cuphyPuschRxRsrpLaunchCfgs_t* pLaunchCfgs,
                                    cudaStream_t                  strm)
{
    if(!puschRxRssiHndl || !pDynDescrsCpu || !pDynDescrsGpu || !pLaunchCfgs)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    puschRx_rssi::puschRxRssi*              pPuschRssi     = static_cast<puschRx_rssi::puschRxRssi*>(puschRxRssiHndl);
    puschRx_rssi::puschRxRsrpDynDescrVec_t& dynDescrVecCpu = *(static_cast<puschRx_rssi::puschRxRsrpDynDescrVec_t*>(pDynDescrsCpu));

    return pPuschRssi->setupRsrpMeas(pDrvdUeGrpPrmsCpu,
                                     pDrvdUeGrpPrmsGpu,
                                     nUeGrps,
                                     nMaxPrb,
                                     dmrsSymbolIdx,
                                     (0 != enableCpuToGpuDescrAsyncCpy) ? true : false,
                                     dynDescrVecCpu,
                                     pDynDescrsGpu,
                                     pLaunchCfgs,
                                     strm);
}

////////////////////////////////////////////////////////////////////////
// cuphyPuschRxRssiGetDescrInfo()
cuphyStatus_t CUPHYWINAPI cuphyPuschRxRssiGetDescrInfo(size_t* pRssiDynDescrSizeBytes, size_t* pRssiDynDescrAlignBytes, size_t* pRsrpDynDescrSizeBytes, size_t* pRsrpDynDescrAlignBytes)
{
    if(!pRssiDynDescrSizeBytes || !pRssiDynDescrAlignBytes || !pRsrpDynDescrSizeBytes || !pRsrpDynDescrAlignBytes)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    puschRx_rssi::puschRxRssi::getDescrInfo(*pRssiDynDescrSizeBytes, *pRssiDynDescrAlignBytes, *pRsrpDynDescrSizeBytes, *pRsrpDynDescrAlignBytes);
    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphyPolarEncRateMatch()
cuphyStatus_t CUPHYWINAPI cuphyPolarEncRateMatch(unsigned int   nInfoBits,
                                                 unsigned int   nTxBits,
                                                 uint8_t const* pInfoBits,
                                                 uint32_t*      pNCodedBits,
                                                 uint8_t*       pCodedBits,
                                                 uint8_t*       pTxBits,
                                                 uint32_t       procModeBmsk,
                                                 cudaStream_t   strm)
{
    //------------------------------------------------------------------
    // Validate inputs
    if((!pInfoBits) || (!pNCodedBits) || (!pCodedBits) || (!pTxBits) ||
       (nInfoBits < 1) || (nInfoBits > CUPHY_POLAR_ENC_MAX_INFO_BITS) ||
       (nTxBits < 1) || (nTxBits > CUPHY_POLAR_ENC_MAX_TX_BITS))
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    // Ensure 4byte (32b) alignment on all input buffers
    if(((reinterpret_cast<uintptr_t>(pInfoBits) & 0x3) != 0) ||
       ((reinterpret_cast<uintptr_t>(pCodedBits) & 0x3) != 0) ||
       ((reinterpret_cast<uintptr_t>(pTxBits) & 0x3) != 0))
    {
        return CUPHY_STATUS_UNSUPPORTED_ALIGNMENT;
    }

    //------------------------------------------------------------------
    polar_encoder::encodeRateMatch(static_cast<uint32_t>(nInfoBits),
                                   static_cast<uint32_t>(nTxBits),
                                   pInfoBits,
                                   pNCodedBits,
                                   pCodedBits,
                                   pTxBits,
                                   procModeBmsk,
                                   strm);


    cudaError_t e = cudaGetLastError();
    DEBUG_PRINTF("CUDA STATUS (%s:%i): %s\n", __FILE__, __LINE__, cudaGetErrorString(e));
    return (e == cudaSuccess) ? CUPHY_STATUS_SUCCESS : CUPHY_STATUS_INTERNAL_ERROR;

    //return CUPHY_STATUS_SUCCESS;
}

cuphyStatus_t CUPHYWINAPI cuphyRunPolarEncRateMatchSSBs(
    cuphyEncoderRateMatchMultiSSBLaunchCfg_t* pEncdRmSSBCfg,
    uint8_t const*                            pInfoBits,
    uint8_t*                                  pCodedBits,
    uint8_t*                                  pTxBits,
    uint16_t                                  nSSBs,
    cudaStream_t                              strm)
{
    //------------------------------------------------------------------
    // Validate inputs
    if((!pInfoBits) || (!pCodedBits) || (!pTxBits) || (nSSBs == 0))
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    // Ensure 4byte (32b) alignment on all input buffers
    if(((reinterpret_cast<uintptr_t>(pInfoBits) & 0x3) != 0) ||
       ((reinterpret_cast<uintptr_t>(pCodedBits) & 0x3) != 0) ||
       ((reinterpret_cast<uintptr_t>(pTxBits) & 0x3) != 0))
    {
        return CUPHY_STATUS_UNSUPPORTED_ALIGNMENT;
    }

    CUresult e = launch_kernel(pEncdRmSSBCfg->kernelNodeParamsDriver, strm);
    return (e == CUDA_SUCCESS) ? CUPHY_STATUS_SUCCESS : CUPHY_STATUS_INTERNAL_ERROR;

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
    //*pDescrSizeBytes = sizeof(ldpcEncodeDescr_t);
    //*pDescrAlignBytes = alignof(ldpcEncodeDescr_t);
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

////////////////////////////////////////////////////////////////////////
// cuphyModulationGetDescrInfo()

cuphyStatus_t CUPHYWINAPI cuphyModulationGetDescrInfo(size_t* pDescrSizeBytes, size_t* pDescrAlignBytes)
{
    if(!pDescrSizeBytes || !pDescrAlignBytes)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    *pDescrSizeBytes  = sizeof(modulationDescr_t);
    *pDescrAlignBytes = alignof(modulationDescr_t);
    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphyPdschDmrsGetDescrInfo()

cuphyStatus_t CUPHYWINAPI cuphyPdschDmrsGetDescrInfo(size_t* pDescrSizeBytes, size_t* pDescrAlignBytes)
{
    if(!pDescrSizeBytes || !pDescrAlignBytes)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    *pDescrSizeBytes  = sizeof(pdschDmrsDescr_t);
    *pDescrAlignBytes = alignof(pdschDmrsDescr_t);
    return CUPHY_STATUS_SUCCESS;
}


////////////////////////////////////////////////////////////////////////
// cuphyPdschCsirsPrepGetDescrInfo()

cuphyStatus_t CUPHYWINAPI cuphyPdschCsirsPrepGetDescrInfo(size_t* pDescrSizeBytes,
                                                          size_t* pDescrAlignBytes)
{
    if(!pDescrSizeBytes || !pDescrAlignBytes)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    *pDescrSizeBytes  = sizeof(pdschCsirsPrepDescr_t);
    *pDescrAlignBytes = alignof(pdschCsirsPrepDescr_t);
    return CUPHY_STATUS_SUCCESS;
}


////////////////////////////////////////////////////////////////////////
// RM decoder

cuphyStatus_t CUPHYWINAPI cuphyRmDecoderGetDescrInfo(size_t* pDynDescrSizeBytes, size_t* pDynDescrAlignBytes)
{
    if(!pDynDescrSizeBytes || !pDynDescrAlignBytes)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    rmDecoder::getDescrInfo(*pDynDescrSizeBytes, *pDynDescrAlignBytes);
    return CUPHY_STATUS_SUCCESS;
}

cuphyStatus_t CUPHYWINAPI cuphyCreateRmDecoder(cuphyContext_t        context,
                                               cuphyRmDecoderHndl_t* pHndl,
                                               unsigned int          flags,
                                               void*                 pMemoryFootprint)
{
    if(!pHndl || !context)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    *pHndl                = nullptr;
    cuphy_i::context& ctx = static_cast<cuphy_i::context&>(*context);
    try
    {
        rmDecoder* d = new rmDecoder(ctx);
        *pHndl       = static_cast<cuphyRmDecoderHndl_t>(d);
        d->init(pMemoryFootprint);
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

cuphyStatus_t CUPHYWINAPI cuphySetupRmDecoder(cuphyRmDecoderHndl_t       hndl,
                                              uint16_t                   nCws,
                                              cuphyRmCwPrm_t*            pCwPrmsGpu,
                                              uint8_t                    enableCpuToGpuDescrAsyncCpy, // option to copy descriptors from CPU to GPU
                                              void*                      pCpuDynDesc,                 // pointer to descriptor in cpu
                                              void*                      pGpuDynDesc,                 // pointer to descriptor in gpu
                                              cuphyRmDecoderLaunchCfg_t* pLaunchCfg,                  // pointer to launch configuration
                                              cudaStream_t               strm)                                      // stream to perform copy
{
    if(!hndl || !pCwPrmsGpu || !pCpuDynDesc || !pGpuDynDesc || !pLaunchCfg)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    rmDecoder* pDecoder = static_cast<rmDecoder*>(hndl);

    pDecoder->setup(nCws,
                    pCwPrmsGpu,
                    static_cast<uint8_t>(enableCpuToGpuDescrAsyncCpy),
                    static_cast<rmDecoderDynDescr_t*>(pCpuDynDesc),
                    pGpuDynDesc,
                    pLaunchCfg,
                    strm);

    return CUPHY_STATUS_SUCCESS;
}

cuphyStatus_t CUPHYWINAPI cuphyDestroyRmDecoder(cuphyRmDecoderHndl_t hndl)
{
    if(!hndl)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    rmDecoder* pRmDecoder = static_cast<rmDecoder*>(hndl);
    delete pRmDecoder;
    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// Simplex decoder

cuphyStatus_t CUPHYWINAPI cuphySimplexDecoderGetDescrInfo(size_t* pDynDescrSizeBytes, size_t* pDynDescrAlignBytes)
{
    if(!pDynDescrSizeBytes || !pDynDescrAlignBytes)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    SimplexDecoder::getDescrInfo(*pDynDescrSizeBytes, *pDynDescrAlignBytes);
    return CUPHY_STATUS_SUCCESS;
}

cuphyStatus_t CUPHYWINAPI cuphyCreateSimplexDecoder(cuphySimplexDecoderHndl_t* pHndl)
{
    if(!pHndl)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    *pHndl = nullptr;
    try
    {
        SimplexDecoder* d = new SimplexDecoder();
        *pHndl            = static_cast<cuphySimplexDecoderHndl_t>(d);
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

cuphyStatus_t CUPHYWINAPI cuphySetupSimplexDecoder(cuphySimplexDecoderHndl_t       simplexDecoderHndl,
                                                   uint16_t                        nCws,
                                                   cuphySimplexCwPrm_t*            pCwPrmsCpu,
                                                   cuphySimplexCwPrm_t*            pCwPrmsGpu,
                                                   uint8_t                         enableCpuToGpuDescrAsyncCpy, // option to copy descriptors from CPU to GPU
                                                   void*                           pCpuDynDesc,                 // pointer to descriptor in cpu
                                                   void*                           pGpuDynDesc,                 // pointer to descriptor in gpu
                                                   cuphySimplexDecoderLaunchCfg_t* pLaunchCfg,                  // pointer to launch configuration
                                                   cudaStream_t                    strm)                                           // stream to perform copy
{
    if(!pCwPrmsCpu || !pCwPrmsGpu || !pCpuDynDesc || !pGpuDynDesc || !pLaunchCfg)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    SimplexDecoder* pDecoder = static_cast<SimplexDecoder*>(simplexDecoderHndl);

    pDecoder->setup(nCws,
                    pCwPrmsCpu,
                    pCwPrmsGpu,
                    static_cast<bool>(enableCpuToGpuDescrAsyncCpy),
                    static_cast<simplexDecoderDynDescr_t*>(pCpuDynDesc),
                    pGpuDynDesc,
                    pLaunchCfg,
                    strm);

    return CUPHY_STATUS_SUCCESS;
}

cuphyStatus_t CUPHYWINAPI cuphyDestroySimplexDecoder(cuphySimplexDecoderHndl_t simplexDecoderHndl)
{
    if(!simplexDecoderHndl)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    SimplexDecoder* pSimplexDecoder = static_cast<SimplexDecoder*>(simplexDecoderHndl);
    delete pSimplexDecoder;
    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphyPucchF0RxGetDescrInfo()

cuphyStatus_t CUPHYWINAPI cuphyPucchF0RxGetDescrInfo(size_t* pDynDescrSizeBytes, size_t* pDynDescrAlignBytes)
{
    if(!pDynDescrSizeBytes || !pDynDescrAlignBytes)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    pucchF0Rx::getDescrInfo(*pDynDescrSizeBytes, *pDynDescrAlignBytes);
    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphyCreatePucchF0Rx()

cuphyStatus_t CUPHYWINAPI cuphyCreatePucchF0Rx(cuphyPucchF0RxHndl_t* pPucchF0RxHndl, cudaStream_t strm)
{
    if(!pPucchF0RxHndl)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    *pPucchF0RxHndl = nullptr;
    try
    {
        pucchF0Rx* pPucchF0Rx = new pucchF0Rx(strm);
        *pPucchF0RxHndl       = static_cast<cuphyPucchF0RxHndl_t>(pPucchF0Rx);
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
// cuphySetupPucchF0Rx()
cuphyStatus_t CUPHYWINAPI cuphySetupPucchF0Rx(cuphyPucchF0RxHndl_t       pucchF0RxHndl,
                                              cuphyTensorPrm_t*          pDataRx,
                                              cuphyPucchF0F1UciOut_t*    pF0UcisOut,
                                              uint16_t                   nCells,
                                              uint16_t                   nF0Ucis,
                                              uint8_t                    enableUlRxBf,
                                              cuphyPucchUciPrm_t*        pF0UciPrms,
                                              cuphyPucchCellPrm_t*       pCmnCellPrms,
                                              uint8_t                    enableCpuToGpuDescrAsyncCpy,
                                              void*                      pCpuDynDesc,
                                              void*                      pGpuDynDesc,
                                              cuphyPucchF0RxLaunchCfg_t* pLaunchCfg,
                                              cudaStream_t               strm)
{
    if(!pucchF0RxHndl || !pDataRx || !pF0UcisOut || !pF0UciPrms || !pCmnCellPrms || !pCpuDynDesc || !pGpuDynDesc || !pLaunchCfg)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    pucchF0Rx* pPucchF0Rx = static_cast<pucchF0Rx*>(pucchF0RxHndl);

    pPucchF0Rx->setup(pDataRx,
                      pF0UcisOut,
                      nCells,
                      nF0Ucis,
                      enableUlRxBf,
                      pF0UciPrms,
                      pCmnCellPrms,
                      static_cast<bool>(enableCpuToGpuDescrAsyncCpy),
                      static_cast<pucchF0RxDynDescr*>(pCpuDynDesc),
                      pGpuDynDesc,
                      pLaunchCfg,
                      strm);

    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphyDestroyPucchF0Rx()
cuphyStatus_t CUPHYWINAPI cuphyDestroyPucchF0Rx(cuphyPucchF0RxHndl_t pucchF0RxHndl)
{
    if(!pucchF0RxHndl)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    pucchF0Rx* pPucchF0Rx = static_cast<pucchF0Rx*>(pucchF0RxHndl);
    delete pPucchF0Rx;
    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////
// cuphyPucchF1RxGetDescrInfo()

cuphyStatus_t CUPHYWINAPI cuphyPucchF1RxGetDescrInfo(size_t* pDynDescrSizeBytes, size_t* pDynDescrAlignBytes)
{
    if(!pDynDescrSizeBytes || !pDynDescrAlignBytes)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    pucchF1Rx::getDescrInfo(*pDynDescrSizeBytes, *pDynDescrAlignBytes);
    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphyCreatePucchF1Rx()

cuphyStatus_t CUPHYWINAPI cuphyCreatePucchF1Rx(cuphyPucchF1RxHndl_t* pPucchF1RxHndl, cudaStream_t strm)
{
    if(!pPucchF1RxHndl)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    *pPucchF1RxHndl = nullptr;
    try
    {
        pucchF1Rx* pPucchF1Rx = new pucchF1Rx(strm);
        *pPucchF1RxHndl       = static_cast<cuphyPucchF1RxHndl_t>(pPucchF1Rx);
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
// cuphySetupPucchF1Rx()
cuphyStatus_t CUPHYWINAPI cuphySetupPucchF1Rx(cuphyPucchF1RxHndl_t       pucchF1RxHndl,
                                              cuphyTensorPrm_t*          pDataRx,
                                              cuphyPucchF0F1UciOut_t*    pF1UcisOut,
                                              uint16_t                   nCells,
                                              uint16_t                   nF1Ucis,
                                              uint8_t                    enableUlRxBf,
                                              cuphyPucchUciPrm_t*        pF1UciPrms,
                                              cuphyPucchCellPrm_t*       pCmnCellPrms,
                                              uint8_t                    enableCpuToGpuDescrAsyncCpy,
                                              void*                      pCpuDynDesc,
                                              void*                      pGpuDynDesc,
                                              cuphyPucchF1RxLaunchCfg_t* pLaunchCfg,
                                              cudaStream_t               strm)
{
    if(!pucchF1RxHndl || !pDataRx || !pF1UcisOut || !pF1UciPrms || !pCmnCellPrms || !pCpuDynDesc || !pGpuDynDesc || !pLaunchCfg)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    pucchF1Rx* pPucchF1Rx = static_cast<pucchF1Rx*>(pucchF1RxHndl);

    pPucchF1Rx->setup(pDataRx,
                      pF1UcisOut,
                      nCells,
                      nF1Ucis,
                      enableUlRxBf,
                      pF1UciPrms,
                      pCmnCellPrms,
                      static_cast<bool>(enableCpuToGpuDescrAsyncCpy),
                      static_cast<pucchF1RxDynDescr*>(pCpuDynDesc),
                      pGpuDynDesc,
                      pLaunchCfg,
                      strm);

    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphyDestroyPucchF1Rx()
cuphyStatus_t CUPHYWINAPI cuphyDestroyPucchF1Rx(cuphyPucchF1RxHndl_t pucchF1RxHndl)
{
    if(!pucchF1RxHndl)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    pucchF1Rx* pPucchF1Rx = static_cast<pucchF1Rx*>(pucchF1RxHndl);
    delete pPucchF1Rx;
    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
// cuphyPucchF2RxGetDescrInfo()

cuphyStatus_t CUPHYWINAPI cuphyPucchF2RxGetDescrInfo(size_t* pDynDescrSizeBytes, size_t* pDynDescrAlignBytes)
{
    if(!pDynDescrSizeBytes || !pDynDescrAlignBytes)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    pucchF2Rx::getDescrInfo(*pDynDescrSizeBytes, *pDynDescrAlignBytes);
    return CUPHY_STATUS_SUCCESS;
}
////////////////////////////////////////////////////////////////////////
// cuphyCreatePucchF2Rx()

cuphyStatus_t CUPHYWINAPI cuphyCreatePucchF2Rx(cuphyPucchF2RxHndl_t* pPucchF2RxHndl, cudaStream_t strm)
{
    if(!pPucchF2RxHndl)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    *pPucchF2RxHndl = nullptr;
    try
    {
        pucchF2Rx* pPucchF2Rx = new pucchF2Rx(strm);
        *pPucchF2RxHndl       = static_cast<cuphyPucchF2RxHndl_t>(pPucchF2Rx);
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
// cuphySetupPucchF2Rx()
cuphyStatus_t CUPHYWINAPI cuphySetupPucchF2Rx(cuphyPucchF2RxHndl_t       pucchF2RxHndl,
                                              cuphyTensorPrm_t*          pDataRx,
                                              __half**                   pDescramLLRaddrs,
                                              uint8_t*                   pDTXflags,
                                              float*                     pSinr,
                                              float*                     pRssi,
                                              float*                     pRsrp,
                                              float*                     pInterf,
                                              float*                     pNoiseVar,
                                              float*                     pTaEst,
                                              uint16_t                   nCells,
                                              uint16_t                   nF2Ucis,
                                              uint8_t                    enableUlRxBf,
                                              cuphyPucchUciPrm_t*        pF2UciPrms,
                                              cuphyPucchCellPrm_t*       pCmnCellPrms,
                                              uint8_t                    enableCpuToGpuDescrAsyncCpy,
                                              void*                      pCpuDynDesc,
                                              void*                      pGpuDynDesc,
                                              cuphyPucchF2RxLaunchCfg_t* pLaunchCfg,
                                              cudaStream_t               strm)
{
    if(!pucchF2RxHndl || !pDataRx || !pDescramLLRaddrs || !pDTXflags || !pSinr || !pRssi || !pRsrp || !pInterf || !pNoiseVar || !pF2UciPrms || !pCmnCellPrms || !pCpuDynDesc || !pGpuDynDesc || !pLaunchCfg)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    pucchF2Rx* pPucchF2Rx = static_cast<pucchF2Rx*>(pucchF2RxHndl);

    pPucchF2Rx->setup(pDataRx,
                      pDescramLLRaddrs,
                      pDTXflags,
                      pSinr,
                      pRssi,
                      pRsrp,
                      pInterf,
                      pNoiseVar,
                      pTaEst,
                      nCells,
                      nF2Ucis,
                      enableUlRxBf,
                      pF2UciPrms,
                      pCmnCellPrms,
                      static_cast<bool>(enableCpuToGpuDescrAsyncCpy),
                      static_cast<pucchF2RxDynDescr*>(pCpuDynDesc),
                      pGpuDynDesc,
                      pLaunchCfg,
                      strm);

    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphyDestroyPucchF2Rx()
cuphyStatus_t CUPHYWINAPI cuphyDestroyPucchF2Rx(cuphyPucchF2RxHndl_t pucchF2RxHndl)
{
    if(!pucchF2RxHndl)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    pucchF2Rx* pPucchF2Rx = static_cast<pucchF2Rx*>(pucchF2RxHndl);
    delete pPucchF2Rx;
    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
// cuphyPucchF3RxGetDescrInfo()

cuphyStatus_t CUPHYWINAPI cuphyPucchF3RxGetDescrInfo(size_t* pDynDescrSizeBytes, size_t* pDynDescrAlignBytes)
{
    if(!pDynDescrSizeBytes || !pDynDescrAlignBytes)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    pucchF3Rx::getDescrInfo(*pDynDescrSizeBytes, *pDynDescrAlignBytes);
    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphyCreatePucchF3Rx()

cuphyStatus_t CUPHYWINAPI cuphyCreatePucchF3Rx(cuphyPucchF3RxHndl_t* pPucchF3RxHndl, cudaStream_t strm)
{
    if(!pPucchF3RxHndl)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    *pPucchF3RxHndl = nullptr;
    try
    {
        pucchF3Rx* pPucchF3Rx = new pucchF3Rx(strm);
        *pPucchF3RxHndl       = static_cast<cuphyPucchF3RxHndl_t>(pPucchF3Rx);
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
// cuphySetupPucchF3Rx()
cuphyStatus_t CUPHYWINAPI cuphySetupPucchF3Rx(cuphyPucchF3RxHndl_t       pucchF3RxHndl,
                                              cuphyTensorPrm_t*          pDataRx,
                                              __half**                   pDescramLLRaddrs,
                                              uint8_t*                   pDTXflags,
                                              float*                     pSinr,
                                              float*                     pRssi,
                                              float*                     pRsrp,
                                              float*                     pInterf,
                                              float*                     pNoiseVar,
                                              float*                     pTaEst,
                                              uint16_t                   nCells,
                                              uint16_t                   nF3Ucis,
                                              uint8_t                    enableUlRxBf,
                                              cuphyPucchUciPrm_t*        pF3UciPrms,
                                              cuphyPucchCellPrm_t*       pCmnCellPrms,
                                              uint8_t                    enableCpuToGpuDescrAsyncCpy,
                                              void*                      pCpuDynDesc,
                                              void*                      pGpuDynDesc,
                                              cuphyPucchF3RxLaunchCfg_t* pLaunchCfg,
                                              cudaStream_t               strm)
{
    if(!pucchF3RxHndl || !pDataRx || !pDescramLLRaddrs || !pDTXflags || !pSinr || !pRssi || !pRsrp || !pInterf || !pNoiseVar || !pF3UciPrms || !pCmnCellPrms || !pCpuDynDesc || !pGpuDynDesc || !pLaunchCfg)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    pucchF3Rx* pPucchF3Rx = static_cast<pucchF3Rx*>(pucchF3RxHndl);

    pPucchF3Rx->setup(pDataRx,
                      pDescramLLRaddrs,
                      pDTXflags,
                      pSinr,
                      pRssi,
                      pRsrp,
                      pInterf,
                      pNoiseVar,
                      pTaEst,
                      nCells,
                      nF3Ucis,
                      enableUlRxBf,
                      pF3UciPrms,
                      pCmnCellPrms,
                      static_cast<bool>(enableCpuToGpuDescrAsyncCpy),
                      static_cast<pucchF3RxDynDescr*>(pCpuDynDesc),
                      pGpuDynDesc,
                      pLaunchCfg,
                      strm);

    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphyDestroyPucchF3Rx()
cuphyStatus_t CUPHYWINAPI cuphyDestroyPucchF3Rx(cuphyPucchF3RxHndl_t pucchF3RxHndl)
{
    if(!pucchF3RxHndl)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    pucchF3Rx* pPucchF3Rx = static_cast<pucchF3Rx*>(pucchF3RxHndl);
    delete pPucchF3Rx;
    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphyPucchF3Csi2CtrlGetDescrInfo()

cuphyStatus_t CUPHYWINAPI cuphyPucchF3Csi2CtrlGetDescrInfo(size_t* pDynDescrSizeBytes, size_t* pDynDescrAlignBytes)
{
    if(!pDynDescrSizeBytes || !pDynDescrAlignBytes)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    pucchF3Csi2Ctrl::getDescrInfo(*pDynDescrSizeBytes, *pDynDescrAlignBytes);
    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphyCreatePucchF3Csi2Ctrl()

cuphyStatus_t CUPHYWINAPI cuphyCreatePucchF3Csi2Ctrl(cuphyPucchF3Csi2CtrlHndl_t* pPucchF3Csi2CtrlHndl)
{
    if(!pPucchF3Csi2CtrlHndl)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    *pPucchF3Csi2CtrlHndl = nullptr;
    try
    {
        pucchF3Csi2Ctrl* pPucchF3Csi2Ctrl = new pucchF3Csi2Ctrl;
        *pPucchF3Csi2CtrlHndl             = static_cast<cuphyPucchF3Csi2CtrlHndl_t>(pPucchF3Csi2Ctrl);
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
// cuphySetupPucchF3Csi2Ctrl()

cuphyStatus_t CUPHYWINAPI cuphySetupPucchF3Csi2Ctrl(cuphyPucchF3Csi2CtrlHndl_t           pucchF3Csi2CtrlHndl,
                                                    uint16_t                             nCsi2Ucis,
                                                    uint16_t*                            pCsi2UciIdxsCpu,
                                                    cuphyPucchUciPrm_t*                  pUciPrmsCpu,
                                                    cuphyPucchUciPrm_t*                  pUciPrmsGpu,
                                                    cuphyPucchCellStatPrm_t*             pCellStatPrmsGpu,
                                                    cuphyPucchF234OutOffsets_t*          pPucchF3OutOffsetsCpu,
                                                    uint8_t*                             pUciPayloadsGpu,
                                                    uint16_t*                            pNumCsi2BitsGpu,
                                                    cuphyPolarUciSegPrm_t*               pCsi2PolarSegPrmsGpu,
                                                    cuphyPolarCwPrm_t*                   pCsi2PolarCwPrmsGpu,
                                                    cuphyRmCwPrm_t*                      pCsi2RmCwPrmsGpu,
                                                    cuphySimplexCwPrm_t*                 pCsi2SpxCwPrmsGpu,
                                                    void*                                pCpuDynDesc,
                                                    void*                                pGpuDynDesc,
                                                    bool                                 enableCpuToGpuDescrAsyncCpy,
                                                    cuphyPucchF3Csi2CtrlLaunchCfg_t*     pLaunchCfg,
                                                    cudaStream_t                         strm)
{
    if(!pucchF3Csi2CtrlHndl || !pCsi2UciIdxsCpu || !pUciPrmsCpu || !pUciPrmsGpu || !pCellStatPrmsGpu || !pPucchF3OutOffsetsCpu || !pUciPayloadsGpu || !pNumCsi2BitsGpu || !pCsi2PolarSegPrmsGpu || !pCsi2PolarCwPrmsGpu || !pCsi2RmCwPrmsGpu || !pCsi2SpxCwPrmsGpu || !pCpuDynDesc || !pGpuDynDesc || !pLaunchCfg)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    // call c++ setup function
    pucchF3Csi2Ctrl* pPucchF3Csi2Ctrl = static_cast<pucchF3Csi2Ctrl*>(pucchF3Csi2CtrlHndl);

    pPucchF3Csi2Ctrl->setup(nCsi2Ucis,
                               pCsi2UciIdxsCpu,
                               pUciPrmsCpu,
                               pUciPrmsGpu,
                               pCellStatPrmsGpu,
                               pPucchF3OutOffsetsCpu,
                               pUciPayloadsGpu,
                               pNumCsi2BitsGpu,
                               pCsi2PolarSegPrmsGpu,
                               pCsi2PolarCwPrmsGpu,
                               pCsi2RmCwPrmsGpu,
                               pCsi2SpxCwPrmsGpu,
                               static_cast<pucchF3Csi2CtrlDynDescr_t*>(pCpuDynDesc),
                               pGpuDynDesc,
                               static_cast<bool>(enableCpuToGpuDescrAsyncCpy),
                               pLaunchCfg,
                               strm);

    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphyDestroyUciOnPuschCsi2Ctrl()

cuphyStatus_t CUPHYWINAPI cuphyDestroyPucchF3Csi2Ctrl(cuphyPucchF3Csi2CtrlHndl_t pucchF3Csi2CtrlHndl)
{
    if(!pucchF3Csi2CtrlHndl)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    pucchF3Csi2Ctrl* pPucchF3Csi2Ctrl = static_cast<pucchF3Csi2Ctrl*>(pucchF3Csi2CtrlHndl);
    delete pPucchF3Csi2Ctrl;
    return CUPHY_STATUS_SUCCESS;
}
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
// cuphyPucchF3SegLLRsGetDescrInfo
cuphyStatus_t CUPHYWINAPI cuphyPucchF3SegLLRsGetDescrInfo(size_t* pDynDescrSizeBytes,
                                                          size_t* pDynDescrAlignBytes)
{
    if(!pDynDescrSizeBytes || !pDynDescrAlignBytes)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    pucchF3SegLLRs::getDescrInfo(*pDynDescrSizeBytes, *pDynDescrAlignBytes);
    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphyCreatePucchF3SegLLRs
cuphyStatus_t CUPHYWINAPI cuphyCreatePucchF3SegLLRs(cuphyPucchF3SegLLRsHndl_t* pPucchF3SegLLRsHndl)
{
    if(!pPucchF3SegLLRsHndl)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    *pPucchF3SegLLRsHndl = nullptr;
    try
    {
        pucchF3SegLLRs* pPucchF3SegLLRs = new pucchF3SegLLRs;
        *pPucchF3SegLLRsHndl             = static_cast<cuphyPucchF3SegLLRsHndl_t>(pPucchF3SegLLRs);
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
// cuphySetupPucchF3SegLLRs
cuphyStatus_t CUPHYWINAPI cuphySetupPucchF3SegLLRs(cuphyPucchF3SegLLRsHndl_t            pucchF3SegLLRsHndl,
                                                   uint16_t                             nF3Ucis,
                                                   cuphyPucchUciPrm_t*                  pF3UciPrms,
                                                   __half**                             pDescramLLRaddrs,
                                                   void*                                pCpuDynDesc,
                                                   void*                                pGpuDynDesc,
                                                   bool                                 enableCpuToGpuDescrAsyncCpy,
                                                   cuphyPucchF3SegLLRsLaunchCfg_t*      pLaunchCfg,
                                                   cudaStream_t                         strm)
{
    if(!pucchF3SegLLRsHndl || !pF3UciPrms || !pDescramLLRaddrs || !pCpuDynDesc || !pGpuDynDesc || !pLaunchCfg)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    // call c++ setup function
    pucchF3SegLLRs* pPucchF3SegLLRs = static_cast<pucchF3SegLLRs*>(pucchF3SegLLRsHndl);

    pPucchF3SegLLRs->setup(nF3Ucis,
                           pF3UciPrms,
                           pDescramLLRaddrs,
                           static_cast<pucchF3SegLLRsDynDescr_t*>(pCpuDynDesc),
                           pGpuDynDesc,
                           enableCpuToGpuDescrAsyncCpy,
                           pLaunchCfg,
                           strm);

    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphyDestroyPucchF3SegLLRs
cuphyStatus_t CUPHYWINAPI cuphyDestroyPucchF3SegLLRs(cuphyPucchF3SegLLRsHndl_t pucchF3SegLLRsHndl)
{
    if(!pucchF3SegLLRsHndl)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    pucchF3SegLLRs* pPucchF3SegLLRs = static_cast<pucchF3SegLLRs*>(pucchF3SegLLRsHndl);
    delete pPucchF3SegLLRs;
    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
// cuphyPucchF234UciSegGetDescrInfo
cuphyStatus_t CUPHYWINAPI cuphyPucchF234UciSegGetDescrInfo(size_t* pDynDescrSizeBytes,
                                                           size_t* pDynDescrAlignBytes)
{
    if(!pDynDescrSizeBytes || !pDynDescrAlignBytes)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    pucchF234UciSeg::getDescrInfo(*pDynDescrSizeBytes, *pDynDescrAlignBytes);
    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphyCreatePucchF234UciSeg
cuphyStatus_t CUPHYWINAPI cuphyCreatePucchF234UciSeg(cuphyPucchF234UciSegHndl_t* pPucchF234UciSegHndl)
{
    if(!pPucchF234UciSegHndl)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    *pPucchF234UciSegHndl = nullptr;
    try
    {
        pucchF234UciSeg* pPucchF234UciSeg = new pucchF234UciSeg;
        *pPucchF234UciSegHndl             = static_cast<cuphyPucchF234UciSegHndl_t>(pPucchF234UciSeg);
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
// cuphySetupPucchF234UciSeg
cuphyStatus_t CUPHYWINAPI cuphySetupPucchF234UciSeg(cuphyPucchF234UciSegHndl_t       pucchF234UciSegHndl,
                                                    uint16_t                         nF2Ucis,
                                                    uint16_t                         nF3Ucis,
                                                    cuphyPucchUciPrm_t*              pF2UciPrms,
                                                    cuphyPucchUciPrm_t*              pF3UciPrms,
                                                    cuphyPucchF234OutOffsets_t*&     pF2OutOffsetsCpu,
                                                    cuphyPucchF234OutOffsets_t*&     pF3OutOffsetsCpu,
                                                    uint8_t*                         uciPayloadsGpu,
                                                    void*                            pCpuDynDesc,
                                                    void*                            pGpuDynDesc,
                                                    bool                             enableCpuToGpuDescrAsyncCpy,
                                                    cuphyPucchF234UciSegLaunchCfg_t* pLaunchCfg,
                                                    cudaStream_t                     strm)
{
    if(!pucchF234UciSegHndl || (!pF2UciPrms && !pF3UciPrms) || (!pF2OutOffsetsCpu && !pF3OutOffsetsCpu) || !uciPayloadsGpu || !pCpuDynDesc || !pGpuDynDesc || !pLaunchCfg)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    // call c++ setup function
    pucchF234UciSeg* pPucchF234UciSeg = static_cast<pucchF234UciSeg*>(pucchF234UciSegHndl);

    pPucchF234UciSeg->setup(nF2Ucis,
                            nF3Ucis,
                            pF2UciPrms,
                            pF3UciPrms,
                            pF2OutOffsetsCpu,
                            pF3OutOffsetsCpu,
                            uciPayloadsGpu,
                            static_cast<pucchF234UciSegDynDescr_t*>(pCpuDynDesc),
                            pGpuDynDesc,
                            enableCpuToGpuDescrAsyncCpy,
                            pLaunchCfg,
                            strm);

    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphyDestroyPucchF234UciSeg
cuphyStatus_t CUPHYWINAPI cuphyDestroyPucchF234UciSeg(cuphyPucchF234UciSegHndl_t pPucchF234UciSegHndl)
{
    if(!pPucchF234UciSegHndl)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    pucchF234UciSeg* pPucchF234UciSeg = static_cast<pucchF234UciSeg*>(pPucchF234UciSegHndl);
    delete pPucchF234UciSeg;
    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
// cuphyModulateSymbol()
cuphyStatus_t cuphyModulateSymbol(cuphyTensorDescriptor_t tSym,
                                  void*                   pSym,
                                  cuphyTensorDescriptor_t tBits,
                                  const void*             pBits,
                                  int                     log2_QAM,
                                  cudaStream_t            strm)
{
    std::array<int, 5> valid_log_mod = {CUPHY_QAM_2,
                                        CUPHY_QAM_4,
                                        CUPHY_QAM_16,
                                        CUPHY_QAM_64,
                                        CUPHY_QAM_256};
    //------------------------------------------------------------------
    // Validate inputs
    if(!tSym ||
       !pSym ||
       !tBits ||
       !pBits ||
       valid_log_mod.end() == std::find(valid_log_mod.begin(),
                                        valid_log_mod.end(),
                                        log2_QAM))
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    //------------------------------------------------------------------
    const tensor_desc& tSymDesc = static_cast<const tensor_desc&>(*tSym);
    const tensor_desc& tBitDesc = static_cast<const tensor_desc&>(*tBits);
    //------------------------------------------------------------------
    // Validate tensor types
    if(CUPHY_BIT != tBitDesc.type() ||
       (CUPHY_C_32F != tSymDesc.type() && (CUPHY_C_16F != tSymDesc.type())))
    {
        return CUPHY_STATUS_UNSUPPORTED_TYPE;
    }
    //------------------------------------------------------------------
    // Validate tensor sizes
    if((0 != (tBitDesc.layout().dimensions[0] % log2_QAM)) ||
       (tBitDesc.layout().dimensions[0] / log2_QAM != tSymDesc.layout().dimensions[0]))
    {
        return CUPHY_STATUS_SIZE_MISMATCH;
    }
    return cuphy_i::symbol_modulate(tSymDesc,
                                    pSym,
                                    tBitDesc,
                                    pBits,
                                    log2_QAM,
                                    strm);
}

////////////////////////////////////////////////////////////////////////
// cuphyDemodulateSymbol()
cuphyStatus_t cuphyDemodulateSymbol(cuphyContext_t          context,
                                    cuphyTensorDescriptor_t tLLR,
                                    void*                   pLLR,
                                    cuphyTensorDescriptor_t tSym,
                                    const void*             pSym,
                                    int                     log2_QAM,
                                    float                   noiseVariance,
                                    cudaStream_t            strm)
{
    //------------------------------------------------------------------
    // Validate inputs
    if(!context ||
       !tLLR ||
       !pLLR ||
       !tSym ||
       !pSym ||
       (log2_QAM < 1) ||
       (log2_QAM > 8) ||
       (noiseVariance <= 0.0f))
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    tensor_desc& tLLRDesc = static_cast<tensor_desc&>(*tLLR);
    tensor_desc& tSymDesc = static_cast<tensor_desc&>(*tSym);
    if((tLLRDesc.type() != CUPHY_R_32F) &&
       (tLLRDesc.type() != CUPHY_R_16F))
    {
        return CUPHY_STATUS_UNSUPPORTED_CONFIG;
    }
    if((tSymDesc.type() != CUPHY_C_32F) &&
       (tSymDesc.type() != CUPHY_C_16F))
    {
        return CUPHY_STATUS_UNSUPPORTED_CONFIG;
    }
    //------------------------------------------------------------------
    cuphy_i::context& ctx = static_cast<cuphy_i::context&>(*context);
    return cuphy_i::soft_demap(ctx,
                               tLLRDesc,
                               pLLR,
                               tSymDesc,
                               pSym,
                               log2_QAM,
                               noiseVariance,
                               strm);
}

// Set pNodeParams for a generic empty kernel node with num_ptr_args pointer arguments followed by a grid constant argument of size descr_size
cuphyStatus_t CUPHYWINAPI cuphySetGenericEmptyKernelNodeGridConstantParams(CUDA_KERNEL_NODE_PARAMS* pNodeParams, void** pKernelParams, int ptrArgsCnt, uint16_t descr_size)
{
    return internalCuphySetGenericEmptyKernelNodeGridConstantParams(pNodeParams, pKernelParams, ptrArgsCnt, descr_size);
}

cuphyStatus_t CUPHYWINAPI cuphySetGenericEmptyKernelNodeParams(CUDA_KERNEL_NODE_PARAMS* pNodeParams, int ptrArgsCnt, void** pKernelParams)
{
    return internalCuphySetGenericEmptyKernelNodeParams(pNodeParams, ptrArgsCnt, pKernelParams);
}

cuphyStatus_t CUPHYWINAPI cuphySetEmptyKernelNodeParams(CUDA_KERNEL_NODE_PARAMS* pNodeParams)
{
    return internalCuphySetEmptyKernelNodeParams(pNodeParams);
}

void CUPHYWINAPI cuphySetD2HMemcpyNodeParams(CUDA_MEMCPY3D *memcpyParams, void* src_d, void* dst_h, size_t size_in_bytes) {
    *memcpyParams = {0};
    memcpyParams->WidthInBytes = size_in_bytes;
    memcpyParams->Height = 1;
    memcpyParams->Depth = 1;
    memcpyParams->dstHost = dst_h;
    memcpyParams->dstMemoryType = CU_MEMORYTYPE_HOST;
    memcpyParams->srcDevice = reinterpret_cast<CUdeviceptr>(src_d);
    memcpyParams->srcMemoryType = CU_MEMORYTYPE_DEVICE;
}

cuphyStatus_t CUPHYWINAPI cuphySetWorkCancelKernelNodeParams(CUDA_KERNEL_NODE_PARAMS* pNodeParams, void** pKernelParams, uint8_t device_graph_launch)
{
    return internalCuphySetWorkCancelKernelNodeParams(pNodeParams, pKernelParams, device_graph_launch);
}



////////////////////////////////////////////////////////////////////////
// cuphyCreateRandomNumberGenerator()
cuphyStatus_t CUPHYWINAPI cuphyCreateRandomNumberGenerator(cuphyRNG_t*        pRNG,
                                                           unsigned long long seed,
                                                           unsigned int       flags,
                                                           cudaStream_t       strm)
{
    if(!pRNG)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    *pRNG = nullptr;
    try
    {
        cuphy_i::rng* r = new cuphy_i::rng(seed, strm);
        *pRNG           = static_cast<cuphyRNG_t>(r);
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
// cuphyDestroyRandomNumberGenerator()
cuphyStatus_t CUPHYWINAPI cuphyDestroyRandomNumberGenerator(cuphyRNG_t rng)
{
    if(!rng)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    cuphy_i::rng* r = static_cast<cuphy_i::rng*>(rng);
    delete r;
    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphyRandomUniform()
cuphyStatus_t CUPHYWINAPI cuphyRandomUniform(cuphyRNG_t              rng,
                                             cuphyTensorDescriptor_t tDst,
                                             void*                   pDst,
                                             const cuphyVariant_t*   minValue,
                                             const cuphyVariant_t*   maxValue,
                                             cudaStream_t            strm)
{
    if(!rng ||
       !tDst ||
       !pDst ||
       !minValue ||
       !maxValue)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    tensor_desc&  tDesc = static_cast<tensor_desc&>(*tDst);
    cuphy_i::rng& r     = static_cast<cuphy_i::rng&>(*rng);
    return r.uniform(tDesc, pDst, *minValue, *maxValue, strm);
}

////////////////////////////////////////////////////////////////////////
// cuphyRandomNormal()
cuphyStatus_t CUPHYWINAPI cuphyRandomNormal(cuphyRNG_t              rng,
                                            cuphyTensorDescriptor_t tDst,
                                            void*                   pDst,
                                            const cuphyVariant_t*   mean,
                                            const cuphyVariant_t*   stddev,
                                            cudaStream_t            strm)
{
    if(!rng ||
       !tDst ||
       !pDst ||
       !mean ||
       !stddev)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    tensor_desc&  tDesc = static_cast<tensor_desc&>(*tDst);
    cuphy_i::rng& r     = static_cast<cuphy_i::rng&>(*rng);
    return r.normal(tDesc, pDst, *mean, *stddev, strm);
}

////////////////////////////////////////////////////////////////////////
// cuphyConvertVariant()
cuphyStatus_t CUPHYWINAPI cuphyConvertVariant(cuphyVariant_t* v,
                                              cuphyDataType_t t)
{
    if(!v ||
       (CUPHY_VOID == t))
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    return cuphy_i::convert_variant(*v, t);
}

////////////////////////////////////////////////////////////////////////
// cuphyFillTensor()
cuphyStatus_t CUPHYWINAPI cuphyFillTensor(cuphyTensorDescriptor_t tDst,
                                          void*                   pDst,
                                          const cuphyVariant_t*   v,
                                          cudaStream_t            strm)
{
    if(!tDst || !pDst || !v || (CUPHY_VOID == v->type))
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    return cuphy_i::tensor_fill(static_cast<tensor_desc&>(*tDst),
                                pDst,
                                *v,
                                strm);
}

////////////////////////////////////////////////////////////////////////
// cuphyTileTensor()
cuphyStatus_t CUPHYWINAPI cuphyTileTensor(cuphyTensorDescriptor_t tDst,
                                          void*                   pDst,
                                          cuphyTensorDescriptor_t tSrc,
                                          const void*             pSrc,
                                          int                     tileRank,
                                          const int*              tileExtents,
                                          cudaStream_t            strm)
{
    //------------------------------------------------------------------
    if(!tDst || !pDst || !tSrc || !pSrc || (0 == tileRank) || !tileExtents || (tileRank > CUPHY_DIM_MAX))
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    //------------------------------------------------------------------
    // Types must match (to avoid having an exponential number of
    // conversion tiling kernels).
    const tensor_desc& descDst = static_cast<tensor_desc&>(*tDst);
    const tensor_desc& descSrc = static_cast<tensor_desc&>(*tSrc);
    if(descDst.type() != descSrc.type())
    {
        return CUPHY_STATUS_UNSUPPORTED_TYPE;
    }
    //------------------------------------------------------------------
    return cuphy_i::tensor_tile(descDst,
                                pDst,
                                descSrc,
                                pSrc,
                                tileRank,
                                tileExtents,
                                strm);
}

////////////////////////////////////////////////////////////////////////
// cuphyTensorElementWiseOperation()
cuphyStatus_t CUPHYWINAPI cuphyTensorElementWiseOperation(cuphyTensorDescriptor_t tDst,
                                                          void*                   pDst,
                                                          cuphyTensorDescriptor_t tSrcA,
                                                          const void*             pSrcA,
                                                          const cuphyVariant_t*   alpha,
                                                          cuphyTensorDescriptor_t tSrcB,
                                                          const void*             pSrcB,
                                                          const cuphyVariant_t*   beta,
                                                          cuphyElementWiseOp_t    elemOp,
                                                          cudaStream_t            strm)
{
    //------------------------------------------------------------------
    // Note that input B is optional for some operations, but the
    // destination and input A are required.
    if(!tDst || !pDst || !tSrcA || !pSrcA)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    // If there is a descriptor for B, there must also be an address
    if((nullptr == tSrcB) != (nullptr == pSrcB))
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    const tensor_desc& descDst  = static_cast<const tensor_desc&>(*tDst);
    const tensor_desc& descSrcA = static_cast<const tensor_desc&>(*tSrcA);
    return cuphy_i::tensor_elementwise(descDst,
                                       pDst,
                                       descSrcA,
                                       pSrcA,
                                       alpha,
                                       tSrcB,
                                       pSrcB,
                                       beta,
                                       elemOp,
                                       strm);
}

////////////////////////////////////////////////////////////////////////
// cuphyTensorReduction()
cuphyStatus_t CUPHYWINAPI cuphyTensorReduction(cuphyTensorDescriptor_t tDst,
                                               void*                   pDst,
                                               cuphyTensorDescriptor_t tSrc,
                                               const void*             pSrc,
                                               cuphyReductionOp_t      redOp,
                                               int                     dim,
                                               size_t                  workspaceSize,
                                               void*                   workspace,
                                               cudaStream_t            strm)
{
    if(!tDst || !pDst || !tSrc || !pSrc)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    if((dim < 0) || (dim >= CUPHY_DIM_MAX))
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    const tensor_desc& descDst = static_cast<const tensor_desc&>(*tDst);
    const tensor_desc& descSrc = static_cast<const tensor_desc&>(*tSrc);
    return cuphy_i::tensor_reduction(descDst,
                                     pDst,
                                     descSrc,
                                     pSrc,
                                     redOp,
                                     dim,
                                     workspaceSize,
                                     workspace,
                                     strm);
}

////////////////////////////////////////////////////////////////////////
// cuphyCompCwTreeTypesGetDescrInfo()

cuphyStatus_t CUPHYWINAPI cuphyCompCwTreeTypesGetDescrInfo(size_t* pDynDescrSizeBytes, size_t* pDynDescrAlignBytes)
{
    if(!pDynDescrSizeBytes || !pDynDescrAlignBytes)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    compCwTreeTypes::getDescrInfo(*pDynDescrSizeBytes, *pDynDescrAlignBytes);
    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphyCreateCompCwTreeTypes()

cuphyStatus_t CUPHYWINAPI cuphyCreateCompCwTreeTypes(cuphyCompCwTreeTypesHndl_t* pCompCwTreeTypesHndl)
{
    if(!pCompCwTreeTypesHndl)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    *pCompCwTreeTypesHndl = nullptr;
    try
    {
        compCwTreeTypes* pCompCwTreeTypes = new compCwTreeTypes;
        *pCompCwTreeTypesHndl             = static_cast<cuphyCompCwTreeTypesHndl_t>(pCompCwTreeTypes);
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
// cuphySetupCompCwTreeTypes()
cuphyStatus_t CUPHYWINAPI cuphySetupCompCwTreeTypes(cuphyCompCwTreeTypesHndl_t       compCwTreeTypesHndl,
                                                    uint16_t                         nPolUciSegs,
                                                    const cuphyPolarUciSegPrm_t*     pPolUciSegPrmsCpu,
                                                    const cuphyPolarUciSegPrm_t*     pPolUciSegPrmsGpu,
                                                    uint8_t**                        pCwTreeTypesAddrs,
                                                    void*                            pCpuDynDescCompTree,
                                                    void*                            pGpuDynDescCompTree,
                                                    void*                            pCpuDynDescCompTreeAddrs,
                                                    uint8_t                          enableCpuToGpuDescrAsyncCpy,
                                                    cuphyCompCwTreeTypesLaunchCfg_t* pLaunchCfg,
                                                    cudaStream_t                     strm)
{
    if(!compCwTreeTypesHndl || !pPolUciSegPrmsCpu || !pPolUciSegPrmsGpu || !pCpuDynDescCompTree || !pGpuDynDescCompTree || !pLaunchCfg)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    compCwTreeTypes* pCompCwTreeTypes = static_cast<compCwTreeTypes*>(compCwTreeTypesHndl);

    auto pCpuDynDesc = static_cast<compCwTreeTypesDynDescr_t*>(pCpuDynDescCompTree);
    pCpuDynDesc->pCwTreeTypesAddrs = static_cast<uint8_t**> (pCpuDynDescCompTreeAddrs);

    pCompCwTreeTypes->setup(nPolUciSegs,
                            pPolUciSegPrmsCpu,
                            pPolUciSegPrmsGpu,
                            pCwTreeTypesAddrs,
                            pCpuDynDesc,
                            pGpuDynDescCompTree,
                            enableCpuToGpuDescrAsyncCpy,
                            pLaunchCfg,
                            strm);

    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphyDestroyCompCwTreeTypes()
cuphyStatus_t CUPHYWINAPI cuphyDestroyCompCwTreeTypes(cuphyCompCwTreeTypesHndl_t compCwTreeTypesHndl)
{
    if(!compCwTreeTypesHndl)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    compCwTreeTypes* pCompCwTreeTypes = static_cast<compCwTreeTypes*>(compCwTreeTypesHndl);
    delete pCompCwTreeTypes;
    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphyPolSegDeRmDeItlDescrInfo()

cuphyStatus_t CUPHYWINAPI cuphyPolSegDeRmDeItlGetDescrInfo(size_t* pDynDescrSizeBytes, size_t* pDynDescrAlignBytes)
{
    if(!pDynDescrSizeBytes || !pDynDescrAlignBytes)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    polSegDeRmDeItl::getDescrInfo(*pDynDescrSizeBytes, *pDynDescrAlignBytes);
    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphyCreatePolSegDeRmDeItl()

cuphyStatus_t CUPHYWINAPI cuphyCreatePolSegDeRmDeItl(cuphyPolSegDeRmDeItlHndl_t* pPolSegDeRmDeItlHndl)
{
    if(!pPolSegDeRmDeItlHndl)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    *pPolSegDeRmDeItlHndl = nullptr;
    try
    {
        polSegDeRmDeItl* pPolSegDeRmDeItl = new polSegDeRmDeItl;
        *pPolSegDeRmDeItlHndl             = static_cast<cuphyPolSegDeRmDeItlHndl_t>(pPolSegDeRmDeItl);
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
// cuphySetupPolSegDeRmDeItl()
cuphyStatus_t CUPHYWINAPI cuphySetupPolSegDeRmDeItl(cuphyPolSegDeRmDeItlHndl_t       polSegDeRmDeItlHndl,
                                                    uint16_t                         nPolUciSegs,
                                                    uint16_t                         nPolCws,
                                                    const cuphyPolarUciSegPrm_t*     pPolUciSegPrmsCpu,
                                                    const cuphyPolarUciSegPrm_t*     pPolUciSegPrmsGpu,
                                                    const cuphyPolarCwPrm_t*         pPolCwPrmsCpu,
                                                    const cuphyPolarCwPrm_t*         pPolCwPrmsGpu,
                                                    __half**                         pUciSegLLRsAddrs,
                                                    __half**                         pCwLLRsAddrs,
                                                    void*                            pCpuDynDescDrDi,
                                                    void*                            pGpuDynDescDrDi,
                                                    void*                            pCpuDynDescDrDiCwAddrs,
                                                    void*                            pCpuDynDescDrDiUciAddrs,
                                                    uint8_t                          enableCpuToGpuDescrAsyncCpy,
                                                    cuphyPolSegDeRmDeItlLaunchCfg_t* pLaunchCfg,
                                                    cudaStream_t                     strm)
{
    if(!polSegDeRmDeItlHndl || !pPolUciSegPrmsCpu || !pPolCwPrmsCpu || !pPolUciSegPrmsGpu || !pPolUciSegPrmsGpu || !pUciSegLLRsAddrs || !pCwLLRsAddrs || !pCpuDynDescDrDi || !pGpuDynDescDrDi || !pLaunchCfg)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    polSegDeRmDeItl* pPolSegDeRmDeItl = static_cast<polSegDeRmDeItl*>(polSegDeRmDeItlHndl);

    auto pCpuDynDesc              = static_cast<polSegDeRmDeItlDynDescr_t*>(pCpuDynDescDrDi);
    pCpuDynDesc->pCwLLRsAddrs     = static_cast<__half**>(pCpuDynDescDrDiCwAddrs);
    pCpuDynDesc->pUciSegLLRsAddrs = static_cast<__half**>(pCpuDynDescDrDiUciAddrs);

    pPolSegDeRmDeItl->setup(nPolUciSegs,
                            nPolCws,
                            pPolUciSegPrmsCpu,
                            pPolUciSegPrmsGpu,
                            pPolCwPrmsCpu,
                            pPolCwPrmsGpu,
                            pUciSegLLRsAddrs,
                            pCwLLRsAddrs,
                            pCpuDynDesc,
                            pGpuDynDescDrDi,
                            enableCpuToGpuDescrAsyncCpy,
                            pLaunchCfg,
                            strm);

    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphyDestroyPolSegDeRmDeItl()
cuphyStatus_t CUPHYWINAPI cuphyDestroyPolSegDeRmDeItl(cuphyPolSegDeRmDeItlHndl_t polSegDeRmDeItlHndl)
{
    if(!polSegDeRmDeItlHndl)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    polSegDeRmDeItl* pPolSegDeRmDeItl = static_cast<polSegDeRmDeItl*>(polSegDeRmDeItlHndl);
    delete pPolSegDeRmDeItl;
    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphyUciOnPuschSegLLrs1DescrInfo()

cuphyStatus_t CUPHYWINAPI cuphyUciOnPuschSegLLRs1GetDescrInfo(size_t* pDynDescrSizeBytes, size_t* pDynDescrAlignBytes)
{
    if(!pDynDescrSizeBytes || !pDynDescrAlignBytes)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    uciOnPuschSegLLRs1::getDescrInfo(*pDynDescrSizeBytes, *pDynDescrAlignBytes);
    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphyCreateUciOnPuschSegLLRs1()

cuphyStatus_t CUPHYWINAPI cuphyCreateUciOnPuschSegLLRs1(cuphyUciOnPuschSegLLRs1Hndl_t* pUciOnPuschSegLLRs1Hndl)
{
    if(!pUciOnPuschSegLLRs1Hndl)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    *pUciOnPuschSegLLRs1Hndl = nullptr;
    try
    {
        uciOnPuschSegLLRs1* pUciOnPuschSegLLRs1 = new uciOnPuschSegLLRs1;
        *pUciOnPuschSegLLRs1Hndl                = static_cast<cuphyUciOnPuschSegLLRs1Hndl_t>(pUciOnPuschSegLLRs1);
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
// cuphySetupUciOnPuschSegLLRs1()

cuphyStatus_t CUPHYWINAPI cuphySetupUciOnPuschSegLLRs1(cuphyUciOnPuschSegLLRs1Hndl_t       uciOnPuschSegLLRs1Hndl,
                                                       uint16_t                            nUciUes,
                                                       uint16_t*                           pUciUserIdxs,
                                                       PerTbParams*                        pTbPrmsCpu,
                                                       PerTbParams*                        pTbPrmsGpu,
                                                       uint16_t                            nUeGrps,
                                                       cuphyTensorPrm_t*                   pTensorPrmsEqOutLLRs,
                                                       uint16_t*                           pNumPrbs,
                                                       uint8_t                             startSym,
                                                       uint8_t                             nPuschSym,
                                                       uint8_t                             nPuschDataSym,
                                                       uint8_t*                            pDataSymIdxs,
                                                       uint8_t                             nPuschDmrsSym,
                                                       uint8_t*                            pDmrsSymIdxs,
                                                       void*                               pCpuDynDesc,
                                                       void*                               pGpuDynDesc,
                                                       uint8_t                             enableCpuToGpuDescrAsyncCpy,
                                                       cuphyUciOnPuschSegLLRs1LaunchCfg_t* pLaunchCfg,
                                                       cudaStream_t                        strm)
{
    if(!uciOnPuschSegLLRs1Hndl || !pUciUserIdxs || !pTbPrmsCpu || !pTbPrmsGpu || !pTensorPrmsEqOutLLRs || !pNumPrbs || !pDataSymIdxs || !pDmrsSymIdxs || !pCpuDynDesc || !pGpuDynDesc || !pLaunchCfg)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    // call c++ setup function
    uciOnPuschSegLLRs1* pUciOnPuschSegLLRs1 = static_cast<uciOnPuschSegLLRs1*>(uciOnPuschSegLLRs1Hndl);

    pUciOnPuschSegLLRs1->setup(uciOnPuschSegLLRs1Hndl,
                               nUciUes,
                               pUciUserIdxs,
                               pTbPrmsCpu,
                               pTbPrmsGpu,
                               nUeGrps,
                               pTensorPrmsEqOutLLRs,
                               pNumPrbs,
                               startSym,
                               nPuschSym,
                               nPuschDataSym,
                               pDataSymIdxs,
                               nPuschDmrsSym,
                               pDmrsSymIdxs,
                               static_cast<uciOnPuschSegLLRs1DynDescr_t*>(pCpuDynDesc),
                               pGpuDynDesc,
                               enableCpuToGpuDescrAsyncCpy,
                               pLaunchCfg,
                               strm);

    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphyDestroyUciOnPuschSegLLRs1()
cuphyStatus_t CUPHYWINAPI cuphyDestroyUciOnPuschSegLLRs1(cuphyUciOnPuschSegLLRs1Hndl_t uciOnPuschSegLLRs1Hndl)
{
    if(!uciOnPuschSegLLRs1Hndl)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    uciOnPuschSegLLRs1* pUciOnPuschSegLLRs1 = static_cast<uciOnPuschSegLLRs1*>(uciOnPuschSegLLRs1Hndl);
    delete pUciOnPuschSegLLRs1;
    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphyUciPolarDecoderGetDescrInfo()

cuphyStatus_t CUPHYWINAPI cuphyPolarDecoderGetDescrInfo(size_t* pDynDescrSizeBytes, size_t* pDynDescrAlignBytes)
{
    if(!pDynDescrSizeBytes || !pDynDescrAlignBytes)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    polarDecoder::getDescrInfo(*pDynDescrSizeBytes, *pDynDescrAlignBytes);
    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphyCreatePolarDecoder()

cuphyStatus_t CUPHYWINAPI cuphyCreatePolarDecoder(cuphyPolarDecoderHndl_t* pPolarDecoderHndl)
{
    if(!pPolarDecoderHndl)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    *pPolarDecoderHndl = nullptr;
    try
    {
        polarDecoder* pPolarDecoder = new polarDecoder;
        *pPolarDecoderHndl          = static_cast<cuphyPolarDecoderHndl_t>(pPolarDecoder);
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
// cuphySetupPolarDecoder()

cuphyStatus_t CUPHYWINAPI cuphySetupPolarDecoder(cuphyPolarDecoderHndl_t       polarDecoderHndl,
                                                 uint16_t                      nPolCws,
                                                 __half**                      pCwTreeLLRsAddrs,
                                                 cuphyPolarCwPrm_t*            pCwPrmsGpu,
                                                 cuphyPolarCwPrm_t*            pCwPrmsCpu,
                                                 uint32_t**                    pPolCbEstAddrs,
                                                 bool**                        pListPolScratchAddrs,
                                                 uint8_t                       nPolarList,
                                                 uint8_t*                      pPolCrcErrorFlags,
                                                 bool                          enableCpuToGpuDescrAsyncCpy,
                                                 void*                         pCpuDynDescPolar,
                                                 void*                         pGpuDynDescPolar,
                                                 void*                         pCpuDynDescPolarLLRAddrs,
                                                 void*                         pCpuDynDescPolarCBAddrs,
                                                 void*                         pCpuDynDescListPolarScratchAddrs,
                                                 cuphyPolarDecoderLaunchCfg_t* pLaunchCfg,
                                                 cudaStream_t                  strm)
{
    if(!pCwTreeLLRsAddrs || !pCwPrmsGpu || !pCwPrmsCpu || !pPolCbEstAddrs || !pPolCrcErrorFlags || !pCpuDynDescPolar || !pGpuDynDescPolar || !pLaunchCfg)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    // call c++ setup function
    polarDecoder* pPolarDecoder = static_cast<polarDecoder*>(polarDecoderHndl);

    auto pCpuDynDesc                 = static_cast<polarDecoderDynDescr_t*>(pCpuDynDescPolar);
    pCpuDynDesc->cwTreeLLRsAddrs     = static_cast<__half**>(pCpuDynDescPolarLLRAddrs);
    pCpuDynDesc->polCbEstAddrs       = static_cast<uint32_t**>(pCpuDynDescPolarCBAddrs);
    pCpuDynDesc->listPolScratchAddrs = static_cast<bool**>(pCpuDynDescListPolarScratchAddrs);

    pPolarDecoder->setup(nPolCws,
                         pCwTreeLLRsAddrs,
                         pCwPrmsGpu,
                         pCwPrmsCpu,
                         pPolCbEstAddrs,
                         pListPolScratchAddrs,
                         nPolarList,
                         pPolCrcErrorFlags,
                         static_cast<bool>(enableCpuToGpuDescrAsyncCpy),
                         pCpuDynDesc,
                         pGpuDynDescPolar,
                         pLaunchCfg,
                         strm);

    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphyDestroyPolarDecoder()
cuphyStatus_t CUPHYWINAPI cuphyDestroyPolarDecoder(cuphyPolarDecoderHndl_t polarDecoderHndl)
{
    if(!polarDecoderHndl)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    polarDecoder* pPolarDecoder = static_cast<polarDecoder*>(polarDecoderHndl);
    delete pPolarDecoder;
    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphyUciOnPuschSegLLrs2DescrInfo()

cuphyStatus_t CUPHYWINAPI cuphyUciOnPuschSegLLRs2GetDescrInfo(size_t* pDynDescrSizeBytes, size_t* pDynDescrAlignBytes)
{
    if(!pDynDescrSizeBytes || !pDynDescrAlignBytes)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    uciOnPuschSegLLRs2::getDescrInfo(*pDynDescrSizeBytes, *pDynDescrAlignBytes);
    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphyCreateUciOnPuschSegLLRs2()

cuphyStatus_t CUPHYWINAPI cuphyCreateUciOnPuschSegLLRs2(cuphyUciOnPuschSegLLRs2Hndl_t* pUciOnPuschSegLLRs2Hndl)
{
    if(!pUciOnPuschSegLLRs2Hndl)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    *pUciOnPuschSegLLRs2Hndl = nullptr;
    try
    {
        uciOnPuschSegLLRs2* pUciOnPuschSegLLRs2 = new uciOnPuschSegLLRs2;
        *pUciOnPuschSegLLRs2Hndl                = static_cast<cuphyUciOnPuschSegLLRs2Hndl_t>(pUciOnPuschSegLLRs2);
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
// // cuphySetupUciOnPuschSegLLRs2()

cuphyStatus_t CUPHYWINAPI cuphySetupUciOnPuschSegLLRs2(cuphyUciOnPuschSegLLRs2Hndl_t       uciOnPuschSegLLRs2Hndl,
                                                       uint16_t                            nCsi2Ues,
                                                       uint16_t*                           pCsi2UeIdxs,
                                                       PerTbParams*                        pTbPrmsCpu,
                                                       PerTbParams*                        pTbPrmsGpu,
                                                       uint16_t                            nUeGrps,
                                                       cuphyTensorPrm_t*                   pTensorPrmsEqOutLLRs,
                                                       cuphyPuschRxUeGrpPrms_t*            pUeGrpPrmsCpu,
                                                       cuphyPuschRxUeGrpPrms_t*            pUeGrpPrmsGpu,
                                                       void*                               pCpuDynDesc,
                                                       void*                               pGpuDynDesc,
                                                       uint8_t                             enableCpuToGpuDescrAsyncCpy,
                                                       cuphyUciOnPuschSegLLRs2LaunchCfg_t* pLaunchCfg,
                                                       cudaStream_t                        strm)
{
    if(!uciOnPuschSegLLRs2Hndl || !pCsi2UeIdxs || !pTbPrmsCpu || !pTbPrmsGpu || !pTensorPrmsEqOutLLRs || !pCpuDynDesc || !pGpuDynDesc || !pLaunchCfg)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    // call c++ setup function
    uciOnPuschSegLLRs2* pUciOnPuschSegLLRs2 = static_cast<uciOnPuschSegLLRs2*>(uciOnPuschSegLLRs2Hndl);

    pUciOnPuschSegLLRs2->setup(nCsi2Ues,
                               pCsi2UeIdxs,
                               pTbPrmsCpu,
                               pTbPrmsGpu,
                               nUeGrps,
                               pTensorPrmsEqOutLLRs,
                               pUeGrpPrmsCpu,
                               pUeGrpPrmsGpu,
                               static_cast<uciOnPuschSegLLRs2DynDescr_t*>(pCpuDynDesc),
                               pGpuDynDesc,
                               enableCpuToGpuDescrAsyncCpy,
                               pLaunchCfg,
                               strm);

    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////

// cuphyDestroyUciOnPuschSegLLRs2()
cuphyStatus_t CUPHYWINAPI cuphyDestroyUciOnPuschSegLLRs2(cuphyUciOnPuschSegLLRs2Hndl_t uciOnPuschSegLLRs2Hndl)
{
    if(!uciOnPuschSegLLRs2Hndl)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    uciOnPuschSegLLRs2* pUciOnPuschSegLLRs2 = static_cast<uciOnPuschSegLLRs2*>(uciOnPuschSegLLRs2Hndl);
    delete pUciOnPuschSegLLRs2;
    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphyUciOnPuschSegLLrs0DescrInfo()

cuphyStatus_t CUPHYWINAPI cuphyUciOnPuschSegLLRs0GetDescrInfo(size_t* pDynDescrSizeBytes, size_t* pDynDescrAlignBytes)
{
    if(!pDynDescrSizeBytes || !pDynDescrAlignBytes)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    uciOnPuschSegLLRs0::getDescrInfo(*pDynDescrSizeBytes, *pDynDescrAlignBytes);
    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphyCreateUciOnPuschSegLLRs0()

cuphyStatus_t CUPHYWINAPI cuphyCreateUciOnPuschSegLLRs0(cuphyUciOnPuschSegLLRs0Hndl_t* pUciOnPuschSegLLRs0Hndl)
{
    if(!pUciOnPuschSegLLRs0Hndl)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    *pUciOnPuschSegLLRs0Hndl = nullptr;
    try
    {
        uciOnPuschSegLLRs0* pUciOnPuschSegLLRs0 = new uciOnPuschSegLLRs0;
        *pUciOnPuschSegLLRs0Hndl                = static_cast<cuphyUciOnPuschSegLLRs0Hndl_t>(pUciOnPuschSegLLRs0);
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
// cuphySetupUciOnPuschSegLLrs0()

cuphyStatus_t CUPHYWINAPI cuphySetupUciOnPuschSegLLRs0(cuphyUciOnPuschSegLLRs0Hndl_t       uciOnPuschSegLLRs0Hndl,
                                                       uint16_t                            nUciUes,
                                                       uint16_t*                           pUciUeIdxs,
                                                       PerTbParams*                        pTbPrmsCpu,
                                                       PerTbParams*                        pTbPrmsGpu,
                                                       uint16_t                            nUeGrps,
                                                       cuphyTensorPrm_t*                   pTensorPrmsEqOutLLRs,
                                                       cuphyPuschRxUeGrpPrms_t*            pUeGrpPrmsCpu,
                                                       cuphyPuschRxUeGrpPrms_t*            pUeGrpPrmsGpu,
                                                       cuphyUciToSeg_t                     uciToSeg,
                                                       void*                               pCpuDynDesc,
                                                       void*                               pGpuDynDesc,
                                                       uint8_t                             enableCpuToGpuDescrAsyncCpy,
                                                       cuphyUciOnPuschSegLLRs0LaunchCfg_t* pLaunchCfg,
                                                       cudaStream_t                        strm)
{
    if(!uciOnPuschSegLLRs0Hndl || !pUciUeIdxs || !pTbPrmsCpu || !pTbPrmsGpu || !pTensorPrmsEqOutLLRs || !pUeGrpPrmsCpu || !pUeGrpPrmsGpu || !pCpuDynDesc || !pGpuDynDesc || !pLaunchCfg)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    // call c++ setup function
    uciOnPuschSegLLRs0* pUciOnPuschSegLLRs0 = static_cast<uciOnPuschSegLLRs0*>(uciOnPuschSegLLRs0Hndl);

    pUciOnPuschSegLLRs0->setup(nUciUes,
                               pUciUeIdxs,
                               pTbPrmsCpu,
                               pTbPrmsGpu,
                               nUeGrps,
                               pTensorPrmsEqOutLLRs,
                               pUeGrpPrmsCpu,
                               pUeGrpPrmsGpu,
                               uciToSeg,
                               static_cast<uciOnPuschSegLLRs0DynDescr_t*>(pCpuDynDesc),
                               pGpuDynDesc,
                               enableCpuToGpuDescrAsyncCpy,
                               pLaunchCfg,
                               strm);

    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphyDestroyUciOnPuschSegLLRs0()

cuphyStatus_t CUPHYWINAPI cuphyDestroyUciOnPuschSegLLRs0(cuphyUciOnPuschSegLLRs0Hndl_t uciOnPuschSegLLRs0Hndl)
{
    if(!uciOnPuschSegLLRs0Hndl)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    uciOnPuschSegLLRs0* pUciOnPuschSegLLRs0 = static_cast<uciOnPuschSegLLRs0*>(uciOnPuschSegLLRs0Hndl);
    delete pUciOnPuschSegLLRs0;
    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphyUciOnPuschCsi2CtrlDescrInfo()

cuphyStatus_t CUPHYWINAPI cuphyUciOnPuschCsi2CtrlGetDescrInfo(size_t* pDynDescrSizeBytes, size_t* pDynDescrAlignBytes)
{
    if(!pDynDescrSizeBytes || !pDynDescrAlignBytes)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    uciOnPuschCsi2Ctrl::getDescrInfo(*pDynDescrSizeBytes, *pDynDescrAlignBytes);
    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphyCreateUciOnPuschCsi2Ctrl()

cuphyStatus_t CUPHYWINAPI cuphyCreateUciOnPuschCsi2Ctrl(cuphyUciOnPuschCsi2CtrlHndl_t* pUciOnPuschCsi2CtrlHndl)
{
    if(!pUciOnPuschCsi2CtrlHndl)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    *pUciOnPuschCsi2CtrlHndl = nullptr;
    try
    {
        uciOnPuschCsi2Ctrl* pUciOnPuschCsi2Ctrl = new uciOnPuschCsi2Ctrl;
        *pUciOnPuschCsi2CtrlHndl                = static_cast<cuphyUciOnPuschCsi2CtrlHndl_t>(pUciOnPuschCsi2Ctrl);
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
// // cuphySetupUciOnPuschCsi2Ctrl()

cuphyStatus_t CUPHYWINAPI cuphySetupUciOnPuschCsi2Ctrl(cuphyUciOnPuschCsi2CtrlHndl_t       uciOnPuschCsi2CtrlHndl,
                                                       uint16_t                            nCsi2Ues,
                                                       uint16_t*                           pCsi2UeIdxsCpu,
                                                       PerTbParams*                        pTbPrmsCpu,
                                                       PerTbParams*                        pTbPrmsGpu,
                                                       cuphyPuschRxUeGrpPrms_t*            pUeGrpPrmsCpu,
                                                       cuphyPuschCellStatPrm_t*            pCellStatPrmsGpu,
                                                       cuphyUciOnPuschOutOffsets_t*        pUciOnPuschOutOffsetsCpu,
                                                       uint8_t*                            pUciPayloadsGpu,
                                                       uint16_t*                           pNumCsi2BitsGpu,
                                                       cuphyPolarUciSegPrm_t*              pCsi2PolarSegPrmsGpu,
                                                       cuphyPolarCwPrm_t*                  pCsi2PolarCwPrmsGpu,
                                                       cuphyRmCwPrm_t*                     pCsi2RmCwPrmsGpu,
                                                       cuphySimplexCwPrm_t*                pCsi2SpxCwPrmsGpu,
                                                       uint16_t                            forcedNumCsi2Bits,
                                                       uint8_t                             enableCsiP2Fapiv3,
                                                       void*                               pCpuDynDesc,
                                                       void*                               pGpuDynDesc,
                                                       uint8_t                             enableCpuToGpuDescrAsyncCpy,
                                                       cuphyUciOnPuschCsi2CtrlLaunchCfg_t* pLaunchCfg,
                                                       cudaStream_t                        strm)
{
    if(!uciOnPuschCsi2CtrlHndl || !pCsi2UeIdxsCpu || !pTbPrmsCpu || !pTbPrmsGpu || !pUeGrpPrmsCpu || !pCellStatPrmsGpu || !pUciOnPuschOutOffsetsCpu || !pUciPayloadsGpu || !pNumCsi2BitsGpu || !pCsi2PolarSegPrmsGpu)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    // call c++ setup function
    uciOnPuschCsi2Ctrl* pUciOnPuschCsi2Ctrl = static_cast<uciOnPuschCsi2Ctrl*>(uciOnPuschCsi2CtrlHndl);

    pUciOnPuschCsi2Ctrl->setup(nCsi2Ues,
                               pCsi2UeIdxsCpu,
                               pTbPrmsCpu,
                               pTbPrmsGpu,
                               pUeGrpPrmsCpu,
                               pCellStatPrmsGpu,
                               pUciOnPuschOutOffsetsCpu,
                               pUciPayloadsGpu,
                               pNumCsi2BitsGpu,
                               pCsi2PolarSegPrmsGpu,
                               pCsi2PolarCwPrmsGpu,
                               pCsi2RmCwPrmsGpu,
                               pCsi2SpxCwPrmsGpu,
                               forcedNumCsi2Bits,
                               enableCsiP2Fapiv3,
                               static_cast<uciOnPuschCsi2CtrlDynDescr_t*>(pCpuDynDesc),
                               pGpuDynDesc,
                               static_cast<bool>(enableCpuToGpuDescrAsyncCpy),
                               pLaunchCfg,
                               strm);

    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphyDestroyUciOnPuschCsi2Ctrl()

cuphyStatus_t CUPHYWINAPI cuphyDestroyUciOnPuschCsi2Ctrl(cuphyUciOnPuschCsi2CtrlHndl_t uciOnPuschCsi2CtrlHndl)
{
    if(!uciOnPuschCsi2CtrlHndl)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    uciOnPuschCsi2Ctrl* pUciOnPuschCsi2Ctrl = static_cast<uciOnPuschCsi2Ctrl*>(uciOnPuschCsi2CtrlHndl);
    delete pUciOnPuschCsi2Ctrl;
    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphySrsChEstGetDescrInfo()

cuphyStatus_t CUPHYWINAPI cuphySrsChEstGetDescrInfo(size_t* pStatDescrSizeBytes, size_t* pStatDescrAlignBytes, size_t* pDynDescrSizeBytes, size_t* pDynDescrAlignBytes)
{
    if(!pDynDescrSizeBytes || !pDynDescrAlignBytes)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    srsChEst::getDescrInfo(*pStatDescrSizeBytes, *pStatDescrAlignBytes, *pDynDescrSizeBytes, *pDynDescrAlignBytes);
    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphyCreateSrsChEst()

cuphyStatus_t CUPHYWINAPI cuphyCreateSrsChEst(cuphySrsChEstHndl_t*     pSrsChEstHndl,
                                               cuphySrsFilterPrms_t*   pSrsFilterPrms,
                                               cuphySrsRkhsPrms_t*     pRkhsPrms,
                                               cuphySrsChEstAlgoType_t chEstAlgo,
                                               uint8_t                 chEstToL2NormalizationAlgo,
                                               float                   chEstToL2ConstantScaler,
                                               uint8_t                 enableDelayOffsetCorrection,
                                               uint8_t                 enableCpuToGpuDescrAsyncCpy,
                                               void*                   pCpuStatDesc,
                                               void*                   pGpuStatDesc,
                                               cudaStream_t            strm)
{
    if(!pSrsChEstHndl)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    *pSrsChEstHndl = nullptr;
    try
    {
        srsChEst* pSrsChEst = new srsChEst;
        *pSrsChEstHndl       = static_cast<cuphySrsChEstHndl_t>(pSrsChEst);

        pSrsChEst->init( pSrsFilterPrms,
                         pRkhsPrms,
                         chEstAlgo,
                         chEstToL2NormalizationAlgo,
                         chEstToL2ConstantScaler,
                         enableDelayOffsetCorrection,
                         (0 != enableCpuToGpuDescrAsyncCpy) ? true : false,
                         static_cast<srsChEstStatDescr_t*>(pCpuStatDesc),
                         pGpuStatDesc,
                         strm);
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
// // cuphySetupSrsChEst()

cuphyStatus_t CUPHYWINAPI cuphySetupSrsChEst(   cuphySrsChEstHndl_t          srsChEstHndl,
                                                 uint16_t                      nSrsUes,
                                                 cuphyUeSrsPrm_t*              h_srsUePrms,
                                                 uint16_t                      nCell,
                                                 cuphyTensorPrm_t*             pTDataRx,
                                                 cuphySrsCellPrms_t*           h_srsCellPrms,
                                                 float*                        d_rbSnrBuff,
                                                 uint32_t*                     h_rbSnrBuffOffsets,
                                                 cuphySrsReport_t*             d_pSrsReports,
                                                 cuphySrsChEstBuffInfo_t*      h_chEstBuffInfo,
                                                 void**                        d_addrsChEstToL2InnerBuff,
                                                 void**                        d_addrsChEstToL2Buff,
                                                 cuphySrsChEstToL2_t*          h_chEstToL2,
                                                 void*                         d_workspace,
                                                 uint8_t                       enableCpuToGpuDescrAsyncCpy,
                                                 void*                         pCpuDynDesc,
                                                 void*                         pGpuDynDesc,
                                                 cuphySrsChEstLaunchCfg_t*     pLaunchCfg,
                                                 cuphySrsChEstNormalizationLaunchCfg_t* pNormalizationLaunchCfg,
                                                 cudaStream_t                  strm)
{
    if(!srsChEstHndl || !h_srsUePrms || !h_srsCellPrms || !d_rbSnrBuff || !h_rbSnrBuffOffsets || !d_addrsChEstToL2InnerBuff || !d_addrsChEstToL2Buff || !h_chEstToL2 || !d_pSrsReports || !h_chEstBuffInfo || !pCpuDynDesc || !pGpuDynDesc || !pLaunchCfg || !pNormalizationLaunchCfg)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    // call c++ setup function
    srsChEst* pSrsChEst = static_cast<srsChEst*>(srsChEstHndl);


    return pSrsChEst->setup(nSrsUes,
                            h_srsUePrms,
                            nCell,
                            pTDataRx,
                            h_srsCellPrms,
                            d_rbSnrBuff,
                            h_rbSnrBuffOffsets,
                            d_pSrsReports,
                            h_chEstBuffInfo,
                            d_addrsChEstToL2InnerBuff,
                            d_addrsChEstToL2Buff,
                            h_chEstToL2,
                            d_workspace,
                            static_cast<bool>(enableCpuToGpuDescrAsyncCpy),
                            static_cast<srsChEstDynDescr_t*>(pCpuDynDesc),
                            pGpuDynDesc,
                            pLaunchCfg,
                            pNormalizationLaunchCfg,
                            strm);

    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// cuphyDestroySrsChEst()

cuphyStatus_t CUPHYWINAPI cuphyDestroySrsChEst(cuphySrsChEstHndl_t srsChEstHndl)
{
    if(!srsChEstHndl)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    srsChEst* pSrsChEst = static_cast<srsChEst*>(srsChEstHndl);
    delete pSrsChEst;
    return CUPHY_STATUS_SUCCESS;
}
