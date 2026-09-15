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

#if !defined(DATASETS_HPP_INCLUDED_)
#define DATASETS_HPP_INCLUDED_

#include "cuphy_api.h"
#include "cuphy.hpp"
#include "cuphy.h"
#include "cuphy_hdf5.hpp"
#include "nvlog.hpp"

// structs holding max parameter values for memory allocation
// NOTE: default value for these parameters is zero, in which case
// memory allocation in PUSCH and PDSCH will use MAX supported values defined in cuphy.h
typedef struct
{
    uint32_t maxNTbs;
    uint32_t maxNCbs;
    uint32_t maxNPrbs;
    uint32_t maxNCbsPerTb;
} maxSCHPrms;

typedef struct : maxSCHPrms
{
    uint32_t maxNRx;
    uint32_t maxNCellsPerSlot;
} maxPUSCHPrms;

typedef struct : maxSCHPrms
{
    uint32_t maxNTx;
} maxPDSCHPrms;

//-------------------------------------------------------------------------------
// DynApiDataset
// Contains dynamic api parameters and data for a single PuschRxPipeline
struct DynApiDataset
{
public:
    // api C parameters:
    std::vector<cuphyPuschCellDynPrm_t> cellDynPrmVec;
    std::vector<cuphyPuschDmrsPrm_t>    dmrsPrmVec;
    std::vector<cuphyPuschUeGrpPrm_t>   ueGrpPrmsVec;
    std::vector<cuphyPuschUePrm_t>      uePrmsVec;
    std::vector<cuphyUciOnPuschPrm_t>   uciPrmsVec;
    std::vector<cuphyCalcCsi2SizePrm_t> calcCsi2PrmsVec;
    cuphyPuschCellGrpDynPrm_t           cellGrpDynPrm;
    cuphyPuschDynPrms_t                 puschDynPrm;
    cuphyPuschDynDbgPrms_t              dbgPrm;
    cuphyPuschDataOut_t                 DataOut;
    cuphyPuschDataInOut_t               DataInOut;
    cuphyPuschDataIn_t                  DataIn;
    cuphyPuschStatusOut_t               StatusOutput;
    
    uint8_t* evalHarqDetectionStatus;
    uint8_t* evalUciPayloads;
    uint8_t* evalUciCrcFlags;
    float*   evalPreEqSinrEhq;
    float*   evalRsrpEhq;
    float*   evalRssiEhq;
    
    uint32_t totNumUes;
    uint32_t nUciPayloadBytes;
    uint32_t nUciSegs;
    
    uint8_t* pPreEarlyHarqWaitKernelStatus;
    uint8_t* pPostEarlyHarqWaitKernelStatus;

    // input tensor parameters:
    std::vector<cuphyTensorPrm_t> tPrmDataRxVec;
    std::vector<cuphyTensorPrm_t> tPrmXVec;
    std::vector<cuphyTensorPrm_t> tPrmReeInvVec;

    DynApiDataset();
    DynApiDataset(const std::vector<std::string>& inputFileNameVec, cudaStream_t cuStrm, uint64_t procModeBmsk = 0, bool cpuCopyOn = false, uint32_t fp16Mode = 1, int apiTVflag = 0, int drmDebug = false, uint8_t openRanFunctionalSplitMode = PUSCH_7_2_A); // construct dataset from h5 file
    DynApiDataset(const DynApiDataset& dynApiDataset);
    ~DynApiDataset();
    DynApiDataset&        operator=(DynApiDataset&& dynApiDataset);
    void                  ResetPointers();
    void                  EasyAllocHarqBuffers(cudaStream_t strm);
    std::vector<uint32_t> bStartOffsetsTbPayloadDatasetsVec;

    const float getOutputTaEst(uint32_t taEstIdx) const { return outTaEsts.addr()[taEstIdx]; }
    const float* getOutputRssiPtr() const { return outRssi.addr(); }
    const float* getOutputRsrpPtr() const { return outRsrp.addr(); }
    const float* getOutputNoiseVarPreEqPtr() const { return outNoiseVarPreEq.addr(); }
    const float* getOutputNoiseVarPostEqPtr() const { return outNoiseVarPostEq.addr(); }
    const float* getOutputSinrPreEqPtr() const { return outSinrPreEq.addr(); }
    const float* getOutputSinrPostEqPtr() const { return outSinrPostEq.addr(); }
    const float* getOutputCfoHzPtr() const { return outCfoHz.addr(); }
    const float* getOutputToMicroSecPtr() const { return outTaEsts.addr(); }

    // buffers containing dynamic data
private:
    std::vector<cuphy::tensor_device>                               tDataRxVec, tXVec, tReeInvVec;
    cuphy::buffer<uint32_t, cuphy::pinned_alloc>                    bCbCrcs;
    cuphy::buffer<uint32_t, cuphy::pinned_alloc>                    bTbCrcs;
    cuphy::buffer<uint8_t, cuphy::pinned_alloc>                     bTbPayloads, bHarqDetectionStatus, bCsiP1DetectionStatus, bCsiP2DetectionStatus, bEvalHarqDetectionStatus;
    cuphy::buffer<float, cuphy::pinned_alloc>                       bEvalPreEqSinr;
    cuphy::buffer<float, cuphy::pinned_alloc>                       bEvalRsrp;
    cuphy::buffer<float, cuphy::pinned_alloc>                       bEvalRssi;
    cuphy::buffer<uint32_t, cuphy::pinned_alloc>                    bStartOffsetsCbCrc, bStartOffsetsTbCrc, bStartOffsetsTbPayload;
    cuphy::buffer<float, cuphy::pinned_alloc>                       outTaEsts, outRssi, outRsrp, outNoiseVarPreEq, outSinrPreEq, outNoiseVarPostEq, outSinrPostEq, outCfoHz;
    std::vector<std::vector<uint16_t>>                              ueGrpToUeIdxs; // Dim: nUeGrps x nUe
    std::vector<uint32_t>                                           bHarqBufferSizeInBytes;
    cuphy::buffer<uint8_t*, cuphy::pinned_alloc>                    bHarqBufferPtrs;
    cuphy::buffer<float*, cuphy::pinned_alloc>                      bFoCompensationBufferPtrs;
    std::vector<cuphy::buffer<uint8_t, cuphy::device_alloc>>        harqBuffers;
    cuphy::buffer<uint8_t, cuphy::pinned_alloc>                     bUciPayloads, bEvalUciPayloads;
    cuphy::buffer<uint8_t, cuphy::pinned_alloc>                     bUciCrcFlags, bEvalUciCrcFlags;
    cuphy::buffer<uint8_t, cuphy::pinned_alloc>                     bEarlyHarqPayloads;
    cuphy::buffer<uint8_t, cuphy::pinned_alloc>                     bEarlyHarqCrcFlags;
    //cuphy::buffer<uint8_t, cuphy::pinned_alloc>                     bUciDTXs;
    cuphy::buffer<uint16_t, cuphy::pinned_alloc>                    bNumCsi2Bits;
    cuphy::buffer<cuphyUciOnPuschOutOffsets_t, cuphy::pinned_alloc> bUciOnPuschOutOffsets;
};

//-------------------------------------------------------------------------------
// StaticApiDataset
// Contains static api parameters and data for a single PuschRxPipeline

struct StaticApiDataset
{
public:
    // api C parameters
    cuphyPuschStatPrms_t                 puschStatPrms{};
    cuphyTracker_t                       puschTracker{};
    cuphyPuschStatDbgPrms_t              dbgPrm{};
    std::vector<cuphyCellStatPrm_t>      cellStatPrmVec;
    std::vector<cuphyPuschCellStatPrm_t> puschCellStatPrmVec;

    StaticApiDataset();
    StaticApiDataset(const std::vector<std::string>& inputFileNameVec, cudaStream_t cuStrm, std::string outFileName = std::string(), int descramblingOn = 1,
                     int apiTVflag = 0, bool enableLdpcThroughputMode = false, const maxPUSCHPrms* puschPrms = nullptr,
                     cuphyPuschLdpcKernelLaunch_t ldpcLaunchMode = PUSCH_RX_LDPC_STREAM_POOL, uint8_t openRanFunctionalSplitMode = PUSCH_7_2_A, uint8_t kernelSelMode = PUSCH_ALL, uint8_t uciKernelSelMode = PUSCH_UCI_ALL, uint32_t delayUs = 0, uint32_t subSlotDelayUs = 0); // construct dataset from h5 file
    StaticApiDataset(const StaticApiDataset& staticApiDataset);
    void              ResetPointers();
    StaticApiDataset& operator=(StaticApiDataset&& staticApiDatatset);
    void              puschInitCellStatPrm(const std::vector<std::string>& inputFileNameVec, int apiTVflag, cudaStream_t cuStrm, const maxPUSCHPrms* puschPrms = nullptr);
    void              setWorkCancelValue(uint8_t work_cancellation);

private:
    // buffers containing static data
    cuphy::buffer<uint32_t, cuphy::device_alloc> bSymRxStatus;
    cuphy::tensor_device tWFreq, tWFreq4, tWFreqSmall, tShiftSeq, tUnShiftSeq, tShiftSeq4, tUnShiftSeq4;
    cuphyTensorPrm_t     tPrmWFreq, tPrmWFreq4, tPrmWFreqSmall, tPrmShiftSeq, tPrmUnShiftSeq, tPrmShiftSeq4, tPrmUnShiftSeq4; //TODO: move tensor parameters into tensor class
    std::string          bOutputFileName;
    cuphy::event         subSlotCompletedEvent;
    cuphy::event         waitCompletedSubSlotEvent;
    cuphy::event         waitCompletedFullSlotEvent;
    cuphy::event         uciOnPuschCompletedEvent;

    cuphy::tensor_device corr_half_nZpDmrsScTable[MAX_N_PRBS_SUPPORTED];
    cuphy::tensor_device eigVecCobTable[MAX_N_PRBS_SUPPORTED];
    cuphy::tensor_device corrTable[MAX_N_PRBS_SUPPORTED];
    cuphy::tensor_device eigValTable[MAX_N_PRBS_SUPPORTED];
    cuphy::tensor_device interpCobTable[MAX_N_PRBS_SUPPORTED];
    cuphy::tensor_device zpInterpVecTable[NUM_RKHS_ZP];
    cuphy::tensor_device zpDmrsScEigenVecTable[NUM_RKHS_ZP];
    cuphy::tensor_device secondStageTwiddleFactors[NUM_RKHS_ZP];
    cuphy::tensor_device secondStageFourierPerm[NUM_RKHS_ZP];

    cuphy::buffer<rkhsPrbPrms_t, cuphy::pinned_alloc>  cpuBufferRkhsPrbPrms;
    cuphy::buffer<rkhsZpPrms_t, cuphy::pinned_alloc>   cpuBufferRkhsZpPrms;
    cuphyPuschRkhsPrms_t puschRkhsPrms{};
    std::vector<cuphy::buffer<uint16_t, cuphy::device_alloc>>          gpuCsi2Maps;
    std::vector<cuphy::buffer<cuphyCsi2MapPrm_t, cuphy::device_alloc>> gpuCsi2MapPrms;

    cuphy::unique_pinned_ptr<uint8_t> hWorkCancelPtr;
};

//-------------------------------------------------------------------------------
// EvalDataset
// 1.) contains true TB bytes used to evaluate BLER
// 2.) if interBufferFlag = true, also contains intermediate buffers (Channel estimates, equalization filters, ...)

struct EvalDataset
{
    struct ueRefBufferOffsets
    {
        uint32_t harqPayloadByteOffset;
        uint32_t nHarqBytes;
        uint32_t harqCrcFlagOffset;
        uint32_t csi1PayloadByteOffset;
        uint32_t nCsi1Bytes;
        uint32_t csi1CrcFlagOffset;
        uint32_t csi2PayloadByteOffset;
        uint32_t nCsi2Bytes;
        uint32_t csi2CrcFlagOffset;
    };

    struct uciSizes
    {
        uint32_t G;
        uint32_t G_csi2;
        uint16_t nBitsCsi2;
    };

    // buffer offsets
    std::vector<ueRefBufferOffsets> ueRefBuffOffsetsVec;

    // BLER evaluation objects:
    uint32_t                          nCbs, nTbs;
    std::vector<uint32_t>             nBytesVec;
    std::vector<uint32_t>             nBytesPerCbVec, nCbsPerTbVec;
    std::vector<cuphy::tensor_pinned> tTrueTbBytesVec;
    std::vector<uint32_t>             nTbsInFileVec;
    std::vector<cuphy::tensor_pinned> tTrueCbCrcErrVec;
    std::vector<cuphy::tensor_pinned> tTrueTbCrcErrVec;

    // uci evaluation objects:
    uint32_t nHarqUcis;
    uint32_t firstHarqUciCbRef;
    uint32_t nHarqUciErrors, nCsi1UciErrors, nCsi2UciErrors;

    // intermediate buffers:
    bool                     interBufferFlag;
    uint32_t                 nUes, nUeGrps, nSchUes, nCsi2Ues;
    std::vector<PerTbParams> perTbPrmsRef;

    // uci buffers
    cuphy::typed_tensor<CUPHY_R_8U, cuphy::pinned_alloc> tRefUciPayloadBytes;
    std::vector<cuphy::tensor_pinned> tRefUciPayloadBytesVec;
    std::vector<cuphy::tensor_pinned> tRefUciCrcFlagsVec;
    std::vector<cuphy::tensor_pinned> tRefUciDTXsVec;
    std::vector<cuphy::tensor_pinned> tRefUciHarqDetStatusVec;
    std::vector<cuphy::tensor_pinned> tRefUciCsi1DetStatusVec;
    std::vector<cuphy::tensor_pinned> tRefUciCsi2DetStatusVec;

    std::vector<uciSizes>                                uciSizesVec;
    std::vector<uint16_t>                                csi2UeIdxsVec;

    using tensor_pinned_C_32F = cuphy::typed_tensor<CUPHY_C_32F, cuphy::pinned_alloc>;
    using tensor_pinned_R_32F = cuphy::typed_tensor<CUPHY_R_32F, cuphy::pinned_alloc>;
    using tensor_pinned_R_16F = cuphy::typed_tensor<CUPHY_R_16F, cuphy::pinned_alloc>;

    std::vector<cuphy::typed_tensor<CUPHY_C_32F, cuphy::pinned_alloc>> HestRef;
    std::vector<tensor_pinned_C_32F>                                   tRefChEstLSHest;
    std::vector<tensor_pinned_R_32F>                                   tRefChEstDelayMean;
    std::vector<tensor_pinned_C_32F>                                   tRefCfoEst;
    std::vector<tensor_pinned_R_32F>                                   tRefTaEsts;
    std::vector<tensor_pinned_R_32F>                                   tRefRssi, tRefRssiEhq;
    std::vector<tensor_pinned_R_32F>                                   tRefRssiFull;
    std::vector<tensor_pinned_R_32F>                                   tRefRsrp, tRefRsrpEhq;
    std::vector<tensor_pinned_R_32F>                                   tRefNoiseVarPreEq, tRefNoiseVarPostEq;
    std::vector<tensor_pinned_R_32F>                                   tRefSinrPreEq, tRefSinrPostEq, tRefSinrPreEqEhq;
    std::vector<tensor_pinned_R_32F>                                   tRefCfoEstHzPerUe;
    std::vector<tensor_pinned_R_32F>                                   tRefToEstMicroSecPerUe;
    std::vector<cuphy::typed_tensor<CUPHY_C_32F, cuphy::pinned_alloc>> chEqCoefRef;

    std::vector<cuphy::typed_tensor<CUPHY_R_16F, cuphy::pinned_alloc>> eqOutLLRsRef;
    std::vector<cuphy::typed_tensor<CUPHY_R_16F, cuphy::pinned_alloc>> schLLRsRef;
    std::vector<cuphy::typed_tensor<CUPHY_R_16F, cuphy::pinned_alloc>> csi1LLRsRef;
    std::vector<cuphy::typed_tensor<CUPHY_R_16F, cuphy::pinned_alloc>> csi2LLRsRef;
    std::vector<cuphy::typed_tensor<CUPHY_R_16F, cuphy::pinned_alloc>> harqLLRsRef;
    std::vector<cuphy::typed_tensor<CUPHY_R_16F, cuphy::pinned_alloc>> rmOutLLRsRef;

    cuphy::typed_tensor<CUPHY_R_32U, cuphy::pinned_alloc> ldpcOutRef;
    cuphy::tensor_pinned                                  tReference_derateCbsIndices;
    cuphy::tensor_pinned                                  tReference_derateCbsIndicesSizes;
    bool                                                  m_drmDebug = false;

    // Functions:
    EvalDataset(const std::vector<std::string>& inputFileNameVec, cudaStream_t cuStrm, int apiTVflag = 0, bool drmDebug = false); // construct dataset from h5 file
    EvalDataset(const EvalDataset& evalDataset);
    EvalDataset();
    EvalDataset& operator=(EvalDataset&& evalDataset);

    double   evalChEstDelayMean(std::vector<cuphy::tensor_device>& tDmrsDelayMeanGpu, int ueGrp, cudaStream_t cuStrm);
    double   evalChEstLS(std::vector<cuphy::tensor_device>& tLSEstGpu, int ueGrp, cudaStream_t cuStrm);
    double   evalChEst(std::vector<cuphy::tensor_device>& tHEstGpu, int ueGrp, cudaStream_t cuStrm);
    double   evalChEqCoef(std::vector<cuphy::tensor_device>& tChEqCoefGpu, int ueGrp, cudaStream_t cuStrm);
    double   evalChEqLLR(std::vector<cuphy::tensor_device>& tChEqLLRGpu, int ueGrp, cudaStream_t cuStrm);
    void     evalCfoTaEst(std::vector<cuphy::tensor_device>& tCfoEstGpu, std::vector<cuphy::tensor_device>& tTaEstGpu, cudaStream_t cuStrm);
    double   evalCfoEst(tensor_pinned_C_32F& tRefCfoEst, cuphy::tensor_pinned& tCfoEstRes, cudaStream_t cuStrm);
    double   evalCfoEstHzPerUe(tensor_pinned_R_32F& tCfoRef, cuphy::tensor_pinned& tCfoRes, cudaStream_t cuStrm, bool verbose = false);
    double   evalToEstMicroSecPerUe(tensor_pinned_R_32F& tToRef, cuphy::tensor_pinned& tToRes, cudaStream_t cuStrm, bool verbose = false);
    double   evalRssi(tensor_pinned_R_32F& tRssiRef, cuphy::tensor_pinned& tRssiRes, cudaStream_t cuStrm, bool verbose = false);
    double   evalRsrp(tensor_pinned_R_32F& tRsrpRef, cuphy::tensor_pinned& tRsrpRes, cudaStream_t cuStrm, bool verbose = false);
    double   evalRsrpDiff(tensor_pinned_R_32F& tRsrpRef, cuphy::tensor_pinned& tRsrpRes, cudaStream_t cuStrm, bool verbose = false);
    double   evalSinr(tensor_pinned_R_32F& tSinrRef, cuphy::tensor_pinned& tSinrRes, cudaStream_t cuStrm, bool verbose = false);
    double   evalNoiseIntfVar(tensor_pinned_R_32F& tNoiseIntfRef, cuphy::tensor_pinned& tNoiseIntfRes, cudaStream_t cuStrm, bool verbose = false);
    void     computeNumUciCbErrors(DynApiDataset const& dynApiDataset, bool evalEarlyHarqFlag, uint8_t openRanFunctionalSplitOption = PUSCH_7_2_A);
    uint32_t computeNumCbErrors(DynApiDataset const& dynApiDataset);
    void     evalPuschCrc(uint32_t* pTbCrc, uint32_t* pCbCrc, uint8_t* pTbPayload, cudaStream_t cuStrm);
    void     evalUciRmSizes(PerTbParams* pTbPrmsCuphy, PerTbParams* pTbPrmsRef, cuphyPuschUePrm_t* pUePuschPrms, uint16_t nUes);
    void     evalUciOnPuschSegLLRs1(uint16_t nUciUes, uint16_t* pUciUserIdxs, PerTbParams* pTbPrmsCpu, cudaStream_t cuStrm);
    void     evalUciOnPuschSegLLRs2(uint16_t nCsi2Ues, uint16_t* pCsi2UserIdxs, PerTbParams* pTbPrmsCpu, cudaStream_t cuStrm);
    void     evalUciOnPuschCsi2Ctrl(PerTbParams* pTbPrmsGpu, cudaStream_t cuStrm);
    void     evalPuschRm(void** pRmOutLLrAddrs, const PerTbParams* pTbPrmsCpu, cudaStream_t cuStrm);
    void     evalPuschRx(std::string const& resultFileName, StaticApiDataset const& statApiDataset, DynApiDataset const& dynApiDataset, cudaStream_t cuStrm);
    void     reportPuschCrcErrors(cuphyPuschDynPrms_t const& dynPrms);
};

//-------------------------------------------------------------------------------
// pucchDynApiDataset
// Contains dynamic api parameters and data for a single PucchRxPipeline
struct pucchDynApiDataset
{
public:
    // api C parameters:
    std::vector<cuphyPucchCellDynPrm_t> cellDynPrm;
    std::vector<cuphyPucchUciPrm_t>     F0UciPrmsVec;
    std::vector<cuphyPucchUciPrm_t>     F1UciPrmsVec;
    std::vector<cuphyPucchUciPrm_t>     F2UciPrmsVec;
    std::vector<cuphyPucchUciPrm_t>     F3UciPrmsVec;
    std::vector<cuphyPucchUciPrm_t>     F4UciPrmsVec;
    cuphyPucchCellGrpDynPrm_t           cellGrpDynPrm;
    cuphyPucchDynPrms_t                 pucchDynPrm;
    cuphyPucchDataOut_t                 DataOut;
    cuphyPucchDataIn_t                  DataIn;
    cuphyPucchDbgPrms_t                 dbgPrm;
    cuphyPucchStatusOut_t               StatusOutput;
    // input tensor parameters:
    std::vector<cuphyTensorPrm_t> tPrmDataRxVec;

    //pucchDynApiDataset();
    pucchDynApiDataset(const std::vector<std::string>& inputFileNameVec, cudaStream_t cuStrm, uint64_t procModeBmsk = 0);  // construct dataset from h5 file
    [[nodiscard]] const cuphyPucchPostPolarData_t* enablePostPolarData(const char* h5path, cudaStream_t cuStrm);
    void enableFrontEndLlrOutput(const char* h5path, cudaStream_t cuStrm);

    // buffers containing dynamic data
private:
    std::vector<cuphy::tensor_device>                              tDataRxVec;
    std::vector<cuphy::tensor_pinned>                              tPostPolarCbEstVec;
    cuphy::tensor_pinned                                           tPostPolarCrcErrorFlags;
    std::vector<cuphyTensorPrm_t>                                  tPrmPostPolarCbEstVec;
    cuphyPucchPostPolarData_t                                      postPolarData;
    std::vector<cuphy::tensor_device>                              tF2FrontEndLLRVec;
    std::vector<cuphy::tensor_device>                              tF3FrontEndLLRVec;
    std::vector<cuphyTensorPrm_t>                                  tPrmF2FrontEndLLRVec;
    std::vector<cuphyTensorPrm_t>                                  tPrmF3FrontEndLLRVec;
    cuphy::buffer<cuphyPucchF0F1UciOut_t, cuphy::pinned_alloc>     bF0UciOut;
    cuphy::buffer<cuphyPucchF0F1UciOut_t, cuphy::pinned_alloc>     bF1UciOut;
    cuphy::buffer<cuphyPucchF234OutOffsets_t, cuphy::pinned_alloc> bF2OutOffsets;
    cuphy::buffer<cuphyPucchF234OutOffsets_t, cuphy::pinned_alloc> bF3OutOffsets;
    cuphy::buffer<uint8_t, cuphy::pinned_alloc>                    bUciPayload;
    cuphy::buffer<uint8_t, cuphy::pinned_alloc>                    bCrcFlag;
    cuphy::buffer<uint8_t, cuphy::pinned_alloc>                    bDtxFlag;
    cuphy::buffer<float,   cuphy::pinned_alloc>                    bRssi;
    cuphy::buffer<float,   cuphy::pinned_alloc>                    bSinr;
    cuphy::buffer<float,   cuphy::pinned_alloc>                    bInterf;
    cuphy::buffer<float,   cuphy::pinned_alloc>                    bRsrp;
    cuphy::buffer<float,   cuphy::pinned_alloc>                    bTaEst;
    cuphy::buffer<uint8_t, cuphy::pinned_alloc>                    bHarqDetectionStatus;
    cuphy::buffer<uint8_t, cuphy::pinned_alloc>                    bCsiP1DetectionStatus;
    cuphy::buffer<uint8_t, cuphy::pinned_alloc>                    bCsiP2DetectionStatus;
    // pNumCsi2Bits not used, hence commented here
    //cuphy::buffer<uint16_t, cuphy::pinned_alloc>  bNumCsi2Bits;

    void populateUciParams(cuphy::cuphyHDF5_struct& uciPrmsH5, cuphyPucchUciPrm_t& uciPrms, int cellIdx, int uciIdx, uint8_t pucchFmt);
};

//-------------------------------------------------------------------------------
// pucchStaticApiDataset
// Contains static Pucch api parameters and data for a single PucchRxPipeline

struct pucchStaticApiDataset
{
public:
    // api C parameters
    cuphyPucchStatPrms_t            pucchStatPrms;
    cuphyTracker_t                  pucchTracker;
    cuphyPucchDbgPrms_t             dbgPrm;
    std::vector<cuphyCellStatPrm_t> cellStatPrm;
    cuphyPucchCellStatPrm_t         pucchCellStatPrm;
    // Owning storage for the input TV path. Mode-specific example checks parse
    // this file in the dataset layer, not inside the PUCCH API.
    std::string                     refH5Path;

    pucchStaticApiDataset(const std::vector<std::string>& inputFileNameVec, cudaStream_t cuStrm, std::string outFileName = std::string());  // construct dataset from h5 file

private:
    // buffers containing static data
    std::string bOutputFileName;
};

// Non-public PUCCH test/example helper for SKIP_BACKEND front-end reference checking.
[[nodiscard]] cuphyStatus_t comparePucchFrontEndRefForBackendSkip(const cuphyPucchDataOut_t* pDataOut,
                                                                  const char*                h5path,
                                                                  float                      tolHalf,
                                                                  cudaStream_t               cuStream);

//-------------------------------------------------------------------------------
// EvalPucchDataset
// contains reference UCI output to evaluate cuPHY PUCCH reciever

struct EvalPucchDataset
{
    struct pucchF234bufferOffsets
    {
        uint32_t harqDetStatOffset;
        uint32_t csiPart1DetStatOffset;
        uint32_t csiPart2DetStatOffset;
        uint32_t dtxFlagOffset;
        uint32_t snrOffset;
        uint32_t RSRPoffset;
        uint32_t RSSIoffset;
        uint32_t InterfOffset;
        uint32_t taEstOffset;
        uint32_t uciSeg1PayloadByteOffset;
        uint32_t nUciSeg1Bytes;
        uint32_t harqPayloadByteOffset;
        uint32_t nHarqBytes;
        uint32_t srPayloadByteOffset;
        uint32_t nSrBytes;
        uint32_t csiP1PayloadByteOffset;
        uint32_t nCsiP1Bytes;
        uint32_t LLRsoffset;
        uint32_t Seg1LLRsoffset;
        uint32_t Seg2LLRsoffset;
        uint32_t nSegLLRs;
        uint32_t nSeg1LLRs;
        uint32_t nSeg2LLRs;
        uint16_t cellIdx;
    };

    uint16_t                            nF0Ucis;
    std::vector<cuphyPucchF0F1UciOut_t> F0UcisOutRefVec;
    std::vector<cuphyPucchF0F1UciOut_t> F0UcisOutVec;

    uint16_t                            nF1Ucis;
    std::vector<cuphyPucchF0F1UciOut_t> F1UcisOutRefVec;

    uint16_t                            nF2Ucis;
    std::vector<pucchF234bufferOffsets> pucchF2bufferOffsetsVec;

    uint16_t                            nF3Ucis;
    std::vector<pucchF234bufferOffsets> pucchF3bufferOffsetsVec;

    float taWrapSpanUs{}; //!< Unambiguous PF1 TA span in microseconds (1e6/scs), used for wrap-aware TA comparison; set from cell numerology (mu) at construction.
    
    std::vector<bool> pucchF1multiplexed; //!< Per-PF1-UCI flag for PRB/symbol sharing (cyclic shift + tOCC); TA checks use relaxed tolerance under multiplexing interference.

    // Buffers holding format 2,3,4 data
    std::vector<cuphy::typed_tensor<CUPHY_R_8U, cuphy::pinned_alloc>>  tRefHarqDetStat;
    std::vector<cuphy::typed_tensor<CUPHY_R_8U, cuphy::pinned_alloc>>  tRefCsiPart1DetStat;
    std::vector<cuphy::typed_tensor<CUPHY_R_8U, cuphy::pinned_alloc>>  tRefCsiPart2DetStat;
    std::vector<cuphy::typed_tensor<CUPHY_R_8U, cuphy::pinned_alloc>>  tRefDtxFlags;
    std::vector<cuphy::typed_tensor<CUPHY_R_32F, cuphy::pinned_alloc>> tRefSinr;
    std::vector<cuphy::typed_tensor<CUPHY_R_32F, cuphy::pinned_alloc>> tRefInterf;
    std::vector<cuphy::typed_tensor<CUPHY_R_32F, cuphy::pinned_alloc>> tRefRssi;
    std::vector<cuphy::typed_tensor<CUPHY_R_32F, cuphy::pinned_alloc>> tRefRsrp;
    std::vector<cuphy::typed_tensor<CUPHY_R_32F, cuphy::pinned_alloc>> tRefTaEst;

    std::vector<cuphy::typed_tensor<CUPHY_R_8U, cuphy::pinned_alloc>>  tRefPayloadBytes;
    std::vector<cuphy::typed_tensor<CUPHY_R_16F, cuphy::pinned_alloc>> tRefLLRs;
    std::vector<cuphy::typed_tensor<CUPHY_R_16F, cuphy::pinned_alloc>> tRefSeg1LLRs;
    std::vector<cuphy::typed_tensor<CUPHY_R_16F, cuphy::pinned_alloc>> tRefSeg2LLRs;

    // functions:
    EvalPucchDataset(const std::vector<std::string>& inputFileNameVec, cudaStream_t cuStrm); // construct dataset from h5 file
    uint16_t evalPucchF0Receiver(cuphyPucchF0F1UciOut_t* pF0UcisOutGpu, cudaStream_t cuStrm);
    uint16_t evalPucchF1Receiver(cuphyPucchF0F1UciOut_t* pF1UcisOutGpu, cudaStream_t cuStrm);
    uint16_t evalPucchF2FrontEnd(__half** pDescramLLRaddrs, uint8_t* pDTXflags, uint16_t* E_seg1, cudaStream_t cuStrm);
    uint16_t evalPucchF3FrontEnd(__half** pDescramLLRaddrs, uint8_t* pDTXflags, uint16_t* E_seg1, cudaStream_t cuStrm);
    uint16_t evalPucchF3SegLLRs(__half** pDescramLLRaddrs, uint16_t* E_seg1, uint16_t* E_seg2, cudaStream_t cuStrm);
    uint16_t evalPucchF234UciSeg(uint8_t* uciPayloadsGpu, uint16_t nF2Ucis, uint16_t nF3Ucis, cuphyPucchF234OutOffsets_t* pF2Cuphyoffsets, cuphyPucchF234OutOffsets_t* pF3Cuphyoffsets, cudaStream_t cuStrm);
    uint16_t compareF0F1UciOutput(uint16_t nUcis, cuphyPucchF0F1UciOut_t* uciOutMeas, std::vector<cuphyPucchF0F1UciOut_t> uciOutRef, std::string pucchFormatName);
    uint16_t compareF234UciOutput(uint16_t nUcis, cuphyPucchDynPrms_t& pucchDynPrm, cuphyPucchF234OutOffsets_t* pCuphyoffsets, pucchF234bufferOffsets* pRefoffsets, std::string pucchFormatName);
    int     evalPucchRxPipeline(cuphyPucchDynPrms_t& pucchDynPrm);
};

//--------------------------------------------------------------------------------
// UciPolarDataset
// 1.) contains parameters and input data consumed by cuPHY uci polar functions
// 2.) contains reference intermediate buffers
// 3.) contains validation functions comparing cuPHY output against reference buffers

struct UciPolarDataset
{
    // parameters:
    uint16_t                           nPolUciSegs, nPolCws;
    std::vector<cuphyPolarUciSegPrm_t> polUciSegPrmsVec;

    // intermediate buffers:
    std::vector<cuphy::typed_tensor<CUPHY_R_8U, cuphy::pinned_alloc>>  refCwTreeTypesVec;
    std::vector<cuphy::typed_tensor<CUPHY_R_16F, cuphy::pinned_alloc>> refCwLLRsVec;
    std::vector<cuphy::typed_tensor<CUPHY_R_16F, cuphy::pinned_alloc>> refUciSegLLRsVec;
    std::vector<cuphy::typed_tensor<CUPHY_R_32U, cuphy::pinned_alloc>> refCbEstsVec;
    std::vector<cuphy::typed_tensor<CUPHY_R_32U, cuphy::pinned_alloc>> refUciSegEstsVec;
    cuphy::typed_tensor<CUPHY_R_8U, cuphy::pinned_alloc>               refCrcErrorFlags;

    // functions:
    UciPolarDataset(const std::string& inputFileName, cudaStream_t cuStrm); // construct dataset from h5 file
    void evalCwTreeTypes(uint8_t** pCwTreeTypesAddrsGpu, cudaStream_t cuStrm);
    void evalCwLLRs(uint16_t nPolCws, cuphyPolarCwPrm_t* pPolarCwPrms, __half** pCwLLRsAddrsGpu, cudaStream_t cuStrm);
    void evalDecoderOutput(uint16_t nPolCbs, cuphyPolarCwPrm_t* pPolarCwPrms, uint32_t** pCbEstGpuAddrs, uint8_t* pCrcErrorFlagsGpu, uint16_t nPolSegs, cuphyPolarUciSegPrm_t* pUciSegPrms, uint32_t** pUciSegEstGpuAddrs, cudaStream_t cuStrm);
};

//--------------------------------------------------------------------------------
// simplexDataset
// 1.) contains parameters and input data consumed by cuPHY simplex functions
// 2.) contains validation functions comparing cuPHY output against reference buffers

struct simplexDataset
{
    // parameters:
    uint16_t                         nCws;
    std::vector<cuphySimplexCwPrm_t> simplexCwPrmsVec;

    // buffers:
    std::vector<cuphy::typed_tensor<CUPHY_R_16F, cuphy::pinned_alloc>> refCwLLRsVec;
    cuphy::typed_tensor<CUPHY_R_32U, cuphy::pinned_alloc>              refCbs;

    // functions:
    simplexDataset(const std::string& inputFileName, cudaStream_t cuStrm);
    void evalDecoderOutput(cuphySimplexCwPrm_t* h_simplexCwPrms, cudaStream_t cuStrm);
};


//-------------------------------------------------------------------------------
// srsDynApiDataset
// Contains dynamic api parameters and data for a single srsRxPipeline
struct srsDynApiDataset
{
public:
    // api C parameters:
    std::vector<cuphySrsCellDynPrm_t>  cellDynPrmVec;
    std::vector<cuphyUeSrsPrm_t>       ueSrsPrmVec;
    cuphySrsCellGrpDynPrm_t            cellGrpDynPrm;
    cuphySrsDynPrms_t                  srsDynPrm;
    cuphySrsDataIn_t                   dataIn;
    cuphySrsDataOut_t                  dataOut;
    cuphySrsDynDbgPrms_t               dynDbgPrm;
    std::vector<cuphySrsChEstToL2_t>   chEstToL2Vec;
    
    cuphySrsStatusOut_t               StatusOutput;

    srsDynApiDataset();
    srsDynApiDataset(const std::vector<std::string>& inputFileNameVec, cudaStream_t cuStrm, uint64_t procModeBmsk = 0); // construct dataset from h5 file

    std::vector<cuphy::tensor_device> tSrsChEstVec;
private:
    std::vector<cuphy::tensor_device>                        tDataRxVec;
    std::vector<cuphyTensorPrm_t>                            tPrmDataRxVec;
    std::vector<cuphy::buffer<uint8_t, cuphy::pinned_alloc>> chEstCpuBuffVec;

    cuphy::buffer<float, cuphy::pinned_alloc> rbSnrVec;
    std::vector<uint32_t>                rbSnrBuffOffsetVec;
    cuphy::buffer<cuphySrsReport_t, cuphy::pinned_alloc> srsReportVec;
    std::vector<cuphySrsChEstBuffInfo_t> srsChEstBuffInfoVec;

    static constexpr uint32_t LINEAR_ALLOC_PAD_BYTES = 128;
};

//-------------------------------------------------------------------------------
// srsStaticApiDataset
// Contains static api parameters and data for a single srsRxPipeline
struct srsStaticApiDataset
{
public:
    // api C parameters:
    cuphySrsStatPrms_t              srsStatPrms{};
    cuphyTracker_t                  srsTracker{};
    cuphySrsStatDbgPrms_t           statDbgPrm{};
    std::vector<cuphyCellStatPrm_t> cellStatPrmVec;

    srsStaticApiDataset(const std::vector<std::string>& inputFileNameVec, cudaStream_t cuStrm, std::string outFileName = std::string()); // construct dataset from h5 file

private:
    // buffers containing static data
    cuphy::tensor_device tFocc_table, tFocc_comb2_table, tFocc_comb4_table;
    cuphyTensorPrm_t     tPrmFocc_table, tPrmFocc_comb2_table, tPrmFocc_comb4_table;
    cuphy::tensor_device tW_comb2_nPorts1_wide, tW_comb2_nPorts2_wide, tW_comb2_nPorts4_wide, tW_comb2_nPorts8_wide, tW_comb4_nPorts1_wide, tW_comb4_nPorts2_wide, tW_comb4_nPorts4_wide, tW_comb4_nPorts6_wide, tW_comb4_nPorts12_wide;
    cuphyTensorPrm_t     tPrmW_comb2_nPorts1_wide, tPrmW_comb2_nPorts2_wide, tPrmW_comb2_nPorts4_wide, tPrmW_comb2_nPorts8_wide, tPrmW_comb4_nPorts1_wide, tPrmW_comb4_nPorts2_wide, tPrmW_comb4_nPorts4_wide, tPrmW_comb4_nPorts6_wide, tPrmW_comb4_nPorts12_wide;
    cuphy::tensor_device tW_comb2_nPorts1_narrow, tW_comb2_nPorts2_narrow, tW_comb2_nPorts4_narrow, tW_comb2_nPorts8_narrow, tW_comb4_nPorts1_narrow, tW_comb4_nPorts2_narrow, tW_comb4_nPorts4_narrow, tW_comb4_nPorts6_narrow, tW_comb4_nPorts12_narrow;
    cuphyTensorPrm_t     tPrmW_comb2_nPorts1_narrow, tPrmW_comb2_nPorts2_narrow, tPrmW_comb2_nPorts4_narrow, tPrmW_comb2_nPorts8_narrow, tPrmW_comb4_nPorts1_narrow, tPrmW_comb4_nPorts2_narrow, tPrmW_comb4_nPorts4_narrow, tPrmW_comb4_nPorts6_narrow, tPrmW_comb4_nPorts12_narrow;

    cuphy::tensor_device eigenVecTable[NUM_RKHS_GRIDS];
    cuphy::tensor_device eigenValueTable[NUM_RKHS_GRIDS];
    cuphy::tensor_device eigenCorrTable[NUM_RKHS_GRIDS];
    cuphy::tensor_device secondStageTwiddleFactorsTable;
    cuphy::tensor_device secondStageFourierPermTable;

    cuphySrsRkhsPrms_t srsRkhsPrms;
    cuphy::buffer<cuphyRkhsGridPrms_t, cuphy::pinned_alloc>  cpuBufferRkhsGridPrms;
    
    std::string bOutputFileName;
};

//-------------------------------------------------------------------------------
// srsEvalDataset
// contains reference srs output to evaluate cuPHY SRS reciever

struct srsEvalDataset
{
    // Buffers hold reference srs outputs:
    std::vector<cuphy::typed_tensor<CUPHY_R_32F, cuphy::pinned_alloc>> rbSnrsRef;
    std::vector<cuphySrsReport_t>                                      srsReportsRef;
    std::vector<uint16_t>                                              srsStartValidPrgRef;
    std::vector<uint16_t>                                              srsNValidPrgRef;
    std::vector<cuphy::typed_tensor<CUPHY_C_32F, cuphy::pinned_alloc>> srsChEstBuffsRef;
    std::vector<cuphy::typed_tensor<CUPHY_C_32I, cuphy::pinned_alloc>> srsChEstToL2BuffsRef;

    // functions:
    srsEvalDataset(const std::vector<std::string>& inputFileNameVec, cudaStream_t cuStrm); // construct dataset from h5 file

    void evalSrsRx(cuphySrsDynPrms_t& srsDynPrm, std::vector<cuphy::tensor_device>& tSrsChEstVec, float* h_rbSnrsCuphy, cuphySrsReport_t* h_srsReportsCuphy, cudaStream_t cuStrm);
};


enum class OutputBufferMode {
    Host,
    Device
};

//-------------------------------------------------------------------------------
// bfwDynApiDataset
// Contains dynamic api parameters and data for a single bfwPipeline

struct bfwDynApiDataset
{
public:
    // api C parameters:
    std::vector<cuphyBfwUeGrpPrm_t>       bfwUeGrpPrmVec;
    std::vector<cuphyBfwLayerPrm_t>       bfwLayerPrmVec;
    cuphyBfwDynPrms_t                     bfwDynPrms;
    cuphyBfwDynPrm_t                      bfwDynPrm;
    cuphyBfwDataIn_t                      dataIn;
    cuphyBfwDataOut_t                     dataOut;
    cuphyBfwDynDbgPrms_t                  dynDbgPrm;
    
    cuphyBfwStatusOut_t                   StatusOutput;


    bfwDynApiDataset(const std::vector<std::string>& inputFileNameVec, cudaStream_t cuStrm, uint64_t procModeBmsk = 0, OutputBufferMode outputBufferMode = OutputBufferMode::Host, int16_t beamIdOffset = -1); // construct dataset from h5 file

#ifdef BFW_BOTH_COMP_FLOAT
    std::vector<uint8_t*>                 bfwBufferVec;
    std::vector<cuphy::tensor_device>     tBfwVec;
#endif
    OutputBufferMode m_outputBufferMode;
    std::vector<std::unique_ptr<cuphy::buffer_wrapper>> bfwCompBufferVec;
    std::vector<uint8_t*>                 bfwComppBufVec;
    std::vector<cuphySrsChEstBuffInfo_t>  srsChEstBufInfoVec;

    std::vector<cuphy::tensor_device> tSrsChEstVec;
};

//-------------------------------------------------------------------------------
// bfwStatApiDataset
// Contains static api parameters and data for a single bfwPipeline

struct bfwStaticApiDataset
{
public:
    // api C parameters:
    cuphyBfwStatPrms_t    bfwStatPrms;
    cuphyTracker_t        bfwTracker;
    cuphyBfwDbgPrms_t     dbgPrm;
    cuphyBfwStatDbgPrms_t statDbgPrm;

    bfwStaticApiDataset(const std::vector<std::string>& inputFileNameVec, cudaStream_t cuStrm, std::string outFileName = std::string()); // construct dataset from h5 file

private:
    // buffers containing static data
    std::string bOutputFileName;
};

//-------------------------------------------------------------------------------
// bfwEvalDataset
// contains reference bfw outputs to evaluate cuPHY bfw compute

struct bfwEvalDataset
{
    bfwEvalDataset(const std::vector<std::string>& inputFileNameVec, cudaStream_t cuStrm); // construct dataset from h5 file

    // functions:
    void bfwEvalCoefs(bfwStaticApiDataset& statApiDataset, bfwDynApiDataset& dynApiDataset, cudaStream_t cuStrm, float refCheckSnrThd, bool verbose = false);
    [[nodiscard]] bool bfwDecompressCompare(uint16_t ueGrpIdx, float beta, int bundleSize, uint8_t* input, int16_t beamIdOffset);

private:
    // Buffers hold reference bfw outputs:
    std::vector<cuphy::typed_tensor<CUPHY_C_32F, cuphy::pinned_alloc>> bfwBufRefVec;
    std::vector<cuphy::typed_tensor<CUPHY_R_8U, cuphy::pinned_alloc>> bfwCompBufRefVec;
    uint16_t m_nCells;
    std::vector<uint32_t> m_nUeGrpsInCell;
};





//-------------------------------------------------------------------------------
// pdschDynApiDataset
// Contains dynamic api parameters and data for a single PdschTxPipeline

struct postFecLdpcOutputHelper_t
{
    uint32_t LDPC_dst_offset;
    bool     test_model;
    uint32_t num_CBs;
    uint32_t N;
    uint32_t tbSize;
};

struct pdschDynApiDataset
{
public:
    // api C parameters
    std::vector<cuphyPdschCellDynPrm_t> CellPrms;
    std::vector<cuphyPdschCellAerialMetrics_t> CellMetrics;
    std::vector<cuphyPdschUeGrpPrm_t>   UeGrpPrms;
    std::vector<cuphyPdschUePrm_t>      UePrms;
    std::vector<cuphyPdschCwPrm_t>      CwPrms;
    std::vector<cuphyPdschDmrsPrm_t>    pdsch_dmrs_pars;
    std::vector<_cuphyCsirsRrcDynPrm>   CsirsPrms;
    std::vector<cuphyPmW_t>             PmwPrms;
    cuphyPdschCellGrpDynPrm_t           cell_grp_dyn_params;
    cuphyPdschDynPrms_t                 pdsch_dyn_params;

    std::vector<cuphyPdschPostFecDataIn_t>  post_fec_data_in;
    std::vector<cuphyPdschDataIn_t>  data_in;
    std::vector<cuphyPdschDataIn_t>  tb_crc_in;
    std::vector<cuphyPdschDataOut_t> output_data;
    std::vector<cuphyPdschStatusOut_t> output_status;
    std::vector<cuphyTensorPrm_t>    output_tensorPrm;

    pdschDynApiDataset();
    pdschDynApiDataset(const std::string& inputFileName, uint32_t max_cells, cudaStream_t cuStrm, cuphyPdschProcMode_t pdsch_proc_mode, cuphyPdschStatPrms_t& stat_params, bool cfg_pdsch_TB_input_on_GPU=false, int cfg_forced_TB_byte_alignment=1, int cell_grp_Emax=PDSCH_MAX_ER_PER_CB_BITS /* computed in PDSCH static dataset, only relevant in POST_FEC_RM_SCRAMBLING_PROCESSING mode*/);
    pdschDynApiDataset(const pdschDynApiDataset&) = delete; // deep copy for GPU memory buffers not implemented
    pdschDynApiDataset& operator=(const pdschDynApiDataset&) = delete;
    pdschDynApiDataset(pdschDynApiDataset&& other) noexcept;
    pdschDynApiDataset& operator=(pdschDynApiDataset&& other) noexcept;
    void                ResetPointers();
    void                cumulativeUpdate(const std::string& inputFileName, cudaStream_t cuStrm, cuphyPdschProcMode_t pdsch_proc_mode);
    void                print();
    void                resetOutputTensors(cudaStream_t cuStrm);

    // buffers containing dynamic data
private:
    uint32_t                                 max_cells;
    uint32_t                                 max_UEs_per_cell_group;
    uint32_t                                 max_CBs_per_TB; // across all TBs in cell group
    cuphy::unique_device_ptr<__half2>        large_buffer; // Allocated during constructor and covers max_cells max. configured PDSCH output tensors
    std::vector<cuphy::tensor_device>        data_tx_tensor;
    std::unique_ptr<uint8_t*[]>              crc_input_data_ptr;
    std::unique_ptr<uint8_t*[]>              tb_crc_data_in_ptr;
    bool                                     pdsch_TB_input_on_GPU;
    int                                      forced_TB_byte_alignment;
    bool                                     read_TB_CRC;

    std::vector<cuphy::typed_tensor<CUPHY_R_8U, cuphy::pinned_alloc>> crc_input_data_pinned;
    std::vector<cuphy::unique_pinned_ptr<uint8_t>> padded_crc_input_data_pinned;
    std::vector<cuphy::typed_tensor<CUPHY_R_8U, cuphy::device_alloc>> crc_input_data_device;
    std::vector<cuphy::unique_device_ptr<uint8_t>> padded_crc_input_data_device;
    std::vector<cuphy::typed_tensor<CUPHY_R_32U, cuphy::pinned_alloc>> tb_crc_input_data;
    uint32_t                                                          large_buffer_elements, large_buffer_bytes;

    // RAII ownership of nested new[] allocations made by read_ue_groups_pars_from_file() / read_ue_pars_from_file() in cuphy_hdf5.hpp
    std::vector<std::unique_ptr<uint8_t[]>>               rb_bitmaps;
    std::vector<std::unique_ptr<uint16_t[]>>              ue_prm_idxs;
    std::vector<std::unique_ptr<cuphyPdschDmrsPrm_t[]>>   dmrs_prms;
    std::vector<std::unique_ptr<uint16_t[]>>              cw_idxs;

    cuphy::unique_device_ptr<uint32_t> d_ldpc_workspace;
    cuphyPdschPipelineMode_t pipeline_processing_mode{cuphyPdschPipelineMode_t::PDSCH_FULL_PROCESSING};
    std::vector<postFecLdpcOutputHelper_t> post_fec_input_helper;

    cuphy::unique_device_ptr<uint32_t> d_rm_scrambling_workspace;
    int cell_grp_Emax; // computed in PDSCH static dataset
};

//-------------------------------------------------------------------------------
// pdschStaticApiDataset
// Contains static api parameters and data for a single PdschTxPipeline

struct pdschStaticApiDataset
{
public:
    // api C parameters
    cuphyPdschStatPrms_t             pdschStatPrms;
    cuphyTracker_t                   pdschTracker;
    std::vector<cuphyPdschDbgPrms_t> dbgPrm;
    std::vector<cuphyCellStatPrm_t>  cellStatPrm;

    pdschStaticApiDataset();
    pdschStaticApiDataset(const std::string& inputFileName, std::string outputFileName, bool ref_check, bool identical_ldpc_configs, int stream_priority, uint32_t max_cells, uint32_t max_CBs_per_TB = 0, uint32_t max_UEs_per_cell_group = 0, uint32_t max_PRBs = 0, bool use_batched_memcpy=false, cuphyPdschPipelineMode_t pipeline_processing_mode=cuphyPdschPipelineMode_t::PDSCH_FULL_PROCESSING, bool read_TB_CRC=false, uint32_t delay_usec=0); // construct dataset from h5 file
    pdschStaticApiDataset(const pdschStaticApiDataset& other);
    void                   ResetPointers();
    pdschStaticApiDataset& operator=(pdschStaticApiDataset&& other) noexcept;
    pdschStaticApiDataset(pdschStaticApiDataset&& other) noexcept;
    pdschStaticApiDataset& operator=(const pdschStaticApiDataset&) = delete;

    void cumulativeUpdate(const std::string& inputFileName, std::string outputFileName, bool ref_check, bool identical_ldpc_configs);
    void print();

    [[nodiscard]] int getEmax() const;

private:
    std::vector<std::string> CfgFileName;
    bool                     compute_max_values;
    int                      cell_grp_Emax;
};

//-------------------------------------------------------------------------------
// pdcchStaticApiDataset
// Contains static API parameters and data for a single PdcchTxPipeline

struct pdcchStaticApiDataset{
public:
    cuphyPdcchStatPrms_t  pdcchStatPrms;
    cuphyTracker_t        pdcchTracker;
    pdcchStaticApiDataset(int cfg_max_cells_per_slot=1);
    void print();
private:
    std::vector<std::string> CfgFileName;
};

//-------------------------------------------------------------------------------
// pdcchDynApiDataset
// Contains dynamic API parameters and data for a single PdcchTxPipeline

struct pdcchDynApiDataset {
public:
    std::vector<cuphyPdcchDciPrm_t>         dci_params;
    std::vector<cuphyPdcchCoresetDynPrm_t>  coreset_params;
    std::vector<cuphyPmWOneLayer_t>         pdcch_precoding_matrix;
    cuphyPdcchDynPrms_t                     pdcch_dyn_params;

    std::vector<cuphyPdcchDataIn_t>      data_in;
    std::vector<cuphyPdcchDataOut_t>     output_data;
    std::vector<cuphyTensorPrm_t>        output_tensorPrm;

    pdcchDynApiDataset(const std::string& inputFileName, uint32_t max_cells, cudaStream_t cuStrm, uint64_t procModeBmsk = 0);
    void cumulativeUpdate(const std::string& inputFileName, cudaStream_t cuStrm);

    int  refCheck(bool verbose=false);
    void revDciOrder();
    void printPayload();

private:
    uint32_t max_cells;
    cuphy::unique_device_ptr<__half2> large_buffer; // Allocated during constructor and covers max_cells max. configured PDCCH output tensors
    std::vector<cuphy::tensor_device> data_tx_tensor;
    uint32_t large_buffer_elements, large_buffer_bytes;
    std::vector<cuphy::typed_tensor<CUPHY_R_8U, cuphy::pinned_alloc>>  input_data;
    std::vector<cuphy::typed_tensor<CUPHY_R_8U, cuphy::pinned_alloc>>  x_input_data;
    std::vector<cuphy::typed_tensor<CUPHY_R_32U, cuphy::pinned_alloc>>  c_input_data;
    std::vector<std::string> CfgFileName;
};

#if 1

//-------------------------------------------------------------------------------
// PrachApiDataset
// Contains PRACH api parameters and data for a single PRACH Pipeline
struct PrachApiDataset
{
public:
    // api C parameters
    cuphyPrachStatPrms_t prachStatPrms;
    cuphyTracker_t       prachTracker;
    cuphyPrachDynPrms_t prachDynPrms;
    cuphyPrachStatDbgPrms_t statDbgPrm;
    cuphyPrachDynDbgPrms_t dynDbgPrm;

    std::vector<cuphy::tensor_device> dataRxTensor;
    std::vector<cuphyTensorPrm_t> dataRxTensorPrm;
    
    cuphy::tensor_device num_detectedPrmb;
    cuphy::tensor_device prmbIndex_estimates;
    cuphy::tensor_device prmbDelay_estimates;
    cuphy::tensor_device prmbPower_estimates;
    cuphy::tensor_pinned ant_rssi;
    cuphy::tensor_pinned rssi;
    cuphy::tensor_pinned interference;

    std::vector<cuphyPrachCellStatPrms_t> prachCellStatPrms;
    std::vector<cuphyPrachOccaStatPrms_t> prachOccaStatPrms;
    std::vector<cuphyPrachOccaDynPrms_t>  prachOccaDynPrms;
    cuphyPrachDataIn_t  dataIn;
    cuphyPrachDataOut_t dataOut;
    
    cuphyPrachStatusOut_t StatusOutput;

    int nCells = 0;

    // number of PRACH occasions in dataset
    int nOccasaions = 0;
    
    hdf5hpp::hdf5_file prach_file;

    PrachApiDataset(const std::string& inputFileName, cudaStream_t cuStrm, uint64_t procModeBmsk, bool refCheck, std::string outputFileName = std::string());
    void cumulativeUpdate(const std::string& inputFileName, std::string outputFileName, cudaStream_t cuStrm);
    void finalize(cudaStream_t cuStrm);
    int evaluateOutput();

private:

    uint64_t procModeBmsk_;
    bool enable_ref_check = false;

    std::string bOutputFileName;

    std::vector<int> ref_num_prmb;
    std::vector<int> ref_num_ant;
    std::vector<float> ref_interference;
    std::vector<float> ref_rssi;
    std::vector<std::vector<float>> ref_ant_rssi;
    std::vector<std::vector<int>> ref_prmb_index;
    std::vector<std::vector<float>> ref_delay_time;
    std::vector<std::vector<float>> ref_peak_power;

    void readReferenceValues(hdf5hpp::hdf5_file& prach_file, int numOccaInCell);
};

#endif

//-------------------------------------------------------------------------------
// ssbStaticApiDataset
// Contains static API parameters and data for a single SsbTxPipeline

struct ssbStaticApiDataset{
public:
    cuphySsbStatPrms_t  ssbStatPrms;
    cuphyTracker_t      ssbTracker;
    ssbStaticApiDataset(int cfg_max_cells_per_slot=1);
private:
    //std::vector<std::string> CfgFileName;
};

//-------------------------------------------------------------------------------
// ssbDynApiDataset
// Contains dynamic API parameters and data for a single SsbTxPipeline

struct ssbDynApiDataset {
public:
    std::vector<cuphyPerCellSsbDynPrms_t>  per_cell_SSB_params;
    std::vector<cuphyPerSsBlockDynPrms_t>  per_SS_block_params;
    std::vector<cuphyPmWOneLayer_t>        ssb_precoding_matrix;
    cuphySsbDynPrms_t                      ssb_dyn_params;

    std::vector<cuphySsbDataIn_t>          data_in;
    std::vector<cuphySsbDataOut_t>         output_data;
    std::vector<cuphyTensorPrm_t>          output_tensorPrm;

    ssbDynApiDataset(const std::string& inputFileName, uint32_t max_cells, cudaStream_t cuStrm, uint64_t procModeBmsk = SSB_PROC_MODE_STREAMS);
    void cumulativeUpdate(const std::string& inputFileName, cudaStream_t cuStrm);
    int  refCheck(bool verbose=false);

private:

    uint32_t max_cells;
    cuphy::unique_device_ptr<__half2> large_buffer; // Allocated during constructor and covers max_cells max. configured SSB output tensors
    std::vector<cuphy::tensor_device> data_tx_tensor;
    uint32_t large_buffer_elements, large_buffer_bytes;
    std::vector<cuphy::typed_tensor<CUPHY_R_32U, cuphy::pinned_alloc>>  input_data;
    std::vector<std::string> CfgFileName;
};

// csirsStaticApiDataset
// Contains static API parameters and data for a single CsirsTxPipeline

struct csirsStaticApiDataset{
public:
    cuphyCsirsStatPrms_t  csirsStatPrms;
    cuphyTracker_t        csirsTracker;
    std::vector<cuphyCellStatPrm_t>  cellStatPrm;
    csirsStaticApiDataset(const std::string& inputFileName, int cfg_max_cells_per_slot=1);
    void print();
    void cumulativeUpdate(const std::string& inputFileName);
private:
    std::vector<std::string> CfgFileName;
};

//-------------------------------------------------------------------------------
// csirsDynApiDataset
// Contains dynamic API parameters and data for a single CsirsTxPipeline

struct csirsDynApiDataset {
public:
    std::vector<cuphyCsirsRrcDynPrm_t>      rrc_params;
    std::vector<cuphyCsirsCellDynPrm_t>     cell_params;
    cuphyCsirsDynPrms_t                     csirs_dyn_params;
    std::vector<cuphyPmWOneLayer_t>         csirs_precoding_matrix;

    std::vector<cuphyCsirsDataOut_t>     output_data;
    std::vector<cuphyTensorPrm_t>        output_tensorPrm;

    csirsDynApiDataset(const std::string& inputFileName, uint32_t max_cells, cudaStream_t cuStrm, uint64_t procModeBmsk = 0);
    void cumulativeUpdate(const std::string& inputFileName, cudaStream_t cuStrm);

    int  refCheck(bool verbose=false);

private:
    uint32_t max_cells;
    uint32_t total_rrc_params;
    cuphy::unique_device_ptr<__half2> large_buffer; // Allocated during constructor and covers max_cells max. configured CSI-RS output tensors
    std::vector<cuphy::tensor_device> data_tx_tensor;
    uint32_t large_buffer_elements, large_buffer_bytes;
    std::vector<std::string> CfgFileName;
};

//-------------------------------------------------------------------------------
// csirsRxDynApiDataset
// Contains dynamic API parameters and data for a single CsirsRxPipeline

struct csirsRxDynApiDataset {
public:
    std::vector<cuphyCsirsRrcDynPrm_t>       rrc_params;
    std::vector<cuphyCsirsCellDynPrm_t>      cell_params;
    std::vector<cuphyCsirsUeDynPrm_t>        ue_params;
    cuphyCsirsRxDynPrms_t                    csirs_rx_dyn_params;
    std::vector<cuphyPmWOneLayer_t>          csirs_precoding_matrix;
    //input
    std::vector<cuphyCsirsRxDataIn_t>        input_data;
    std::vector<cuphyTensorPrm_t>            input_tensorPrm;
    //output
    std::vector<cuphyCsirsRxDataOut_t>       output_data;
    std::vector<cuphyCsirsRxChEstBuffInfo_t> output_chest_data;
    std::vector<cuphyTensorPrm_t>            output_chest_tensorPrm;

    csirsRxDynApiDataset(const std::string& inputFileName, uint32_t max_cells, uint32_t max_ues, cudaStream_t cuStrm, uint64_t procModeBmsk = 0);
    void cumulativeUpdate(const std::string& inputFileName, cudaStream_t cuStrm);

    int  refCheck(bool verbose=false);

private:
    uint32_t max_ues;
    uint32_t max_cells;
    uint32_t total_rrc_params;
    uint16_t nPrecodingMatrices;
    std::vector<std::string> CfgFileName;
    //input
    std::vector<cuphy::tensor_device> data_rx_tensor;
    //output
    std::vector<uint16_t> startPrbChEst;
    std::vector<uint16_t> sizePrbChEst;
    std::vector<cuphy::tensor_device> ch_est_tensor;
};



// srsTxStaticApiDataset
// Contains static API parameters and data for a single SrsTxPipeline

struct srsTxStaticApiDataset{
public:
    cuphySrsTxStatPrms_t  srsTxStatPrms;
    cuphyTracker_t        srsTxTracker;
    srsTxStaticApiDataset(const std::string& inputFileName, int cfg_max_ues_per_slot=1);
    void print();
private:
    std::vector<std::string> CfgFileName;
};

//-------------------------------------------------------------------------------
// srsTxDynApiDataset
// Contains dynamic API parameters and data for a single SrsTxPipeline

struct srsTxDynApiDataset {
public:
    std::vector<cuphyUeSrsTxPrm_t>       ue_srstx_params;
    cuphySrsTxDynPrms_t                  srstx_dyn_params;
    std::vector<cuphyTensorPrm_t>        output_tensorPrm;

    srsTxDynApiDataset(const std::string& inputFileName, uint32_t max_ues, cudaStream_t cuStrm);
    void cumulativeUpdate(const std::string& inputFileName, cudaStream_t cuStrm);

    int  refCheck(bool verbose=false);

private:
    uint32_t max_ues;
    cuphy::unique_device_ptr<__half2> large_buffer; // Allocated during constructor and covers max_cells max. configured CSI-RS output tensors
    std::vector<cuphy::tensor_device> data_tx_tensor;
    uint32_t large_buffer_elements, large_buffer_bytes;
    std::vector<std::string> CfgFileName;
};

// for testing performance curve of pusch, pucch, prach; use to resovle the path of input files
#if defined(_WIN32) || defined(_WIN64)
const char pathSeparator = '\\';
#else
const char pathSeparator = '/';
#endif

#endif // !defined(DATASETS_HPP_INCLUDED_)
