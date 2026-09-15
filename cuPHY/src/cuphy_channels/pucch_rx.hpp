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
#include <vector>
#include <string>
#include "cuphy_hdf5.hpp"
#include "cuphy.hpp"
#include "tensor_desc.hpp"

#if !defined(PUCCH_RX_HPP_INCLUDED_)
#define PUCCH_RX_HPP_INCLUDED_

#define PUCCH_USE_BATCHED_MEMCPY 1 // enables batched async memcpy in combination with static parameter field
#define PUCCH_MAX_OUTPUT_TO_CPU_COPIES 13 //runtime check present in copyOutputToCpu() if batched memcpy enabled

struct cuphyPucchRx
{

};

class PucchRx : public cuphyPucchRx {
public:
    enum Component
    {
        PUCCH_CELL_INFO              = 0,
        PUCCH_F0_Rx                  = 1,
        PUCCH_F1_Rx                  = 2,
        PUCCH_F2_RX                  = 3,
        PUCCH_F3_RX                  = 4,
        PUCCH_F234_UCI_SEG           = 5,
        POL_COMP_CW_TREE             = 6,
        POL_COMP_CW_TREE_ADDRS       = 7,
        POL_SEG_DERM_DEITL           = 8,
        POL_SEG_DERM_DEITL_CW_ADDRS  = 9,
        POL_SEG_DERM_DEITL_UCI_ADDRS = 10,
        POL_DECODE                   = 11,
        POL_DECODE_LLR_ADDRS         = 12,
        POL_DECODE_CB_ADDRS          = 13,
        LIST_POL_DECODE_SCRATCH_ADDRS= 14,
        RM_DECODE                    = 15,
        N_PUCCH_COMPONENTS           = 16
    };
    typedef struct F234RmSizes
    {
        uint16_t E_seg1;
        //uint16_t E_seg2; // Currently unused
    } F234RmSizes_t;
    struct OutputParams
    {
        // flag to copy outputs to CPU after run
        bool                        cpuCopyOn;

        // sizes
        uint16_t                    nF234Ucis;
        uint16_t                    nUciSegs;
        uint32_t                    nUciPayloadBytes;

        // device (GPU) output addresses
        cuphyPucchF0F1UciOut_t* pF0UciOutGpu;
        cuphyPucchF0F1UciOut_t* pF1UciOutGpu;
        uint8_t*                pUciPayloadsGpu;
        uint8_t*                pCrcFlagsGpu;
        uint8_t*                pDtxFlagsGpu;
        float*                  pRssiGpu;
        float*                  pRsrpGpu;
        float*                  pSinrGpu;
        float*                  pInterfGpu;
        float*                  pNoiseVarGpu;
        float*                  pTaEstGpu;
        uint16_t*               pNumCsi2BitsGpu;

        uint8_t*                    pHarqDetectionStatusGpu;
        uint8_t*                    pCsiP1DetectionStatusGpu;
        uint8_t*                    pCsiP2DetectionStatusGpu;

        // host (CPU) output addresses
        cuphyPucchF234OutOffsets_t* pPucchF2OutOffsetsCpu;
        cuphyPucchF234OutOffsets_t* pPucchF3OutOffsetsCpu;
        cuphyPucchF0F1UciOut_t*     pF0UciOutCpu;
        cuphyPucchF0F1UciOut_t*     pF1UciOutCpu;
        uint8_t*                    pUciPayloadsCpu;
        uint8_t*                    pCrcFlagsCpu;
        //uint8_t*                    pDtxFlagsCpu;
        float*                      pRssiCpu;
        float*                      pRsrpCpu;
        float*                      pSinrCpu;
        float*                      pInterfCpu;
        float*                      pNoiseVarCpu;
        float*                      pTaEstCpu;
        //uint16_t*                   pNumCsi2BitsCpu;
        cuphyTensorPrm_t*           pF2FrontEndLLRs;
        cuphyTensorPrm_t*           pF3FrontEndLLRs;

        // HARQ/CSI part 1/CSI part 2 detection status. Refer to SCF FAPIv10.04, table 3–125, 126, 127 
        uint8_t*                    pHarqDetectionStatusCpu;
        uint8_t*                    pCsiP1DetectionStatusCpu;
        uint8_t*                    pCsiP2DetectionStatusCpu;

        // debug parameters
        bool                        debugOutputFlag;
        hdf5hpp::hdf5_file          outHdf5File;
    };
    PucchRx(cuphyPucchStatPrms_t const* pStatPrms, cudaStream_t strm);
    PucchRx(PucchRx const&) = delete;
    PucchRx& operator=(PucchRx const&) = delete;
    ~PucchRx();

    [[nodiscard]] cuphyStatus_t setup(cuphyPucchDynPrms_t *pDynPrm);
    [[nodiscard]] cuphyStatus_t run();
    void writeDbgBufSynch(cudaStream_t cuStream);
    const void* getMemoryTracker();

    // Load caller-provided post-polar outputs for PUCCH_PIPELINE_SKIP_POLAR.
    // This populates the internal buffers normally written by the polar decoder
    // so the downstream UCI segmentation stage can run unchanged.
    [[nodiscard]] cuphyStatus_t loadPostPolarDataForSkip(const cuphyPucchPostPolarData_t& postPolarData,
                                                         cudaStream_t                    cuStream);

    template <fmtlog::LogLevel log_level=fmtlog::DBG>
    static void printUciPrms(const cuphyPucchUciPrm_t* uciPrms);
    template <fmtlog::LogLevel log_level=fmtlog::DBG>
    static void printDynApiPrms(cuphyPucchDynPrms_t* pDynPrm);
    template <fmtlog::LogLevel log_level=fmtlog::DBG>
    static void printStaticApiPrms(cuphyPucchStatPrms_t const* pStaticPrm);

private:
    cuphyMemoryFootprint m_memoryFootprint;

    // creation functions
    size_t getBufferSize(cuphyPucchStatPrms_t const* pStatPrms);
    void   allocateDescr(void);
    void   createComponents(void);

    // setup functions
    F234RmSizes_t compRateMatchSizesF2(cuphyPucchUciPrm_t& F2uciPrms);
    F234RmSizes_t compRateMatchSizesF3(cuphyPucchUciPrm_t& F3uciPrms);

    void        expandF234UciParameters(cuphyPucchUciPrm_t& uciPrms, F234RmSizes_t& rmSizes, uint16_t nBitsUciSeg1);
    void        expandUciCodingPrms(cuphyPucchUciPrm_t& uciPrms, uint32_t nInfoBits, uint32_t nRmBits);

    cuphyStatus_t setupCmn(cuphyPucchDynPrms_t *pDynPrm);

    void allocateDeviceMemory(void);
    void* getFrontEndLlrOutputAddr(cuphyTensorPrm_t* pFrontEndLLRs,
                                    uint16_t uciIdx,
                                    uint16_t expectedElems,
                                    const char* fmtName) const;
    void bindFrontEndLlrOutputAddrs();
    void allocateBackendBuffers(cuphyPucchUciPrm_t&           uciPrms, 
                                cuphyPucchF234OutOffsets_t&   outOffsets, 
                                F234RmSizes_t&                rmSizes,
                                void*                         pSeg1LLRs,
                                uint16_t                      nBitsUciSeg1,
                                uint16_t&                     F234uciIdx,
                                uint16_t&                     rmCwIdx,
                                uint16_t&                     polSegIdx,
                                uint32_t&                     polarWordOffset);
    
    cuphyStatus_t setupComponents(bool enableCpuToGpuDescrAsyncCpy, cuphyPucchDynPrms_t *pDynPrm);

    // graph functions
    void createGraph();
    void updateGraph();

    // run functions
    cuphyStatus_t copyOutputToCPU();
    // destroy functions
    void destroyComponents();

    // kernel descriptors
    cuphy::kernelDescrs<N_PUCCH_COMPONENTS>   m_kernelStatDescr;
    cuphy::kernelDescrs<N_PUCCH_COMPONENTS>   m_kernelDynDescr;

    // stream worker
    cudaStream_t  m_cuStream;

    // pipeline parameters
    cuphyPucchCellPrm_t*         m_pCellCmnPrms{nullptr};
    cuphyPucchStatPrms_t         m_cuphyPucchStatPrms;
    cuphyPucchCellGrpDynPrm_t    m_cuphyPucchCellGrpDynPrm;

    cuphyCellStatPrm_t*&         m_pCellStatPrms = m_cuphyPucchStatPrms.pCellStatPrms;
    cuphyPucchCellDynPrm_t*&     m_pCellDynPrms  = m_cuphyPucchCellGrpDynPrm.pCellPrms;
    uint16_t&                    m_nF0Ucis       = m_cuphyPucchCellGrpDynPrm.nF0Ucis;
    cuphyPucchUciPrm_t*&         m_pF0UciPrms    = m_cuphyPucchCellGrpDynPrm.pF0UciPrms;  
    uint16_t&                    m_nF1Ucis       = m_cuphyPucchCellGrpDynPrm.nF1Ucis;
    cuphyPucchUciPrm_t*&         m_pF1UciPrms    = m_cuphyPucchCellGrpDynPrm.pF1UciPrms;
    uint16_t&                    m_nF2Ucis       = m_cuphyPucchCellGrpDynPrm.nF2Ucis;
    cuphyPucchUciPrm_t*&         m_pF2UciPrms    = m_cuphyPucchCellGrpDynPrm.pF2UciPrms;
    uint16_t&                    m_nF3Ucis       = m_cuphyPucchCellGrpDynPrm.nF3Ucis;
    cuphyPucchUciPrm_t*&         m_pF3UciPrms    = m_cuphyPucchCellGrpDynPrm.pF3UciPrms;

    // input/intermediate/output buffers
    cuphy::linear_alloc<128, cuphy::device_alloc>         m_LinearAlloc;
    OutputParams                                          m_outputPrms;
    cuphy::buffer<cuphyTensorPrm_t, cuphy::pinned_alloc>  m_tPrmDataRxBufCpu;
    std::vector<cuphy::tensor_ref>                        m_tRefDataRxBufCpu;
    cuphy::tensor_ref                                     m_tUciPayload;
    cuphy::tensor_ref                                     m_tDtxFlags;
    cuphy::tensor_ref                                     m_tHarqDetectionStatus;
    cuphy::tensor_ref                                     m_tCsiP1DetectionStatus;
    cuphy::tensor_ref                                     m_tCsiP2DetectionStatus;
    cuphy::tensor_ref                                     m_tSinr;
    cuphy::tensor_ref                                     m_tRssi;
    cuphy::tensor_ref                                     m_tRsrp;
    cuphy::tensor_ref                                     m_tInterf;
    cuphy::tensor_ref                                     m_tNoiseVar;
    cuphy::tensor_ref                                     m_tTaEst;

    // Component handles 
    cuphyPucchF0RxHndl_t         m_pucchF0RxHndl;
    cuphyPucchF1RxHndl_t         m_pucchF1RxHndl;
    cuphyPucchF2RxHndl_t         m_pucchF2RxHndl;
    cuphyPucchF3RxHndl_t         m_pucchF3RxHndl;
    cuphyPucchF234UciSegHndl_t   m_pucchF234UciSegHndl;
    cuphyRmDecoderHndl_t         m_rmDecodeHndl;
    cuphyCompCwTreeTypesHndl_t   m_compCwTreeTypesHndl;
    cuphyPolSegDeRmDeItlHndl_t   m_polSegDeRmDeItlHndl;
    cuphyPolarDecoderHndl_t      m_polarDecoderHndl;

    // kernel launch configurations
    cuphyPucchF0RxLaunchCfg_t        m_pucchF0RxLaunchCfg;
    cuphyPucchF1RxLaunchCfg_t        m_pucchF1RxLaunchCfg;
    cuphyPucchF2RxLaunchCfg_t        m_pucchF2RxLaunchCfg;
    cuphyPucchF3RxLaunchCfg_t        m_pucchF3RxLaunchCfg;
    cuphyPucchF234UciSegLaunchCfg_t  m_pucchF234UciSegLaunchCfg;
    cuphyCompCwTreeTypesLaunchCfg_t  m_compCwTreeTypesLaunchCfg;
    cuphyPolSegDeRmDeItlLaunchCfg_t  m_polSegDeRmDeItlLaunchCfg;
    cuphyPolarDecoderLaunchCfg_t     m_polarDecoderLaunchCfg;
    cuphyRmDecoderLaunchCfg_t        m_rmDecoderLaunchCfg;

    // graph parameters
    bool        m_cudaGraphModeEnabled;
    bool        m_skipPolarDecoder;
    bool        m_skipBackend; // PUCCH_PIPELINE_SKIP_BACKEND: also skip RM + UciSeg parser
    const cuphyPucchPostPolarData_t* m_pPostPolarData; // optional SKIP_POLAR source data
    uint32_t    m_pipelineDelayUs;
    CUgraph     m_graph;
    CUgraphExec m_graphExec;
    CUgraphNode m_emptyRootNode;
    CUgraphNode m_pucchF0RxKernelNode;
    CUgraphNode m_pucchF1RxKernelNode;
    CUgraphNode m_pucchF2RxKernelNode;
    CUgraphNode m_pucchF3RxKernelNode;
    CUgraphNode m_rmDecoderKernelNode;
    CUgraphNode m_compCwTreeTypesKernelNode;
    CUgraphNode m_polSegDeRmDeItlKernelNode;
    CUgraphNode m_polarDecoderKernelNode;
    CUgraphNode m_pucchF234RxKernelNode;
    CUgraphNode m_delayKernelNode;

    // Used to avoid calling cuGraphNodeSetEnabled if status unchanged
    bool m_pucchF0RxKernelNodeEnabled;
    bool m_pucchF1RxKernelNodeEnabled;
    bool m_pucchF2RxKernelNodeEnabled;
    bool m_pucchF3RxKernelNodeEnabled;
    bool m_rmDecoderKernelNodeEnabled;
    bool m_polSegKernelNodesEnabled; // covers m_compCwTreeTypesKernelNode, m_polSegDeRmDeItlKernelNode and  m_polarDecoderKernelNode
    bool m_pucchF234RxKernelNodeEnabled;
    bool m_delayKernelNodeEnabled;

    CUDA_KERNEL_NODE_PARAMS m_emptyNode0paramDriver, m_emptyNode1paramDriver;
    CUDA_KERNEL_NODE_PARAMS m_delayKernelNodeParamsDriver;
    uint32_t                m_delayKernelArgUs;
    void*                   m_delayKernelArgs[1];


    // F2 buffers
    std::vector<__half*>        m_F2seg1LLRaddrsVec;
    std::vector<F234RmSizes_t>  m_F2RmSizesVec;
    uint8_t*                    m_pF2dtxFlags{nullptr};
    float*                      m_pF2pSinr{nullptr};
    float*                      m_pF2pRssi{nullptr};
    float*                      m_pF2pRsrp{nullptr};
    float*                      m_pF2pInterf{nullptr};
    float*                      m_pF2pNoiseVar{nullptr};
    float*                      m_pF2pTaEst{nullptr};
    std::vector<uint16_t>       m_F2nBitsUciSeg1;

    // F3 buffers
    std::vector<__half*>        m_F3seg1LLRaddrsVec;
    std::vector<F234RmSizes_t>  m_F3RmSizesVec;
    uint8_t*                    m_pF3dtxFlags{nullptr};
    float*                      m_pF3pSinr{nullptr};
    float*                      m_pF3pRssi{nullptr};
    float*                      m_pF3pRsrp{nullptr};
    float*                      m_pF3pInterf{nullptr};
    float*                      m_pF3pNoiseVar{nullptr};
    float*                      m_pF3pTaEst{nullptr};
    std::vector<uint16_t>       m_F3nBitsUciSeg1;

    // Polar decoder parameters
    uint8_t                                                   m_polDcdrListSz;
    uint16_t                                                  m_nPolSegs{0};
    uint16_t                                                  m_nPolCbs{0};
    cuphy::buffer<cuphyPolarUciSegPrm_t, cuphy::pinned_alloc> m_polSegPrmsBufCpu;
    cuphyPolarUciSegPrm_t*                                    m_pPolSegPrmsGpu{nullptr};
    cuphy::buffer<cuphyPolarCwPrm_t, cuphy::pinned_alloc>     m_polCwPrmsBufCpu;
    cuphyPolarCwPrm_t*                                        m_pPolCwPrmsGpu{nullptr};
    std::vector<uint8_t*>                                     m_polCwTreeTypesAddrVec;
    std::vector<__half*>                                      m_polSegLLRsAddrVec;
    std::vector<__half*>                                      m_polCwLLRsAddrVec;
    std::vector<__half*>                                      m_polCwTreeLLRsAddrVec;
    std::vector<uint32_t*>                                    m_polCbEstAddrVec;
    std::vector<bool*>                                        m_listPolScratchAddrVec;
    uint8_t*                                                  m_pPolCrcFlags{nullptr};
    std::vector<uint32_t*>                                    m_pUciSegEst;

    // Reed Muller decoder parameters
    uint16_t                                                  m_nRmCws{0};
    cuphy::buffer<cuphyRmCwPrm_t, cuphy::pinned_alloc>        m_rmCwPrmsBufCpu;
    cuphyRmCwPrm_t*                                           m_pRmCwPrmsGpu{nullptr};

    uint8_t                                                   m_pucchEnableUlRxBf;

    // Batched memcpy helper object
    cuphyBatchedMemcpyHelper m_batchedMemcpyHelper;
};

#endif // !defined(PUCCH_RX_HPP_INCLUDED_)



