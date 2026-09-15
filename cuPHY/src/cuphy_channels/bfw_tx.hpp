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

#if !defined(BFW_TX_HPP_INCLUDED_)
#define BFW_TX_HPP_INCLUDED_
#include "cuphy_hdf5.hpp"
#include <cuda_runtime.h>

// enables batched async memcpy in combination with static parameter field
static constexpr bool BFW_TX_USE_BATCHED_MEMCPY = true;

struct cuphyBfwTx
{};

class BfwTx : public cuphyBfwTx
{
public:
    enum DescriptorTypes
    {
        BFW_COEF_COMP                    = 0,
        BFW_COEF_COMP_HET_CFG_UE_GRP_MAP = BFW_COEF_COMP                    + 1,
        BFW_COEF_COMP_UE_GRP_PRMS        = BFW_COEF_COMP_HET_CFG_UE_GRP_MAP + 1,
        BFW_COEF_COMP_LAYER_PRMS         = BFW_COEF_COMP_UE_GRP_PRMS        + 1,
        N_BFW_TX_DESCR_TYPES             = BFW_COEF_COMP_LAYER_PRMS         + 1
    };

    BfwTx(cuphyBfwStatPrms_t const* pStatPrms, cudaStream_t cuStrm);
    BfwTx(BfwTx const&)            = delete;
    BfwTx& operator=(BfwTx const&) = delete;
    ~BfwTx();

    // void writeDbgBufSynch(cudaStream_t cuStrm);

    [[nodiscard]] cuphyStatus_t setup(cuphyBfwDynPrms_t* pDynPrm);
    [[nodiscard]] cuphyStatus_t run(uint64_t procModeBmsk);
    void destroyComponents();
    const void* getMemoryTracker();

    // uint32_t processApi(cuphyBfwDynPrms_t* pDynPrm, cuphyPuschRxUeGrpPrms_t* pDrvdUeGrpPrms, uint8_t enableRssiMeasurement);

    // debug functions:
    template <fmtlog::LogLevel log_level=fmtlog::DBG>
    static void printStaticApiPrms(cuphyBfwStatPrms_t const* pStaticPrms);
    template <fmtlog::LogLevel log_level=fmtlog::DBG>
    static void printDynApiPrms(cuphyBfwDynPrms_t const* pDynPrms);
    void writeDbgBufSynch(cudaStream_t cuStream);

private:
    size_t getBufferSize(cuphyBfwStatPrms_t const* pStatPrms);
    void allocateDescrs(cuphyBfwStatPrms_t const* pStatPrm);
    void createComponents(cuphyBfwStatPrms_t const* pStatPrm);
    void createGraphExec();
    void updateGraphExec();

    static constexpr int getNoOfMemcopies() {
        #ifdef BFW_BOTH_COMP_FLOAT
            return 2;  // 2 device to host copies when BFW_BOTH_COMP_FLOAT is defined in BfwTx::run
        #else
            return 1;  // just 1 device to host copy when BFW_BOTH_COMP_FLOAT is not defined in BfwTx::run
        #endif
    }

    // intermediate/output buffers
    static constexpr uint32_t LINEAR_ALLOC_PAD_BYTES = 128;

    /// count GPU memory usage
    /// @note:
    ///  m_LinearAlloc is using memoryFootprint, so m_memoryFootprint needs appear before
    ///  in the class layout.
    cuphyMemoryFootprint m_memoryFootprint;

    // Maximum bundle size (assumes beam ID header present). Used for allocations.
    size_t             m_bundleSize{};
    // Actual bundle size for the current run (may exclude beam ID header)
    size_t             m_bundleSizeRun{};

    cuphy::linear_alloc<LINEAR_ALLOC_PAD_BYTES, cuphy::device_alloc> m_LinearAlloc; // linear buffer holding intermediate results
#ifdef BFW_BOTH_COMP_FLOAT    
    std::vector<cuphyTensorPrm_t>             m_tPrmBfwVec;
    std::vector<cuphy::tensor_ref>            m_tRefBfwVec;
#endif
    std::vector<uint8_t*>                     m_bfwComppVec;

    cuphy::kernelDescrs<N_BFW_TX_DESCR_TYPES> m_kernelStatDescr;
    cuphy::kernelDescrs<N_BFW_TX_DESCR_TYPES> m_kernelDynDescr;

    cuphyBfwCoefCompHndl_t m_bfwCoefCompHndl{};
    cuphyBfwCoefCompLaunchCfgs_t m_bfwCoefCompLaunchCfgs{};

    CUgraphNode m_emptyRootNode{};
    CUDA_KERNEL_NODE_PARAMS m_emptyNodePrms{};

    // Paramaters: 
    cuphyBfwDynPrms_t* m_pDynPrms{};
    uint8_t            m_compressBitwidth{};
    uint16_t           m_nMaxGnbAnt{};

    // debug paramaters:
    bool               m_debugOutputFlag{};
    hdf5hpp::hdf5_file m_outHdf5File;

    // CUDA stream on which are launched
    cudaStream_t m_cuStrm{};

    bool m_hostOutputBuffer = true;

    // Graph parameters
    bool m_enableGraph = false;
    cudaGraph_t     m_graph{};
    cudaGraphExec_t m_graphExec{};

#if CUDA_VERSION >= 13020
    // Attribute value for kernel nodes that may need dynamic shared memory opt-in
    CUkernelNodeAttrValue m_kernelNodeAttrValue{.sharedMemoryMode = CU_SHARED_MEMORY_MODE_ALLOW_NON_PORTABLE};
#endif
    CUgraphNode m_bfwCoefCompNodes[CUPHY_BFW_COEF_COMP_N_MAX_HET_CFGS]{};
    int32_t m_prevBfwCoefCompNodesCfgs{};
    // Batched memcpy helper object
    cuphyBatchedMemcpyHelper m_batchedMemcpyHelperD2H;

    bool               m_useKernelCopy{};
};
#endif // !defined(BFW_TX_HPP_INCLUDED_)
