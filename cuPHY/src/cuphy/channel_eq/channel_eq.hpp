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

#if !defined(CHANNEL_EQ_HPP_INCLUDED_)
#define CHANNEL_EQ_HPP_INCLUDED_

#include "tensor_desc.hpp"
#include <vector>
#include <unordered_map>

// Implementation of the channel equalizer interface exposed as an opaque data type to abstract out implementation
// details (channel equalizer C++ class). The channel equalizer is implemented as a C++ class which inherits
// from this interface structure defiend as an empty shell (opaque type is a struct since the interface is C
// compatible). Pointer to the opaque type is also exposed in the interface as a handle to the underlying
// implementation
struct cuphyPuschRxChEq
{};

namespace channel_eq
{

// By choosing return value of getThreadsPerFreqBin() to be a divisor of 32,
// we ensure each frequency-bin group fits entirely within a single warp, preventing any bin from being split across warps.
// (otherwise we'd need to use __syncthreads() instead of __syncwarp() in eqMmseIrcCoefCompLowMimoKernel())
template <uint32_t N_BS_ANTS, uint32_t N_LAYERS>
constexpr uint32_t getThreadsPerFreqBin()
{
    constexpr uint32_t need = std::max(N_BS_ANTS * N_LAYERS, N_BS_ANTS + N_LAYERS);
    if constexpr (need <= 1)  return 1;
    else if constexpr (need <= 2)  return 2;
    else if constexpr (need <= 4)  return 4;
    else if constexpr (need <= 8)  return 8;
    else if constexpr (need <= 16) return 16;
    else return 32;
}

// QAM levels
enum class QAM_t : uint8_t
{
    QAM_4   = CUPHY_QAM_4,
    QAM_16  = CUPHY_QAM_16,
    QAM_64  = CUPHY_QAM_64,
    QAM_256 = CUPHY_QAM_256
};

// Tensor parameters needed to access Channel estimator input/output tensors
template <size_t NDim>
struct puschRxChEqTensorPrm
{
    void* pAddr;
    int   strides[NDim];
};
template <size_t NDim>
using puschRxChEqTensorPrm_t = puschRxChEqTensorPrm<NDim>;

// Channel estimator static descriptor
typedef struct _puschRxChEqStatDescr
{
    cudaTextureObject_t demapper_tex;
    // If enableDebugEqOutput is non-zero, write the post-eq output to tInfoDataEq
    uint8_t enableDebugEqOutput;
} puschRxChEqStatDescr_t;

typedef struct _puschRxChEqIdftStatDescr
{
    cuphyTensorInfo2_t tInfoDftBluesteinWorkspaceTime;
    cuphyTensorInfo2_t tInfoDftBluesteinWorkspaceFreq;
    uint16_t           nBluesteinWorkspaceRows{};    //!< Number of valid Bluestein workspace rows provisioned.
    uint16_t           bluesteinWorkspaceFftWidth{}; //!< FFT width (FFT128..FFT8192) provisioned for Bluestein workspace.
} puschRxChEqIdftStatDescr_t;

// Channel estimator dynamic descriptor
typedef struct _puschRxChEqCoefCompDynDescr
{
    uint8_t                  chEqTimeInstIdx; // Time domain instance of channel estimation
    uint32_t                 hetCfgUeGrpMap[MAX_N_USER_GROUPS_SUPPORTED]; // Mapping of Heterogenous config to UE group
    cuphyPuschRxUeGrpPrms_t* pDrvdUeGrpPrms;
} puschRxChEqCoefCompDynDescr_t;

typedef struct _puschRxChEqSoftDemapDynDescr
{
    uint32_t                 hetCfgUeGrpMap[MAX_N_USER_GROUPS_SUPPORTED]; // Mapping of Heterogenous config to UE group
    cuphyPuschRxUeGrpPrms_t* pDrvdUeGrpPrms;
} puschRxChEqSoftDemapDynDescr_t;

// Channel estimator kernel arguments (supplied via descriptors)
typedef struct _puschRxChEqCoefCompKernelArgs
{
    // puschRxDescrs_t const* pPuschRxDescrs;
    puschRxChEqStatDescr_t*        pStatDescr; // pointer to an array of static descriptors (1 to CUPHY_PUSCH_RX_CH_EQ_N_HOM_CFG)
    puschRxChEqCoefCompDynDescr_t* pDynDescr;  // pointer to an array of CUPHY_PUSCH_RX_CH_EQ_N_HOM_CFG dynamic descriptors
} puschRxChEqCoefCompKernelArgs_t;

typedef struct _puschRxChEqSoftDemapKernelArgs
{
    // puschRxDescrs_t const* pPuschRxDescrs;
    puschRxChEqStatDescr_t*         pStatDescr; // pointer to an array of static descriptors (1 to CUPHY_PUSCH_RX_CH_EQ_N_HOM_CFG)
    puschRxChEqSoftDemapDynDescr_t* pDynDescr;  // pointer to an array of CUPHY_PUSCH_RX_CH_EQ_N_HOM_CFG dynamic descriptors
} puschRxChEqSoftDemapKernelArgs_t;

typedef struct _puschRxChEqSoftDemapIdftKernelArgs
{
    // puschRxDescrs_t const* pPuschRxDescrs;
    puschRxChEqStatDescr_t*         pStatDescr; // pointer to an array of static descriptors (1 to CUPHY_PUSCH_RX_CH_EQ_N_HOM_CFG)
    puschRxChEqSoftDemapDynDescr_t* pDynDescr;  // pointer to an array of CUPHY_PUSCH_RX_CH_EQ_N_HOM_CFG dynamic descriptors
    puschRxChEqIdftStatDescr_t*     pIdftStatDescr;
} puschRxChEqSoftDemapIdftKernelArgs_t;

// Forward declaration for launch parameters
using puschRxChEqCoefCompDynDescrVec_t    = std::array<puschRxChEqCoefCompDynDescr_t, CUPHY_PUSCH_RX_CH_EQ_N_MAX_HET_CFGS>;
using puschRxChEqCoefCompStrmVec_t        = std::array<cudaStream_t, CUPHY_PUSCH_RX_CH_EQ_N_MAX_HET_CFGS>;

using puschRxChEqSoftDemapDynDescrVec_t    = std::array<puschRxChEqSoftDemapDynDescr_t, CUPHY_PUSCH_RX_CH_EQ_N_MAX_HET_CFGS>;
using puschRxChEqSoftDemapStrmVec_t        = std::array<cudaStream_t, CUPHY_PUSCH_RX_CH_EQ_N_MAX_HET_CFGS>;

using puschRxChEqCoefCompKernelArgsArr_t      = std::array<puschRxChEqCoefCompKernelArgs_t, CUPHY_PUSCH_RX_CH_EQ_N_MAX_HET_CFGS>;
using puschRxChEqSoftDemapKernelArgsArr_t     = std::array<puschRxChEqSoftDemapKernelArgs_t, CUPHY_PUSCH_RX_CH_EQ_N_MAX_HET_CFGS>;
using puschRxChEqSoftDemapIdftKernelArgsArr_t = std::array<puschRxChEqSoftDemapIdftKernelArgs_t, CUPHY_PUSCH_RX_CH_EQ_N_MAX_HET_CFGS>;

// Class implementation of the channel equalization component
class puschRxChEq : public cuphyPuschRxChEq {
public:
    puschRxChEq()
    {
        // reserve in unordered map only works for hash buckets, not nodes (key/values pairs)
        // FixMe a custom allocator could be used to preallocate and help to avoid dyn mem alloc at runtime
        m_chEqCoefCompHashTable.reserve(MAX_N_USER_GROUPS_SUPPORTED);
        m_chEqSoftDmpHashTable.reserve(MAX_N_USER_GROUPS_SUPPORTED);
    }
    ~puschRxChEq()                  = default;
    puschRxChEq(puschRxChEq const&) = delete;
    puschRxChEq& operator=(puschRxChEq const&) = delete;

    // initialize channel equalizer object and static component descriptor
    cuphyStatus_t init(cuphyContext_t          ctx,
                       cuphyTensorInfo2_t&     tInfoDftBluesteinWorkspaceTime,
                       cuphyTensorInfo2_t&     tInfoDftBluesteinWorkspaceFreq,
                       uint                    cudaDeviceArch,
                       uint8_t                 enableDftSOfdm,
                       uint8_t                 enableDebugEqOutput,
                       bool                    enableCpuToGpuDescrAsyncCpy,
                       void **                 ppStatDescrCpu,
                       void **                 ppStatDescrGpu,
                       void **                 ppIdftStatDescrCpu,
                       void **                 ppIdftStatDescrGpu,
                       cudaStream_t            strm);

    // setup object state and dynamic component descriptor in prepration towards execution
    // @todo: replace with new API structures once integrated
    cuphyStatus_t setupCoefCompute(cuphyPuschRxUeGrpPrms_t*          pDrvdUeGrpPrmsCpu,
                                   cuphyPuschRxUeGrpPrms_t*          pDrvdUeGrpPrmsGpu,
                                   uint16_t                          nUeGrps,
                                   uint16_t                          nMaxPrb,
                                   uint8_t                           enableCfoCorrection,
                                   uint8_t                           enablePuschTdi,
                                   bool                              enableCpuToGpuDescrAsyncCpy,
                                   void**                            ppDynDescrsCpu,
                                   void**                            ppDynDescrsGpu,
                                   cuphyPuschRxChEqLaunchCfgs_t*     pLaunchCfgs,
                                   cudaStream_t                      strm);

    cuphyStatus_t setupSoftDemap(cuphyPuschRxUeGrpPrms_t*           pDrvdUeGrpPrmsCpu,
                                 cuphyPuschRxUeGrpPrms_t*           pDrvdUeGrpPrmsGpu,
                                 uint16_t                           nUeGrps,
                                 uint16_t                           nMaxPrb,                        
                                 uint8_t                            enableCfoCorrection,
                                 uint8_t                            enablePuschTdi,
                                 uint8_t                            openRanFunctionalSplitOption,
                                 uint8_t                            kernelSelOption,
                                 uint16_t                           symbolBitmask,
                                 bool                               enableCpuToGpuDescrAsyncCpy,
                                 puschRxChEqSoftDemapDynDescrVec_t& dynDescrVecCpu,
                                 void*                              pDynDescrsGpu,
                                 cuphyPuschRxChEqLaunchCfgs_t*      pLaunchCfgs,
                                 cudaStream_t                       strm);
                                 
    cuphyStatus_t setupSoftDemapIdft(cuphyPuschRxUeGrpPrms_t*           pDrvdUeGrpPrmsCpu,
                                     cuphyPuschRxUeGrpPrms_t*           pDrvdUeGrpPrmsGpu,
                                     uint16_t                           nUeGrps,
                                     uint16_t                           nMaxPrb, 
                                     uint                               cudaDeviceArch,                          
                                     uint8_t                            enableCfoCorrection,
                                     uint8_t                            enablePuschTdi,
                                     uint16_t                           symbolBitmask,
                                     bool                               enableCpuToGpuDescrAsyncCpy,
                                     puschRxChEqSoftDemapDynDescrVec_t& dynDescrVecCpu,
                                     void*                              pDynDescrsGpu,
                                     cuphyPuschRxChEqLaunchCfgs_t*      pLaunchCfgs,
                                     cudaStream_t                       strm);
                                 
    cuphyStatus_t setupSoftDemapAfterDft(cuphyPuschRxUeGrpPrms_t*           pDrvdUeGrpPrmsCpu,
                                         cuphyPuschRxUeGrpPrms_t*           pDrvdUeGrpPrmsGpu,
                                         uint16_t                           nUeGrps,
                                         uint16_t                           nMaxPrb,                        
                                         uint8_t                            enableCfoCorrection,
                                         uint8_t                            enablePuschTdi,
                                         uint16_t                           symbolBitmask,
                                         bool                               enableCpuToGpuDescrAsyncCpy,
                                         puschRxChEqSoftDemapDynDescrVec_t& dynDescrVecCpu,
                                         void*                              pDynDescrsGpu,
                                         cuphyPuschRxChEqLaunchCfgs_t*      pLaunchCfgs,
                                         cudaStream_t                       strm);

    static void getDescrInfo(size_t& statDescrSizeBytes,
                             size_t& statDescrAlignBytes,
                             size_t& idftStatDescrSizeBytes,
                             size_t& idftStatDescrAlignBytes,
                             size_t& coefCompDynDescrSizeBytes,
                             size_t& coefCompDynDescrAlignBytes,
                             size_t& softDemapDynDescrSizeBytes,
                             size_t& softDemapDynDescrAlignBytes);

private:
    cuphyStatus_t batchEqCoefComp(uint32_t                           chEqInstIdx,
                         cuphyPuschRxUeGrpPrms_t*           pDrvdUeGrpPrms,
                         uint16_t                           nUeGrps,
                         uint32_t&                          nHetCfgs,
                         puschRxChEqCoefCompDynDescrVec_t&  dynDescrVecCpu);

    void coefCompKernelSelectL1(uint16_t                     nBSAnts,
                                uint8_t                      nLayers,
                                uint16_t                     nPrb,
                                uint16_t                     nUeGrps,
                                cuphyDataType_t              hEstType,
                                cuphyDataType_t              coefType,
                                cuphyPuschRxChEqLaunchCfg_t& launchCfg);

    template <typename TStorageIn, typename TStorageOut, typename TCompute>
    void coefCompKernelSelectL0(uint16_t                     nBSAnts,
                                uint8_t                      nLayers,
                                uint16_t                     nPrb,
                                uint16_t                     nUeGrps,
                                cuphyPuschRxChEqLaunchCfg_t& launchCfg);

    template <typename TStorageIn, typename TStorageOut, typename TCompute, uint32_t N_BS_ANTS, uint32_t N_LAYERS>
    void eqMmseCoefCompMassiveMimo(uint16_t                     nPrb,
                                   uint16_t                     nUeGrps,
                                   cuphyPuschRxChEqLaunchCfg_t& launchCfg);

  template <uint32_t N_LAYERS, uint32_t N_THRD_BLK_TONES, uint32_t N_TONES_PER_ITER>
    void coefCompMassiveMimoKernelLaunchGeo(uint16_t nPrb,
                                            uint16_t nUeGrps,
                                            dim3&    gridDim,
                                            dim3&    blockDim);

    template <typename TStorageIn, typename TStorageOut, typename TCompute, uint32_t N_BS_ANTS, uint32_t N_LAYERS>
    void eqMmseCoefCompHighMimo(uint16_t                     nPrb,
                                uint16_t                     nUeGrps,
                                cuphyPuschRxChEqLaunchCfg_t& launchCfg);

    template <uint32_t N_THRD_BLK_PER_PRB, uint32_t N_TONES_PER_ITER, uint32_t N_THRDS_PER_TONE>
    void coefCompHighMimoKernelLaunchGeo(uint16_t nPrb,
                                         uint16_t nUeGrps,
                                         dim3&    gridDim,
                                         dim3&    blockDim);

    template <typename TStorageIn, typename TStorageOut, typename TCompute, uint32_t N_BS_ANTS, uint32_t N_LAYERS>
    void eqMmseCoefCompLowMimo(uint16_t                     nPrb,
                               uint16_t                     nUeGrps,
                               cuphyPuschRxChEqLaunchCfg_t& launchCfg);

    template <uint32_t N_BS_ANTS, uint32_t N_LAYERS, uint32_t N_FREQ_BINS_PER_ITER>
    void coefCompLowMimoKernelLaunchGeo(uint16_t nPrb,
                                        uint16_t nUeGrps,
                                        dim3&    gridDim,
                                        dim3&    blockDim);

    cuphyStatus_t batchEqSoftDemap(cuphyPuschRxUeGrpPrms_t*           pDrvdUeGrpPrms,
                                   uint16_t                           nUeGrps,
                                   uint8_t                            openRanFunctionalSplitOption,
                                   uint8_t                            kernelSelOption,
                                   uint16_t                           symbolBitmask,
                                   uint32_t&                          nHetCfgs,
                                   puschRxChEqSoftDemapDynDescrVec_t& dynDescrVecCpu);

    cuphyStatus_t batchEqSoftDemapIdft(cuphyPuschRxUeGrpPrms_t*           pDrvdUeGrpPrms,
                                       uint16_t                           nUeGrps,
                                       uint16_t                           symbolBitmask,
                                       uint                               cudaDeviceArch,   
                                       uint32_t&                          nHetCfgs,
                                       puschRxChEqSoftDemapDynDescrVec_t& dynDescrVecCpu);
                          
    cuphyStatus_t batchEqSoftDemapAfterDft(cuphyPuschRxUeGrpPrms_t*           pDrvdUeGrpPrms,
                                           uint16_t                           nUeGrps,
                                           uint16_t                           symbolBitmask,
                                           uint32_t&                          nHetCfgs,
                                           puschRxChEqSoftDemapDynDescrVec_t& dynDescrVecCpu);

    void softDemapKernelSelectL1(uint16_t                     nBSAnts,
                                 uint8_t                      nLayers,
                                 uint8_t                      Nd,
                                 uint16_t                     nPrb,
                                 uint16_t                     nUeGrps,
                                 uint8_t                      openRanFunctionalSplitOption,
                                 uint8_t                      kernelSelOption,
                                 uint16_t                     symbolBitmask,
                                 cuphyDataType_t              coefType,
                                 cuphyDataType_t              dataRxType,
                                 cuphyDataType_t              llrType,
                                 cuphyPuschRxChEqLaunchCfg_t& launchCfg);
                                 
    void softDemapIdftKernelSelectL1(uint8_t                      Nd,
                                     uint16_t                     nPrb,
                                     uint16_t                     nUeGrps,
                                     uint16_t                     symbolBitmask,
                                     uint                         cudaDeviceArch,   
                                     cuphyDataType_t              coefType,
                                     cuphyDataType_t              dataRxType,
                                     cuphyDataType_t              llrType,
                                     cuphyPuschRxChEqLaunchCfg_t& launchCfg);
    
    void softDemapAfterDftKernelSelectL1(uint16_t                     nBSAnts,
                                         uint8_t                      nLayers,
                                         uint8_t                      Nd,
                                         uint16_t                     nPrb,
                                         uint16_t                     nUeGrps,
                                         uint16_t                     symbolBitmask,
                                         cuphyDataType_t              coefType,
                                         cuphyDataType_t              dataRxType,
                                         cuphyDataType_t              llrType,
                                         cuphyPuschRxChEqLaunchCfg_t& launchCfg);

    template <typename TStorageIn, typename TDataRx, typename TStorageOut, typename TCompute>
    void softDemapKernelSelectL0(uint16_t                     nBSAnts,
                                 uint8_t                      nLayers,
                                 uint8_t                      Nd,
                                 uint16_t                     Nprb,
                                 uint16_t                     nUeGrps,
                                 uint8_t                      openRanFunctionalSplitOption,
                                 uint8_t                      kernelSelOption,
                                 uint16_t                     symbolBitmask,
                                 cuphyPuschRxChEqLaunchCfg_t& launchCfg);
                                 
    template <typename TStorageIn, typename TDataRx, typename TStorageOut, typename TCompute>
    void softDemapIdftKernelSelectL0(uint8_t                      Nd,
                                     uint16_t                     Nprb,
                                     uint16_t                     nUeGrps,
                                     uint16_t                     symbolBitmask,
                                     uint                         cudaDeviceArch,   
                                     cuphyPuschRxChEqLaunchCfg_t& launchCfg);
                                 
    template <typename TStorageIn, typename TDataRx, typename TStorageOut, typename TCompute>
    void softDemapAfterDftKernelSelectL0(uint16_t                     nBSAnts,
                                         uint8_t                      nLayers,
                                         uint8_t                      Nd,
                                         uint16_t                     Nprb,
                                         uint16_t                     nUeGrps,
                                         uint16_t                     symbolBitmask,
                                         cuphyPuschRxChEqLaunchCfg_t& launchCfg);

    void softDemapKernelLaunchGeo(uint8_t  Nd,
                                  uint16_t nPrb,
                                  uint16_t nLayers,
                                  uint16_t nUeGrps,
                                  uint8_t  openRanFunctionalSplitOption,
                                  dim3&    gridDim,
                                  dim3&    blockDim);
                                  
    template <uint32_t N_SYMBS_PER_THRD_BLK>
    void softDemapAfterDftKernelLaunchGeo(uint8_t  Nd,
                                  uint16_t nPrb,
                                  uint16_t nUeGrps,
                                  dim3&    gridDim,
                                  dim3&    blockDim);
                                  
    template <uint32_t N_SYMBS_PER_THRD_BLK>
    void softDemapKernelLaunchGeo_64R(uint8_t  Nd,
                                                   uint16_t nPrb,
                                                   uint16_t nUeGrps,
                                                   dim3&    gridDim,
                                                   dim3&    blockDim);

  template <typename TStorageIn, typename TDataRx, typename TStorageOut, typename TCompute, uint32_t N_BS_ANTS>
    void eqMmseSoftDemap(uint8_t                      Nd,
                         uint16_t                     nPrb,
                         uint16_t                     nLayers,
                         uint16_t                     nUeGrps,
                         uint8_t                      openRanFunctionalSplitOption,
                         uint8_t                      kernelSelOption,
                         uint16_t                     symbolBitmask,
                         cuphyPuschRxChEqLaunchCfg_t& launchCfg);
                         
  template <typename TStorageIn, typename TDataRx, typename TStorageOut, typename TCompute>
    void eqMmseSoftDemapIdft(uint8_t                      Nd,
                             uint16_t                     nPrb,
                             uint16_t                     nUeGrps,
                             uint16_t                     symbolBitmask,
                             uint                         cudaDeviceArch,
                             cuphyPuschRxChEqLaunchCfg_t& launchCfg);
                         
  template <typename TStorageIn, typename TDataRx, typename TStorageOut, typename TCompute, uint32_t N_BS_ANTS, uint32_t N_SYMBS_PER_THRD_BLK>
    void eqMmseSoftDemapAfterDft(uint8_t                      Nd,
                                 uint16_t                     nPrb,
                                 uint16_t                     nUeGrps,
                                 uint16_t                     symbolBitmask,
                                 cuphyPuschRxChEqLaunchCfg_t& launchCfg);
                         
  template <typename TStorageIn, typename TDataRx, typename TStorageOut, typename TCompute, uint32_t N_LAYERS, uint32_t N_SYMBS_PER_THRD_BLK>
    void eqMmseSoftDemap_64R(uint8_t                      Nd,
                         uint16_t                     nPrb,
                         uint16_t                     nUeGrps,
                         cuphyPuschRxChEqLaunchCfg_t& launchCfg);

    // class state modifed by setup saved in data member
    puschRxChEqCoefCompKernelArgsArr_t  m_coefCompKernelArgsArr[CUPHY_PUSCH_RX_MAX_N_TIME_CH_EQ];
    puschRxChEqSoftDemapKernelArgsArr_t m_softDemapKernelArgsArr[2], m_softDemapAfterDftKernelArgsArr[2];
    puschRxChEqSoftDemapIdftKernelArgsArr_t m_softDemapIdftKernelArgsArr[2];

    // cached in order to use for template specialization
    bool m_enableDebugEqOutput{false};

    // 1. nRxAnt: 16, nLayer: [1, 2, 4, 8, 16] : 5 templates
    // 2. nRxAnt:  8, nLayer: [1, 2, 4, 8]     : 4 templates 
    // 3. nRxAnt:  4, nLayer: [1, 2, 4]        : 3 templates
    // 4. nRxAnt:  2, nLayer: [1, 2]           : 2 templates
    // # of heterogenous configs needed for nRxAnt x nLayer combos = 14 templates
    // # of heterogenous configs needed for time                   =  4 templates
    // Max # of heterogenous configs needed = 14 * 4 = 56 but capping CUPHY_PUSCH_RX_CH_EQ_N_MAX_HET_CFGS to 8
    // @todo: reduce the number of nRxAnt x nLayer and time-domain templates

    typedef struct _puschRxChEqCoefCompHetCfg
    {
        CUfunction func;
        uint16_t   nMaxPrb; // Maximum number of PRBs across all UE groups corresponding to this heterogenous config
        uint16_t   nUeGrps; // Number of user groups corresponding to this heterogenous config
    } puschRxChEqCoefCompHetCfg_t;

    using puschRxChEqCoefCompHetCfgVec_t = std::vector<puschRxChEqCoefCompHetCfg_t>;
    std::array<puschRxChEqCoefCompHetCfgVec_t, CUPHY_PUSCH_RX_MAX_N_TIME_CH_EQ> m_coefCompHetCfgsVecArr;

    typedef struct _puschRxChEqSoftDemapHetCfg
    {
        CUfunction func;
        uint16_t   nMaxPrb;     // Maximum number of PRBs across all UE groups corresponding to this heterogenous config
        uint16_t   nMaxDataSym; // Maximum number of data symbols across all UE groups corresponding to this heterogenous config
        uint16_t   nMaxLayers;  // Maximum number of layers across all UE groups corresponding to this heterogenous config
        uint16_t   nUeGrps;     // Number of user groups corresponding to this heterogenous config
    } puschRxChEqSoftDemapHetCfg_t;

    using puschRxChEqSoftDemapHetCfgVec_t = std::vector<puschRxChEqSoftDemapHetCfg_t>;
    std::array<puschRxChEqSoftDemapHetCfgVec_t, 2> m_softDemapHetCfgVec;
    
    std::array<puschRxChEqSoftDemapHetCfgVec_t, 2> m_softDemapIdftHetCfgVec;
    std::array<puschRxChEqSoftDemapHetCfgVec_t, 2> m_softDemapAfterDftHetCfgVec;

    struct puschRxChEqHash_t
    {
        std::size_t operator()(const std::tuple<int, int>& comb) const
        {
            // Combine hashes of three integer indices using XOR and multiplication
            return std::get<0>(comb) ^ (std::get<1>(comb) * 17);
        }
    };

    struct chEqHashVal
    {
        CUfunction func;
        int32_t    hetCfgIdx;

        chEqHashVal() = default;
        chEqHashVal(CUfunction f, int32_t idx)
            : func(f), hetCfgIdx(idx) {};
    };

    using chEqHashMap_t     = std::unordered_map<std::tuple<int, int>, chEqHashVal, puschRxChEqHash_t>;
    using softDemapperMap_t = std::unordered_map<int, chEqHashVal>;

    // used in setup functions for coef comp and softdemp to check if a het config has been used previously
    chEqHashMap_t     m_chEqCoefCompHashTable;
    softDemapperMap_t m_chEqSoftDmpHashTable;
};


} // namespace channel_eq

#endif // !defined(CHANNEL_EQ_HPP_INCLUDED_)
