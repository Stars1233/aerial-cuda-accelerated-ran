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

#if !defined(CH_EST_HPP_INCLUDED_)
#define CH_EST_HPP_INCLUDED_

#include "tensor_desc.hpp"
#include <unordered_map>

#include "IModule.hpp"
#include "pusch_start_kernels.hpp"
#include "ch_est_graph_mgr.hpp"
#include "ch_est_types.hpp"
#include "ch_est_settings.hpp"
#include "ch_est_stream.hpp"

namespace ch_est
{
/**
 * @brief concreate implementation of IKernelBuilder
 * @see IKernelBuilder for API declaration.
 */
class puschRxChEstKernelBuilder final : public IKernelBuilder {
public:
    puschRxChEstKernelBuilder() {
        // reserve in unordered map only works for hash buckets, not nodes (key/values pairs)
        m_ChEstHashTable.reserve(MAX_N_USER_GROUPS_SUPPORTED);
    }

    void init(gsl_lite::span<uint8_t*> ppStatDescrsCpu,
              gsl_lite::span<uint8_t*> ppStatDescrsGpu,
              bool enableCpuToGpuDescrAsyncCpy,
              cudaStream_t strm) final;

    // IKernelBuilder interface
    [[nodiscard]] cuphyStatus_t
    build(gsl_lite::span<cuphyPuschRxUeGrpPrms_t>         pDrvdUeGrpPrmsCpu,
          gsl_lite::span<cuphyPuschRxUeGrpPrms_t>         pDrvdUeGrpPrmsGpu,
          uint16_t                                   nUeGrps,
          uint8_t                                    maxDmrsMaxLen,
          uint8_t                                    enableDftSOfdm,
          uint8_t                                    chEstAlgo,
          uint8_t                                    enableUlRxBf,
          uint8_t                                    enablePerPrgChEst,
          uint8_t*                                   pPreEarlyHarqWaitKernelStatusGpu,
          uint8_t*                                   pPostEarlyHarqWaitKernelStatusGpu,
          uint16_t                                   waitTimeOutPreEarlyHarqUs,
          uint16_t                                   waitTimeOutPostEarlyHarqUs,
          bool                                       enableCpuToGpuDescrAsyncCpy,
          gsl_lite::span<uint8_t*>                        ppDynDescrsCpu,
          gsl_lite::span<uint8_t*>                        ppDynDescrsGpu,
          pusch::IStartKernels*                      pStartKernels,
          gsl_lite::span<cuphyPuschRxChEstLaunchCfgs_t>   launchCfgs,
          uint8_t                               enableEarlyHarqProc,
          uint8_t                               enableFrontLoadedDmrsProc,
          uint8_t                               enableDeviceGraphLaunch,
          CUgraphExec*                          pSubSlotDeviceGraphExec,
          CUgraphExec*                          pFullSlotDeviceGraphExec,
          cuphyPuschRxWaitLaunchCfg_t*          pWaitKernelLaunchCfgsPreSubSlot,
          cuphyPuschRxWaitLaunchCfg_t*          pWaitKernelLaunchCfgsPostSubSlot,
          cuphyPuschRxDglLaunchCfg_t*           pDglKernelLaunchCfgsPreSubSlot,
          cuphyPuschRxDglLaunchCfg_t*           pDglKernelLaunchCfgsPostSubSlot,
          cudaStream_t                          strm) final;

private:
    // kernelSelectL0 -  5 templates (PRB cluster sizes)
    // kernelSelectL1 - 11 templates (Layers, DMRS symbols, DMRS grids)
    // Max # of heterogenous configs needed = 11 * 5 = 55 but capping CUPHY_PUSCH_RX_CH_EST_N_MAX_HET_CFGS to 16
    // @todo: reduce the number of nRxAnt x nLayer and time-domain templates

    template <typename TCompute>
    cuphyStatus_t batch(uint32_t                                 chEstInstIdx,
                        gsl_lite::span<cuphyPuschRxUeGrpPrms_t>       pDrvdUeGrpPrms,
                        uint16_t                       nUeGrps,
                        uint8_t                        enableDftSOfdm,
                        uint8_t                        chEstAlgo,
                        uint8_t                        enablePerPrgChEst,
                        uint32_t&                      nHetCfgs,
                        puschRxChEstDynDescrVec_t&     dynDescrVecCpu);

    template <typename TCompute>
    void kernelSelectL2(uint16_t                        nBSAnts,
                        uint8_t                         nLayers,
                        uint8_t                         nDmrsSyms,
                        uint8_t                         nDmrsGridsPerPrb,
                        uint8_t                         enablePerPrgChEstPerUeg,
                        uint16_t                        nTotalDataPrb,
                        uint8_t                         Nh,
                        uint16_t                        nUeGrps,
                        uint8_t                         enableDftSOfdm,
                        uint8_t                         chEstAlgo,
                        uint8_t                         enablePerPrgChEst,
                        cuphyDataType_t                 dataRxType,
                        cuphyDataType_t                 hEstType,
                        cuphyPuschRxChEstLaunchCfg_t&   launchCfg);

    template <typename TStorage, typename TDataRx, typename TCompute>
    void kernelSelectL1(uint16_t                        nBSAnts,
                        uint8_t                         nLayers,
                        uint8_t                         nDmrsSyms,
                        uint8_t                         nDmrsGridsPerPrb,
                        uint8_t                         enablePerPrgChEstPerUeg,
                        uint16_t                        nTotalDataPrb,
                        uint8_t                         Nh,
                        uint16_t                        nUeGrps,
                        uint8_t                         enableDftSOfdm,
                        uint8_t                         chEstAlgo,
                        uint8_t                         enablePerPrgChEst,
                        cuphyPuschRxChEstLaunchCfg_t&   launchCfg);

    void rkhsKernelSelectL1(uint16_t nTotalDataPrb, cuphyPuschRxChEstLaunchCfg_t&  launchCfg);

    template <typename TStorage,
              typename TDataRx,
              typename TCompute,
              uint32_t N_LAYERS,
              uint32_t N_DMRS_GRIDS_PER_PRB,
              uint32_t N_DMRS_SYMS>
    void kernelSelectL0(uint8_t                         enablePerPrgChEstPerUeg,
                        uint16_t                        nTotalDataPrb,
                        uint16_t                        nUeGrps,
                        uint32_t                        nRxAnt,
                        uint8_t                         enableDftSOfdm,
                        uint8_t                         chEstAlgo,
                        uint8_t                         enablePerPrgChEst,
                        cuphyPuschRxChEstLaunchCfg_t&   launchCfg);


    template <typename TStorage,
              typename TDataRx,
              typename TCompute,
              uint32_t N_LAYERS,
              uint32_t N_DMRS_GRIDS_PER_PRB,
              uint32_t N_DMRS_PRB_IN_PER_CLUSTER,
              uint32_t N_DMRS_INTERP_PRB_OUT_PER_CLUSTER,
              uint32_t N_DMRS_SYMS>
    void windowedChEst(uint16_t                        nTotalDataPrb,
                       uint16_t                        nUeGrps,
                       uint32_t                        nRxAnt,
                       uint8_t                         enabelDftSOfdm,
                       cuphyPuschRxChEstLaunchCfg_t&   launchCfg);


    template <typename TStorage,
              typename TDataRx,
              typename TCompute,
              uint32_t N_LAYERS,
              uint32_t N_PRBS,
              uint32_t N_DMRS_GRIDS_PER_PRB,
              uint32_t N_DMRS_SYMS>
    void smallChEst(uint16_t                        nUeGrps,
                    uint32_t                        nRxAnt,
                    uint8_t                         enabelDftSOfdm,
                    cuphyPuschRxChEstLaunchCfg_t&   launchCfg);

    template <uint32_t N_DMRS_GRIDS_PER_PRB,
              uint32_t N_DMRS_PRB_IN_PER_CLUSTER,
              uint32_t N_DMRS_INTERP_PRB_OUT_PER_CLUSTER>
    void
    computeKernelLaunchGeo(uint16_t nTotalDataPrb,
                           uint16_t nUeGrps,
                           uint32_t nRxAnt,
                           dim3&    gridDim,
                           dim3&    blockDim);

    template <typename TStorage,
              typename TDataRx,
              typename TCompute,
              uint32_t N_LAYERS,
              uint32_t N_DMRS_GRIDS_PER_PRB,
              uint32_t N_DMRS_PRB_IN_PER_CLUSTER,
              uint32_t N_DMRS_INTERP_PRB_OUT_PER_CLUSTER,
              uint32_t N_DMRS_SYMS>
    void multiStageChEst(uint8_t                         enablePerPrgChEstPerUeg,
                         uint16_t                        nTotalDataPrb,
                         uint16_t                        nUeGrps,
                         uint32_t                        nRxAnt,
                         uint8_t                         enableDftSOfdm,
                         uint8_t                         enablePerPrgChEst,
                         cuphyPuschRxChEstLaunchCfg_t&   launchCfg);

    template <typename TStorage,
              typename TDataRx,
              typename TCompute,
              uint32_t N_LAYERS,
              uint32_t N_DMRS_GRIDS_PER_PRB,
              uint32_t N_DMRS_PRB_IN_PER_CLUSTER,
              uint32_t N_DMRS_INTERP_PRB_OUT_PER_CLUSTER,
              uint32_t N_DMRS_SYMS>
    void lsChEst(uint16_t                        nTotalDataPrb,
                 uint16_t                        nUeGrps,
                 uint32_t                        nRxAnt,
                 uint8_t                         enableDftSOfdm,
                 cuphyPuschRxChEstLaunchCfg_t&   launchCfg);

    // class state modified by setup saved in data member
    puschRxChEstKernelArgsArr_t m_kernelArgsArr[CUPHY_PUSCH_RX_MAX_N_TIME_CH_EST]{};

    typedef struct _puschRxChEstHetCfg
    {
        CUfunction func{};
        uint16_t   nMaxPrb{}; // Maximum number of PRBs across all UE groups
        uint16_t   nMaxRxAnt{}; // Maximum number of Rx Antenna across all UE groups
        uint16_t   nUeGrps{};
    } puschRxChEstHetCfg_t;
    using puschRxChEstHetCfgArr_t = std::array<puschRxChEstHetCfg_t, CUPHY_PUSCH_RX_CH_EST_ALL_ALGS_N_MAX_HET_CFGS>;
    puschRxChEstHetCfgArr_t m_hetCfgsArr[CUPHY_PUSCH_RX_MAX_N_TIME_CH_EST]{};

    struct puschRxChEstHash_t
    {
        std::size_t operator()(const std::tuple<int, int, int, int, int, int>& comb) const
        {
            // Combine hashes of three integer indices using XOR and multiplication
            return std::get<0>(comb) ^ (std::get<1>(comb) * 7) ^ (std::get<2>(comb) * 13) ^ (std::get<3>(comb) * 17) ^ (std::get<4>(comb) * 31) ^ (std::get<5>(comb) * 63);
        }
    };

    struct chEstHashVal
    {
        CUfunction func{};
        int32_t    hetCfgIdx{};

        chEstHashVal() = default;
        chEstHashVal(CUfunction f, int32_t idx)
                : func(f), hetCfgIdx(idx) {};
    };

    using chEstHashMap_t = std::unordered_map<std::tuple<int, int, int, int, int, int>, chEstHashVal, puschRxChEstHash_t>;

    // used in setup function for channel estimation to check if a het config has been used previously
    chEstHashMap_t m_ChEstHashTable;
};

/**
 * @brief Class implementation of the channel estimation component
 */
class puschRxChEst final : public IModule
{
public:
    puschRxChEst(const cuphyChEstSettings& chEstSettings,
                 bool earlyHarqModeEnabled);
    puschRxChEst(puschRxChEst const&) = delete;
    puschRxChEst& operator=(puschRxChEst const&) = delete;

    // initialize channel estimator object and static component descriptor
    void init(IKernelBuilder*     pKernelBuilder,
              bool                enableCpuToGpuDescrAsyncCpy,
              gsl_lite::span<uint8_t*> ppStatDescrsCpu,
              gsl_lite::span<uint8_t*> ppStatDescrsGpu,
              cudaStream_t        strm) final;

    // setup object state and dynamic component descriptor in prepration towards execution
    // @todo: replace with new API structures once integrated
    [[nodiscard]]
    cuphyStatus_t setup(IKernelBuilder*                       pKernelBuilder,
                        gsl_lite::span<cuphyPuschRxUeGrpPrms_t>    pDrvdUeGrpPrmsCpu,
                        gsl_lite::span<cuphyPuschRxUeGrpPrms_t>    pDrvdUeGrpPrmsGpu,
                        uint16_t                              nUeGrps,
                        uint8_t                               maxDmrsMaxLen,
                        uint8_t*                              pPreEarlyHarqWaitKernelStatusGpu,
                        uint8_t*                              pPostEarlyHarqWaitKernelStatusGpu,
                        uint16_t                              waitTimeOutPreEarlyHarqUs,
                        uint16_t                              waitTimeOutPostEarlyHarqUs,
                        bool                                  enableCpuToGpuDescrAsyncCpy,
                        gsl_lite::span<uint8_t*>                   ppDynDescrsCpu,
                        gsl_lite::span<uint8_t*>                   ppDynDescrsGpu,
                        uint8_t                               enableEarlyHarqProc,
                        uint8_t                               enableFrontLoadedDmrsProc,
                        uint8_t                               enableDeviceGraphLaunch,
                        CUgraphExec*                          pSubSlotDeviceGraphExec,
                        CUgraphExec*                          pFullSlotDeviceGraphExec,
                        cuphyPuschRxWaitLaunchCfg_t*          pWaitKernelLaunchCfgsPreSubSlot,
                        cuphyPuschRxWaitLaunchCfg_t*          pWaitKernelLaunchCfgsPostSubSlot,
                        cuphyPuschRxDglLaunchCfg_t*           pDglKernelLaunchCfgsPreSubSlot,
                        cuphyPuschRxDglLaunchCfg_t*           pDglKernelLaunchCfgsPostSubSlot,
                        cudaStream_t                          strm) final;

    static void getDescrInfo(size_t& statDescrSizeBytes, size_t& statDescrAlignBytes, size_t& dynDescrSizeBytes, size_t& dynDescrAlignBytes);

    void setEarlyHarqModeEnabled(bool earlyHarqModeEnabled) final;

    // Set of functions related to the graph management API/abstract interface
    IChestGraphNodes& chestGraph() final;
    IChestStream&     chestStream() final;
    IChestSubSlotNodes& earlyHarqGraph() final;
    IChestSubSlotNodes& frontDmrsGraph() final;
    pusch::IStartKernels& startKernels() final;
private:
    const cuphyChEstSettings&      m_chEstSettings;
    ChannelEstimateGraphMgr        m_chestGraphMgr;
    pusch::StartKernels            m_startKernels;
    ChestStream                    m_chestStream; // Stream CUDA operations
};

} // namespace ch_est

#endif // !defined(CH_EST_HPP_INCLUDED_)
