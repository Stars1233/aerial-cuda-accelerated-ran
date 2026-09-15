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

#ifndef CUPHY_UTILS_HPP
#define CUPHY_UTILS_HPP

#include "cuphy.hpp"

namespace cuphy_utils {
    // template function to enable/disable graph nodes for het cfgs

    /**
     * Enable or Disable Graph Nodes according to provided params.
     * [0..NumActiveNodes) will be enabled if the node is disabled.
     * [numActiveNodes..total) will be disabled if enabled.
     * @pre  numActiveNodes <= numTotalNodes
     * @tparam Tcfg Launch Kernel Config type, containing the Kernel node params driver.
     * @param numActiveNodes The number of active nodes to iterate over [0..NumActiveNodes)
     * @param numTotalNodes The total number of notes
     * @param cfgs Launch Kernel Config instance, containing the Kernel node params driver.
     * @param enableStats container that has info on whether a current node is enabled or disabled
     * @param nodes Nodes array to use for enabling/disabling
     * @param graphExec GrapgExec to use when calling cuGraph* functions
     */
    template<typename Tcfg, std::size_t N>
    void setHetCfgNodeStatus(const int numActiveNodes, const int numTotalNodes, const Tcfg (&cfgs)[N],
                             std::vector<uint8_t>& enableStats, CUgraphNode nodes[], CUgraphExec graphExec,
                             CUDA_KERNEL_NODE_PARAMS Tcfg::*kernelParams = &Tcfg::kernelNodeParamsDriver)
    {
        // enable and update active CUDA graph nodes
        for(int hetCfgIdx = 0; hetCfgIdx < numActiveNodes; ++hetCfgIdx)
        {
            if(enableStats[hetCfgIdx] != 1)
            {
                enableStats[hetCfgIdx] = 1;
                CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(graphExec, nodes[hetCfgIdx], 1));
            }
            CU_CHECK_EXCEPTION(cuGraphExecKernelNodeSetParams(graphExec, nodes[hetCfgIdx], &(cfgs[hetCfgIdx].*kernelParams)));
        }
        // disable the remaining CUDA graph nodes
        for(int hetCfgIdx = numActiveNodes; hetCfgIdx < numTotalNodes; ++hetCfgIdx)
        {
            if(enableStats[hetCfgIdx] != 0)
            {
                enableStats[hetCfgIdx] = 0;
                CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(graphExec, nodes[hetCfgIdx], 0));
            }
        }
    }

    /**
     * @see @fn setHetCfgNodeStatus. It is the same except kernelNodeParamsDriverSecond
     */
    template<typename Tcfg, std::size_t N>
    void setHetCfgNodeStatusSecond(const int numActiveNodes, const int numTotalNodes, const Tcfg (&cfgs)[N],
                                   std::vector<uint8_t> &enableStats, CUgraphNode nodes[], CUgraphExec graphExec)
    {
        setHetCfgNodeStatus(numActiveNodes, numTotalNodes, cfgs, enableStats, nodes, graphExec,
                            &Tcfg::kernelNodeParamsDriverSecond);
    }
}

#endif //CUPHY_UTILS_HPP
