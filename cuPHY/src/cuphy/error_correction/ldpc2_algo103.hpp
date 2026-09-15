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

#if !defined(LDPC2_ALGO103_HPP_INCLUDED_)
#define LDPC2_ALGO103_HPP_INCLUDED_

#include "ldpc.hpp"

namespace ldpc2
{

////////////////////////////////////////////////////////////////////////
// bg1_z384_p4_x2
// BG1, p=4, Z=384 half2 row-layered min-sum experiment.
// Fixed to uniform BP iterations:
// each iteration visits rows 0,1,2,3 exactly once.
class bg1_z384_p4_x2 : public ldpc::decode_algo
{
public:
    explicit bg1_z384_p4_x2(ldpc::decoder& desc);

    virtual cuphyStatus_t decode(ldpc::decoder&                     dec,
                                 LDPC_output_t&                     tDst,
                                 const_tensor_pair&                 tLLR,
                                 const cuphy_optional<tensor_pair>& optSoftOutputs,
                                 const cuphyLDPCDecodeConfigDesc_t& config,
                                 cudaStream_t                       strm) override;

    virtual cuphyStatus_t decode_tb(ldpc::decoder&               dec,
                                    const cuphyLDPCDecodeDesc_t& decodeDesc,
                                    cudaStream_t                 strm) override;

    virtual std::pair<bool, size_t> get_workspace_size(const ldpc::decoder&               dec,
                                                       const cuphyLDPCDecodeConfigDesc_t& config,
                                                       int                                num_cw) override;

    virtual bool can_decode_config(const ldpc::decoder&               dec,
                                   const cuphyLDPCDecodeConfigDesc_t& config) override;

    virtual cuphyStatus_t get_launch_config(const ldpc::decoder&           dec,
                                            cuphyLDPCDecodeLaunchConfig_t& launchConfig) override;

    // Stated once here so a refusal can explain itself; see
    // decode_algo::supported_config_desc().
    const char* supported_config_desc() const override
    {
        return "the transport-block interface, and punctured input (it never "
                   "loads the punctured columns V0/V1)";
    }
    //------------------------------------------------------------------
    // decode_algo::requirements(). The machine-readable form of the string
    // above -- keep the two in step.
    [[nodiscard]]
    uint32_t requirements() const override
    {
        return CUPHY_LDPC_ALGO_REQUIRES_PUNCTURED_INPUT |
               CUPHY_LDPC_ALGO_REQUIRES_TB_INTERFACE;
    }
};

} // namespace ldpc2

#endif // !defined(LDPC2_ALGO103_HPP_INCLUDED_)
