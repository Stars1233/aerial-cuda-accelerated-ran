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

#if !defined(LDPC2_INDEX_BP_X2_LOWP_HPP_INCLUDED_)
#define LDPC2_INDEX_BP_X2_LOWP_HPP_INCLUDED_

#include "ldpc.hpp"

namespace ldpc2
{

////////////////////////////////////////////////////////////////////////
// index_bp_x2_lowp
// Low-p x2 variant (ALGO 56): BG1 only, p<=5, min-sum core,
// NUM_REG_PARITY_ROWS=5 (all rows in registers), targeting 2 CTAs/SM
// for better occupancy at small parity node counts.
class index_bp_x2_lowp : public ldpc::decode_algo
{
public:
    //------------------------------------------------------------------
    // Constructor
    index_bp_x2_lowp(ldpc::decoder& desc);
    //------------------------------------------------------------------
    // decode()
    virtual cuphyStatus_t decode(ldpc::decoder&                     dec,
                                 LDPC_output_t&                     tDst,
                                 const_tensor_pair&                 tLLR,
                                 const cuphy_optional<tensor_pair>& optSoftOutputs,
                                 const cuphyLDPCDecodeConfigDesc_t& config,
                                 cudaStream_t                       strm) override;
    //------------------------------------------------------------------
    // decode_tb()
    virtual cuphyStatus_t decode_tb(ldpc::decoder&               dec,
                                    const cuphyLDPCDecodeDesc_t& decodeDesc,
                                    cudaStream_t                 strm) override;
    //------------------------------------------------------------------
    // get_workspace_size()
    virtual std::pair<bool, size_t> get_workspace_size(const ldpc::decoder&               dec,
                                                       const cuphyLDPCDecodeConfigDesc_t& config,
                                                       int                                num_cw) override;
    //------------------------------------------------------------------
    // can_decode_config()
    virtual bool can_decode_config(const ldpc::decoder&               dec,
                                   const cuphyLDPCDecodeConfigDesc_t& config) override;
    //------------------------------------------------------------------
    // get_launch_config()
    virtual cuphyStatus_t get_launch_config(const ldpc::decoder&           dec,
                                            cuphyLDPCDecodeLaunchConfig_t& launchConfig) override;
};

} // namespace ldpc2

#endif // !defined(LDPC2_INDEX_BP_X2_LOWP_HPP_INCLUDED_)
