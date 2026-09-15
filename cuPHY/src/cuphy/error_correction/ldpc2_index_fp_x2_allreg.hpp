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

#if !defined(LDPC2_INDEX_FP_X2_ALLREG_HPP_INCLUDED_)
#define LDPC2_INDEX_FP_X2_ALLREG_HPP_INCLUDED_

#include "ldpc.hpp"

namespace ldpc2
{

////////////////////////////////////////////////////////////////////////
// index_fp_x2_allreg
// All-register x2 variant (ALGO 51): all C2V in registers, shmem = APP
// only. Enables 2 CW/CTA decoding at high parity node counts where
// normal C2V shmem overflow would exceed device limits.
class index_fp_x2_allreg : public ldpc::decode_algo
{
public:
    //------------------------------------------------------------------
    // Constructor
    index_fp_x2_allreg(ldpc::decoder& desc);
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

#endif // !defined(LDPC2_INDEX_FP_X2_ALLREG_HPP_INCLUDED_)
