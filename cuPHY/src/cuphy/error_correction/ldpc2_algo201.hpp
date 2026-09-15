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

#if !defined(LDPC2_ALGO201_HPP_INCLUDED_)
#define LDPC2_ALGO201_HPP_INCLUDED_

#include "ldpc.hpp"

namespace ldpc2
{

////////////////////////////////////////////////////////////////////////
// bg1_z256up_bp_allreg_x2
// BG1 / Z=256..384 (runtime; the five 38.212 liftings 256/288/320/352/384),
// p=4..11 (runtime), 2 codewords per CTA exact box-plus layered decoder. ONE
// kernel: no per-Z or per-p specialization. Body is the evolved all-register
// stack -- deferred normalization, register-resident APP columns,
// iteration-0 puncture collapse, final-pass dead-store elision, V24
// forwarding -- with the compile-time Z-immediate addressing replaced by the
// runtime bg_desc descriptor path (dp_desc form, base folded), and the row
// band extended to row 10 (p<=11; hard limit -- see the column-ownership
// expiry note in the .cu). Transport-block interface, hard-output only.
// Targets GB203.
// __launch_bounds__(A201_Z_MAX, 1), so 384 threads by default. Compiles to
// REG:168 / STACK:0. Do NOT restate that as a per-Z register budget: the .cu
// keeps the arithmetic that made the zone stop at 352 precisely because it was
// wrong -- registers are allocated per warp within a subpartition, so Z=288..384
// all land on the same 168 and extending to 384 cost nothing.
class bg1_z256up_bp_allreg_x2 : public ldpc::decode_algo
{
public:
    explicit bg1_z256up_bp_allreg_x2(ldpc::decoder& desc);

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
        return "the transport-block interface, hard decisions only (no soft "
                   "outputs), and punctured input (it never loads the punctured "
                   "columns V0/V1)";
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

#endif // !defined(LDPC2_ALGO201_HPP_INCLUDED_)
