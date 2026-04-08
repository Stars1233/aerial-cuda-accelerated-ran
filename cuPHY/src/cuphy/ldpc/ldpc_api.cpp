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

#include "ldpc/ldpc_api.hpp"

namespace cuphy {

LDPC_decode_config::LDPC_decode_config(cuphyDataType_t llr_type_in,
                                int16_t         num_parity_nodes_in,
                                int16_t         Z_in,
                                int16_t         max_iterations_in,
                                float           clamp_value_in,
                                int16_t         Kb_in,
                                float           norm_in,
                                uint32_t        flags_in,
                                int16_t         BG_in,
                                int16_t         algo_in,
                                void*           workspace_in) : cuphyLDPCDecodeConfigDesc_t{}
{
    // This not a member initializer list because cuphyLDPCDecodeConfigDesc_t is a C struct
    llr_type         = llr_type_in;
    num_parity_nodes = num_parity_nodes_in;
    Z                = Z_in;
    max_iterations   = max_iterations_in;
    Kb               = Kb_in;
    // Normalization union member must match the input LLR type
    if(CUPHY_R_16F == llr_type)
    {
        norm.f16x2   =  static_cast<__half2_raw>(__float2half2_rn(norm_in));
    }
    else
    {
        norm.f32     = norm_in;
    }
    flags            = flags_in;
    BG               = BG_in;
    algo             = algo_in;
    clamp_value      = clamp_value_in;
    workspace        = workspace_in;
}

void LDPC_decode_desc::add_tensor_as_tb(const tensor_desc& llrTensorDesc,
                                        void*              llrAddr,
                                        const tensor_desc& decodeTensorDesc,
                                        void*              decodeAddr)
{
    if(num_tbs >= max_tbs_per_desc)
    {
        throw std::runtime_error("Max number of TBS in LDPC descriptor exceeded");
    }
    llr_input[num_tbs].addr             = llrAddr;
    llr_input[num_tbs].stride_elements  = llrTensorDesc.get_stride(1);
    llr_input[num_tbs].num_codewords    = llrTensorDesc.get_dim(1);
    tb_output[num_tbs].addr             = static_cast<uint32_t*>(decodeAddr);
    // Convert bit stride to uint32_t word stride
    tb_output[num_tbs].stride_words     = decodeTensorDesc.get_stride(1) / 32;
    tb_output[num_tbs].num_codewords    = decodeTensorDesc.get_dim(1);
    // Soft output slots are unused
    llr_output[num_tbs].addr            = nullptr;
    llr_output[num_tbs].stride_elements = 0;
    llr_output[num_tbs].num_codewords   = 0;
    ++num_tbs;
}

void LDPC_decode_desc::add_tensor_as_tb(const tensor_desc& llrTensorDesc,
                                        void*              llrAddr,
                                        const tensor_desc& decodeTensorDesc,
                                        void*              decodeAddr,
                                        const tensor_desc& softOutputsTensorDesc,
                                        void*              softOutputsAddr)
{
    if(num_tbs >= max_tbs_per_desc)
    {
        throw std::runtime_error("Max number of TBS in LDPC descriptor exceeded");
    }
    llr_input[num_tbs].addr             = llrAddr;
    llr_input[num_tbs].stride_elements  = llrTensorDesc.get_stride(1);
    llr_input[num_tbs].num_codewords    = llrTensorDesc.get_dim(1);
    tb_output[num_tbs].addr             = static_cast<uint32_t*>(decodeAddr);
    // Convert bit stride to uint32_t word stride
    tb_output[num_tbs].stride_words     = decodeTensorDesc.get_stride(1) / 32;
    tb_output[num_tbs].num_codewords    = decodeTensorDesc.get_dim(1);
    llr_output[num_tbs].addr            = softOutputsAddr;
    llr_output[num_tbs].stride_elements = softOutputsTensorDesc.get_stride(1);
    llr_output[num_tbs].num_codewords   = softOutputsTensorDesc.get_dim(1);
    // Set the flag in the decoder descriptor to indicate that soft outputs
    // are desired.
    config.flags |= CUPHY_LDPC_DECODE_WRITE_SOFT_OUTPUTS;
    ++num_tbs;
}

LDPC_decode_desc& LDPC_decode_desc_set::find(int16_t BG, int Z, int num_parity, cuphyLdpcMaxItrAlgoType_t ldpcMaxNumItrAlgo, uint8_t ldpcMaxNumItrPerUe)
{
    for(unsigned int i = 0; i < count_; ++i)
    {
        if(descs_[i].has_config(BG, Z, num_parity, ldpcMaxNumItrAlgo, ldpcMaxNumItrPerUe) && !descs_[i].is_full())
        {
            return descs_[i];
        }
    }
    if((count_ + 1) < max_count_)
    {
        LDPC_decode_desc& d = descs_[count_++];
        d.config.BG               = BG;
        d.config.Z                = Z;
        d.config.num_parity_nodes = num_parity;
        if(ldpcMaxNumItrAlgo == LDPC_MAX_NUM_ITR_ALGO_TYPE_PER_UE)
        {
            d.config.max_iterations   = ldpcMaxNumItrPerUe;
        }
        d.num_tbs                 = 0;
        return d;
    }
    throw std::runtime_error("LDPC_decode_desc_set size exceeded");
}

LDPC_decoder::LDPC_decoder(context& ctx, unsigned int flags)
{
    cuphyLDPCDecoder_t dec = nullptr;
    cuphyStatus_t      s   = cuphyCreateLDPCDecoder(ctx.handle(),
                                                    &dec,
                                                    flags);
    if(CUPHY_STATUS_SUCCESS != s)
    {
        throw cuphy::cuphy_fn_exception(s, "cuphyCreateLDPCDecoder()");
    }
    dec_.reset(dec);
}

size_t LDPC_decoder::get_workspace_size(const cuphyLDPCDecodeConfigDesc_t& cfg,
                                        int                                numCodeWords) const
{
    size_t        szBuf = 0;
    cuphyStatus_t s     = cuphyErrorCorrectionLDPCDecodeGetWorkspaceSize(handle(),
                                                                         &cfg,
                                                                         numCodeWords, // numCodeblocks
                                                                         &szBuf);      // output size
    if(CUPHY_STATUS_SUCCESS != s)
    {
        throw cuphy_fn_exception(s, "cuphyErrorCorrectionLDPCDecodeGetWorkspaceSize()");
    }
    return szBuf;
}

void LDPC_decoder::decode(const LDPC_decode_tensor_params& params,
                          cudaStream_t                     strm) const
{
    cuphyStatus_t s = cuphyErrorCorrectionLDPCDecode(handle(),
                                                     params.dst_desc,
                                                     params.dst_addr,
                                                     params.LLR_desc,
                                                     params.LLR_addr,
                                                     params.softOutputs_desc,
                                                     params.softOutputs_addr,
                                                     &params.config,
                                                     strm);
    if(CUPHY_STATUS_SUCCESS != s)
    {
        throw cuphy_fn_exception(s, "cuphyErrorCorrectionLDPCDecode()");
    }
}

void LDPC_decoder::decode(const cuphyLDPCDecodeDesc_t& desc,
                          cudaStream_t                 strm) const
{

    cuphyStatus_t s = cuphyErrorCorrectionLDPCTransportBlockDecode(handle(),
                                                                   &desc,
                                                                   strm);
    if(CUPHY_STATUS_SUCCESS != s)
    {
        throw cuphy_fn_exception(s, "cuphyErrorCorrectionLDPCTransportBlockDecode()");
    }
}

void LDPC_decoder::set_normalization(cuphyLDPCDecodeConfigDesc_t& config) const
{
    cuphyStatus_t s = cuphyErrorCorrectionLDPCDecodeSetNormalization(dec_.get(),
                                                                     &config);
    if(CUPHY_STATUS_SUCCESS != s)
    {
        throw cuphy_fn_exception(s, "cuphyErrorCorrectionLDPCDecodeSetNormalization()");
    }
}

void LDPC_decoder::get_launch_config(cuphyLDPCDecodeLaunchConfig_t& cfg) const
{
    cuphyStatus_t s = cuphyErrorCorrectionLDPCDecodeGetLaunchDescriptor(dec_.get(),
                                                                        &cfg);
    if(CUPHY_STATUS_SUCCESS != s)
    {
        throw cuphy_fn_exception(s, "cuphyErrorCorrectionLDPCDecodeGetLaunchDescriptor()");
    }
}

} // namespace cuphy
