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

#include <stdio.h>
#include <sstream>
#include "hdf5hpp.hpp"
#include "cuphy_hdf5.hpp"
#include "ldpc_decode_test_vec_pusch.hpp"
#include "ldpc/ldpc_params.hpp"

#include <limits>

using namespace cuphy::ldpc;

////////////////////////////////////////////////////////////////////////
// ldpc_decode_test_vec_pusch::ldpc_decode_test_vec_pusch()
ldpc_decode_test_vec_pusch::ldpc_decode_test_vec_pusch(const test_vec_pusch_params& fparams) :
  ldpc_decode_test_vec(fparams.LLRtype),
  filename_(fparams.filename),
  TB_index_(fparams.TB_index),
  num_cw_limit_(fparams.num_cw_limit),
  nCb_(0),
  BG_(0),
  Kb_(0),
  Zc_(0)
{
    //------------------------------------------------------------------
    // Open the HDF5 file
    hdf5hpp::hdf5_file fInput = hdf5hpp::hdf5_file::open(filename_.c_str());

    //------------------------------------------------------------------
    // Read gnb_pars to verify we have TBs
    hdf5hpp::hdf5_dataset gnb_pars_ds = fInput.open_dataset("gnb_pars");
    uint32_t numTb = gnb_pars_ds[0]["numTb"].as<uint32_t>();

    if (TB_index_ >= static_cast<int>(numTb))
    {
        throw std::runtime_error("TB index out of range");
    }

    //------------------------------------------------------------------
    // Read tb_pars to get nCb, tbSize, codeRate, and derive BG/Zc
    hdf5hpp::hdf5_dataset tb_pars_ds = fInput.open_dataset("tb_pars");
    const int64_t hdf5NCb = tb_pars_ds[TB_index_]["nCb"].as<int64_t>();
    uint32_t tbSize = tb_pars_ds[TB_index_]["nTbByte"].as<uint32_t>() * 8;  // Convert bytes to bits
    double codeRate = tb_pars_ds[TB_index_]["targetCodeRate"].as<double>() / 10240.0;

    const auto ldpcParams = derive_ldpc_params(tbSize, codeRate);
    if(hdf5NCb <= 0 || hdf5NCb > std::numeric_limits<int>::max())
    {
        throw std::runtime_error("PUSCH test vector has an invalid codeblock count");
    }
    if(ldpcParams.nCb == 0 || ldpcParams.nCb > static_cast<uint32_t>(std::numeric_limits<int>::max()))
    {
        throw std::runtime_error("PUSCH test vector has an invalid derived codeblock count");
    }
    if (static_cast<uint32_t>(hdf5NCb) != ldpcParams.nCb)
    {
        char errmsg[256];
        snprintf(errmsg,
                 sizeof(errmsg),
                 "HDF5 nCb (%u) does not match derived nCb (%u) for TB index %d",
                 static_cast<uint32_t>(hdf5NCb),
                 ldpcParams.nCb,
                 TB_index_);
        throw std::runtime_error(errmsg);
    }
    nCb_ = static_cast<int>(ldpcParams.nCb);
    BG_ = ldpcParams.bg;
    Kb_ = ldpcParams.Kb;
    Zc_ = ldpcParams.Zc;

    //------------------------------------------------------------------
    // Load LLR data for this TB
    std::ostringstream llr_name;
    llr_name << "reference_rmOutLLRs" << TB_index_;

    tLLR_ = cuphy::tensor_from_dataset(fInput.open_dataset(llr_name.str().c_str()),
                                       fparams.LLRtype,
                                       cuphy::tensor_flags::align_coalesce);

    //------------------------------------------------------------------
    // Load the reference decoded bits for this TB
    std::ostringstream ref_name;
    ref_name << "reference_TbCbs_est" << TB_index_;

    tSrcData_ = cuphy::tensor_from_dataset(fInput.open_dataset(ref_name.str().c_str()),
                                          CUPHY_BIT,
                                          cuphy::tensor_flags::align_coalesce);

    //------------------------------------------------------------------
    // Verify dimensions match expectations
    if(tSrcData_.dimensions()[1] != nCb_)
    {
        char errmsg[256];
        snprintf(errmsg, sizeof(errmsg),
                 "Expected normalized PUSCH reference data dimensions [K_prime, nCb] with nCb=%d, got [%d, %d]",
                 nCb_, tSrcData_.dimensions()[0], tSrcData_.dimensions()[1]);
        throw std::runtime_error(errmsg);
    }
    set_src_bits_desc(tSrcData_.desc());

    //------------------------------------------------------------------
    // Populate LDPC "configuration" data
    populate_config();

    //------------------------------------------------------------------
    // Reshape tensors to match decoder expectations
    // LLR data is flattened [total_LLRs, 1], need to reshape to [LLRs_per_CB, nCb]
    // Reference data is normalized by tensor_from_dataset to [K_prime, nCb].

    // Reshape LLR tensor from flattened [total_LLRs, 1] to [LLRs_per_CB, nCb]
    int totalLLRs = tLLR_.dimensions()[0] * tLLR_.dimensions()[1];
    int LLRs_per_CB = totalLLRs / nCb_;

    cuphyDataType_t llr_dtype = tLLR_.desc().get_info().type();
    cuphy::tensor_desc llr_reshaped_desc(llr_dtype,
                                         LLRs_per_CB, nCb_,
                                         cuphy::tensor_flags::align_coalesce);

    //------------------------------------------------------------------
    // If we want to process only part of the input data, create a
    // tensor descriptor for that subset.
    if(num_cw_limit_ > 0)
    {
        if(num_cw_limit_ > nCb_)
        {
            throw std::runtime_error("Number of codewords to use exceeds file contents");
        }
        // Slice the reshaped descriptor
        limit_desc_ = cuphy::index_group(cuphy::dim_all(),
                                        cuphy::index_range(0, num_cw_limit_)).get_tensor_desc(llr_reshaped_desc);
        set_LLR_desc(limit_desc_);
        set_src_bits_desc(cuphy::index_group(cuphy::dim_all(),
                                             cuphy::index_range(0, num_cw_limit_)).get_tensor_desc(tSrcData_.desc()));
    }
    else
    {
        set_LLR_desc(llr_reshaped_desc);
    }
}

////////////////////////////////////////////////////////////////////////
// ldpc_decode_test_vec_pusch::desc()
const char* ldpc_decode_test_vec_pusch::desc() const
{
    return filename_.c_str();
}

////////////////////////////////////////////////////////////////////////
// ldpc_decode_test_vec_pusch::populate_config()
void ldpc_decode_test_vec_pusch::populate_config()
{
    config_.num_cw = (num_cw_limit_ > 0) ? num_cw_limit_ : nCb_;

    // Use BG, Kb, and Zc derived from tbSize and codeRate in constructor
    config_.BG = BG_;
    config_.Kb = Kb_;
    config_.Z = Zc_;

    // K = Kb * Zc for the codeblock systematic bits
    config_.K = Kb_ * Zc_;

    // Get actual info bits per CB from reference data to calculate filler bits
    // Reference data is [K_prime, nCb] where K_prime = K - F
    int K_prime;
    K_prime = tSrcData_.dimensions()[0];

    // F = K - K_prime (filler bits)
    config_.F = config_.K - K_prime;
    config_.B = K_prime;  // Actual info bits per CB (without filler)
    config_.P = 0;  // No puncturing assumed

    const int totalLLRs = tLLR_.dimensions()[0] * tLLR_.dimensions()[1];
    if (totalLLRs % nCb_ != 0)
    {
        throw std::runtime_error("PUSCH LLR tensor cannot be split into codeblocks");
    }

    const int llrsPerCodeblock = totalLLRs / nCb_;
    const int systematicNodes = (BG_ == 1) ? 22 : 10;
    if (llrsPerCodeblock % Zc_ != 0 || llrsPerCodeblock / Zc_ < systematicNodes)
    {
        throw std::runtime_error("PUSCH LLR tensor has an invalid LDPC codeblock length");
    }
    config_.mb = llrsPerCodeblock / Zc_ - systematicNodes;

    // Use the base class function to calculate the number of modulated bits
    update_config_modulated_bits();

    config_.R = static_cast<float>(config_.B) / static_cast<float>(config_.N);
    config_.QAM = "PUSCH";
    config_.punc = LLR_PUNCTURE_STATUS_UNKNOWN;
}

////////////////////////////////////////////////////////////////////////
// ldpc_decode_test_vec_pusch::generate()
void ldpc_decode_test_vec_pusch::generate()
{
    // Nothing to do for file-based test vectors
}
