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

#include <stdio.h>
#include "hdf5hpp.hpp"
#include "cuphy_hdf5.hpp"
#include "ldpc_decode_test_vec_file.hpp"

////////////////////////////////////////////////////////////////////////
// ldpc_decode_test_vec_file::ldpc_decode_test_vec_file()
ldpc_decode_test_vec_file::ldpc_decode_test_vec_file(const test_vec_file_params& fparams) :
  ldpc_decode_test_vec(fparams.LLRtype),
  filename_(fparams.filename),
  num_cw_limit_(fparams.num_cw_limit)
{
    //------------------------------------------------------------------
    // Open the HDF5 file
    hdf5hpp::hdf5_file fInput = hdf5hpp::hdf5_file::open(filename_.c_str());
    //------------------------------------------------------------------
    // Load LLR data, converting to the desired LLR type if necessary
    tLLR_                     = cuphy::tensor_from_dataset(fInput.open_dataset("inputLLR"),
                                                           fparams.LLRtype,
                                                           cuphy::tensor_flags::align_coalesce);
    //------------------------------------------------------------------
    // Load the source (truth) data, converting to a CUPHY_BIT tensor
    tSrcData_                 = cuphy::tensor_from_dataset(fInput.open_dataset("sourceData"),
                                                           CUPHY_BIT,
                                                           cuphy::tensor_flags::align_coalesce);
    set_src_bits_desc(tSrcData_.desc());
    //------------------------------------------------------------------
    // Populate LDPC "configuration" data
    populate_config(fparams);
    //------------------------------------------------------------------
    // If we want to process only part of the input data, create a
    // tensor descriptor for that subset.
    if(num_cw_limit_ > 0)
    {
        if(num_cw_limit_ > tLLR_.dimensions()[1])
        {
            throw std::runtime_error("Number of codewords to use exceeds file contents");
        }
        // We are using the start of the tensor and slicing off the "end",
        // so we don't need to adjust the tensor address. We just create
        // a modified tensor descriptor.
        limit_desc_ = cuphy::index_group(cuphy::dim_all(),
                                         cuphy::index_range(0, num_cw_limit_)).get_tensor_desc(tLLR_.desc());
        set_LLR_desc(limit_desc_);
        set_src_bits_desc(cuphy::index_group(cuphy::dim_all(),
                                             cuphy::index_range(0, num_cw_limit_)).get_tensor_desc(tSrcData_.desc()));
    }
    else
    {
        set_LLR_desc(tLLR_.desc());
    }
    //printf("tLLR_:      %s\n", tLLR_.desc().get_info().to_string().c_str());
    //printf("tSrcData_:  %s\n", tSrcData_.desc().get_info().to_string().c_str());
}

////////////////////////////////////////////////////////////////////////
// ldpc_decode_test_vec_file::desc()
const char* ldpc_decode_test_vec_file::desc() const
{
    return filename_.c_str();
}

////////////////////////////////////////////////////////////////////////
// ldpc_decode_test_vec_file::populate_config()
void ldpc_decode_test_vec_file::populate_config(const test_vec_file_params& fparams)
{
    config_.BG       = fparams.BG;
    // For file input, we are assuming that the input contains no filler
    // bits, and that no parity bits are punctured.
    config_.K        = tSrcData_.dimensions()[0];
    config_.F        = 0;
    config_.P        = 0;
    config_.B        = config_.K;
    config_.num_cw   = (fparams.num_cw_limit > 0) ?
                       fparams.num_cw_limit       :
                       tLLR_.dimensions()[1];
    config_.Kb       = get_num_info_nodes(config_.BG, config_.B);
    config_.Z        = find_lifting_size(config_.BG, config_.B);
    config_.mb       = fparams.num_parity;

    // Use the base class function to calculate the number of modulated
    // bits
    update_config_modulated_bits();

    config_.R        = static_cast<float>(config_.B) / static_cast<float>(config_.N);

    config_.QAM      = "Unknown";
    config_.punc     = LLR_PUNCTURE_STATUS_UNKNOWN;
}

////////////////////////////////////////////////////////////////////////
// ldpc_decode_test_vec_file::generate()
void ldpc_decode_test_vec_file::generate()
{
    // Nothing to do for file-based test vectors. We might want to
    // allow for advancing to a new subset of the test data in the
    // future?
}
