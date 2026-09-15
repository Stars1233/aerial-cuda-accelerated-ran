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

#if !defined(LDPC_DECODE_TEST_VEC_GEN_HPP_INCLUDED_)
#define LDPC_DECODE_TEST_VEC_GEN_HPP_INCLUDED_

#include <stdexcept>
#include <string>
#include "ldpc_decode_test_vec.hpp"

class ldpc_invalid_crc_config : public std::runtime_error
{
public:
    explicit ldpc_invalid_crc_config(const std::string& msg) :
      std::runtime_error(msg)
    {
    }
};

struct test_vec_gen_params
{
    test_vec_gen_params(cuphyDataType_t llr_type,
                        int             bg,
                        int             Z,
                        int             nparity,
                        int             num_cw_in,
                        int             blockSize,
                        float           codeRate,
                        int             modBits,
                        int             log2QAM,
                        float           SNR_in,
                        bool            puncture_in,
                        uint32_t        crc_type_in) :
      LLRtype(llr_type),
      BG(bg),
      num_parity(nparity),
      lifting_size(Z),
      num_cw(num_cw_in),
      block_size(blockSize),
      code_rate(codeRate),
      num_modulated_bits(modBits),
      log2_QAM(log2QAM),
      SNR(SNR_in),
      puncture(puncture_in),
      crc_type(crc_type_in)
    {
    }
    cuphyDataType_t LLRtype;
    int             BG;
    int             num_parity;
    int             lifting_size;
    int             num_cw;
    int             block_size;
    float           code_rate;
    int             num_modulated_bits;
    int             log2_QAM;
    float           SNR;
    bool            puncture;
    uint32_t        crc_type;
};

////////////////////////////////////////////////////////////////////////
// ldpc_decode_test_vec_gen
class ldpc_decode_test_vec_gen : public ldpc_decode_test_vec
{
public:
    //------------------------------------------------------------------
    // ldpc_decode_test_vec_gen()
    ldpc_decode_test_vec_gen(cuphy::context&            ctx,
                             cuphy::rng&                rand_gen,
                             const test_vec_gen_params& fparams);
    //------------------------------------------------------------------
    // ~ldpc_decode_test_vec_gen()
    virtual ~ldpc_decode_test_vec_gen() {}
    //------------------------------------------------------------------
    // desc()
    // Descriptive string
    virtual const char* desc() const override;
    //------------------------------------------------------------------
    // Prepare a test vector
    virtual void generate() override;
private:
    //------------------------------------------------------------------
    // populate_config()
    void populate_config(const test_vec_gen_params& fparams);
    //------------------------------------------------------------------
    // Data
    cuphy::context&      ctx_;
    cuphy::rng&          rng_gen_;
    int                  num_cw_;
    int                  log2_QAM_;
    cuphy::tensor_device tSymbols_;
    float                SNR_;
};

#endif // !defined(LDPC_DECODE_TEST_VEC_GEN_HPP_INCLUDED_)
