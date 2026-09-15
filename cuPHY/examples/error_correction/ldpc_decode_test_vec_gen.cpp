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
#include <limits>
#include "hdf5hpp.hpp"
#include "cuphy_hdf5.hpp"
#include "ldpc_decode_test_vec_gen.hpp"
#include "ldpc_decode_test_vec_gen_kernels.hpp"
#include "ldpc/ldpc_api.hpp"

static const char* generated_crc_type_name(uint32_t crc_type)
{
    switch(crc_type)
    {
        case CUPHY_LDPC_CRC_16:  return "CRC-16";
        case CUPHY_LDPC_CRC_24A: return "CRC-24A";
        case CUPHY_LDPC_CRC_24B: return "CRC-24B";
        default:                 return "CRC-NONE";
    }
}

////////////////////////////////////////////////////////////////////////
// ldpc_decode_test_vec_gen::ldpc_decode_test_vec_gen()
ldpc_decode_test_vec_gen::ldpc_decode_test_vec_gen(cuphy::context&            ctx,
                                                   cuphy::rng&                rng_gen,
                                                   const test_vec_gen_params& gparams) :
    ldpc_decode_test_vec(gparams.LLRtype),
    ctx_(ctx),
    rng_gen_(rng_gen),
    num_cw_(std::max(gparams.num_cw, 1)),
    log2_QAM_(gparams.log2_QAM),
    SNR_(gparams.SNR)
{
    //------------------------------------------------------------------
    // Populate LDPC "configuration" data using input parameters
    populate_config(gparams);
    //------------------------------------------------------------------
    // Generate data for input to the encoder with CRC-24B appended.
    // We work with scalar values (1 bit per byte) so we can leverage
    // the cuPHY fill function for filler bits.
    cuphy::tensor_device tSrcWithFiller(CUPHY_R_8U, config_.K);
    rng_gen.uniform(tSrcWithFiller, 0, 1, 0);

    // Apply CRC to the codeword and set filler bits to zero
    {
        cuphyTensorDescriptor_t tDescriptor = tSrcWithFiller.desc().handle();
        tensor_desc& tDesc = static_cast<tensor_desc&>(*tDescriptor);
        create_codeword_with_crc(tDesc.layout(),
                                 static_cast<uint8_t*>(tSrcWithFiller.addr()),
                                 config_.K,
                                 config_.F,
                                 config_.crc_type,
                                 true, // Use existing info bits not including CRC
                                 true);  // MSB first
    }

    //------------------------------------------------------------------
    // Populate the source data tensor with only the source bits
    tSrcData_ = cuphy::tensor_device(CUPHY_BIT, config_.B);
    set_src_bits_desc(tSrcData_.desc());
    {
        cuphy::index_group grp(cuphy::index_range(0, config_.B));
        cuphy::tensor_ref  tSrcB = tSrcWithFiller.subset(grp);
        cuphy::tensor_convert(tSrcData_, tSrcB);
    }
    //------------------------------------------------------------------
    // Convert source data to type CUPHY_BIT, as required by the encoder
    cuphy::tensor_device tSrcWithFillerBits(CUPHY_BIT, config_.K);
    cuphy::tensor_convert(tSrcWithFillerBits, tSrcWithFiller);
    //------------------------------------------------------------------
    // Encode
    const int            MAXV = (1 == config_.BG)            ?
                                CUPHY_LDPC_MAX_BG1_VAR_NODES :
                                CUPHY_LDPC_MAX_BG2_VAR_NODES;
    cuphy::tensor_device tEncode(CUPHY_BIT, config_.Z * MAXV);
    cuphy::ldpc_encode(tEncode,
                       tSrcWithFillerBits,
                       config_.BG,
                       config_.Z);

    if (0) {
        cuphyTensorDescriptor_t tDescriptor = tEncode.desc().handle();
        tensor_desc&  tDesc = static_cast<tensor_desc&>(*tDescriptor);
        const tensor_layout_any& tLayout = tDesc.layout();
        print_codeword(tDesc.layout(), static_cast<uint8_t*>(tEncode.addr()), config_.K, config_.N, true);
    }

    //------------------------------------------------------------------
    // Modulate (bits to complex values)
    const int            V               = config_.mb + ((1 == config_.BG) ? 22 : 10);
    const int            NUM_SYMBOLS     = (config_.Z * V + gparams.log2_QAM - 1) / gparams.log2_QAM;
    cuphy::index_group   grp(cuphy::index_range(0, config_.Z * V));
    cuphy::tensor_ref    tEncodedPartial = tEncode.subset(grp);
    tSymbols_ = cuphy::tensor_device(CUPHY_C_16F, NUM_SYMBOLS);
    cuphy::modulate_symbol(tSymbols_, tEncodedPartial, gparams.log2_QAM);

    //------------------------------------------------------------------
    // Allocate a tensor to hold LLR data
    const int            NUM_LLR = NUM_SYMBOLS * log2_QAM_;
    tLLR_ = cuphy::tensor_device(gparams.LLRtype, NUM_LLR, num_cw_, cuphy::tensor_flags::align_coalesce);
    //------------------------------------------------------------------
    // Set the base class LLR descriptor
    set_LLR_desc(tLLR_.desc());
    //------------------------------------------------------------------
    cudaStreamSynchronize(0);
}

////////////////////////////////////////////////////////////////////////
// ldpc_decode_test_vec_gen::desc()
const char* ldpc_decode_test_vec_gen::desc() const
{
    return "(generated at runtime)";
}

////////////////////////////////////////////////////////////////////////
// ldpc_decode_test_vec_gen::populate_config()
void ldpc_decode_test_vec_gen::populate_config(const test_vec_gen_params& gparams)
{

    config_.BG       = gparams.BG;
    config_.num_cw   = num_cw_;
    if(gparams.block_size > 0)
    {
        config_.B  = gparams.block_size;
        config_.Z  = find_lifting_size(config_.BG, config_.B);
        config_.Kb = get_num_info_nodes(config_.BG, config_.B);
        config_.F  = (config_.Z * ((1 == config_.BG) ? 22 : 10)) - config_.B;
        if(gparams.num_modulated_bits > 0)
        {
            config_.N = gparams.num_modulated_bits;
            config_.R = static_cast<float>(config_.B) / static_cast<float>(config_.N);
        }
        else if(gparams.code_rate > 0.0f)
        {
            config_.R = gparams.code_rate;
            config_.N = std::lroundf(config_.B / gparams.code_rate);
        }
        else
        {
            throw std::runtime_error(std::string("Code rate or modulated bits must be "
                                                 "provided with input block size"));
        }
        // N: Number of modulated bits
        // B: Input block size
        // mb: number of parity nodes
        // P: punctured parity bits (0 <= P < Z)
        //
        // N = B + Z(mb - 2) - P
        // N - B             P
        // ----- + 2 = mb - ---
        //   Z               Z
        //
        //
        // 0 <= P/Z < 1
        //
        // mb = ceil(2 + (N - B)/Z) = ceil((2Z + N - B) / Z)
        config_.mb = static_cast<int>(std::ceil((config_.N - config_.B + (2 * config_.Z)) / static_cast<float>(config_.Z)));
        config_.P  = config_.B + (config_.Z * (config_.mb - 2)) - config_.N;
    }
    else
    {
        // Using values for Z, mb. Assuming no filler bits (except
        // for BG2 cases for Kb < 10), no punctured parity bits.
        config_.Z  = gparams.lifting_size;
        config_.mb = gparams.num_parity;
        config_.Kb = get_num_info_nodes_from_Z(config_.BG, config_.Z);
        config_.B  = config_.Z * config_.Kb;
        if(1 == config_.BG)
        {
            config_.F  = 0;
        }
        else
        {
            config_.F = (10 - config_.Kb) * config_.Z;
        }
        config_.P = 0;
        // Use the base class function to calculate the number of modulated
        // bits
        update_config_modulated_bits();
        config_.R = static_cast<float>(config_.B) / static_cast<float>(config_.N);
    }
    // mb must fit the base graph. generate() allocates the encode output as
    // Z*MAXV bits but the modulation source spans Z*(mb + 22|10); an mb past the
    // base-graph maximum therefore reads off the end of that allocation inside
    // the modulation mapper (compute-sanitizer memcheck: invalid __global__ read
    // in sym_mod_util). Reject it here instead of letting it reach the GPU.
    {
        const int maxVar    = (1 == config_.BG) ? CUPHY_LDPC_MAX_BG1_VAR_NODES
                                                 : CUPHY_LDPC_MAX_BG2_VAR_NODES;
        const int kbNominal = (1 == config_.BG) ? 22 : 10;  // matches V in generate()
        const int maxMb     = maxVar - kbNominal;            // BG1 46, BG2 42
        if(config_.mb > maxMb)
        {
            throw std::runtime_error(
                std::string("Invalid parity node count for BG") + std::to_string(config_.BG) +
                ": mb=" + std::to_string(config_.mb) + " exceeds the base graph maximum " +
                std::to_string(maxMb));
        }
    }

    config_.K    = config_.B + config_.F;
    config_.QAM  = get_QAM_desc(gparams.log2_QAM);
    config_.punc = gparams.puncture ? LLR_PUNCTURE_STATUS_ENABLED :  LLR_PUNCTURE_STATUS_DISABLED;
    config_.crc_type = gparams.crc_type;

    const int crc_len = (config_.crc_type == CUPHY_LDPC_CRC_16)  ? 16 :
                        (config_.crc_type == CUPHY_LDPC_CRC_24A) ? 24 :
                        (config_.crc_type == CUPHY_LDPC_CRC_24B) ? 24 : 0;
    if((crc_len > 0) && (config_.B <= crc_len))
    {
        throw ldpc_invalid_crc_config(
            std::string("Requested CRC type ") + generated_crc_type_name(config_.crc_type) +
            " requires more than " + std::to_string(crc_len) +
            " information bits, but B=" + std::to_string(config_.B) +
            ". This generated LDPC test-vector configuration is invalid; skipping run.");
    }
}

////////////////////////////////////////////////////////////////////////
// ldpc_decode_test_vec_gen::generate()
void ldpc_decode_test_vec_gen::generate()
{
    //------------------------------------------------------------------
    // Generate random noise. Assuming half of noise power is real and
    // half is complex, and that the input SNR is the total (real +
    // complex).
    const float NOISE_VAR              = std::pow(10.0f, SNR_ / (-10.0f));
    const float NOISE_VAR_HALF         = NOISE_VAR / 2.0f;
    const float NOISE_COMPONENT_STDDEV = std::sqrt(NOISE_VAR_HALF);
    const cuComplex mean               = make_cuFloatComplex(0.0f, 0.0f);
    const cuComplex stddev             = make_cuFloatComplex(NOISE_COMPONENT_STDDEV,
                                                             NOISE_COMPONENT_STDDEV);
    //printf("NOISE_STDDEV = %f\n", NOISE_STDDEV);
    const int NUM_SYMBOLS    = tSymbols_.dimensions()[0];
    cuphy::tensor_device tNoise(CUPHY_C_16F, NUM_SYMBOLS, num_cw_);
    rng_gen_.normal(tNoise, mean, stddev);
    //------------------------------------------------------------------
    // Add noise to modulated symbols
    cuphy::tensor_device tSymbolsPlusNoise(CUPHY_C_16F, NUM_SYMBOLS, num_cw_);
    // Using broadcast semantics here - the symbols "column" vector is
    // automatically replicated for each column of the noise.
    cuphy::tensor_sum(tSymbolsPlusNoise,
                      tSymbols_,
                      tNoise);
    //------------------------------------------------------------------
    // Demodulate symbols (complex values to LLRs)
    ctx_.demodulate_symbol(tLLR_,
                           tSymbolsPlusNoise,
                           log2_QAM_,
                           NOISE_VAR);
    //------------------------------------------------------------------
    // Set LLRs for punctured bits to 0
    // tLLR(0:2Z,:) = 0
    if(LLR_PUNCTURE_STATUS_ENABLED == config_.punc)
    {
        cuphy::index_group p_grp(cuphy::index_range(0, 2 * config_.Z),
                                 cuphy::dim_all());
        cuphy::tensor_ref tLLR_p = tLLR_.subset(p_grp);
        cuphy::tensor_fill(tLLR_p, 0);
    }
    //------------------------------------------------------------------
    // Set LLRs for filler bits to Inf
    // tLLR(K-F:K,:) = Inf
    if(config_.F > 0)
    {
        cuphy::index_group f_grp(cuphy::index_range(config_.K - config_.F,
                                                    config_.K),
                                 cuphy::dim_all());
        cuphy::tensor_ref  tLLR_f = tLLR_.subset(f_grp);
        cuphy::tensor_fill(tLLR_f, std::numeric_limits<float>::infinity());
    }
#if 0
    // For debugging, convert to FP8 and back
    if(CUPHY_R_16F == tLLR_.type())
    {
        const int NUM_LLR = tLLR_.dimensions()[0];
        cuphy::tensor_device tQuant(CUPHY_R_8F_E4M3, NUM_LLR, config_.num_cw);
        cuphy::tensor_convert(tQuant, tLLR_);
        cuphy::tensor_convert(tLLR_, tQuant);
    }
#endif
#if 0
    //------------------------------------------------------------------
    // Enable this code for debugging purposes, to write an HDF5 file
    // with the generated data.
    {
        const int            MAXV = (1 == config_.BG)            ?
                                    CUPHY_LDPC_MAX_BG1_VAR_NODES :
                                    CUPHY_LDPC_MAX_BG2_VAR_NODES;

        // TODO: make tEncode a member instead of local scope in the constructor
        //cuphy::tensor_device tEncodeDebug(CUPHY_R_8U, config_.Z * MAXV);
        //cuphy::tensor_convert(tEncodeDebug, tEncode);

        cuphy::tensor_device tSymbolsDebug(CUPHY_C_32F, NUM_SYMBOLS);
        cuphy::tensor_convert(tSymbolsDebug, tSymbols_);

        cuphy::tensor_device tNoiseDebug(CUPHY_C_32F, NUM_SYMBOLS, config_.num_cw);
        cuphy::tensor_convert(tNoiseDebug, tNoise);

        cuphy::tensor_device tSymbolsPlusNoiseDebug(CUPHY_C_32F, NUM_SYMBOLS, config_.num_cw);
        cuphy::tensor_convert(tSymbolsPlusNoiseDebug, tSymbolsPlusNoise);

        const int NUM_LLR = tLLR_.dimensions()[0];
        cuphy::tensor_device tLLRDebug(CUPHY_R_32F, NUM_LLR, config_.num_cw);
        cuphy::tensor_convert(tLLRDebug, tLLR_);

        cudaStreamSynchronize(0);
        hdf5hpp::hdf5_file f = hdf5hpp::hdf5_file::create("debug_gen.h5");
        //cuphy::write_HDF5_dataset(f, tSrcWithFiller,         "srcWithFiller",     0);
        //cuphy::write_HDF5_dataset(f, tEncodeDebug,           "tEncode",           0);
        cuphy::write_HDF5_dataset(f, tSymbolsDebug,          "tSymbols",          0);
        cuphy::write_HDF5_dataset(f, tNoiseDebug,            "tNoise",            0);
        cuphy::write_HDF5_dataset(f, tSymbolsPlusNoiseDebug, "tSymbolsPlusNoise", 0);
        cuphy::write_HDF5_dataset(f, tLLRDebug,              "tLLR",              0);
        cudaStreamSynchronize(0);
    }
#endif
    cudaStreamSynchronize(0);
}
