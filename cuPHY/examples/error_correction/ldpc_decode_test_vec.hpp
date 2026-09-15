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

#if !defined(LDPC_DECODE_TEST_VEC_HPP_INCLUDED_)
#define LDPC_DECODE_TEST_VEC_HPP_INCLUDED_

#include "cuphy.hpp"
#include "hdf5hpp.hpp"

enum LLR_puncture_status
{
    LLR_PUNCTURE_STATUS_ENABLED,
    LLR_PUNCTURE_STATUS_DISABLED,
    LLR_PUNCTURE_STATUS_UNKNOWN
};

struct ldpc_decode_test_vec_config
{
    int                 BG;     // Base graph (1 or 2)
    int                 Kb;     // Number of information nodes {22, 6, 8, 9, 10 }
    int                 Z;      // Lifting size
    int                 mb;     // Number of parity nodes
    int                 F;      // Number of filler bits
    int                 num_cw; // Number of codewords
    int                 K;      // Information bits (including filler bits)
    int                 B;      // Input data block size
    int                 P;      // Punctured parity bits
    int                 N;      // Modulated bits
    float               R;      // Code rate
    const char*         QAM;    // Modulation description
    LLR_puncture_status punc;   // 2Z info bits puncture status
    uint32_t            crc_type; // CRC type
};

////////////////////////////////////////////////////////////////////////
// ldpc_decode_test_vec
class ldpc_decode_test_vec
{
public:
    virtual                              ~ldpc_decode_test_vec() {}
    cuphyDataType_t                      LLR_type()              { return LLRtype_; }
    [[nodiscard]] virtual const char*    desc() const = 0;
    const ldpc_decode_test_vec_config&   config() const          { return config_; }
    void                                 print_config() const;
    const cuphy::tensor_desc&            LLR_desc() const { return LLR_desc_; }
    void*                                LLR_addr() const { return tLLR_.addr(); }
    const cuphy::tensor_device&          src_bits() const { return tSrcData_; }
    cuphy::tensor_ref                    src_bits_view() { return cuphy::tensor_ref(src_bits_desc_, tSrcData_.addr()); }
    virtual void                         generate() = 0;
    void                                 export_hdf5(hdf5hpp::hdf5_file& f, cudaStream_t strm = 0);
protected:
    //------------------------------------------------------------------
    // find_lifting_size()
    static int find_lifting_size(int BG, int block_sz);
    //------------------------------------------------------------------
    // get_num_info_nodes()
    static int get_num_info_nodes(int BG, int block_sz);
    //------------------------------------------------------------------
    // get_num_info_nodes_from_Z()
    static int get_num_info_nodes_from_Z(int BG, int Z);
    //------------------------------------------------------------------
    // get_QAM_desc()
    static const char* get_QAM_desc(int log2QAM);
    //------------------------------------------------------------------
    // update_config_modulated_bits()
    void update_config_modulated_bits();
    //------------------------------------------------------------------
    // Constructor
    ldpc_decode_test_vec(cuphyDataType_t llrType) :
      LLRtype_(llrType)
    {
    }
    //------------------------------------------------------------------
    // set_LLR_desc()
    void set_LLR_desc(const cuphy::tensor_desc desc) { LLR_desc_ = desc; }
    void set_src_bits_desc(const cuphy::tensor_desc desc) { src_bits_desc_ = desc; }
    //------------------------------------------------------------------
    // Data
    cuphyDataType_t               LLRtype_;      // Data type for LLR inputs
    cuphy::tensor_device          tLLR_;         // Input LLR data
    cuphy::tensor_device          tSrcData_;     // Source (truth) bits
    ldpc_decode_test_vec_config   config_;
private:
    cuphy::tensor_desc            LLR_desc_;     // raw tensor descriptor handle
    cuphy::tensor_desc            src_bits_desc_;
};

#endif // !defined(LDPC_DECODE_TEST_VEC_HPP_INCLUDED_)
