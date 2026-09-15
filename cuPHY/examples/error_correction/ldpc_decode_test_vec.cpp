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

#include "ldpc_decode_test_vec.hpp"
#include "cuphy_hdf5.hpp"
#include "ldpc/ldpc_params.hpp"

namespace
{

int max_block_size[3] = {0, 8448, 3840 };

std::array<int, 51> Z_values =
{
      2,   3,   4,   5,   6,   7,   8,   9,  10,  11,
     12,  13,  14,  15,  16,  18,  20,  22,  24,  26,
     28,  30,  32,  36,  40,  44,  48,  52,  56,  60,
     64,  72,  80,  88,  96, 104, 112, 120, 128, 144,
    160, 176, 192, 208, 224, 240, 256, 288, 320, 352,
    384
};

}

////////////////////////////////////////////////////////////////////////
// ldpc_decode_test_vec::get_num_info_nodes()
int ldpc_decode_test_vec::get_num_info_nodes(int BG, int B)
{
    //------------------------------------------------------------------
    // Validate inputs
    if((BG < 1) || (BG > 2))
    {
        throw std::runtime_error("Invalid base graph");
    }
    if(B > max_block_size[BG])
    {
        throw std::runtime_error("Invalid block size for single segment");
    }
    //------------------------------------------------------------------
    return cuphy::ldpc::derive_Kb(BG, static_cast<uint32_t>(B));
}

////////////////////////////////////////////////////////////////////////
// ldpc_decode_test_vec::get_num_info_nodes_from_Z()
int ldpc_decode_test_vec::get_num_info_nodes_from_Z(int BG, int Z)
{
    if(1 == BG)
    {
        return 22;
    }
    else if(Z > 64)
    {
        return 10;
    }
    else if(Z > 56)
    {
        return 9;
    }
    else if(Z > 24)
    {
        return 8;
    }
    else
    {
        return 6;
    }
}

////////////////////////////////////////////////////////////////////////
// ldpc_decode_test_vec::find_lifting_size()
int ldpc_decode_test_vec::find_lifting_size(int BG, int block_sz)
{
    //------------------------------------------------------------------
    // Validate inputs
    if((BG < 1) || (BG > 2))
    {
        throw std::runtime_error("Invalid base graph");
    }
    if(block_sz > max_block_size[BG])
    {
        throw std::runtime_error("Invalid block size for single segment");
    }
    // See 3GPP TS 38.212, Section 5.2.2
    // For LDPC decode test vectors, we assume a single segment.
    // Therefore, in the notation of the 3GPP spec, L = 0, C = 1,
    // B' = B, and K' = B'/C = B/C = B.
    int Kb = get_num_info_nodes(BG, block_sz);
    for(int i = 0; i < Z_values.size(); ++i)
    {
        if(Z_values[i] * Kb >= block_sz)
        {
            return Z_values[i];
        }
    }
    throw std::runtime_error("Unable to find lifting size for given block size");
}

////////////////////////////////////////////////////////////////////////
// update_config_modulated_bits()
void ldpc_decode_test_vec::update_config_modulated_bits()
{
    // 2*Z information bits are punctured
    // mb gives the number of parity nodes, but some bits in the last
    // node may be punctured.
    config_.N = config_.B - (2 * config_.Z) + (config_.mb * config_.Z - config_.P);
}

////////////////////////////////////////////////////////////////////////
// ldpc_decode_test_vec::print_config()
void ldpc_decode_test_vec::print_config() const
{
    printf("*********************************************************************\n");
    printf("LDPC Configuration:\n");
    printf("*********************************************************************\n");
    printf("Source                           = %s\n", desc());
    printf("BG (base graph)                  = %i\n", config_.BG);
    printf("Kb (info nodes)                  = %i\n", config_.Kb);
    printf("Z  (lifting size)                = %i\n", config_.Z);
    printf("B  (code block size in bits)     = %i\n", config_.B);
    printf("F  (filler bits)                 = %i\n", config_.F);
    printf("K  (info bits, B + F)            = %i\n", config_.K);
    printf("mb (parity nodes)                = %i\n", config_.mb);
    printf("P  (punctured parity bits)       = %i\n", config_.P);
    printf("N  (modulated bits, B+Z(mb-2)-P) = %i\n", config_.N);
    printf("Number of codewords              = %i\n", config_.num_cw);
    printf("R  (code rate = B / N)           = %0.3f\n", config_.R);
    printf("Modulation                       = %s\n", config_.QAM);
    const char* pstr = "";
    switch(config_.punc)
    {
    case LLR_PUNCTURE_STATUS_ENABLED:  pstr = "ENABLED";  break;
    case LLR_PUNCTURE_STATUS_DISABLED: pstr = "DISABLED"; break;
    case LLR_PUNCTURE_STATUS_UNKNOWN:  pstr = "UNKNOWN";  break;
    }
    printf("Punctured info nodes             = %s\n", pstr);

    std::string crc_type_str;
    switch(config_.crc_type)
    {
        case CUPHY_LDPC_CRC_24A: crc_type_str = "CRC-24A"; break;
        case CUPHY_LDPC_CRC_24B: crc_type_str = "CRC-24B"; break;
        case CUPHY_LDPC_CRC_16: crc_type_str = "CRC-16"; break;
        case CUPHY_LDPC_CRC_NONE: crc_type_str = "NONE"; break;
        default: crc_type_str = "UNKNOWN"; break;
    }

    printf("CRC type                         = %s\n", crc_type_str.c_str());
}

////////////////////////////////////////////////////////////////////////
// ldpc_decode_test_vec::get_QAM_desc()
const char* ldpc_decode_test_vec::get_QAM_desc(int log2QAM)
{
    const char* s = "Unknown";
    switch(log2QAM)
    {
    case CUPHY_QAM_2:   s = "BPSK";   break;
    case CUPHY_QAM_4:   s = "QPSK";   break;
    case CUPHY_QAM_16:  s = "QAM16";  break;
    case CUPHY_QAM_64:  s = "QAM64";  break;
    case CUPHY_QAM_256: s = "QAM256"; break;
    default:                          break;
    }
    return s;
}

////////////////////////////////////////////////////////////////////////
// ldpc_decode_test_vec::export_hdf5()
void ldpc_decode_test_vec::export_hdf5(hdf5hpp::hdf5_file& f, cudaStream_t strm)
{
    //------------------------------------------------------------------
    // Convert source data to uint8_t, since HDF5 support for bit
    // tensors doesn't exist (or isn't implement in cuPHY)
    {
        cuphy::tensor_device tSrcData_u8(CUPHY_R_8U, tSrcData_.layout());
        cuphy::tensor_convert(tSrcData_u8, tSrcData_);
        cuphy::write_HDF5_dataset(f, tSrcData_u8, "sourceData", strm);
    }
    //------------------------------------------------------------------
    // Convert LLR data fo FP32 before writing to file. (Not strictly
    // necessary, but there are some questions about HDF5 support for
    // FP16 in terms of accuracy.
    {
        cuphy::tensor_device tLLR_f32(CUPHY_R_32F, LLR_desc().get_info().layout());
        cuphy::tensor_convert(tLLR_f32, LLR_desc(), LLR_addr(), strm);
        cuphy::write_HDF5_dataset(f, tLLR_f32, "inputLLR", strm);
    }

}
