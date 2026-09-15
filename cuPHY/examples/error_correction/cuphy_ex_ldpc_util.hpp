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

#pragma once

#include "cuphy.h"
#include "cuphy.hpp"

#include <cstdint>
#include <string>
#include <vector>

// Self-contained 5G-NR PUSCH (data-only, no UCI) rate-match / de-rate-match + CRC + MCS
// math for the LDPC test bench. Deliberately re-implemented here rather than linked from
// the production PUSCH RX/TX pipeline, so the bench depends only on the public
// cuphy.h / cuphy.hpp -- no PUSCH-pipeline headers or link-time coupling. Keep it that way.
namespace cuphy_ex_ldpc_rm
{

struct pusch_ldpc_case
{
    bool        valid{false};
    int         num_prb{15};
    int         num_layers{1};
    int         mcs{27};
    int         mcs_table{2};
    int         rv{0};
    int         Qm{8};
    int         target_code_rate_x10240{9480};
    float       code_rate{948.0f / 1024.0f};
    int         tb_size{0};
    int         tb_crc_len{0};
    int         cb_crc_len{0};
    int         bg{1};
    int         K_cb{8448};
    int         C{1};
    int         K_prime{0};
    int         Kb{22};
    int         Z{384};
    int         K{8448};
    int         F{0};
    int         Ncb{0};
    int         Ncb_padded{0};
    int         k0{0};
    int         G{0};
    int         p{4};
    int         llr_len{0};
    std::string mod{"QAM256"};
    std::vector<int> E;
    std::vector<int> cb_rm_offset;
};

struct error_stats
{
    uint64_t bit_errors{0};
    uint64_t bit_count{0};
    uint32_t cb_errors{0};
    uint32_t cb_count{0};
    uint32_t tb_errors{0};
    uint32_t tb_count{0};
};

// Modulation name (e.g. "QPSK".."QAM256") for a modulation order Qm.
const char* mod_from_Qm(int Qm);
// LLR data type for an fptype string ("fp16" / "fp32" / "auto").
cuphyDataType_t llr_type_from_string(const std::string& fptype);
// Build the PUSCH LDPC case (TBS, BG, Z, CRC lengths, ratematching geometry) from the MCS +
// allocation. num_dmrs / cdm_no_data set the data-RE count.
pusch_ldpc_case derive_pusch_ldpc_case(int num_prb, int num_layers, int mcs, int mcs_table, int rv,
                                       int num_dmrs = 2, int cdm_no_data = 2);
// Code-block CRC type for the code blocks of case c.
uint32_t cb_crc_type(const pusch_ldpc_case& c);
// Fill cb_bits with num_tbs TBs of random code-block input for case c; deterministic in seed.
void build_random_cb_inputs(const pusch_ldpc_case& c,
                            int                   num_tbs,
                            uint64_t              seed,
                            std::vector<uint8_t>& cb_bits);
// TX ratematching: encoded_u8 -> rm_bits_u8 for num_tbs TBs of case c, on stream strm.
void launch_pusch_tx_rate_match(const pusch_ldpc_case&    c,
                                const cuphy::tensor_device& encoded_u8,
                                cuphy::tensor_device&       rm_bits_u8,
                                int                         num_tbs,
                                cudaStream_t                strm = 0);
// Tile source_num_tbs modulated TBs up to num_tbs (the reuse-TB path), on stream strm.
void launch_repeat_modulated_symbols(const cuphy::tensor_device& source_symbols,
                                     cuphy::tensor_device&       repeated_symbols,
                                     int                         symbols_per_tb,
                                     int                         source_num_tbs,
                                     int                         num_tbs,
                                     cudaStream_t                strm = 0);
// RX de-ratematching: rm_llr -> dec_llr, fp16 LLRs clamped to +-clamp_value, num_tbs TBs, on strm.
void launch_pusch_rx_derate_match_fp16(const pusch_ldpc_case&    c,
                                       const cuphy::tensor_device& rm_llr,
                                       cuphy::tensor_device&       dec_llr,
                                       int                         num_tbs,
                                       float                       clamp_value,
                                       cudaStream_t                strm = 0);
// fp32 counterpart of launch_pusch_rx_derate_match_fp16.
void launch_pusch_rx_derate_match_fp32(const pusch_ldpc_case&    c,
                                       const cuphy::tensor_device& rm_llr,
                                       cuphy::tensor_device&       dec_llr,
                                       int                         num_tbs,
                                       float                       clamp_value,
                                       cudaStream_t                strm = 0);
// Compare decoded_bits against reference cb_bits; returns bit/CB/TB error_stats.
// reuse_tb: all TBs are compared against the single reused reference TB.
error_stats compare_decoded_bits(const pusch_ldpc_case&          c,
                                 int                             num_tbs,
                                 const cuphy::tensor_device&     decoded_bits,
                                 const std::vector<uint8_t>&     cb_bits,
                                 bool                            reuse_tb = false);

} // namespace cuphy_ex_ldpc_rm
