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
#include "cuphy.hpp"

/**
 * Create a single codeword with CRC-16, CRC-24A, or CRC-24B appended (or no CRC).
 *
 * This function optionally generates a debug info byte pattern and optionally
 * computes a CRC on the info byte payload (replacing the last 2 or 3 bytes).
 * The output is 1 bit per byte.
 *
 * @param[in]  tLayout     Tensor layout for the output data
 * @param[out] tData_addr  Output buffer address (1 bit per byte)
 * @param[in]  K           Total information bits (including filler bits)
 * @param[in]  F           Number of filler bits
 * @param[in]  crc_type    CRC type (CUPHY_LDPC_CRC_16, CUPHY_LDPC_CRC_24A, CUPHY_LDPC_CRC_24B, or CUPHY_LDPC_CRC_NONE)
 * @param[in]  use_existing_info_bits If true, use existing info bits not including CRC
 * @param[in]  msb_first   If true, bits are stored MSB first within each byte
 */
void create_codeword_with_crc(const tensor_layout_any& tLayout,
                              uint8_t*                 tData_addr,
                              int                      K,
                              int                      F,
                              uint32_t                 crc_type,
                              bool                     use_existing_info_bits,
                              bool                     msb_first);

/**
 * Print a codeword for debugging purposes.
 *
 * @param[in] tLayout              Tensor layout for the data
 * @param[in] tData_addr           Data buffer address (byte-packed bits)
 * @param[in] K                    Information bits
 * @param[in] N                    Total bits (info + parity)
 * @param[in] reverse_bits_in_byte If true, reverse bit order within each byte
 */
void print_codeword(const tensor_layout_any& tLayout,
                    const uint8_t*           tData_addr,
                    int                      K,
                    int                      N,
                    bool                     reverse_bits_in_byte);

