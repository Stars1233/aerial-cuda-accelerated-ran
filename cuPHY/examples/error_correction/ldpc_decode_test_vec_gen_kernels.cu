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

#include "ldpc_decode_test_vec_gen_kernels.hpp"

// CRC-24A polynomial as defined in 3GPP TS 38.212
// g_CRC24A(D) = D^24 + D^23 + D^18 + D^17 + D^14 + D^11 + D^10 + D^7 + D^6 + D^5 + D^4 + D^3 + D + 1
// in hex with implicit leading 1
static constexpr uint32_t CRC24A_POLY = 0x864CFB;

// CRC-24B polynomial as defined in 3GPP TS 38.212
// g_CRC24B(D) = D^24 + D^23 + D^6 + D^5 + D + 1
// in hex with implicit leading 1
static constexpr uint32_t CRC24B_POLY = 0x800063;

// CRC-16 polynomial as defined in 3GPP TS 38.212
// g_CRC16(D) = D^16 + D^12 + D^5 + 1
// in hex with implicit leading 1
static constexpr uint32_t CRC16_POLY = 0x1021;

static constexpr int CRC24_LENGTH = 24;
static constexpr int CRC16_LENGTH = 16;

////////////////////////////////////////////////////////////////////////
// create_codeword_kernel()
// Creates a single codeword with random data and appends CRC-24B.
// Output is 1 bit per byte format.
__global__
void create_codeword_kernel(tensor_layout_any tLayout,
                            uint8_t*          tData_addr,
                            int               K,
                            int               F,
                            uint32_t          crc_type,
                            bool              use_existing_info_bits,
                            bool              msb_first)
{
    uint32_t msbMask;
    uint32_t allOnesMask;
    int crc_length;
    uint32_t crc_poly;

    if (crc_type == CUPHY_LDPC_CRC_24A)
    {
        msbMask = 1 << (CRC24_LENGTH - 1);
        allOnesMask = (1 << CRC24_LENGTH) - 1;
        crc_length = CRC24_LENGTH;
        crc_poly = CRC24A_POLY;
    }
    else if (crc_type == CUPHY_LDPC_CRC_24B)
    {
        msbMask = 1 << (CRC24_LENGTH - 1);
        allOnesMask = (1 << CRC24_LENGTH) - 1;
        crc_length = CRC24_LENGTH;
        crc_poly = CRC24B_POLY;
    }
    else if (crc_type == CUPHY_LDPC_CRC_16)
    {
        msbMask = 1 << (CRC16_LENGTH - 1);
        allOnesMask = (1 << CRC16_LENGTH) - 1;
        crc_length = CRC16_LENGTH;
        crc_poly = CRC16_POLY;
    }
    else
    {
        crc_type = CUPHY_LDPC_CRC_NONE;
        msbMask = 0;
        allOnesMask = 0;
        crc_poly = 0;
        crc_length = 0;
    }

    uint32_t crc = 0;
    const int layout_bits = tLayout.dimensions[0];
    const int info_bits_with_crc = K - F;

    if((crc_length > 0) && (info_bits_with_crc <= crc_length))
    {
        crc_type = CUPHY_LDPC_CRC_NONE;
        crc_length = 0;
    }

    // Number of payload bits excluding CRC and filler bits. Full bytes use
    // the byte-oriented CRC update below; leftover bits are handled exactly
    // after the full-byte loop so non-byte-aligned payloads contribute to CRC.
    const int totalPayloadBits = max(0, info_bits_with_crc - crc_length);
    const int numDataBytes = totalPayloadBits / 8;
    const int remainderBits = totalPayloadBits % 8;
    const bool compute_crc = (crc_length > 0);

    // Generate data bytes and compute CRC
    for(int kByte = 0; kByte < numDataBytes; ++kByte)
    {
        uint8_t byte;

        if (use_existing_info_bits)
        {
            byte = 0;
            for(int l = 0; l < 8; ++l)
            {
                const int bit_idx = kByte * 8 + l;
                int n[5] = {bit_idx, 0, 0, 0, 0};
                uint8_t bit = 0;
                if((bit_idx >= 0) && (bit_idx < layout_bits))
                {
                    size_t in = tLayout.offset(n);
                    bit = tData_addr[in];
                }
                if (msb_first)
                {
                    byte |= bit << (7 - l);
                }
                else
                {
                    byte |= bit << l;
                }
            }
        }
        else
        {
            // Use a simple deterministic pattern: (kByte + 1) & 0xFF
            // This ensures reproducible test data
            byte = static_cast<uint8_t>((kByte + 1) & 0xFF);
        }

        // Update CRC with this byte
        if(compute_crc)
        {
            crc ^= static_cast<uint32_t>(byte) << (crc_length - 8);
        }

        // Process each bit of the byte
        for(int l = 0; l < 8; ++l)
        {
            if (use_existing_info_bits == false)
            {
                const int bit_idx = kByte * 8 + l;
                int n[5] = {bit_idx, 0, 0, 0, 0};
                if((bit_idx >= 0) && (bit_idx < layout_bits))
                {
                    size_t out = tLayout.offset(n);
                    uint8_t bit = msb_first ? (byte >> (7 - l)) & 0x01 : (byte >> l) & 0x01;
                    tData_addr[out] = bit;
                }
            }

            // Shift CRC and apply polynomial if MSB is set
            if(compute_crc)
            {
                uint32_t pred = crc & msbMask;
                crc <<= 1;
                if(pred)
                {
                    crc ^= crc_poly;
                }
                crc &= allOnesMask;
            }
        }
    }

    if(remainderBits > 0)
    {
        const int base_bit_idx = numDataBytes * 8;
        uint8_t byte = 0;

        if(use_existing_info_bits)
        {
            for(int l = 0; l < remainderBits; ++l)
            {
                const int bit_idx = base_bit_idx + l;
                int n[5] = {bit_idx, 0, 0, 0, 0};
                uint8_t bit = 0;
                if((bit_idx >= 0) && (bit_idx < layout_bits))
                {
                    size_t in = tLayout.offset(n);
                    bit = tData_addr[in];
                }
                if(msb_first)
                {
                    byte |= bit << (7 - l);
                }
                else
                {
                    byte |= bit << l;
                }
            }
        }
        else
        {
            byte = static_cast<uint8_t>((numDataBytes + 1) & 0xFF);
            byte &= msb_first ? static_cast<uint8_t>(0xFFu << (8 - remainderBits))
                              : static_cast<uint8_t>((1u << remainderBits) - 1u);
        }

        for(int l = 0; l < remainderBits; ++l)
        {
            const int bit_idx = base_bit_idx + l;
            const uint8_t bit = msb_first ? (byte >> (7 - l)) & 0x01 : (byte >> l) & 0x01;

            if(use_existing_info_bits == false)
            {
                int n[5] = {bit_idx, 0, 0, 0, 0};
                if((bit_idx >= 0) && (bit_idx < layout_bits))
                {
                    size_t out = tLayout.offset(n);
                    tData_addr[out] = bit;
                }
            }

            if(compute_crc)
            {
                crc ^= static_cast<uint32_t>(bit) << (crc_length - 1);
                uint32_t pred = crc & msbMask;
                crc <<= 1;
                if(pred)
                {
                    crc ^= crc_poly;
                }
                crc &= allOnesMask;
            }
        }
    }

    if ((crc_type == CUPHY_LDPC_CRC_24A) || (crc_type == CUPHY_LDPC_CRC_24B))
    {
        // Append 3 byte CRC, big endian
        uint8_t crc_bytes[3] = {
            static_cast<uint8_t>((crc >> 16) & 0xFF),
            static_cast<uint8_t>((crc >> 8) & 0xFF),
            static_cast<uint8_t>(crc & 0xFF)
        };

        for(int kByte = 0; kByte < 3; ++kByte)
        {
            for(int l = 0; l < 8; ++l)
            {
                const int bit_idx = K - F - CRC24_LENGTH + kByte * 8 + l;
                int n[5] = {bit_idx, 0, 0, 0, 0};
                if((bit_idx >= 0) && (bit_idx < layout_bits))
                {
                    size_t out = tLayout.offset(n);
                    uint8_t bit = msb_first ? (crc_bytes[kByte] >> (7 - l)) & 0x01
                                            : (crc_bytes[kByte] >> l) & 0x01;
                    tData_addr[out] = bit;
                }
            }
        }
    }
    else if (crc_type == CUPHY_LDPC_CRC_16)
    {
        // Append 2 byte CRC, big endian
        uint8_t crc_bytes[2] = {
            static_cast<uint8_t>((crc >> 8) & 0xFF),
            static_cast<uint8_t>(crc & 0xFF)
        };

        for(int kByte = 0; kByte < 2; ++kByte)
        {
            for(int l = 0; l < 8; ++l)
            {
                const int bit_idx = K - F - CRC16_LENGTH + kByte * 8 + l;
                int n[5] = {bit_idx, 0, 0, 0, 0};
                if((bit_idx >= 0) && (bit_idx < layout_bits))
                {
                    size_t out = tLayout.offset(n);
                    uint8_t bit = msb_first ? (crc_bytes[kByte] >> (7 - l)) & 0x01
                                            : (crc_bytes[kByte] >> l) & 0x01;
                    tData_addr[out] = bit;
                }
            }
        }
    }

    // Set filler bits to 0
    for(int k = max(0, K - F); (k < K) && (k < layout_bits); ++k)
    {
        int n[5] = {k, 0, 0, 0, 0};
        size_t idx = tLayout.offset(n);
        tData_addr[idx] = 0;
    }
}

////////////////////////////////////////////////////////////////////////
// print_codeword_kernel()
// Prints a codeword for debugging purposes.
__global__
void print_codeword_kernel(const tensor_layout_any tLayout,
                           const uint8_t*          tData_addr,
                           int                     K,
                           int                     N,
                           bool                    reverse_bits_in_byte)
{
    printf("tLayout.dimensions()[0] = %d\n", tLayout.dimensions[0]);
    printf("tLayout.dimensions()[1] = %d\n", tLayout.dimensions[1]);

    const int kBytes = K / 8;

    printf("Codeword info bytes (K=%d)\n", K);
    for(int k = 0; k < kBytes; ++k)
    {
        int n[5] = {k, 0, 0, 0, 0};
        size_t idx = tLayout.offset(n);
        uint8_t byte = tData_addr[idx];
        if(reverse_bits_in_byte)
        {
            byte = static_cast<uint8_t>(__brev(static_cast<uint32_t>(byte)) >> 24);
        }
        printf("%02X", byte);
        if((k % 4) == 3)
        {
            printf(" ");
        }
        if((k % 64) == 63)
        {
            printf("\n");
        }
    }
    printf("\n");

    printf("Codeword parity bytes (N-K=%d)\n", N - K);
    const int parityBytes = (N - K) / 8;
    for(int k = 0; k < parityBytes; ++k)
    {
        int n[5] = {kBytes + k, 0, 0, 0, 0};
        size_t idx = tLayout.offset(n);
        uint8_t byte = tData_addr[idx];
        printf("%02X", byte);
        if((k % 4) == 3)
        {
            printf(" ");
        }
        if((k % 64) == 63)
        {
            printf("\n");
        }
    }
    printf("\n");
}

////////////////////////////////////////////////////////////////////////
// create_codeword_with_crc24b()
void create_codeword_with_crc(const tensor_layout_any& tLayout,
                              uint8_t*                 tData_addr,
                              int                      K,
                              int                      F,
                              uint32_t                 crc_type,
                              bool                     use_existing_info_bits,
                              bool                     msb_first)
{
    create_codeword_kernel<<<1, 1>>>(tLayout, tData_addr, K, F, crc_type, use_existing_info_bits, msb_first);
    cudaDeviceSynchronize();
}

////////////////////////////////////////////////////////////////////////
// print_codeword()
void print_codeword(const tensor_layout_any& tLayout,
                    const uint8_t*           tData_addr,
                    int                      K,
                    int                      N,
                    bool                     reverse_bits_in_byte)
{
    print_codeword_kernel<<<1, 1>>>(tLayout, tData_addr, K, N, reverse_bits_in_byte);
    cudaDeviceSynchronize();
}

