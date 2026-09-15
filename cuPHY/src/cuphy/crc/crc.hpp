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

#if !defined(CRC_HPP_INCLUDED_)
#define CRC_HPP_INCLUDED_

#include "cuphy.h"

namespace crc
{
const uint32_t     WARP_SIZE                      = 32;
const uint32_t     GLOBAL_BLOCK_SIZE              = 128;
const uint64_t     MAX_PTABLE_SIZE                = MAX_WORDS_PER_TRANSPORT_BLOCK * 32;
const uint64_t     MAX_LARGE_RADIX_TABLE_SIZE     = MAX_WORDS_PER_TRANSPORT_BLOCK / 100;
const uint32_t     G_CRC_24_A                     = 0x01864CFB;
const uint32_t     G_CRC_24_B                     = 0x01800063;
const uint32_t     G_CRC_24_C                     = 0x01B2B117;
const uint32_t     G_CRC_16                       = 0x11021; // MSB is implicit and equal to 1
const uint16_t     G_CRC_11                       = 0x0E21;
const uint8_t      G_CRC_6                        = 0x61;
const uint32_t     MAX_SMALL_A_BITS               = 3824;
const uint32_t     MAX_CB_BIT_SIZE_FOR_CRC16      = 3840;
const uint32_t     MAX_CB_BYTE_SIZE_FOR_CRC16     = (MAX_CB_BIT_SIZE_FOR_CRC16 >> 3);
const uint32_t     SMALL_L_BITS                   = 16;
constexpr uint32_t MAX_SMALL_A_BYTES              = (MAX_SMALL_A_BITS >> 3);
constexpr uint32_t MAX_A_BITS                     = 8424;
constexpr uint32_t MAX_A_BYTES                    = (MAX_A_BITS >> 3);
constexpr uint32_t SMALL_L_BYTES                  = (SMALL_L_BITS >> 3);
const uint32_t     LARGE_L_BITS                   = 24;
constexpr uint32_t LARGE_L_BYTES                  = (LARGE_L_BITS >> 3);
constexpr uint32_t MAX_BYTES_PER_SMALL_CODE_BLOCK = (MAX_SMALL_A_BITS) >> 3; //3824 bits including the size of the CRC value
constexpr uint32_t MAX_WORDS_PER_SMALL_CODE_BLOCK = (MAX_BYTES_PER_SMALL_CODE_BLOCK + sizeof(uint32_t) - 1) / sizeof(uint32_t);
constexpr uint32_t MAX_BYTES_PER_CODE_BLOCK       = 1056; //8448 bits including the size of the CRC value
constexpr uint32_t MAX_WORDS_PER_CODE_BLOCK       = MAX_BYTES_PER_CODE_BLOCK / sizeof(uint32_t);
// Must be power of 2
const uint32_t BITS_PROCESSED_PER_LUT_ENTRY = 32;

template <typename uintCRC_t, size_t uintCRCBitLength>
uintCRC_t computeCRC(const uint8_t* input,
                     uint32_t       size,
                     uintCRC_t      poly,
                     uintCRC_t      initVal = 0,
                     uint64_t       stride  = 1)
{
    uintCRC_t crc         = initVal;
    uintCRC_t msbMask     = (1 << (uintCRCBitLength - 1));
    uintCRC_t allOnesMask = static_cast<uintCRC_t>(-1);
    if((sizeof(uintCRC_t) * 8 - uintCRCBitLength) > 0)
        allOnesMask >>= (sizeof(uintCRC_t) * 8 - uintCRCBitLength - 1);
    const uint64_t limit = static_cast<uint64_t>(size) * stride;
    for(uint64_t i = 0; i < limit; i += stride)
    {
        crc ^= static_cast<uintCRC_t>(input[i] << (uintCRCBitLength - 8));
        for(int b = 0; b < 8; b++)
        {
            uintCRC_t pred = (crc & msbMask) == 0;
            crc <<= 1;
            crc ^= (poly & (pred + allOnesMask));
        }
    }

    return crc;
}
} // namespace crc

#endif
