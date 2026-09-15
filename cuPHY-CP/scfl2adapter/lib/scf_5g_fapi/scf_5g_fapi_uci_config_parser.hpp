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

#ifndef SCF_5G_FAPI_UCI_CONFIG_PARSER_HPP_
#define SCF_5G_FAPI_UCI_CONFIG_PARSER_HPP_

#ifdef SCF_FAPI_10_04

#include "cuphy.h"
#include "scf_5g_fapi.h"

#include <cstdint>

namespace scf_5g_fapi
{

/**
 * Result of parsing a UCI configuration TLV.
 */
struct uci_config_parse_result final
{
    bool ok{};            //!< True when parsing succeeded and output buffers were updated.
    const char* error{};  //!< Failure reason when ok is false; nullptr when parsing succeeded.
    uint16_t nCsi2Maps{}; //!< Number of CSI-2 maps parsed from a valid TLV.
};

/**
 * Parse a UCI CONFIG TLV into CSI-2 map buffers.
 *
 * The function validates the entire TLV before copying into the destination
 * buffers. It records only small map metadata and payload offsets during
 * validation, then copies map data after every bound has been checked. It
 * returns ok == false without modifying the destination buffers for null
 * destinations, truncated payloads, excessive map counts, invalid sigma-derived
 * map sizes, size overflows, trailing bytes, or destination-capacity overruns.
 * This function does not throw exceptions.
 *
 * @param[in] tlv UCI CONFIG TLV payload to parse.
 * @param[out] destMapBuf Caller-owned destination buffer for CSI-2 map entries;
 *     must be non-null and hold CUPHY_CSI2_SIZE_MAP_BUFFER_SIZE_PER_CELL entries.
 * @param[out] destMapParamBuf Caller-owned destination buffer for CSI-2 map parameters;
 *     must be non-null and hold CUPHY_MAX_NUM_CSI2_SIZE_MAPS_PER_CELL entries.
 * @return uci_config_parse_result. This [[nodiscard]] return value must be checked:
 *     ok == true and nCsi2Maps set on success; otherwise ok == false,
 *     nCsi2Maps == 0, and error describes the rejected input.
 */
[[nodiscard]] uci_config_parse_result parse_uci_config_tlv(const scf_fapi_tl_t& tlv,
                                                           uint16_t* destMapBuf,
                                                           cuphyCsi2MapPrm_t* destMapParamBuf) noexcept;

} // namespace scf_5g_fapi

#endif // SCF_FAPI_10_04

#endif // SCF_5G_FAPI_UCI_CONFIG_PARSER_HPP_
