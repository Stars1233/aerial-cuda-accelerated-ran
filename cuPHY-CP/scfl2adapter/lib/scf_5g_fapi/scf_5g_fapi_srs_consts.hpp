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

#if !defined(SCF_5G_FAPI_SRS_CONSTS_HPP_INCLUDED_)
#define SCF_5G_FAPI_SRS_CONSTS_HPP_INCLUDED_

#include <array>
#include <cstdint>

namespace scf_5g_fapi::detail {

// SCF FAPI 10.04 SRS PDU lookup tables. Each table maps a FAPI bitfield
// index to its 3GPP TS 38.211 / SCF FAPI 10.04 SRS PDU spec value.
//
// FAPI defines exactly 3 valid index values per field (0, 1, 2); indices >= 3
// are rejected upstream by SrsPduParser::validate_pdu() before these tables
// are read. Centralized here so the parser, the legacy
// scf_5g_slot_commands.cpp:99 update_cell_command path, and the RU emulator
// (cuPHY-CP/ru-emulator/ru_emulator/utils.hpp:402) can converge on one
// definition of the spec mapping.

// SCF FAPI 10.04 SRS PDU §5.2.6.2.6 nrOfAntennaPorts: index → port count.
inline constexpr std::array<uint8_t, 3> srs_ant_idx_to_port{1, 2, 4};

// SCF FAPI 10.04 SRS PDU §5.2.6.2.6 nrOfSymbols: index → symbol count.
inline constexpr std::array<uint8_t, 3> srs_symb_idx_to_num_symb{1, 2, 4};

// SCF FAPI 10.04 SRS PDU §5.2.6.2.6 repetitionFactor: index → repetition count.
inline constexpr std::array<uint8_t, 3> srs_rep_factor_idx_to_num_rep_factor{1, 2, 4};

// SCF FAPI 10.04 SRS PDU §5.2.6.2.6 combSize: index → comb size (k_TC).
inline constexpr std::array<uint8_t, 3> srs_comb_idx_to_comb_size{2, 4, 8};

} // namespace scf_5g_fapi::detail

#endif // SCF_5G_FAPI_SRS_CONSTS_HPP_INCLUDED_
