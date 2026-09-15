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

#include <cstddef>
#include <cstdint>

#include "scf_5g_fapi.h"

namespace cplane_tb {

/// FAPI messages for a single slot (DL and/or UL).
struct SlotFapiMessages {
    uint16_t sfn{};
    uint16_t slot{};

    // DL
    scf_fapi_dl_tti_req_t* dl_tti_req{nullptr};
    size_t dl_body_len{};
    scf_fapi_ul_dci_t* ul_dci_req{nullptr};
    size_t ul_dci_body_len{};

    // UL
    scf_fapi_ul_tti_req_t* ul_tti_req{nullptr};
    size_t ul_body_len{};

    bool has_dl{false};
    bool has_ul{false};
};

/// Abstract interface for providing FAPI messages per slot.
/// Implementations can use testMAC launch patterns, FAPI file replay, etc.
class FapiSource {
public:
    virtual ~FapiSource() = default;

    /// Total number of slots in the test pattern.
    [[nodiscard]] virtual size_t get_total_slots() const = 0;

    /// Number of cells.
    [[nodiscard]] virtual size_t get_num_cells() const = 0;

    /// Get FAPI messages for a given slot index and cell.
    /// The returned pointers are owned by the FapiSource and valid until the next call.
    [[nodiscard]] virtual SlotFapiMessages get_slot(size_t slot_index, size_t cell_index) = 0;
};

} // namespace cplane_tb
