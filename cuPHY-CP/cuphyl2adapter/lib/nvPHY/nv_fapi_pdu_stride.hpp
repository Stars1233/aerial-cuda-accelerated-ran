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

/**
 * @file nv_fapi_pdu_stride.hpp
 * @brief Single canonical FAPI generic-PDU stride helper.
 *
 * Shared by the nvPHY sidecar walk, the SCF FAPI parser helpers, and the
 * dispatch-path message walk so the pdu_size-advance contract lives in one
 * place instead of being copied per call site.
 */

#if !defined(NV_FAPI_PDU_STRIDE_HPP_)
#define NV_FAPI_PDU_STRIDE_HPP_

#include <cstddef>
#include <cstdint>

#include "scf_5g_fapi.h"

namespace nv {

/// Advance a FAPI generic PDU pointer by its own reported size.
///
/// pdu_size is set by the sender as sizeof(scf_fapi_generic_pdu_info_t) +
/// payload_bytes, so advancing by pdu_size -- without adding
/// sizeof(scf_fapi_generic_pdu_info_t) again -- lands exactly on the next
/// PDU header.
///
/// @return Pointer to the next PDU, or nullptr if any of the following hold:
///         - pdu is nullptr
///         - pdu->pdu_size is less than sizeof(scf_fapi_generic_pdu_info_t)
///           (would loop backward or in place)
///         - the resulting pointer would overflow or not advance
[[nodiscard]] inline const scf_fapi_generic_pdu_info_t*
next_pdu(const scf_fapi_generic_pdu_info_t* pdu) noexcept
{
    if (pdu == nullptr) { return nullptr; }
    if (pdu->pdu_size < sizeof(scf_fapi_generic_pdu_info_t)) { return nullptr; }
    const auto base   = reinterpret_cast<uintptr_t>(pdu);
    const auto offset = static_cast<uintptr_t>(pdu->pdu_size);
    if (base + offset <= base) { return nullptr; }
    return reinterpret_cast<const scf_fapi_generic_pdu_info_t*>(base + offset);
}

} // namespace nv

#endif // NV_FAPI_PDU_STRIDE_HPP_
