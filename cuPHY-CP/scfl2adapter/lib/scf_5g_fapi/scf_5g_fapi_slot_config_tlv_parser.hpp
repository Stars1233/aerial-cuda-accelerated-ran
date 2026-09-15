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

#ifndef SCF_5G_FAPI_SLOT_CONFIG_TLV_PARSER_HPP_
#define SCF_5G_FAPI_SLOT_CONFIG_TLV_PARSER_HPP_

#include "scf_5g_fapi.h"
#include "nv_phy_fapi_msg_common.hpp"

#include <tl/expected.hpp>
#include <wise_enum/wise_enum.h>

#include <cstddef>
#include <cstdint>
#include <span>

namespace scf_5g_fapi
{

/**
 * Reason a TDD SLOT_CONFIG TLV was rejected.
 *
 * Reflected via wise_enum so callers and tests can log/branch on the named
 * cause (wise_enum::to_string) instead of a free-text string.
 */
enum class SlotConfigError : std::uint8_t
{
    NullDestination,       //!< Destination span has no backing storage (data() == nullptr).
    InvalidNumerology,     //!< subCarrierCommon (mu) outside the 3GPP [0,4] range.
    EntriesExceedCapacity, //!< Derived slot count overruns the destination buffer.
    PayloadTooSmall        //!< TLV payload holds fewer bytes than the slot count requires.
};

/**
 * Parse a CONFIG.request SLOT_CONFIG TLV into a TDD slot-detail buffer.
 *
 * The slot count is derived from peer-supplied @p subCarrierCommon (mu) and
 * @p tddPeriodNum as <tt>(2^mu) * period_ms</tt>. The function validates the
 * numerology (3GPP TS 38.211, section 4.2), the derived count against
 * <tt>dest.size()</tt>, and the payload
 * length against the bytes the classification loop reads — all before any write —
 * then populates @c dest[0, count). On any rejection it returns the failure cause
 * and leaves @p dest untouched. This function does not throw.
 *
 * @param[in]  tlv              SLOT_CONFIG TLV; @c tlv.val holds OFDM_SYMBOLS_PER_SLOT
 *                              bytes per slot and @c tlv.length bounds the read.
 * @param[in]  subCarrierCommon Numerology mu from CONFIG_TLV_SUB_C_COMMON; must be <= 4.
 * @param[in]  tddPeriodNum     TDD periodicity index from CONFIG_TLV_TDD_PERIOD.
 * @param[out] dest             Destination slot-detail span; @c dest.size() bounds the write.
 *                              A span over null storage is rejected (NullDestination).
 * @return On success the number of slot entries written (<= dest.size());
 *         otherwise a SlotConfigError describing the rejected input. This
 *         [[nodiscard]] return value must be checked.
 */
[[nodiscard]] tl::expected<std::uint16_t, SlotConfigError> parse_slot_config_tlv(
    const scf_fapi_tl_t& tlv,
    std::uint8_t subCarrierCommon,
    std::uint8_t tddPeriodNum,
    std::span<nv::slot_detail_t> dest) noexcept;

} // namespace scf_5g_fapi

// Enable wise_enum reflection
WISE_ENUM_ADAPT(scf_5g_fapi::SlotConfigError,
                NullDestination,
                InvalidNumerology,
                EntriesExceedCapacity,
                PayloadTooSmall)

#endif // SCF_5G_FAPI_SLOT_CONFIG_TLV_PARSER_HPP_
