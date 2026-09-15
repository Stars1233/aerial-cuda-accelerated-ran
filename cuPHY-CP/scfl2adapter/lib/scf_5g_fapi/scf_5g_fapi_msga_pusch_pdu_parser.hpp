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

#if !defined(SCF_5G_FAPI_MSGA_PUSCH_PDU_PARSER_HPP_INCLUDED_)
#define SCF_5G_FAPI_MSGA_PUSCH_PDU_PARSER_HPP_INCLUDED_

#include <cstdint>

#include <gsl-lite/gsl-lite.hpp>

#include "scf_5g_fapi_slot_concepts.hpp"
#include "scf_5g_fapi_tags.hpp"
#include "scf_5g_fapi.h"

namespace scf_5g_fapi
{

/**
 * Parses a single MsgA-PUSCH PDU from a UL_TTI.request message.
 *
 * MsgA-PUSCH uses scf_fapi_pusch_pdu_t per the current FAPI spec.
 * TODO: update pdu_t when a dedicated MsgA-PUSCH struct is introduced.
 *
 * @tparam V  Type satisfying UlModuleView.
 */
template<UlModuleView V>
class MsgAPuschPduParser final
{
public:
    using pdu_t = scf_fapi_pusch_pdu_t; // TODO: dedicated MsgA struct when available
    static constexpr scf_fapi_ul_tti_pdu_type_t pdu_type = UL_TTI_PDU_TYPE_MsgA_PUSCH;

    explicit MsgAPuschPduParser(V& v) noexcept : view_{&v} {}

    /**
     * Process one MsgA-PUSCH PDU for the given sfn/slot.
     *
     * @param[in] sfn   System Frame Number.
     * @param[in] slot  Slot number.
     * @param[in] pdu   MsgA-PUSCH PDU to process.
     * @return true on success; false if processing could not be completed.
     *         Return value must be checked.
     */
    [[nodiscard]] bool parse(uint16_t sfn, uint16_t slot, const pdu_t& pdu) noexcept;

private:
    gsl_lite::not_null<V*> view_; //!< Non-owning; must outlive this parser.
};

template<UlModuleView V>
bool MsgAPuschPduParser<V>::parse(uint16_t sfn, uint16_t slot,
                                   [[maybe_unused]] const pdu_t& pdu) noexcept
{
    NVLOGW_FMT(detail::k_tag,
               "MsgAPuschPduParser::parse stub called at sfn={} slot={}",
               sfn, slot);
    return true;
}

} // namespace scf_5g_fapi

#endif // SCF_5G_FAPI_MSGA_PUSCH_PDU_PARSER_HPP_INCLUDED_
