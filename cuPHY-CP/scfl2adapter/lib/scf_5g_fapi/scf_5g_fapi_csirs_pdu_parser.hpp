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

#if !defined(SCF_5G_FAPI_CSIRS_PDU_PARSER_HPP_INCLUDED_)
#define SCF_5G_FAPI_CSIRS_PDU_PARSER_HPP_INCLUDED_

#include <cstdint>

#include <gsl-lite/gsl-lite.hpp>

#include "scf_5g_fapi_slot_concepts.hpp"
#include "scf_5g_fapi_tags.hpp"
#include "scf_5g_fapi.h"

namespace scf_5g_fapi
{

/**
 * Parses a single CSI-RS PDU from a DL_TTI.request message.
 *
 * Calls update_cell_command for CSI-RS, forwarding group_command,
 * cell_sub_command, config_options, pm_map, mmimo_enabled,
 * and cell_view (cell_params, slot_detail) from the module view.
 *
 * @tparam V  Type satisfying DlModuleView.
 */
template<DlModuleView V>
class CsiRsPduParser final
{
public:
    using pdu_t = scf_fapi_csi_rsi_pdu_t;
    static constexpr scf_fapi_dl_tti_pdu_type_t pdu_type = DL_TTI_PDU_TYPE_CSI_RS;

    explicit CsiRsPduParser(V& v) noexcept : view_{&v} {}

    /**
     * Process one CSI-RS PDU for the given sfn/slot.
     *
     * @param[in] sfn   System Frame Number.
     * @param[in] slot  Slot number.
     * @param[in] pdu   CSI-RS PDU to process.
     * @return true on success; false if processing could not be completed.
     *         Return value must be checked.
     */
    [[nodiscard]] bool parse(uint16_t sfn, uint16_t slot, const pdu_t& pdu) noexcept;

private:
    gsl_lite::not_null<V*> view_; //!< Non-owning; must outlive this parser.
};

template<DlModuleView V>
bool CsiRsPduParser<V>::parse(uint16_t sfn, uint16_t slot,
                               [[maybe_unused]] const pdu_t& pdu) noexcept
{
    NVLOGW_FMT(detail::k_tag,
               "CsiRsPduParser::parse stub called at sfn={} slot={}",
               sfn, slot);
    return true;
}

} // namespace scf_5g_fapi

#endif // SCF_5G_FAPI_CSIRS_PDU_PARSER_HPP_INCLUDED_
